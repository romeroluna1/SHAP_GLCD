import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False


@dataclass
class RedundancyRecord:
    feature_a: str
    feature_b: str
    assoc_value: float
    assoc_measure: str
    target_score_a: float
    target_score_b: float
    dropped_feature: str
    kept_feature: str


class ProcessingSelectionPipeline:
    """
    Pipeline para:
      1) trabajar con variables ORIGINALES (sin dumificar),
      2) filtrar redundancia usando distintas medidas de asociación,
      3) codificar,
      4) balancear SOLO el entrenamiento,
      5) devolver X_train/X_test listos para el modelo.

    Importante:
      - NO hace train/test split.
      - NO hace escalado.
      - El split debe hacerse afuera.
      - El escalado debe hacerse afuera, después de este pipeline.
    """

    def __init__(
        self,
        association_threshold: float = 0.90,
        ordinal_cols: Optional[List[str]] = None,
        ordinal_categories: Optional[Dict[str, List]] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None,
        balance_mode: str = "none",
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        balance_mode:
            - "none"
            - "class_weight"
            - "oversample"
            - "undersample"
            - "smote"
        """
        self.association_threshold = association_threshold
        self.ordinal_cols = ordinal_cols or []
        self.ordinal_categories = ordinal_categories or {}
        self.categorical_cols_user = categorical_cols
        self.numeric_cols_user = numeric_cols
        self.binary_cols_user = binary_cols
        self.balance_mode = balance_mode.lower().strip()
        self.random_state = random_state
        self.verbose = verbose

        valid_balance_modes = {"none", "class_weight", "oversample", "undersample", "smote"}
        if self.balance_mode not in valid_balance_modes:
            raise ValueError(f"balance_mode debe ser uno de {valid_balance_modes}")

        self.feature_types_: Dict[str, str] = {}
        self.target_scores_: Dict[str, float] = {}
        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.redundancy_records_: List[RedundancyRecord] = []
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.class_weight_: Optional[Dict] = None
        self.output_feature_names_: Optional[List[str]] = None

    # ============================================================
    # UTILIDADES BÁSICAS
    # ============================================================

    @staticmethod
    def _safe_unique_count(s: pd.Series) -> int:
        return s.dropna().nunique()

    @staticmethod
    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        x = pd.Series(x).astype("category")
        y = pd.Series(y).astype("category")

        confusion = pd.crosstab(x, y)
        if confusion.empty:
            return 0.0

        n = confusion.values.sum()
        if n == 0:
            return 0.0

        observed = confusion.values
        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        expected = row_sums @ col_sums / n

        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((observed - expected) ** 2 / expected)

        r, k = observed.shape
        if min(r - 1, k - 1) <= 0:
            return 0.0

        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
        rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
        kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)

        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return 0.0

        return float(np.sqrt(phi2corr / denom))

    @staticmethod
    def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
        categories = pd.Series(categories)
        measurements = pd.Series(measurements)

        valid = categories.notna() & measurements.notna()
        categories = categories[valid]
        measurements = measurements[valid]

        if len(categories) == 0:
            return 0.0

        cat_codes = pd.Categorical(categories).codes
        y_avg = np.mean(measurements)
        classes = np.unique(cat_codes)

        numerator = 0.0
        denominator = np.sum((measurements - y_avg) ** 2)

        if denominator == 0:
            return 0.0

        for c in classes:
            vals = measurements[cat_codes == c]
            if len(vals) == 0:
                continue
            numerator += len(vals) * (np.mean(vals) - y_avg) ** 2

        eta = np.sqrt(numerator / denominator) if denominator > 0 else 0.0
        return float(abs(eta))

    def _ordinal_to_numeric(self, s: pd.Series, col_name: str) -> pd.Series:
        if col_name in self.ordinal_categories:
            cat_type = pd.CategoricalDtype(categories=self.ordinal_categories[col_name], ordered=True)
            codes = s.astype(cat_type).cat.codes
        else:
            # Si no se provee orden, usa categorías observadas ordenadas alfabéticamente
            categories = sorted(pd.Series(s).dropna().unique().tolist())
            cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
            codes = s.astype(cat_type).cat.codes
        return pd.Series(codes, index=s.index)

    # ============================================================
    # TIPOS DE VARIABLES
    # ============================================================

    def infer_feature_types(self, X_train_df: pd.DataFrame) -> Dict[str, str]:
        feature_types = {}

        for col in X_train_df.columns:
            s = X_train_df[col]

            # Prioridad a lo que el usuario pase explícitamente
            if self.numeric_cols_user and col in self.numeric_cols_user:
                feature_types[col] = "numeric"
                continue
            if self.binary_cols_user and col in self.binary_cols_user:
                feature_types[col] = "binary"
                continue
            if self.categorical_cols_user and col in self.categorical_cols_user:
                feature_types[col] = "categorical"
                continue
            if col in self.ordinal_cols:
                feature_types[col] = "ordinal"
                continue

            nunique = self._safe_unique_count(s)

            if self._is_numeric_series(s):
                if nunique <= 2:
                    feature_types[col] = "binary"
                else:
                    feature_types[col] = "numeric"
            else:
                if nunique <= 2:
                    feature_types[col] = "binary"
                else:
                    feature_types[col] = "categorical"

        self.feature_types_ = feature_types

        if self.verbose:
            report = pd.Series(feature_types).value_counts()
            print("\nTipos de variables inferidos:")
            print(report.to_string())

        return feature_types

    # ============================================================
    # ASOCIACIÓN ENTRE VARIABLES
    # ============================================================

    def _feature_association(
        self,
        x: pd.Series,
        y: pd.Series,
        x_type: str,
        y_type: str,
        x_name: str,
        y_name: str,
    ) -> Tuple[float, str]:
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]

        if len(x) < 3:
            return 0.0, "insufficient_data"

        # numeric - numeric
        if x_type == "numeric" and y_type == "numeric":
            try:
                p = abs(pearsonr(x, y)[0]) if x.nunique() > 1 and y.nunique() > 1 else 0.0
            except Exception:
                p = 0.0
            try:
                s = abs(spearmanr(x, y, nan_policy="omit")[0]) if x.nunique() > 1 and y.nunique() > 1 else 0.0
            except Exception:
                s = 0.0
            return float(max(p, s)), "max(|pearson|,|spearman|)"

        # ordinal - ordinal
        if x_type == "ordinal" and y_type == "ordinal":
            x_num = self._ordinal_to_numeric(x, x_name)
            y_num = self._ordinal_to_numeric(y, y_name)
            try:
                s = abs(spearmanr(x_num, y_num, nan_policy="omit")[0]) if x_num.nunique() > 1 and y_num.nunique() > 1 else 0.0
            except Exception:
                s = 0.0
            return float(s), "|spearman_ordinal|"

        # numeric - ordinal
        if (x_type == "numeric" and y_type == "ordinal") or (x_type == "ordinal" and y_type == "numeric"):
            if x_type == "ordinal":
                x_num = self._ordinal_to_numeric(x, x_name)
                y_num = y
            else:
                x_num = x
                y_num = self._ordinal_to_numeric(y, y_name)

            try:
                s = abs(spearmanr(x_num, y_num, nan_policy="omit")[0]) if pd.Series(x_num).nunique() > 1 and pd.Series(y_num).nunique() > 1 else 0.0
            except Exception:
                s = 0.0
            return float(s), "|spearman_numeric_ordinal|"

        # binary - numeric
        if (x_type == "binary" and y_type == "numeric") or (x_type == "numeric" and y_type == "binary"):
            if x_type == "binary":
                xb = pd.Categorical(x).codes
                yn = y.astype(float)
            else:
                xb = pd.Categorical(y).codes
                yn = x.astype(float)

            # Asegurar binario real
            uniq = np.unique(xb)
            if len(uniq) != 2:
                return self._correlation_ratio(
                    categories=x if x_type == "binary" else y,
                    measurements=y if y_type == "numeric" else x
                ), "eta"

            try:
                r = abs(pointbiserialr(xb, yn)[0]) if len(np.unique(yn)) > 1 else 0.0
            except Exception:
                r = 0.0
            return float(r), "|pointbiserial|"

        # numeric - categorical / binary - ordinal / binary - categorical / categorical - ordinal
        if (
            (x_type == "numeric" and y_type in {"categorical", "binary"}) or
            (y_type == "numeric" and x_type in {"categorical", "binary"})
        ):
            if x_type == "numeric":
                return self._correlation_ratio(y, x.astype(float)), "eta"
            return self._correlation_ratio(x, y.astype(float)), "eta"

        # categorical / binary / ordinal con categóricas
        return float(self._cramers_v(x, y)), "cramers_v"

    # ============================================================
    # RELACIÓN CON EL TARGET
    # ============================================================

    def _compute_target_scores(
        self,
        X_train_df: pd.DataFrame,
        y_train: np.ndarray
    ) -> Dict[str, float]:
        """
        Usa mutual information como score general de relevancia con el target.
        """
        target_scores = {}
        y_train = np.asarray(y_train)

        for col in X_train_df.columns:
            s = X_train_df[col]
            ftype = self.feature_types_[col]

            try:
                if ftype == "numeric":
                    x_col = s.astype(float).values.reshape(-1, 1)
                    score = mutual_info_classif(
                        x_col,
                        y_train,
                        discrete_features=False,
                        random_state=self.random_state
                    )[0]

                elif ftype == "ordinal":
                    x_num = self._ordinal_to_numeric(s, col).values.reshape(-1, 1)
                    score = mutual_info_classif(
                        x_num,
                        y_train,
                        discrete_features=True,
                        random_state=self.random_state
                    )[0]

                else:
                    x_codes = pd.Categorical(s).codes.reshape(-1, 1)
                    score = mutual_info_classif(
                        x_codes,
                        y_train,
                        discrete_features=True,
                        random_state=self.random_state
                    )[0]
            except Exception:
                score = 0.0

            target_scores[col] = float(score)

        self.target_scores_ = target_scores

        if self.verbose:
            print("\nTop 15 variables por relevancia con el target (mutual information):")
            top_scores = (
                pd.Series(target_scores)
                .sort_values(ascending=False)
                .head(15)
            )
            print(top_scores.to_string())

        return target_scores

    # ============================================================
    # FILTRO DE REDUNDANCIA
    # ============================================================

    def _select_redundancy_filtered_features(self, X_train_df: pd.DataFrame) -> List[str]:
        cols = list(X_train_df.columns)
        selected = cols.copy()
        dropped = set()
        records: List[RedundancyRecord] = []

        # Recorremos pares sobre las variables originales
        for i in range(len(cols)):
            col_i = cols[i]
            if col_i in dropped:
                continue

            for j in range(i + 1, len(cols)):
                col_j = cols[j]
                if col_j in dropped:
                    continue

                assoc_value, assoc_measure = self._feature_association(
                    X_train_df[col_i],
                    X_train_df[col_j],
                    self.feature_types_[col_i],
                    self.feature_types_[col_j],
                    col_i,
                    col_j,
                )

                if assoc_value >= self.association_threshold:
                    score_i = self.target_scores_.get(col_i, 0.0)
                    score_j = self.target_scores_.get(col_j, 0.0)

                    # Mantener la variable con mayor relación al target
                    if score_i > score_j:
                        kept, dropped_col = col_i, col_j
                    elif score_j > score_i:
                        kept, dropped_col = col_j, col_i
                    else:
                        # Si empatan, mantener la primera por estabilidad
                        kept, dropped_col = col_i, col_j

                    dropped.add(dropped_col)

                    records.append(
                        RedundancyRecord(
                            feature_a=col_i,
                            feature_b=col_j,
                            assoc_value=float(assoc_value),
                            assoc_measure=assoc_measure,
                            target_score_a=float(score_i),
                            target_score_b=float(score_j),
                            dropped_feature=dropped_col,
                            kept_feature=kept,
                        )
                    )

        selected = [c for c in cols if c not in dropped]

        self.selected_features_ = selected
        self.dropped_features_ = sorted(list(dropped))
        self.redundancy_records_ = records

        if self.verbose:
            print(f"\nUmbral de asociación para redundancia: {self.association_threshold}")
            print(f"Variables originales antes del filtro: {len(cols)}")
            print(f"Variables originales después del filtro: {len(selected)}")
            print(f"Variables eliminadas por redundancia: {len(self.dropped_features_)}")

            if len(records) > 0:
                df_records = pd.DataFrame([r.__dict__ for r in records]).sort_values(
                    by="assoc_value", ascending=False
                )
                print("\nPrimeros pares eliminados por redundancia:")
                print(df_records.head(15).to_string(index=False))

        return selected

    # ============================================================
    # BALANCEO
    # ============================================================

    def _apply_balance_original_space(
        self,
        X_train_sel: pd.DataFrame,
        y_train: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        y_train = np.asarray(y_train)

        if self.balance_mode in {"none", "class_weight"}:
            if self.balance_mode == "class_weight":
                classes = np.unique(y_train)
                weights = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=y_train
                )
                self.class_weight_ = {cls: float(w) for cls, w in zip(classes, weights)}

                if self.verbose:
                    print("\nclass_weight calculado:")
                    print(self.class_weight_)

            return X_train_sel.copy(), y_train.copy()

        if not IMBLEARN_AVAILABLE:
            raise ImportError(
                "imblearn no está instalado. Instálalo si quieres usar oversample/undersample/smote."
            )

        if self.balance_mode == "oversample":
            sampler = RandomOverSampler(random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X_train_sel, y_train)

            if self.verbose:
                print("\nBalanceo aplicado: RandomOverSampler")
                print("Distribución antes:")
                print(pd.Series(y_train).value_counts().sort_index().to_string())
                print("Distribución después:")
                print(pd.Series(y_res).value_counts().sort_index().to_string())

            return pd.DataFrame(X_res, columns=X_train_sel.columns), np.asarray(y_res)

        if self.balance_mode == "undersample":
            sampler = RandomUnderSampler(random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X_train_sel, y_train)

            if self.verbose:
                print("\nBalanceo aplicado: RandomUnderSampler")
                print("Distribución antes:")
                print(pd.Series(y_train).value_counts().sort_index().to_string())
                print("Distribución después:")
                print(pd.Series(y_res).value_counts().sort_index().to_string())

            return pd.DataFrame(X_res, columns=X_train_sel.columns), np.asarray(y_res)

        # Para SMOTE se manejará más adelante, ya en espacio codificado
        return X_train_sel.copy(), y_train.copy()

    def _apply_smote_encoded_space(
        self,
        X_train_enc: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.balance_mode != "smote":
            return X_train_enc, y_train

        if not IMBLEARN_AVAILABLE:
            raise ImportError("imblearn no está instalado. Instálalo para usar SMOTE.")

        warnings.warn(
            "Estás usando SMOTE sobre variables ya codificadas. "
            "Si luego trabajas con muchas variables one-hot, revisa con cuidado si los ejemplos sintéticos tienen sentido.",
            UserWarning
        )

        sampler = SMOTE(random_state=self.random_state)
        X_res, y_res = sampler.fit_resample(X_train_enc, y_train)

        if self.verbose:
            print("\nBalanceo aplicado: SMOTE")
            print("Distribución antes:")
            print(pd.Series(y_train).value_counts().sort_index().to_string())
            print("Distribución después:")
            print(pd.Series(y_res).value_counts().sort_index().to_string())

        return X_res, y_res

    # ============================================================
    # CODIFICACIÓN
    # ============================================================

    def _build_preprocessor(self, X_train_sel: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = [c for c in X_train_sel.columns if self.feature_types_[c] == "numeric"]
        binary_cols = [c for c in X_train_sel.columns if self.feature_types_[c] == "binary"]
        ordinal_cols = [c for c in X_train_sel.columns if self.feature_types_[c] == "ordinal"]
        categorical_cols = [c for c in X_train_sel.columns if self.feature_types_[c] == "categorical"]

        transformers = []

        if numeric_cols:
            transformers.append(("num", "passthrough", numeric_cols))

        if binary_cols:
            # Si son numéricas 0/1, passthrough. Si son texto, ordinal encoder a 1 columna.
            binary_numeric = [c for c in binary_cols if self._is_numeric_series(X_train_sel[c])]
            binary_non_numeric = [c for c in binary_cols if not self._is_numeric_series(X_train_sel[c])]

            if binary_numeric:
                transformers.append(("bin_num", "passthrough", binary_numeric))

            if binary_non_numeric:
                transformers.append((
                    "bin_cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    binary_non_numeric
                ))

        if ordinal_cols:
            categories = []
            for col in ordinal_cols:
                if col in self.ordinal_categories:
                    categories.append(self.ordinal_categories[col])
                else:
                    categories.append(sorted(pd.Series(X_train_sel[col]).dropna().unique().tolist()))

            transformers.append((
                "ord",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                ordinal_cols
            ))

        if categorical_cols:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

            transformers.append(("cat", ohe, categorical_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True
        )

        return preprocessor

    @staticmethod
    def _clean_feature_names(feature_names: List[str]) -> List[str]:
        cleaned = []
        for f in feature_names:
            if "__" in f:
                cleaned.append(f.split("__", 1)[1])
            else:
                cleaned.append(f)
        return cleaned

    # ============================================================
    # MÉTODO PRINCIPAL
    # ============================================================

    def fit_transform_train_test(
        self,
        X_train_df: pd.DataFrame,
        y_train: np.ndarray,
        X_test_df: pd.DataFrame,
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        if not isinstance(X_train_df, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train_df)

        if not isinstance(X_test_df, pd.DataFrame):
            X_test_df = pd.DataFrame(X_test_df)

        if list(X_train_df.columns) != list(X_test_df.columns):
            raise ValueError("X_train_df y X_test_df deben tener las mismas columnas y en el mismo orden.")

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # 1) Inferir tipos en TRAIN
        self.infer_feature_types(X_train_df)

        # 2) Score con target usando TRAIN
        self._compute_target_scores(X_train_df, y_train)

        # 3) Filtrar redundancia usando TRAIN
        selected_features = self._select_redundancy_filtered_features(X_train_df)

        X_train_sel = X_train_df[selected_features].copy()
        X_test_sel = X_test_df[selected_features].copy()

        # 4) Balancear SOLO TRAIN en espacio original cuando aplique
        if self.balance_mode in {"oversample", "undersample", "none", "class_weight"}:
            X_train_bal, y_train_bal = self._apply_balance_original_space(X_train_sel, y_train)
        else:
            X_train_bal, y_train_bal = X_train_sel.copy(), y_train.copy()

        # 5) Construir codificador y transformar
        self.preprocessor_ = self._build_preprocessor(X_train_bal)
        X_train_enc = self.preprocessor_.fit_transform(X_train_bal)
        X_test_enc = self.preprocessor_.transform(X_test_sel)

        try:
            output_names = list(self.preprocessor_.get_feature_names_out())
        except Exception:
            output_names = [f"feature_{i}" for i in range(X_train_enc.shape[1])]

        self.output_feature_names_ = self._clean_feature_names(output_names)

        # 6) SMOTE solo si se pidió
        if self.balance_mode == "smote":
            X_train_enc, y_train_bal = self._apply_smote_encoded_space(X_train_enc, y_train_bal)

        if self.verbose:
            print("\nResumen final del procesamiento:")
            print(f"Variables originales de entrada: {X_train_df.shape[1]}")
            print(f"Variables originales seleccionadas: {len(selected_features)}")
            print(f"Variables después de codificación: {X_train_enc.shape[1]}")
            print(f"X_train final: {X_train_enc.shape}")
            print(f"X_test final : {X_test_enc.shape}")

        return X_train_enc, X_test_enc, y_train_bal, y_test, self.output_feature_names_

    # ============================================================
    # REPORTES AUXILIARES
    # ============================================================

    def get_dropped_features(self) -> List[str]:
        return self.dropped_features_

    def get_selected_original_features(self) -> List[str]:
        return self.selected_features_

    def get_redundancy_report(self) -> pd.DataFrame:
        if not self.redundancy_records_:
            return pd.DataFrame(
                columns=[
                    "feature_a", "feature_b", "assoc_value", "assoc_measure",
                    "target_score_a", "target_score_b", "dropped_feature", "kept_feature"
                ]
            )
        return pd.DataFrame([r.__dict__ for r in self.redundancy_records_])

    def get_feature_type_report(self) -> pd.DataFrame:
        if not self.feature_types_:
            return pd.DataFrame(columns=["feature", "type", "target_score"])

        rows = []
        for feat, ftype in self.feature_types_.items():
            rows.append({
                "feature": feat,
                "type": ftype,
                "target_score": self.target_scores_.get(feat, np.nan)
            })
        return pd.DataFrame(rows).sort_values(by=["type", "target_score"], ascending=[True, False])