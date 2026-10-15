import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from scipy.stats import ks_2samp


class Gexplainer:
    def __init__(
        self,
        csv_path,
        target_column,
        model,
        shap_pickle="shap_values.pickle",
        j_sel=1,
        r_sel=2,
        top=7,
        random_state=42,
        class_mapping=None,
        figure_font_scale=1.20
    ):
        self.csv_path = csv_path
        self.target_column = target_column
        self.model = model
        self.shap_pickle = shap_pickle
        self.j_sel = j_sel
        self.r_sel = r_sel
        self.TOP = top
        self.RANDOM_STATE = random_state

        # Información necesaria para verificar que los valores SHAP
        # reconstruyen predict_proba y no la salida raw de XGBoost.
        self.shap_expected_value = None
        self.shap_output_space = None
        self.shap_background_size = None

        # =====================================================
        # Parámetro visual
        # =====================================================
        # Solo modifica el tamaño de las letras en las figuras.
        # No afecta modelos, métricas, SHAP, rankings ni estadísticos.
        self.figure_font_scale = float(figure_font_scale)

        # =====================================================
        # Cargar dataset
        # =====================================================
        self.df = pd.read_csv(csv_path)

        self.features = [
            c for c in self.df.columns
            if c != target_column
        ]

        self.X = self.df[self.features]
        self.y = self.df[target_column].values
        self.class_names = sorted(np.unique(self.y))

        if class_mapping is None:
            self.class_mapping = {
                i: f"class_{i}"
                for i in self.class_names
            }
        else:
            self.class_mapping = class_mapping

        if self.j_sel not in self.class_mapping:
            raise ValueError(
                f"j_sel={self.j_sel} no existe en class_mapping."
            )

        if self.r_sel not in self.class_mapping:
            raise ValueError(
                f"r_sel={self.r_sel} no existe en class_mapping."
            )

        # =====================================================
        # Split y normalización
        # =====================================================
        # Primero se separan entrenamiento y prueba.
        # Después, el escalador se ajusta únicamente con train.
        self.X_tr_raw, self.X_te_raw, self.y_tr, self.y_te = train_test_split(
            self.X,
            self.y,
            test_size=0.3,
            stratify=self.y,
            random_state=self.RANDOM_STATE
        )

        self.scaler = MinMaxScaler()

        self.X_tr = pd.DataFrame(
            self.scaler.fit_transform(self.X_tr_raw),
            columns=self.features,
            index=self.X_tr_raw.index
        )

        self.X_te = pd.DataFrame(
            self.scaler.transform(self.X_te_raw),
            columns=self.features,
            index=self.X_te_raw.index
        )

        # Se mantiene por compatibilidad con código previo.
        self.X_norm = pd.DataFrame(
            self.scaler.transform(self.X),
            columns=self.features,
            index=self.X.index
        )

        self.N_te, self.M = self.X_te.shape
        self.K = len(self.class_names)

        # =====================================================
        # Configuración visual
        # =====================================================
        self._setup_plot_style()

    # =========================================================
    # CONFIGURACIÓN VISUAL
    # =========================================================

    def _fs(self, size):
        """
        Escala el tamaño de fuente para figuras de publicación.
        """
        return int(round(float(size) * self.figure_font_scale))

    def _setup_plot_style(self):
        """
        Configuración visual general.

        figure_font_scale solo modifica la apariencia gráfica.
        """

        sns.set_context("paper", font_scale=1.4)
        sns.set_style("whitegrid")

        self.palette = sns.color_palette(
            "colorblind",
            max(3, self.K, 10)
        )

        self.class_color = {
            c: self.palette[i % len(self.palette)]
            for i, c in enumerate(self.class_names)
        }

    def _legend_below(
        self,
        ax,
        fig,
        ncol=2,
        fontsize=22,
        anchor_y=-0.24,
        bottom_margin=0.30
    ):
        """
        Coloca la leyenda fuera del área gráfica, centrada debajo.

        Esta ubicación evita que la leyenda cubra barras,
        porcentajes o valores numéricos.
        """

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, anchor_y),
            ncol=ncol,
            fontsize=self._fs(fontsize),
            frameon=True,
            framealpha=0.90,
            borderaxespad=0.0
        )

        fig.subplots_adjust(
            bottom=bottom_margin
        )

    def _format_feature_label(self, label, max_chars=24):
        """
        Divide nombres largos en varias líneas solo para visualización.

        No modifica los nombres reales usados en cálculos.
        """

        label = str(label)

        if len(label) <= max_chars:
            return label

        parts = label.split("_")
        lines = []
        current = ""

        for part in parts:
            candidate = (
                part
                if current == ""
                else current + "_" + part
            )

            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)

                current = part

        if current:
            lines.append(current)

        return "\n".join(lines)

    # =========================================================
    # VALIDACIÓN CRUZADA
    # =========================================================

    def validacion_cruzada(self):
        """
        Realiza validación cruzada estratificada sobre el 70% destinado
        a entrenamiento.

        Dentro de cada fold, MinMaxScaler se ajusta exclusivamente con
        los datos de entrenamiento del fold correspondiente.
        """

        print("\n== Validación cruzada ==")

        cv = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=self.RANDOM_STATE
        )

        resultados = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(self.X_tr_raw, self.y_tr),
            start=1
        ):
            X_fold_train_raw = self.X_tr_raw.iloc[train_idx]
            X_fold_val_raw = self.X_tr_raw.iloc[val_idx]

            y_fold_train = self.y_tr[train_idx]
            y_fold_val = self.y_tr[val_idx]

            fold_scaler = MinMaxScaler()

            X_fold_train = pd.DataFrame(
                fold_scaler.fit_transform(X_fold_train_raw),
                columns=self.features,
                index=X_fold_train_raw.index
            )

            X_fold_val = pd.DataFrame(
                fold_scaler.transform(X_fold_val_raw),
                columns=self.features,
                index=X_fold_val_raw.index
            )

            modelo = self._clone_model()
            modelo.fit(X_fold_train, y_fold_train)

            y_pred = modelo.predict(X_fold_val)
            y_proba = modelo.predict_proba(X_fold_val)

            resultados.append({
                "Pliegue": f"Pliegue {fold}",
                "Accuracy": accuracy_score(
                    y_fold_val,
                    y_pred
                ),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y_fold_val,
                    y_pred
                ),
                "F1_Macro": f1_score(
                    y_fold_val,
                    y_pred,
                    average="macro"
                ),
                "MCC": matthews_corrcoef(
                    y_fold_val,
                    y_pred
                ),
                "Macro_AUC_OvO": self._calculate_auc(
                    y_fold_val,
                    y_proba
                )
            })

        df_resultados = pd.DataFrame(resultados)

        mean_results = (
            df_resultados
            .drop(columns="Pliegue")
            .mean()
        )

        std_results = (
            df_resultados
            .drop(columns="Pliegue")
            .std()
        )

        print("\nResultados por pliegue:")
        print(df_resultados)

        print("\nPromedios (CV):")

        for metric in mean_results.index:
            print(
                f"{metric:18s}: "
                f"{mean_results[metric]:.4f} ± "
                f"{std_results[metric]:.4f}"
            )

        self.cv_results_ = df_resultados

        self.cv_summary_ = pd.DataFrame({
            "Mean": mean_results,
            "Std": std_results
        })

        return df_resultados

    def _clone_model(self):
        """
        Clona el modelo usando sus parámetros originales.
        """

        params = self.model.get_params()

        return self.model.__class__(
            **params
        )

    def _calculate_auc(self, y_true, y_proba):
        """
        Calcula AUC para problemas binarios o Macro-AUC OvO
        para problemas multiclase.
        """

        try:
            if len(np.unique(self.y)) == 2:
                return roc_auc_score(
                    y_true,
                    y_proba[:, 1]
                )

            return roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovo",
                average="macro"
            )

        except ValueError:
            return np.nan

    def entrenar_modelo_final(self):
        """
        Entrena el modelo final con el 70% de entrenamiento.
        """

        print("\n== Entrenamiento final con 70% ==")

        self.modelo_final = self._clone_model()

        self.modelo_final.fit(
            self.X_tr,
            self.y_tr
        )

    # =========================================================
    # SHAP VALUES
    # =========================================================

    def _format_shap_values(self, shap_values, n_expected):
        """
        Convierte la salida SHAP a la estructura requerida:

            (n_instancias, n_caracteristicas, n_clases)

        Permite compatibilidad con versiones que retornan una lista
        de arreglos o un arreglo con las clases en el primer eje.
        """

        if isinstance(shap_values, list):
            shap_values = np.stack(
                shap_values,
                axis=2
            )
        else:
            shap_values = np.asarray(
                shap_values
            )

        expected_shape = (
            n_expected,
            len(self.features),
            len(self.class_names)
        )

        if shap_values.shape == expected_shape:
            return shap_values

        alternative_shape = (
            len(self.class_names),
            n_expected,
            len(self.features)
        )

        if shap_values.shape == alternative_shape:
            return np.transpose(
                shap_values,
                (1, 2, 0)
            )

        raise ValueError(
            "Los valores SHAP no tienen la forma esperada. "
            f"Se esperaba {expected_shape} y se recibió "
            f"{shap_values.shape}."
        )

    def calcular_y_guardar_shap(
        self,
        filename=None,
        max_test_samples=100,
        background_size=100
    ):
        """
        Calcula y guarda los valores SHAP del conjunto de prueba.

        - Para XGBClassifier se utiliza KernelExplainer sobre
          predict_proba. Por tanto, los valores SHAP quedan expresados
          en escala de probabilidad.
        - Para otros modelos de árboles se utiliza TreeExplainer.
        - Para modelos no basados en árboles se utiliza KernelExplainer.
        """

        import shap

        if not hasattr(self, "modelo_final"):
            self.entrenar_modelo_final()

        print("\n== Calculando SHAP values ==")

        model_class = type(
            self.modelo_final
        ).__name__.lower()

        is_xgb = "xgb" in model_class or "lgbm" in model_class

        is_tree = any(
            x in model_class
            for x in [
                "tree",
                "forest",
                "gbm",
                "boost",
                "lgbm",
                "xgb",
                "catboost"
            ]
        )

        X_te_eval = self.X_te.copy()

        # Para modelos no basados en árboles, limitar tamaño por velocidad.
        if not is_tree and len(self.X_te) > max_test_samples:
            print(
                f"Advertencia: KernelExplainer es lento. "
                f"Se usarán solo {max_test_samples} muestras de test."
            )

            X_te_eval = self.X_te.sample(
                max_test_samples,
                random_state=self.RANDOM_STATE
            )

        # =====================================================
        # Caso especial: XGBoost EN PROBABILIDADES
        # =====================================================
        if is_xgb:
            print(
                "Usando KernelExplainer sobre predict_proba para "
                "obtener SHAP en escala de probabilidad."
            )

            n_background = min(
                int(background_size),
                len(self.X_tr)
            )

            background = shap.sample(
                self.X_tr,
                n_background,
                random_state=self.RANDOM_STATE
            )

            # No se pasa predict_proba como método ligado directamente porque
            # algunas combinaciones SHAP/XGBoost intentan modificar la
            # propiedad de solo lectura feature_names_in_ del estimador.
            # La función envolvente devuelve exactamente las mismas
            # probabilidades, pero evita esa incompatibilidad técnica.
            def predict_proba_for_shap(data):
                return self.modelo_final.predict_proba(data)

            explainer = shap.KernelExplainer(
                predict_proba_for_shap,
                background,
                link="identity"
            )

            np.random.seed(self.RANDOM_STATE)
            shap_values = explainer.shap_values(X_te_eval)

            self.shap_expected_value = np.asarray(
                explainer.expected_value,
                dtype=float
            ).reshape(-1)
            self.shap_output_space = "probability"
            self.shap_background_size = n_background

        # =====================================================
        # Otros modelos basados en árboles
        # =====================================================
        elif is_tree:
            explainer = shap.TreeExplainer(
                self.modelo_final
            )

            shap_values = explainer.shap_values(
                X_te_eval
            )

            self.shap_expected_value = np.asarray(
                explainer.expected_value,
                dtype=float
            ).reshape(-1)

            # En clasificadores de scikit-learn como RandomForestClassifier
            # y DecisionTreeClassifier, TreeExplainer reconstruye directamente
            # predict_proba. Se declara esa escala de forma explícita y se
            # verifica numéricamente más abajo antes de continuar.
            if hasattr(self.modelo_final, "predict_proba"):
                self.shap_output_space = "probability"
            else:
                self.shap_output_space = "model_output"

        # =====================================================
        # Modelos no basados en árboles
        # =====================================================
        else:
            print(
                "Usando shap.KernelExplainer "
                "(esto puede tomar varios minutos)..."
            )

            explainer = shap.KernelExplainer(
                self.modelo_final.predict_proba,
                self.X_tr
            )

            shap_values = explainer.shap_values(
                X_te_eval
            )

            self.shap_expected_value = np.asarray(
                explainer.expected_value,
                dtype=float
            ).reshape(-1)
            self.shap_output_space = "probability"

        shap_values = self._format_shap_values(
            shap_values,
            len(X_te_eval)
        )

        fname = filename or self.shap_pickle

        payload = {
            "shap_values": shap_values,
            "expected_value": self.shap_expected_value,
            "output_space": self.shap_output_space,
            "background_size": self.shap_background_size,
            "random_state": self.RANDOM_STATE,
            "feature_names": list(self.features),
            "class_names": list(self.class_names)
        }

        with open(fname, "wb") as f:
            pickle.dump(
                payload,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        print(
            f"SHAP values calculados y guardados en: {fname}"
        )

        self.shap_vals = shap_values

        if self.shap_output_space == "probability":
            self.verificar_escala_probabilidad(X_te_eval)

        # Si KernelExplainer utilizó un subconjunto de test,
        # mantener la correspondencia exacta con X_te e y_te.
        if not is_tree and len(X_te_eval) < len(self.X_te):
            self.X_te = X_te_eval.copy()

            self.y_te = self.df.loc[
                X_te_eval.index,
                self.target_column
            ].values

            self.N_te, self.M = self.X_te.shape

    def cargar_shap(self, filename=None):
        """
        Carga valores SHAP previamente guardados y valida su estructura.
        """

        fname = filename or self.shap_pickle

        print(f"\n== Cargando SHAP desde: {fname} ==")

        with open(fname, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "shap_values" not in payload:
            raise ValueError(
                "El archivo SHAP cargado pertenece al formato antiguo y "
                "no contiene información verificable sobre la escala. "
                "Use un nombre de pickle nuevo y ejecute con "
                "shap_mode='always'."
            )

        shap_values = payload["shap_values"]
        self.shap_expected_value = np.asarray(
            payload.get("expected_value"),
            dtype=float
        ).reshape(-1)
        self.shap_output_space = payload.get("output_space")
        self.shap_background_size = payload.get("background_size")

        if self.shap_output_space != "probability":
            raise ValueError(
                "El archivo SHAP no declara escala de probabilidad. "
                f"Escala encontrada: {self.shap_output_space!r}."
            )

        self.shap_vals = self._format_shap_values(
            shap_values,
            len(self.X_te)
        )

        self.N_te, self.M, self.K = self.shap_vals.shape

        print(
            f"SHAP shape validado: {self.shap_vals.shape}"
        )

        self.verificar_escala_probabilidad(self.X_te)

    def verificar_escala_probabilidad(
        self,
        X_eval,
        n_check=10,
        atol=1e-5,
        rtol=1e-5
    ):
        """Verifica base + suma(SHAP) = predict_proba por clase."""

        if self.shap_expected_value is None:
            raise RuntimeError(
                "No existe expected_value para verificar la escala SHAP."
            )

        n = min(int(n_check), len(X_eval), len(self.shap_vals))
        if n <= 0:
            raise ValueError("No hay instancias para verificar SHAP.")

        probabilities = self.modelo_final.predict_proba(
            X_eval.iloc[:n] if hasattr(X_eval, "iloc") else X_eval[:n]
        )

        reconstructed = (
            self.shap_expected_value.reshape(1, -1)
            + np.sum(self.shap_vals[:n], axis=1)
        )

        if probabilities.shape != reconstructed.shape:
            raise ValueError(
                "La reconstrucción SHAP y predict_proba tienen formas "
                f"distintas: {reconstructed.shape} y {probabilities.shape}."
            )

        errors = np.abs(probabilities - reconstructed)
        max_error = float(np.max(errors))
        mean_error = float(np.mean(errors))

        print("\n== Verificación de escala SHAP ==")
        print("Salida explicada: predict_proba")
        print(f"Instancias verificadas: {n}")
        print(f"Error medio: {mean_error:.3e}")
        print(f"Error máximo: {max_error:.3e}")

        if not np.allclose(
            probabilities,
            reconstructed,
            atol=atol,
            rtol=rtol
        ):
            raise RuntimeError(
                "La verificación falló: base + suma(SHAP) no coincide "
                "con predict_proba. No se generarán resultados SHAP-GLCD."
            )

        print("VERIFICACIÓN SUPERADA: SHAP está en probabilidad.")

    # =========================================================
    # CÁLCULO SHAP-GLCD
    # =========================================================

    def _seleccionar_muestra_balanceada_par(self):
        """
        Selecciona, dentro del conjunto de prueba, una muestra balanceada
        formada exclusivamente por instancias cuya ETIQUETA VERDADERA
        pertenece al par (j_sel, r_sel).

        Criterios metodológicos:
        - selección por etiqueta verdadera;
        - mismo número de instancias de Cj y Cr;
        - semilla RANDOM_STATE;
        - no se filtra por predicción, por lo que se conservan tanto
          aciertos como errores del clasificador.

        La selección se realiza sobre posiciones del test para conservar
        la correspondencia exacta entre X_te, y_te y shap_vals.
        """

        if not hasattr(self, "shap_vals"):
            raise AttributeError(
                "Primero deben calcularse o cargarse los valores SHAP."
            )

        if len(self.shap_vals) != len(self.y_te):
            raise ValueError(
                "No existe correspondencia uno a uno entre SHAP y y_te: "
                f"len(shap_vals)={len(self.shap_vals)}, "
                f"len(y_te)={len(self.y_te)}."
            )

        idx_j = np.flatnonzero(
            self.y_te == self.j_sel
        )

        idx_r = np.flatnonzero(
            self.y_te == self.r_sel
        )

        if len(idx_j) == 0 or len(idx_r) == 0:
            raise ValueError(
                "No hay suficientes instancias en test para construir "
                f"el par {self.j_sel} vs {self.r_sel}. "
                f"Disponibles: Cj={len(idx_j)}, Cr={len(idx_r)}."
            )

        n_per_class = min(
            len(idx_j),
            len(idx_r)
        )

        rng = np.random.RandomState(
            self.RANDOM_STATE
        )

        selected_j = rng.choice(
            idx_j,
            size=n_per_class,
            replace=False
        )

        selected_r = rng.choice(
            idx_r,
            size=n_per_class,
            replace=False
        )

        selected_idx = np.concatenate([
            selected_j,
            selected_r
        ])

        # El orden no afecta las métricas agregadas, pero se mezcla de
        # manera reproducible para no dejar las clases agrupadas por filas.
        rng.shuffle(selected_idx)

        self.pair_indices = selected_idx
        self.N_pair = len(selected_idx)

        self.X_te_pair = self.X_te.iloc[
            selected_idx
        ].copy()

        # Versión sin escalar de la misma muestra pairwise. Se conserva
        # para identificar de forma inequívoca las variables binarias 0/1
        # y calcular sus proporciones originales mediante Cohen h.
        self.X_te_pair_raw = self.X_te_raw.iloc[
            selected_idx
        ].copy()

        self.y_te_pair = self.y_te[
            selected_idx
        ].copy()

        self.shap_vals_pair = self.shap_vals[
            selected_idx,
            :,
            :
        ].copy()

        original_indices = self.X_te.index.to_numpy()[
            selected_idx
        ]

        metadata = pd.DataFrame({
            "test_position": selected_idx,
            "original_index": original_indices,
            "y_true": self.y_te_pair
        })

        # Si el modelo final está disponible, registrar predicción solo
        # para verificar explícitamente que se conservan aciertos y errores.
        # La predicción NO interviene en la selección.
        if hasattr(self, "modelo_final"):
            y_pred_pair = self.modelo_final.predict(
                self.X_te_pair
            )

            metadata["y_pred"] = y_pred_pair
            metadata["correct_prediction"] = (
                y_pred_pair == self.y_te_pair
            )

        pair_tag = f"{self.j_sel}_vs_{self.r_sel}"

        metadata.to_csv(
            f"selected_pair_metadata_{pair_tag}.csv",
            index=False
        )

        summary = pd.DataFrame([{
            "class_j": self.j_sel,
            "class_j_name": self.class_mapping[self.j_sel],
            "class_r": self.r_sel,
            "class_r_name": self.class_mapping[self.r_sel],
            "available_j_in_test": int(len(idx_j)),
            "available_r_in_test": int(len(idx_r)),
            "selected_j": int(n_per_class),
            "selected_r": int(n_per_class),
            "N_selected": int(self.N_pair),
            "random_state": int(self.RANDOM_STATE),
            "selection_basis": "true_label",
            "balanced_pair": True,
            "prediction_filter": False
        }])

        summary.to_csv(
            f"selected_pair_summary_{pair_tag}.csv",
            index=False
        )

        print("\n== Selección global SHAP-GLCD ==")
        print(
            f"Par: {self.class_mapping[self.j_sel]} "
            f"vs {self.class_mapping[self.r_sel]}"
        )
        print("Selección: etiqueta verdadera")
        print("Balanceo del par: sí")
        print("Filtro por predicción: no")
        print(f"Semilla: {self.RANDOM_STATE}")
        print(
            f"Disponibles en test -> "
            f"{self.class_mapping[self.j_sel]}: {len(idx_j)}, "
            f"{self.class_mapping[self.r_sel]}: {len(idx_r)}"
        )
        print(
            f"Seleccionadas -> "
            f"{self.class_mapping[self.j_sel]}: {n_per_class}, "
            f"{self.class_mapping[self.r_sel]}: {n_per_class}"
        )
        print(f"N global balanceado: {self.N_pair}")

        return selected_idx

    def calcular_delta(self):
        """
        Calcula:

            ΔS = SHAP(clase j) - SHAP(clase r)

        exclusivamente sobre una muestra balanceada del conjunto de prueba,
        seleccionada mediante la etiqueta verdadera de las clases j_sel y
        r_sel. No se filtra por predicción del modelo.
        """

        self._seleccionar_muestra_balanceada_par()

        self.delta = (
            self.shap_vals_pair[:, :, self.j_sel]
            - self.shap_vals_pair[:, :, self.r_sel]
        )

        print("\nΔSHAP calculado sobre el par balanceado:")

        print(
            f"delta shape: {self.delta.shape}"
        )

        print(
            f"Valores positivos favorecen "
            f"{self.class_mapping[self.j_sel]}; "
            f"valores negativos favorecen "
            f"{self.class_mapping[self.r_sel]}."
        )

    def ranking_global(self):
        """
        Calcula el ranking global de características según:

        - Mean |ΔSHAP|
        - Median |ΔSHAP|
        - Signed Mean ΔSHAP
        """

        mean_abs = np.mean(
            np.abs(self.delta),
            axis=0
        )

        median_abs = np.median(
            np.abs(self.delta),
            axis=0
        )

        signed_mean = np.mean(
            self.delta,
            axis=0
        )

        self.rank = pd.DataFrame({
            "Feature": self.features,
            "Mean_|ΔSHAP|": mean_abs,
            "Median_|ΔSHAP|": median_abs,
            "SignedMeanΔSHAP": signed_mean
        }).sort_values(
            "Mean_|ΔSHAP|",
            ascending=False
        ).reset_index(drop=True)

        self.rank.to_csv(
            "global_delta_shap_ranking.csv",
            index=False
        )

        print(
            "▶ CSV guardado: global_delta_shap_ranking.csv"
        )

        return self.rank

    # =========================================================
    # GRÁFICAS PRINCIPALES
    # =========================================================

    def plot_barras(self, dpi=600, show=True):
        """
        Genera las gráficas:

        - Signed Mean ΔS
        - Mean |ΔS|
        - Median |ΔS|
        """

        def barh_plot(
            df,
            col,
            title,
            fname,
            show_values=False,
            title_fontsize=38,
            label_fontsize=38,
            tick_fontsize=36,
            value_fontsize=28
        ):
            top_df = (
                df.head(self.TOP)
                .iloc[::-1]
                .copy()
            )

            vals = top_df[col].values.astype(float)

            labs = [
                self._format_feature_label(
                    f,
                    max_chars=24
                )
                for f in top_df["Feature"].astype(str).values
            ]

            fig, ax = plt.subplots(
                figsize=(18, 11)
            )

            bars = ax.barh(
                labs,
                vals,
                color=self.palette[:len(labs)]
            )

            if show_values:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))

                span = max(
                    abs(vmin),
                    abs(vmax),
                    1e-12
                )

                dx = 0.04 * span

                ax.set_xlim(
                    vmin - 6 * dx,
                    vmax + 6 * dx
                )

                bbox_kw = dict(
                    facecolor="white",
                    alpha=0.70,
                    edgecolor="none",
                    pad=1.0
                )

                for b in bars:
                    w = b.get_width()
                    y = (
                        b.get_y()
                        + b.get_height() / 2.0
                    )

                    if w >= 0:
                        ax.text(
                            w + dx,
                            y,
                            f"{w:.2f}",
                            va="center",
                            ha="left",
                            fontsize=self._fs(value_fontsize),
                            bbox=bbox_kw
                        )

                    else:
                        ax.text(
                            w - dx,
                            y,
                            f"{w:.3f}",
                            va="center",
                            ha="right",
                            fontsize=self._fs(value_fontsize),
                            bbox=bbox_kw
                        )

            ax.set_title(
                f"{title} – top {self.TOP}\n"
                f"{self.class_mapping[self.j_sel]} and "
                f"{self.class_mapping[self.r_sel]}",
                fontsize=self._fs(title_fontsize),
                pad=self._fs(12)
            )

            ax.set_xlabel(
                title,
                fontsize=self._fs(label_fontsize)
            )

            ax.set_ylabel(
                "Feature",
                fontsize=self._fs(label_fontsize)
            )

            ax.tick_params(
                axis="x",
                labelsize=self._fs(tick_fontsize)
            )

            ax.tick_params(
                axis="y",
                labelsize=self._fs(min(tick_fontsize, 30))
            )

            fig.tight_layout()

            fig.savefig(
                fname,
                dpi=dpi,
                bbox_inches="tight"
            )

            if show:
                plt.show()

            plt.close(fig)

        barh_plot(
            self.rank,
            "SignedMeanΔSHAP",
            "Signed Mean $\\Delta S$",
            "signed_mean_delta.png",
            show_values=True
        )

        barh_plot(
            self.rank,
            "Mean_|ΔSHAP|",
            "Mean $|\\Delta S|$",
            "mean_abs_delta.png",
            show_values=False,
            title_fontsize=49,
            label_fontsize=48,
            tick_fontsize=43
        )

        rank_med = self.rank.sort_values(
            "Median_|ΔSHAP|",
            ascending=False
        )

        barh_plot(
            rank_med,
            "Median_|ΔSHAP|",
            "Median $|\\Delta S|$",
            "median_abs_delta.png",
            show_values=False,
            title_fontsize=49,
            label_fontsize=48,
            tick_fontsize=43
        )

        print("✓ Barras PNG generadas.")

    def plot_barras_alineadas(self, dpi=600, show=True):
        """
        Genera el gráfico de medias condicionales:

        - media positiva de ΔS
        - media negativa de ΔS

        La leyenda se coloca fuera del área gráfica, centrada debajo,
        con espacio adicional para evitar que quede cerca del eje X.
        """

        delta = self.delta
        M = delta.shape[1]

        mean_pos = np.zeros(M)
        mean_neg = np.zeros(M)

        for j in range(M):
            v = delta[:, j]

            pos = v[v > 0]
            neg = v[v < 0]

            mean_pos[j] = (
                pos.mean()
                if pos.size
                else 0.0
            )

            mean_neg[j] = (
                neg.mean()
                if neg.size
                else 0.0
            )

        mean_abs = np.mean(
            np.abs(delta),
            axis=0
        )

        rank = pd.DataFrame({
            "Feature": self.features,
            "Mean_|ΔSHAP|": mean_abs,
            "MeanPosΔ": mean_pos,
            "MeanNegΔ": mean_neg
        })

        top = (
            rank
            .sort_values(
                "Mean_|ΔSHAP|",
                ascending=False
            )
            .head(self.TOP)
            .iloc[::-1]
            .reset_index(drop=True)
        )

        labels = [
            self._format_feature_label(
                f,
                max_chars=24
            )
            for f in top["Feature"].astype(str)
        ]

        pos_vals = top["MeanPosΔ"].values
        neg_vals = top["MeanNegΔ"].values
        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(
            figsize=(18, 14)
        )

        ax.barh(
            y_pos,
            pos_vals,
            color="#56B4E9",
            label=(
                f"$\\overline{{\\Delta S}}^{{+}}$ "
                f"(favours {self.class_mapping[self.j_sel]})"
            )
        )

        ax.barh(
            y_pos,
            neg_vals,
            color="#ECE133",
            label=(
                f"$\\overline{{\\Delta S}}^{{-}}$ "
                f"(favours {self.class_mapping[self.r_sel]})"
            )
        )

        ax.axvline(
            0,
            color="k",
            linewidth=1
        )

        ax.set_yticks(y_pos)

        ax.set_yticklabels(
            labels,
            fontsize=self._fs(30)
        )

        ax.tick_params(
            axis="x",
            labelsize=self._fs(32)
        )

        ax.set_xlabel(
            "Conditional Mean ΔS",
            fontsize=self._fs(34),
            labelpad=self._fs(10)
        )

        ax.set_ylabel(
            "Feature",
            fontsize=self._fs(34)
        )

        ax.set_title(
            f"Conditional Mean ΔS for top "
            f"{len(labels)} Features\n"
            f"{self.class_mapping[self.j_sel]} and "
            f"{self.class_mapping[self.r_sel]}",
            fontsize=self._fs(36),
            pad=self._fs(14)
        )

        max_pos = (
            np.max(pos_vals)
            if len(pos_vals)
            else 0.0
        )

        min_neg = (
            np.min(neg_vals)
            if len(neg_vals)
            else 0.0
        )

        span = max(
            max_pos,
            abs(min_neg),
            1e-12
        )

        dx = 0.04 * span

        ax.set_xlim(
            min_neg - 6 * dx,
            max_pos + 6 * dx
        )

        bbox_kw = dict(
            facecolor="white",
            alpha=0.70,
            edgecolor="none",
            pad=1.0
        )

        for yi, pv, nv in zip(
            y_pos,
            pos_vals,
            neg_vals
        ):
            if pv != 0:
                ax.text(
                    pv + dx,
                    yi,
                    f"{pv:.2f}",
                    va="center",
                    ha="left",
                    fontsize=self._fs(26),
                    bbox=bbox_kw
                )

            if nv != 0:
                ax.text(
                    nv - dx,
                    yi,
                    f"{nv:.3f}",
                    va="center",
                    ha="right",
                    fontsize=self._fs(26),
                    bbox=bbox_kw
                )

        fig.tight_layout()

        self._legend_below(
            ax=ax,
            fig=fig,
            ncol=2,
            fontsize=24,
            anchor_y=-0.24,
            bottom_margin=0.30
        )

        fig.savefig(
            "cond_mean_delta_aligned.png",
            dpi=dpi,
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close(fig)

    def plot_figura_B_100stack_weighted(
        self,
        dpi=600,
        fname="figB_weighted_100stack.png",
        show=True
    ):
        """
        Figura B / Dominance Plot:

            w+ = p+ * |mean(ΔS | ΔS > 0)|
            w- = p- * |mean(ΔS | ΔS < 0)|
            W+ = w+ / (w+ + w-)
            W- = 1 - W+

        La leyenda se ubica fuera del área gráfica, debajo y centrada,
        dejando mayor espacio respecto de la etiqueta del eje X.
        """

        delta = self.delta
        N, M = delta.shape

        mean_pos = np.zeros(M)
        mean_neg = np.zeros(M)

        n_pos = np.zeros(
            M,
            dtype=int
        )

        n_neg = np.zeros(
            M,
            dtype=int
        )

        for j in range(M):
            v = delta[:, j]

            pos = v[v > 0]
            neg = v[v < 0]

            n_pos[j] = pos.size
            n_neg[j] = neg.size

            mean_pos[j] = (
                pos.mean()
                if pos.size
                else 0.0
            )

            mean_neg[j] = (
                neg.mean()
                if neg.size
                else 0.0
            )

        p_pos = n_pos / N
        p_neg = n_neg / N

        w_pos = p_pos * np.abs(
            mean_pos
        )

        w_neg = p_neg * np.abs(
            mean_neg
        )

        w_tot = w_pos + w_neg

        eps = 1e-12

        W_pos = np.where(
            w_tot > 0,
            w_pos / (w_tot + eps),
            0.5
        )

        W_neg = 1.0 - W_pos

        df = pd.DataFrame({
            "Feature": self.features,
            "W_plus": W_pos,
            "W_minus": W_neg,
            "Mass_total": w_tot
        })

        top = (
            df
            .sort_values(
                "Mass_total",
                ascending=False
            )
            .head(self.TOP)
            .iloc[::-1]
            .reset_index(drop=True)
        )

        labels = [
            self._format_feature_label(
                f,
                max_chars=24
            )
            for f in top["Feature"].astype(str).tolist()
        ]

        w_minus = top["W_minus"].values
        w_plus = top["W_plus"].values

        y = np.arange(
            len(labels)
        )

        fig, ax = plt.subplots(
            figsize=(18, 14)
        )

        ax.barh(
            y,
            w_minus,
            color="#ECE133",
            label=(
                rf"$W^-_m$ "
                rf"(favours {self.class_mapping[self.r_sel]})"
            )
        )

        ax.barh(
            y,
            w_plus,
            left=w_minus,
            color="#56B4E9",
            label=(
                rf"$W^+_m$ "
                rf"(favours {self.class_mapping[self.j_sel]})"
            )
        )

        ax.set_yticks(y)

        ax.set_yticklabels(
            labels,
            fontsize=self._fs(30)
        )

        ax.set_xticks(
            [0, 0.25, 0.5, 0.75, 1.0]
        )

        ax.set_xticklabels(
            ["0%", "25%", "50%", "75%", "100%"],
            fontsize=self._fs(30)
        )

        ax.set_xlim(
            0,
            1
        )

        ax.axvline(
            0.5,
            color="k",
            linewidth=1,
            alpha=0.6
        )

        ax.set_xlabel(
            "Row-normalized weighted contribution",
            fontsize=self._fs(34),
            labelpad=self._fs(10)
        )

        ax.set_ylabel(
            "Feature",
            fontsize=self._fs(34)
        )

        ax.set_title(
            f"Dominance Plot (DP)\n"
            f"{self.class_mapping[self.j_sel]} and "
            f"{self.class_mapping[self.r_sel]}",
            fontsize=self._fs(36),
            pad=self._fs(14)
        )

        for yi, a, b in zip(
            y,
            w_minus,
            w_plus
        ):
            if a >= 0.12:
                ax.text(
                    a / 2,
                    yi,
                    f"{100 * a:.0f}%",
                    va="center",
                    ha="center",
                    fontsize=self._fs(24)
                )

            if b >= 0.12:
                ax.text(
                    a + b / 2,
                    yi,
                    f"{100 * b:.0f}%",
                    va="center",
                    ha="center",
                    fontsize=self._fs(24)
                )

        fig.tight_layout()

        self._legend_below(
            ax=ax,
            fig=fig,
            ncol=2,
            fontsize=24,
            anchor_y=-0.24,
            bottom_margin=0.30
        )

        fig.savefig(
            fname,
            dpi=dpi,
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close(fig)

        return df

    # =========================================================
    # UMBRALES Y ESTADÍSTICOS
    # =========================================================

    def _es_caracteristica_binaria_01(self, feature_name):
        """
        Determina si una característica está codificada exclusivamente
        mediante 0 y 1 en los datos originales.

        La detección se realiza antes del escalado para evitar clasificar
        erróneamente como binaria una variable cuantitativa transformada.
        """

        values = pd.to_numeric(
            self.X[feature_name],
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        if values.size == 0:
            return False

        unique_values = np.unique(values)

        return bool(
            np.all(
                np.isclose(
                    unique_values[:, None],
                    np.array([0.0, 1.0])[None, :]
                ).any(axis=1)
            )
        )

    @staticmethod
    def _cohens_d(a, b):
        """
        Cohen d con desviación estándar combinada para una
        característica cuantitativa.
        """

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        n1, n2 = len(a), len(b)

        if n1 < 2 or n2 < 2:
            return np.nan

        s1 = a.std(ddof=1)
        s2 = b.std(ddof=1)

        pooled_var = (
            (
                (n1 - 1) * s1 ** 2
                + (n2 - 1) * s2 ** 2
            )
            / (n1 + n2 - 2)
        )

        pooled_std = np.sqrt(pooled_var)

        if np.isclose(pooled_std, 0.0):
            return np.nan

        return float(
            (a.mean() - b.mean())
            / pooled_std
        )

    @staticmethod
    def _cohens_h_binaria(a, b):
        """
        Cohen h para una característica binaria codificada como 0/1.

        a corresponde a Cj y b a Cr. El signo positivo indica una
        proporción mayor de valores 1 en Cj; el signo negativo indica
        una proporción mayor en Cr.
        """

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) == 0 or len(b) == 0:
            return np.nan, np.nan, np.nan

        valid_values = np.r_[a, b]

        if not np.all(
            np.isclose(
                valid_values[:, None],
                np.array([0.0, 1.0])[None, :]
            ).any(axis=1)
        ):
            raise ValueError(
                "Cohen h solo puede aplicarse aquí a una "
                "característica binaria codificada como 0/1."
            )

        p_j = float(np.mean(a))
        p_r = float(np.mean(b))

        h_signed = (
            2.0 * np.arcsin(np.sqrt(p_j))
            - 2.0 * np.arcsin(np.sqrt(p_r))
        )

        return float(h_signed), p_j, p_r

    def _best_balanced_threshold_bidir(
        self,
        pos,
        neg
    ):
        """
        Encuentra el umbral que maximiza balanced accuracy probando:

        - pred = 1 si x >= t
        - pred = 1 si x <= t

        pos = valores de la clase positiva j_sel.
        neg = valores de la clase negativa r_sel.
        """

        if len(pos) == 0 or len(neg) == 0:
            return None, None, None

        y_true = np.r_[
            np.ones(len(pos)),
            np.zeros(len(neg))
        ]

        vals = np.r_[
            pos,
            neg
        ]

        uniq = np.unique(
            vals
        )

        if uniq.size == 1:
            return float(uniq[0]), 0.5, ">="

        best_ba = -1.0
        best_t = None
        best_dir = None

        for t in uniq:
            pred_ge = (
                vals >= t
            ).astype(int)

            ba_ge = balanced_accuracy_score(
                y_true,
                pred_ge
            )

            pred_le = (
                vals <= t
            ).astype(int)

            ba_le = balanced_accuracy_score(
                y_true,
                pred_le
            )

            if ba_ge > best_ba:
                best_ba = ba_ge
                best_t = float(t)
                best_dir = ">="

            if ba_le > best_ba:
                best_ba = ba_le
                best_t = float(t)
                best_dir = "<="

        return best_t, float(best_ba), best_dir

    def _boundary_pairwise(
        self,
        pos,
        neg,
        direction
    ):
        """
        Boundary coherente con la dirección elegida.
        """

        if (
            len(pos) == 0
            or len(neg) == 0
            or direction is None
        ):
            return None

        if direction == ">=":
            return 0.5 * (
                np.max(neg)
                + np.min(pos)
            )

        return 0.5 * (
            np.max(pos)
            + np.min(neg)
        )

    def plot_kde_top_features(self, dpi=600, show=True):
        """
        Genera KDE para las dos características principales del ranking
        y una KDE de la característica principal considerando todas
        las clases.

        Las leyendas KDE se mantienen dentro de la figura, conservando
        el comportamiento de la clase original.
        """

        print(
            "Clases presentes en self.y_te:",
            set(self.y_te)
        )

        print(
            "Claves en self.class_mapping:",
            set(self.class_mapping.keys())
        )

        if not hasattr(self, "pair_indices"):
            self._seleccionar_muestra_balanceada_par()

        # X_te_df se conserva completo para la figura contextual de todas
        # las clases. X_pair_df contiene únicamente el par balanceado y es
        # el utilizado en las KDE y umbrales pairwise.
        X_te_df = self.X_te.copy()
        X_pair_df = self.X_te_pair.copy()

        rank = self.rank

        # =====================================================
        # KDE para las dos primeras características del ranking
        # sobre la MISMA muestra balanceada usada por SHAP-GLCD.
        # =====================================================
        for f in rank["Feature"].iloc[:2]:
            fig, ax = plt.subplots(
                figsize=(12, 8)
            )

            vals_j = X_pair_df.loc[
                self.y_te_pair == self.j_sel,
                f
            ].values

            vals_r = X_pair_df.loc[
                self.y_te_pair == self.r_sel,
                f
            ].values

            # Como el scaler se ajusta solo con train,
            # algunos valores test pueden quedar fuera de [0, 1].
            x_min = min(
                0.0,
                float(np.min(vals_j)),
                float(np.min(vals_r))
            )

            x_max = max(
                1.0,
                float(np.max(vals_j)),
                float(np.max(vals_r))
            )

            plotted = False

            for c in [
                self.j_sel,
                self.r_sel
            ]:
                vals_c = X_pair_df.loc[
                    self.y_te_pair == c,
                    f
                ].dropna()

                if (
                    len(vals_c) < 2
                    or vals_c.nunique() < 2
                ):
                    print(
                        f"Advertencia: no se grafica KDE para "
                        f"{f}, clase {c}, porque tiene pocos "
                        "valores o varianza nula."
                    )

                    continue

                sns.kdeplot(
                    vals_c,
                    fill=True,
                    alpha=0.4,
                    color=(
                        "#56B4E9"
                        if c == self.j_sel
                        else "#ECE133"
                    ),
                    label=self.class_mapping[c],
                    cut=0,
                    clip=(x_min, x_max),
                    ax=ax
                )

                plotted = True

            t_opt, ba_opt, direction = (
                self._best_balanced_threshold_bidir(
                    vals_j,
                    vals_r
                )
            )

            t_bnd = self._boundary_pairwise(
                vals_j,
                vals_r,
                direction
            )

            if t_opt is not None:
                ax.axvline(
                    t_opt,
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.95,
                    color="tab:orange",
                    label=(
                        rf"$t_{{opt}}$={t_opt:.3f} "
                        rf"(BA={ba_opt:.2f}, "
                        rf"rule: $x{direction}t$)"
                    )
                )

            if t_bnd is not None:
                ax.axvline(
                    t_bnd,
                    linestyle=":",
                    linewidth=2.5,
                    alpha=0.95,
                    color="tab:green",
                    label=rf"$t_{{bnd}}$={t_bnd:.3f}"
                )

            ax.set_xlim(
                x_min,
                x_max
            )

            ax.tick_params(
                axis="x",
                labelsize=self._fs(26)
            )

            ax.tick_params(
                axis="y",
                labelsize=self._fs(26)
            )

            ax.set_xlabel(
                f"{f} (scaled)",
                fontsize=self._fs(30)
            )

            ax.set_ylabel(
                "Density",
                fontsize=self._fs(30)
            )

            ax.set_title(
                f"Density of {f} – "
                f"{self.class_mapping[self.j_sel]} and "
                f"{self.class_mapping[self.r_sel]}",
                fontsize=self._fs(30),
                pad=self._fs(10)
            )

            if plotted:
                ax.legend(
                    fontsize=self._fs(22),
                    loc="upper right"
                )

                fig.tight_layout()

                fig.savefig(
                    f"density_{f}.png",
                    dpi=dpi,
                    bbox_inches="tight"
                )

                if show:
                    plt.show()

            plt.close(fig)

        # =====================================================
        # KDE top-1 para todas las clases
        # =====================================================
        top_feat = rank.iloc[0]["Feature"]

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        vals_all = X_te_df[
            top_feat
        ].values

        x_min_all = min(
            0.0,
            float(np.min(vals_all))
        )

        x_max_all = max(
            1.0,
            float(np.max(vals_all))
        )

        plotted_any = False

        for c in self.class_names:
            vals_c = X_te_df.loc[
                self.y_te == c,
                top_feat
            ].dropna()

            if (
                len(vals_c) < 2
                or vals_c.nunique() < 2
            ):
                print(
                    f"Advertencia: no se grafica KDE all-classes "
                    f"para {top_feat}, clase {c}, porque tiene "
                    "pocos valores o varianza nula."
                )

                continue

            sns.kdeplot(
                vals_c,
                fill=True,
                alpha=0.25,
                color=self.class_color[c],
                label=self.class_mapping[c],
                cut=0,
                clip=(x_min_all, x_max_all),
                ax=ax
            )

            plotted_any = True

        ax.set_title(
            f"Density of {top_feat} across all classes",
            fontsize=self._fs(30),
            pad=self._fs(10)
        )

        ax.set_xlabel(
            f"{top_feat} (scaled)",
            fontsize=self._fs(30)
        )

        ax.set_ylabel(
            "Density",
            fontsize=self._fs(30)
        )

        ax.set_xlim(
            x_min_all,
            x_max_all
        )

        ax.tick_params(
            axis="x",
            labelsize=self._fs(26)
        )

        ax.tick_params(
            axis="y",
            labelsize=self._fs(26)
        )

        if plotted_any:
            ax.legend(
                ncol=2,
                fontsize=self._fs(18)
            )

            fig.tight_layout()

            fig.savefig(
                "kde_all_classes.png",
                dpi=dpi,
                bbox_inches="tight"
            )

            if show:
                plt.show()

        plt.close(fig)

    def estadisticos_top1(self):
        """
        Calcula medidas complementarias sobre la característica top-1
        identificada por Mean |ΔS|.

        Todos los cálculos pairwise utilizan la MISMA muestra balanceada,
        seleccionada por etiqueta verdadera, que se empleó en SHAP-GLCD.

        Se reportan:
        - Cohen d para una característica cuantitativa;
        - Cohen h para una característica binaria 0/1;
        - estadístico D del test Kolmogorov--Smirnov y su p-value;
        - umbral bidireccional que maximiza balanced accuracy;
        - boundary threshold coherente con la dirección seleccionada.
        """

        if not hasattr(self, "pair_indices"):
            self._seleccionar_muestra_balanceada_par()

        rank = self.rank

        top_feat = rank.iloc[0]["Feature"]
        top_sign = rank.iloc[0]["SignedMeanΔSHAP"]

        favours = (
            self.class_mapping[self.j_sel]
            if top_sign > 0
            else self.class_mapping[self.r_sel]
        )

        print(
            f"\n=== Top-1 feature: {top_feat} "
            f"(favorece {favours}) ==="
        )

        X_pair_df = self.X_te_pair.copy()

        vals_j = X_pair_df.loc[
            self.y_te_pair == self.j_sel,
            top_feat
        ].values

        vals_r = X_pair_df.loc[
            self.y_te_pair == self.r_sel,
            top_feat
        ].values

        feature_is_binary = self._es_caracteristica_binaria_01(
            top_feat
        )

        if feature_is_binary:
            # Para Cohen h se utilizan los valores originales 0/1 de
            # exactamente la misma muestra pairwise balanceada.
            vals_j_effect = self.X_te_pair_raw.loc[
                self.y_te_pair == self.j_sel,
                top_feat
            ].values

            vals_r_effect = self.X_te_pair_raw.loc[
                self.y_te_pair == self.r_sel,
                top_feat
            ].values

            effect_signed, proportion_j, proportion_r = (
                self._cohens_h_binaria(
                    vals_j_effect,
                    vals_r_effect
                )
            )

            effect_metric = "Cohen h"
            effect_symbol = "h"
            feature_type = "binary_0_1"

        else:
            effect_signed = self._cohens_d(
                vals_j,
                vals_r
            )

            proportion_j = np.nan
            proportion_r = np.nan
            effect_metric = "Cohen d"
            effect_symbol = "d"
            feature_type = "quantitative"

        ks_result = ks_2samp(
            vals_j,
            vals_r
        )

        ks_statistic = float(
            ks_result.statistic
        )

        p_ks = float(
            ks_result.pvalue
        )

        thr_BA, ba, direction = (
            self._best_balanced_threshold_bidir(
                vals_j,
                vals_r
            )
        )

        print("\n=== Top-1 statistics ===")

        print(
            f"Feature type            : {feature_type}"
        )

        if np.isnan(effect_signed):
            print(
                f"{effect_metric:<24}: nan "
                "(no computable por varianza cero)"
            )
            print(
                f"Cohen |{effect_symbol}|"
                f"{'':<14}: nan"
            )
        else:
            print(
                f"{effect_metric:<24}: "
                f"{effect_signed:.3f}"
            )
            print(
                f"Cohen |{effect_symbol}|"
                f"{'':<14}: "
                f"{abs(effect_signed):.3f}"
            )

            if feature_is_binary:
                print(
                    f"Proportion "
                    f"{self.class_mapping[self.j_sel]:<12}: "
                    f"{proportion_j:.6f}"
                )

                print(
                    f"Proportion "
                    f"{self.class_mapping[self.r_sel]:<12}: "
                    f"{proportion_r:.6f}"
                )

        print(
            f"KS statistic D           : {ks_statistic:.3f}"
        )

        print(
            f"p-value KS               : {p_ks:.3e}"
        )

        print(
            f"Best threshold (BA)      : {thr_BA}"
        )

        if ba is None:
            print(
                "Balanced-Acc @thr        : None"
            )

            return

        print(
            f"Balanced-Acc @thr        : {ba:.3f}\n"
        )

        # =====================================================
        # Precision-recall sobre la muestra balanceada j vs r
        # =====================================================
        x_pair = X_pair_df[
            top_feat
        ].values

        y_true_pair = (
            self.y_te_pair == self.j_sel
        ).astype(int)

        thr_vals = np.sort(
            np.unique(x_pair)
        )

        records = []

        for t in thr_vals:
            if direction == "<=":
                pred = (
                    x_pair <= t
                ).astype(int)
            else:
                pred = (
                    x_pair >= t
                ).astype(int)

            records.append({
                "thr": t,
                "precision": precision_score(
                    y_true_pair,
                    pred,
                    zero_division=0
                ),
                "recall": recall_score(
                    y_true_pair,
                    pred,
                    zero_division=0
                )
            })

        pr_df = pd.DataFrame(
            records
        )

        pr_df.to_csv(
            "pr_thresholds.csv",
            index=False
        )

        print(
            "Tabla P–R completa → pr_thresholds.csv"
        )

        PREC_REQ = 0.90

        meet = pr_df.loc[
            pr_df["precision"] >= PREC_REQ
        ]

        if not meet.empty:
            row = meet.iloc[0]

            print(
                f"\nFirst thr with precision ≥ "
                f"{PREC_REQ:.0%}:"
            )

            print(
                f"thr={row['thr']:.6f} | "
                f"precision={row['precision']:.3f} | "
                f"recall={row['recall']:.3f}"
            )

        else:
            print(
                f"\nNo threshold reaches precision ≥ "
                f"{PREC_REQ:.0%}\n"
            )

        # =====================================================
        # Boundary threshold pairwise
        # =====================================================
        boundary = self._boundary_pairwise(
            vals_j,
            vals_r,
            direction
        )

        if boundary is None:
            print(
                "No se pudo calcular boundary threshold."
            )

            return

        if direction == "<=":
            pred_pair = (
                x_pair <= boundary
            ).astype(int)
        else:
            pred_pair = (
                x_pair >= boundary
            ).astype(int)

        bal_rule = balanced_accuracy_score(
            y_true_pair,
            pred_pair
        )

        print(
            "=== Boundary threshold (pairwise) ==="
        )

        if direction == ">=":
            print(
                f"max {self.class_mapping[self.r_sel]}: "
                f"{vals_r.max():.6f}"
            )
            print(
                f"min {self.class_mapping[self.j_sel]}: "
                f"{vals_j.min():.6f}"
            )
        else:
            print(
                f"max {self.class_mapping[self.j_sel]}: "
                f"{vals_j.max():.6f}"
            )
            print(
                f"min {self.class_mapping[self.r_sel]}: "
                f"{vals_r.min():.6f}"
            )

        print(
            f"boundary thr       : {boundary:.6f}"
        )

        print(
            f"Balanced-Acc @bnd  : {bal_rule:.3f}\n"
        )

        top1_stats = pd.DataFrame([{
            "feature": top_feat,
            "class_j": self.j_sel,
            "class_r": self.r_sel,
            "N_pair": int(self.N_pair),
            "feature_type": feature_type,
            "effect_size_metric": effect_metric,
            "effect_size_symbol": effect_symbol,
            "effect_size_signed": (
                float(effect_signed)
                if not np.isnan(effect_signed)
                else np.nan
            ),
            "effect_size_absolute": (
                float(abs(effect_signed))
                if not np.isnan(effect_signed)
                else np.nan
            ),
            "proportion_class_j": proportion_j,
            "proportion_class_r": proportion_r,
            "ks_statistic_D": ks_statistic,
            "ks_pvalue": p_ks,
            "t_opt": float(thr_BA) if thr_BA is not None else np.nan,
            "t_opt_direction": direction,
            "balanced_accuracy_t_opt": float(ba),
            "t_bnd": float(boundary),
            "balanced_accuracy_t_bnd": float(bal_rule)
        }])

        top1_stats.to_csv(
            f"top1_statistics_{self.j_sel}_vs_{self.r_sel}.csv",
            index=False
        )

    # =========================================================
    # MÉTRICAS DEL MODELO
    # =========================================================

    def metricas_modelo(self):
        """
        Calcula las métricas finales sobre el 30% de prueba:

        - Accuracy
        - Balanced accuracy
        - F1-score macro
        - MCC
        - Macro-AUC OvO
        - Matriz de confusión
        """

        y_pred = self.modelo_final.predict(
            self.X_te
        )

        y_proba = self.modelo_final.predict_proba(
            self.X_te
        )

        acc = accuracy_score(
            self.y_te,
            y_pred
        )

        bal_acc = balanced_accuracy_score(
            self.y_te,
            y_pred
        )

        f1_macro = f1_score(
            self.y_te,
            y_pred,
            average="macro"
        )

        mcc = matthews_corrcoef(
            self.y_te,
            y_pred
        )

        auc_mac = self._calculate_auc(
            self.y_te,
            y_proba
        )

        print(
            "\n=== Model metrics (30 % test) ==="
        )

        print(
            f"Accuracy           : {acc:.4f}"
        )

        print(
            f"Balanced accuracy  : {bal_acc:.4f}"
        )

        print(
            f"F1-score macro     : {f1_macro:.4f}"
        )

        print(
            f"MCC                : {mcc:.4f}"
        )

        print(
            f"Macro-AUC (OvO)    : {auc_mac:.4f}"
        )

        print(
            "Confusion matrix:"
        )

        print(
            confusion_matrix(
                self.y_te,
                y_pred
            )
        )

    # =========================================================
    # ANÁLISIS ADICIONAL DE UNA CARACTERÍSTICA
    # =========================================================

    def extra_feature_analysis(
        self,
        feature_name
    ):
        ci = self.delta[
            :,
            self.features.index(feature_name)
        ]

        n_total = len(ci)

        n_pos = np.sum(
            ci > 0
        )

        n_neg = np.sum(
            ci < 0
        )

        mean_pos = (
            ci[ci > 0].mean()
            if n_pos
            else 0.0
        )

        mean_neg = (
            ci[ci < 0].mean()
            if n_neg
            else 0.0
        )

        signed_mean = ci.mean()

        print(
            f"n_+={n_pos}, "
            f"n_-={n_neg}, "
            f"N={n_total}"
        )

        print(
            f"p_+={n_pos / n_total:.2f}, "
            f"p_-={n_neg / n_total:.2f}"
        )

        print(
            f"mean_+={mean_pos:.2f}, "
            f"mean_-={mean_neg:.2f}"
        )

        print(
            f"signed_mean={signed_mean:.2f}"
        )

    # =========================================================
    # PIPELINE PRINCIPAL
    # =========================================================

    def run_all(self, shap_mode="auto"):
        """
        Ejecuta el flujo completo del análisis.

        shap_mode:
            - "auto":
              Si existe el archivo SHAP, lo carga.
              Si no existe, lo calcula y lo guarda.

            - "always":
              Siempre recalcula y guarda los SHAP values.

            - "only_load":
              Solo carga el archivo SHAP previamente generado.

        Nota:
            El Dominance Plot se invoca desde cada main para conservar
            el nombre personalizado por dataset y par de clases.
        """

        self.validacion_cruzada()
        self.entrenar_modelo_final()

        if shap_mode == "auto":
            if os.path.exists(
                self.shap_pickle
            ):
                self.cargar_shap()
            else:
                self.calcular_y_guardar_shap()

        elif shap_mode == "always":
            self.calcular_y_guardar_shap()

        elif shap_mode == "only_load":
            self.cargar_shap()

        else:
            raise ValueError(
                "shap_mode debe ser 'auto', "
                "'always', o 'only_load'"
            )

        self.calcular_delta()
        self.ranking_global()

        self.plot_barras(
            dpi=600,
            show=True
        )

        self.plot_barras_alineadas(
            dpi=600,
            show=True
        )

        self.plot_kde_top_features(
            dpi=600,
            show=True
        )

        self.estadisticos_top1()
        self.metricas_modelo()

        if "color_intensity" in self.features:
            print(
                "\n=== Extra: color_intensity analysis ==="
            )

            self.extra_feature_analysis(
                "color_intensity"
            )
