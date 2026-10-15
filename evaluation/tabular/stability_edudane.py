#duDane / EduFormalDANE
#
# Evaluación separada de confiabilidad de SHAP-LCD usando el
# MODELO FINAL DEFINITIVO entrenado con:
#
#   - split externo 80/20 estratificado
#   - validación cruzada de 7 pliegues sin fuga
#   - RandomOverSampler SOLO en entrenamiento
#   - Random Forest optimizado
#   - semilla 42
#
# Este programa NO reentrena el modelo.
# Este programa NO recalcula selección de variables.
# Este programa NO ajusta nuevamente encoder ni scaler.
#
# Carga los artefactos definitivos y calcula:
#
#   1) Stability_S
#   2) Consistency_C_imp
#   3) Coverage_imp
#   4) PredKeep_imp
#
# Protocolo de perturbaciones válidas:
#
#   - Las perturbaciones se generan en el ESPACIO ORIGINAL.
#   - Numéricas continuas:
#       ruido gaussiano local con sigma = 0.005 del rango observado
#       en TRAIN y clipping al min/max de TRAIN.
#   - Numéricas discretas/enteras:
#       mismo ruido local, seguido de proyección al valor válido
#       observado en TRAIN más cercano.
#   - Binarias:
#       permanecen en estados válidos observados en TRAIN;
#       cambio con probabilidad 0.05.
#   - Categóricas:
#       cambio con probabilidad 0.05; la nueva categoría se toma
#       únicamente entre categorías observadas en TRAIN y de acuerdo
#       con su distribución empírica.
#   - Ordinales:
#       se mantienen dentro de los valores observados válidos;
#       cambio con probabilidad 0.05.
#
# Después de perturbar:
#
#   raw perturbado
#      -> MISMAS variables seleccionadas
#      -> MISMO encoder ajustado
#      -> MISMO MinMaxScaler ajustado
#      -> MISMO modelo final
#      -> nuevo SHAP
#      -> nuevo SHAP-LCD
#
# Selección de instancias:
#
#   - etiqueta VERDADERA (filter_mode="true")
#   - muestra balanceada entre las dos clases
#   - 50 instancias por par = 25 + 25 cuando hay disponibilidad
#   - incluye predicciones correctas e incorrectas
#   - semilla 42
#
# Parámetros:
#
#   N = 50 instancias por par
#   M = 40 perturbaciones por instancia
#   sigma = 0.005
#   p_change = 0.05
#   tau = 0.95
#   top_k = 10
#   normalize = "l2"
#   seed = 42
#
# IMPORTANTE:
#   Stability_S menor = explicación más estable.
#
#   Consistency_C_imp = proporción de perturbaciones que satisfacen
#   la implicación:
#
#       similitud(exp_base, exp_pert) >= tau
#              =>
#       predicción perturbada = predicción base
#
#   Coverage_imp = proporción de perturbaciones donde se cumple
#   el antecedente de similitud.
#
#   PredKeep_imp = proporción de perturbaciones donde se conserva
#   la predicción base.
#
# ============================================================

import os
import sys
import json
import time
import random
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import shap

from sklearn.ensemble import RandomForestClassifier

# Necesario para deserializar correctamente el processor guardado.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.preprocessing import ProcessingSelectionPipeline


# ============================================================
# CLASE DEL MODELO
# ============================================================
#
# Debe existir con el MISMO nombre que cuando se guardó con joblib.
# No se usa para reentrenar. Su definición permite deserializar
# correctamente el objeto random_forest_optimizado.joblib si fue
# guardado desde un main donde esta clase pertenecía a __main__.
# ============================================================

class MyOptimizedRandomForestClassifier(RandomForestClassifier):
    def __init__(self, random_state=42):
        super().__init__(
            n_estimators=800,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features=0.5,
            class_weight=None,
            n_jobs=-1,
            random_state=random_state
        )


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_STATE = 42

N_INSTANCES = 50
M_PERTURB = 40

SIGMA_RANGE = 0.005
P_CHANGE = 0.05

TAU = 0.95
TOP_K = 10
NORMALIZE = "l2"

FILTER_MODE = "true"
BALANCED_SAMPLING = True

# ------------------------------------------------------------
# Par definitivo del análisis SHAP-GLCD para EduFormalDANE
# ------------------------------------------------------------
#
#   (0, 3) = Diario vs Ocasional
#
# Se usa este mismo par en Stability/Consistency para mantener
# coherencia con el análisis global principal del artículo.
# ------------------------------------------------------------

PAIRS_TO_EVALUATE = [
    (0, 3),
]

# Overrides opcionales para el tipo de perturbación.
# Por defecto, las variables "numeric" que toman exclusivamente
# valores enteros en TRAIN se consideran discretas/enteras.
FORCE_CONTINUOUS = set()
FORCE_DISCRETE = set()
FORCE_BINARY = set()
FORCE_CATEGORICAL = set()
FORCE_ORDINAL = set()

# Verificación de alineación raw -> processor -> scaler -> X_test_final
N_ALIGNMENT_CHECK = 20
ALIGNMENT_ATOL = 1e-9
ALIGNMENT_RTOL = 1e-9


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = PROJECT_ROOT

OUTPUT_ROOT = PROJECT_ROOT / "artifacts"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "edudane_smoke_test"

METRICS_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "edudane_smoke_test"
)
METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "random_forest_optimizado.joblib"
PROCESSOR_PATH = ARTIFACT_DIR / "processing_selection_pipeline.joblib"
SCALER_PATH = ARTIFACT_DIR / "minmax_scaler.joblib"

X_TRAIN_RAW_PATH = ARTIFACT_DIR / "X_train_raw_before_processing.csv"
X_TEST_RAW_PATH = ARTIFACT_DIR / "X_test_raw_before_processing.csv"

X_TEST_FINAL_PATH = ARTIFACT_DIR / "X_test_final.csv"

Y_TEST_PATH = ARTIFACT_DIR / "y_test_final.csv"
Y_TEST_ORIGINAL_PATH = ARTIFACT_DIR / "y_test_original_split.csv"

TEST_ORIGINAL_INDICES_PATH = ARTIFACT_DIR / "test_original_indices.csv"

FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.csv"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"

SHAP_3D_PATH = ARTIFACT_DIR / "shap_values_3d.npy"


# ============================================================
# SEMILLA
# ============================================================

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# HELPERS GENERALES
# ============================================================

def safe_name(text):
    text = str(text)
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")


def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "\nFaltan archivos necesarios:\n - "
            + "\n - ".join(missing)
            + "\n\nVerifica ARTIFACT_DIR."
        )


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}

    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}

    raise ValueError("Formato inesperado en class_names.json")


def load_y_vector(path):
    df = pd.read_csv(path)
    if df.shape[1] != 1:
        raise ValueError(
            f"{path.name} debería tener una sola columna y tiene {df.shape[1]}."
        )
    return df.iloc[:, 0].to_numpy()


def read_raw_split(path):
    # Los raw se guardaron con index=True.
    # index_col=0 recupera el índice original sin convertirlo en predictor.
    df = pd.read_csv(path, index_col=0)

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )

    return df


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    den = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / den)


def sparsify_and_normalize(v, top_k=10, normalize="l2"):
    v = np.asarray(v, dtype=float).copy()

    if top_k is not None and top_k < v.size:
        idx = np.argsort(np.abs(v))[::-1][:top_k]
        sparse = np.zeros_like(v)
        sparse[idx] = v[idx]
        v = sparse

    if normalize == "l1":
        v = v / (np.sum(np.abs(v)) + 1e-12)
    elif normalize == "l2":
        v = v / (np.linalg.norm(v) + 1e-12)
    elif normalize in {None, "none"}:
        pass
    else:
        raise ValueError("NORMALIZE debe ser 'l1', 'l2' o 'none'.")

    return v


def normalize_shap_to_3d(shap_raw, n_rows, n_features, n_classes):
    """
    Devuelve SHAP con forma:
        (n_rows, n_features, n_classes)
    """

    if isinstance(shap_raw, list):
        if len(shap_raw) != n_classes:
            raise ValueError(
                f"SHAP list contiene {len(shap_raw)} clases; "
                f"se esperaban {n_classes}."
            )

        arr = np.stack(
            [np.asarray(x, dtype=float) for x in shap_raw],
            axis=-1
        )

        if arr.shape != (n_rows, n_features, n_classes):
            raise ValueError(
                f"Forma SHAP inesperada después de stack: {arr.shape}"
            )
        return arr

    arr = np.asarray(shap_raw, dtype=float)

    if arr.shape == (n_rows, n_features, n_classes):
        return arr

    if arr.shape == (n_classes, n_rows, n_features):
        return np.transpose(arr, (1, 2, 0))

    if n_rows == 1 and arr.shape == (n_features, n_classes):
        return arr.reshape(1, n_features, n_classes)

    raise ValueError(
        "No fue posible normalizar SHAP a 3D. "
        f"Forma recibida: {arr.shape}; "
        f"esperada: ({n_rows}, {n_features}, {n_classes})."
    )


def transform_raw_df(raw_df, processor, scaler):
    """
    Aplica EXACTAMENTE:
      variables seleccionadas -> encoder ya ajustado -> scaler ya ajustado.

    No hace fit de ningún componente.
    """

    selected_cols = list(processor.get_selected_original_features())

    missing = [c for c in selected_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(
            "Faltan variables seleccionadas en el DataFrame raw:\n"
            + "\n".join(missing)
        )

    raw_sel = raw_df[selected_cols].copy()

    x_proc = processor.preprocessor_.transform(raw_sel)
    x_final = scaler.transform(x_proc)

    return np.asarray(x_final, dtype=float)


def resolve_pair_positions(model, pair):
    model_classes = list(model.classes_)

    ci_lbl, cj_lbl = pair

    if ci_lbl not in model_classes or cj_lbl not in model_classes:
        raise ValueError(
            f"Par {pair} no compatible con model.classes_={model_classes}"
        )

    ci_pos = model_classes.index(ci_lbl)
    cj_pos = model_classes.index(cj_lbl)

    return ci_pos, cj_pos, ci_lbl, cj_lbl


# ============================================================
# CLASIFICACIÓN DEL TIPO DE PERTURBACIÓN
# ============================================================

def is_integer_valued_numeric(series):
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)

    if vals.size == 0:
        return False

    return bool(np.all(np.isclose(vals, np.round(vals), atol=1e-12)))


def build_perturbation_profiles(X_train_raw, processor):
    """
    Construye el perfil de perturbación SOLO para las variables
    originales seleccionadas que realmente entran al modelo.
    """

    selected = list(processor.get_selected_original_features())

    rows = []
    profiles = {}

    for feature in selected:
        if feature not in X_train_raw.columns:
            raise ValueError(
                f"La variable seleccionada '{feature}' no está en X_train_raw."
            )

        base_type = processor.feature_types_.get(feature, None)
        series = X_train_raw[feature]

        # Overrides explícitos
        if feature in FORCE_CONTINUOUS:
            perturb_type = "continuous"
        elif feature in FORCE_DISCRETE:
            perturb_type = "discrete_numeric"
        elif feature in FORCE_BINARY:
            perturb_type = "binary"
        elif feature in FORCE_CATEGORICAL:
            perturb_type = "categorical"
        elif feature in FORCE_ORDINAL:
            perturb_type = "ordinal"
        else:
            if base_type == "binary":
                perturb_type = "binary"
            elif base_type == "categorical":
                perturb_type = "categorical"
            elif base_type == "ordinal":
                perturb_type = "ordinal"
            elif base_type == "numeric":
                if is_integer_valued_numeric(series):
                    perturb_type = "discrete_numeric"
                else:
                    perturb_type = "continuous"
            else:
                raise ValueError(
                    f"Tipo de variable no reconocido para {feature}: {base_type}"
                )

        profile = {
            "feature": feature,
            "processor_type": base_type,
            "perturbation_type": perturb_type,
        }

        if perturb_type in {"continuous", "discrete_numeric"}:
            vals = (
                pd.to_numeric(series, errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )

            if vals.size == 0:
                raise ValueError(
                    f"No hay valores numéricos válidos en TRAIN para {feature}."
                )

            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            vrange = float(vmax - vmin)
            observed = np.unique(vals)

            profile.update({
                "min": vmin,
                "max": vmax,
                "range": vrange,
                "sigma_abs": float(SIGMA_RANGE * vrange),
                "observed_values": observed,
                "n_valid_values": int(len(observed)),
            })

        else:
            valid = series.dropna()
            counts = valid.value_counts(dropna=True)

            if counts.empty:
                raise ValueError(
                    f"No hay estados/categorías válidas en TRAIN para {feature}."
                )

            values = counts.index.to_numpy(dtype=object)
            probs = (counts / counts.sum()).to_numpy(dtype=float)

            profile.update({
                "valid_values": values,
                "valid_probs": probs,
                "n_valid_values": int(len(values)),
            })

        profiles[feature] = profile

        rows.append({
            "feature": feature,
            "processor_type": base_type,
            "perturbation_type": perturb_type,
            "n_valid_values": profile["n_valid_values"],
            "train_min": profile.get("min", np.nan),
            "train_max": profile.get("max", np.nan),
            "train_range": profile.get("range", np.nan),
            "sigma_abs": profile.get("sigma_abs", np.nan),
            "p_change": (
                P_CHANGE
                if perturb_type in {"binary", "categorical", "ordinal"}
                else np.nan
            ),
        })

    report = pd.DataFrame(rows)

    return profiles, report


# ============================================================
# GENERACIÓN DE PERTURBACIONES VÁLIDAS
# ============================================================

def draw_empirical_alternative(current_value, valid_values, valid_probs, rng):
    valid_values = np.asarray(valid_values, dtype=object)
    valid_probs = np.asarray(valid_probs, dtype=float)

    if valid_values.size == 0:
        return current_value

    if pd.isna(current_value):
        probs = valid_probs / valid_probs.sum()
        return rng.choice(valid_values, p=probs)

    mask = np.array(
        [not (v == current_value) for v in valid_values],
        dtype=bool
    )

    alternatives = valid_values[mask]
    probs = valid_probs[mask]

    if alternatives.size == 0:
        return current_value

    probs = probs / probs.sum()
    return rng.choice(alternatives, p=probs)


def nearest_observed_value(candidate, observed):
    observed = np.asarray(observed, dtype=float)

    if observed.size == 0:
        return candidate

    idx = int(np.argmin(np.abs(observed - candidate)))
    return float(observed[idx])


def perturb_one_raw_row(base_row, profiles, rng):
    """
    Genera UNA perturbación válida de una instancia raw.
    Solo modifica variables originales seleccionadas.
    """

    row = base_row.copy()

    for feature, profile in profiles.items():
        ptype = profile["perturbation_type"]
        current = row[feature]

        # --------------------------------------------------------
        # Numérica continua
        # --------------------------------------------------------
        if ptype == "continuous":
            if pd.isna(current):
                continue

            current_num = float(current)
            sigma_abs = profile["sigma_abs"]

            if sigma_abs <= 0:
                new_value = current_num
            else:
                new_value = current_num + rng.normal(0.0, sigma_abs)

            new_value = float(
                np.clip(
                    new_value,
                    profile["min"],
                    profile["max"]
                )
            )

            row[feature] = new_value

        # --------------------------------------------------------
        # Numérica discreta / entera
        # --------------------------------------------------------
        elif ptype == "discrete_numeric":
            if pd.isna(current):
                continue

            current_num = float(current)
            sigma_abs = profile["sigma_abs"]

            if sigma_abs <= 0:
                candidate = current_num
            else:
                candidate = current_num + rng.normal(0.0, sigma_abs)

            candidate = float(
                np.clip(
                    candidate,
                    profile["min"],
                    profile["max"]
                )
            )

            row[feature] = nearest_observed_value(
                candidate,
                profile["observed_values"]
            )

        # --------------------------------------------------------
        # Binaria / categórica / ordinal
        # --------------------------------------------------------
        elif ptype in {"binary", "categorical", "ordinal"}:
            if rng.random() < P_CHANGE:
                row[feature] = draw_empirical_alternative(
                    current,
                    profile["valid_values"],
                    profile["valid_probs"],
                    rng
                )

        else:
            raise ValueError(
                f"Tipo de perturbación no reconocido: {ptype}"
            )

    return row


def generate_valid_perturbations(base_row, profiles, M, seed):
    rng = np.random.default_rng(seed)

    rows = [
        perturb_one_raw_row(base_row, profiles, rng)
        for _ in range(M)
    ]

    return pd.DataFrame(rows)


# ============================================================
# SELECCIÓN BALANCEADA POR ETIQUETA VERDADERA
# ============================================================

def select_balanced_true_pair_indices(
    y_test,
    pair,
    n_instances=50,
    seed=42
):
    ci_lbl, cj_lbl = pair

    idx_i = np.where(y_test == ci_lbl)[0]
    idx_j = np.where(y_test == cj_lbl)[0]

    if len(idx_i) == 0 or len(idx_j) == 0:
        raise ValueError(
            f"No hay suficientes instancias para el par {pair}. "
            f"Disponibles: {len(idx_i)} y {len(idx_j)}."
        )

    if n_instances is None:
        n_each = min(len(idx_i), len(idx_j))
    else:
        # Para mantener balance exacto se usa el total par.
        n_each_requested = int(n_instances) // 2
        n_each = min(n_each_requested, len(idx_i), len(idx_j))

    if n_each <= 0:
        raise ValueError(f"No se pudo formar muestra balanceada para {pair}.")

    rng = np.random.default_rng(seed)

    sel_i = rng.choice(idx_i, size=n_each, replace=False)
    sel_j = rng.choice(idx_j, size=n_each, replace=False)

    selected = np.concatenate([sel_i, sel_j])
    rng.shuffle(selected)

    return selected.astype(int)


# ============================================================
# EVALUACIÓN DE UNA INSTANCIA
# ============================================================

def evaluate_one_instance(
    idx_test,
    pair,
    instance_position,
    X_test_raw,
    X_test_final,
    y_test,
    original_indices,
    model,
    tree_explainer,
    processor,
    scaler,
    profiles,
    shap_base_3d,
    class_names,
):
    ci_pos, cj_pos, ci_lbl, cj_lbl = resolve_pair_positions(model, pair)

    # --------------------------------------------------------
    # Instancia original
    # --------------------------------------------------------
    base_raw = X_test_raw.iloc[idx_test].copy()
    base_x = np.asarray(X_test_final[idx_test], dtype=float)

    base_pred = model.predict(base_x.reshape(1, -1))[0]
    base_prob = model.predict_proba(base_x.reshape(1, -1))[0]

    base_delta = (
        shap_base_3d[idx_test, :, ci_pos]
        - shap_base_3d[idx_test, :, cj_pos]
    )

    base_exp = sparsify_and_normalize(
        base_delta,
        top_k=TOP_K,
        normalize=NORMALIZE
    )

    # --------------------------------------------------------
    # Perturbaciones válidas EN ESPACIO ORIGINAL
    # --------------------------------------------------------
    #
    # Regla común del protocolo:
    #
    #     seed_instancia = seed + t
    #
    # donde t es la posición local de la instancia seleccionada.
    # --------------------------------------------------------
    local_seed = (
        RANDOM_STATE
        + int(instance_position)
    )

    raw_pert_df = generate_valid_perturbations(
        base_row=base_raw,
        profiles=profiles,
        M=M_PERTURB,
        seed=local_seed
    )

    # --------------------------------------------------------
    # MISMO procesamiento ya ajustado
    # --------------------------------------------------------
    X_pert = transform_raw_df(
        raw_df=raw_pert_df,
        processor=processor,
        scaler=scaler
    )

    # --------------------------------------------------------
    # Predicciones perturbadas
    # --------------------------------------------------------
    pred_pert = model.predict(X_pert)

    # --------------------------------------------------------
    # SHAP y SHAP-LCD perturbados
    # --------------------------------------------------------
    shap_raw_pert = tree_explainer.shap_values(X_pert)

    shap_pert_3d = normalize_shap_to_3d(
        shap_raw=shap_raw_pert,
        n_rows=X_pert.shape[0],
        n_features=X_pert.shape[1],
        n_classes=len(model.classes_)
    )

    similarities = []
    squared_distances = []
    pred_keep_flags = []
    antecedent_flags = []
    implication_ok_flags = []

    for m in range(M_PERTURB):
        delta_pert = (
            shap_pert_3d[m, :, ci_pos]
            - shap_pert_3d[m, :, cj_pos]
        )

        exp_pert = sparsify_and_normalize(
            delta_pert,
            top_k=TOP_K,
            normalize=NORMALIZE
        )

        sim = cosine_similarity(base_exp, exp_pert)
        sq_dist = float(np.linalg.norm(exp_pert - base_exp) ** 2)

        A = bool(sim >= TAU)
        B = bool(pred_pert[m] == base_pred)

        implication_ok = (not A) or B

        similarities.append(sim)
        squared_distances.append(sq_dist)
        pred_keep_flags.append(B)
        antecedent_flags.append(A)
        implication_ok_flags.append(implication_ok)

    # --------------------------------------------------------
    # Métricas por instancia
    # --------------------------------------------------------
    Stability_S = float(np.mean(squared_distances))
    Consistency_C_imp = float(np.mean(implication_ok_flags))
    Coverage_imp = float(np.mean(antecedent_flags))
    PredKeep_imp = float(np.mean(pred_keep_flags))

    violation_rate = float(
        np.mean(
            np.asarray(antecedent_flags, dtype=bool)
            & ~np.asarray(pred_keep_flags, dtype=bool)
        )
    )

    mean_cosine = float(np.mean(similarities))

    true_label = y_test[idx_test]

    original_index = (
        original_indices[idx_test]
        if original_indices is not None
        else np.nan
    )

    return {
        "idx_test": int(idx_test),
        "original_index": original_index,
        "true_label": int(true_label),
        "true_class": class_names[int(true_label)],
        "base_prediction": int(base_pred),
        "base_prediction_class": class_names[int(base_pred)],
        "base_prediction_correct": bool(base_pred == true_label),
        "pair_i": int(ci_lbl),
        "pair_i_name": class_names[int(ci_lbl)],
        "pair_j": int(cj_lbl),
        "pair_j_name": class_names[int(cj_lbl)],
        "pair": (
            f"{class_names[int(ci_lbl)]} vs "
            f"{class_names[int(cj_lbl)]}"
        ),
        "abs_deltaP_ij": float(
            abs(base_prob[ci_pos] - base_prob[cj_pos])
        ),
        "Stability_S": Stability_S,
        "Consistency_C_imp": Consistency_C_imp,
        "Coverage_imp": Coverage_imp,
        "PredKeep_imp": PredKeep_imp,
        "ImplicationViolationRate": violation_rate,
        "MeanCosineSimilarity": mean_cosine,
        "n_antecedent_true": int(np.sum(antecedent_flags)),
        "n_prediction_kept": int(np.sum(pred_keep_flags)),
        "M_perturb": int(M_PERTURB),
    }


# ============================================================
# RESUMEN
# ============================================================

def summarize_metrics(df_inst):
    metrics = [
        "Stability_S",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]

    summary = df_inst[metrics].agg(["mean", "std", "median"]).T

    summary["SE"] = (
        summary["std"]
        / np.sqrt(len(df_inst))
    )

    summary["mean±std"] = (
        summary["mean"].map(lambda x: f"{x:.6f}")
        + " ± "
        + summary["std"].map(lambda x: f"{x:.6f}")
    )

    return summary.reset_index(names="metric")


# ============================================================
# VERIFICACIONES PREVIAS
# ============================================================

def verify_configuration(
    model,
    processor,
    scaler,
    X_train_raw,
    X_test_raw,
    X_test_final,
    y_test,
    shap_3d,
    feature_names,
    class_names,
):
    print("\n============================================================")
    print(" VERIFICACIÓN DE CONFIGURACIÓN")
    print("============================================================")

    selected = list(processor.get_selected_original_features())

    print(f"ARTIFACT_DIR       : {ARTIFACT_DIR}")
    print(f"Modelo             : {type(model).__name__}")
    print(f"Clases modelo      : {list(model.classes_)}")
    print(f"Clases nombres     : {class_names}")
    print(f"Train raw          : {X_train_raw.shape}")
    print(f"Test raw           : {X_test_raw.shape}")
    print(f"Test final         : {X_test_final.shape}")
    print(f"y_test             : {y_test.shape}")
    print(f"SHAP 3D            : {shap_3d.shape}")
    print(f"Vars seleccionadas : {len(selected)}")
    print(f"Vars procesadas    : {len(feature_names)}")
    print(f"N por par          : {N_INSTANCES}")
    print(f"M perturbaciones   : {M_PERTURB}")
    print(f"sigma rango        : {SIGMA_RANGE}")
    print(f"p_change           : {P_CHANGE}")
    print(f"tau                : {TAU}")
    print(f"top_k              : {TOP_K}")
    print(f"normalize          : {NORMALIZE}")
    print(f"filter_mode        : {FILTER_MODE}")
    print(f"balanced_sampling  : {BALANCED_SAMPLING}")
    print(f"seed               : {RANDOM_STATE}")
    print(f"Pares              : {PAIRS_TO_EVALUATE}")

    if FILTER_MODE != "true":
        raise ValueError(
            "Este protocolo definitivo debe usar FILTER_MODE='true'."
        )

    if not BALANCED_SAMPLING:
        raise ValueError(
            "Este protocolo definitivo debe usar muestreo balanceado."
        )

    if X_test_final.shape[0] != len(y_test):
        raise ValueError(
            "X_test_final y y_test no tienen el mismo número de filas."
        )

    if shap_3d.shape[0] != len(y_test):
        raise ValueError(
            "SHAP 3D y y_test no tienen el mismo número de instancias."
        )

    if shap_3d.shape[1] != X_test_final.shape[1]:
        raise ValueError(
            "SHAP 3D y X_test_final no tienen el mismo número de variables."
        )

    if shap_3d.shape[2] != len(model.classes_):
        raise ValueError(
            "SHAP 3D y el modelo no tienen el mismo número de clases."
        )

    if len(feature_names) != X_test_final.shape[1]:
        raise ValueError(
            "feature_names no coincide con X_test_final."
        )

    missing_selected = [
        f for f in selected
        if f not in X_train_raw.columns or f not in X_test_raw.columns
    ]
    if missing_selected:
        raise ValueError(
            "Variables seleccionadas ausentes en raw:\n"
            + "\n".join(missing_selected)
        )

    # --------------------------------------------------------
    # Verificación de alineación RAW -> processor -> scaler
    # --------------------------------------------------------
    n_check = min(N_ALIGNMENT_CHECK, len(X_test_raw))

    check_raw = X_test_raw.iloc[:n_check].copy()

    check_transformed = transform_raw_df(
        check_raw,
        processor=processor,
        scaler=scaler
    )

    check_reference = X_test_final[:n_check]

    max_abs_diff = float(
        np.max(np.abs(check_transformed - check_reference))
    )

    print(
        f"Máxima diferencia alineación raw->final "
        f"(primeras {n_check} filas): {max_abs_diff:.12e}"
    )

    if not np.allclose(
        check_transformed,
        check_reference,
        atol=ALIGNMENT_ATOL,
        rtol=ALIGNMENT_RTOL
    ):
        raise ValueError(
            "FALLÓ la verificación de alineación raw -> processor -> scaler. "
            "No se ejecutarán las métricas."
        )

    # --------------------------------------------------------
    # Verificar pares
    # --------------------------------------------------------
    for pair in PAIRS_TO_EVALUATE:
        resolve_pair_positions(model, pair)

        c1 = np.sum(y_test == pair[0])
        c2 = np.sum(y_test == pair[1])

        print(
            f"Par {pair}: "
            f"{class_names[pair[0]]} vs {class_names[pair[1]]} | "
            f"disponibles en test = {c1} / {c2}"
        )

        if min(c1, c2) < (N_INSTANCES // 2):
            print(
                "ADVERTENCIA: no hay 25 observaciones por clase. "
                "Se usará el máximo balanceado disponible."
            )

    print(
        "\nProtocolo de perturbación válido confirmado:"
    )
    print(
        "  Numéricas continuas/discretas: "
        f"sigma = {SIGMA_RANGE} del rango TRAIN"
    )
    print(
        "  Binarias/categóricas/ordinales: "
        f"p_change = {P_CHANGE}"
    )
    print(
        f"  M = {M_PERTURB}, tau = {TAU}, "
        f"top_k = {TOP_K}, normalize = {NORMALIZE}"
    )
    print(
        "  Semilla por instancia: seed + t"
    )

    print("VERIFICACIÓN COMPLETADA CORRECTAMENTE.")
    print("============================================================\n")


# ============================================================
# MAIN
# ============================================================

def main():
    start_total = time.perf_counter()

    set_seed(RANDOM_STATE)

    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("SHAP:", shap.__version__)
    print("OS:", platform.platform())
    print("Processor:", platform.processor())

    # --------------------------------------------------------
    # 1) Verificar archivos
    # --------------------------------------------------------
    require_files([
        MODEL_PATH,
        PROCESSOR_PATH,
        SCALER_PATH,
        X_TRAIN_RAW_PATH,
        X_TEST_RAW_PATH,
        X_TEST_FINAL_PATH,
        Y_TEST_PATH,
        FEATURE_NAMES_PATH,
        CLASS_NAMES_PATH,
        SHAP_3D_PATH,
    ])

    # --------------------------------------------------------
    # 2) Cargar artefactos definitivos
    # --------------------------------------------------------
    print("\nCargando artefactos definitivos de EduFormalDANE...")

    model = joblib.load(MODEL_PATH)
    processor = joblib.load(PROCESSOR_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_train_raw = read_raw_split(X_TRAIN_RAW_PATH)
    X_test_raw = read_raw_split(X_TEST_RAW_PATH)

    X_test_final_df = pd.read_csv(X_TEST_FINAL_PATH)
    X_test_final = X_test_final_df.to_numpy(dtype=float)

    y_test = load_y_vector(Y_TEST_PATH)

    # Verificación redundante contra y_test_original_split si existe.
    if Y_TEST_ORIGINAL_PATH.exists():
        y_test_original = load_y_vector(Y_TEST_ORIGINAL_PATH)
        if not np.array_equal(y_test, y_test_original):
            raise ValueError(
                "y_test_final y y_test_original_split no están alineados."
            )

    original_indices = None
    if TEST_ORIGINAL_INDICES_PATH.exists():
        df_idx = pd.read_csv(TEST_ORIGINAL_INDICES_PATH)
        if "original_index" in df_idx.columns:
            original_indices = df_idx["original_index"].to_numpy()

    feature_names = (
        pd.read_csv(FEATURE_NAMES_PATH)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )

    class_names = load_class_names(CLASS_NAMES_PATH)

    shap_base_3d = np.load(SHAP_3D_PATH)

    # --------------------------------------------------------
    # 3) Verificar configuración ANTES de calcular
    # --------------------------------------------------------
    verify_configuration(
        model=model,
        processor=processor,
        scaler=scaler,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_test_final=X_test_final,
        y_test=y_test,
        shap_3d=shap_base_3d,
        feature_names=feature_names,
        class_names=class_names,
    )

    # --------------------------------------------------------
    # 4) Construir perfiles válidos por tipo de variable
    # --------------------------------------------------------
    profiles, type_report = build_perturbation_profiles(
        X_train_raw=X_train_raw,
        processor=processor
    )

    type_report.to_csv(
        METRICS_OUTPUT_DIR / "feature_perturbation_types.csv",
        index=False
    )

    print("\nTipos de perturbación utilizados:")
    print(
        type_report["perturbation_type"]
        .value_counts()
        .to_string()
    )

    print("\nDetalle de variables:")
    print(
        type_report[
            [
                "feature",
                "processor_type",
                "perturbation_type",
                "n_valid_values",
                "train_min",
                "train_max",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 5) Guardar protocolo
    # --------------------------------------------------------
    protocol = {
        "dataset_code": "EduDane",
        "dataset_article": "EduFormalDANE",
        "artifact_dir": str(ARTIFACT_DIR),
        "model": type(model).__name__,
        "selection_mode": "true_label_balanced",
        "includes_correct_and_incorrect_predictions": True,
        "pairs": [list(p) for p in PAIRS_TO_EVALUATE],
        "n_instances_per_pair": N_INSTANCES,
        "M_perturbations_per_instance": M_PERTURB,
        "sigma_fraction_train_range": SIGMA_RANGE,
        "p_change_binary_categorical_ordinal": P_CHANGE,
        "tau_cosine_similarity": TAU,
        "top_k": TOP_K,
        "normalize": NORMALIZE,
        "random_state": RANDOM_STATE,
        "instance_seed_rule": "random_state + local_position",
        "perturbation_parameters_note": (
            "Dataset-specific valid-perturbation parameters: "
            "sigma=0.005 of TRAIN range for numeric variables and "
            "p_change=0.05 for binary/categorical/ordinal variables."
        ),
        "perturbation_space": "original_feature_space",
        "continuous": (
            "Gaussian local noise with SD=sigma*train_range; "
            "clip to train min/max."
        ),
        "discrete_numeric": (
            "Gaussian local noise with SD=sigma*train_range; "
            "project to nearest train-observed valid value."
        ),
        "binary": (
            "Keep valid states; switch with p_change to another "
            "train-observed state."
        ),
        "categorical": (
            "Switch with p_change; sample only train-observed categories "
            "according to empirical train distribution."
        ),
        "ordinal": (
            "Switch with p_change; remain within train-observed valid "
            "ordinal values."
        ),
        "transform_after_perturbation": (
            "same fitted feature selection + same fitted encoder + "
            "same fitted MinMaxScaler; no refit"
        ),
        "metrics": {
            "Stability_S": (
                "Mean squared L2 distance between normalized sparse "
                "base SHAP-LCD explanation and perturbed explanations; "
                "lower is more stable."
            ),
            "Consistency_C_imp": (
                "Mean truth value of implication "
                "(cosine_similarity >= tau) => prediction_preserved."
            ),
            "Coverage_imp": (
                "Fraction with cosine_similarity >= tau."
            ),
            "PredKeep_imp": (
                "Fraction whose predicted class equals base predicted class."
            ),
        },
    }

    with open(
        METRICS_OUTPUT_DIR / "protocol_valid_perturbations.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(protocol, f, ensure_ascii=False, indent=4)

    # --------------------------------------------------------
    # 6) TreeExplainer del MISMO modelo final
    # --------------------------------------------------------
    print("\nInicializando TreeExplainer del modelo final...")
    tree_explainer = shap.TreeExplainer(model)

    # --------------------------------------------------------
    # 7) Evaluar pares
    # --------------------------------------------------------
    all_instance_results = []
    all_pair_summaries = []

    for pair_number, pair in enumerate(PAIRS_TO_EVALUATE):
        pair_start = time.perf_counter()

        ci_lbl, cj_lbl = pair
        pair_name = (
            f"{class_names[ci_lbl]} vs {class_names[cj_lbl]}"
        )
        pair_safe = safe_name(pair_name)

        print("\n============================================================")
        print(f" PAR: {pair_name}")
        print("============================================================")

        selected_idxs = select_balanced_true_pair_indices(
            y_test=y_test,
            pair=pair,
            n_instances=N_INSTANCES,
            seed=RANDOM_STATE
        )

        # Composición por etiqueta verdadera
        selected_true = pd.Series(y_test[selected_idxs])
        print("\nDistribución VERDADERA de la muestra seleccionada:")
        print(selected_true.value_counts().sort_index().to_string())

        # Sin filtrar por predicción: se reporta cuántas son correctas/incorrectas.
        base_preds_selected = model.predict(X_test_final[selected_idxs])
        correct_flags = (base_preds_selected == y_test[selected_idxs])

        print(
            "\nPredicciones base en la muestra: "
            f"correctas={int(np.sum(correct_flags))}, "
            f"incorrectas={int(np.sum(~correct_flags))}"
        )

        selection_df = pd.DataFrame({
            "idx_test": selected_idxs,
            "original_index": (
                original_indices[selected_idxs]
                if original_indices is not None
                else np.nan
            ),
            "true_label": y_test[selected_idxs].astype(int),
            "true_class": [
                class_names[int(v)]
                for v in y_test[selected_idxs]
            ],
            "base_prediction": base_preds_selected.astype(int),
            "base_prediction_class": [
                class_names[int(v)]
                for v in base_preds_selected
            ],
            "base_prediction_correct": correct_flags,
        })

        selection_df.to_csv(
            METRICS_OUTPUT_DIR
            / f"selected_instances_{pair_safe}.csv",
            index=False
        )

        rows = []

        n_sel = len(selected_idxs)

        for position, idx_test in enumerate(selected_idxs, start=1):
            row_start = time.perf_counter()

            result = evaluate_one_instance(
                idx_test=int(idx_test),
                pair=pair,
                instance_position=position - 1,
                X_test_raw=X_test_raw,
                X_test_final=X_test_final,
                y_test=y_test,
                original_indices=original_indices,
                model=model,
                tree_explainer=tree_explainer,
                processor=processor,
                scaler=scaler,
                profiles=profiles,
                shap_base_3d=shap_base_3d,
                class_names=class_names,
            )

            rows.append(result)
            all_instance_results.append(result)

            elapsed_row = time.perf_counter() - row_start

            print(
                f"[{position:02d}/{n_sel}] "
                f"idx_test={idx_test:5d} | "
                f"true={result['true_class']:<10s} | "
                f"pred={result['base_prediction_class']:<10s} | "
                f"S={result['Stability_S']:.6f} | "
                f"Cimp={result['Consistency_C_imp']:.6f} | "
                f"Cov={result['Coverage_imp']:.6f} | "
                f"PredKeep={result['PredKeep_imp']:.6f} | "
                f"{elapsed_row:.2f}s"
            )

        df_inst = pd.DataFrame(rows)
        summary = summarize_metrics(df_inst)

        # Agregar información del par
        summary.insert(0, "pair_i", ci_lbl)
        summary.insert(1, "pair_i_name", class_names[ci_lbl])
        summary.insert(2, "pair_j", cj_lbl)
        summary.insert(3, "pair_j_name", class_names[cj_lbl])
        summary.insert(4, "pair", pair_name)
        summary.insert(5, "N_instances", len(df_inst))
        summary.insert(6, "M_perturb", M_PERTURB)

        all_pair_summaries.append(summary)

        df_inst.to_csv(
            METRICS_OUTPUT_DIR
            / f"metrics_valid_perturbations_{pair_safe}.csv",
            index=False
        )

        summary.to_csv(
            METRICS_OUTPUT_DIR
            / f"summary_valid_perturbations_{pair_safe}.csv",
            index=False
        )

        print("\nResumen media ± std:")
        print(
            summary[
                ["metric", "mean±std", "median", "SE"]
            ].to_string(index=False)
        )

        pair_elapsed = time.perf_counter() - pair_start

        print(
            f"\nTiempo del par {pair_name}: "
            f"{pair_elapsed / 60.0:.2f} minutos"
        )

    # --------------------------------------------------------
    # 8) Guardar resultados combinados
    # --------------------------------------------------------
    df_all = pd.DataFrame(all_instance_results)
    summary_all = pd.concat(
        all_pair_summaries,
        axis=0,
        ignore_index=True
    )

    df_all.to_csv(
        METRICS_OUTPUT_DIR
        / "all_pairs_instance_metrics_valid_perturbations.csv",
        index=False
    )

    summary_all.to_csv(
        METRICS_OUTPUT_DIR
        / "all_pairs_summary_valid_perturbations.csv",
        index=False
    )

    # --------------------------------------------------------
    # 9) Resumen final en consola
    # --------------------------------------------------------
    print("\n============================================================")
    print(" RESUMEN FINAL - PERTURBACIONES VÁLIDAS")
    print("============================================================")

    for pair_name, grp in summary_all.groupby("pair", sort=False):
        print(f"\n{pair_name}")
        print(
            grp[
                ["metric", "mean±std", "median", "SE"]
            ].to_string(index=False)
        )

    total_elapsed = time.perf_counter() - start_total

    print("\n============================================================")
    print(" PROCESO FINALIZADO")
    print("============================================================")
    print(f"Resultados guardados en:")
    print(METRICS_OUTPUT_DIR.resolve())
    print(f"\nTiempo total: {total_elapsed / 60.0:.2f} minutos")
    print("\nArchivos principales:")
    print(" - feature_perturbation_types.csv")
    print(" - protocol_valid_perturbations.json")
    print(" - selected_instances_<par>.csv")
    print(" - metrics_valid_perturbations_<par>.csv")
    print(" - summary_valid_perturbations_<par>.csv")
    print(" - all_pairs_instance_metrics_valid_perturbations.csv")
    print(" - all_pairs_summary_valid_perturbations.csv")
    print("============================================================")


if __name__ == "__main__":
    main()
