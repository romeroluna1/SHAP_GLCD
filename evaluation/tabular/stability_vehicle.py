"""
Stability y Consistency de SHAP-LCD para Vehicle (bus vs van).

La explicacion base se toma del pickle SHAP-GLCD calculado en probabilidad.
Las explicaciones de las instancias perturbadas se calculan con
KernelExplainer sobre predict_proba y link="identity".
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.tabular import Gexplainer


# ============================================================
# 1. CONFIGURACION DEL PROTOCOLO
# ============================================================

RANDOM_STATE = 42
N_INSTANCES_REQUESTED = 50
M_PERTURB = 40
SIGMA = 0.02
TAU = 0.95
TOP_K = 10
BACKGROUND_SIZE = 100

J_SEL = 0  # bus
R_SEL = 3  # van

VEHICLE_CLASS_MAPPING = {
    0: "bus",
    1: "saab",
    2: "opel",
    3: "van",
}

PAIR_TAG = f"{J_SEL}_vs_{R_SEL}"
SHAP_PICKLE_NAME = f"shapvalues_vehicle_xgb_probability_{PAIR_TAG}.pickle"

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "statlog_vehicle_clean_abreviation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "vehicle"
)


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def find_shap_pickle():
    pickle_path = (
        PROJECT_ROOT
        / "artifacts"
        / SHAP_PICKLE_NAME
    )

    if not pickle_path.exists():
        raise FileNotFoundError(
            "No se encontró el pickle SHAP de Vehicle:\n"
            f"{pickle_path}"
        )

    return pickle_path


def normalize_l2(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def top_k_sparse(vector, top_k=10):
    vector = np.asarray(vector, dtype=float)
    if top_k is None or top_k >= len(vector):
        return vector.copy()

    indices = np.argsort(np.abs(vector))[::-1][:top_k]
    result = np.zeros_like(vector)
    result[indices] = vector[indices]
    return result


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def delta_s_from_shap(shap_instance, class_j, class_r, top_k=10):
    """Transforma SHAP (features, classes) en DeltaS -> Top-K -> L2."""

    shap_instance = np.asarray(shap_instance, dtype=float)
    if shap_instance.ndim != 2:
        raise ValueError(
            "Se esperaba SHAP con forma (features, classes), "
            f"pero se recibio {shap_instance.shape}."
        )

    delta_s = shap_instance[:, class_j] - shap_instance[:, class_r]
    return normalize_l2(top_k_sparse(delta_s, top_k=top_k))


def format_kernel_shap(shap_raw, n_instances, n_features, n_classes):
    """Normaliza las distintas salidas de KernelExplainer."""

    if isinstance(shap_raw, list):
        arr = np.stack(shap_raw, axis=2)
    else:
        arr = np.asarray(shap_raw)

    expected = (n_instances, n_features, n_classes)
    if arr.shape == expected:
        return arr

    alternative = (n_classes, n_instances, n_features)
    if arr.shape == alternative:
        return np.transpose(arr, (1, 2, 0))

    if n_instances == 1 and arr.shape == (n_features, n_classes):
        return arr.reshape(expected)

    raise ValueError(
        "Formato SHAP inesperado. "
        f"Recibido: {arr.shape}; esperado: {expected}."
    )


def perturb_normalized_instance(original_scaled, rng, sigma=0.02):
    """Aplica ruido gaussiano en el espacio normalizado y clip [0,1]."""

    values = np.asarray(original_scaled, dtype=float)
    noise = rng.normal(0.0, sigma, size=values.shape)
    return np.clip(values + noise, 0.0, 1.0)


def select_balanced_pair(y_test, class_j, class_r, n_requested=50, seed=42):
    """Selecciona por etiqueta verdadera, balanceado y sin reemplazo."""

    y_test = np.asarray(y_test)
    rng = np.random.default_rng(seed)
    idx_j = np.where(y_test == class_j)[0]
    idx_r = np.where(y_test == class_r)[0]

    requested_each = n_requested // 2
    n_each = min(requested_each, len(idx_j), len(idx_r))
    if n_each <= 0:
        raise RuntimeError("No hay instancias suficientes para formar el par.")

    rng.shuffle(idx_j)
    rng.shuffle(idx_r)
    selected = np.concatenate([idx_j[:n_each], idx_r[:n_each]])
    rng.shuffle(selected)
    return selected, n_each


def build_probability_explainer(gexp):
    """Reproduce el explicador usado para crear el pickle SHAP-GLCD."""

    n_background = min(BACKGROUND_SIZE, len(gexp.X_tr))
    background = shap.sample(
        gexp.X_tr,
        n_background,
        random_state=RANDOM_STATE,
    )

    # La envoltura evita la incompatibilidad de SHAP con la propiedad
    # feature_names_in_ de algunas versiones de XGBoost.
    def predict_proba_for_shap(data):
        return gexp.modelo_final.predict_proba(data)

    explainer = shap.KernelExplainer(
        predict_proba_for_shap,
        background,
        link="identity",
    )
    return explainer, n_background


def verify_probability_pickle(gexp, selected_indices, atol=1e-5, rtol=1e-5):
    """Verifica base + suma(SHAP) = predict_proba para el pickle cargado."""

    if gexp.shap_output_space != "probability":
        raise RuntimeError(
            "El pickle no declara escala de probabilidad: "
            f"{gexp.shap_output_space!r}."
        )

    if gexp.shap_expected_value is None:
        raise RuntimeError("El pickle no contiene expected_value.")

    check_indices = np.asarray(selected_indices[:10], dtype=int)
    reconstructed = (
        np.asarray(gexp.shap_expected_value, dtype=float).reshape(1, -1)
        + np.sum(gexp.shap_vals[check_indices], axis=1)
    )
    probabilities = gexp.modelo_final.predict_proba(
        gexp.X_te.iloc[check_indices]
    )
    errors = np.abs(reconstructed - probabilities)

    print("\n== Verificacion del pickle SHAP en probabilidad ==")
    print(f"Instancias verificadas: {len(check_indices)}")
    print(f"Error medio: {np.mean(errors):.3e}")
    print(f"Error maximo: {np.max(errors):.3e}")

    if not np.allclose(reconstructed, probabilities, atol=atol, rtol=rtol):
        raise RuntimeError(
            "El pickle no reconstruye predict_proba dentro de la tolerancia."
        )

    print("VERIFICACION SUPERADA: SHAP esta en probabilidad.")


# ============================================================
# 3. EVALUACION DE STABILITY Y CONSISTENCY
# ============================================================

def evaluate_stability_consistency(gexp, kernel_explainer, selected_indices):
    fitted_model = gexp.modelo_final
    feature_names = list(gexp.features)
    n_features = len(feature_names)
    n_classes = len(fitted_model.classes_)
    y_test = np.asarray(gexp.y_te)
    rows = []

    for t, test_position in enumerate(selected_indices):
        instance_seed = RANDOM_STATE + t
        rng = np.random.default_rng(instance_seed)

        original = gexp.X_te.iloc[test_position].to_numpy(dtype=float)
        original_df = pd.DataFrame([original], columns=feature_names)
        true_class = int(y_test[test_position])
        base_pred = int(fitted_model.predict(original_df)[0])

        # La explicacion base es exactamente la almacenada por SHAP-GLCD.
        base_exp = delta_s_from_shap(
            gexp.shap_vals[test_position],
            class_j=J_SEL,
            class_r=R_SEL,
            top_k=TOP_K,
        )

        # Se generan juntas las M perturbaciones de la instancia.
        perturbed_array = np.vstack([
            perturb_normalized_instance(original, rng, sigma=SIGMA)
            for _ in range(M_PERTURB)
        ])
        perturbed_df = pd.DataFrame(perturbed_array, columns=feature_names)
        perturbed_pred = fitted_model.predict(perturbed_df).astype(int)

        # Todas las explicaciones perturbadas se calculan en probabilidad.
        np.random.seed(instance_seed)
        shap_raw = kernel_explainer.shap_values(perturbed_df)
        shap_3d = format_kernel_shap(
            shap_raw,
            n_instances=M_PERTURB,
            n_features=n_features,
            n_classes=n_classes,
        )

        stability_values = []
        consistency_values = []
        coverage_values = []
        predkeep_values = []
        cosine_values = []

        for m in range(M_PERTURB):
            perturbed_exp = delta_s_from_shap(
                shap_3d[m],
                class_j=J_SEL,
                class_r=R_SEL,
                top_k=TOP_K,
            )

            squared_distance = float(
                np.linalg.norm(perturbed_exp - base_exp) ** 2
            )
            similarity = cosine_similarity(base_exp, perturbed_exp)
            antecedent = bool(similarity >= TAU)
            consequent = bool(perturbed_pred[m] == base_pred)
            implication_holds = (not antecedent) or consequent

            stability_values.append(squared_distance)
            cosine_values.append(similarity)
            coverage_values.append(int(antecedent))
            predkeep_values.append(int(consequent))
            consistency_values.append(int(implication_holds))

        row = {
            "local_position": int(t),
            "test_position": int(test_position),
            "perturbation_seed": int(instance_seed),
            "base_SHAP_source": "SHAP-GLCD probability pickle",
            "perturbed_SHAP_source": "KernelExplainer predict_proba",
            "explanation_dimension": int(n_features),
            "top_k": int(TOP_K),
            "nonzero_base_exp": int(np.count_nonzero(base_exp)),
            "base_exp_l2_norm": float(np.linalg.norm(base_exp)),
            "true_class_id": true_class,
            "true_class": VEHICLE_CLASS_MAPPING[true_class],
            "predicted_class_id": base_pred,
            "predicted_class": VEHICLE_CLASS_MAPPING[base_pred],
            "correct_prediction": bool(base_pred == true_class),
            "Stability_S": float(np.mean(stability_values)),
            "Consistency_C_imp": float(np.mean(consistency_values)),
            "Coverage_imp": float(np.mean(coverage_values)),
            "PredKeep_imp": float(np.mean(predkeep_values)),
            "Cosine_mean": float(np.mean(cosine_values)),
        }
        rows.append(row)

        print(
            f"[{t + 1:02d}/{len(selected_indices)}] "
            f"idx_test={test_position:3d} | "
            f"true={VEHICLE_CLASS_MAPPING[true_class]:4s} | "
            f"pred={VEHICLE_CLASS_MAPPING[base_pred]:4s} | "
            f"S={row['Stability_S']:.6f} | "
            f"Cimp={row['Consistency_C_imp']:.6f} | "
            f"Cov={row['Coverage_imp']:.6f} | "
            f"PredKeep={row['PredKeep_imp']:.6f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. MAIN
# ============================================================

def main():
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.90,
        colsample_bytree=0.90,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    shap_pickle_path = find_shap_pickle()

    print("\n============================================================")
    print(" VEHICLE - STABILITY / CONSISTENCY SHAP-LCD")
    print(" SHAP EN ESCALA DE PROBABILIDAD")
    print("============================================================")
    print("Par               : bus vs van")
    print(f"N solicitado      : {N_INSTANCES_REQUESTED}")
    print(f"M perturbaciones  : {M_PERTURB}")
    print(f"sigma             : {SIGMA}")
    print(f"tau               : {TAU}")
    print(f"top-k             : {TOP_K}")
    print("Normalizacion     : L2")
    print("Perturbacion      : Gaussian en espacio normalizado")
    print("Clip              : [0,1]")
    print("Base SHAP         : pickle SHAP-GLCD en probabilidad")
    print("Perturbed SHAP    : KernelExplainer(predict_proba)")
    print(f"Pickle            : {shap_pickle_path}")
    print("============================================================")

    gexp = Gexplainer(
        csv_path=CSV_PATH,
        target_column="class_num",
        model=model,
        shap_pickle=str(shap_pickle_path),
        j_sel=J_SEL,
        r_sel=R_SEL,
        top=TOP_K,
        random_state=RANDOM_STATE,
        class_mapping=VEHICLE_CLASS_MAPPING,
        figure_font_scale=1.20,
    )

    gexp.entrenar_modelo_final()
    gexp.cargar_shap(filename=str(shap_pickle_path))

    expected_shape = (len(gexp.X_te), len(gexp.features), len(gexp.class_names))
    if gexp.shap_vals.shape != expected_shape:
        raise RuntimeError(
            f"Shape SHAP inesperado: {gexp.shap_vals.shape}; "
            f"esperado: {expected_shape}."
        )

    selected_indices, n_each = select_balanced_pair(
        y_test=gexp.y_te,
        class_j=J_SEL,
        class_r=R_SEL,
        n_requested=N_INSTANCES_REQUESTED,
        seed=RANDOM_STATE,
    )

    selected_true = np.asarray(gexp.y_te)[selected_indices]
    selected_pred = gexp.modelo_final.predict(
        gexp.X_te.iloc[selected_indices]
    ).astype(int)

    print("\n== Muestra balanceada ==")
    print(f"N efectivo: {len(selected_indices)} ({n_each} bus + {n_each} van)")
    print("Filtro por prediccion: NO")
    print(f"Aciertos: {np.sum(selected_true == selected_pred)}")
    print(f"Errores : {np.sum(selected_true != selected_pred)}")

    verify_probability_pickle(gexp, selected_indices)

    kernel_explainer, n_background = build_probability_explainer(gexp)
    print(f"Background KernelExplainer: {n_background} instancias")

    metadata = pd.DataFrame({
        "test_position": selected_indices,
        "true_class_id": selected_true,
        "true_class": [VEHICLE_CLASS_MAPPING[int(v)] for v in selected_true],
        "predicted_class_id": selected_pred,
        "predicted_class": [VEHICLE_CLASS_MAPPING[int(v)] for v in selected_pred],
        "correct_prediction": selected_true == selected_pred,
    })
    metadata.to_csv(
        OUTPUT_DIR / "selected_instances_vehicle_bus_vs_van.csv",
        index=False,
        encoding="utf-8-sig",
    )

    results = evaluate_stability_consistency(
        gexp=gexp,
        kernel_explainer=kernel_explainer,
        selected_indices=selected_indices,
    )
    results.to_csv(
        OUTPUT_DIR / "vehicle_bus_vs_van_stability_consistency_instances.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "Stability_S",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]
    summary = results[metric_columns].agg(["mean", "std", "median"]).T
    summary["SE"] = summary["std"] / np.sqrt(len(results))
    summary["mean+-std"] = (
        summary["mean"].map(lambda x: f"{x:.6f}")
        + " +- "
        + summary["std"].map(lambda x: f"{x:.6f}")
    )
    summary.insert(0, "Dataset", "Vehicle")
    summary.insert(1, "Pair", "bus vs van")
    summary.insert(2, "N", len(selected_indices))

    summary_out = summary.reset_index().rename(columns={"index": "Metric"})
    summary_out.to_csv(
        OUTPUT_DIR / "vehicle_bus_vs_van_stability_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n============================================================")
    print(" RESULTADOS FINALES - VEHICLE")
    print("============================================================")
    print(summary[["mean", "std", "median", "SE", "mean+-std"]].to_string())
    print(f"\nResultados guardados en:\n{OUTPUT_DIR.resolve()}")
    print("============================================================")


if __name__ == "__main__":
    main()
