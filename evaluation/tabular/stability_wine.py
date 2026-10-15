"""Estabilidad y consistencia SHAP-GLCD para Wine (clases 1 vs. 2).

Reproduce el protocolo de stability_vehicle.py con LightGBM:
N solicitado=50, M=40, sigma=0.02, tau=0.95, top-k=10 y semilla=42.
Las explicaciones SHAP se calculan en espacio de probabilidad.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.datasets import load_wine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.tabular import Gexplainer

RANDOM_STATE = 42
N_INSTANCES_REQUESTED = 50
M_PERTURB = 40
SIGMA = 0.02
TAU = 0.95
TOP_K = 10
BACKGROUND_SIZE = 100
J_SEL, R_SEL = 1, 2
CLASS_MAPPING = {0: "wine_class_0", 1: "wine_class_1", 2: "wine_class_2"}
PAIR_TAG = f"{J_SEL}_vs_{R_SEL}"
DATA_PATH = PROJECT_ROOT / "data" / "wine.csv"
SHAP_PICKLE_PATH = (
    PROJECT_ROOT / "artifacts" /
    f"shapvalues_wine_LGBM_probability_{PAIR_TAG}.pickle"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation" / "wine"


def normalize_l2(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return np.zeros_like(vector) if norm <= 1e-12 else vector / norm


def top_k_sparse(vector, top_k=10):
    vector = np.asarray(vector, dtype=float)
    if top_k is None or top_k >= len(vector):
        return vector.copy()
    indices = np.argsort(np.abs(vector))[::-1][:top_k]
    result = np.zeros_like(vector)
    result[indices] = vector[indices]
    return result


def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def delta_s_from_shap(shap_instance):
    shap_instance = np.asarray(shap_instance, dtype=float)
    if shap_instance.ndim != 2 or shap_instance.shape[1] != len(CLASS_MAPPING):
        raise ValueError(f"Forma SHAP inesperada: {shap_instance.shape}")
    delta = shap_instance[:, J_SEL] - shap_instance[:, R_SEL]
    return normalize_l2(top_k_sparse(delta, TOP_K))


def format_kernel_shap(shap_raw, n_instances, n_features, n_classes):
    if isinstance(shap_raw, list):
        arr = np.stack(shap_raw, axis=2)
    else:
        arr = np.asarray(shap_raw)
    expected = (n_instances, n_features, n_classes)
    if arr.shape == expected:
        return arr
    if arr.shape == (n_classes, n_instances, n_features):
        return np.transpose(arr, (1, 2, 0))
    if n_instances == 1 and arr.shape == (n_features, n_classes):
        return arr.reshape(expected)
    raise ValueError(f"Forma SHAP recibida {arr.shape}; esperada {expected}")


def perturb_normalized_instance(original_scaled, rng):
    values = np.asarray(original_scaled, dtype=float)
    return np.clip(values + rng.normal(0.0, SIGMA, size=values.shape), 0.0, 1.0)


def select_balanced_pair(y_test):
    y_test = np.asarray(y_test)
    rng = np.random.default_rng(RANDOM_STATE)
    idx_j = np.where(y_test == J_SEL)[0]
    idx_r = np.where(y_test == R_SEL)[0]
    n_each = min(N_INSTANCES_REQUESTED // 2, len(idx_j), len(idx_r))
    if n_each <= 0:
        raise RuntimeError("No hay instancias suficientes para el par.")
    rng.shuffle(idx_j)
    rng.shuffle(idx_r)
    selected = np.concatenate((idx_j[:n_each], idx_r[:n_each]))
    rng.shuffle(selected)
    return selected, n_each


def verify_probability_pickle(gexp, selected_indices):
    if gexp.shap_output_space != "probability":
        raise RuntimeError(f"El pickle no declara probabilidades: {gexp.shap_output_space!r}")
    if gexp.shap_expected_value is None:
        raise RuntimeError("El pickle no contiene expected_value.")
    check_indices = np.asarray(selected_indices[:10], dtype=int)
    reconstructed = (
        np.asarray(gexp.shap_expected_value, dtype=float).reshape(1, -1)
        + np.sum(gexp.shap_vals[check_indices], axis=1)
    )
    probabilities = gexp.modelo_final.predict_proba(gexp.X_te.iloc[check_indices])
    errors = np.abs(reconstructed - probabilities)
    print("\n== Verificación del pickle SHAP en probabilidad ==")
    print(f"Instancias verificadas: {len(check_indices)}")
    print(f"Error medio: {np.mean(errors):.3e}")
    print(f"Error máximo: {np.max(errors):.3e}")
    if not np.allclose(reconstructed, probabilities, atol=1e-5, rtol=1e-5):
        raise RuntimeError("El pickle no reconstruye predict_proba del modelo reentrenado.")
    print("VERIFICACIÓN SUPERADA: SHAP está en probabilidad.")


def build_probability_explainer(gexp):
    n_background = min(BACKGROUND_SIZE, len(gexp.X_tr))
    background = shap.sample(gexp.X_tr, n_background, random_state=RANDOM_STATE)
    feature_names = list(gexp.features)

    def predict_proba_for_shap(data):
        # SHAP puede entregar un array sin nombres de columnas.
        # Se restaura el DataFrame sin alterar valores ni orden de columnas.
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_names)
        return gexp.modelo_final.predict_proba(data)

    explainer = shap.KernelExplainer(
        predict_proba_for_shap, background, link="identity"
    )
    return explainer, n_background


def evaluate_stability_consistency(gexp, kernel_explainer, selected_indices):
    model = gexp.modelo_final
    feature_names = list(gexp.features)
    n_features = len(feature_names)
    n_classes = len(model.classes_)
    y_test = np.asarray(gexp.y_te)
    rows = []

    for t, test_position in enumerate(selected_indices):
        instance_seed = RANDOM_STATE + t
        rng = np.random.default_rng(instance_seed)
        original = gexp.X_te.iloc[test_position].to_numpy(dtype=float)
        original_df = pd.DataFrame([original], columns=feature_names)
        true_class = int(y_test[test_position])
        base_pred = int(model.predict(original_df)[0])
        base_exp = delta_s_from_shap(gexp.shap_vals[test_position])

        perturbed_array = np.vstack([
            perturb_normalized_instance(original, rng)
            for _ in range(M_PERTURB)
        ])
        perturbed_df = pd.DataFrame(perturbed_array, columns=feature_names)
        perturbed_pred = model.predict(perturbed_df).astype(int)

        np.random.seed(instance_seed)
        shap_raw = kernel_explainer.shap_values(perturbed_df)
        shap_3d = format_kernel_shap(
            shap_raw, M_PERTURB, n_features, n_classes
        )

        stability_values, consistency_values = [], []
        coverage_values, predkeep_values, cosine_values = [], [], []
        for m in range(M_PERTURB):
            perturbed_exp = delta_s_from_shap(shap_3d[m])
            squared_distance = float(np.linalg.norm(perturbed_exp - base_exp) ** 2)
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
            "explanation_dimension": n_features,
            "top_k": TOP_K,
            "nonzero_base_exp": int(np.count_nonzero(base_exp)),
            "base_exp_l2_norm": float(np.linalg.norm(base_exp)),
            "true_class_id": true_class,
            "true_class": CLASS_MAPPING[true_class],
            "predicted_class_id": base_pred,
            "predicted_class": CLASS_MAPPING[base_pred],
            "correct_prediction": bool(base_pred == true_class),
            "Stability_S": float(np.mean(stability_values)),
            "Consistency_C_imp": float(np.mean(consistency_values)),
            "Coverage_imp": float(np.mean(coverage_values)),
            "PredKeep_imp": float(np.mean(predkeep_values)),
            "Cosine_mean": float(np.mean(cosine_values)),
        }
        rows.append(row)
        print(
            f"[{t+1:02d}/{len(selected_indices)}] "
            f"idx_test={test_position:3d} | "
            f"true={CLASS_MAPPING[true_class]} | "
            f"pred={CLASS_MAPPING[base_pred]} | "
            f"S={row['Stability_S']:.6f} | "
            f"Cimp={row['Consistency_C_imp']:.6f} | "
            f"Cov={row['Coverage_imp']:.6f} | "
            f"PredKeep={row['PredKeep_imp']:.6f}"
        )

    return pd.DataFrame(rows)


def main():
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        wine = load_wine()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df["target"] = wine.target
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset Wine creado en: {DATA_PATH}")

    if not SHAP_PICKLE_PATH.exists():
        raise FileNotFoundError(
            "Primero ejecuta examples/tabular/run_wine.py para generar "
            f"el pickle SHAP en probabilidad:\n{SHAP_PICKLE_PATH}"
        )

    # Misma configuración de LightGBM que examples/tabular/run_wine.py.
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    print("\n============================================================")
    print(" WINE - STABILITY / CONSISTENCY SHAP-GLCD")
    print("============================================================")
    print("Par             : wine_class_1 vs wine_class_2")
    print(f"N solicitado    : {N_INSTANCES_REQUESTED}")
    print(f"M perturbaciones: {M_PERTURB}")
    print(f"Sigma           : {SIGMA}")
    print(f"Tau             : {TAU}")
    print(f"Top-K           : {TOP_K}")
    print("Normalización   : L2")
    print("Perturbación    : Gaussiana en espacio normalizado, clip [0, 1]")
    print("SHAP original   : pickle en probabilidad")
    print("SHAP perturbado : KernelExplainer(predict_proba)")
    print(f"Pickle          : {SHAP_PICKLE_PATH}")
    print("============================================================")

    gexp = Gexplainer(
        csv_path=DATA_PATH,
        target_column="target",
        model=model,
        shap_pickle=str(SHAP_PICKLE_PATH),
        j_sel=J_SEL,
        r_sel=R_SEL,
        top=TOP_K,
        random_state=RANDOM_STATE,
        class_mapping=CLASS_MAPPING,
        figure_font_scale=1.20,
    )
    gexp.entrenar_modelo_final()
    gexp.cargar_shap(filename=str(SHAP_PICKLE_PATH))

    expected_shape = (len(gexp.X_te), len(gexp.features), len(gexp.class_names))
    if gexp.shap_vals.shape != expected_shape:
        raise RuntimeError(
            f"Forma SHAP inesperada {gexp.shap_vals.shape}; "
            f"esperada {expected_shape}"
        )

    selected_indices, n_each = select_balanced_pair(gexp.y_te)
    selected_true = np.asarray(gexp.y_te)[selected_indices]
    selected_pred = gexp.modelo_final.predict(
        gexp.X_te.iloc[selected_indices]
    ).astype(int)

    print("\n== Muestra balanceada por etiqueta verdadera ==")
    print(f"N efectivo: {len(selected_indices)} ({n_each} + {n_each})")
    print("Filtro por predicción: NO")
    print(f"Aciertos: {np.sum(selected_true == selected_pred)}")
    print(f"Errores: {np.sum(selected_true != selected_pred)}")
    verify_probability_pickle(gexp, selected_indices)

    kernel_explainer, n_background = build_probability_explainer(gexp)
    print(f"Background KernelExplainer: {n_background} instancias")

    metadata = pd.DataFrame({
        "test_position": selected_indices,
        "true_class_id": selected_true,
        "true_class": [CLASS_MAPPING[int(c)] for c in selected_true],
        "predicted_class_id": selected_pred,
        "predicted_class": [CLASS_MAPPING[int(c)] for c in selected_pred],
        "correct_prediction": selected_true == selected_pred,
    })
    metadata.to_csv(
        OUTPUT_DIR / "selected_instances_wine_1_vs_2.csv",
        index=False, encoding="utf-8-sig"
    )

    results = evaluate_stability_consistency(
        gexp, kernel_explainer, selected_indices
    )
    results.to_csv(
        OUTPUT_DIR / "wine_1_vs_2_stability_consistency_instances.csv",
        index=False, encoding="utf-8-sig"
    )

    metrics = ["Stability_S", "Consistency_C_imp", "Coverage_imp", "PredKeep_imp"]
    summary = results[metrics].agg(["mean", "std", "median"]).T
    summary["SE"] = summary["std"] / np.sqrt(len(results))
    summary["mean+-std"] = (
        summary["mean"].map(lambda x: f"{x:.6f}") + " +- "
        + summary["std"].map(lambda x: f"{x:.6f}")
    )
    summary.insert(0, "Dataset", "Wine")
    summary.insert(1, "Pair", "wine_class_1 vs wine_class_2")
    summary.insert(2, "N", len(selected_indices))
    summary.reset_index().rename(columns={"index": "Metric"}).to_csv(
        OUTPUT_DIR / "wine_1_vs_2_stability_consistency_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\n============================================================")
    print(" RESULTADOS FINALES - WINE")
    print("============================================================")
    print(summary[["mean", "std", "median", "SE", "mean+-std"]].to_string())
    print(f"\nResultados guardados en:\n{OUTPUT_DIR.resolve()}")
    print("============================================================")


if __name__ == "__main__":
    main()
