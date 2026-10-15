"""Stability and consistency evaluation for Iris SHAP-GLCD explanations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.tabular import Gexplainer  # noqa: E402


# -----------------------------------------------------------------------------
# Protocol configuration
# -----------------------------------------------------------------------------
QUICK_TEST = False

RANDOM_STATE = 42
J_SEL = 0  # setosa
R_SEL = 2  # virginica
SIGMA_RANGE = 0.005
TAU = 0.95
TOP_K = 10

if QUICK_TEST:
    N_INSTANCES_REQUESTED = 2
    M_PERTURB = 2
    RUN_NAME = "quick_test"
else:
    # The independent test split contains 15 setosa and 15 virginica
    # instances. Requesting 30 therefore evaluates the complete pair.
    N_INSTANCES_REQUESTED = 30
    M_PERTURB = 40
    RUN_NAME = "final"

CLASS_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "iris"
CSV_PATH = ARTIFACT_DIR / "iris.csv"
SHAP_PICKLE = (
    ARTIFACT_DIR
    / "shap_values_iris_rf_probability_0_vs_2.pickle"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "iris"
    / RUN_NAME
)


# -----------------------------------------------------------------------------
# Mathematical helpers
# -----------------------------------------------------------------------------
def normalize_l2(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def top_k_sparse(vector: np.ndarray, top_k: int | None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if top_k is None or top_k >= vector.size:
        return vector.copy()

    indices = np.argsort(np.abs(vector))[::-1][:top_k]
    sparse = np.zeros_like(vector)
    sparse[indices] = vector[indices]
    return sparse


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def delta_s_from_shap(
    shap_instance: np.ndarray,
    class_j: int,
    class_r: int,
) -> np.ndarray:
    shap_instance = np.asarray(shap_instance, dtype=float)
    if shap_instance.ndim != 2:
        raise ValueError(
            "Expected SHAP with shape (features, classes), received "
            f"{shap_instance.shape}."
        )

    delta_s = shap_instance[:, class_j] - shap_instance[:, class_r]
    return normalize_l2(top_k_sparse(delta_s, TOP_K))


def format_tree_shap(
    shap_raw,
    n_instances: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    if isinstance(shap_raw, list):
        array = np.stack(shap_raw, axis=2)
    else:
        array = np.asarray(shap_raw)

    expected = (n_instances, n_features, n_classes)
    if array.shape == expected:
        return array

    alternative = (n_classes, n_instances, n_features)
    if array.shape == alternative:
        return np.transpose(array, (1, 2, 0))

    if n_instances == 1 and array.shape == (n_features, n_classes):
        return array.reshape(expected)

    raise ValueError(
        f"Unexpected SHAP shape {array.shape}; expected {expected}."
    )


def select_balanced_pair(
    y_test: np.ndarray,
    n_requested: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(RANDOM_STATE)
    idx_j = np.flatnonzero(np.asarray(y_test) == J_SEL)
    idx_r = np.flatnonzero(np.asarray(y_test) == R_SEL)

    requested_each = n_requested // 2
    n_each = min(requested_each, len(idx_j), len(idx_r))
    if n_each <= 0:
        raise RuntimeError("Not enough test instances to build the pair.")

    rng.shuffle(idx_j)
    rng.shuffle(idx_r)
    selected = np.concatenate([idx_j[:n_each], idx_r[:n_each]])
    rng.shuffle(selected)
    return selected, n_each


def perturb_raw_continuous(
    raw_instance: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb in original units and keep values inside the train range."""
    feature_range = train_max - train_min
    sigma = SIGMA_RANGE * feature_range
    noise = rng.normal(0.0, sigma, size=raw_instance.shape)
    return np.clip(raw_instance + noise, train_min, train_max)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(gexp: Gexplainer, selected_indices: np.ndarray) -> pd.DataFrame:
    model = gexp.modelo_final
    feature_names = list(gexp.features)
    n_features = len(feature_names)
    n_classes = len(gexp.class_names)

    tree_explainer = shap.TreeExplainer(model)

    train_min = gexp.X_tr_raw.min(axis=0).to_numpy(dtype=float)
    train_max = gexp.X_tr_raw.max(axis=0).to_numpy(dtype=float)
    rows = []

    for local_position, test_position in enumerate(selected_indices):
        instance_seed = RANDOM_STATE + local_position
        rng = np.random.default_rng(instance_seed)

        original_raw = gexp.X_te_raw.iloc[test_position].to_numpy(dtype=float)
        original_scaled = gexp.X_te.iloc[[test_position]]
        true_class = int(gexp.y_te[test_position])
        base_prediction = int(model.predict(original_scaled)[0])

        base_explanation = delta_s_from_shap(
            gexp.shap_vals[test_position],
            J_SEL,
            R_SEL,
        )

        perturbed_raw = np.vstack([
            perturb_raw_continuous(
                original_raw,
                train_min,
                train_max,
                rng,
            )
            for _ in range(M_PERTURB)
        ])
        perturbed_raw_df = pd.DataFrame(
            perturbed_raw,
            columns=feature_names,
        )
        perturbed_scaled = pd.DataFrame(
            gexp.scaler.transform(perturbed_raw_df),
            columns=feature_names,
        )

        perturbed_predictions = model.predict(perturbed_scaled).astype(int)
        shap_raw = tree_explainer.shap_values(perturbed_scaled)
        shap_3d = format_tree_shap(
            shap_raw,
            n_instances=M_PERTURB,
            n_features=n_features,
            n_classes=n_classes,
        )

        stability = []
        consistency = []
        coverage = []
        prediction_keep = []
        cosine_values = []

        for perturbation_index in range(M_PERTURB):
            perturbed_explanation = delta_s_from_shap(
                shap_3d[perturbation_index],
                J_SEL,
                R_SEL,
            )
            similarity = cosine_similarity(
                base_explanation,
                perturbed_explanation,
            )
            antecedent = bool(similarity >= TAU)
            consequent = bool(
                perturbed_predictions[perturbation_index] == base_prediction
            )

            stability.append(float(
                np.linalg.norm(perturbed_explanation - base_explanation) ** 2
            ))
            consistency.append(int((not antecedent) or consequent))
            coverage.append(int(antecedent))
            prediction_keep.append(int(consequent))
            cosine_values.append(similarity)

        row = {
            "local_position": int(local_position),
            "test_position": int(test_position),
            "perturbation_seed": int(instance_seed),
            "true_class_id": true_class,
            "true_class": CLASS_MAPPING[true_class],
            "predicted_class_id": base_prediction,
            "predicted_class": CLASS_MAPPING[base_prediction],
            "correct_prediction": bool(base_prediction == true_class),
            "Stability_S": float(np.mean(stability)),
            "Consistency_C_imp": float(np.mean(consistency)),
            "Coverage_imp": float(np.mean(coverage)),
            "PredKeep_imp": float(np.mean(prediction_keep)),
            "Cosine_mean": float(np.mean(cosine_values)),
        }
        rows.append(row)

        print(
            f"[{local_position + 1:02d}/{len(selected_indices)}] "
            f"idx_test={test_position:2d} | "
            f"true={CLASS_MAPPING[true_class]:10s} | "
            f"pred={CLASS_MAPPING[base_prediction]:10s} | "
            f"S={row['Stability_S']:.6f} | "
            f"Cimp={row['Consistency_C_imp']:.6f} | "
            f"Cov={row['Coverage_imp']:.6f} | "
            f"PredKeep={row['PredKeep_imp']:.6f}"
        )

    return pd.DataFrame(rows)


def main() -> None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required = [CSV_PATH, SHAP_PICKLE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Iris artifacts:\n - " + "\n - ".join(missing)
        )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        class_weight=None,
    )

    print("=" * 72)
    print(" IRIS - STABILITY / CONSISTENCY SHAP-GLCD")
    print("=" * 72)
    print("Pair               : setosa vs virginica")
    print("Selection          : true labels, balanced")
    print("Prediction filter  : no")
    print(f"Quick test         : {QUICK_TEST}")
    print(f"N requested        : {N_INSTANCES_REQUESTED}")
    print(f"M perturbations    : {M_PERTURB}")
    print(f"sigma              : {SIGMA_RANGE} of train range")
    print(f"tau                : {TAU}")
    print(f"top-k              : {TOP_K}")
    print("Perturbation space : original feature units")
    print("Scaling            : fitted training MinMaxScaler")
    print(f"Results            : {OUTPUT_DIR}")
    print("=" * 72)

    gexp = Gexplainer(
        csv_path=str(CSV_PATH),
        target_column="target",
        model=model,
        shap_pickle=str(SHAP_PICKLE),
        j_sel=J_SEL,
        r_sel=R_SEL,
        top=7,
        random_state=RANDOM_STATE,
        class_mapping=CLASS_MAPPING,
        figure_font_scale=1.20,
    )
    gexp.entrenar_modelo_final()
    gexp.cargar_shap(filename=str(SHAP_PICKLE))

    selected_indices, n_each = select_balanced_pair(
        gexp.y_te,
        N_INSTANCES_REQUESTED,
    )

    print("\nBalanced evaluation sample")
    print(f"Effective N: {len(selected_indices)} ({n_each} + {n_each})")

    results = evaluate(gexp, selected_indices)
    results.to_csv(
        OUTPUT_DIR / "iris_stability_consistency_instances.csv",
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
        summary["mean"].map(lambda value: f"{value:.6f}")
        + " +- "
        + summary["std"].map(lambda value: f"{value:.6f}")
    )
    summary.insert(0, "Dataset", "Iris")
    summary.insert(1, "Pair", "setosa vs virginica")
    summary.insert(2, "N", len(results))

    summary.reset_index(names="Metric").to_csv(
        OUTPUT_DIR / "iris_stability_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        "test_position": selected_indices,
        "true_class_id": np.asarray(gexp.y_te)[selected_indices],
    }).to_csv(
        OUTPUT_DIR / "selected_instances_iris.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print(" FINAL RESULTS - IRIS")
    print("=" * 72)
    print(summary[["mean", "std", "median", "SE", "mean+-std"]].to_string())
    print(f"\nResults saved in:\n{OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
