"""Reproducible SHAP-GLCD stability evaluation for BloodMNIST."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from medmnist import BloodMNIST


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bloodmnist"
MODEL_PATH = ARTIFACT_DIR / "bloodmnist_resnet18_best.keras"
SHAP_PATH = ARTIFACT_DIR / "shap_values_bloodmnist_resnet18_best_6_vs_1.pickle"


# -----------------------------------------------------------------------------
# Reproducible configuration
# -----------------------------------------------------------------------------
SEED = 42
CLASS_J = 6  # neutrophil
CLASS_R = 1  # eosinophil

CLASS_NAMES = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature granulocytes",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet",
}

# Start with the small verification run.
QUICK_TEST = False

if QUICK_TEST:
    N_INSTANCES = 2       # 1 neutrophil + 1 eosinophil
    M_PERTURBATIONS = 2
    RUN_NAME = "quick_test"
else:
    N_INSTANCES = 50      # 25 neutrophils + 25 eosinophils
    M_PERTURBATIONS = 40
    RUN_NAME = "final"

SIGMA = 0.005
TAU = 0.95
TOP_K = 10
BACKGROUND_SIZE = 100

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "bloodmnist"
    / RUN_NAME
)


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
        print("Deterministic operations enabled.")
    except Exception as exc:
        print(f"Deterministic operations unavailable: {exc}")


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = BloodMNIST(split=split, download=True)
    images = dataset.imgs.astype("float32") / 255.0
    labels = np.asarray(dataset.labels).reshape(-1).astype("int64")
    return images, labels


def to_shap_array(values) -> np.ndarray:
    """Convert SHAP output to (N,H,W,C,K)."""
    if isinstance(values, list):
        return np.stack([np.asarray(item) for item in values], axis=-1)

    array = np.asarray(values)
    if array.ndim != 5:
        raise ValueError(
            "Expected SHAP values with shape (N,H,W,C,K); "
            f"received {array.shape}."
        )
    return array


def topk_l2(values: np.ndarray, top_k: int, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1).copy()
    if top_k < vector.size:
        selected = np.argpartition(np.abs(vector), -top_k)[-top_k:]
        mask = np.zeros(vector.size, dtype=bool)
        mask[selected] = True
        vector[~mask] = 0.0
    return vector / (np.linalg.norm(vector) + eps)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denominator = (np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)
    return float(np.dot(a, b) / denominator)


def choose_balanced_cache_positions(
    cached_labels: np.ndarray,
    n_total: int,
    seed: int,
) -> np.ndarray:
    if n_total % 2 != 0:
        raise ValueError("N_INSTANCES must be even for balanced sampling.")

    positions_j = np.flatnonzero(cached_labels == CLASS_J)
    positions_r = np.flatnonzero(cached_labels == CLASS_R)
    n_each = n_total // 2

    if len(positions_j) < n_each or len(positions_r) < n_each:
        raise ValueError("The cached SHAP sample is too small for the request.")

    rng = np.random.default_rng(seed)
    selected_j = rng.choice(positions_j, size=n_each, replace=False)
    selected_r = rng.choice(positions_r, size=n_each, replace=False)
    selected = np.concatenate([selected_j, selected_r])
    rng.shuffle(selected)
    return selected


def summarize(instance_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Stability_S",
        "Consistency_C_predexp",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]
    rows = []
    for metric in metrics:
        values = instance_results[metric].dropna().astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        median = float(values.median())
        se = std / np.sqrt(len(values)) if len(values) else np.nan
        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "median": median,
                "SE": se,
                "mean+-std": f"{mean:.6f} +- {std:.6f}",
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def main() -> None:
    start_time = time.perf_counter()
    set_reproducibility(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    missing = [path for path in (MODEL_PATH, SHAP_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing BloodMNIST artifacts:\n - "
            + "\n - ".join(str(path) for path in missing)
        )

    print("=" * 76)
    print(" BLOODMNIST - STABILITY / CONSISTENCY SHAP-GLCD")
    print("=" * 76)
    print("Pair               : Neutrophil vs Eosinophil")
    print("Selection          : true labels, balanced")
    print("Prediction filter  : no")
    print(f"Quick test         : {QUICK_TEST}")
    print(f"N requested        : {N_INSTANCES}")
    print(f"M perturbations    : {M_PERTURBATIONS}")
    print(f"sigma              : {SIGMA}")
    print(f"tau                : {TAU}")
    print(f"top-k              : {TOP_K}")
    print("Perturbation       : Gaussian pixel noise + clip [0,1]")
    print("Base SHAP          : original cached pickle")
    print("Perturbed SHAP     : GradientExplainer")
    print(f"Seed rule          : seed + t")
    print(f"Results            : {RESULTS_DIR}")
    print("=" * 76)

    print("\nLoading BloodMNIST train and test sets...")
    x_train, _ = load_split("train")
    x_test, y_test = load_split("test")
    print(f"Train: {x_train.shape}")
    print(f"Test : {x_test.shape} | {y_test.shape}")

    print(f"\nLoading definitive model:\n{MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nLoading original cached SHAP values...")
    with SHAP_PATH.open("rb") as file:
        cached = pickle.load(file)
    if not isinstance(cached, dict) or "shap_values" not in cached or "idx_eval" not in cached:
        raise ValueError("The BloodMNIST pickle must contain shap_values and idx_eval.")

    shap_cached = to_shap_array(cached["shap_values"])
    idx_cached = np.asarray(cached["idx_eval"], dtype=int)
    if len(shap_cached) != len(idx_cached):
        raise ValueError("SHAP values and idx_eval have different lengths.")

    y_cached = y_test[idx_cached]
    selected_positions = choose_balanced_cache_positions(
        y_cached,
        n_total=N_INSTANCES,
        seed=SEED,
    )
    selected_test_indices = idx_cached[selected_positions]
    selected_labels = y_cached[selected_positions]

    print("\nBalanced evaluation sample")
    print(f"Effective N: {len(selected_positions)}")
    print(f"Neutrophil: {int(np.sum(selected_labels == CLASS_J))}")
    print(f"Eosinophil: {int(np.sum(selected_labels == CLASS_R))}")

    # Reconstruct the same background rule used by the global image class.
    np.random.seed(SEED)
    background_indices = np.random.choice(
        len(x_train),
        size=BACKGROUND_SIZE,
        replace=False,
    )
    background = x_train[background_indices]
    print(f"Background: {background.shape}")

    print("\nBuilding GradientExplainer...")
    gradient_explainer = shap.GradientExplainer(model, background)

    rows = []
    for t, (cache_position, test_index) in enumerate(
        zip(selected_positions, selected_test_indices),
        start=0,
    ):
        instance_seed = SEED + t
        x = x_test[test_index]
        true_label = int(y_test[test_index])

        base_probabilities = model.predict(x[None, ...], verbose=0)[0]
        base_prediction = int(np.argmax(base_probabilities))

        rng = np.random.default_rng(instance_seed)
        noise = rng.normal(
            0.0,
            SIGMA,
            size=(M_PERTURBATIONS,) + x.shape,
        ).astype("float32")
        perturbed = np.clip(x[None, ...] + noise, 0.0, 1.0).astype("float32")

        perturbed_probabilities = model.predict(perturbed, verbose=0)
        perturbed_predictions = np.argmax(perturbed_probabilities, axis=1)
        prediction_kept = perturbed_predictions == base_prediction

        shap_perturbed = to_shap_array(
            gradient_explainer.shap_values(perturbed)
        )

        delta_base = (
            shap_cached[cache_position, ..., CLASS_J]
            - shap_cached[cache_position, ..., CLASS_R]
        )
        delta_perturbed = (
            shap_perturbed[..., CLASS_J]
            - shap_perturbed[..., CLASS_R]
        )

        explanation_base = topk_l2(delta_base, TOP_K)
        explanations_perturbed = np.stack(
            [topk_l2(delta_perturbed[m], TOP_K) for m in range(M_PERTURBATIONS)]
        )

        differences = explanations_perturbed - explanation_base[None, :]
        stability = float(np.mean(np.sum(differences ** 2, axis=1)))
        similarities = np.asarray(
            [cosine(explanation_base, item) for item in explanations_perturbed]
        )
        similar_explanation = similarities >= TAU

        kept_count = int(prediction_kept.sum())
        consistency_predexp = (
            float(np.mean(similarities[prediction_kept] >= TAU))
            if kept_count > 0
            else np.nan
        )
        consistency_imp = float(
            np.mean((~similar_explanation) | prediction_kept)
        )
        coverage_imp = float(np.mean(similar_explanation))
        predkeep_imp = float(np.mean(prediction_kept))

        rows.append(
            {
                "cache_position": int(cache_position),
                "idx_test": int(test_index),
                "true_label": true_label,
                "true_class": CLASS_NAMES[true_label],
                "base_prediction": base_prediction,
                "base_prediction_name": CLASS_NAMES[base_prediction],
                "instance_seed": instance_seed,
                "Stability_S": stability,
                "Consistency_C_predexp": consistency_predexp,
                "Consistency_C_imp": consistency_imp,
                "Coverage_imp": coverage_imp,
                "PredKeep_imp": predkeep_imp,
                "M_keep": kept_count,
            }
        )

        print(
            f"[{t + 1:02d}/{len(selected_positions)}] "
            f"idx_test={test_index:4d} | "
            f"true={CLASS_NAMES[true_label]:12s} | "
            f"pred={CLASS_NAMES[base_prediction]:12s} | "
            f"S={stability:.6f} | "
            f"Cimp={consistency_imp:.6f} | "
            f"Cov={coverage_imp:.6f} | "
            f"PredKeep={predkeep_imp:.6f}"
        )

    instance_results = pd.DataFrame(rows)
    summary = summarize(instance_results)

    instance_results.to_csv(
        RESULTS_DIR / "metrics_instances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        RESULTS_DIR / "metrics_summary.csv",
        encoding="utf-8-sig",
    )

    configuration = {
        "quick_test": QUICK_TEST,
        "seed": SEED,
        "class_j": CLASS_J,
        "class_r": CLASS_R,
        "N_instances": N_INSTANCES,
        "M_perturbations": M_PERTURBATIONS,
        "sigma": SIGMA,
        "tau": TAU,
        "top_k": TOP_K,
        "background_size": BACKGROUND_SIZE,
        "selection": "true_labels_balanced",
        "prediction_filter": False,
        "base_shap": "original_cached_pickle",
        "perturbed_shap": "GradientExplainer",
    }
    with (RESULTS_DIR / "run_configuration.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)

    elapsed_minutes = (time.perf_counter() - start_time) / 60.0
    print("\n" + "=" * 76)
    print(" FINAL RESULTS - BLOODMNIST")
    print("=" * 76)
    print(summary.to_string())
    print(f"\nTotal time: {elapsed_minutes:.2f} minutes")
    print(f"Results saved in:\n{RESULTS_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()
