"""Reproducible global SHAP-GLCD experiment for BloodMNIST."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from medmnist import BloodMNIST, INFO
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.image import SHAP_GLCD_ImageExplainer_MapsOnly  # noqa: E402


# -----------------------------------------------------------------------------
# Reproducible configuration
# -----------------------------------------------------------------------------
SEED = 42
J_SEL = 6  # neutrophil
R_SEL = 1  # eosinophil

BACKGROUND_SIZE = 100
TOP_PERCENT = 12.0
DPI = 600

# The definitive SHAP cache already exists. Using only_load prevents an
# accidental and computationally expensive recalculation.
SHAP_MODE = "only_load"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bloodmnist"
MODEL_PATH = ARTIFACT_DIR / "bloodmnist_resnet18_best.keras"
SHAP_PATH = (
    ARTIFACT_DIR
    / "shap_values_bloodmnist_resnet18_best_6_vs_1.pickle"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "bloodmnist"


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
    if images.ndim == 3:
        images = images[..., np.newaxis]
    return images, labels


def print_distribution(
    labels: np.ndarray,
    title: str,
    class_names: dict[int, str],
) -> None:
    print(f"\n{title}")
    for class_id in sorted(class_names):
        count = int(np.sum(labels == class_id))
        print(f"{class_id} ({class_names[class_id]}): {count}")


def main() -> None:
    set_reproducibility(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        str(path)
        for path in (MODEL_PATH, SHAP_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required BloodMNIST artifacts:\n - "
            + "\n - ".join(missing)
        )

    class_names = {
        int(key): value
        for key, value in INFO["bloodmnist"]["label"].items()
    }
    display_names = {
        0: "Basophil",
        1: "Eosinophil",
        2: "Erythroblast",
        3: "Immature granulocytes",
        4: "Lymphocyte",
        5: "Monocyte",
        6: "Neutrophil",
        7: "Platelet",
    }

    print("=" * 76)
    print(" SHAP-GLCD - BLOODMNIST - RESNET-18")
    print("=" * 76)
    print(f"Model          : {MODEL_PATH}")
    print(f"SHAP pickle    : {SHAP_PATH}")
    print(f"Results        : {RESULTS_DIR}")
    print(
        f"Pair           : {display_names[J_SEL]} vs "
        f"{display_names[R_SEL]}"
    )
    print("Selection      : true labels, strictly balanced pair")
    print("Pred. filter   : no")
    print(f"Seed           : {SEED}")
    print(f"SHAP mode      : {SHAP_MODE}")
    print("=" * 76)

    print("\nLoading BloodMNIST...")
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    print(f"Train      : {x_train.shape} | {y_train.shape}")
    print(f"Validation : {x_val.shape} | {y_val.shape}")
    print(f"Test       : {x_test.shape} | {y_test.shape}")
    print_distribution(y_test, "Official test distribution", class_names)

    print(f"\nLoading definitive model:\n{MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nCalculating final model metrics on the official test set...")
    probabilities = model.predict(x_test, batch_size=64, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, predictions),
        "F1_Macro": f1_score(y_test, predictions, average="macro"),
        "MCC": matthews_corrcoef(y_test, predictions),
        "Macro_AUC_OvO": roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovo",
            average="macro",
        ),
    }

    print("\nFinal model metrics")
    for name, value in metrics.items():
        print(f"{name:20s}: {value:.6f}")

    matrix = confusion_matrix(y_test, predictions)
    print("\nConfusion matrix:")
    print(matrix)

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        matrix,
        index=[display_names[index] for index in sorted(display_names)],
        columns=[display_names[index] for index in sorted(display_names)],
    ).to_csv(
        RESULTS_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    print("\nLoading cached SHAP values and generating global maps...")
    explainer = SHAP_GLCD_ImageExplainer_MapsOnly(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        j_sel=J_SEL,
        r_sel=R_SEL,
        class_names=class_names,
        display_class_names=display_names,
        shap_pickle=str(SHAP_PATH),
        out_dir=str(RESULTS_DIR),
        dpi=DPI,
        random_state=SEED,
        show_plots=True,
        strict_pair_balance=True,
    )

    # compute_kwargs are retained for complete reproducibility if a future
    # user explicitly chooses a calculation mode. In this run only_load is
    # mandatory and therefore the existing pickle is never overwritten.
    compute_kwargs = {
        "background_size": BACKGROUND_SIZE,
        "max_test": len(x_test),
        "explainer_type": "gradient",
        "eval_pair_only": True,
    }

    explainer.run_all_maps(
        shap_mode=SHAP_MODE,
        compute_kwargs=compute_kwargs,
        top_percent=TOP_PERCENT,
        make_parallel=True,
    )

    selected_counts = {
        display_names[class_id]: int(np.sum(explainer.y_eval == class_id))
        for class_id in (J_SEL, R_SEL)
    }
    print("\nFinal balanced SHAP-GLCD sample:")
    for name, count in selected_counts.items():
        print(f"{name}: {count}")

    pd.DataFrame(
        {
            "test_index": explainer.idx_eval,
            "true_class_id": explainer.y_eval,
            "true_class": [
                display_names[int(value)] for value in explainer.y_eval
            ],
        }
    ).to_csv(
        RESULTS_DIR / "selected_balanced_test_indices.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 76)
    print(" BLOODMNIST EXECUTION COMPLETED")
    print("=" * 76)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
