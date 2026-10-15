"""Verify class balance and index alignment in the BloodMNIST SHAP cache."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from medmnist import BloodMNIST, INFO


PROJECT_ROOT = Path(__file__).resolve().parent
SHAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "bloodmnist"
    / "shap_values_bloodmnist_resnet18_best_6_vs_1.pickle"
)

J_SEL = 6  # neutrophil
R_SEL = 1  # eosinophil


def main() -> None:
    print("=" * 72)
    print(" BLOODMNIST SHAP SAMPLING VERIFICATION")
    print("=" * 72)

    if not SHAP_PATH.is_file():
        raise FileNotFoundError(f"SHAP pickle not found:\n{SHAP_PATH}")

    test_dataset = BloodMNIST(split="test", download=True)
    y_test = np.asarray(test_dataset.labels).reshape(-1).astype(int)
    class_names = {
        int(key): value
        for key, value in INFO["bloodmnist"]["label"].items()
    }

    with SHAP_PATH.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict):
        raise TypeError("The SHAP pickle is not a dictionary.")
    if "shap_values" not in payload or "idx_eval" not in payload:
        raise KeyError("The pickle must contain shap_values and idx_eval.")

    shap_values = np.asarray(payload["shap_values"])
    idx_eval = np.asarray(payload["idx_eval"], dtype=int)

    if len(shap_values) != len(idx_eval):
        raise ValueError(
            "SHAP/index misalignment: "
            f"{len(shap_values)} SHAP rows vs {len(idx_eval)} indices."
        )
    if idx_eval.min() < 0 or idx_eval.max() >= len(y_test):
        raise IndexError("idx_eval contains indices outside the test set.")

    y_eval = y_test[idx_eval]
    unique, counts = np.unique(y_eval, return_counts=True)

    print(f"Official test size : {len(y_test)}")
    print(f"SHAP sample size   : {len(idx_eval)}")
    print(f"Unique indices     : {len(np.unique(idx_eval))}")
    print(f"SHAP shape         : {shap_values.shape}")

    print("\nDistribution in the complete official test set:")
    for class_id in sorted(class_names):
        count = int(np.sum(y_test == class_id))
        print(f"{class_id} ({class_names[class_id]}): {count}")

    print("\nDistribution in the cached SHAP sample:")
    for class_id, count in zip(unique, counts):
        print(f"{class_id} ({class_names[int(class_id)]}): {int(count)}")

    count_j = int(np.sum(y_eval == J_SEL))
    count_r = int(np.sum(y_eval == R_SEL))
    n_each = min(count_j, count_r)

    print("\nPair verification:")
    print(f"Class j: {class_names[J_SEL]} -> {count_j}")
    print(f"Class r: {class_names[R_SEL]} -> {count_r}")
    print(f"Currently balanced: {count_j == count_r}")
    print(f"Strict balanced size available: {n_each} + {n_each} = {2 * n_each}")

    print("\n" + "=" * 72)
    print(" VERIFICATION COMPLETED — NO FILE WAS MODIFIED")
    print("=" * 72)


if __name__ == "__main__":
    main()
