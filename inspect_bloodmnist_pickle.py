"""Inspect the definitive BloodMNIST model and cached SHAP artifact."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bloodmnist"

MODEL_PATH = ARTIFACT_DIR / "bloodmnist_resnet18_best.keras"
SHAP_PATH = (
    ARTIFACT_DIR
    / "shap_values_bloodmnist_resnet18_best_6_vs_1.pickle"
)


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def describe(name: str, value) -> None:
    print(f"\n{name}")
    print(f"Type : {type(value)}")

    if isinstance(value, np.ndarray):
        print(f"Shape: {value.shape}")
        print(f"Dtype: {value.dtype}")
        if value.size:
            if np.issubdtype(value.dtype, np.number):
                print(f"Min  : {np.nanmin(value)}")
                print(f"Max  : {np.nanmax(value)}")
            if value.ndim == 1 and np.issubdtype(value.dtype, np.integer):
                print(f"Unique values: {len(np.unique(value))}")
                print(f"First values : {value[:20].tolist()}")
        return

    if isinstance(value, (list, tuple)):
        print(f"Length: {len(value)}")
        if value:
            first = np.asarray(value[0])
            print(f"First element shape: {first.shape}")
        return

    if isinstance(value, dict):
        print(f"Keys: {list(value.keys())}")
        return

    print(f"Value: {value!r}")


def main() -> None:
    print("=" * 72)
    print(" BLOODMNIST ARTIFACT INSPECTION")
    print("=" * 72)

    for label, path in (
        ("Model", MODEL_PATH),
        ("SHAP pickle", SHAP_PATH),
    ):
        print(f"\n{label}:")
        print(path)
        print(f"Exists: {path.is_file()}")
        if path.is_file():
            print(f"Size  : {size_mb(path):.2f} MB")

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

    print("\nLoading the pickle in read-only mode...")
    with SHAP_PATH.open("rb") as file:
        payload = pickle.load(file)

    describe("Main object", payload)

    if isinstance(payload, dict):
        for key, value in payload.items():
            describe(f"Key: {key!r}", value)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload[:10]):
            describe(f"Element {index}", value)
    else:
        array = np.asarray(payload)
        describe("SHAP array", array)

    print("\n" + "=" * 72)
    print(" INSPECTION COMPLETED — NO FILE WAS MODIFIED")
    print("=" * 72)


if __name__ == "__main__":
    main()
