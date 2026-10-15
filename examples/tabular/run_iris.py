"""Reproducible SHAP-GLCD experiment for the Iris dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
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
# Reproducible configuration
# -----------------------------------------------------------------------------
SEED = 42
J_SEL = 0  # setosa
R_SEL = 2  # virginica
TOP = 7
FIGURE_FONT_SCALE = 1.20

# Use "always" on the first execution. After the new probability-scale
# artifact has been created and verified, change it to "auto".
SHAP_MODE = "auto"

CLASS_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "iris"
RESULTS_DIR = PROJECT_ROOT / "results" / "iris"

CSV_PATH = ARTIFACT_DIR / "iris.csv"
SHAP_PICKLE = (
    ARTIFACT_DIR
    / f"shap_values_iris_rf_probability_{J_SEL}_vs_{R_SEL}.pickle"
)
FIGURE_B = (
    f"figB_iris_rf_probability_weighted_100stack_"
    f"{J_SEL}_vs_{R_SEL}.png"
)


def prepare_dataset() -> None:
    """Create a reproducible CSV from scikit-learn's Iris dataset."""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data["target"] = iris.target
    data.to_csv(CSV_PATH, index=False)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_dataset()

    # The tabular class currently writes figures and result tables using
    # relative filenames. Running from RESULTS_DIR keeps every generated
    # result inside the repository's results/iris directory.
    os.chdir(RESULTS_DIR)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=SEED,
        class_weight=None,
    )

    print("=" * 72)
    print(" SHAP-GLCD - IRIS - RANDOM FOREST")
    print("=" * 72)
    print(f"Dataset       : {CSV_PATH}")
    print(f"SHAP pickle   : {SHAP_PICKLE}")
    print(f"Results       : {RESULTS_DIR}")
    print("Pair          : setosa vs virginica")
    print("Selection     : true labels, balanced pair")
    print("Pred. filter  : no")
    print(f"SHAP mode     : {SHAP_MODE}")
    print("=" * 72)

    gexp = Gexplainer(
        csv_path=str(CSV_PATH),
        target_column="target",
        model=model,
        shap_pickle=str(SHAP_PICKLE),
        j_sel=J_SEL,
        r_sel=R_SEL,
        top=TOP,
        random_state=SEED,
        class_mapping=CLASS_MAPPING,
        figure_font_scale=FIGURE_FONT_SCALE,
    )

    gexp.run_all(shap_mode=SHAP_MODE)

    gexp.plot_figura_B_100stack_weighted(
        dpi=300,
        fname=FIGURE_B,
    )

    top_feature = gexp.rank.iloc[0]["Feature"]
    print(
        "\n=== Extra conditional mean analysis "
        f"for top-1 feature: {top_feature} ==="
    )
    gexp.extra_feature_analysis(top_feature)

    print("\n" + "=" * 72)
    print(" IRIS EXECUTION COMPLETED")
    print("=" * 72)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
