"""Reproducible SHAP-GLCD experiment for the Seeds dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sklearn.neural_network import MLPClassifier


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
J_SEL = 0  # Kama
R_SEL = 1  # Rosa
TOP = 7
FIGURE_FONT_SCALE = 1.20

# First execution: "always". After the new artifact has been generated and
# its probability scale verified, change this value to "auto".
SHAP_MODE = "auto"

CLASS_MAPPING = {
    0: "Kama",
    1: "Rosa",
    2: "Canadian",
}

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "seeds"
RESULTS_DIR = PROJECT_ROOT / "results" / "seeds"

CSV_PATH = ARTIFACT_DIR / "seeds.csv"
SHAP_PICKLE = (
    ARTIFACT_DIR
    / f"shap_values_seeds_mlp_probability_{J_SEL}_vs_{R_SEL}.pickle"
)
FIGURE_B = (
    f"figB_seeds_mlp_probability_weighted_100stack_"
    f"{J_SEL}_vs_{R_SEL}.png"
)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            "The Seeds dataset was not found:\n"
            f"{CSV_PATH}\n\n"
            "Copy seeds.csv into artifacts/seeds before running this example."
        )

    # The current tabular implementation writes result files using relative
    # paths. Running from RESULTS_DIR keeps every output in results/seeds.
    os.chdir(RESULTS_DIR)

    model = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        max_iter=2000,
        random_state=SEED,
    )

    print("=" * 72)
    print(" SHAP-GLCD - SEEDS - MLP")
    print("=" * 72)
    print(f"Dataset       : {CSV_PATH}")
    print(f"SHAP pickle   : {SHAP_PICKLE}")
    print(f"Results       : {RESULTS_DIR}")
    print("Pair          : Kama vs Rosa")
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
    print(" SEEDS EXECUTION COMPLETED")
    print("=" * 72)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
