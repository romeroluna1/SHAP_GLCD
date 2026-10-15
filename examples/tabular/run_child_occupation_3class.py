"""SHAP-GLCD: ocupacion de menores (3 clases), desde artefactos existentes.

Ubicar en: SHAP_GLCD_GITHUB/examples/tabular/run_child_occupation_3class.py
Ejecutar desde la raiz del repositorio:
    python examples/tabular/run_child_occupation_3class.py
No entrena el modelo ni recalcula SHAP.
"""
from pathlib import Path
import os
import sys
import json
import random
import platform

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier

# Compatibilidad con el modelo guardado originalmente desde __main__.
# Debe declararse antes de que la clase comun cargue el archivo joblib.
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
            random_state=random_state,
        )

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.artifacts import GLCDExplainerFromArtifacts

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

J_SEL = 0  # trabaj
R_SEL = 2  # oficio_hogar
TOP = 7
FIGURE_FONT_SCALE = 1.0
PAIR_TAG = f"{J_SEL}_vs_{R_SEL}"

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "child_occupation_3class"
OUTPUT_DIR = REPO_ROOT / "results" / "child_occupation_3class" / f"pair_{PAIR_TAG}"

REQUIRED_FILES = (
    "X_test_final.csv",
    "y_test_final.csv",
    "feature_names.csv",
    "class_names.json",
    "shap_values_3d.npy",
    "random_forest_optimizado.joblib",
    "test_probabilities.csv",
    "test_predictions.csv",
    "test_original_indices.csv",
    "metadata.json",
)


def main():
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("seaborn:", sns.__version__)
    print("joblib:", joblib.__version__)
    print("OS:", platform.platform())
    print("Artefactos:", ARTIFACT_DIR)
    print("Resultados:", OUTPUT_DIR)

    missing = [name for name in REQUIRED_FILES if not (ARTIFACT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Faltan artefactos en " + str(ARTIFACT_DIR) + ": " + ", ".join(missing))

    with (ARTIFACT_DIR / "class_names.json").open(encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, dict):
        raise ValueError("class_names.json debe contener un diccionario de clases.")
    class_names = {int(k): str(v) for k, v in content.items()}
    expected = {0: "trabaj", 1: "bustrabajo", 2: "oficio_hogar"}
    if class_names != expected:
        raise ValueError(f"Clases diferentes de las esperadas: {class_names}")

    y_df = pd.read_csv(ARTIFACT_DIR / "y_test_final.csv")
    y_test = y_df.iloc[:, 0].to_numpy()
    count_j = int(np.sum(y_test == J_SEL))
    count_r = int(np.sum(y_test == R_SEL))
    print("Clases:", class_names)
    print(f"Casos reales disponibles para el par: {J_SEL}={count_j}, {R_SEL}={count_r}")
    print("IMPORTANTE: verificar en consola que la clase comun seleccione por etiqueta verdadera,")
    print("mantenga balance 1:1, incluya aciertos/errores y use semilla 42.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gexp = GLCDExplainerFromArtifacts(
        artifact_dir=str(ARTIFACT_DIR),
        j_sel=J_SEL,
        r_sel=R_SEL,
        top=TOP,
        random_state=RANDOM_STATE,
        output_dir=str(OUTPUT_DIR),
        load_model=True,
        figure_font_scale=FIGURE_FONT_SCALE,
    )
    gexp.run_glcd(make_plots=True, show=True, dpi=600)

    fig_b_name = f"figB_child_occupation_3class_RF_weighted_100stack_{PAIR_TAG}.png"
    gexp.plot_figura_B_100stack_weighted(dpi=600, fname=fig_b_name, show=True)

    top_feature = gexp.rank.iloc[0]["Feature"]
    print("Analisis adicional de la primera caracteristica:", top_feature)
    gexp.extra_feature_analysis(top_feature)

    print("\nSHAP-GLCD 3 clases finalizado")
    print("Par:", J_SEL, class_names[J_SEL], "vs", R_SEL, class_names[R_SEL])
    print("Resultados:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
