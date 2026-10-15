
from pathlib import Path
import os
import sys

import pandas as pd
from sklearn.datasets import load_wine
from lightgbm import LGBMClassifier


# =========================================================
# 1) Rutas generales del repositorio
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "wine"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.tabular import Gexplainer

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# La clase guarda algunas salidas en el directorio actual.
os.chdir(RESULTS_DIR)


# =========================================================
# 2) Configuración del experimento
# =========================================================
SEED = 42
FIGURE_FONT_SCALE = 1.20

wine_class_mapping = {
    0: "wine_class_0",
    1: "wine_class_1",
    2: "wine_class_2",
}


# =========================================================
# 3) Par de clases para SHAP-GLCD
# =========================================================
J_SEL = 1
R_SEL = 2

pair_tag = f"{J_SEL}_vs_{R_SEL}"


# =========================================================
# 4) Preparar el conjunto de datos Wine
# =========================================================
wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = wine.target

df = X.copy()
df["target"] = y

csv_path = DATA_DIR / "wine.csv"

df.to_csv(
    csv_path,
    index=False
)

print(f"Dataset Wine guardado en: {csv_path}")
print(f"Instancias: {len(df)}")
print(f"Características: {X.shape[1]}")
print(f"Clases: {sorted(df['target'].unique())}")


# =========================================================
# 5) Archivos de salida
# =========================================================
shap_pickle_name = (
    ARTIFACTS_DIR
    / f"shapvalues_wine_LGBM_probability_{pair_tag}.pickle"
)

figB_name = (
    RESULTS_DIR
    / f"figB_wine_LGBM_probability_weighted_100stack_{pair_tag}.png"
)

if not shap_pickle_name.exists():
    print(
        "\nNo se encontró un archivo SHAP en escala de probabilidad."
        "\nSe calcularán nuevos valores SHAP para Wine."
    )


# =========================================================
# 6) Modelo LightGBM original
# =========================================================
model = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    verbose=-1
)


# =========================================================
# 7) Instanciar SHAP-GLCD
# =========================================================
gexp = Gexplainer(
    csv_path=csv_path,
    target_column="target",
    model=model,
    shap_pickle=shap_pickle_name,
    j_sel=J_SEL,
    r_sel=R_SEL,
    top=7,
    random_state=SEED,
    class_mapping=wine_class_mapping,
    figure_font_scale=FIGURE_FONT_SCALE
)


# =========================================================
# 8) Ejecutar el pipeline
# =========================================================
gexp.run_all(shap_mode="auto")


# =========================================================
# 9) Análisis adicional
# =========================================================
top_feat = gexp.rank.iloc[0]["Feature"]

print(
    f"\n=== Extra conditional mean analysis "
    f"for top-1 feature: {top_feat} ==="
)

# Se conserva el comportamiento original de Wine.
if top_feat != "color_intensity":
    gexp.extra_feature_analysis(top_feat)


# =========================================================
# 10) Weighted Dominance
# =========================================================
gexp.plot_figura_B_100stack_weighted(
    dpi=300,
    fname=figB_name
)


print("\n============================================================")
print("EJECUCIÓN DE WINE FINALIZADA")
print("============================================================")
print(f"Resultados guardados en:\n{RESULTS_DIR}")
