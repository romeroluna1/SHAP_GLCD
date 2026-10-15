from pathlib import Path
import os
import sys

from xgboost import XGBClassifier


# =========================================================
# 1) Rutas generales del repositorio
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "vehicle"

# Permite importar el código ubicado dentro de src.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.tabular import Gexplainer


# Crear la carpeta de resultados si todavía no existe.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# La clase actual guarda algunas salidas en el directorio de ejecución.
# Por eso establecemos como directorio de trabajo la carpeta de resultados.
os.chdir(RESULTS_DIR)


# =========================================================
# 2) Mapping de clases
# =========================================================
vehicle_class_mapping = {
    0: "bus",
    1: "saab",
    2: "opel",
    3: "van",
}


# =========================================================
# 3) Par de clases para SHAP-GLCD
# =========================================================
J_SEL = 0  # Cj: bus
R_SEL = 3  # Cr: van

pair_tag = f"{J_SEL}_vs_{R_SEL}"


# =========================================================
# 4) Archivos de entrada y salida
# =========================================================
csv_path = DATA_DIR / "statlog_vehicle_clean_abreviation.csv"

shap_pickle_name = (
    ARTIFACTS_DIR
    / f"shapvalues_vehicle_xgb_probability_{pair_tag}.pickle"
)

figB_name = (
    RESULTS_DIR
    / f"figB_vehicle_xgb_probability_weighted_100stack_{pair_tag}.png"
)


# =========================================================
# 5) Verificar archivos necesarios
# =========================================================
if not csv_path.exists():
    raise FileNotFoundError(
        "No se encontró el conjunto de datos de Vehicle:\n"
        f"{csv_path}"
    )

if not shap_pickle_name.exists():
    print(
        "\nNo se encontró el pickle SHAP en:\n"
        f"{shap_pickle_name}\n"
        "SHAP se calculará nuevamente en escala de probabilidad.\n"
    )


# =========================================================
# 6) Modelo XGBoost
# =========================================================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.90,
    colsample_bytree=0.90,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


# =========================================================
# 7) Instanciar SHAP-GLCD
# =========================================================
FIGURE_FONT_SCALE = 1.20

gexp = Gexplainer(
    csv_path=csv_path,
    target_column="class_num",
    model=model,
    shap_pickle=shap_pickle_name,
    j_sel=J_SEL,
    r_sel=R_SEL,
    top=10,
    random_state=42,
    class_mapping=vehicle_class_mapping,
    figure_font_scale=FIGURE_FONT_SCALE,
)


# =========================================================
# 8) Ejecutar el pipeline
# =========================================================
gexp.run_all(shap_mode="auto")


# =========================================================
# 9) Análisis adicional para la característica top-1
# =========================================================
top_feat = gexp.rank.iloc[0]["Feature"]

print(
    f"\n=== Extra conditional mean analysis "
    f"for top-1 feature: {top_feat} ==="
)

gexp.extra_feature_analysis(top_feat)


# =========================================================
# 10) Weighted Dominance
# =========================================================
gexp.plot_figura_B_100stack_weighted(
    dpi=300,
    fname=figB_name,
)


print("\n============================================================")
print("EJECUCIÓN DE VEHICLE FINALIZADA")
print("============================================================")
print(f"Resultados guardados en:\n{RESULTS_DIR}")