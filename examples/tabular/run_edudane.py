# -*- coding: utf-8 -*-
# ============================================================
# main_glcd_edudane_fruson_OVERSAMPLE_CORREGIDA_PAIR_TRUE_BALANCED_020926.py
# ============================================================
#
# SHAP-GLCD global para EduFormalDANE / FRUSON.
#
# IMPORTANTE:
# - NO reentrena el modelo.
# - NO recalcula SHAP.
# - NO modifica los artefactos originales.
# - Carga exclusivamente los artefactos definitivos obtenidos con:
#       RandomOverSampler + CV de 7 pliegues sin fuga + seed 42.
#
# CLASE GLCD UTILIZADA:
#   glcd_explainer_from_artifacts_CORREGIDA_PAIR_TRUE_BALANCED010926.py
#
# Esa clase ya implementa el protocolo global definitivo:
#   1) selección por ETIQUETA VERDADERA;
#   2) muestra balanceada entre Cj y Cr;
#   3) semilla 42;
#   4) muestreo sin reemplazo;
#   5) NO filtra por predicción: conserva aciertos y errores;
#   6) la MISMA muestra se usa para ΔS y todas las métricas pairwise;
#   7) KDE pairwise, Cohen d, KS, t_opt, t_bnd y P/R usan esa muestra;
#   8) las métricas predictivas del modelo se calculan sobre TODO test;
#   9) KDE all-classes usa TODO el test.
#
# NUEVO:
#   10) si la característica top-1 es binaria, genera una figura
#       complementaria de proporciones 0/1 usando EXACTAMENTE
#       la misma muestra balanceada SHAP-GLCD.
#
# Par actual:
#   J_SEL = 0 -> Diario
#   R_SEL = 3 -> Ocasional
#
# Convención:
#   ΔS = SHAP(Cj) - SHAP(Cr)
#   ΔS > 0 favorece Cj
#   ΔS < 0 favorece Cr
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# RUTAS GENERALES DEL REPOSITORIO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.artifacts import GLCDExplainerFromArtifacts


# ============================================================
# IMPORTAR LA CLASE METODOLÓGICAMENTE CORREGIDA
# ============================================================
#
# IMPORTANTE:
# Deben estar en la misma carpeta del proyecto:
#
#   - glcd_explainer_from_artifacts.py
#   - glcd_explainer_from_artifacts_CORREGIDA_PAIR_TRUE_BALANCED010926.py
#
# La clase corregida hereda de la clase original.
# ============================================================

#from glcd_explainer_from_artifacts_CORREGIDA_PAIR_TRUE_BALANCED010926 import (
 #   GLCDExplainerFromArtifacts,
#)


# ============================================================
# IMPORTANTE PARA CARGAR EL MODELO GUARDADO
# ============================================================
#
# Esta clase se define con el mismo nombre utilizado en el main
# definitivo de SHAP-LCD de EduFormalDANE.
#
# De esta forma joblib puede reconstruir correctamente el modelo
# guardado desde __main__.
# ============================================================

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


# ============================================================
# INFORMACIÓN DEL ENTORNO
# ============================================================

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
print("joblib:", joblib.__version__)
print("OS:", platform.platform())
print("Processor:", platform.processor())


# ============================================================
# SEMILLA
# ============================================================

RANDOM_STATE = 42


def set_seed(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)


set_seed(RANDOM_STATE)


# ============================================================
# CONFIGURACIÓN SHAP-GLCD
# ============================================================

BASE_DIR = PROJECT_ROOT


# ------------------------------------------------------------
# ARTEFACTOS DEFINITIVOS:
# RandomOverSampler + CV sin fuga + seed 42
# ------------------------------------------------------------

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "edudane"


# ------------------------------------------------------------
# CLASES DE FRUSON
# ------------------------------------------------------------

CLASS_NAMES = {

    0: "Diario",

    1: "Semanal",

    2: "Mensual",

    3: "Ocasional",

    4: "Nunca",
}


# ------------------------------------------------------------
# PAR ACTUAL
# ------------------------------------------------------------
#
# 0 = Diario
# 1 = Semanal
# 2 = Mensual
# 3 = Ocasional
# 4 = Nunca
# ------------------------------------------------------------

J_SEL = 0       # Diario

R_SEL = 3       # Ocasional


TOP = 10

FIGURE_FONT_SCALE = 1.20


PAIR_TAG = f"{J_SEL}_vs_{R_SEL}"


# Carpeta de resultados definitiva de esta ejecución.

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "edudane_smoke_test"
    / f"pair_{PAIR_TAG}"
)


# ============================================================
# VERIFICAR CARPETA Y ARCHIVOS DE ARTEFACTOS
# ============================================================

print("\nDirectorio actual de ejecución:")

print(BASE_DIR)


print("\nRuta absoluta de la carpeta de artefactos:")

print(ARTIFACT_DIR.resolve())


if not ARTIFACT_DIR.exists():

    raise FileNotFoundError(

        f"No se encontró la carpeta:\n{ARTIFACT_DIR}\n\n"

        "Debe utilizarse la carpeta definitiva con oversample_010926."
    )


required_files = [

    "X_test_final.csv",

    "y_test_final.csv",

    "feature_names.csv",

    "class_names.json",

    "shap_values_3d.npy",

    "random_forest_optimizado.joblib",
]


missing_files = [

    filename

    for filename in required_files

    if not (ARTIFACT_DIR / filename).exists()
]


if missing_files:

    raise FileNotFoundError(

        "Faltan archivos requeridos para ejecutar SHAP-GLCD:\n"

        + "\n".join(
            f" - {filename}"
            for filename in missing_files
        )
    )


print("\nArchivos requeridos encontrados correctamente:")

for filename in required_files:

    print(f" - {filename}")


# ============================================================
# VERIFICAR NOMBRES DE CLASE DESDE LOS ARTEFACTOS
# ============================================================

with open(

    ARTIFACT_DIR / "class_names.json",

    "r",

    encoding="utf-8",

) as f:

    artifact_class_names_raw = json.load(f)


artifact_class_names = {

    int(k): str(v)

    for k, v in artifact_class_names_raw.items()
}


if artifact_class_names != CLASS_NAMES:

    raise ValueError(

        "\nLas clases de class_names.json no coinciden con CLASS_NAMES.\n"

        f"Artefactos: {artifact_class_names}\n"

        f"Main      : {CLASS_NAMES}"
    )


# ============================================================
# CREAR CARPETA DE RESULTADOS
# ============================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,
)


# ============================================================
# MOSTRAR CONFIGURACIÓN ANTES DE EJECUTAR
# ============================================================

print("\n============================================================")

print(" VERIFICACIÓN DE CONFIGURACIÓN SHAP-GLCD")

print("============================================================")

print("Dataset código       : EduDane")

print("Dataset artículo     : EduFormalDANE")

print("Variable objetivo    : FRUSON")

print(
    f"Clase j              : "
    f"{J_SEL} - {CLASS_NAMES[J_SEL]}"
)

print(
    f"Clase r              : "
    f"{R_SEL} - {CLASS_NAMES[R_SEL]}"
)

print(
    f"Top features         : "
    f"{TOP}"
)

print(
    "Selección            : "
    "etiqueta verdadera"
)

print(
    "Balanceo muestra     : "
    "sí, máximo balanceado disponible"
)

print(
    "Muestreo             : "
    "sin reemplazo"
)

print(
    "Filtro predicción    : "
    "NO"
)

print(
    "Aciertos + errores   : "
    "incluidos"
)

print(
    f"Semilla              : "
    f"{RANDOM_STATE}"
)

print(
    f"Escala de fuente     : "
    f"{FIGURE_FONT_SCALE}"
)

print(
    "Clase GLCD           : "
    "CORREGIDA_PAIR_TRUE_BALANCED010926"
)

print(
    "Gráfico binario top1 : "
    "sí, cuando la top-1 sea binaria"
)

print(
    f"Resultados en        : "
    f"{OUTPUT_DIR}"
)

print("============================================================")


# ============================================================
# INSTANCIAR SHAP-GLCD DESDE ARTEFACTOS
# ============================================================
#
# NO se crea ninguna subclase dentro del main.
#
# La selección true-label balanceada ya está implementada dentro
# de GLCDExplainerFromArtifacts de la clase corregida.
# ============================================================

gexp = GLCDExplainerFromArtifacts(

    artifact_dir=ARTIFACT_DIR,

    j_sel=J_SEL,

    r_sel=R_SEL,

    top=TOP,

    random_state=RANDOM_STATE,

    output_dir=OUTPUT_DIR,

    load_model=True,

    figure_font_scale=FIGURE_FONT_SCALE,
)


# ============================================================
# PIPELINE PRINCIPAL SHAP-GLCD
# ============================================================
#
# Este paso, mediante la clase corregida:
#
# - selecciona por etiqueta verdadera;
# - balancea Cj y Cr con el máximo N posible;
# - conserva predicciones correctas e incorrectas;
# - usa seed 42;
# - calcula ΔS sobre esa muestra;
# - calcula Mean |ΔS|;
# - calcula Median |ΔS|;
# - calcula Signed Mean;
# - genera Conditional Mean;
# - genera KDE pairwise;
# - calcula Cohen d;
# - calcula |d|;
# - calcula KS-D;
# - calcula p-KS;
# - calcula t_opt;
# - calcula t_bnd;
# - calcula precision / recall;
# - calcula métricas predictivas sobre TODO el test externo.
#
# NO recalcula SHAP.
#
# NO vuelve a entrenar el modelo.
# ============================================================

gexp.run_glcd(

    make_plots=True,

    show=True,

    dpi=600,
)


# ============================================================
# FIGURA B:
# DOMINANCE PLOT / 100% STACKED WEIGHTED
# ============================================================

figB_name = (

    f"figB_edudane_FRUSON_RF_"

    f"weighted_100stack_{PAIR_TAG}.png"
)


gexp.plot_figura_B_100stack_weighted(

    dpi=600,

    fname=figB_name,

    show=True,
)


# ============================================================
# ANÁLISIS ADICIONAL PARA TOP-1
# ============================================================

if (
    not hasattr(gexp, "rank")
    or gexp.rank is None
    or gexp.rank.empty
):

    raise RuntimeError(

        "No existe ranking global para identificar "
        "la característica top-1."
    )


top_feat = gexp.rank.iloc[0]["Feature"]


print(

    f"\n=== Extra conditional mean analysis "

    f"for top-1 feature: {top_feat} ==="
)


gexp.extra_feature_analysis(

    top_feat
)


# ============================================================
# NUEVO:
# DISTRIBUCIÓN DE PROPORCIONES PARA TOP-1 SI ES BINARIA
# ============================================================
#
# OBJETIVO:
#
# Para características binarias como NIVNY_Preescolar,
# una KDE no es la representación más natural porque los valores
# reales son únicamente 0 y 1.
#
# Por ello se genera adicionalmente una figura de proporciones:
#
#   valor 0 -> ausencia de la categoría
#   valor 1 -> presencia de la categoría
#
# IMPORTANTE:
#
# - utiliza EXACTAMENTE gexp.X_te_pair y gexp.y_te_pair;
# - por tanto utiliza la MISMA muestra balanceada del SHAP-GLCD;
# - NO usa todo el test;
# - NO vuelve a seleccionar observaciones;
# - NO recalcula SHAP;
# - NO vuelve a entrenar el modelo.
#
# La KDE original se conserva como salida diagnóstica.
# Para el artículo puede utilizarse la figura de proporciones
# cuando la top-1 sea binaria.
# ============================================================

print("\n============================================================")

print(" DISTRIBUCIÓN BINARIA DE LA CARACTERÍSTICA TOP-1")

print("============================================================")


# ------------------------------------------------------------
# Verificar que la muestra pairwise está disponible
# ------------------------------------------------------------

if (
    not hasattr(gexp, "X_te_pair")
    or not hasattr(gexp, "y_te_pair")
):

    raise RuntimeError(

        "No está disponible la muestra pairwise balanceada.\n"

        "Debe ejecutarse primero gexp.run_glcd()."
    )


if top_feat not in gexp.X_te_pair.columns:

    raise ValueError(

        f"La característica top-1 '{top_feat}' "
        "no está en X_te_pair."
    )


# ------------------------------------------------------------
# Extraer valores observados
# ------------------------------------------------------------

top_values = (

    gexp.X_te_pair[top_feat]

    .dropna()

    .astype(float)
)


unique_values = np.sort(

    top_values.unique()
)


# ------------------------------------------------------------
# Verificar si los valores son efectivamente 0 / 1
#
# Se usa np.isclose para tolerar pequeños errores numéricos.
# ------------------------------------------------------------

is_binary = False


if len(unique_values) <= 2:

    is_binary = all(

        np.isclose(value, 0.0)
        or np.isclose(value, 1.0)

        for value in unique_values
    )


print(
    f"Característica top-1 : "
    f"{top_feat}"
)

print(
    f"Valores observados   : "
    f"{unique_values}"
)

print(
    f"Variable binaria     : "
    f"{is_binary}"
)


# ============================================================
# SI LA TOP-1 ES BINARIA
# ============================================================

if is_binary:

    # --------------------------------------------------------
    # Calcular conteos y proporciones exactos dentro de
    # cada clase de la muestra SHAP-GLCD balanceada.
    # --------------------------------------------------------

    binary_records = []


    for class_id in [J_SEL, R_SEL]:

        mask_class = (

            gexp.y_te_pair == class_id
        )


        class_values = (

            gexp.X_te_pair.loc[
                mask_class,
                top_feat
            ]

            .dropna()

            .astype(float)
        )


        n_class = len(class_values)


        if n_class == 0:

            raise RuntimeError(

                f"No hay observaciones para la clase "
                f"{class_id} en la muestra pairwise."
            )


        # ----------------------------------------------------
        # Contar aproximadamente 0 y 1.
        # np.isclose evita problemas de precisión numérica.
        # ----------------------------------------------------

        count_zero = int(

            np.sum(
                np.isclose(
                    class_values.to_numpy(),
                    0.0
                )
            )
        )


        count_one = int(

            np.sum(
                np.isclose(
                    class_values.to_numpy(),
                    1.0
                )
            )
        )


        # ----------------------------------------------------
        # Comprobación:
        # todos los valores deben corresponder a 0 o 1.
        # ----------------------------------------------------

        if count_zero + count_one != n_class:

            raise RuntimeError(

                f"La variable {top_feat} fue identificada como binaria, "

                f"pero para la clase {class_id} existen valores "

                "que no corresponden a 0 o 1."
            )


        for value, count in [

            (0, count_zero),

            (1, count_one),

        ]:

            proportion = count / n_class


            binary_records.append({

                "feature": top_feat,

                "class_id": class_id,

                "class_name": gexp.class_mapping[class_id],

                "value": value,

                "count": count,

                "n_class": n_class,

                "proportion": proportion,

                "percentage": 100.0 * proportion,
            })


    # --------------------------------------------------------
    # Crear DataFrame
    # --------------------------------------------------------

    binary_df = pd.DataFrame(

        binary_records
    )


    # --------------------------------------------------------
    # Guardar tabla exacta
    # --------------------------------------------------------

    binary_csv_path = (

        OUTPUT_DIR

        / (

            f"binary_distribution_"

            f"{top_feat}_"

            f"{PAIR_TAG}.csv"
        )
    )


    binary_df.to_csv(

        binary_csv_path,

        index=False
    )


    print("\nDistribución binaria exacta:")

    print(

        binary_df.to_string(
            index=False
        )
    )


    print(

        "\nCSV distribución binaria guardado en:\n"

        f"{binary_csv_path}"
    )


    # --------------------------------------------------------
    # Extraer porcentajes para Cj
    # --------------------------------------------------------

    percentages_j = []


    # --------------------------------------------------------
    # Extraer porcentajes para Cr
    # --------------------------------------------------------

    percentages_r = []


    for value in [0, 1]:

        value_j = binary_df.loc[

            (
                binary_df["class_id"] == J_SEL
            )

            &

            (
                binary_df["value"] == value
            ),

            "percentage"
        ]


        value_r = binary_df.loc[

            (
                binary_df["class_id"] == R_SEL
            )

            &

            (
                binary_df["value"] == value
            ),

            "percentage"
        ]


        percentages_j.append(

            float(value_j.iloc[0])

            if not value_j.empty

            else 0.0
        )


        percentages_r.append(

            float(value_r.iloc[0])

            if not value_r.empty

            else 0.0
        )


    # --------------------------------------------------------
    # Etiquetas del eje x
    #
    # NIVNY_Preescolar es una dummy:
    #
    # 0 = el registro no corresponde a Preescolar
    # 1 = el registro corresponde a Preescolar
    #
    # Para cualquier otra top-1 binaria se utilizan 0 y 1.
    # --------------------------------------------------------

    if top_feat == "NIVNY_Preescolar":

        x_labels = [

            "0 – No Preescolar",

            "1 – Preescolar",
        ]

    else:

        x_labels = [

            "0",

            "1",
        ]


    # --------------------------------------------------------
    # Crear figura
    # --------------------------------------------------------

    x = np.arange(2)


    width = 0.36


    fig, ax = plt.subplots(

        figsize=(13, 8)
    )


    # --------------------------------------------------------
    # Barras Cj
    # --------------------------------------------------------

    bars_j = ax.bar(

        x - width / 2,

        percentages_j,

        width,

        color="#56B4E9",

        label=gexp.class_mapping[J_SEL],
    )


    # --------------------------------------------------------
    # Barras Cr
    # --------------------------------------------------------

    bars_r = ax.bar(

        x + width / 2,

        percentages_r,

        width,

        color="#ECE133",

        label=gexp.class_mapping[R_SEL],
    )


    # --------------------------------------------------------
    # Escribir porcentajes sobre las barras
    # --------------------------------------------------------

    for bars in [

        bars_j,

        bars_r,

    ]:

        for bar in bars:

            height = float(
                bar.get_height()
            )


            ax.text(

                bar.get_x()
                + bar.get_width() / 2,

                height + 1.5,

                f"{height:.1f}%",

                ha="center",

                va="bottom",

                fontsize=gexp._fs(20),
            )


    # --------------------------------------------------------
    # Configuración del eje x
    # --------------------------------------------------------

    ax.set_xticks(

        x
    )


    ax.set_xticklabels(

        x_labels,

        fontsize=gexp._fs(22),
    )


    # --------------------------------------------------------
    # Eje y:
    # porcentaje dentro de cada clase
    # --------------------------------------------------------

    ax.set_ylim(

        0,

        105
    )


    ax.set_ylabel(

        "Within-class proportion (%)",

        fontsize=gexp._fs(26),
    )


    # --------------------------------------------------------
    # Etiqueta eje x
    # --------------------------------------------------------

    ax.set_xlabel(

        f"{top_feat} (binary)",

        fontsize=gexp._fs(26),
    )


    # --------------------------------------------------------
    # Título
    # --------------------------------------------------------

    ax.set_title(

        f"Distribution of {top_feat}\n"

        f"{gexp.class_mapping[J_SEL]} and "

        f"{gexp.class_mapping[R_SEL]}",

        fontsize=gexp._fs(30),
    )


    # --------------------------------------------------------
    # Tamaño etiquetas eje y
    # --------------------------------------------------------

    ax.tick_params(

        axis="y",

        labelsize=gexp._fs(20),
    )


    # --------------------------------------------------------
    # Leyenda
    # --------------------------------------------------------

    ax.legend(

        fontsize=gexp._fs(21),

        loc="upper right",
    )


    # --------------------------------------------------------
    # Rejilla solo horizontal
    # --------------------------------------------------------

    ax.grid(

        axis="y",

        alpha=0.25,
    )


    ax.set_axisbelow(

        True
    )


    fig.tight_layout()


    # --------------------------------------------------------
    # Guardar PNG
    # --------------------------------------------------------

    binary_png_path = (

        OUTPUT_DIR

        / (

            f"binary_distribution_"

            f"{top_feat}_"

            f"{PAIR_TAG}.png"
        )
    )


    # --------------------------------------------------------
    # Guardar PDF
    # --------------------------------------------------------

    binary_pdf_path = (

        OUTPUT_DIR

        / (

            f"binary_distribution_"

            f"{top_feat}_"

            f"{PAIR_TAG}.pdf"
        )
    )


    fig.savefig(

        binary_png_path,

        dpi=600,

        bbox_inches="tight",
    )


    fig.savefig(

        binary_pdf_path,

        bbox_inches="tight",
    )


    plt.show()


    plt.close(fig)


    print(

        "\nFigura PNG guardada en:\n"

        f"{binary_png_path}"
    )


    print(

        "\nFigura PDF guardada en:\n"

        f"{binary_pdf_path}"
    )


    print("============================================================")


# ============================================================
# SI LA TOP-1 NO ES BINARIA
# ============================================================

else:

    print(

        f"\nLa característica top-1 '{top_feat}' "

        "no es binaria."
    )


    print(

        "No se genera el gráfico de proporciones 0/1."
    )


    print(

        "En este caso se conserva la visualización de densidad "

        "generada por SHAP-GLCD."
    )


    print("============================================================")


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n==========================================")

print(" PROCESO SHAP-GLCD EDUFORMALDANE FINALIZADO")

print("==========================================")


print(
    "Dataset           : "
    "EduFormalDANE"
)


print(
    "Variable objetivo : "
    "FRUSON"
)


print(
    f"Par analizado     : "
    f"{J_SEL} vs {R_SEL}"
)


print(
    f"Clase j           : "
    f"{gexp.class_mapping[J_SEL]}"
)


print(
    f"Clase r           : "
    f"{gexp.class_mapping[R_SEL]}"
)


if hasattr(gexp, "N_pair"):

    print(
        f"N global usado    : "
        f"{gexp.N_pair}"
    )

else:

    print(
        "N global usado    : "
        "no disponible"
    )


print(
    "Selección         : "
    "true label, balanced"
)


print(
    "Muestreo          : "
    "sin reemplazo"
)


print(
    "Incluye errores   : "
    "sí; no se filtró por predicción"
)


print(
    f"Top features      : "
    f"{TOP}"
)


print(
    f"Top-1             : "
    f"{top_feat}"
)


print(
    f"Top-1 binaria     : "
    f"{is_binary}"
)


print(
    f"Escala de fuente  : "
    f"{FIGURE_FONT_SCALE}"
)


print(
    "Clase GLCD        : "
    "CORREGIDA_PAIR_TRUE_BALANCED010926"
)


print(
    f"Resultados en     : "
    f"{OUTPUT_DIR}"
)


print("==========================================\n")