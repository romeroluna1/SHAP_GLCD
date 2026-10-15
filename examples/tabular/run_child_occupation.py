# ============================================================
# run_child_occupation.py (adaptado del main original de menores)
# ============================================================
# SHAP-GLCD global para el conjunto de trabajo infantil (4 clases)
# usando artefactos generados previamente por SHAP-LCD.
#
# Corrección metodológica principal:
# - selección por ETIQUETA VERDADERA;
# - muestra balanceada entre las dos clases;
# - semilla 42;
# - muestreo sin reemplazo;
# - se incluyen predicciones correctas e incorrectas;
# - NO se filtra por clase predicha;
# - todas las métricas pairwise usan exactamente la misma muestra;
# - las métricas predictivas del modelo se calculan sobre TODO el test.
#
# Este main:
# - NO reentrena el modelo;
# - NO recalcula SHAP;
# - NO modifica los artefactos originales.
# - Usa la clase común del repositorio, como EduDane.
# ============================================================

import os
import sys
import json
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier

# Mismo import utilizado por run_edudane.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from shap_glcd.artifacts import GLCDExplainerFromArtifacts


# ============================================================
# IMPORTANTE PARA CARGAR EL MODELO GUARDADO
# ============================================================
# Esta clase se define aquí con el mismo nombre usado en el main
# original de SHAP-LCD. Esto permite que joblib reconstruya el
# modelo si fue guardado desde __main__.
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
            random_state=random_state
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
# CONFIGURACIÓN DEL ANÁLISIS
# ============================================================

# ------------------------------------------------------------
# Carpeta de artefactos generada por el main SHAP-LCD.
#
# Debe contener, entre otros:
# - X_test_final.csv
# - y_test_final.csv
# - feature_names.csv
# - class_names.json
# - shap_values_3d.npy
# - random_forest_optimizado.joblib
#
# Si ya copiaste la carpeta al proyecto GLCD, deja esta ruta.
# ------------------------------------------------------------

ORIGINAL_ARTIFACT_DIR = str(PROJECT_ROOT / "artifacts" / "child_occupation")

# ------------------------------------------------------------
# Limpieza de etiquetas para publicación.
#
# True:
#   crea una copia *_clean_labels y corrige problemas de encoding
#   SOLO en nombres/textos. No altera los valores numéricos ni SHAP.
#
# False:
#   usa directamente la carpeta original.
# ------------------------------------------------------------

USE_CLEAN_LABELS = True

# Mientras verificamos resultados, conviene regenerar la copia limpia
# para que siempre provenga de los artefactos actuales.
OVERWRITE_CLEAN_ARTIFACTS = True

ARTIFACT_DIR = ORIGINAL_ARTIFACT_DIR


# ------------------------------------------------------------
# Clases:
#
# 0 -> asoTrabajo
# 1 -> estudia
# 2 -> incapacitado
# 3 -> otraactividad
#
# Análisis principal actual:
# asoTrabajo vs estudia
# ------------------------------------------------------------

J_SEL = 0   # clase j: asoTrabajo
R_SEL = 1   # clase r: estudia

TOP = 7

# Escala de fuente para figuras de publicación.
FIGURE_FONT_SCALE = 1.0

PAIR_TAG = f"{J_SEL}_vs_{R_SEL}"

OUTPUT_DIR = str(PROJECT_ROOT / "results" / "child_occupation" / f"pair_{PAIR_TAG}")


# ============================================================
# COMPROBACIÓN BÁSICA DE CONFIGURACIÓN
# ============================================================

EXPECTED_CLASSES = {
    0: "asoTrabajo",
    1: "estudia",
    2: "incapacitado",
    3: "otraactividad"
}

if J_SEL == R_SEL:
    raise ValueError(
        "J_SEL y R_SEL deben representar clases diferentes."
    )

if J_SEL not in EXPECTED_CLASSES:
    raise ValueError(
        f"J_SEL={J_SEL} no pertenece a las clases esperadas."
    )

if R_SEL not in EXPECTED_CLASSES:
    raise ValueError(
        f"R_SEL={R_SEL} no pertenece a las clases esperadas."
    )

print("\n============================================================")
print(" CONFIGURACIÓN PRINCIPAL SHAP-GLCD")
print("============================================================")
print("Dataset                  : menores5to18_4class")
print(f"Clase j                  : {J_SEL} - {EXPECTED_CLASSES[J_SEL]}")
print(f"Clase r                  : {R_SEL} - {EXPECTED_CLASSES[R_SEL]}")
print("Selección global         : etiqueta verdadera")
print("Balanceo pairwise        : sí, 1:1")
print("Muestreo                 : sin reemplazo")
print("Filtro por predicción    : NO")
print("Aciertos + errores       : incluidos")
print(f"Semilla                  : {RANDOM_STATE}")
print(f"Top features             : {TOP}")
print("Métricas predictivas     : test completo")
print("============================================================\n")


# ============================================================
# LIMPIEZA DE NOMBRES CON PROBLEMAS DE CODIFICACIÓN
# ============================================================

def fix_mojibake(text):
    """
    Corrige problemas comunes de codificación en nombres de variables.

    Ejemplos:
        ï»¿No_Cuartos_Hog -> No_Cuartos_Hog
        niÃ±os -> niños
        leÃ±a -> leña
        carbÃ³n -> carbón

    Solo modifica texto/etiquetas.
    """

    if not isinstance(text, str):
        return text

    s = (
        text
        .replace("\ufeff", "")
        .replace("ï»¿", "")
    )

    # Intenta corregir cadenas tipo niÃ±os, leÃ±a, carbÃ³n.
    for _ in range(2):
        try:
            new_s = (
                s.encode("latin1")
                .decode("utf-8")
            )
            s = new_s
        except Exception:
            break

    s = (
        s
        .replace("\ufeff", "")
        .replace("ï»¿", "")
    )

    return s


def clean_dataframe_text(dataframe):
    """
    Limpia nombres de columnas y valores tipo texto.
    """

    dataframe = dataframe.copy()

    dataframe.columns = [
        fix_mojibake(str(column))
        for column in dataframe.columns
    ]

    for column in dataframe.select_dtypes(
        include=["object"]
    ).columns:
        dataframe[column] = dataframe[column].map(
            fix_mojibake
        )

    return dataframe


def load_json_safely(path):
    """
    Lee JSON evitando problemas de BOM.
    """

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:
        return json.load(file)


def save_json_safely(content, path):
    """
    Guarda JSON preservando caracteres especiales.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=4
        )


def clean_json_content(content):
    """
    Limpia strings dentro de listas o diccionarios JSON.
    """

    if isinstance(content, list):
        return [
            (
                fix_mojibake(value)
                if isinstance(value, str)
                else clean_json_content(value)
                if isinstance(value, (list, dict))
                else value
            )
            for value in content
        ]

    if isinstance(content, dict):
        return {
            key: (
                fix_mojibake(value)
                if isinstance(value, str)
                else clean_json_content(value)
                if isinstance(value, (list, dict))
                else value
            )
            for key, value in content.items()
        }

    return content


def prepare_clean_artifact_dir(
    original_artifact_dir,
    overwrite=True
):
    """
    Crea una copia de la carpeta de artefactos con nombres de variables
    corregidos para las figuras.

    IMPORTANTE:
    - no modifica la carpeta original;
    - no modifica los valores numéricos;
    - no modifica shap_values_3d.npy;
    - no modifica el modelo.
    """

    if not os.path.exists(
        original_artifact_dir
    ):
        raise FileNotFoundError(
            "No se encontró la carpeta original de artefactos:\n"
            f"{original_artifact_dir}"
        )

    clean_artifact_dir = (
        original_artifact_dir
        + "_clean_labels"
    )

    if (
        os.path.exists(clean_artifact_dir)
        and overwrite
    ):
        shutil.rmtree(
            clean_artifact_dir
        )

    if not os.path.exists(
        clean_artifact_dir
    ):
        shutil.copytree(
            original_artifact_dir,
            clean_artifact_dir
        )

    # --------------------------------------------------------
    # JSON donde pueden aparecer etiquetas
    # --------------------------------------------------------

    json_files = [
        "feature_names.json",
        "class_names.json",
        "metadata.json",
    ]

    for filename in json_files:
        path = os.path.join(
            clean_artifact_dir,
            filename
        )

        if os.path.exists(path):
            try:
                content = load_json_safely(
                    path
                )

                content = clean_json_content(
                    content
                )

                save_json_safely(
                    content,
                    path
                )

            except Exception as error:
                print(
                    f"Advertencia: no se pudo limpiar "
                    f"{filename}: {error}"
                )

    # --------------------------------------------------------
    # CSV donde pueden aparecer columnas o nombres de variables
    # --------------------------------------------------------

    csv_files = [
        "feature_names.csv",
        "X_train_final.csv",
        "X_test_final.csv",
        "y_train_final.csv",
        "y_test_final.csv",
        "glcd_train_final_with_target.csv",
        "glcd_test_final_with_target.csv",
        "glcd_full_final_with_target.csv",
        "test_probabilities.csv",
        "test_predictions.csv",
        "classification_report_test.csv",
        "confusion_matrix_test.csv",
        "test_original_indices.csv",
        "train_original_indices.csv",
        "selected_original_features.csv",
        "reporte_redundancia.csv",
        "reporte_tipos_y_scores.csv",
    ]

    for filename in csv_files:
        path = os.path.join(
            clean_artifact_dir,
            filename
        )

        if os.path.exists(path):
            try:
                dataframe = pd.read_csv(
                    path,
                    encoding="utf-8-sig"
                )

                dataframe = clean_dataframe_text(
                    dataframe
                )

                dataframe.to_csv(
                    path,
                    index=False,
                    encoding="utf-8-sig"
                )

            except Exception as error:
                print(
                    f"Advertencia: no se pudo limpiar "
                    f"{filename}: {error}"
                )

    return clean_artifact_dir


# ============================================================
# PREPARAR CARPETA DE ARTEFACTOS
# ============================================================

if USE_CLEAN_LABELS:

    print(
        "\nPreparando carpeta de artefactos "
        "con etiquetas limpias..."
    )

    ARTIFACT_DIR = (
        prepare_clean_artifact_dir(
            ORIGINAL_ARTIFACT_DIR,
            overwrite=OVERWRITE_CLEAN_ARTIFACTS
        )
    )

    print("Carpeta limpia usada:")
    print(
        os.path.abspath(
            ARTIFACT_DIR
        )
    )

else:

    ARTIFACT_DIR = ORIGINAL_ARTIFACT_DIR


# ============================================================
# VERIFICAR CARPETA DE ARTEFACTOS
# ============================================================

print("\nDirectorio actual de ejecución:")
print(os.getcwd())

print(
    "\nRuta absoluta de la carpeta "
    "de artefactos:"
)
print(
    os.path.abspath(
        ARTIFACT_DIR
    )
)

if not os.path.exists(
    ARTIFACT_DIR
):
    raise FileNotFoundError(
        f"No se encontró la carpeta: {ARTIFACT_DIR}\n"
        "Copia la carpeta "
        "'artifacts/child_occupation' "
        "dentro de la raíz del repositorio, "
        "o reemplaza ORIGINAL_ARTIFACT_DIR por la ruta "
        "absoluta correcta."
    )


# ============================================================
# VERIFICAR ARCHIVOS PRINCIPALES REQUERIDOS
# ============================================================

required_files = [
    "X_test_final.csv",
    "y_test_final.csv",
    "feature_names.csv",
    "class_names.json",
    "shap_values_3d.npy"
]

# El modelo es necesario solo para recalcular las métricas directamente.
# Si no carga, la clase intentará usar test_predictions/test_probabilities.
model_file = (
    "random_forest_optimizado.joblib"
)

missing_files = [
    filename
    for filename in required_files
    if not os.path.exists(
        os.path.join(
            ARTIFACT_DIR,
            filename
        )
    )
]

if missing_files:
    raise FileNotFoundError(
        "Faltan archivos requeridos para ejecutar SHAP-GLCD:\n"
        + "\n".join(
            f" - {filename}"
            for filename in missing_files
        )
    )

print(
    "\nArchivos requeridos encontrados correctamente:"
)

for filename in required_files:
    print(f" - {filename}")

if os.path.exists(
    os.path.join(
        ARTIFACT_DIR,
        model_file
    )
):
    print(f" - {model_file}")
else:
    print(
        f"Advertencia: no se encontró {model_file}. "
        "GLCD puede continuar con SHAP; para métricas predictivas "
        "se intentarán usar los archivos de predicciones guardados."
    )


# ============================================================
# VERIFICACIÓN BÁSICA DE ARTEFACTOS
# ============================================================

def load_feature_names_from_artifacts(
    artifact_dir
):
    json_path = os.path.join(
        artifact_dir,
        "feature_names.json"
    )

    csv_path = os.path.join(
        artifact_dir,
        "feature_names.csv"
    )

    if os.path.exists(
        json_path
    ):
        content = load_json_safely(
            json_path
        )

        if isinstance(
            content,
            list
        ):
            return [
                fix_mojibake(
                    str(value)
                )
                for value in content
            ]

        if (
            isinstance(content, dict)
            and "feature_names" in content
        ):
            return [
                fix_mojibake(
                    str(value)
                )
                for value in content[
                    "feature_names"
                ]
            ]

        if isinstance(
            content,
            dict
        ):
            return [
                fix_mojibake(
                    str(value)
                )
                for value in content.values()
            ]

    if os.path.exists(
        csv_path
    ):
        df_features = pd.read_csv(
            csv_path,
            encoding="utf-8-sig"
        )

        df_features = (
            clean_dataframe_text(
                df_features
            )
        )

        if "feature" in df_features.columns:
            return (
                df_features["feature"]
                .astype(str)
                .tolist()
            )

        return (
            df_features.iloc[:, 0]
            .astype(str)
            .tolist()
        )

    raise FileNotFoundError(
        "No se encontró feature_names.json "
        "ni feature_names.csv en la carpeta:\n"
        f"{artifact_dir}"
    )


def load_class_names_from_artifacts(
    artifact_dir
):
    json_path = os.path.join(
        artifact_dir,
        "class_names.json"
    )

    if not os.path.exists(
        json_path
    ):
        print(
            "\nAdvertencia: no se encontró class_names.json."
        )
        return None

    content = load_json_safely(
        json_path
    )

    return {
        int(k): fix_mojibake(v)
        for k, v in content.items()
    }


feature_names_check = (
    load_feature_names_from_artifacts(
        ARTIFACT_DIR
    )
)

class_names_check = (
    load_class_names_from_artifacts(
        ARTIFACT_DIR
    )
)

print(
    "\nNúmero de características "
    "cargadas desde artefactos:"
)
print(
    len(
        feature_names_check
    )
)

if class_names_check is not None:

    print(
        "\nClases cargadas desde artefactos:"
    )
    print(
        class_names_check
    )

    if class_names_check != EXPECTED_CLASSES:
        raise ValueError(
            "Las clases cargadas desde artefactos no coinciden "
            "con las esperadas.\n"
            f"Esperadas: {EXPECTED_CLASSES}\n"
            f"Encontradas: {class_names_check}"
        )

print(
    "\nVerificación básica correcta: "
    "se encontraron los artefactos principales."
)


# ============================================================
# CREAR CARPETA DE RESULTADOS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("\nConfiguración del análisis:")
print("Dataset           : menores5to18_4class")
print(f"Clase j           : {J_SEL} - {EXPECTED_CLASSES[J_SEL]}")
print(f"Clase r           : {R_SEL} - {EXPECTED_CLASSES[R_SEL]}")
print(f"Top features      : {TOP}")
print(f"Resultados en     : {OUTPUT_DIR}")


# ============================================================
# INSTANCIAR SHAP-GLCD DESDE ARTEFACTOS
# ============================================================

gexp = GLCDExplainerFromArtifacts(
    artifact_dir=ARTIFACT_DIR,
    j_sel=J_SEL,
    r_sel=R_SEL,
    top=TOP,
    random_state=RANDOM_STATE,
    output_dir=OUTPUT_DIR,
    load_model=True,
    figure_font_scale=FIGURE_FONT_SCALE
)


# ============================================================
# PIPELINE PRINCIPAL SHAP-GLCD
# ============================================================
# La clase corregida:
# - selecciona por etiqueta verdadera;
# - balancea las clases;
# - conserva aciertos y errores;
# - calcula ΔS con la misma muestra balanceada;
# - usa esa muestra en Mean/Median/Signed/Conditional/Weighted,
#   KDE pairwise, Cohen d, KS y thresholds;
# - mantiene las métricas predictivas sobre todo el test.
# ============================================================

gexp.run_glcd(
    make_plots=True,
    show=True,
    dpi=600
)


# ============================================================
# FIGURA B: DOMINANCE PLOT / 100% STACKED WEIGHTED
# ============================================================

figB_name = (
    f"figB_trabajo_infantil_RF_"
    f"menores5to18_4class_"
    f"weighted_100stack_{PAIR_TAG}.png"
)

gexp.plot_figura_B_100stack_weighted(
    dpi=600,
    fname=figB_name,
    show=True
)


# ============================================================
# EXTRA ANALYSIS PARA LA TOP-1 FEATURE
# ============================================================

top_feat = (
    gexp.rank.iloc[0][
        "Feature"
    ]
)

print(
    f"\n=== Extra conditional mean analysis "
    f"for top-1 feature: {top_feat} ==="
)

gexp.extra_feature_analysis(
    top_feat
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n============================================================")
print(" PROCESO SHAP-GLCD FINALIZADO")
print("============================================================")
print("Experimento             : menores5to18_4class")
print(f"Artefactos usados       : {os.path.abspath(ARTIFACT_DIR)}")
print(f"Par analizado           : {J_SEL} vs {R_SEL}")
print(f"Clase j                 : {gexp.class_mapping[J_SEL]}")
print(f"Clase r                 : {gexp.class_mapping[R_SEL]}")
print(f"Top features            : {TOP}")
print(f"Escala de fuente        : {FIGURE_FONT_SCALE}")

if hasattr(
    gexp,
    "N_pair"
):
    print(
        f"N pairwise balanceado   : {gexp.N_pair}"
    )

print("Métricas predictivas    : test completo")
print(f"Resultados en           : {OUTPUT_DIR}")
print("============================================================\n")


