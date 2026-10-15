# ============================================================
# Main_AGNews_SHAP_GLCD_TwoColumns_Ranking.py
# ============================================================
# SHAP-GLCD global para texto usando AG News.
#
# Este main NO entrena el modelo.
# Carga el modelo ya entrenado, calcula o recupera SHAP-GLCD
# y genera exclusivamente las figuras ranked two-column
# utilizadas para el artículo.
#
# Salidas:
#
#   1) Archivos globales:
#      - deltaS_records_<Cj>_vs_<Cr>.csv
#      - global_token_summary_unfiltered_<Cj>_vs_<Cr>.csv
#      - global_token_summary_<Cj>_vs_<Cr>.csv
#
#   2) Figuras ranked two-column para el artículo:
#      - 26_article_two_column_ranked_mean_abs_<Cj>_vs_<Cr>.png/pdf
#      - 27_article_two_column_ranked_median_abs_<Cj>_vs_<Cr>.png/pdf
#      - 28_article_two_column_ranked_signed_mean_<Cj>_vs_<Cr>.png/pdf
#      - 29_article_two_column_ranked_cdm_<Cj>_vs_<Cr>.png/pdf
#      - 30_article_two_column_ranked_dp_<Cj>_vs_<Cr>.png/pdf
#
# NO genera:
#      - Figuras originales 01-05.
#      - Figuras antiguas tipo bloque.
#      - Figuras two-column direccionales antiguas.
#
# Convención de las figuras ranked:
#   - Las columnas NO separan las clases.
#   - Los tokens se ordenan por importancia.
#   - La columna izquierda contiene la primera mitad del ranking.
#   - La columna derecha continúa con la segunda mitad.
#
# Convención de color:
#   - Mean |ΔS| y Median |ΔS|: color neutral.
#   - Signed Mean, CdM y DP:
#       Magenta → favorece Cj.
#       Verde   → favorece Cr.
# ============================================================

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.text import (
    SHAPTextExplainerLCD,
    SHAPGlobalTextGLCD,
    SHAPGlobalTextTwoColumnRankedArticleVisualizer,
)


# ============================================================
# 0) Configuración general
# ============================================================

SEED = 42

MODEL_NAME = "distilbert-base-uncased"
MODEL_SAFE_NAME = (
    MODEL_NAME
    .replace("/", "_")
    .replace("-", "_")
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "ag_news"
MODEL_DIR = ARTIFACT_DIR / f"model_agnews_{MODEL_SAFE_NAME}"
SHAP_CACHE_DIR = ARTIFACT_DIR / "shap_cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "ag_news"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

os.makedirs(
    SHAP_CACHE_DIR,
    exist_ok=True,
)

MAX_LENGTH = 128

# Número máximo de textos usados para el análisis global.
# Al balancear el par:
#   150 textos de Cj + 150 textos de Cr = 300 textos.
MAX_EVAL_GLCD = 300

# Tokens que se muestran en la consola.
TOP_K = 20

# Frecuencia mínima total para incluir un token
# en el resumen global inicial.
MIN_TOKEN_FREQUENCY = 5

# Frecuencia documental mínima para conservar un token
# en el resumen final y en las figuras del artículo.
# El token debe aparecer en al menos 3 documentos diferentes.
MIN_DOCUMENT_FREQUENCY = 3

# Selección de textos:
#   "pred" → según la predicción del modelo.
#   "true" → según la etiqueta verdadera.
FILTER_MODE = "true"

# Balancear las dos clases del par.
BALANCE_PAIR_SAMPLE = True


# ============================================================
# 0.1) Pares que se evaluarán
# ============================================================

# Par utilizado actualmente en el artículo.
PAIRS_TO_EVALUATE = [
    ("Business", "Sci/Tech"),
]


# Para analizar posteriormente otros pares,
# se pueden agregar aquí:
#
# PAIRS_TO_EVALUATE = [
#     ("World", "Sports"),
#     ("World", "Business"),
#     ("World", "Sci/Tech"),
#     ("Sports", "Business"),
#     ("Sports", "Sci/Tech"),
#     ("Business", "Sci/Tech"),
# ]


# ============================================================
# 0.2) Configuración de las figuras para el artículo
# ============================================================

# Número total de tokens mostrados:
#   10 en la columna izquierda.
#   10 en la columna derecha.
RANKED_TWO_COLUMN_TOP_K = 20

RANKED_TWO_COLUMN_MIN_FONT = 14
RANKED_TWO_COLUMN_MAX_FONT = 46
RANKED_TWO_COLUMN_TITLE_FONT = 22
RANKED_TWO_COLUMN_LEGEND_FONT = 15

RANKED_TWO_COLUMN_DPI = 600

RANKED_TWO_COLUMN_FIG_WIDTH = 10.5
RANKED_TWO_COLUMN_FIG_HEIGHT = 6.2

# Para el artículo no se muestran números
# junto a los tokens.
RANKED_TWO_COLUMN_SHOW_RANK_NUMBERS = False


# Colores finales.
NEUTRAL_COLOR = "#31688E"
CJ_COLOR = "#C2185B"
CR_COLOR = "#00897B"


# ============================================================
# 1) Semillas y dispositivo
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Dispositivo usado: {device}")


# ============================================================
# 2) Clases de AG News
# ============================================================

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

LABEL_TO_ID = {
    class_name: class_id
    for class_id, class_name in CLASS_NAMES.items()
}


# ============================================================
# 3) Funciones auxiliares
# ============================================================

def safe_filename(text):
    """
    Convierte un texto en una cadena segura
    para utilizarla como nombre de archivo.
    """

    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )


def build_text_column(example):
    """
    Construye la columna text.

    Para sh0416/ag_news se espera:
        title + description
    """

    if (
        "title" in example
        and "description" in example
    ):
        title = str(example["title"])
        description = str(example["description"])

        example["text"] = (
            title
            + ". "
            + description
        )

    elif "text" in example:
        example["text"] = str(
            example["text"]
        )

    else:
        raise ValueError(
            "No se encontró una estructura compatible "
            "para construir la columna text."
        )

    return example


def normalize_agnews_labels(dataset):
    """
    Normaliza las etiquetas de AG News a 0-3
    cuando el conjunto las presenta como 1-4.
    """

    labels_train = np.array(
        dataset["train"]["label"]
    )

    minimum_label = labels_train.min()
    maximum_label = labels_train.max()

    if (
        minimum_label == 1
        and maximum_label == 4
    ):
        print(
            "Normalizando etiquetas de 1-4 a 0-3..."
        )

        def normalize_label(example):
            example["label"] = (
                int(example["label"]) - 1
            )
            return example

        dataset = dataset.map(
            normalize_label
        )

    elif (
        minimum_label == 0
        and maximum_label == 3
    ):
        print(
            "Las etiquetas ya están en formato 0-3."
        )

    else:
        raise ValueError(
            "Rango inesperado de etiquetas: "
            f"min={minimum_label}, "
            f"max={maximum_label}"
        )

    return dataset


def select_pair_sample(
    texts,
    y_true,
    y_pred,
    class_j_name,
    class_r_name,
    filter_mode="pred",
    max_eval=100,
    balance=True,
    seed=42,
):
    """
    Selecciona una muestra de textos pertenecientes
    al par de clases solicitado.

    La misma semilla se utiliza para todos los pares.
    Por tanto, el resultado no depende del orden
    de PAIRS_TO_EVALUATE.
    """

    rng = np.random.default_rng(seed)

    class_j_idx = LABEL_TO_ID[class_j_name]
    class_r_idx = LABEL_TO_ID[class_r_name]

    if filter_mode == "pred":
        labels_for_filter = y_pred

    elif filter_mode == "true":
        labels_for_filter = y_true

    else:
        raise ValueError(
            "filter_mode debe ser 'pred' o 'true'."
        )

    idx_j = np.where(
        labels_for_filter == class_j_idx
    )[0]

    idx_r = np.where(
        labels_for_filter == class_r_idx
    )[0]

    if (
        len(idx_j) == 0
        or len(idx_r) == 0
    ):
        raise ValueError(
            "No hay suficientes textos para el par "
            f"{class_j_name} vs {class_r_name} "
            f"con filter_mode={filter_mode}."
        )

    if balance:
        n_each = min(
            len(idx_j),
            len(idx_r),
            max_eval // 2,
        )

        selected_j = rng.choice(
            idx_j,
            size=n_each,
            replace=False,
        )

        selected_r = rng.choice(
            idx_r,
            size=n_each,
            replace=False,
        )

        selected_idx = np.concatenate(
            [
                selected_j,
                selected_r,
            ]
        )

    else:
        candidate_idx = np.concatenate(
            [
                idx_j,
                idx_r,
            ]
        )

        n_total = min(
            len(candidate_idx),
            max_eval,
        )

        selected_idx = rng.choice(
            candidate_idx,
            size=n_total,
            replace=False,
        )

    selected_idx = np.sort(
        selected_idx
    )

    selected_texts = [
        texts[index]
        for index in selected_idx
    ]

    selected_y_true = y_true[
        selected_idx
    ]

    selected_y_pred = y_pred[
        selected_idx
    ]

    print("\nMuestra seleccionada para el par:")
    print(
        f"Par: {class_j_name} vs {class_r_name}"
    )
    print(
        f"Modo de selección: {filter_mode}"
    )
    print(
        f"Semilla de muestreo: {seed}"
    )
    print(
        f"Número de textos seleccionados: "
        f"{len(selected_idx)}"
    )

    print("\nDistribución por etiquetas verdaderas:")

    unique_true, counts_true = np.unique(
        selected_y_true,
        return_counts=True,
    )

    for label, count in zip(
        unique_true,
        counts_true,
    ):
        print(
            f"  {CLASS_NAMES[int(label)]}: "
            f"{count}"
        )

    print("\nDistribución por etiquetas predichas:")

    unique_pred, counts_pred = np.unique(
        selected_y_pred,
        return_counts=True,
    )

    for label, count in zip(
        unique_pred,
        counts_pred,
    ):
        print(
            f"  {CLASS_NAMES[int(label)]}: "
            f"{count}"
        )

    return (
        selected_texts,
        selected_y_true,
        selected_y_pred,
        selected_idx,
    )


def save_selected_metadata(
    selected_texts,
    selected_y_true,
    selected_y_pred,
    selected_indices,
    output_path,
):
    """
    Guarda la información de los textos utilizados
    en el análisis global.
    """

    rows = []

    for local_idx, original_idx in enumerate(
        selected_indices
    ):
        true_label = int(
            selected_y_true[local_idx]
        )

        pred_label = int(
            selected_y_pred[local_idx]
        )

        rows.append(
            {
                "local_index": int(local_idx),
                "original_test_index": int(
                    original_idx
                ),
                "true_label": true_label,
                "true_label_name": CLASS_NAMES[
                    true_label
                ],
                "pred_label": pred_label,
                "pred_label_name": CLASS_NAMES[
                    pred_label
                ],
                "text": selected_texts[
                    local_idx
                ],
            }
        )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(
        "Metadatos de los textos guardados en:"
    )
    print(
        os.path.abspath(output_path)
    )


# ============================================================
# 4) Mostrar las rutas reales
# ============================================================

print("\nDirectorio actual de ejecución:")
print(os.getcwd())

print("\nModelo esperado en:")
print(os.path.abspath(MODEL_DIR))

print("\nResultados se guardarán en:")
print(os.path.abspath(RESULTS_DIR))


# ============================================================
# 5) Verificar la existencia del modelo
# ============================================================

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        "No se encontró el modelo en:\n"
        f"{os.path.abspath(MODEL_DIR)}\n\n"
        "Primero debes entrenarlo con "
        "Main_AGNews_SHAP_LCD.py."
    )


# ============================================================
# 6) Cargar el modelo y el tokenizador
# ============================================================

print(
    f"\nCargando modelo desde: "
    f"{os.path.abspath(MODEL_DIR)}"
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(MODEL_DIR)
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(MODEL_DIR)
)

model.to(device)
model.eval()


# ============================================================
# 7) Cargar AG News
# ============================================================

print("\nCargando AG News...")

dataset = load_dataset(
    "sh0416/ag_news"
)

dataset = dataset.map(
    build_text_column
)

dataset = normalize_agnews_labels(
    dataset
)

test_dataset = dataset["test"]

texts_test = list(
    test_dataset["text"]
)

y_test = np.array(
    test_dataset["label"]
)

print(
    "\nNúmero total de textos en test: "
    f"{len(texts_test)}"
)


# ============================================================
# 8) Obtener predicciones del modelo
# ============================================================

print(
    "\nObteniendo predicciones del modelo "
    "en el conjunto de prueba..."
)

tmp_explainer = SHAPTextExplainerLCD(
    model=model,
    tokenizer=tokenizer,
    texts_eval=texts_test,
    y_eval=y_test,
    class_names=CLASS_NAMES,
    max_length=MAX_LENGTH,
    batch_size=16,
    device=device,
    random_state=SEED,
)

y_pred = tmp_explainer.predict_labels(
    texts_test
)

print(
    "\nDistribución de predicciones en test:"
)

unique_pred, counts_pred = np.unique(
    y_pred,
    return_counts=True,
)

for label, count in zip(
    unique_pred,
    counts_pred,
):
    print(
        f"{CLASS_NAMES[int(label)]}: "
        f"{count}"
    )


# ============================================================
# 9) Ejecutar SHAP-GLCD global por pares
# ============================================================

for (
    class_j_name,
    class_r_name,
) in PAIRS_TO_EVALUATE:

    print(
        "\n\n"
        "============================================================"
    )

    print(
        "Iniciando SHAP-GLCD global para el par: "
        f"{class_j_name} vs {class_r_name}"
    )

    print(
        "============================================================"
    )

    (
        selected_texts,
        selected_y_true,
        selected_y_pred,
        selected_indices,
    ) = select_pair_sample(
        texts=texts_test,
        y_true=y_test,
        y_pred=y_pred,
        class_j_name=class_j_name,
        class_r_name=class_r_name,
        filter_mode=FILTER_MODE,
        max_eval=MAX_EVAL_GLCD,
        balance=BALANCE_PAIR_SAMPLE,

        # Todos los pares utilizan la misma semilla.
        seed=SEED,
    )

    safe_j = safe_filename(
        class_j_name
    )

    safe_r = safe_filename(
        class_r_name
    )

    pair_results_dir = os.path.join(
        RESULTS_DIR,
        f"pair_{safe_j}_vs_{safe_r}",
    )

    os.makedirs(
        pair_results_dir,
        exist_ok=True,
    )


    # ========================================================
    # 9.1) Guardar los metadatos de la muestra
    # ========================================================

    metadata_path = os.path.join(
        pair_results_dir,
        "selected_text_metadata.csv",
    )

    save_selected_metadata(
        selected_texts=selected_texts,
        selected_y_true=selected_y_true,
        selected_y_pred=selected_y_pred,
        selected_indices=selected_indices,
        output_path=metadata_path,
    )

    selected_indices_path = os.path.join(
        pair_results_dir,
        "selected_test_indices.npy",
    )

    np.save(
        selected_indices_path,
        selected_indices,
    )

    print(
        "\nÍndices seleccionados guardados en:"
    )

    print(
        os.path.abspath(
            selected_indices_path
        )
    )


    # ========================================================
    # 9.2) Calcular o cargar los valores SHAP
    # ========================================================

    shap_pickle_path = os.path.join(
        SHAP_CACHE_DIR,
        (
            f"shap_values_glcd_"
            f"{safe_j}_vs_{safe_r}"
            f"_N{len(selected_texts)}"
            f"_{FILTER_MODE}.pickle"
        ),
    )

    pair_explainer = SHAPTextExplainerLCD(
        model=model,
        tokenizer=tokenizer,
        texts_eval=selected_texts,
        y_eval=selected_y_true,
        class_names=CLASS_NAMES,
        max_length=MAX_LENGTH,
        batch_size=8,
        device=device,
        random_state=SEED,
    )

    pair_explainer.original_indices = (
        selected_indices
    )

    if os.path.exists(
        shap_pickle_path
    ):
        print(
            "\nCargando valores SHAP desde:"
        )

        print(
            os.path.abspath(
                shap_pickle_path
            )
        )

        pair_explainer.load_shap_values(
            shap_pickle_path
        )

    else:
        print(
            "\nNo se encontró el archivo SHAP "
            "para esta muestra."
        )

        print(
            "Calculando valores SHAP..."
        )

        pair_explainer.calculate_shap_values(
            max_eval=len(selected_texts)
        )

        pair_explainer.save_shap_values(
            shap_pickle_path
        )

        print(
            "\nValores SHAP guardados en:"
        )

        print(
            os.path.abspath(
                shap_pickle_path
            )
        )


    # ========================================================
    # 9.3) Agregación global SHAP-GLCD
    # ========================================================

    glcd_global = SHAPGlobalTextGLCD(
        lcd_explainer=pair_explainer,
        output_dir=RESULTS_DIR,
        min_token_frequency=(
            MIN_TOKEN_FREQUENCY
        ),
        normalize_token_case=True,
    )

    class_j_idx = LABEL_TO_ID[
        class_j_name
    ]

    class_r_idx = LABEL_TO_ID[
        class_r_name
    ]

    # IMPORTANTE:
    #
    # No se utiliza run_pair_analysis porque ese método
    # genera automáticamente las figuras antiguas 01-05.
    #
    # aggregate_pair_global calcula únicamente:
    #   - los registros ΔS;
    #   - el resumen global de tokens.
    #
    # Las únicas figuras se generarán posteriormente con
    # SHAPGlobalTextTwoColumnRankedArticleVisualizer.

    df_records, df_global_unfiltered = (
        glcd_global.aggregate_pair_global(
            class_j=class_j_idx,
            class_r=class_r_idx,
        )
    )

    if "document_frequency" not in df_global_unfiltered.columns:
        raise RuntimeError(
            "El resumen global no contiene la columna "
            "'document_frequency'."
        )

    # Segundo filtro metodológico:
    # además de la frecuencia total mínima aplicada por la clase,
    # el token debe aparecer en al menos MIN_DOCUMENT_FREQUENCY
    # documentos diferentes.
    df_global = df_global_unfiltered[
        df_global_unfiltered["document_frequency"]
        >= MIN_DOCUMENT_FREQUENCY
    ].copy()

    # Mantiene un orden determinista para la consola, los CSV
    # y las figuras finales.
    df_global = df_global.sort_values(
        by=[
            "mean_abs_deltaS",
            "document_frequency",
            "frequency",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    if df_records.empty:
        raise RuntimeError(
            "No se generaron registros deltaS."
        )

    if df_global.empty:
        raise RuntimeError(
            "El resumen global quedó vacío después de aplicar "
            "los filtros. Revise MIN_TOKEN_FREQUENCY y "
            "MIN_DOCUMENT_FREQUENCY."
        )


    # ========================================================
    # 9.4) Guardar los archivos globales
    # ========================================================

    records_path = os.path.join(
        pair_results_dir,
        (
            f"deltaS_records_"
            f"{safe_j}_vs_{safe_r}.csv"
        ),
    )

    global_summary_unfiltered_path = os.path.join(
        pair_results_dir,
        (
            f"global_token_summary_unfiltered_"
            f"{safe_j}_vs_{safe_r}.csv"
        ),
    )

    global_summary_path = os.path.join(
        pair_results_dir,
        (
            f"global_token_summary_"
            f"{safe_j}_vs_{safe_r}.csv"
        ),
    )

    df_records.to_csv(
        records_path,
        index=False,
        encoding="utf-8",
    )

    df_global_unfiltered.to_csv(
        global_summary_unfiltered_path,
        index=False,
        encoding="utf-8",
    )

    df_global.to_csv(
        global_summary_path,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nRegistros ΔS guardados en:"
    )

    print(
        os.path.abspath(
            records_path
        )
    )

    print(
        "\nResumen global sin filtro documental guardado en:"
    )

    print(
        os.path.abspath(
            global_summary_unfiltered_path
        )
    )

    print(
        "\nResumen global filtrado guardado en:"
    )

    print(
        os.path.abspath(
            global_summary_path
        )
    )

    print(
        "\nTokens antes/después del filtro documental: "
        f"{len(df_global_unfiltered)} -> {len(df_global)}"
    )

    print(
        "Criterio documental aplicado: "
        f"document_frequency >= {MIN_DOCUMENT_FREQUENCY}"
    )


    # ========================================================
    # 9.5) Mostrar los principales tokens en consola
    # ========================================================

    columns_to_show = [
        "token",
        "frequency",
        "document_frequency",
        "mean_abs_deltaS",
        "median_abs_deltaS",
        "signed_mean_deltaS",
        "positive_conditional_mean",
        "negative_conditional_mean",
        "cdm_value",
        "W_positive",
        "W_negative",
        "dominant_class",
        "signed_direction",
    ]

    # Evita errores si alguna versión de la clase
    # no contiene una de estas columnas.
    available_columns = [
        column
        for column in columns_to_show
        if column in df_global.columns
    ]

    print(
        "\nPrincipales tokens globales "
        "según Mean |ΔS|:"
    )

    print(
        df_global[
            available_columns
        ]
        .head(TOP_K)
        .to_string(index=False)
    )


    # ========================================================
    # 10) Generar exclusivamente las figuras del artículo
    # ========================================================

    print(
        "\nGenerando exclusivamente las figuras "
        "ranked two-column para el artículo..."
    )

    ranked_two_column_visualizer = (
        SHAPGlobalTextTwoColumnRankedArticleVisualizer(
            output_dir=pair_results_dir,
            token_col=None,
            min_font_size=(
                RANKED_TWO_COLUMN_MIN_FONT
            ),
            max_font_size=(
                RANKED_TWO_COLUMN_MAX_FONT
            ),
            title_font_size=(
                RANKED_TWO_COLUMN_TITLE_FONT
            ),
            legend_font_size=(
                RANKED_TWO_COLUMN_LEGEND_FONT
            ),
            dpi=(
                RANKED_TWO_COLUMN_DPI
            ),
            neutral_color=NEUTRAL_COLOR,
            cj_color=CJ_COLOR,
            cr_color=CR_COLOR,
            fig_width=(
                RANKED_TWO_COLUMN_FIG_WIDTH
            ),
            fig_height=(
                RANKED_TWO_COLUMN_FIG_HEIGHT
            ),
            show_rank_numbers=(
                RANKED_TWO_COLUMN_SHOW_RANK_NUMBERS
            ),
        )
    )

    ranked_two_column_visualizer.plot_all(
        df_global=df_global,
        class_j=class_j_name,
        class_r=class_r_name,
        top_k=RANKED_TWO_COLUMN_TOP_K,
        include_signed_mean=True,
        include_cdm=True,
        include_dp=True,
    )

    print(
        "\nFiguras del artículo guardadas en:"
    )

    print(
        os.path.abspath(
            pair_results_dir
        )
    )


# ============================================================
# 11) Finalización
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "Proceso SHAP-GLCD global para texto finalizado."
)

print(
    "Solo se generaron las figuras ranked "
    "two-column del artículo."
)

print(
    "No se generaron las figuras originales 01-05 "
    "ni las figuras antiguas tipo bloque."
)

print(
    "Resultados guardados en:"
)

print(
    os.path.abspath(
        RESULTS_DIR
    )
)

print(
    "============================================================"
)
