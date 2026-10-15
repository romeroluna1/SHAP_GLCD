# ============================================================
# stability_ag_news.py
# ============================================================
# Calcula métricas de estabilidad y consistencia para SHAP-LCD
# en texto usando el modelo AG News ya entrenado.
#
# Reporte en consola:
#
# Pair | N | Bal. | S_bar ± sS | Cimp_bar ± sC |
# Cov_imp / PredKeep_imp
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
    SHAPTextLCDMetrics,
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

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "ag_news"
    / "model_agnews_distilbert_base_uncased"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "ag_news"
)

METRICS_DIR = RESULTS_DIR / "metrics_stability_consistency_text"

os.makedirs(
    METRICS_DIR,
    exist_ok=True,
)

MAX_LENGTH = 128


# ============================================================
# 0.1) Parámetros metodológicos
# ============================================================

# ------------------------------------------------------------
# Prueba rápida
# ------------------------------------------------------------

M_PERTURBATIONS = 40

# Debe ser un número par porque la clase actualizada
# selecciona la misma cantidad de instancias de cada clase.
MAX_INSTANCES_PER_PAIR = 50


# ------------------------------------------------------------
# Corrida final para el artículo
# ------------------------------------------------------------
#
# Cuando la prueba rápida termine correctamente,
# comenta los valores anteriores y activa:
#
# M_PERTURBATIONS = 50
# MAX_INSTANCES_PER_PAIR = 20


# Umbral de similitud coseno utilizado
# en la consistencia basada en implicación.
TAU = 0.95


# Proporción de tokens que se enmascaran.
# 0.05 equivale aproximadamente al 5 %.
PERTURBATION_RATE = 0.05


# "pred":
# selecciona las instancias según la predicción del modelo.
#
# "true":
# selecciona las instancias según la etiqueta verdadera.
FILTER_MODE = "true"


# Par utilizado en el artículo.
PAIRS_TO_EVALUATE = [
    ("Business", "Sci/Tech"),
]


# Para evaluar posteriormente todos los pares:
#
# PAIRS_TO_EVALUATE = [
#     ("World", "Sports"),
#     ("World", "Business"),
#     ("World", "Sci/Tech"),
#     ("Sports", "Business"),
#     ("Sports", "Sci/Tech"),
#     ("Business", "Sci/Tech"),
# ]


# Verificación para la selección balanceada.
if (
    MAX_INSTANCES_PER_PAIR is not None
    and MAX_INSTANCES_PER_PAIR % 2 != 0
):
    raise ValueError(
        "MAX_INSTANCES_PER_PAIR debe ser un número par "
        "para construir una muestra balanceada."
    )


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

print(
    f"Dispositivo usado: {device}"
)


# ============================================================
# 2) Nombres de las clases
# ============================================================

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


# ============================================================
# 3) Funciones auxiliares
# ============================================================

def build_text_column(example):
    """
    Construye la columna text para AG News.
    """

    if (
        "title" in example
        and "description" in example
    ):
        title = str(
            example["title"]
        )

        description = str(
            example["description"]
        )

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
    Convierte las etiquetas 1-4 a 0-3
    cuando sea necesario.
    """

    train_labels = np.asarray(
        dataset["train"]["label"]
    )

    minimum_label = int(
        train_labels.min()
    )

    maximum_label = int(
        train_labels.max()
    )

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


def print_metrics_summary_console(
    summary_df,
    instances_df,
):
    """
    Muestra en consola el resumen general
    y los resultados por instancia.
    """

    print(
        "\n\n"
        "================================================================"
        "================"
    )

    print(
        " Robustness of SHAP-LCD explanations under perturbations"
    )

    print(
        "================================================================"
        "================"
    )

    if not summary_df.empty:

        tau_value = float(
            summary_df["tau"].iloc[0]
        )

        perturbation_value = float(
            summary_df[
                "perturbation_rate"
            ].iloc[0]
        )

        m_value = int(
            summary_df[
                "M_perturbations"
            ].iloc[0]
        )

        print(
            f"Text experiment with tau = {tau_value:.2f}, "
            f"perturbation_rate = {perturbation_value:.2f}, "
            f"M = {m_value}"
        )

        print("")

    header = (
        f"{'Pair (Cj, Cr)':<30}"
        f"{'N':>6}"
        f"{'Bal.':>10}"
        f"{'S_bar ± sS':>22}"
        f"{'Cimp_bar ± sC':>22}"
        f"{'Cov_imp / PredKeep_imp':>28}"
    )

    print(header)
    print("-" * len(header))

    for _, row in summary_df.iterrows():

        pair_name = row.get(
            "pair",
            (
                f"{row['class_j']} "
                f"vs {row['class_r']}"
            ),
        )

        stability_text = (
            f"{row['stability_mean']:.3f} "
            f"± {row['stability_std']:.3f}"
        )

        consistency_text = (
            f"{row['consistency_mean']:.3f} "
            f"± {row['consistency_std']:.3f}"
        )

        coverage_prediction_text = (
            f"{row['coverage_imp_mean']:.3f} / "
            f"{row['prediction_keep_imp_mean']:.3f}"
        )

        print(
            f"{pair_name:<30}"
            f"{int(row['N_instances']):>6}"
            f"{row['Bal.']:>10}"
            f"{stability_text:>22}"
            f"{consistency_text:>22}"
            f"{coverage_prediction_text:>28}"
        )

    print("-" * len(header))

    print(
        "\nDetalles por instancia:"
    )

    print("-" * 130)

    columns_to_show = [
        "test_index",
        "true_label_name",
        "base_prediction_name",
        "class_j",
        "class_r",
        "stability",
        "consistency",
        "coverage_imp",
        "prediction_keep_imp",
        "cosine_mean",
        "explanation_dimension",
        "sampling_seed",
        "instance_seed",
    ]

    available_columns = [
        column
        for column in columns_to_show
        if column in instances_df.columns
    ]

    print(
        instances_df[
            available_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )

    print(
        "\n"
        "================================================================"
        "================"
    )

    print(
        " End of SHAP-LCD text robustness metrics"
    )

    print(
        "================================================================"
        "================\n"
    )


# ============================================================
# 4) Mostrar configuración
# ============================================================

print(
    "\n"
    "=================================================="
)

print(
    "Configuración de métricas SHAP-LCD"
)

print(
    "=================================================="
)

print(
    f"Semilla: {SEED}"
)

print(
    f"Pares evaluados: {PAIRS_TO_EVALUATE}"
)

print(
    f"FILTER_MODE: {FILTER_MODE}"
)

print(
    f"M_PERTURBATIONS: {M_PERTURBATIONS}"
)

print(
    "MAX_INSTANCES_PER_PAIR: "
    f"{MAX_INSTANCES_PER_PAIR}"
)

print(
    "PERTURBATION_RATE: "
    f"{PERTURBATION_RATE}"
)

print(
    f"TAU: {TAU}"
)

print(
    "=================================================="
)


# ============================================================
# 5) Verificar que el modelo exista
# ============================================================

if not os.path.exists(
    MODEL_DIR
):
    raise FileNotFoundError(
        "No se encontró el modelo en:\n"
        f"{os.path.abspath(MODEL_DIR)}\n\n"
        "Primero debes entrenar el modelo final con "
        "Main_AGNews_SHAP_LCD.py."
    )


# ============================================================
# 6) Cargar modelo y tokenizador
# ============================================================

print(
    "\nCargando modelo desde:"
)

print(
    os.path.abspath(
        MODEL_DIR
    )
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_DIR
    )
)

model.to(
    device
)

model.eval()


# ============================================================
# 7) Cargar AG News
# ============================================================

print(
    "\nCargando AG News..."
)

dataset = load_dataset(
    "sh0416/ag_news"
)

dataset = dataset.map(
    build_text_column
)

dataset = normalize_agnews_labels(
    dataset
)

test_dataset = dataset[
    "test"
]

texts_test = list(
    test_dataset["text"]
)

y_test = np.asarray(
    test_dataset["label"]
)

print(
    "\nNúmero de textos en test: "
    f"{len(texts_test)}"
)


# ============================================================
# 8) Inicializar SHAP-LCD
# ============================================================

print(
    "\nInicializando SHAP-LCD para texto..."
)

lcd_explainer = SHAPTextExplainerLCD(
    model=model,
    tokenizer=tokenizer,
    texts_eval=texts_test,
    y_eval=y_test,
    class_names=CLASS_NAMES,
    max_length=MAX_LENGTH,
    batch_size=8,
    device=device,
    random_state=SEED,
)


# ============================================================
# 9) Inicializar la clase de métricas
# ============================================================

metrics_evaluator = SHAPTextLCDMetrics(
    lcd_explainer=lcd_explainer
)


# ============================================================
# 10) Calcular estabilidad y consistencia
# ============================================================

print(
    "\nCalculando métricas SHAP-LCD para texto..."
)

summary_df, instances_df = (
    metrics_evaluator.evaluate_multiple_pairs(
        texts=texts_test,
        y_true=y_test,
        pairs=PAIRS_TO_EVALUATE,
        filter_mode=FILTER_MODE,
        M=M_PERTURBATIONS,
        perturbation_rate=(
            PERTURBATION_RATE
        ),
        tau=TAU,
        max_instances_per_pair=(
            MAX_INSTANCES_PER_PAIR
        ),
        seed=SEED,
        output_dir=METRICS_DIR,
    )
)


# ============================================================
# 11) Mostrar el resumen en consola
# ============================================================

print_metrics_summary_console(
    summary_df=summary_df,
    instances_df=instances_df,
)


# ============================================================
# 12) Rutas de los resultados
# ============================================================

summary_path = os.path.join(
    METRICS_DIR,
    "metrics_summary_all_pairs.csv",
)

instances_path = os.path.join(
    METRICS_DIR,
    "metrics_instances_all_pairs.csv",
)

table_path = os.path.join(
    METRICS_DIR,
    "metrics_table_format_all_pairs.csv",
)

print(
    "\nArchivos guardados:"
)

print(
    "\nResumen completo:"
)

print(
    os.path.abspath(
        summary_path
    )
)

print(
    "\nValores por instancia:"
)

print(
    os.path.abspath(
        instances_path
    )
)

print(
    "\nTabla con formato para el artículo:"
)

print(
    os.path.abspath(
        table_path
    )
)

print(
    "\nProceso finalizado."
)
