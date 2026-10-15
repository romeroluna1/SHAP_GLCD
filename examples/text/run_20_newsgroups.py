# ============================================================
# Main_20Newsgroups_SHAP_GLCD_Global.py
# ============================================================
# SHAP-GLCD global para texto usando 20 Newsgroups + RoBERTa-base.
#
# Este main NO entrena el modelo.
# Carga:
#   - model_20newsgroups_roberta_base
#   - results_20newsgroups_roberta_base/label_mapping.csv
#   - results_20newsgroups_roberta_base/test_split.csv
#   - results_20newsgroups_roberta_base/test_predictions.csv
#
# Flujo:
#   1) Selecciona un par de clases.
#   2) Extrae una muestra balanceada del test según predicción o
#      etiqueta verdadera.
#   3) Calcula o recupera SHAP para la muestra seleccionada.
#   4) Reconstruye de forma segura las palabras de RoBERTa.
#   5) Calcula los registros globales deltaS y el resumen por token.
#   6) Genera únicamente las figuras ranked two-column del artículo.
#
# Convención:
#   deltaS = SHAP(Cj) - SHAP(Cr)
#   deltaS > 0  -> favorece Cj
#   deltaS < 0  -> favorece Cr
#
# Salidas principales:
#   - selected_text_metadata.csv
#   - selected_test_indices.npy
#   - run_configuration.csv
#   - alignment_summary.csv
#   - deltaS_records_<Cj>_vs_<Cr>.csv
#   - global_token_summary_<Cj>_vs_<Cr>.csv
#   - 26_article_two_column_ranked_mean_abs_...
#   - 27_article_two_column_ranked_median_abs_...
#   - 28_article_two_column_ranked_signed_mean_...
#   - 29_article_two_column_ranked_cdm_...
#   - 30_article_two_column_ranked_dp_...
# ============================================================

import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.text import (
    SHAPTextExplainerLCD,
    SHAPGlobalTextGLCD20Newsgroups,
    SHAPGlobalTextTwoColumnRankedArticleVisualizer,
)


# ============================================================
# 0) Configuración general
# ============================================================

SEED = 42

DATASET_NAME = "20newsgroups"
MODEL_SAFE_NAME = "roberta_base"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "20_newsgroups"
MODEL_DIR = ARTIFACT_DIR / f"model_{DATASET_NAME}_{MODEL_SAFE_NAME}"
TRAIN_RESULTS_DIR = ARTIFACT_DIR / f"results_{DATASET_NAME}_{MODEL_SAFE_NAME}"
SHAP_CACHE_DIR = ARTIFACT_DIR / "shap_cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "20_newsgroups"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SHAP_CACHE_DIR, exist_ok=True)

# Debe coincidir con el valor usado durante el entrenamiento.
MAX_LENGTH = 256

# Corrida global ampliada: 200 textos en total, 100 por clase cuando
# FILTER_MODE="pred" y BALANCE_PAIR_SAMPLE=True.
# En CPU puede tardar alrededor de 1 a 2 horas, según el equipo.
MAX_EVAL_GLCD = 200

# Frecuencia mínima total y frecuencia documental mínima para conservar
# un token en el resumen global utilizado por las figuras.
MIN_TOKEN_FREQUENCY = 5
MIN_DOCUMENT_FREQUENCY = 3

# Antes de calcular SHAP, cada texto se recorta de forma explícita a la
# misma longitud máxima usada durante el entrenamiento. Esto evita avisos
# por secuencias superiores a 512 tokens y mejora la alineación RoBERTa-SHAP.
PRETRUNCATE_TEXTS_FOR_SHAP = True

# Diagnóstico de alineación. Si la tasa queda por debajo de este valor se
# imprime una advertencia. El proceso no se detiene salvo que se active
# FAIL_ON_LOW_ALIGNMENT.
MIN_EXACT_ALIGNMENT_RATE = 0.90
FAIL_ON_LOW_ALIGNMENT = False

# Tokens mostrados en consola y en las figuras finales.
TOP_K_CONSOLE = 20
RANKED_TWO_COLUMN_TOP_K = 20

# "pred" -> selecciona textos según la clase predicha por el modelo.
# "true" -> selecciona textos según la etiqueta verdadera.
FILTER_MODE = "true"

# Con True se toma la misma cantidad de textos para Cj y Cr.
BALANCE_PAIR_SAMPLE = True

# Selección del par por consola. Al presionar Enter se conservan
# los valores por defecto.
USE_INTERACTIVE_CONSOLE_SELECTION = False
DEFAULT_CLASS_J_NAME = "sci.space"
DEFAULT_CLASS_R_NAME = "comp.graphics"

# Generar únicamente las figuras finales ranked two-column.
GENERATE_RANKED_TWO_COLUMN_FIGURES = True

# Diagnóstico de alineación. False evita imprimir una advertencia
# por cada instancia; al final siempre se imprime el resumen.
VERBOSE_TOKEN_ALIGNMENT = False


# ============================================================
# 0.1) Configuración visual
# ============================================================

RANKED_TWO_COLUMN_MIN_FONT = 14
RANKED_TWO_COLUMN_MAX_FONT = 46
RANKED_TWO_COLUMN_TITLE_FONT = 22
RANKED_TWO_COLUMN_LEGEND_FONT = 15
RANKED_TWO_COLUMN_DPI = 600
RANKED_TWO_COLUMN_FIG_WIDTH = 10.5
RANKED_TWO_COLUMN_FIG_HEIGHT = 6.2
RANKED_TWO_COLUMN_SHOW_RANK_NUMBERS = False

NEUTRAL_COLOR = "#31688E"
CJ_COLOR = "#C2185B"
CR_COLOR = "#00897B"


# ============================================================
# 1) Utilidades de nombres y carga de datos
# ============================================================


def safe_filename(text: str) -> str:
    """Convierte nombres de clases en nombres seguros para Windows."""

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
        .replace(".", "_")
        .replace("-", "_")
    )



def visualizer_safe_name(text: str) -> str:
    """
    Reproduce el saneamiento usado internamente por la clase de
    figuras ranked. Se usa solo para localizar y renombrar sus salidas.
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



def safe_prob_col_name(text: str) -> str:
    """Nombre usado para columnas de probabilidad guardadas."""

    return (
        str(text)
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )



def load_class_names(
    label_mapping_path: str,
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Carga label_mapping.csv como diccionarios id<->nombre."""

    label_df = pd.read_csv(label_mapping_path)

    required_columns = {"label", "label_name"}
    missing = required_columns.difference(label_df.columns)

    if missing:
        raise ValueError(
            "label_mapping.csv no contiene las columnas requeridas: "
            f"{sorted(missing)}"
        )

    label_df = label_df.sort_values("label").reset_index(drop=True)

    class_names = {
        int(row["label"]): str(row["label_name"])
        for _, row in label_df.iterrows()
    }

    label_to_id = {
        class_name: class_id
        for class_id, class_name in class_names.items()
    }

    return class_names, label_to_id



def find_probability_column(df: pd.DataFrame, class_name: str) -> str:
    """Busca la columna de probabilidad correspondiente a una clase."""

    candidates = [
        f"prob_{class_name}",
        f"prob_{safe_prob_col_name(class_name)}",
        f"prob_{safe_filename(class_name)}",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"No se encontró columna de probabilidad para '{class_name}'. "
        f"Candidatas revisadas: {candidates}"
    )



def load_saved_predictions(
    test_split_path: str,
    predictions_path: str,
    class_names: Dict[int, str],
) -> Tuple[
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Carga textos, etiquetas, predicciones y probabilidades desde los
    archivos producidos durante el entrenamiento de RoBERTa.
    """

    df_test = pd.read_csv(test_split_path)
    df_predictions = pd.read_csv(predictions_path)

    if len(df_test) != len(df_predictions):
        raise ValueError(
            "test_split.csv y test_predictions.csv tienen tamaños "
            f"diferentes: {len(df_test)} vs {len(df_predictions)}"
        )

    if "text" not in df_test.columns:
        raise ValueError("test_split.csv debe contener la columna 'text'.")

    if "label" in df_predictions.columns:
        y_true = df_predictions["label"].astype(int).to_numpy()
    elif "true_label" in df_predictions.columns:
        y_true = df_predictions["true_label"].astype(int).to_numpy()
    elif "label" in df_test.columns:
        y_true = df_test["label"].astype(int).to_numpy()
    else:
        raise ValueError(
            "No se encontró la etiqueta verdadera en test_predictions.csv "
            "ni en test_split.csv."
        )

    if "pred_label" not in df_predictions.columns:
        raise ValueError(
            "test_predictions.csv debe contener la columna 'pred_label'."
        )

    y_pred = df_predictions["pred_label"].astype(int).to_numpy()

    probability_columns = []
    for class_id in sorted(class_names):
        probability_columns.append(
            find_probability_column(
                df=df_predictions,
                class_name=class_names[class_id],
            )
        )

    probabilities = (
        df_predictions[probability_columns]
        .astype(float)
        .to_numpy()
    )

    texts = df_test["text"].astype(str).tolist()

    return (
        texts,
        y_true,
        y_pred,
        probabilities,
        df_test,
        df_predictions,
    )


def prepare_texts_for_shap(
    texts: Sequence[str],
    tokenizer,
    max_length: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Recorta cada texto a ``max_length`` tokens antes de entregarlo a SHAP.

    La salida se vuelve a convertir a texto para que PartitionExplainer
    trabaje únicamente con el contenido que realmente recibe el modelo.
    Esto evita que SHAP intente segmentar documentos de más de 512 tokens
    aunque la función de predicción los recorte internamente.

    Returns
    -------
    prepared_texts:
        Textos utilizados realmente por SHAP.
    original_token_counts:
        Número de tokens del texto original, incluidos tokens especiales.
    truncated_flags:
        Indica qué textos fueron recortados.
    """

    prepared_texts: List[str] = []
    original_token_counts: List[int] = []
    truncated_flags: List[bool] = []

    for text_index, text in enumerate(texts):
        text = str(text)

        full_encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )

        full_ids = list(full_encoding["input_ids"])

        truncated_encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            verbose=False,
        )

        truncated_ids = list(truncated_encoding["input_ids"])

        prepared_text = tokenizer.decode(
            truncated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if prepared_text == "":
            raise ValueError(
                "El texto preparado para SHAP quedó vacío en la posición "
                f"{text_index}."
            )

        verification = tokenizer(
            prepared_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )

        verification_length = len(verification["input_ids"])

        if verification_length > max_length:
            # Salvaguarda excepcional: recortar nuevamente después de decode.
            second_encoding = tokenizer(
                prepared_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=False,
                verbose=False,
            )
            prepared_text = tokenizer.decode(
                second_encoding["input_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        prepared_texts.append(prepared_text)
        original_token_counts.append(len(full_ids))
        truncated_flags.append(len(full_ids) > max_length)

    counts_array = np.asarray(original_token_counts, dtype=int)
    flags_array = np.asarray(truncated_flags, dtype=bool)

    print("\nPreparación explícita de textos para SHAP:")
    print(f"  Textos procesados: {len(prepared_texts)}")
    print(f"  Textos recortados: {int(flags_array.sum())}")
    print(f"  Longitud máxima permitida: {max_length} tokens")

    if len(counts_array) > 0:
        print(
            "  Longitud original máxima observada: "
            f"{int(counts_array.max())} tokens"
        )

    return prepared_texts, counts_array, flags_array


# ============================================================
# 2) Selección del par por consola
# ============================================================


def print_available_classes(class_names: Dict[int, str]) -> None:
    print("\nClases disponibles:")
    print("------------------------------------------------------------")

    for class_id, class_name in class_names.items():
        print(f"{class_id:2d}: {class_name}")

    print("------------------------------------------------------------")



def parse_class_choice(
    raw_value: str,
    class_names: Dict[int, str],
) -> Optional[str]:
    """Admite índice numérico o nombre de clase."""

    raw_value = str(raw_value).strip()

    if raw_value == "":
        return None

    if raw_value.isdigit():
        class_id = int(raw_value)

        if class_id not in class_names:
            raise ValueError(f"Índice de clase no válido: {class_id}")

        return class_names[class_id]

    lower_map = {
        class_name.lower(): class_name
        for class_name in class_names.values()
    }

    key = raw_value.lower()

    if key not in lower_map:
        raise ValueError(
            f"Clase no encontrada: {raw_value}. Usa el índice o el "
            "nombre exacto."
        )

    return lower_map[key]



def ask_class_choice(
    prompt: str,
    class_names: Dict[int, str],
    default_name: str,
) -> str:
    while True:
        raw_value = input(f"{prompt} [{default_name}]: ").strip()

        if raw_value == "":
            return default_name

        try:
            class_name = parse_class_choice(
                raw_value=raw_value,
                class_names=class_names,
            )

            if class_name is not None:
                return class_name

        except ValueError as error:
            print(f"Entrada no válida: {error}")



def select_pair_configuration(
    class_names: Dict[int, str],
) -> Tuple[str, str]:
    """Selecciona Cj y Cr o usa los valores por defecto."""

    if not USE_INTERACTIVE_CONSOLE_SELECTION:
        return DEFAULT_CLASS_J_NAME, DEFAULT_CLASS_R_NAME

    print_available_classes(class_names)
    print("\nSeleccione el par de clases para SHAP-GLCD global.")
    print("Puede ingresar el índice numérico o el nombre exacto.")

    class_j_name = ask_class_choice(
        prompt="Ingrese Cj",
        class_names=class_names,
        default_name=DEFAULT_CLASS_J_NAME,
    )

    while True:
        class_r_name = ask_class_choice(
            prompt="Ingrese Cr",
            class_names=class_names,
            default_name=DEFAULT_CLASS_R_NAME,
        )

        if class_r_name != class_j_name:
            break

        print("Cj y Cr no pueden ser la misma clase.")

    return class_j_name, class_r_name


# ============================================================
# 3) Selección balanceada de textos del par
# ============================================================


def print_label_distribution(
    labels: np.ndarray,
    class_names: Dict[int, str],
    title: str,
) -> None:
    print(f"\n{title}:")

    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        label_int = int(label)
        label_name = class_names.get(label_int, f"class_{label_int}")
        print(f"  {label_name}: {int(count)}")



def select_pair_sample(
    texts: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_j_name: str,
    class_r_name: str,
    label_to_id: Dict[str, int],
    class_names: Dict[int, str],
    filter_mode: str = "pred",
    max_eval: int = 20,
    balance: bool = True,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Selecciona una muestra reproducible para el análisis global."""

    if max_eval < 2:
        raise ValueError("MAX_EVAL_GLCD debe ser al menos 2.")

    rng = np.random.default_rng(seed)

    class_j_id = label_to_id[class_j_name]
    class_r_id = label_to_id[class_r_name]

    if filter_mode == "pred":
        labels_for_filter = y_pred
    elif filter_mode == "true":
        labels_for_filter = y_true
    else:
        raise ValueError("FILTER_MODE debe ser 'pred' o 'true'.")

    indices_j = np.where(labels_for_filter == class_j_id)[0]
    indices_r = np.where(labels_for_filter == class_r_id)[0]

    if len(indices_j) == 0 or len(indices_r) == 0:
        raise ValueError(
            "No hay textos suficientes para construir el par "
            f"{class_j_name} vs {class_r_name} con FILTER_MODE="
            f"'{filter_mode}'."
        )

    if balance:
        n_each = min(
            len(indices_j),
            len(indices_r),
            max_eval // 2,
        )

        if n_each < 1:
            raise ValueError(
                "No fue posible seleccionar al menos un texto por clase."
            )

        selected_j = rng.choice(
            indices_j,
            size=n_each,
            replace=False,
        )

        selected_r = rng.choice(
            indices_r,
            size=n_each,
            replace=False,
        )

        selected_indices = np.concatenate([selected_j, selected_r])

    else:
        candidate_indices = np.concatenate([indices_j, indices_r])
        n_total = min(len(candidate_indices), max_eval)

        selected_indices = rng.choice(
            candidate_indices,
            size=n_total,
            replace=False,
        )

    # Ordenar conserva el orden original del conjunto de test y garantiza
    # resultados deterministas en los archivos y la caché.
    selected_indices = np.sort(selected_indices.astype(int))

    selected_texts = [texts[index] for index in selected_indices]
    selected_y_true = y_true[selected_indices].astype(int)
    selected_y_pred = y_pred[selected_indices].astype(int)

    print("\nMuestra seleccionada para SHAP-GLCD:")
    print("------------------------------------------------------------")
    print(f"Par: {class_j_name} vs {class_r_name}")
    print(f"Filtro: {filter_mode}")
    print(f"Balanceada: {balance}")
    print(f"Semilla: {seed}")
    print(f"N seleccionado: {len(selected_indices)}")
    print("------------------------------------------------------------")

    print_label_distribution(
        labels=selected_y_true,
        class_names=class_names,
        title="Distribución por etiquetas verdaderas",
    )

    print_label_distribution(
        labels=selected_y_pred,
        class_names=class_names,
        title="Distribución por predicciones",
    )

    return (
        selected_texts,
        selected_y_true,
        selected_y_pred,
        selected_indices,
    )



def save_selected_metadata(
    original_texts: Sequence[str],
    shap_input_texts: Sequence[str],
    original_token_counts: np.ndarray,
    truncated_flags: np.ndarray,
    selected_y_true: np.ndarray,
    selected_y_pred: np.ndarray,
    selected_indices: np.ndarray,
    probabilities: np.ndarray,
    class_names: Dict[int, str],
    class_j_name: str,
    class_r_name: str,
    label_to_id: Dict[str, int],
    output_path: str,
) -> pd.DataFrame:
    """Guarda el origen, las etiquetas y las probabilidades del par."""

    class_j_id = label_to_id[class_j_name]
    class_r_id = label_to_id[class_r_name]

    rows = []

    for local_index, original_index in enumerate(selected_indices):
        true_id = int(selected_y_true[local_index])
        pred_id = int(selected_y_pred[local_index])
        original_index_int = int(original_index)

        rows.append({
            "local_index": int(local_index),
            "original_test_index": original_index_int,
            "true_label": true_id,
            "true_label_name": class_names[true_id],
            "pred_label": pred_id,
            "pred_label_name": class_names[pred_id],
            f"P_{safe_filename(class_j_name)}": float(
                probabilities[original_index_int, class_j_id]
            ),
            f"P_{safe_filename(class_r_name)}": float(
                probabilities[original_index_int, class_r_id]
            ),
            "original_token_count": int(original_token_counts[local_index]),
            "was_truncated_for_shap": bool(truncated_flags[local_index]),
            "original_text": str(original_texts[local_index]),
            "shap_input_text": str(shap_input_texts[local_index]),
        })

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Metadatos de selección guardados en: {output_path}")

    return metadata_df


# ============================================================
# 4) Caché, salidas y diagnósticos
# ============================================================


def selection_signature(selected_indices: np.ndarray) -> str:
    """Huella corta para evitar reutilizar una caché de otra muestra."""

    contiguous = np.ascontiguousarray(
        selected_indices,
        dtype=np.int64,
    )

    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:12]



def save_run_configuration(
    output_path: str,
    class_j_name: str,
    class_r_name: str,
    n_selected: int,
    cache_path: str,
) -> None:
    configuration = pd.DataFrame([
        {
            "seed": SEED,
            "dataset": DATASET_NAME,
            "model_dir": MODEL_DIR,
            "class_j": class_j_name,
            "class_r": class_r_name,
            "filter_mode": FILTER_MODE,
            "balance_pair_sample": BALANCE_PAIR_SAMPLE,
            "max_eval_requested": MAX_EVAL_GLCD,
            "n_selected": int(n_selected),
            "max_length": MAX_LENGTH,
            "min_token_frequency": MIN_TOKEN_FREQUENCY,
            "min_document_frequency": MIN_DOCUMENT_FREQUENCY,
            "pretruncate_texts_for_shap": PRETRUNCATE_TEXTS_FOR_SHAP,
            "minimum_exact_alignment_rate": MIN_EXACT_ALIGNMENT_RATE,
            "ranked_top_k": RANKED_TWO_COLUMN_TOP_K,
            "shap_cache": os.path.basename(cache_path),
        }
    ])

    configuration.to_csv(output_path, index=False, encoding="utf-8")



def calculate_or_load_shap(
    pair_explainer: SHAPTextExplainerLCD,
    shap_pickle_path: str,
    n_selected: int,
) -> str:
    """Carga una caché válida o calcula SHAP y la guarda."""

    if os.path.exists(shap_pickle_path):
        print(f"\nCargando SHAP desde caché: {shap_pickle_path}")

        try:
            pair_explainer.load_shap_values(shap_pickle_path)
            return "loaded_from_cache"

        except Exception as error:
            print("La caché no pudo utilizarse y SHAP será recalculado.")
            print(f"Motivo: {error}")

    print("\nCalculando SHAP para el análisis global...")
    print(
        "En CPU puede tardar varios minutos, porque se explicarán "
        f"{n_selected} textos."
    )

    pair_explainer.calculate_shap_values(max_eval=n_selected)
    pair_explainer.save_shap_values(shap_pickle_path)

    return "calculated"



def rename_ranked_outputs_with_safe_names(
    output_dir: str,
    class_j_name: str,
    class_r_name: str,
) -> None:
    """
    Conserva los títulos con los nombres reales, pero reemplaza puntos y
    guiones en los nombres finales de los archivos.
    """

    old_suffix = (
        f"{visualizer_safe_name(class_j_name)}_vs_"
        f"{visualizer_safe_name(class_r_name)}"
    )

    new_suffix = (
        f"{safe_filename(class_j_name)}_vs_"
        f"{safe_filename(class_r_name)}"
    )

    if old_suffix == new_suffix:
        return

    prefixes = [
        "26_article_two_column_ranked_mean_abs_",
        "27_article_two_column_ranked_median_abs_",
        "28_article_two_column_ranked_signed_mean_",
        "29_article_two_column_ranked_cdm_",
        "30_article_two_column_ranked_dp_",
    ]

    for prefix in prefixes:
        for extension in (".png", ".pdf"):
            old_path = os.path.join(
                output_dir,
                f"{prefix}{old_suffix}{extension}",
            )
            new_path = os.path.join(
                output_dir,
                f"{prefix}{new_suffix}{extension}",
            )

            if os.path.exists(old_path):
                os.replace(old_path, new_path)
                print(f"Figura guardada en: {new_path}")


# ============================================================
# 5) Semillas y dispositivo
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("============================================================")
print("SHAP-GLCD global para 20 Newsgroups + RoBERTa-base")
print("VERSIÓN ACTUALIZADA: N=200 | freq>=5 | doc_freq>=3 | pretruncado")
print("============================================================")
print(f"Dispositivo usado: {device}")


# ============================================================
# 6) Verificar archivos requeridos
# ============================================================

print("\nDirectorio actual de ejecución:")
print(os.getcwd())

print("\nModelo esperado en:")
print(os.path.abspath(MODEL_DIR))

print("\nResultados del entrenamiento esperados en:")
print(os.path.abspath(TRAIN_RESULTS_DIR))

print("\nResultados SHAP-GLCD se guardarán en:")
print(os.path.abspath(RESULTS_DIR))

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"No se encontró el modelo en: {MODEL_DIR}\n"
        "Primero debe existir el modelo RoBERTa entrenado."
    )

label_mapping_path = os.path.join(
    TRAIN_RESULTS_DIR,
    "label_mapping.csv",
)
test_split_path = os.path.join(
    TRAIN_RESULTS_DIR,
    "test_split.csv",
)
predictions_path = os.path.join(
    TRAIN_RESULTS_DIR,
    "test_predictions.csv",
)

for required_path in (
    label_mapping_path,
    test_split_path,
    predictions_path,
):
    if not os.path.isfile(required_path):
        raise FileNotFoundError(f"No se encontró: {required_path}")


# ============================================================
# 7) Cargar mapeo, test y predicciones guardadas
# ============================================================

CLASS_NAMES, LABEL_TO_ID = load_class_names(label_mapping_path)

print("\nClases cargadas:")
for class_id, class_name in CLASS_NAMES.items():
    print(f"  {class_id}: {class_name}")

(
    texts_test,
    y_test,
    y_pred,
    probabilities_test,
    df_test,
    df_predictions,
) = load_saved_predictions(
    test_split_path=test_split_path,
    predictions_path=predictions_path,
    class_names=CLASS_NAMES,
)

print(f"\nNúmero total de textos en test: {len(texts_test)}")

print_label_distribution(
    labels=y_pred,
    class_names=CLASS_NAMES,
    title="Distribución de predicciones en test",
)


# ============================================================
# 8) Seleccionar y validar el par
# ============================================================

CLASS_J_NAME, CLASS_R_NAME = select_pair_configuration(CLASS_NAMES)

if CLASS_J_NAME not in LABEL_TO_ID:
    raise ValueError(f"La clase Cj no existe: {CLASS_J_NAME}")

if CLASS_R_NAME not in LABEL_TO_ID:
    raise ValueError(f"La clase Cr no existe: {CLASS_R_NAME}")

if CLASS_J_NAME == CLASS_R_NAME:
    raise ValueError("Cj y Cr no pueden ser la misma clase.")

print("\nConfiguración global seleccionada:")
print("------------------------------------------------------------")
print(f"Cj: {CLASS_J_NAME}")
print(f"Cr: {CLASS_R_NAME}")
print(f"FILTER_MODE: {FILTER_MODE}")
print(f"BALANCE_PAIR_SAMPLE: {BALANCE_PAIR_SAMPLE}")
print(f"MAX_EVAL_GLCD: {MAX_EVAL_GLCD}")
print(f"MIN_TOKEN_FREQUENCY: {MIN_TOKEN_FREQUENCY}")
print(f"MIN_DOCUMENT_FREQUENCY: {MIN_DOCUMENT_FREQUENCY}")
print(f"PRETRUNCATE_TEXTS_FOR_SHAP: {PRETRUNCATE_TEXTS_FOR_SHAP}")
print("------------------------------------------------------------")


# ============================================================
# 9) Cargar modelo y tokenizador
# ============================================================

print(f"\nCargando modelo desde: {MODEL_DIR}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Mantiene explícitamente la misma longitud máxima del entrenamiento.
tokenizer.model_max_length = MAX_LENGTH

model.to(device)
model.eval()


# ============================================================
# 10) Seleccionar muestra balanceada
# ============================================================

(
    selected_texts,
    selected_y_true,
    selected_y_pred,
    selected_indices,
) = select_pair_sample(
    texts=texts_test,
    y_true=y_test,
    y_pred=y_pred,
    class_j_name=CLASS_J_NAME,
    class_r_name=CLASS_R_NAME,
    label_to_id=LABEL_TO_ID,
    class_names=CLASS_NAMES,
    filter_mode=FILTER_MODE,
    max_eval=MAX_EVAL_GLCD,
    balance=BALANCE_PAIR_SAMPLE,
    seed=SEED,
)

if PRETRUNCATE_TEXTS_FOR_SHAP:
    (
        shap_input_texts,
        original_token_counts,
        truncated_flags,
    ) = prepare_texts_for_shap(
        texts=selected_texts,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
else:
    shap_input_texts = list(selected_texts)
    original_token_counts = np.asarray(
        [
            len(
                tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False,
                    verbose=False,
                )["input_ids"]
            )
            for text in selected_texts
        ],
        dtype=int,
    )
    truncated_flags = np.zeros(len(selected_texts), dtype=bool)

safe_j = safe_filename(CLASS_J_NAME)
safe_r = safe_filename(CLASS_R_NAME)

pair_results_dir = os.path.join(
    RESULTS_DIR,
    f"pair_{safe_j}_vs_{safe_r}",
)

os.makedirs(pair_results_dir, exist_ok=True)

metadata_path = os.path.join(
    pair_results_dir,
    "selected_text_metadata.csv",
)

save_selected_metadata(
    original_texts=selected_texts,
    shap_input_texts=shap_input_texts,
    original_token_counts=original_token_counts,
    truncated_flags=truncated_flags,
    selected_y_true=selected_y_true,
    selected_y_pred=selected_y_pred,
    selected_indices=selected_indices,
    probabilities=probabilities_test,
    class_names=CLASS_NAMES,
    class_j_name=CLASS_J_NAME,
    class_r_name=CLASS_R_NAME,
    label_to_id=LABEL_TO_ID,
    output_path=metadata_path,
)

selected_indices_path = os.path.join(
    pair_results_dir,
    "selected_test_indices.npy",
)
np.save(selected_indices_path, selected_indices)
print(f"Índices seleccionados guardados en: {selected_indices_path}")

sample_signature = selection_signature(selected_indices)

shap_pickle_path = os.path.join(
    SHAP_CACHE_DIR,
    (
        f"shap_values_glcd_{safe_j}_vs_{safe_r}_"
        f"N{len(selected_texts)}_{FILTER_MODE}_"
        f"seed{SEED}_L{MAX_LENGTH}_pretruncated{int(PRETRUNCATE_TEXTS_FOR_SHAP)}_"
        f"{sample_signature}.pickle"
    ),
)

configuration_path = os.path.join(
    pair_results_dir,
    "run_configuration.csv",
)

save_run_configuration(
    output_path=configuration_path,
    class_j_name=CLASS_J_NAME,
    class_r_name=CLASS_R_NAME,
    n_selected=len(selected_texts),
    cache_path=shap_pickle_path,
)
print(f"Configuración guardada en: {configuration_path}")


# ============================================================
# 11) Calcular o cargar SHAP
# ============================================================

pair_explainer = SHAPTextExplainerLCD(
    model=model,
    tokenizer=tokenizer,
    texts_eval=shap_input_texts,
    y_eval=selected_y_true,
    class_names=CLASS_NAMES,
    max_length=MAX_LENGTH,
    batch_size=8,
    device=device,
    random_state=SEED,
)

# Permite que los registros globales conserven el índice original del test.
pair_explainer.original_indices = selected_indices

shap_source = calculate_or_load_shap(
    pair_explainer=pair_explainer,
    shap_pickle_path=shap_pickle_path,
    n_selected=len(selected_texts),
)

print(f"SHAP disponible: {shap_source}")


# ============================================================
# 12) Agregación global SHAP-GLCD para RoBERTa
# ============================================================

print("\nCalculando registros deltaS y resumen global por token...")

glcd_global = SHAPGlobalTextGLCD20Newsgroups(
    lcd_explainer=pair_explainer,
    output_dir=pair_results_dir,
    min_token_frequency=MIN_TOKEN_FREQUENCY,
    normalize_token_case=True,
    combine_subwords=True,
    remove_punctuation_only=True,
    verbose_alignment=VERBOSE_TOKEN_ALIGNMENT,
)

df_records, df_global_unfiltered = glcd_global.aggregate_pair_global(
    class_j=CLASS_J_NAME,
    class_r=CLASS_R_NAME,
)

if "document_frequency" not in df_global_unfiltered.columns:
    raise RuntimeError(
        "El resumen global no contiene la columna document_frequency."
    )

# Segundo filtro: exige presencia en varios documentos, no solo varias
# repeticiones dentro de un único texto.
df_global = df_global_unfiltered[
    df_global_unfiltered["document_frequency"] >= MIN_DOCUMENT_FREQUENCY
].copy()

df_global = df_global.sort_values(
    by=["mean_abs_deltaS", "document_frequency", "frequency"],
    ascending=[False, False, False],
).reset_index(drop=True)

if df_records.empty:
    raise RuntimeError("No se generaron registros deltaS.")

if df_global.empty:
    raise RuntimeError(
        "El resumen global quedó vacío después de aplicar los filtros. "
        "Revise MIN_TOKEN_FREQUENCY y MIN_DOCUMENT_FREQUENCY."
    )

records_path = os.path.join(
    pair_results_dir,
    f"deltaS_records_{safe_j}_vs_{safe_r}.csv",
)

global_summary_unfiltered_path = os.path.join(
    pair_results_dir,
    f"global_token_summary_unfiltered_{safe_j}_vs_{safe_r}.csv",
)

global_summary_path = os.path.join(
    pair_results_dir,
    f"global_token_summary_{safe_j}_vs_{safe_r}.csv",
)

df_records.to_csv(records_path, index=False, encoding="utf-8")
df_global_unfiltered.to_csv(
    global_summary_unfiltered_path,
    index=False,
    encoding="utf-8",
)
df_global.to_csv(global_summary_path, index=False, encoding="utf-8")

print(f"Registros deltaS guardados en: {records_path}")
print(
    "Resumen global sin filtro documental guardado en: "
    f"{global_summary_unfiltered_path}"
)
print(f"Resumen global filtrado guardado en: {global_summary_path}")
print(
    "Tokens antes/después del filtro documental: "
    f"{len(df_global_unfiltered)} -> {len(df_global)}"
)

glcd_global.print_alignment_summary()

alignment_data = glcd_global.get_alignment_summary()
alignment_summary = pd.DataFrame([alignment_data])

exact_alignment_rate = float(alignment_data["exact_alignment_rate"])

if exact_alignment_rate < MIN_EXACT_ALIGNMENT_RATE:
    warning_message = (
        "La tasa de alineación exacta quedó por debajo del umbral: "
        f"{exact_alignment_rate:.3f} < {MIN_EXACT_ALIGNMENT_RATE:.3f}."
    )

    if FAIL_ON_LOW_ALIGNMENT:
        raise RuntimeError(warning_message)

    print(f"ADVERTENCIA: {warning_message}")
else:
    print(
        "Alineación validada: "
        f"{exact_alignment_rate:.3f} >= {MIN_EXACT_ALIGNMENT_RATE:.3f}"
    )

alignment_summary_path = os.path.join(
    pair_results_dir,
    "alignment_summary.csv",
)

alignment_summary.to_csv(
    alignment_summary_path,
    index=False,
    encoding="utf-8",
)

print(f"Resumen de alineación guardado en: {alignment_summary_path}")

columns_to_show = [
    "token",
    "frequency",
    "document_frequency",
    "mean_abs_deltaS",
    "median_abs_deltaS",
    "signed_mean_deltaS",
    "cdm_value",
    "W_positive",
    "W_negative",
    "dp_strength",
    "dominant_class",
]

available_columns = [
    column
    for column in columns_to_show
    if column in df_global.columns
]

print(f"\nTop {TOP_K_CONSOLE} tokens globales por mean |deltaS|:")
print(
    df_global[available_columns]
    .head(TOP_K_CONSOLE)
    .to_string(index=False)
)


# ============================================================
# 13) Figuras globales ranked two-column para artículo
# ============================================================

if GENERATE_RANKED_TWO_COLUMN_FIGURES:
    print("\nGenerando figuras globales ranked two-column...")

    ranked_visualizer = SHAPGlobalTextTwoColumnRankedArticleVisualizer(
        output_dir=pair_results_dir,
        token_col=None,
        min_font_size=RANKED_TWO_COLUMN_MIN_FONT,
        max_font_size=RANKED_TWO_COLUMN_MAX_FONT,
        title_font_size=RANKED_TWO_COLUMN_TITLE_FONT,
        legend_font_size=RANKED_TWO_COLUMN_LEGEND_FONT,
        dpi=RANKED_TWO_COLUMN_DPI,
        neutral_color=NEUTRAL_COLOR,
        cj_color=CJ_COLOR,
        cr_color=CR_COLOR,
        fig_width=RANKED_TWO_COLUMN_FIG_WIDTH,
        fig_height=RANKED_TWO_COLUMN_FIG_HEIGHT,
        show_rank_numbers=RANKED_TWO_COLUMN_SHOW_RANK_NUMBERS,
    )

    ranked_visualizer.plot_all(
        df_global=df_global,
        class_j=CLASS_J_NAME,
        class_r=CLASS_R_NAME,
        top_k=RANKED_TWO_COLUMN_TOP_K,
        include_signed_mean=True,
        include_cdm=True,
        include_dp=True,
    )

    rename_ranked_outputs_with_safe_names(
        output_dir=pair_results_dir,
        class_j_name=CLASS_J_NAME,
        class_r_name=CLASS_R_NAME,
    )


# ============================================================
# 14) Resumen final
# ============================================================

print("\n============================================================")
print("Proceso SHAP-GLCD global finalizado.")
print("============================================================")
print(f"Par explicado: {CLASS_J_NAME} vs {CLASS_R_NAME}")
print(f"Textos analizados: {len(shap_input_texts)}")
print(f"Textos recortados antes de SHAP: {int(truncated_flags.sum())}")
print(f"Tokens del resumen final: {len(df_global)}")
print(f"Fuente SHAP: {shap_source}")
print(f"Resultados guardados en: {os.path.abspath(pair_results_dir)}")
