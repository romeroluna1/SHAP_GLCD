# ============================================================
# stability_20_newsgroups.py
# ============================================================
# Stability / Consistency de explicaciones diferenciales
# SHAP-LCD / SHAP-GLCD para 20 Newsgroups + RoBERTa-base.
#
# MISMO PROTOCOLO CIENTIFICO DEL MAIN FINAL:
#   - NO entrena el modelo.
#   - NO recalcula SHAP-GLCD global.
#   - mismo test guardado durante el entrenamiento.
#   - par: sci.space vs comp.graphics.
#   - seleccion por ETIQUETA VERDADERA.
#   - N=50, balance 25+25, sin filtro por aciertos.
#   - M=40 perturbaciones por instancia.
#   - perturbation_rate=0.05.
#   - tau=0.95.
#   - Top-K=10 + normalizacion L2.
#   - seed=42; por instancia seed=42+t.
#   - MAX_LENGTH=256.
#   - token masking + alineacion estricta de la MISMA clase
#     SHAPTextLCDMetrics usada en AG News.
#
# OPTIMIZACIONES (NO CAMBIAN LA DEFINICION DEL EXPERIMENTO):
#   1) Se seleccionan primero las 50 instancias y SOLO ellas se
#      pretruncan a 256 tokens.
#   2) La funcion de prediccion que usa SHAP incorpora una cache
#      exacta de textos -> probabilidades. Si SHAP solicita de nuevo
#      exactamente el mismo texto enmascarado, se reutiliza la misma
#      prediccion en vez de volver a ejecutar RoBERTa.
#   3) El batch interno de predict_proba se fija en 10, que coincide
#      con el batch por defecto de PartitionExplainer; no se cambia
#      max_evals ni el algoritmo de SHAP.
#   4) Se guarda checkpoint DESPUES DE CADA INSTANCIA. Si la corrida
#      se interrumpe, puede reanudarse sin repetir las ya terminadas.
#
# IMPORTANTE:
#   No se modifica max_evals de PartitionExplainer. Por tanto, esta
#   version no acelera a costa de aproximar SHAP con menos evaluaciones.
# ============================================================

import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.text import SHAPTextExplainerLCD, SHAPTextLCDMetrics


# ============================================================
# 0) CONFIGURACION FINAL DEL ARTICULO
# ============================================================

SEED = 42

DATASET_NAME = "20newsgroups"
MODEL_SAFE_NAME = "roberta_base"
MODEL_DISPLAY_NAME = "roberta-base"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "20_newsgroups"
MODEL_DIR = ARTIFACT_DIR / "model_20newsgroups_roberta_base"
TRAIN_RESULTS_DIR = ARTIFACT_DIR / "results_20newsgroups_roberta_base"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation" / "20_newsgroups"

# Mantiene la prueba pequeña separada de la corrida definitiva para que
# sus checkpoints nunca se mezclen.
QUICK_TEST = False
RUN_NAME = "quick_test" if QUICK_TEST else "final"
METRICS_DIR = RESULTS_DIR / RUN_NAME
os.makedirs(METRICS_DIR, exist_ok=True)

MAX_LENGTH = 256

# PartitionExplainer usa batch_size=10 por defecto. El predict_proba
# original estaba en 8, por lo que cada bloque de 10 podia dividirse
# en dos forwards (8+2). Con 10 se atiende en un solo forward.
# Esto NO cambia masks, max_evals ni definicion de SHAP.
MODEL_PRED_BATCH_SIZE = 10

PRETRUNCATE_TEXTS_FOR_SHAP = True

# Protocolo comun Stability / Consistency. La corrida definitiva conserva
# exactamente N=50 y M=40; QUICK_TEST solo valida la instalación con N=2, M=2.
N_INSTANCES = 2 if QUICK_TEST else 50
N_EACH = N_INSTANCES // 2
M_PERTURBATIONS = 2 if QUICK_TEST else 40
PERTURBATION_RATE = 0.05
TAU = 0.95
TOP_K = 10
NORMALIZE = "l2"
FILTER_MODE = "true"
BALANCED_SAMPLING = True

CLASS_J_NAME = "sci.space"
CLASS_R_NAME = "comp.graphics"

# Cache exacta solo para las llamadas del modelo hechas internamente por SHAP.
# Se vacia al comenzar cada instancia; 30000 entradas cubren holgadamente
# una instancia con 41 textos y max_evals=500.
ENABLE_EXACT_SHAP_PREDICTION_CACHE = True
CACHE_MAX_ENTRIES = 30000

CHECKPOINT_PATH = os.path.join(
    METRICS_DIR,
    "metrics_instances_CHECKPOINT.csv",
)
SELECTED_INDICES_PATH = os.path.join(
    METRICS_DIR,
    "selected_test_indices.npy",
)
RUN_CONFIGURATION_PATH = os.path.join(
    METRICS_DIR,
    "run_configuration.csv",
)


# ============================================================
# 0.1) VALIDACION DE CONFIGURACION
# ============================================================

if N_INSTANCES < 2 or N_INSTANCES % 2 != 0:
    raise ValueError("N_INSTANCES debe ser par y >= 2.")
if M_PERTURBATIONS < 1:
    raise ValueError("M_PERTURBATIONS debe ser >= 1.")
if not 0 < PERTURBATION_RATE <= 1:
    raise ValueError("PERTURBATION_RATE debe estar en (0,1].")
if not 0 < TAU <= 1:
    raise ValueError("TAU debe estar en (0,1].")
if TOP_K < 1:
    raise ValueError("TOP_K debe ser >= 1.")
if str(NORMALIZE).lower() != "l2":
    raise ValueError("NORMALIZE debe ser 'l2'.")
if FILTER_MODE != "true":
    raise ValueError("El protocolo final requiere FILTER_MODE='true'.")
if not BALANCED_SAMPLING:
    raise ValueError("El protocolo final requiere muestra balanceada.")
if not PRETRUNCATE_TEXTS_FOR_SHAP:
    raise ValueError(
        "20 Newsgroups debe mantener el pretruncado explicito a 256 tokens."
    )


# ============================================================
# 1) SEMILLAS Y DISPOSITIVO
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2) FUNCIONES DE CARGA
# ============================================================

def safe_filename(text: str) -> str:
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


def load_class_names(label_mapping_path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
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


def load_saved_test(
    test_split_path: str,
    predictions_path: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Carga exactamente el test y las predicciones guardadas del entrenamiento."""
    df_test = pd.read_csv(test_split_path)
    df_predictions = pd.read_csv(predictions_path)

    if len(df_test) != len(df_predictions):
        raise ValueError(
            "test_split.csv y test_predictions.csv tienen tamanos diferentes: "
            f"{len(df_test)} vs {len(df_predictions)}"
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
        raise ValueError("No se encontro la etiqueta verdadera del test.")

    if "pred_label" not in df_predictions.columns:
        raise ValueError(
            "test_predictions.csv debe contener la columna 'pred_label'."
        )

    y_pred_saved = df_predictions["pred_label"].astype(int).to_numpy()
    texts = df_test["text"].astype(str).tolist()

    return texts, y_true, y_pred_saved, df_test, df_predictions


# ============================================================
# 3) SELECCION EXACTA DEL PAR ANTES DEL PRETRUNCADO
# ============================================================

def select_pair_indices_exactly_like_metrics(
    y_true: np.ndarray,
    class_j_id: int,
    class_r_id: int,
    n_each: int,
    seed: int,
) -> np.ndarray:
    """
    Replica exactamente la seleccion de SHAPTextLCDMetrics:
      rng=default_rng(seed), choice Cj, choice Cr, concatenate, sort.
    """
    indices_j = np.where(y_true == class_j_id)[0]
    indices_r = np.where(y_true == class_r_id)[0]

    if len(indices_j) < n_each or len(indices_r) < n_each:
        raise ValueError(
            "No hay suficientes instancias para construir la muestra balanceada. "
            f"Disponibles: Cj={len(indices_j)}, Cr={len(indices_r)}, "
            f"requeridas={n_each} por clase."
        )

    rng = np.random.default_rng(seed)
    selected_j = rng.choice(indices_j, size=n_each, replace=False)
    selected_r = rng.choice(indices_r, size=n_each, replace=False)
    selected_indices = np.concatenate([selected_j, selected_r]).astype(int)
    selected_indices = np.sort(selected_indices)
    return selected_indices


# ============================================================
# 4) PRETRUNCADO EXPLICITO - SOLO DE LAS 50 INSTANCIAS
# ============================================================

def prepare_texts_for_shap(
    texts: Sequence[str],
    tokenizer,
    max_length: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Misma logica de pretruncado del main SHAP-GLCD global."""
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
                "El texto preparado para SHAP quedo vacio en la posicion "
                f"{text_index}."
            )

        verification = tokenizer(
            prepared_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )

        if len(verification["input_ids"]) > max_length:
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

    print(
        f"\nPreparacion explicita de los {N_INSTANCES} "
        "textos seleccionados para SHAP:"
    )
    print(f"  Textos procesados: {len(prepared_texts)}")
    print(f"  Textos recortados: {int(flags_array.sum())}")
    print(f"  Longitud maxima permitida: {max_length} tokens")
    if len(counts_array) > 0:
        print(
            "  Longitud original maxima observada en la muestra: "
            f"{int(counts_array.max())} tokens"
        )

    return prepared_texts, counts_array, flags_array


# ============================================================
# 5) CACHE EXACTA PARA LAS PREDICCIONES INTERNAS DE SHAP
# ============================================================

class ExactTextPredictionCache:
    """
    Cache LRU exacta: texto -> vector de probabilidades.

    No aproxima ni interpola. Si el texto no esta en cache, llama a la misma
    funcion predict_proba de SHAPTextExplainerLCD. Si ya fue evaluado, devuelve
    exactamente el vector guardado.
    """

    def __init__(self, predict_fn, max_entries=30000):
        self.predict_fn = predict_fn
        self.max_entries = int(max_entries)
        self.cache = OrderedDict()
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0
        self.instance_hits = 0
        self.instance_misses = 0
        self.instance_evictions = 0

    def reset_instance(self):
        # No tiene utilidad conservar cache entre documentos diferentes.
        # Vaciarla limita memoria y concentra la cache en la instancia actual.
        self.cache.clear()
        self.instance_hits = 0
        self.instance_misses = 0
        self.instance_evictions = 0

    def _put(self, key, value):
        self.cache[key] = np.asarray(value).copy()
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
            self.total_evictions += 1
            self.instance_evictions += 1

    def __call__(self, texts):
        if isinstance(texts, str):
            text_list = [texts]
        else:
            # SHAP normalmente entrega ndarray/list de strings.
            text_list = [str(x) for x in list(texts)]

        outputs = [None] * len(text_list)
        missing_keys = []
        missing_positions = {}

        for position, text in enumerate(text_list):
            key = str(text)
            if key in self.cache:
                self.total_hits += 1
                self.instance_hits += 1
                self.cache.move_to_end(key)
                outputs[position] = self.cache[key]
            else:
                self.total_misses += 1
                self.instance_misses += 1
                if key not in missing_positions:
                    missing_positions[key] = []
                    missing_keys.append(key)
                missing_positions[key].append(position)

        if missing_keys:
            missing_probabilities = np.asarray(self.predict_fn(missing_keys))
            if missing_probabilities.shape[0] != len(missing_keys):
                raise RuntimeError(
                    "La funcion predict_proba devolvio un numero inesperado "
                    "de filas durante la cache SHAP."
                )

            for key, probabilities in zip(missing_keys, missing_probabilities):
                self._put(key, probabilities)
                for position in missing_positions[key]:
                    outputs[position] = self.cache[key]

        return np.vstack(outputs)

    def instance_stats(self):
        requests = self.instance_hits + self.instance_misses
        hit_rate = (
            self.instance_hits / requests
            if requests > 0
            else 0.0
        )
        return {
            "cache_hits": int(self.instance_hits),
            "cache_misses": int(self.instance_misses),
            "cache_hit_rate": float(hit_rate),
            "cache_evictions": int(self.instance_evictions),
        }


# ============================================================
# 6) FUNCIONES DE SALIDA
# ============================================================

def print_configuration():
    print("\n" + "=" * 76)
    print(" 20 NEWSGROUPS - STABILITY / CONSISTENCY SHAP-GLCD - OPTIMIZADO")
    print("=" * 76)
    print("Modelo                 :", MODEL_DISPLAY_NAME)
    print("Par                    :", f"{CLASS_J_NAME} vs {CLASS_R_NAME}")
    print("Seleccion              : etiqueta verdadera")
    print("Filtro por aciertos    : NO")
    print("N                      :", N_INSTANCES)
    print("Balance                :", f"{N_EACH} + {N_EACH}")
    print("M                      :", M_PERTURBATIONS)
    print("Perturbation rate      :", PERTURBATION_RATE)
    print("Tau                    :", TAU)
    print("Top-K                  :", TOP_K)
    print("Normalizacion          :", NORMALIZE.upper())
    print("Seed                   :", SEED)
    print("Seed por instancia     : 42+t, t=0,...,N-1")
    print("MAX_LENGTH             :", MAX_LENGTH)
    print(f"Pretruncado            : SOLO las {N_INSTANCES} seleccionadas")
    print("Perturbacion           : token masking")
    print("Alineacion             : estricta")
    print("Partition max_evals    : SIN CAMBIOS (default de la clase/SHAP)")
    print("Batch predict_proba    :", MODEL_PRED_BATCH_SIZE)
    print("Cache exacta SHAP      :", ENABLE_EXACT_SHAP_PREDICTION_CACHE)
    print("Checkpoint             : despues de cada instancia")
    print("Dispositivo            :", device)
    print("=" * 76)


def save_run_configuration(
    selected_indices: np.ndarray,
    original_token_counts: np.ndarray,
    truncated_flags: np.ndarray,
):
    configuration = pd.DataFrame([
        {
            "seed": SEED,
            "dataset": DATASET_NAME,
            "model": MODEL_DISPLAY_NAME,
            "model_dir": MODEL_DIR,
            "train_results_dir": TRAIN_RESULTS_DIR,
            "class_j": CLASS_J_NAME,
            "class_r": CLASS_R_NAME,
            "filter_mode": FILTER_MODE,
            "balanced_sampling": BALANCED_SAMPLING,
            "N_instances": N_INSTANCES,
            "N_each_class": N_EACH,
            "M_perturbations": M_PERTURBATIONS,
            "perturbation_rate": PERTURBATION_RATE,
            "tau": TAU,
            "top_k": TOP_K,
            "normalize": NORMALIZE,
            "instance_seed_rule": "seed+t",
            "max_length": MAX_LENGTH,
            "pretruncate_texts_for_shap": PRETRUNCATE_TEXTS_FOR_SHAP,
            "pretruncate_scope": "selected_50_only",
            "n_selected_truncated": int(truncated_flags.sum()),
            "max_original_tokens_selected": int(original_token_counts.max()),
            "model_predict_batch_size": MODEL_PRED_BATCH_SIZE,
            "exact_shap_prediction_cache": ENABLE_EXACT_SHAP_PREDICTION_CACHE,
            "cache_max_entries": CACHE_MAX_ENTRIES,
            "partition_max_evals": "unchanged_default",
            "perturbation": "SHAP text masker token masking",
            "token_alignment": "strict_shap_masker_exact_ids",
            "selected_indices_file": os.path.basename(SELECTED_INDICES_PATH),
        }
    ])
    configuration.to_csv(RUN_CONFIGURATION_PATH, index=False, encoding="utf-8")


def validate_or_load_checkpoint(selected_indices: np.ndarray) -> List[dict]:
    if not os.path.isfile(CHECKPOINT_PATH):
        return []

    checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
    if checkpoint_df.empty:
        return []

    required = {"sample_position", "test_index", "instance_seed"}
    missing = required.difference(checkpoint_df.columns)
    if missing:
        raise RuntimeError(
            "El checkpoint existente no tiene las columnas requeridas: "
            f"{sorted(missing)}. Eliminalo o cambiale el nombre antes de continuar."
        )

    checkpoint_df = checkpoint_df.sort_values("sample_position").reset_index(drop=True)

    expected_positions = np.arange(len(checkpoint_df), dtype=int)
    actual_positions = checkpoint_df["sample_position"].astype(int).to_numpy()
    if not np.array_equal(expected_positions, actual_positions):
        raise RuntimeError(
            "El checkpoint no corresponde a un prefijo continuo de la muestra actual."
        )

    expected_indices = selected_indices[:len(checkpoint_df)].astype(int)
    actual_indices = checkpoint_df["test_index"].astype(int).to_numpy()
    if not np.array_equal(expected_indices, actual_indices):
        raise RuntimeError(
            "El checkpoint pertenece a otra seleccion de instancias. "
            "No se reanudara para evitar mezclar corridas."
        )

    expected_seeds = SEED + expected_positions
    actual_seeds = checkpoint_df["instance_seed"].astype(int).to_numpy()
    if not np.array_equal(expected_seeds, actual_seeds):
        raise RuntimeError(
            "El checkpoint usa una regla de semillas distinta de seed+t."
        )

    print(
        f"\nCheckpoint encontrado: {len(checkpoint_df)}/{N_INSTANCES} "
        "instancias ya terminadas."
    )
    print("La corrida continuara desde la siguiente instancia.")
    return checkpoint_df.to_dict(orient="records")


def save_checkpoint(rows: List[dict]):
    pd.DataFrame(rows).to_csv(
        CHECKPOINT_PATH,
        index=False,
        encoding="utf-8",
    )


def build_summary(instances_df: pd.DataFrame) -> pd.DataFrame:
    stability_values = instances_df["stability"].to_numpy(dtype=float)
    consistency_values = instances_df["consistency"].to_numpy(dtype=float)
    coverage_values = instances_df["coverage_imp"].to_numpy(dtype=float)
    predkeep_values = instances_df["prediction_keep_imp"].to_numpy(dtype=float)

    n = len(instances_df)
    stability_std = float(np.std(stability_values, ddof=1)) if n > 1 else 0.0
    consistency_std = float(np.std(consistency_values, ddof=1)) if n > 1 else 0.0

    n_correct = int(instances_df["prediction_correct"].sum())
    n_incorrect = int(n - n_correct)

    summary = {
        "class_j": CLASS_J_NAME,
        "class_r": CLASS_R_NAME,
        "pair": f"{CLASS_J_NAME} vs {CLASS_R_NAME}",
        "filter_mode": FILTER_MODE,
        "sampling_seed": SEED,
        "N_instances": n,
        "Bal.": f"{N_EACH}/{N_EACH}",
        "M_perturbations": M_PERTURBATIONS,
        "perturbation_rate": PERTURBATION_RATE,
        "tau": TAU,
        "top_k": TOP_K,
        "normalize": NORMALIZE,
        "stability_mean": float(np.mean(stability_values)),
        "stability_std": stability_std,
        "stability_se": float(stability_std / np.sqrt(n)),
        "stability_median": float(np.median(stability_values)),
        "consistency_mean": float(np.mean(consistency_values)),
        "consistency_std": consistency_std,
        "consistency_se": float(consistency_std / np.sqrt(n)),
        "coverage_imp_mean": float(np.mean(coverage_values)),
        "prediction_keep_imp_mean": float(np.mean(predkeep_values)),
        "cosine_mean_avg": float(instances_df["cosine_mean"].mean()),
        "cosine_std_avg": float(instances_df["cosine_std"].mean()),
        "N_correct": n_correct,
        "N_incorrect": n_incorrect,
        "token_alignment": "strict_shap_masker_exact_ids",
        "base_vs_saved_prediction_agreement": float(
            instances_df["base_vs_saved_prediction_match"].mean()
        ),
        "cache_hit_rate_mean": float(instances_df["cache_hit_rate"].mean()),
    }
    return pd.DataFrame([summary])


def build_article_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    row = summary_df.iloc[0]
    return pd.DataFrame([
        {
            "Pair (Cj, Cr)": row["pair"],
            "N": int(row["N_instances"]),
            "Bal.": row["Bal."],
            "S_bar ± sS": (
                f"{row['stability_mean']:.3f} ± {row['stability_std']:.3f}"
            ),
            "Cimp_bar ± sC": (
                f"{row['consistency_mean']:.3f} ± {row['consistency_std']:.3f}"
            ),
            "Cov_imp / PredKeep_imp": (
                f"{row['coverage_imp_mean']:.3f} / "
                f"{row['prediction_keep_imp_mean']:.3f}"
            ),
        }
    ])


# ============================================================
# 7) MOSTRAR CONFIGURACION
# ============================================================

print_configuration()


# ============================================================
# 8) VERIFICAR Y CARGAR ARTEFACTOS
# ============================================================

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        "No se encontro el modelo ya entrenado en:\n"
        f"{os.path.abspath(MODEL_DIR)}"
    )

label_mapping_path = os.path.join(TRAIN_RESULTS_DIR, "label_mapping.csv")
test_split_path = os.path.join(TRAIN_RESULTS_DIR, "test_split.csv")
predictions_path = os.path.join(TRAIN_RESULTS_DIR, "test_predictions.csv")

for required_path in (label_mapping_path, test_split_path, predictions_path):
    if not os.path.isfile(required_path):
        raise FileNotFoundError(
            f"No se encontro el archivo requerido: {os.path.abspath(required_path)}"
        )

print("\nCargando mapeo de clases y test guardado...")
CLASS_NAMES, LABEL_TO_ID = load_class_names(label_mapping_path)

for required_class in (CLASS_J_NAME, CLASS_R_NAME):
    if required_class not in LABEL_TO_ID:
        raise ValueError(
            f"La clase requerida '{required_class}' no aparece en label_mapping.csv."
        )

texts_test, y_test, y_pred_saved, df_test, df_predictions = load_saved_test(
    test_split_path=test_split_path,
    predictions_path=predictions_path,
)
_ = df_test
_ = df_predictions

class_j_id = LABEL_TO_ID[CLASS_J_NAME]
class_r_id = LABEL_TO_ID[CLASS_R_NAME]

print("✓ Test guardado cargado.")
print("Numero total de textos en test:", len(texts_test))
print("Disponibilidad por etiqueta verdadera para el par:")
print(f"  {CLASS_J_NAME}: {int(np.sum(y_test == class_j_id))}")
print(f"  {CLASS_R_NAME}: {int(np.sum(y_test == class_r_id))}")


# ============================================================
# 9) SELECCIONAR EXACTAMENTE LAS N INSTANCIAS
# ============================================================

selected_indices = select_pair_indices_exactly_like_metrics(
    y_true=y_test,
    class_j_id=class_j_id,
    class_r_id=class_r_id,
    n_each=N_EACH,
    seed=SEED,
)
np.save(SELECTED_INDICES_PATH, selected_indices)

selected_texts_raw = [texts_test[int(i)] for i in selected_indices]
selected_y_true = y_test[selected_indices].astype(int)
selected_y_pred_saved = y_pred_saved[selected_indices].astype(int)

print("\nMuestra seleccionada ANTES de pretruncar:")
print(f"  N total: {len(selected_indices)}")
print(f"  {CLASS_J_NAME}: {int(np.sum(selected_y_true == class_j_id))}")
print(f"  {CLASS_R_NAME}: {int(np.sum(selected_y_true == class_r_id))}")
print("  Seleccion: etiqueta verdadera, sin filtro por aciertos")
print("  Semilla: 42")


# ============================================================
# 10) CARGAR MODELO Y TOKENIZADOR
# ============================================================

print("\nCargando modelo desde:")
print(os.path.abspath(MODEL_DIR))

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.model_max_length = MAX_LENGTH
model.to(device)
model.eval()

print("✓ Modelo y tokenizador cargados.")


# ============================================================
# 11) PRETRUNCAR SOLO LAS INSATANCIAS SELECCIONADAS
# ============================================================

selected_texts_for_shap, original_token_counts, truncated_flags = (
    prepare_texts_for_shap(
        texts=selected_texts_raw,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
)

save_run_configuration(
    selected_indices=selected_indices,
    original_token_counts=original_token_counts,
    truncated_flags=truncated_flags,
)


# ============================================================
# 12) INICIALIZAR EL MISMO EXPLICADOR Y LA MISMA CLASE DE METRICAS
# ============================================================

print("\nInicializando SHAP para texto...")

lcd_explainer = SHAPTextExplainerLCD(
    model=model,
    tokenizer=tokenizer,
    texts_eval=selected_texts_for_shap,
    y_eval=selected_y_true,
    class_names=CLASS_NAMES,
    max_length=MAX_LENGTH,
    batch_size=MODEL_PRED_BATCH_SIZE,
    device=device,
    random_state=SEED,
)

metrics_evaluator = SHAPTextLCDMetrics(lcd_explainer=lcd_explainer)


# ============================================================
# 13) ACTIVAR CACHE EXACTA EN EL MODELO INTERNO DE SHAP
# ============================================================

prediction_cache = None

if ENABLE_EXACT_SHAP_PREDICTION_CACHE:
    shap_model = getattr(lcd_explainer.explainer, "model", None)
    if shap_model is None or not hasattr(shap_model, "inner_model"):
        raise RuntimeError(
            "No fue posible acceder al modelo interno de SHAP para activar "
            "la cache exacta."
        )

    original_shap_predict_fn = shap_model.inner_model
    prediction_cache = ExactTextPredictionCache(
        predict_fn=original_shap_predict_fn,
        max_entries=CACHE_MAX_ENTRIES,
    )
    shap_model.inner_model = prediction_cache
    print("✓ Cache exacta de predicciones SHAP activada.")


# ============================================================
# 14) EVALUAR INSTANCIA POR INSTANCIA + CHECKPOINT/RESUME
# ============================================================

rows = validate_or_load_checkpoint(selected_indices)
start_t = len(rows)

if start_t >= N_INSTANCES:
    print("\nEl checkpoint ya contiene las 50 instancias. Se reutilizara.")
else:
    print("\nCalculando Stability / Consistency...")

for t in range(start_t, N_INSTANCES):
    original_test_index = int(selected_indices[t])
    instance_seed = int(SEED + t)

    print("\n" + "=" * 76)
    print(
        f"Instancia {t + 1}/{N_INSTANCES} | indice test={original_test_index} "
        f"| instance_seed={instance_seed}"
    )
    print("=" * 76)

    if prediction_cache is not None:
        prediction_cache.reset_instance()

    start_time = time.time()

    instance_metrics = metrics_evaluator.evaluate_instance_metrics(
        text=selected_texts_for_shap[t],
        class_j=class_j_id,
        class_r=class_r_id,
        M=M_PERTURBATIONS,
        perturbation_rate=PERTURBATION_RATE,
        tau=TAU,
        top_k=TOP_K,
        normalize=NORMALIZE,
        seed=instance_seed,
    )

    elapsed_seconds = float(time.time() - start_time)

    true_label = int(selected_y_true[t])
    base_prediction = int(instance_metrics["base_prediction"])
    saved_prediction = int(selected_y_pred_saved[t])

    if prediction_cache is not None:
        cache_stats = prediction_cache.instance_stats()
    else:
        cache_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "cache_evictions": 0,
        }

    row = {
        "sample_position": int(t),
        "test_index": original_test_index,
        "true_label": true_label,
        "true_label_name": CLASS_NAMES[true_label],
        "class_j": CLASS_J_NAME,
        "class_r": CLASS_R_NAME,
        "pair": f"{CLASS_J_NAME} vs {CLASS_R_NAME}",
        "filter_mode": FILTER_MODE,
        "sampling_seed": SEED,
        "instance_seed": instance_seed,
        "prediction_correct": int(base_prediction == true_label),
        "saved_test_prediction": saved_prediction,
        "base_vs_saved_prediction_match": int(base_prediction == saved_prediction),
        "original_token_count": int(original_token_counts[t]),
        "was_truncated_for_shap": int(truncated_flags[t]),
        "elapsed_seconds": elapsed_seconds,
        **cache_stats,
        **instance_metrics,
    }

    rows.append(row)
    save_checkpoint(rows)

    print(
        "Resultado instancia: "
        f"S={row['stability']:.6f} | "
        f"Cimp={row['consistency']:.6f} | "
        f"Cov={row['coverage_imp']:.6f} | "
        f"PredKeep={row['prediction_keep_imp']:.6f}"
    )
    print(
        f"Tiempo: {elapsed_seconds / 60.0:.2f} min | "
        f"cache hit rate: {row['cache_hit_rate']:.3f} "
        f"({row['cache_hits']} hits / "
        f"{row['cache_hits'] + row['cache_misses']} solicitudes)"
    )
    print(f"Checkpoint actualizado: {CHECKPOINT_PATH}")


# ============================================================
# 15) RESUMEN FINAL Y ARCHIVOS
# ============================================================

instances_df = pd.DataFrame(rows)
instances_df = instances_df.sort_values("sample_position").reset_index(drop=True)

if len(instances_df) != N_INSTANCES:
    raise RuntimeError(
        f"Se esperaban {N_INSTANCES} instancias y hay {len(instances_df)}."
    )

summary_df = build_summary(instances_df)
article_table_df = build_article_table(summary_df)

safe_j = safe_filename(CLASS_J_NAME)
safe_r = safe_filename(CLASS_R_NAME)

pair_summary_path = os.path.join(
    METRICS_DIR,
    f"metrics_summary_{safe_j}_vs_{safe_r}.csv",
)
pair_instances_path = os.path.join(
    METRICS_DIR,
    f"metrics_instances_{safe_j}_vs_{safe_r}.csv",
)
summary_all_path = os.path.join(METRICS_DIR, "metrics_summary_all_pairs.csv")
instances_all_path = os.path.join(METRICS_DIR, "metrics_instances_all_pairs.csv")
table_path = os.path.join(METRICS_DIR, "metrics_table_format_all_pairs.csv")

summary_df.to_csv(pair_summary_path, index=False, encoding="utf-8")
instances_df.to_csv(pair_instances_path, index=False, encoding="utf-8")
summary_df.to_csv(summary_all_path, index=False, encoding="utf-8")
instances_df.to_csv(instances_all_path, index=False, encoding="utf-8")
article_table_df.to_csv(table_path, index=False, encoding="utf-8")

row = summary_df.iloc[0]

print("\n" + "=" * 104)
print(" RESULTADOS - STABILITY / CONSISTENCY DE EXPLICACIONES SHAP-GLCD")
print("=" * 104)
print(f"Par                     : {row['pair']}")
print(f"N / balance             : {int(row['N_instances'])} / {row['Bal.']}")
print(
    "S_bar ± sS              : "
    f"{row['stability_mean']:.4f} ± {row['stability_std']:.4f}"
)
print(f"Mediana S               : {row['stability_median']:.4f}")
print(
    "Cimp_bar ± sC           : "
    f"{row['consistency_mean']:.4f} ± {row['consistency_std']:.4f}"
)
print(
    "Cov_imp / PredKeep_imp  : "
    f"{row['coverage_imp_mean']:.4f} / "
    f"{row['prediction_keep_imp_mean']:.4f}"
)
print(f"Predicciones correctas  : {int(row['N_correct'])}")
print(f"Predicciones incorrectas: {int(row['N_incorrect'])}")
print(
    "Acuerdo pred. guardada  : "
    f"{row['base_vs_saved_prediction_agreement']:.4f}"
)
print(f"Cache hit rate promedio : {row['cache_hit_rate_mean']:.4f}")
print("Alineacion tokenizada   :", row["token_alignment"])
print("=" * 104)

print("\nArchivos guardados:")
print("Resumen:", os.path.abspath(summary_all_path))
print("Instancias:", os.path.abspath(instances_all_path))
print("Tabla articulo:", os.path.abspath(table_path))
print("Checkpoint:", os.path.abspath(CHECKPOINT_PATH))
print("Configuracion:", os.path.abspath(RUN_CONFIGURATION_PATH))
print("\nProceso finalizado.")
