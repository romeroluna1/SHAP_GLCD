# ============================================================
# main_stability_consistency_fashion_mnist_070926.py
# ============================================================
#
# STABILITY + CONSISTENCY DE LAS EXPLICACIONES DIFERENCIALES
# QUE SIRVEN DE BASE A SHAP-GLCD
#
# Dataset:
#   Fashion-MNIST
#
# Modelo:
#   CNN definitiva del experimento:
#   main_fashion_glcd_v2_CORREGIDO_TRAIN_VAL_TEST_020926.py
#
# Par:
#   T-shirt/Top vs Trouser
#
# ============================================================
# PRINCIPIOS
# ============================================================
#
# - NO reentrena la CNN.
# - NO recalcula los SHAP globales originales.
# - Carga:
#
#     fashion_cnn_best_corrected_train_val_test.keras
#
# - Carga:
#
#     shap_values_fashion_corrected_0_vs_1.pickle
#
# - Usa como explicación base EXACTAMENTE el SHAP almacenado
#   que sirvió para SHAP-GLCD.
#
# - Los SHAP de las perturbaciones se calculan con
#   GradientExplainer y el MISMO background reconstruido.
#
# ============================================================
# PROTOCOLO
# ============================================================
#
# N = 50 = 25 T-shirt/Top + 25 Trouser
# selección por etiqueta verdadera
# sin filtro por predicción
#
# M = 40 perturbaciones por instancia
# sigma = 0.005
# tau = 0.95
# top-k = 10
# normalización L2
# seed = 42
#
# ============================================================
# PERTURBACIONES DE IMAGEN
# ============================================================
#
# Las imágenes ya están normalizadas a [0,1].
#
# Cada píxel se considera una variable numérica continua:
#
#   x'_p = x_p + N(0, sigma^2)
#
# seguido de:
#
#   clip(x'_p, 0, 1)
#
# No se generan valores fuera del dominio de entrada del modelo.
#
# ============================================================


import os
import sys
import json
import time
import pickle
import random
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

SEED = 42

VALIDATION_SIZE = 0.10

BATCH_SIZE = 128

J_SEL = 0
R_SEL = 1

# Configuración definitiva utilizada en el artículo.
N_INSTANCES = 50
M_PERTURB = 40

SIGMA = 0.005
TAU = 0.95
TOP_K = 10

SHAP_BACKGROUND_SIZE = 100


CLASS_NAMES = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


# ============================================================
# 2. RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "fashion_mnist"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "fashion_cnn_best_corrected_train_val_test.keras"
)

SHAP_PICKLE_PATH = (
    ARTIFACT_DIR
    / "shap_values_fashion_corrected_0_vs_1.pickle"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "fashion_mnist"
    / "pair_0_vs_1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 3. DETERMINISMO
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("✅ Deterministic ops enabled")
except Exception as e:
    print(
        "⚠️ Deterministic ops not available:",
        e
    )


# ============================================================
# 4. HELPERS
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    denominator = (
        (np.linalg.norm(a) + 1e-12)
        *
        (np.linalg.norm(b) + 1e-12)
    )

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================

def normalize_l2(v):

    v = np.asarray(
        v,
        dtype=float
    )

    return (
        v
        /
        (
            np.linalg.norm(v)
            +
            1e-12
        )
    )


# ============================================================

def top_k_sparse(v, top_k=10):

    v = np.asarray(
        v,
        dtype=float
    )

    if (
        top_k is None
        or
        top_k >= len(v)
    ):
        return v.copy()

    idx = np.argsort(
        np.abs(v)
    )[::-1][:top_k]

    result = np.zeros_like(v)

    result[idx] = v[idx]

    return result


# ============================================================
# 5. CONVERSIÓN SHAP EXACTAMENTE COMO LA CLASE ORIGINAL
# ============================================================

def to_shap_array_npk(shap_values):
    """
    Convierte SHAP al formato:

        (N, P, K)

    donde:
        N = instancias
        P = píxeles = 784
        K = clases = 10

    Compatible con los formatos manejados por
    glcd_image_explainer2.py.
    """

    if isinstance(shap_values, list):

        K = len(shap_values)

        sv0 = np.asarray(
            shap_values[0]
        )

        N = sv0.shape[0]

        matrices = []

        for k in range(K):

            svk = np.asarray(
                shap_values[k]
            )

            if svk.ndim == 4:

                matrices.append(
                    svk.reshape(
                        N,
                        -1
                    )
                )

            elif svk.ndim == 2:

                matrices.append(
                    svk
                )

            else:

                raise ValueError(
                    f"Formato SHAP inesperado "
                    f"para clase {k}: "
                    f"{svk.shape}"
                )

        return np.stack(
            matrices,
            axis=2
        )

    arr = np.asarray(
        shap_values
    )

    if arr.ndim == 5:

        return arr.reshape(
            arr.shape[0],
            -1,
            arr.shape[-1]
        )

    if arr.ndim == 3:

        return arr

    raise ValueError(
        "Formato inesperado de SHAP: "
        f"{arr.shape}"
    )


# ============================================================
# 6. EXPLICACIÓN DIFERENCIAL
# ============================================================

def delta_explanation(
    shap_instance,
    class_j,
    class_r,
    top_k=10
):
    """
    ΔS = SHAP(Cj) - SHAP(Cr)

    Después:
        top-k por |ΔS|
        normalización L2
    """

    shap_instance = np.asarray(
        shap_instance,
        dtype=float
    )

    delta = (
        shap_instance[:, class_j]
        -
        shap_instance[:, class_r]
    )

    delta = top_k_sparse(
        delta,
        top_k=top_k
    )

    return normalize_l2(
        delta
    )


# ============================================================
# 7. CONVERTIR MUCHAS EXPLICACIONES
# ============================================================

def build_delta_matrix(
    shap_arr,
    class_j,
    class_r,
    top_k=10
):

    shap_arr = np.asarray(
        shap_arr,
        dtype=float
    )

    raw_delta = (
        shap_arr[:, :, class_j]
        -
        shap_arr[:, :, class_r]
    )

    result = np.zeros_like(
        raw_delta,
        dtype=float
    )

    for i in range(
        len(raw_delta)
    ):

        v = top_k_sparse(
            raw_delta[i],
            top_k=top_k
        )

        result[i] = normalize_l2(
            v
        )

    return result


# ============================================================
# 8. MUESTREO BALANCEADO
# ============================================================

def sample_pair_indices(
    y_true,
    class_j,
    class_r,
    n_instances=50,
    seed=42
):

    if n_instances % 2 != 0:

        raise ValueError(
            "N_INSTANCES debe ser par."
        )

    y_true = np.asarray(
        y_true
    )

    rng = np.random.default_rng(
        seed
    )

    idx_j = np.where(
        y_true == class_j
    )[0]

    idx_r = np.where(
        y_true == class_r
    )[0]

    n_each = (
        n_instances
        //
        2
    )

    if (
        len(idx_j) < n_each
        or
        len(idx_r) < n_each
    ):
        raise RuntimeError(
            "No hay suficientes imágenes para seleccionar "
            f"{n_each} instancias de cada clase."
        )

    rng.shuffle(
        idx_j
    )

    rng.shuffle(
        idx_r
    )

    selected = np.concatenate(
        [
            idx_j[:n_each],
            idx_r[:n_each],
        ]
    )

    rng.shuffle(
        selected
    )

    return selected


# ============================================================
# 9. RECONSTRUIR BACKGROUND E idx_eval ORIGINALES
# ============================================================

def reconstruct_original_shap_sampling(
    x_train,
    y_test
):
    """
    Reproduce el orden exacto utilizado por
    glcd_image_explainer2.compute_shap():

      1. np.random.seed(42)
      2. seleccionar background del train
      3. seleccionar imágenes de Cj
      4. seleccionar imágenes de Cr
      5. mezclar idx_eval

    En Fashion-MNIST se usó max_test=10000.
    """

    np.random.seed(
        SEED
    )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    idx_background = np.random.choice(
        len(x_train),
        size=SHAP_BACKGROUND_SIZE,
        replace=False
    )

    # --------------------------------------------------------
    # Selección original de SHAP
    # --------------------------------------------------------

    max_test = len(
        y_test
    )

    idx_j = np.where(
        y_test == J_SEL
    )[0]

    idx_r = np.where(
        y_test == R_SEL
    )[0]

    half = (
        max_test
        //
        2
    )

    n_j = min(
        len(idx_j),
        half
    )

    n_r = min(
        len(idx_r),
        max_test - n_j
    )

    if n_j + n_r < max_test:

        missing = (
            max_test
            -
            (n_j + n_r)
        )

        rem_j = (
            len(idx_j)
            -
            n_j
        )

        rem_r = (
            len(idx_r)
            -
            n_r
        )

        if rem_j >= rem_r:

            extra_j = min(
                rem_j,
                missing
            )

        else:

            extra_j = min(
                rem_j,
                max(
                    0,
                    missing
                    -
                    min(
                        rem_r,
                        missing
                    )
                )
            )

        extra_r = min(
            rem_r,
            missing - extra_j
        )

        n_j += extra_j
        n_r += extra_r

    sel_j = np.random.choice(
        idx_j,
        size=n_j,
        replace=False
    )

    sel_r = np.random.choice(
        idx_r,
        size=n_r,
        replace=False
    )

    idx_eval = np.concatenate(
        [
            sel_j,
            sel_r
        ]
    )

    np.random.shuffle(
        idx_eval
    )

    # Estado NumPy justo antes de la llamada SHAP original.
    state_before_shap = (
        np.random.get_state()
    )

    return (
        idx_background,
        idx_eval,
        state_before_shap
    )


# ============================================================
# 10. PERTURBACIONES DE IMAGEN
# ============================================================

def perturb_image(
    base_image,
    rng,
    sigma=0.005
):
    """
    Píxeles continuos normalizados en [0,1].

    x' = x + N(0, sigma^2)
    clip [0,1]
    """

    base_image = np.asarray(
        base_image,
        dtype=np.float32
    )

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=base_image.shape
    ).astype(
        np.float32
    )

    perturbed = (
        base_image
        +
        noise
    )

    perturbed = np.clip(
        perturbed,
        0.0,
        1.0
    )

    return perturbed.astype(
        np.float32
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    start_time = time.time()

    # ========================================================
    # INFORMACIÓN
    # ========================================================

    print(
        "Python:",
        sys.version
    )

    print(
        "NumPy:",
        np.__version__
    )

    print(
        "TensorFlow:",
        tf.__version__
    )

    print(
        "SHAP:",
        shap.__version__
    )

    print(
        "pandas:",
        pd.__version__
    )

    print(
        "OS:",
        platform.platform()
    )

    print(
        "Processor:",
        platform.processor()
    )


    print(
        "\n============================================================"
    )

    print(
        " FASHION-MNIST - STABILITY / CONSISTENCY SHAP-GLCD"
    )

    print(
        "============================================================"
    )

    print(
        "Dataset           : Fashion-MNIST"
    )

    print(
        "Modelo            : CNN"
    )

    print(
        "Par               : "
        "T-shirt/Top vs Trouser"
    )

    print(
        "Selección         : true label"
    )

    print(
        "Filtro predicción : NO"
    )

    print(
        "N                 :",
        N_INSTANCES
    )

    print(
        "Balance           : "
        f"{N_INSTANCES // 2} + {N_INSTANCES // 2}"
    )

    print(
        "M perturbaciones  :",
        M_PERTURB
    )

    print(
        "sigma             :",
        SIGMA
    )

    print(
        "tau               :",
        TAU
    )

    print(
        "top-k             :",
        TOP_K
    )

    print(
        "seed              :",
        SEED
    )

    print(
        "Perturbación      : Gaussian pixel noise + clip [0,1]"
    )

    print(
        "SHAP base         : pickle original SHAP-GLCD"
    )

    print(
        "SHAP perturbado   : GradientExplainer"
    )

    print(
        "============================================================"
    )


    # ========================================================
    # ARCHIVOS
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nNo se encontró el modelo definitivo:\n"
            f"{MODEL_PATH}"
        )

    if not SHAP_PICKLE_PATH.exists():

        raise FileNotFoundError(
            "\nNo se encontró el pickle SHAP definitivo:\n"
            f"{SHAP_PICKLE_PATH}"
        )


    print(
        "\nModelo encontrado:"
    )

    print(
        MODEL_PATH.resolve()
    )

    print(
        "\nSHAP pickle encontrado:"
    )

    print(
        SHAP_PICKLE_PATH.resolve()
    )


    # ========================================================
    # DATOS
    # ========================================================

    (
        (x_train_full, y_train_full),
        (x_test, y_test)
    ) = tf.keras.datasets.fashion_mnist.load_data()


    x_train_full = (
        x_train_full
        .astype("float32")
        /
        255.0
    )[..., np.newaxis]


    x_test = (
        x_test
        .astype("float32")
        /
        255.0
    )[..., np.newaxis]


    (
        x_train,
        x_val,
        y_train,
        y_val
    ) = train_test_split(

        x_train_full,
        y_train_full,

        test_size=VALIDATION_SIZE,

        stratify=y_train_full,

        random_state=SEED
    )


    print(
        "\n============================================================"
    )

    print(
        " DATASET"
    )

    print(
        "============================================================"
    )

    print(
        "Train      :",
        x_train.shape,
        y_train.shape
    )

    print(
        "Validation :",
        x_val.shape,
        y_val.shape
    )

    print(
        "Test       :",
        x_test.shape,
        y_test.shape
    )


    if x_train.shape != (
        54000,
        28,
        28,
        1
    ):

        raise RuntimeError(
            "Train inesperado."
        )

    if x_val.shape != (
        6000,
        28,
        28,
        1
    ):

        raise RuntimeError(
            "Validation inesperada."
        )

    if x_test.shape != (
        10000,
        28,
        28,
        1
    ):

        raise RuntimeError(
            "Test inesperado."
        )


    # ========================================================
    # MODELO
    # ========================================================

    print(
        "\nCargando CNN definitiva..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # ========================================================
    # MÉTRICAS TEST COMPLETO
    # ========================================================

    test_loss, keras_accuracy = (
        model.evaluate(
            x_test,
            y_test,
            verbose=0
        )
    )

    y_prob = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    y_pred = np.argmax(
        y_prob,
        axis=1
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred
        )
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    auc_ovo = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovo",
        average="macro"
    )


    print(
        "\n============================================================"
    )

    print(
        " MÉTRICAS DEL MODELO - TEST OFICIAL"
    )

    print(
        "============================================================"
    )

    print(
        f"Loss                : "
        f"{test_loss:.6f}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.6f}"
    )

    print(
        f"Balanced accuracy   : "
        f"{balanced_accuracy:.6f}"
    )

    print(
        f"F1 macro            : "
        f"{f1_macro:.6f}"
    )

    print(
        f"MCC                 : "
        f"{mcc:.6f}"
    )

    print(
        f"Macro-AUC (OvO)     : "
        f"{auc_ovo:.6f}"
    )


    expected = np.array([
        0.897500,
        0.897500,
        0.897207,
        0.886165,
        0.992785,
    ])

    observed = np.array([
        accuracy,
        balanced_accuracy,
        f1_macro,
        mcc,
        auc_ovo,
    ])


    max_metric_difference = float(
        np.max(
            np.abs(
                expected
                -
                observed
            )
        )
    )


    if max_metric_difference > 1e-5:

        raise RuntimeError(
            "\nEl modelo cargado NO reproduce "
            "el experimento definitivo.\n"
            f"Máxima diferencia: "
            f"{max_metric_difference:.12e}"
        )


    print(
        "\n✓ Modelo compatible con "
        "el experimento definitivo."
    )


    # ========================================================
    # CARGAR SHAP ORIGINAL
    # ========================================================

    print(
        "\nCargando SHAP original..."
    )

    with open(
        SHAP_PICKLE_PATH,
        "rb"
    ) as f:

        shap_object = pickle.load(
            f
        )


    if not isinstance(
        shap_object,
        dict
    ):

        raise RuntimeError(
            "\nEl pickle definitivo debería "
            "contener un diccionario."
        )


    if "shap_values" not in shap_object:

        raise RuntimeError(
            "El pickle no contiene "
            "'shap_values'."
        )


    if "idx_eval" not in shap_object:

        raise RuntimeError(
            "\nEl pickle no contiene idx_eval.\n"
            "No se puede garantizar alineación exacta."
        )


    stored_shap_raw = (
        shap_object[
            "shap_values"
        ]
    )

    stored_idx_eval = np.asarray(
        shap_object[
            "idx_eval"
        ],
        dtype=int
    )


    stored_shap = (
        to_shap_array_npk(
            stored_shap_raw
        )
    )


    print(
        "\n============================================================"
    )

    print(
        " SHAP ORIGINAL"
    )

    print(
        "============================================================"
    )

    print(
        "SHAP shape      :",
        stored_shap.shape
    )

    print(
        "idx_eval shape  :",
        stored_idx_eval.shape
    )


    if stored_shap.shape != (
        2000,
        784,
        10
    ):

        raise RuntimeError(
            "\nSHAP shape inesperado.\n"
            f"Encontrado: {stored_shap.shape}\n"
            "Esperado: (2000, 784, 10)"
        )


    if len(
        stored_idx_eval
    ) != 2000:

        raise RuntimeError(
            "idx_eval debería contener "
            "2000 posiciones."
        )


    y_eval = y_test[
        stored_idx_eval
    ]


    eval_counts = pd.Series(
        y_eval
    ).value_counts().sort_index()


    print(
        "\nDistribución y_eval:"
    )

    for cls, count in (
        eval_counts.items()
    ):

        print(
            f"{int(cls)} "
            f"({CLASS_NAMES[int(cls)]}): "
            f"{count}"
        )


    if (
        np.sum(y_eval == J_SEL) != 1000
        or
        np.sum(y_eval == R_SEL) != 1000
    ):

        raise RuntimeError(
            "El pickle SHAP no contiene "
            "1000 + 1000 imágenes del par."
        )


    # ========================================================
    # RECONSTRUIR BACKGROUND + idx_eval
    # ========================================================

    (
        idx_background,
        reconstructed_idx_eval,
        state_before_original_shap
    ) = reconstruct_original_shap_sampling(
        x_train=x_train,
        y_test=y_test
    )


    same_idx_eval = np.array_equal(
        stored_idx_eval,
        reconstructed_idx_eval
    )


    print(
        "\n============================================================"
    )

    print(
        " VERIFICACIÓN DEL MUESTREO SHAP ORIGINAL"
    )

    print(
        "============================================================"
    )

    print(
        "Background size:",
        len(idx_background)
    )

    print(
        "idx_eval almacenado == "
        "idx_eval reconstruido:",
        same_idx_eval
    )


    if not same_idx_eval:

        raise RuntimeError(
            "\nNo se pudo reproducir exactamente "
            "idx_eval del SHAP original.\n"
            "No continuaremos para evitar mezclar "
            "otro background/muestreo."
        )


    print(
        "\n✓ Muestreo SHAP original reproducido."
    )


    X_background = x_train[
        idx_background
    ]


    print(
        "Background shape:",
        X_background.shape
    )


    # ========================================================
    # CREAR GRADIENT EXPLAINER
    # ========================================================

    print(
        "\nConstruyendo GradientExplainer "
        "con el mismo background..."
    )

    gradient_explainer = shap.GradientExplainer(
        model,
        X_background
    )


    # ========================================================
    # CONTROL DE SHAP DE UNA IMAGEN ORIGINAL
    # ========================================================
    #
    # No recalculamos los 2000 SHAP.
    # Solo una imagen para comprobar formato/comportamiento.
    # ========================================================

    control_test_index = int(
        stored_idx_eval[0]
    )


    np.random.set_state(
        state_before_original_shap
    )


    control_raw = (
        gradient_explainer.shap_values(
            x_test[
                control_test_index:
                control_test_index + 1
            ]
        )
    )


    control_shap = (
        to_shap_array_npk(
            control_raw
        )[0]
    )


    stored_control = (
        stored_shap[0]
    )


    max_control_difference = float(
        np.max(
            np.abs(
                stored_control
                -
                control_shap
            )
        )
    )

    mean_control_difference = float(
        np.mean(
            np.abs(
                stored_control
                -
                control_shap
            )
        )
    )


    stored_control_delta = (
        delta_explanation(
            stored_control,
            J_SEL,
            R_SEL,
            TOP_K
        )
    )

    direct_control_delta = (
        delta_explanation(
            control_shap,
            J_SEL,
            R_SEL,
            TOP_K
        )
    )


    control_cosine = cosine_similarity(
        stored_control_delta,
        direct_control_delta
    )


    print(
        "\n============================================================"
    )

    print(
        " CONTROL SHAP ORIGINAL"
    )

    print(
        "============================================================"
    )

    print(
        "Test index:",
        control_test_index
    )

    print(
        f"Max |stored-direct|  : "
        f"{max_control_difference:.12e}"
    )

    print(
        f"Mean |stored-direct| : "
        f"{mean_control_difference:.12e}"
    )

    print(
        f"Cosine ΔS top-k      : "
        f"{control_cosine:.12f}"
    )

    print(
        "\nNOTA: GradientExplainer utiliza "
        "aproximación numérica/Monte Carlo."
    )

    print(
        "La explicación BASE de las métricas "
        "seguirá siendo siempre la almacenada "
        "en el pickle original."
    )


    # ========================================================
    # SELECCIÓN 50 = 25 + 25
    # ========================================================

    selected_indices = sample_pair_indices(

        y_true=y_test,

        class_j=J_SEL,

        class_r=R_SEL,

        n_instances=N_INSTANCES,

        seed=SEED
    )


    selected_true = y_test[
        selected_indices
    ]

    selected_pred = y_pred[
        selected_indices
    ]


    print(
        "\n============================================================"
    )

    print(
        " MUESTRA STABILITY / CONSISTENCY"
    )

    print(
        "============================================================"
    )

    print(
        "N seleccionado:",
        len(selected_indices)
    )

    print(
        "\nDistribución verdadera:"
    )

    print(
        pd.Series(
            selected_true
        )
        .map(
            CLASS_NAMES
        )
        .value_counts()
        .sort_index()
    )


    n_correct = int(
        np.sum(
            selected_true
            ==
            selected_pred
        )
    )

    n_errors = int(
        len(selected_indices)
        -
        n_correct
    )


    print(
        "\nPredicciones correctas :",
        n_correct
    )

    print(
        "Predicciones incorrectas:",
        n_errors
    )

    print(
        "Filtro predicción       : NO"
    )


    # ========================================================
    # MAPEAR ÍNDICE TEST -> FILA SHAP
    # ========================================================

    test_to_shap_row = {

        int(test_idx):
        int(shap_row)

        for shap_row, test_idx
        in enumerate(
            stored_idx_eval
        )
    }


    missing_from_shap = [

        int(idx)

        for idx in selected_indices

        if int(idx)
        not in test_to_shap_row
    ]


    if missing_from_shap:

        raise RuntimeError(
            "\nHay instancias seleccionadas que "
            "no aparecen en el SHAP original:\n"
            f"{missing_from_shap}"
        )


    selected_shap_rows = np.asarray(
        [
            test_to_shap_row[
                int(idx)
            ]
            for idx
            in selected_indices
        ],
        dtype=int
    )


    # ========================================================
    # EXPLICACIONES BASE ALMACENADAS
    # ========================================================

    base_shap = stored_shap[
        selected_shap_rows
    ]


    base_delta = build_delta_matrix(

        shap_arr=base_shap,

        class_j=J_SEL,

        class_r=R_SEL,

        top_k=TOP_K
    )


    # ========================================================
    # GENERAR N_INSTANCES * M_PERTURB PERTURBACIONES
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        " GENERANDO PERTURBACIONES"
    )

    print(
        "============================================================"
    )


    all_perturbations = []

    perturbation_owner = []

    perturbation_number = []

    perturbation_seeds = []

    perturbation_diagnostics = []


    for t, test_idx in enumerate(
        selected_indices
    ):

        # Regla común del protocolo SHAP-LCD:
        # seed_instancia = seed + t,
        # donde t es la posición local de la instancia seleccionada.
        instance_seed = (
            SEED
            +
            t
        )


        rng = np.random.default_rng(
            instance_seed
        )


        base_image = x_test[
            int(test_idx)
        ]


        abs_changes = []


        for m in range(
            M_PERTURB
        ):

            perturbed = perturb_image(

                base_image=base_image,

                rng=rng,

                sigma=SIGMA
            )


            abs_changes.append(
                float(
                    np.mean(
                        np.abs(
                            perturbed
                            -
                            base_image
                        )
                    )
                )
            )


            all_perturbations.append(
                perturbed
            )

            perturbation_owner.append(
                t
            )

            perturbation_number.append(
                m
            )

            perturbation_seeds.append(
                instance_seed
            )


        perturbation_diagnostics.append({

            "local_position":
            t,

            "test_index":
            int(test_idx),

            "seed":
            instance_seed,

            "mean_abs_pixel_change":
            float(
                np.mean(
                    abs_changes
                )
            ),
        })


    all_perturbations = np.asarray(
        all_perturbations,
        dtype=np.float32
    )


    print(
        "Perturbaciones generadas:",
        all_perturbations.shape
    )


    if all_perturbations.shape != (
        N_INSTANCES * M_PERTURB,
        28,
        28,
        1
    ):

        raise RuntimeError(
            "Shape inesperado para "
            "las perturbaciones."
        )


    print(
        "Mínimo:",
        float(
            all_perturbations.min()
        )
    )

    print(
        "Máximo:",
        float(
            all_perturbations.max()
        )
    )


    if (
        all_perturbations.min() < 0.0
        or
        all_perturbations.max() > 1.0
    ):

        raise RuntimeError(
            "Hay píxeles fuera de [0,1]."
        )


    # ========================================================
    # PREDICCIONES DE PERTURBACIONES
    # ========================================================

    print(
        "\nCalculando predicciones de "
        f"las {N_INSTANCES * M_PERTURB} perturbaciones..."
    )


    perturbed_probabilities = model.predict(

        all_perturbations,

        batch_size=BATCH_SIZE,

        verbose=0
    )


    perturbed_predictions = np.argmax(

        perturbed_probabilities,

        axis=1
    )


    # ========================================================
    # SHAP PERTURBADO
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        " CALCULANDO SHAP DE LAS PERTURBACIONES"
    )

    print(
        "============================================================"
    )

    print(
        "Número total:",
        len(all_perturbations)
    )

    print(
        "Explainer    : GradientExplainer"
    )

    print(
        "Background   :",
        X_background.shape
    )

    print(
        "\nEste paso puede tardar."
    )


    shap_start = time.time()


    # --------------------------------------------------------
    # Restauramos el estado NumPy que existía justo antes de
    # la llamada GradientExplainer del experimento original.
    # --------------------------------------------------------

    np.random.set_state(
        state_before_original_shap
    )


    perturbed_shap_raw = (
        gradient_explainer.shap_values(
            all_perturbations
        )
    )


    perturbed_shap = (
        to_shap_array_npk(
            perturbed_shap_raw
        )
    )


    shap_seconds = (
        time.time()
        -
        shap_start
    )


    print(
        "\nSHAP perturbado shape:",
        perturbed_shap.shape
    )

    print(
        f"Tiempo SHAP: "
        f"{shap_seconds / 60.0:.2f} minutos"
    )


    expected_perturbed_shape = (
        N_INSTANCES * M_PERTURB,
        784,
        10
    )


    if perturbed_shap.shape != (
        expected_perturbed_shape
    ):

        raise RuntimeError(
            "\nShape SHAP perturbado inesperado.\n"
            f"Encontrado: {perturbed_shap.shape}\n"
            f"Esperado: {expected_perturbed_shape}"
        )


    # ========================================================
    # ΔS PERTURBADO TOP-K + L2
    # ========================================================

    perturbed_delta = build_delta_matrix(

        shap_arr=perturbed_shap,

        class_j=J_SEL,

        class_r=R_SEL,

        top_k=TOP_K
    )


    perturbed_delta = (
        perturbed_delta.reshape(
            N_INSTANCES,
            M_PERTURB,
            -1
        )
    )


    perturbed_predictions = (
        perturbed_predictions.reshape(
            N_INSTANCES,
            M_PERTURB
        )
    )


    # ========================================================
    # CALCULAR MÉTRICAS
    # ========================================================

    rows = []


    for t, test_idx in enumerate(
        selected_indices
    ):

        original_exp = base_delta[
            t
        ]

        original_pred = int(
            selected_pred[
                t
            ]
        )

        stability_values = []

        consistency_values = []

        coverage_values = []

        predkeep_values = []

        cosine_values = []


        for m in range(
            M_PERTURB
        ):

            pert_exp = (
                perturbed_delta[
                    t,
                    m
                ]
            )

            pert_pred = int(
                perturbed_predictions[
                    t,
                    m
                ]
            )


            squared_distance = float(
                np.sum(
                    (
                        pert_exp
                        -
                        original_exp
                    ) ** 2
                )
            )


            similarity = cosine_similarity(
                original_exp,
                pert_exp
            )


            antecedent = bool(
                similarity
                >=
                TAU
            )


            consequent = bool(
                pert_pred
                ==
                original_pred
            )


            implication_holds = (
                (not antecedent)
                or
                consequent
            )


            stability_values.append(
                squared_distance
            )

            cosine_values.append(
                similarity
            )

            coverage_values.append(
                int(
                    antecedent
                )
            )

            predkeep_values.append(
                int(
                    consequent
                )
            )

            consistency_values.append(
                int(
                    implication_holds
                )
            )


        true_class = int(
            y_test[
                test_idx
            ]
        )


        p_base = y_prob[
            test_idx
        ]


        rows.append({

            "local_position":
            int(t),

            "test_index":
            int(test_idx),

            "shap_row":
            int(
                selected_shap_rows[
                    t
                ]
            ),

            "true_class_id":
            true_class,

            "true_class":
            CLASS_NAMES[
                true_class
            ],

            "predicted_class_id":
            original_pred,

            "predicted_class":
            CLASS_NAMES[
                original_pred
            ],

            "correct_prediction":
            bool(
                true_class
                ==
                original_pred
            ),

            "abs_deltaP":
            float(
                abs(
                    p_base[J_SEL]
                    -
                    p_base[R_SEL]
                )
            ),

            "Stability_S":
            float(
                np.mean(
                    stability_values
                )
            ),

            "Consistency_C_imp":
            float(
                np.mean(
                    consistency_values
                )
            ),

            "Coverage_imp":
            float(
                np.mean(
                    coverage_values
                )
            ),

            "PredKeep_imp":
            float(
                np.mean(
                    predkeep_values
                )
            ),

            "Cosine_mean":
            float(
                np.mean(
                    cosine_values
                )
            ),
        })


        print(
            f"[{t + 1:02d}/{N_INSTANCES}] "
            f"idx_test={int(test_idx):5d} | "
            f"true="
            f"{CLASS_NAMES[true_class]:12s} | "
            f"pred="
            f"{CLASS_NAMES[original_pred]:12s} | "
            f"S="
            f"{rows[-1]['Stability_S']:.6f} | "
            f"Cimp="
            f"{rows[-1]['Consistency_C_imp']:.6f} | "
            f"Cov="
            f"{rows[-1]['Coverage_imp']:.6f} | "
            f"PredKeep="
            f"{rows[-1]['PredKeep_imp']:.6f}"
        )


    results = pd.DataFrame(
        rows
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    metric_columns = [
        "Stability_S",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]


    summary = (
        results[
            metric_columns
        ]
        .agg([
            "mean",
            "std",
            "median",
        ])
        .T
    )


    summary[
        "mean±std"
    ] = (
        summary[
            "mean"
        ].map(
            lambda x:
            f"{x:.6f}"
        )
        +
        " ± "
        +
        summary[
            "std"
        ].map(
            lambda x:
            f"{x:.6f}"
        )
    )


    # ========================================================
    # GUARDAR
    # ========================================================

    results.to_csv(

        OUTPUT_DIR
        /
        "fashion_mnist_Tshirt_vs_Trouser_"
        "stability_consistency_instances.csv",

        index=False,

        encoding="utf-8-sig"
    )


    summary_out = (
        summary
        .reset_index()
        .rename(
            columns={
                "index":
                "metric"
            }
        )
    )


    summary_out.insert(
        0,
        "Dataset",
        "Fashion-MNIST"
    )

    summary_out.insert(
        1,
        "Pair",
        "T-shirt/Top vs Trouser"
    )

    summary_out.insert(
        2,
        "N",
        N_INSTANCES
    )


    summary_out.to_csv(

        OUTPUT_DIR
        /
        "fashion_mnist_Tshirt_vs_Trouser_"
        "stability_consistency_summary.csv",

        index=False,

        encoding="utf-8-sig"
    )


    selected_df = pd.DataFrame({

        "test_index":
        selected_indices,

        "shap_row":
        selected_shap_rows,

        "true_class_id":
        selected_true,

        "true_class":
        [
            CLASS_NAMES[int(v)]
            for v
            in selected_true
        ],

        "predicted_class_id":
        selected_pred,

        "predicted_class":
        [
            CLASS_NAMES[int(v)]
            for v
            in selected_pred
        ],

        "correct_prediction":
        (
            selected_true
            ==
            selected_pred
        ),
    })


    selected_df.to_csv(

        OUTPUT_DIR
        /
        "selected_instances_"
        "Tshirt_vs_Trouser.csv",

        index=False,

        encoding="utf-8-sig"
    )


    diagnostics_df = pd.DataFrame(
        perturbation_diagnostics
    )


    diagnostics_df.to_csv(

        OUTPUT_DIR
        /
        "perturbation_diagnostics.csv",

        index=False,

        encoding="utf-8-sig"
    )


    metadata = {

        "dataset":
        "Fashion-MNIST",

        "model":
        str(
            MODEL_PATH.resolve()
        ),

        "shap_pickle":
        str(
            SHAP_PICKLE_PATH.resolve()
        ),

        "pair": {
            "j":
            J_SEL,

            "j_name":
            CLASS_NAMES[J_SEL],

            "r":
            R_SEL,

            "r_name":
            CLASS_NAMES[R_SEL],
        },

        "selection":
        "true labels, balanced, no prediction filter",

        "n_instances":
        N_INSTANCES,

        "n_per_class":
        N_INSTANCES // 2,

        "M_perturbations":
        M_PERTURB,

        "sigma":
        SIGMA,

        "tau":
        TAU,

        "top_k":
        TOP_K,

        "normalization":
        "L2",

        "seed":
        SEED,

        "instance_seed_rule":
        "SEED + local_position",

        "image_range":
        "[0,1]",

        "perturbation":
        (
            "independent Gaussian noise per normalized "
            "pixel followed by clipping to [0,1]"
        ),

        "base_explanation":
        (
            "stored SHAP values from definitive "
            "SHAP-GLCD pickle"
        ),

        "perturbed_explanation":
        (
            "GradientExplainer using reconstructed "
            "original training background"
        ),

        "background_size":
        SHAP_BACKGROUND_SIZE,

        "stored_shap_shape":
        list(
            stored_shap.shape
        ),

        "perturbed_shap_shape":
        list(
            perturbed_shap.shape
        ),

        "idx_eval_exactly_reconstructed":
        bool(
            same_idx_eval
        ),

        "control_shap_max_abs_difference":
        max_control_difference,

        "control_shap_mean_abs_difference":
        mean_control_difference,

        "control_delta_topk_cosine":
        control_cosine,
    }


    with open(

        OUTPUT_DIR
        /
        "protocol_fashion_mnist_"
        "stability_consistency.json",

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # RESULTADOS FINALES
    # ========================================================

    total_seconds = (
        time.time()
        -
        start_time
    )


    print(
        "\n============================================================"
    )

    print(
        " RESULTADOS FINALES - FASHION-MNIST"
    )

    print(
        "============================================================"
    )

    print(
        "Par       : "
        "T-shirt/Top vs Trouser"
    )

    print(
        "N efectivo:",
        N_INSTANCES,
        f"({N_INSTANCES // 2} + {N_INSTANCES // 2})"
    )

    print()


    print(
        summary[
            [
                "mean",
                "std",
                "median",
                "mean±std",
            ]
        ].to_string()
    )


    print(
        "\n------------------------------------------------------------"
    )

    print(
        "VALORES PRINCIPALES PARA LA TABLA"
    )

    print(
        "------------------------------------------------------------"
    )


    print(
        "\nStability:"
    )

    print(
        summary.loc[
            "Stability_S",
            "mean±std"
        ]
    )


    print(
        "\nConsistency C_imp:"
    )

    print(
        summary.loc[
            "Consistency_C_imp",
            "mean±std"
        ]
    )


    print(
        "\nCoverage:"
    )

    print(
        summary.loc[
            "Coverage_imp",
            "mean±std"
        ]
    )


    print(
        "\nPredKeep:"
    )

    print(
        summary.loc[
            "PredKeep_imp",
            "mean±std"
        ]
    )


    print(
        "\nTiempo total:"
        f" {total_seconds / 60.0:.2f} minutos"
    )


    print(
        "\nResultados guardados en:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


    print(
        "============================================================"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()