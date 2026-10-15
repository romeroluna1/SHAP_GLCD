# ============================================================
# main_stability_consistency_ham10000_070926.py
# ============================================================
#
# STABILITY + CONSISTENCY DE LAS EXPLICACIONES DIFERENCIALES
# QUE SIRVEN DE BASE A SHAP-GLCD
#
# Dataset:
#   HAM10000
#
# Modelo:
#   ResNet50 definitivo con split por lesion_id
#
# Par:
#   BCC vs BKL
#   1 = Basal cell carcinoma
#   2 = Benign keratosis-like lesions
#
# ============================================================
#
# IMPORTANTE
#
# - NO reentrena el modelo.
# - NO recalcula los 204 SHAP originales.
# - Usa directamente el pickle definitivo de SHAP-GLCD.
# - Reconstruye exactamente:
#       * split por lesion_id
#       * pool de 500 imágenes de train
#       * background SHAP de 30 imágenes
#       * idx_eval del SHAP-GLCD
#
# - Stability/Consistency se calcula sobre:
#       N = 50 = 25 BCC + 25 BKL
#   seleccionadas dentro de la muestra SHAP-GLCD definitiva
#   de 204 imágenes.
#
# - Selección por etiqueta verdadera.
# - Sin filtro por predicción.
#
# ============================================================
# PROTOCOLO
# ============================================================
#
# N      = 50
# M      = 40
# sigma  = 0.005
# tau    = 0.95
# top-k  = {10, 50, 100}  (análisis de sensibilidad)
# K=10   = referencia fiel al protocolo SHAP-LCD
# norm   = L2
# seed   = 42
#
# ============================================================
# PERTURBACIONES
# ============================================================
#
# Las imágenes llegan al modelo en [0,1].
#
# Para cada componente RGB:
#
#   x' = x + N(0, sigma^2)
#   x' = clip(x', 0, 1)
#
# ============================================================
# MEMORIA
# ============================================================
#
# A diferencia de Fashion-MNIST, una explicación HAM10000 tiene
# 224*224*3 = 150528 componentes.
#
# Por ello NO se calculan los SHAP de las 2000 perturbaciones
# simultáneamente.
#
# Se procesan 40 perturbaciones por instancia:
#
#   50 llamadas x 40 perturbaciones
#
# Esto mantiene exactamente el mismo protocolo estadístico
# evitando un consumo excesivo de memoria.
#
# ============================================================


import os
import sys
import gc
import json
import time
import pickle
import random
import platform
from pathlib import Path

import kagglehub
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

from tensorflow.keras import layers, models


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 16

J_SEL = 1       # BCC
R_SEL = 2       # BKL

N_INSTANCES = 50
M_PERTURB = 40

SIGMA = 0.005
TAU = 0.95
TOP_K_VALUES = (10, 50, 100)
REFERENCE_TOP_K = 10

SHAP_TRAIN_POOL = 500
SHAP_BACKGROUND_SIZE = 30

KAGGLE_DATASET = (
    "kmader/skin-cancer-mnist-ham10000"
)


CLASS_ORDER = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


CLASS_CODE_TO_NAME = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesions",
}


CODE_TO_ID = {
    code: idx
    for idx, code
    in enumerate(CLASS_ORDER)
}


ID_TO_CODE = {
    idx: code
    for code, idx
    in CODE_TO_ID.items()
}


CLASS_NAMES = {
    idx: CLASS_CODE_TO_NAME[code]
    for idx, code
    in ID_TO_CODE.items()
}


# ============================================================
# 2. RUTAS
# ============================================================

# ============================================================
# 2. RUTAS DEL REPOSITORIO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "ham10000"
)

WEIGHTS_PATH = (
    ARTIFACT_DIR
    / "ham10000_resnet50_best_overall_"
      "lesion_split_030926.weights.h5"
)

SHAP_PICKLE_PATH = (
    ARTIFACT_DIR
    / "shap_values_ham10000_resnet50_"
      "lesion_split_030926_1_vs_2.pickle"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "ham10000"
    / "pair_1_vs_2"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 3. SEMILLAS / DETERMINISMO
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
# 4. UTILIDADES DATASET
# ============================================================

def find_metadata_csv(root):

    candidates = list(
        Path(root).rglob(
            "HAM10000_metadata.csv"
        )
    )

    if len(candidates) == 0:

        raise FileNotFoundError(
            "No se encontró HAM10000_metadata.csv."
        )

    return candidates[0]


# ============================================================

def build_image_path_map(root):

    root = Path(root)

    path_map = {}

    for ext in [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ]:

        for img_path in root.rglob(ext):

            path_map[
                img_path.stem
            ] = str(img_path)

    return path_map


# ============================================================

def prepare_ham10000_dataframe(
    metadata_csv,
    dataset_root
):

    df = pd.read_csv(
        metadata_csv
    )

    required = {
        "image_id",
        "lesion_id",
        "dx",
    }

    missing = required.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            "Faltan columnas requeridas: "
            f"{sorted(missing)}"
        )


    image_path_map = (
        build_image_path_map(
            dataset_root
        )
    )


    print(
        "\nTotal image files found:",
        len(image_path_map)
    )


    df["path"] = (
        df["image_id"]
        .map(image_path_map)
    )


    df = (
        df
        .dropna(subset=["path"])
        .copy()
    )


    df = (
        df[
            df["dx"].isin(
                CLASS_ORDER
            )
        ]
        .copy()
    )


    df["label"] = (
        df["dx"]
        .map(CODE_TO_ID)
        .astype(int)
    )


    if df["image_id"].duplicated().any():

        raise RuntimeError(
            "Hay image_id duplicados."
        )


    dx_per_lesion = (
        df
        .groupby("lesion_id")["dx"]
        .nunique()
    )


    if (
        dx_per_lesion
        >
        1
    ).any():

        raise RuntimeError(
            "Hay lesion_id asociados "
            "a más de una clase."
        )


    print(
        "\nHAM10000 cargado:"
    )

    print(
        "Imágenes:",
        len(df)
    )

    print(
        "Lesiones únicas:",
        df["lesion_id"].nunique()
    )


    return df


# ============================================================

def split_by_lesion(df):

    lesion_df = (
        df
        .groupby(
            "lesion_id",
            as_index=False
        )
        .agg(
            dx=("dx", "first"),
            label=("label", "first"),
            n_images=("image_id", "size"),
        )
    )


    lesions_trainval, lesions_test = (
        train_test_split(
            lesion_df,
            test_size=0.20,
            random_state=SEED,
            stratify=lesion_df["label"],
        )
    )


    lesions_train, lesions_val = (
        train_test_split(
            lesions_trainval,
            test_size=0.20,
            random_state=SEED,
            stratify=lesions_trainval["label"],
        )
    )


    train_ids = set(
        lesions_train["lesion_id"]
    )

    val_ids = set(
        lesions_val["lesion_id"]
    )

    test_ids = set(
        lesions_test["lesion_id"]
    )


    if not train_ids.isdisjoint(
        val_ids
    ):
        raise RuntimeError(
            "Fuga train-validation."
        )


    if not train_ids.isdisjoint(
        test_ids
    ):
        raise RuntimeError(
            "Fuga train-test."
        )


    if not val_ids.isdisjoint(
        test_ids
    ):
        raise RuntimeError(
            "Fuga validation-test."
        )


    df_train = (
        df[
            df["lesion_id"].isin(
                train_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )


    df_val = (
        df[
            df["lesion_id"].isin(
                val_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )


    df_test = (
        df[
            df["lesion_id"].isin(
                test_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )


    return (
        df_train,
        df_val,
        df_test,
        train_ids,
        val_ids,
        test_ids,
    )


# ============================================================
# 5. PREPROCESAMIENTO DE IMAGEN
# ============================================================

def decode_resize_image(
    path,
    label
):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    image = (
        tf.cast(
            image,
            tf.float32
        )
        /
        255.0
    )

    return image, label


# ============================================================

def make_dataset(
    df
):

    paths = (
        df["path"]
        .values
    )

    labels = (
        df["label"]
        .values
        .astype("int64")
    )


    ds = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                labels
            )
        )
    )


    ds = ds.map(
        decode_resize_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    ds = ds.batch(
        BATCH_SIZE
    )


    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )


    return ds


# ============================================================

def load_images_from_rows(
    df_rows
):

    images = []


    for row in (
        df_rows
        .itertuples(index=False)
    ):

        image_tensor, _ = (
            decode_resize_image(
                tf.convert_to_tensor(
                    str(row.path)
                ),
                tf.convert_to_tensor(
                    int(row.label),
                    dtype=tf.int64
                ),
            )
        )

        images.append(
            image_tensor.numpy()
        )


    return np.stack(
        images
    ).astype(
        "float32"
    )


# ============================================================
# 6. MODELO EXACTO
# ============================================================

def make_resnet50_ham10000(
    input_shape,
    num_classes
):

    inputs = layers.Input(
        shape=input_shape,
        name="input_image"
    )


    data_augmentation = (
        tf.keras.Sequential(
            [
                layers.RandomFlip(
                    "horizontal"
                ),
                layers.RandomRotation(
                    0.05
                ),
                layers.RandomTranslation(
                    0.03,
                    0.03
                ),
                layers.RandomZoom(
                    0.05
                ),
            ],
            name="data_augmentation"
        )
    )


    x = data_augmentation(
        inputs
    )


    x = layers.Rescaling(
        255.0,
        name="to_255"
    )(x)


    x = layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input,
        name="resnet50_preprocess"
    )(x)


    base_model = (
        tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    )


    base_model.trainable = False


    x = base_model(
        x,
        training=False
    )


    x = layers.Dropout(
        0.40,
        name="dropout_1"
    )(x)


    x = layers.Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)


    x = layers.BatchNormalization(
        name="dense_bn"
    )(x)


    x = layers.Dropout(
        0.30,
        name="dropout_2"
    )(x)


    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier"
    )(x)


    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_HAM10000"
    )


    return model


# ============================================================
# 7. SHAP HELPERS
# ============================================================

def stored_shap_shape_npk(
    shap_values
):
    """
    Devuelve shape equivalente:
        (N, P, K)
    sin crear innecesariamente una copia gigante.
    """

    if isinstance(
        shap_values,
        list
    ):

        K = len(
            shap_values
        )

        first = np.asarray(
            shap_values[0]
        )

        N = first.shape[0]

        P = int(
            np.prod(
                first.shape[1:]
            )
        )

        return (
            N,
            P,
            K
        )


    arr = np.asarray(
        shap_values
    )


    if arr.ndim == 5:

        return (
            arr.shape[0],
            int(
                np.prod(
                    arr.shape[1:-1]
                )
            ),
            arr.shape[-1]
        )


    if arr.ndim == 3:

        return arr.shape


    raise ValueError(
        "Formato SHAP inesperado: "
        f"{arr.shape}"
    )


# ============================================================

def get_shap_instance_npk(
    shap_values,
    row
):
    """
    Extrae UNA instancia como (P,K),
    evitando convertir todo el pickle.
    """

    if isinstance(
        shap_values,
        list
    ):

        vectors = []

        for class_values in shap_values:

            v = np.asarray(
                class_values[row]
            ).reshape(-1)

            vectors.append(v)

        return np.stack(
            vectors,
            axis=1
        )


    arr = np.asarray(
        shap_values
    )


    if arr.ndim == 5:

        return (
            arr[row]
            .reshape(
                -1,
                arr.shape[-1]
            )
        )


    if arr.ndim == 3:

        return arr[row]


    raise ValueError(
        "Formato SHAP inesperado."
    )


# ============================================================

def extract_delta_batch(
    shap_values,
    class_j,
    class_r
):
    """
    Devuelve solamente:

        ΔS = SHAP(Cj) - SHAP(Cr)

    shape:
        (N, P)

    Evita conservar copias de las 7 clases.
    """

    if isinstance(
        shap_values,
        list
    ):

        phi_j = np.asarray(
            shap_values[
                class_j
            ]
        )

        phi_r = np.asarray(
            shap_values[
                class_r
            ]
        )

        delta = (
            phi_j
            -
            phi_r
        )

        return delta.reshape(
            delta.shape[0],
            -1
        )


    arr = np.asarray(
        shap_values
    )


    if arr.ndim == 5:

        delta = (
            arr[..., class_j]
            -
            arr[..., class_r]
        )

        return delta.reshape(
            delta.shape[0],
            -1
        )


    if arr.ndim == 3:

        return (
            arr[:, :, class_j]
            -
            arr[:, :, class_r]
        )


    raise ValueError(
        "Formato SHAP inesperado: "
        f"{arr.shape}"
    )


# ============================================================
# 8. ΔS TOP-K + L2
# ============================================================

def top_k_l2(
    vector,
    top_k=10
):

    vector = np.asarray(
        vector,
        dtype=np.float64
    )


    if (
        top_k is not None
        and
        top_k < vector.size
    ):

        idx = (
            np.argsort(
                np.abs(vector)
            )[::-1][:top_k]
        )


        sparse = np.zeros_like(
            vector
        )


        sparse[idx] = (
            vector[idx]
        )


        vector = sparse


    norm = np.linalg.norm(
        vector
    )


    return (
        vector
        /
        (
            norm
            +
            1e-12
        )
    )


# ============================================================


def top_k_l2_from_ranked(
    vector,
    ranked_indices,
    top_k
):
    """
    Construye la representación Top-K + L2 usando un ranking
    de |ΔS| calculado previamente.

    Esto permite evaluar K = 10, 50 y 100 sin volver a ordenar
    el vector para cada K.
    """

    vector = np.asarray(
        vector,
        dtype=np.float64
    )

    ranked_indices = np.asarray(
        ranked_indices,
        dtype=int
    )

    if (
        top_k is None
        or
        top_k >= vector.size
    ):
        selected = vector.copy()
    else:
        selected = np.zeros_like(
            vector
        )

        idx = ranked_indices[
            :top_k
        ]

        selected[idx] = vector[idx]

    norm = np.linalg.norm(
        selected
    )

    return (
        selected
        /
        (
            norm
            +
            1e-12
        )
    )


# ============================================================

def cosine_similarity(
    a,
    b
):

    denominator = (
        (
            np.linalg.norm(a)
            +
            1e-12
        )
        *
        (
            np.linalg.norm(b)
            +
            1e-12
        )
    )


    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# 9. MUESTREO BALANCEADO
# ============================================================

def sample_pair_indices(
    y_true,
    class_j,
    class_r,
    n_instances,
    seed
):

    if n_instances % 2 != 0:

        raise ValueError(
            "N debe ser par."
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
            "No hay suficientes instancias "
            "para 25 + 25."
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
# 10. PERTURBACIÓN DE IMAGEN
# ============================================================

def perturb_image(
    base_image,
    rng,
    sigma
):

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

    start_total = time.time()


    # ========================================================
    # INFORMACIÓN DE ENTORNO
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
        " HAM10000 - STABILITY / CONSISTENCY SHAP-GLCD"
    )

    print(
        "============================================================"
    )

    print(
        "Par               : BCC vs BKL"
    )

    print(
        "Selección         : etiqueta verdadera"
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
        "M                 :",
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
        "top-k values      :",
        TOP_K_VALUES
    )

    print(
        "K referencia      :",
        REFERENCE_TOP_K
    )

    print(
        "seed              :",
        SEED
    )

    print(
        "Background        :",
        SHAP_BACKGROUND_SIZE
    )

    print(
        "Perturbación      : "
        "Gaussian RGB + clip [0,1]"
    )

    print(
        "============================================================"
    )


    # ========================================================
    # COMPROBAR ARTEFACTOS
    # ========================================================

    if not WEIGHTS_PATH.exists():

        raise FileNotFoundError(
            "\nNo se encontraron "
            "los pesos definitivos:\n"
            f"{WEIGHTS_PATH}"
        )


    if not SHAP_PICKLE_PATH.exists():

        raise FileNotFoundError(
            "\nNo se encontró "
            "el pickle SHAP definitivo:\n"
            f"{SHAP_PICKLE_PATH}"
        )


    print(
        "\nPesos:"
    )

    print(
        WEIGHTS_PATH.resolve()
    )


    print(
        "\nSHAP pickle:"
    )

    print(
        SHAP_PICKLE_PATH.resolve()
    )


    # ========================================================
    # DATASET
    # ========================================================

    print(
        "\nLocalizando HAM10000..."
    )


    dataset_path = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )


    metadata_csv = (
        find_metadata_csv(
            dataset_path
        )
    )


    df_ham = (
        prepare_ham10000_dataframe(
            metadata_csv,
            dataset_path
        )
    )


    (
        df_train,
        df_val,
        df_test,
        train_ids,
        val_ids,
        test_ids,
    ) = split_by_lesion(
        df_ham
    )


    print(
        "\n============================================================"
    )

    print(
        " SPLIT VERIFICATION"
    )

    print(
        "============================================================"
    )

    print(
        "Train images     :",
        len(df_train)
    )

    print(
        "Validation images:",
        len(df_val)
    )

    print(
        "Test images      :",
        len(df_test)
    )


    if (
        len(df_train) != 6404
        or
        len(df_val) != 1613
        or
        len(df_test) != 1998
    ):

        raise RuntimeError(
            "El split no coincide con "
            "el experimento definitivo."
        )


    test_counts = (
        df_test["label"]
        .value_counts()
        .sort_index()
    )


    print(
        "\nDistribución test:"
    )

    for cls, count in (
        test_counts.items()
    ):

        print(
            f"{int(cls)} "
            f"({CLASS_NAMES[int(cls)]}): "
            f"{count}"
        )


    if int(
        np.sum(
            df_test["label"].values
            ==
            J_SEL
        )
    ) != 102:

        raise RuntimeError(
            "BCC test != 102."
        )


    if int(
        np.sum(
            df_test["label"].values
            ==
            R_SEL
        )
    ) != 208:

        raise RuntimeError(
            "BKL test != 208."
        )


    # ========================================================
    # MODELO
    # ========================================================

    print(
        "\nConstruyendo ResNet50..."
    )


    model = (
        make_resnet50_ham10000(
            input_shape=(
                IMG_SIZE,
                IMG_SIZE,
                3
            ),
            num_classes=7
        )
    )


    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


    model.load_weights(
        WEIGHTS_PATH
    )


    print(
        "✓ Pesos definitivos cargados."
    )


    # ========================================================
    # MÉTRICAS TEST COMPLETO
    # ========================================================

    ds_test = make_dataset(
        df_test
    )


    test_loss, keras_acc = (
        model.evaluate(
            ds_test,
            verbose=0
        )
    )


    y_test_true = (
        df_test["label"]
        .values
        .astype("int64")
    )


    y_prob = model.predict(
        ds_test,
        verbose=0
    )


    y_pred = np.argmax(
        y_prob,
        axis=1
    )


    accuracy = accuracy_score(
        y_test_true,
        y_pred
    )


    balanced_acc = (
        balanced_accuracy_score(
            y_test_true,
            y_pred
        )
    )


    f1_macro = f1_score(
        y_test_true,
        y_pred,
        average="macro"
    )


    mcc = matthews_corrcoef(
        y_test_true,
        y_pred
    )


    auc_ovo = roc_auc_score(
        y_test_true,
        y_prob,
        multi_class="ovo",
        average="macro",
        labels=np.arange(7),
    )


    print(
        "\n============================================================"
    )

    print(
        " MÉTRICAS DEL MODELO - TEST"
    )

    print(
        "============================================================"
    )

    print(
        f"Loss              : {test_loss:.6f}"
    )

    print(
        f"Accuracy          : {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy : {balanced_acc:.6f}"
    )

    print(
        f"F1 macro          : {f1_macro:.6f}"
    )

    print(
        f"MCC               : {mcc:.6f}"
    )

    print(
        f"Macro-AUC OvO     : {auc_ovo:.6f}"
    )


    expected = np.array([
        0.634835,
        0.788789,
        0.594840,
        0.628399,
        0.601748,
        0.928156,
    ])


    observed = np.array([
        test_loss,
        accuracy,
        balanced_acc,
        f1_macro,
        mcc,
        auc_ovo,
    ])


    max_metric_diff = float(
        np.max(
            np.abs(
                expected
                -
                observed
            )
        )
    )


    if max_metric_diff > 1e-5:

        raise RuntimeError(
            "\nLas métricas no reproducen "
            "el experimento definitivo.\n"
            f"Máxima diferencia: "
            f"{max_metric_diff:.12e}"
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
            "El pickle SHAP no es "
            "el formato definitivo esperado."
        )


    if "shap_values" not in shap_object:

        raise RuntimeError(
            "Falta 'shap_values'."
        )


    if "idx_eval" not in shap_object:

        raise RuntimeError(
            "Falta 'idx_eval'."
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


    stored_shape = (
        stored_shap_shape_npk(
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
        "Shape equivalente (N,P,K):",
        stored_shape
    )

    print(
        "idx_eval:",
        stored_idx_eval.shape
    )


    expected_shape = (
        204,
        150528,
        7
    )


    if stored_shape != expected_shape:

        raise RuntimeError(
            "\nShape SHAP inesperado.\n"
            f"Encontrado: {stored_shape}\n"
            f"Esperado: {expected_shape}"
        )


    # ========================================================
    # RECONSTRUIR POOL TRAIN ORIGINAL
    # ========================================================

    df_train_pool = (
        df_train
        .sample(
            n=SHAP_TRAIN_POOL,
            random_state=SEED
        )
        .reset_index(drop=True)
    )


    y_test_shap = (
        df_test["label"]
        .values
        .astype("int64")
    )


    # ========================================================
    # RECONSTRUIR BACKGROUND + idx_eval ORIGINAL
    # ========================================================

    np.random.seed(
        SEED
    )


    idx_background = (
        np.random.choice(
            SHAP_TRAIN_POOL,
            size=SHAP_BACKGROUND_SIZE,
            replace=False
        )
    )


    idx_j = np.where(
        y_test_shap
        ==
        J_SEL
    )[0]


    idx_r = np.where(
        y_test_shap
        ==
        R_SEL
    )[0]


    # SHAP-GLCD usó 102 + 102 = 204
    n_each_original = 102


    sel_j = np.random.choice(
        idx_j,
        size=n_each_original,
        replace=False
    )


    sel_r = np.random.choice(
        idx_r,
        size=n_each_original,
        replace=False
    )


    reconstructed_idx_eval = (
        np.concatenate(
            [
                sel_j,
                sel_r
            ]
        )
    )


    np.random.shuffle(
        reconstructed_idx_eval
    )


    state_before_original_shap = (
        np.random.get_state()
    )


    same_idx = np.array_equal(
        stored_idx_eval,
        reconstructed_idx_eval
    )


    print(
        "\n============================================================"
    )

    print(
        " VERIFICACIÓN DEL MUESTREO SHAP-GLCD"
    )

    print(
        "============================================================"
    )

    print(
        "Background original:",
        SHAP_BACKGROUND_SIZE
    )

    print(
        "idx_eval almacenado == reconstruido:",
        same_idx
    )


    if not same_idx:

        raise RuntimeError(
            "\nNo se reprodujo idx_eval.\n"
            "No se continuará para evitar "
            "mezclar explicaciones."
        )


    print(
        "\n✓ idx_eval reproducido exactamente."
    )


    # ========================================================
    # CARGAR SOLO LAS 30 IMÁGENES DEL BACKGROUND
    # ========================================================

    background_rows = (
        df_train_pool
        .iloc[
            idx_background
        ]
        .copy()
    )


    X_background = (
        load_images_from_rows(
            background_rows
        )
    )


    print(
        "Background shape:",
        X_background.shape
    )


    # ========================================================
    # GRADIENT EXPLAINER
    # ========================================================

    print(
        "\nConstruyendo GradientExplainer..."
    )


    gradient_explainer = (
        shap.GradientExplainer(
            model,
            X_background
        )
    )


    # ========================================================
    # DISTRIBUCIÓN DE LA MUESTRA GLOBAL SHAP
    # ========================================================

    y_eval = (
        y_test_shap[
            stored_idx_eval
        ]
    )


    print(
        "\nDistribución SHAP-GLCD:"
    )

    print(
        pd.Series(
            y_eval
        )
        .map(
            CLASS_NAMES
        )
        .value_counts()
    )


    if (
        np.sum(y_eval == J_SEL) != 102
        or
        np.sum(y_eval == R_SEL) != 102
    ):

        raise RuntimeError(
            "La muestra SHAP no es 102 + 102."
        )


    # ========================================================
    # CONTROL SHAP ORIGINAL
    # ========================================================

    control_shap_row = 0

    control_test_position = int(
        stored_idx_eval[
            control_shap_row
        ]
    )


    control_row = (
        df_test
        .iloc[
            [
                control_test_position
            ]
        ]
    )


    control_image = (
        load_images_from_rows(
            control_row
        )
    )


    np.random.set_state(
        state_before_original_shap
    )


    control_direct_raw = (
        gradient_explainer.shap_values(
            control_image
        )
    )


    control_direct = (
        get_shap_instance_npk(
            control_direct_raw,
            0
        )
    )


    control_stored = (
        get_shap_instance_npk(
            stored_shap_raw,
            control_shap_row
        )
    )


    max_control_diff = float(
        np.max(
            np.abs(
                control_direct
                -
                control_stored
            )
        )
    )


    mean_control_diff = float(
        np.mean(
            np.abs(
                control_direct
                -
                control_stored
            )
        )
    )


    control_delta_stored = (
        top_k_l2(
            control_stored[:, J_SEL]
            -
            control_stored[:, R_SEL],
            REFERENCE_TOP_K
        )
    )


    control_delta_direct = (
        top_k_l2(
            control_direct[:, J_SEL]
            -
            control_direct[:, R_SEL],
            REFERENCE_TOP_K
        )
    )


    control_cosine = (
        cosine_similarity(
            control_delta_stored,
            control_delta_direct
        )
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
        "SHAP row     :",
        control_shap_row
    )

    print(
        "Test position:",
        control_test_position
    )

    print(
        f"Max |stored-direct|  : "
        f"{max_control_diff:.12e}"
    )

    print(
        f"Mean |stored-direct| : "
        f"{mean_control_diff:.12e}"
    )

    print(
        f"Cosine ΔS top-k      : "
        f"{control_cosine:.12f}"
    )


    if control_cosine < 0.95:

        raise RuntimeError(
            "\nLa explicación de control no "
            "es suficientemente cercana al "
            "SHAP almacenado.\n"
            "Revisar antes de continuar."
        )


    print(
        "\n✓ Control SHAP compatible."
    )


    del control_direct_raw
    del control_direct
    del control_stored
    del control_image

    gc.collect()


    # ========================================================
    # SELECCIÓN N=50 DENTRO DEL SUBCONJUNTO SHAP-GLCD
    # ========================================================
    #
    # Esto garantiza que TODAS las explicaciones base utilizadas
    # sean exactamente las almacenadas en el experimento global.
    #
    # selected_shap_rows son posiciones dentro de las 204
    # explicaciones.
    #
    # selected_test_positions son posiciones dentro de df_test.
    # ========================================================

    selected_shap_rows = (
        sample_pair_indices(
            y_true=y_eval,
            class_j=J_SEL,
            class_r=R_SEL,
            n_instances=N_INSTANCES,
            seed=SEED
        )
    )


    selected_test_positions = (
        stored_idx_eval[
            selected_shap_rows
        ]
    )


    selected_df_rows = (
        df_test
        .iloc[
            selected_test_positions
        ]
        .copy()
        .reset_index(drop=True)
    )


    selected_images = (
        load_images_from_rows(
            selected_df_rows
        )
    )


    selected_true = (
        selected_df_rows[
            "label"
        ]
        .values
        .astype(int)
    )


    selected_prob = (
        model.predict(
            selected_images,
            batch_size=BATCH_SIZE,
            verbose=0
        )
    )


    selected_pred = (
        np.argmax(
            selected_prob,
            axis=1
        )
    )


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
        "Fuente             : "
        "muestra SHAP-GLCD del test"
    )

    print(
        "N seleccionado     :",
        len(selected_shap_rows)
    )

    print(
        "Balance esperado   : "
        f"{N_INSTANCES // 2} + {N_INSTANCES // 2}"
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
    )


    correct = int(
        np.sum(
            selected_true
            ==
            selected_pred
        )
    )


    incorrect = (
        N_INSTANCES
        -
        correct
    )


    print(
        "\nPredicciones correctas :",
        correct
    )

    print(
        "Predicciones incorrectas:",
        incorrect
    )

    print(
        "Filtro por predicción   : NO"
    )


    # ========================================================
    # GUARDAR MUESTRA
    # ========================================================

    selected_metadata = (
        selected_df_rows[
            [
                "image_id",
                "lesion_id",
                "dx",
                "label",
                "path",
            ]
        ]
        .copy()
    )


    selected_metadata.insert(
        0,
        "local_position",
        np.arange(
            N_INSTANCES
        )
    )


    selected_metadata[
        "shap_row"
    ] = selected_shap_rows


    selected_metadata[
        "test_position"
    ] = selected_test_positions


    selected_metadata[
        "predicted_class_id"
    ] = selected_pred


    selected_metadata[
        "predicted_class"
    ] = [
        CLASS_NAMES[int(v)]
        for v
        in selected_pred
    ]


    selected_metadata[
        "correct_prediction"
    ] = (
        selected_true
        ==
        selected_pred
    )


    selected_metadata.to_csv(
        RESULTS_DIR
        /
        "selected_instances_BCC_vs_BKL.csv",
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # STABILITY / CONSISTENCY
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        " CALCULANDO STABILITY / CONSISTENCY"
    )

    print(
        "============================================================"
    )

    print(
        "Modo memoria       : "
        f"{M_PERTURB} perturbaciones por instancia"
    )

    print(
        "Total perturbaciones:",
        N_INSTANCES
        *
        M_PERTURB
    )


    # --------------------------------------------------------
    # Reiniciamos el RNG global de NumPy en el mismo estado
    # usado antes del SHAP original.
    #
    # Los RNG de las perturbaciones usan default_rng y son
    # independientes de este estado.
    # --------------------------------------------------------

    np.random.set_state(
        state_before_original_shap
    )


    result_rows = []

    perturbation_diagnostics = []


    for t in range(
        N_INSTANCES
    ):

        instance_start = (
            time.time()
        )


        shap_row = int(
            selected_shap_rows[
                t
            ]
        )


        test_position = int(
            selected_test_positions[
                t
            ]
        )


        base_image = (
            selected_images[
                t
            ]
        )


        true_class = int(
            selected_true[
                t
            ]
        )


        base_pred = int(
            selected_pred[
                t
            ]
        )


        # ----------------------------------------------------
        # Explicación BASE = SHAP almacenado
        # ----------------------------------------------------

        stored_instance = (
            get_shap_instance_npk(
                stored_shap_raw,
                shap_row
            )
        )

        base_delta_raw = (
            stored_instance[
                :,
                J_SEL
            ]
            -
            stored_instance[
                :,
                R_SEL
            ]
        )

        # Un único ranking de |ΔS| para construir las tres
        # representaciones Top-K de la explicación base.
        base_ranked_indices = (
            np.argsort(
                np.abs(
                    base_delta_raw
                )
            )[::-1]
        )

        base_exp_by_k = {
            int(k):
            top_k_l2_from_ranked(
                base_delta_raw,
                base_ranked_indices,
                int(k)
            )
            for k in TOP_K_VALUES
        }

        del stored_instance
        del base_delta_raw
        del base_ranked_indices


        # ----------------------------------------------------
        # RNG EXACTO DEL PROTOCOLO
        # ----------------------------------------------------

        # Regla común del protocolo SHAP-LCD:
        # seed_instancia = seed + t,
        # donde t es la posición local de la instancia seleccionada.
        instance_seed = (
            SEED
            +
            t
        )

        rng = (
            np.random.default_rng(
                instance_seed
            )
        )


        # ----------------------------------------------------
        # M_PERTURB perturbaciones
        # ----------------------------------------------------

        perturbations = []
        mean_abs_changes = []

        for m in range(
            M_PERTURB
        ):

            perturbed = (
                perturb_image(
                    base_image,
                    rng,
                    SIGMA
                )
            )

            perturbations.append(
                perturbed
            )

            mean_abs_changes.append(
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

        perturbations = (
            np.stack(
                perturbations
            )
            .astype(
                "float32"
            )
        )

        if (
            perturbations.min()
            <
            0.0
            or
            perturbations.max()
            >
            1.0
        ):

            raise RuntimeError(
                "Perturbación fuera de [0,1]."
            )


        # ----------------------------------------------------
        # Predicciones perturbadas
        # ----------------------------------------------------

        pert_prob = (
            model.predict(
                perturbations,
                batch_size=BATCH_SIZE,
                verbose=0
            )
        )

        pert_pred = (
            np.argmax(
                pert_prob,
                axis=1
            )
        )


        # ----------------------------------------------------
        # SHAP de SOLO estas M_PERTURB perturbaciones
        # SE CALCULA UNA SOLA VEZ, independientemente de K.
        # ----------------------------------------------------

        shap_start = (
            time.time()
        )

        pert_shap_raw = (
            gradient_explainer
            .shap_values(
                perturbations
            )
        )

        shap_elapsed = (
            time.time()
            -
            shap_start
        )

        pert_delta_raw = (
            extract_delta_batch(
                pert_shap_raw,
                J_SEL,
                R_SEL
            )
        )

        # Liberar inmediatamente las 7 clases SHAP
        del pert_shap_raw
        gc.collect()


        # ----------------------------------------------------
        # Acumuladores separados para cada K
        # ----------------------------------------------------

        metrics_by_k = {
            int(k): {
                "stability": [],
                "consistency": [],
                "coverage": [],
                "predkeep": [],
                "cosine": [],
            }
            for k in TOP_K_VALUES
        }


        # ----------------------------------------------------
        # Métricas para las M_PERTURB perturbaciones
        #
        # Para cada perturbación se calcula UNA sola vez el
        # ranking de |ΔS| y se reutiliza para K=10,50,100.
        # ----------------------------------------------------

        for m in range(
            M_PERTURB
        ):

            pert_vector = (
                pert_delta_raw[
                    m
                ]
            )

            ranked_indices = (
                np.argsort(
                    np.abs(
                        pert_vector
                    )
                )[::-1]
            )

            B = bool(
                int(
                    pert_pred[
                        m
                    ]
                )
                ==
                base_pred
            )

            for k in TOP_K_VALUES:

                k = int(k)

                pert_exp = (
                    top_k_l2_from_ranked(
                        pert_vector,
                        ranked_indices,
                        k
                    )
                )

                base_exp = (
                    base_exp_by_k[
                        k
                    ]
                )

                squared_distance = float(
                    np.sum(
                        (
                            pert_exp
                            -
                            base_exp
                        ) ** 2
                    )
                )

                sim = cosine_similarity(
                    base_exp,
                    pert_exp
                )

                A = bool(
                    sim
                    >=
                    TAU
                )

                implication = (
                    (not A)
                    or
                    B
                )

                metrics_by_k[
                    k
                ][
                    "stability"
                ].append(
                    squared_distance
                )

                metrics_by_k[
                    k
                ][
                    "cosine"
                ].append(
                    sim
                )

                metrics_by_k[
                    k
                ][
                    "coverage"
                ].append(
                    int(A)
                )

                metrics_by_k[
                    k
                ][
                    "predkeep"
                ].append(
                    int(B)
                )

                metrics_by_k[
                    k
                ][
                    "consistency"
                ].append(
                    int(
                        implication
                    )
                )


        # ----------------------------------------------------
        # Resultado por instancia y por K
        # ----------------------------------------------------

        p_base = (
            selected_prob[
                t
            ]
        )

        elapsed = (
            time.time()
            -
            instance_start
        )

        instance_print_parts = []

        for k in TOP_K_VALUES:

            k = int(k)

            stability = float(
                np.mean(
                    metrics_by_k[
                        k
                    ][
                        "stability"
                    ]
                )
            )

            consistency = float(
                np.mean(
                    metrics_by_k[
                        k
                    ][
                        "consistency"
                    ]
                )
            )

            coverage = float(
                np.mean(
                    metrics_by_k[
                        k
                    ][
                        "coverage"
                    ]
                )
            )

            predkeep = float(
                np.mean(
                    metrics_by_k[
                        k
                    ][
                        "predkeep"
                    ]
                )
            )

            cosine_mean = float(
                np.mean(
                    metrics_by_k[
                        k
                    ][
                        "cosine"
                    ]
                )
            )

            result_rows.append({

                "local_position":
                int(t),

                "top_k":
                k,

                "shap_row":
                shap_row,

                "test_position":
                test_position,

                "image_id":
                str(
                    selected_df_rows
                    .iloc[t][
                        "image_id"
                    ]
                ),

                "lesion_id":
                str(
                    selected_df_rows
                    .iloc[t][
                        "lesion_id"
                    ]
                ),

                "true_class_id":
                true_class,

                "true_class":
                CLASS_NAMES[
                    true_class
                ],

                "predicted_class_id":
                base_pred,

                "predicted_class":
                CLASS_NAMES[
                    base_pred
                ],

                "correct_prediction":
                bool(
                    true_class
                    ==
                    base_pred
                ),

                "abs_deltaP":
                float(
                    abs(
                        p_base[
                            J_SEL
                        ]
                        -
                        p_base[
                            R_SEL
                        ]
                    )
                ),

                "Stability_S":
                stability,

                "Consistency_C_imp":
                consistency,

                "Coverage_imp":
                coverage,

                "PredKeep_imp":
                predkeep,

                "Cosine_mean":
                cosine_mean,

                "mean_abs_pixel_change":
                float(
                    np.mean(
                        mean_abs_changes
                    )
                ),

                "seconds":
                elapsed,
            })

            instance_print_parts.append(
                (
                    f"K={k}: "
                    f"S={stability:.6f}, "
                    f"Cimp={consistency:.6f}, "
                    f"Cov={coverage:.6f}, "
                    f"PredKeep={predkeep:.6f}"
                )
            )


        perturbation_diagnostics.append({

            "local_position":
            int(t),

            "test_position":
            test_position,

            "seed":
            int(
                instance_seed
            ),

            "mean_abs_pixel_change":
            float(
                np.mean(
                    mean_abs_changes
                )
            ),

            "min_perturbed":
            float(
                perturbations.min()
            ),

            "max_perturbed":
            float(
                perturbations.max()
            ),

            "shap_seconds":
            float(
                shap_elapsed
            ),
        })


        print(
            f"[{t + 1:02d}/{N_INSTANCES}] "
            f"test={test_position:4d} | "
            f"true={CLASS_NAMES[true_class]:31s} | "
            f"pred={CLASS_NAMES[base_pred]:31s} | "
            f"{elapsed:.2f}s"
        )

        for text_k in instance_print_parts:
            print(
                "    "
                +
                text_k
            )


        # ----------------------------------------------------
        # Guardado incremental
        # ----------------------------------------------------

        pd.DataFrame(
            result_rows
        ).to_csv(
            RESULTS_DIR
            /
            "ham10000_BCC_vs_BKL_"
            "stability_consistency_instances_partial.csv",
            index=False,
            encoding="utf-8-sig"
        )


        # ----------------------------------------------------
        # LIBERAR MEMORIA DE LA INSTANCIA
        # ----------------------------------------------------

        del perturbations
        del pert_prob
        del pert_pred
        del pert_delta_raw
        del base_exp_by_k
        del metrics_by_k

        gc.collect()


    # ========================================================
    # RESULTADOS
    # ========================================================

    results = pd.DataFrame(
        result_rows
    )

    metric_columns = [
        "Stability_S",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]

    summary_rows = []

    for k in TOP_K_VALUES:

        k = int(k)

        df_k = (
            results[
                results[
                    "top_k"
                ]
                ==
                k
            ]
            .copy()
        )

        if len(df_k) != N_INSTANCES:

            raise RuntimeError(
                f"K={k}: se esperaban "
                f"{N_INSTANCES} filas y se obtuvieron "
                f"{len(df_k)}."
            )

        for metric in metric_columns:

            values = (
                df_k[
                    metric
                ]
                .astype(float)
            )

            mean_value = float(
                values.mean()
            )

            std_value = float(
                values.std(
                    ddof=1
                )
            )

            median_value = float(
                values.median()
            )

            se_value = float(
                std_value
                /
                np.sqrt(
                    len(values)
                )
            )

            summary_rows.append({

                "Dataset":
                "HAM10000",

                "Pair":
                "BCC vs BKL",

                "N":
                N_INSTANCES,

                "top_k":
                k,

                "metric":
                metric,

                "mean":
                mean_value,

                "std":
                std_value,

                "median":
                median_value,

                "SE":
                se_value,

                "mean±std":
                (
                    f"{mean_value:.6f}"
                    " ± "
                    f"{std_value:.6f}"
                ),
            })


    summary_output = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # GUARDAR
    # ========================================================

    results.to_csv(
        RESULTS_DIR
        /
        "ham10000_BCC_vs_BKL_"
        "stability_consistency_instances_MULTIK.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_output.to_csv(
        RESULTS_DIR
        /
        "ham10000_BCC_vs_BKL_"
        "stability_consistency_summary_MULTIK.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        perturbation_diagnostics
    ).to_csv(
        RESULTS_DIR
        /
        "perturbation_diagnostics.csv",
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # PROTOCOLO
    # ========================================================

    protocol = {

        "dataset":
        "HAM10000",

        "model":
        "ResNet50",

        "pair":
        "BCC vs BKL",

        "class_j":
        J_SEL,

        "class_r":
        R_SEL,

        "selection_source":
        (
            "definitive SHAP-GLCD "
            "balanced test subset"
        ),

        "selection":
        (
            "true labels, balanced, "
            "no prediction filter"
        ),

        "global_shap_pair_N":
        204,

        "global_shap_pair_balance":
        "102 + 102",

        "n_instances":
        N_INSTANCES,

        "n_per_class":
        N_INSTANCES // 2,

        "M":
        M_PERTURB,

        "sigma":
        SIGMA,

        "tau":
        TAU,

        "top_k_values":
        list(
            TOP_K_VALUES
        ),

        "reference_top_k":
        REFERENCE_TOP_K,

        "top_k_sensitivity_analysis":
        (
            "K=10,50,100 evaluated from the same SHAP "
            "perturbations; SHAP is not recomputed for each K"
        ),

        "normalization":
        "L2",

        "seed":
        SEED,

        "instance_seed_rule":
        "SEED + local_position",

        "image_input_range":
        "[0,1]",

        "perturbation":
        (
            "independent Gaussian noise "
            "per RGB component followed "
            "by clipping to [0,1]"
        ),

        "base_SHAP":
        (
            "stored definitive "
            "SHAP-GLCD pickle"
        ),

        "perturbed_SHAP":
        "GradientExplainer",

        "SHAP_background_size":
        SHAP_BACKGROUND_SIZE,

        "SHAP_train_pool":
        SHAP_TRAIN_POOL,

        "stored_SHAP_shape_NPK":
        list(
            stored_shape
        ),

        "idx_eval_exact_match":
        bool(
            same_idx
        ),

        "control_max_abs_difference":
        max_control_diff,

        "control_mean_abs_difference":
        mean_control_diff,

        "control_delta_topk_cosine":
        control_cosine,

        "memory_strategy":
        (
            f"{M_PERTURB} perturbed images processed "
            "per instance; no simultaneous "
            f"{N_INSTANCES * M_PERTURB}-image SHAP allocation"
        ),
    }


    with open(
        RESULTS_DIR
        /
        "protocol_ham10000_"
        "stability_consistency.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            protocol,
            f,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    total_time = (
        time.time()
        -
        start_total
    )

    print(
        "\n============================================================"
    )

    print(
        " RESULTADOS FINALES - HAM10000 - SENSIBILIDAD TOP-K"
    )

    print(
        "============================================================"
    )

    print(
        "Par       : BCC vs BKL"
    )

    print(
        "N efectivo:",
        N_INSTANCES,
        "(25 + 25)"
    )

    print(
        "K evaluados:",
        TOP_K_VALUES
    )

    print(
        "SHAP por perturbación calculado UNA sola vez."
    )


    for k in TOP_K_VALUES:

        k = int(k)

        print(
            "\n------------------------------------------------------------"
        )

        print(
            f"TOP-K = {k}"
        )

        print(
            "------------------------------------------------------------"
        )

        df_print = (
            summary_output[
                summary_output[
                    "top_k"
                ]
                ==
                k
            ][
                [
                    "metric",
                    "mean",
                    "std",
                    "median",
                    "SE",
                    "mean±std",
                ]
            ]
            .copy()
        )

        print(
            df_print.to_string(
                index=False
            )
        )


    print(
        "\n============================================================"
    )

    print(
        " COMPARACIÓN DIRECTA ENTRE K"
    )

    print(
        "============================================================"
    )

    comparison = (
        summary_output
        .pivot(
            index="metric",
            columns="top_k",
            values="mean"
        )
        .reindex(
            [
                "Stability_S",
                "Consistency_C_imp",
                "Coverage_imp",
                "PredKeep_imp",
            ]
        )
    )

    print(
        comparison.to_string(
            float_format=(
                lambda x:
                f"{x:.6f}"
            )
        )
    )

    print(
        "\nTiempo total:"
        f" {total_time / 60.0:.2f} minutos"
    )

    print(
        "\nResultados guardados en:"
    )

    print(
        RESULTS_DIR.resolve()
    )

    print(
        "============================================================"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()