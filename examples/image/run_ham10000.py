# ============================================================
# main_ham10000_glcd_resnet50_CORREGIDO_LESION_SPLIT_030926.py
# ============================================================
# HAM10000 + ResNet50 + SHAP-GLCD global para imágenes.
#
# CORRECCIONES METODOLÓGICAS PRINCIPALES:
#   1) Split TRAIN / VALIDATION / TEST por lesion_id, no por imagen.
#      Una misma lesión nunca puede aparecer en más de una partición.
#   2) Test completamente independiente del entrenamiento y selección del modelo.
#   3) Métricas finales completas sobre test:
#      Accuracy, Balanced Accuracy, Macro-F1, MCC y Macro-AUC OvO.
#   4) SHAP-GLCD selecciona el par mediante etiqueta verdadera y sin filtro
#      por predicción.
#   5) La muestra SHAP del par se fuerza a ser balanceada.
#   6) El preprocesamiento de imágenes para SHAP es exactamente el mismo
#      que el usado por el pipeline de evaluación.
#   7) Se usan nombres nuevos de pesos/pickle/salidas para no mezclar
#      artefactos de ejecuciones anteriores.
#   8) Se guarda el historial de entrenamiento en CSV.
#
# PRIMERA EJECUCIÓN DEFINITIVA:
#   TRAIN_MODEL = True
#   SHAP_MODE   = "always"
#
# DESPUÉS DE UNA EJECUCIÓN COMPLETA Y CORRECTA:
#   TRAIN_MODEL = False
#   SHAP_MODE   = "auto"
# ============================================================

import os
import random
import re
import gc
import sys

from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras import layers, models
# ============================================================
# Acceso al paquete local src/shap_glcd
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_glcd.image import SHAP_GLCD_ImageExplainer_MapsOnly


# ============================================================
# 1) Semilla / determinismo
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("✅ Deterministic ops enabled")
except Exception as e:
    print("⚠️ Deterministic ops not available:", e)


# ============================================================
# 2) Parámetros generales
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 16

EPOCHS_HEAD = 15
EPOCHS_FINE = 25

LR_HEAD = 1e-3
LR_FINE = 1e-5

# Se conserva en False: no activamos ponderación de clases en este experimento.
USE_CLASS_WEIGHT = False

SHOW_PLOTS = True
DPI = 600

# ------------------------------------------------------------
# IMPORTANTE PARA LA PRIMERA EJECUCIÓN CORREGIDA:
# ------------------------------------------------------------
TRAIN_MODEL = False

# Ejecutar SHAP-GLCD
RUN_SHAP = True

# Par de clases para SHAP-GLCD
# 0 = Actinic keratoses
# 1 = Basal cell carcinoma
# 2 = Benign keratosis-like lesions
# 3 = Dermatofibroma
# 4 = Melanoma
# 5 = Melanocytic nevus
# 6 = Vascular lesions
j_sel = 1
r_sel = 2


# ============================================================
# 3) Parámetros SHAP-GLCD
# ============================================================
# Pool de entrenamiento disponible para construir el background SHAP.
SHAP_TRAIN_POOL = 500

# Tamaño efectivo del background usado por GradientExplainer.
SHAP_BACKGROUND_SIZE = 30

# Máximo TOTAL deseado para el par. El main lo ajustará automáticamente
# hacia abajo si la clase minoritaria del par tiene menos de 150 imágenes
# en test, garantizando SIEMPRE n_j = n_r.
SHAP_MAX_TEST = 300

# Primera ejecución corregida: "always".
# Después de confirmar que el pickle nuevo quedó bien: "auto".
SHAP_MODE = "auto"

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# Protección para evitar entrenar un modelo nuevo y cargar por accidente
# un pickle SHAP antiguo.
if TRAIN_MODEL and SHAP_MODE == "auto":
    raise ValueError(
        "Configuración insegura: TRAIN_MODEL=True con SHAP_MODE='auto'. "
        "Si entrenas un modelo nuevo, usa SHAP_MODE='always'."
    )


# ============================================================
# 4) Carpeta de salida NUEVA
# ============================================================
ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "ham10000"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ham10000"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Se conserva este nombre para no modificar el resto del main.
OUTPUT_DIR = RESULTS_DIR

BEST_HEAD_WEIGHTS_PATH = (
    ARTIFACT_DIR
    / "ham10000_resnet50_best_head_lesion_split_030926.weights.h5"
)

BEST_FINE_WEIGHTS_PATH = (
    ARTIFACT_DIR
    / "ham10000_resnet50_best_fine_lesion_split_030926.weights.h5"
)

BEST_OVERALL_WEIGHTS_PATH = (
    ARTIFACT_DIR
    / "ham10000_resnet50_best_overall_lesion_split_030926.weights.h5"
)

FINAL_WEIGHTS_PATH = (
    ARTIFACT_DIR
    / "ham10000_resnet50_final_best_lesion_split_030926.weights.h5"
)

HISTORY_CSV_PATH = (
    RESULTS_DIR
    / "ham10000_resnet50_training_history_lesion_split_030926.csv"
)

SPLIT_MANIFEST_PATH = (
    RESULTS_DIR
    / "ham10000_split_manifest_by_lesion_030926.csv"
)

print("\n📁 Current working directory:")
print(os.getcwd())

print("\n📁 Output directory:")
print(OUTPUT_DIR)


# ============================================================
# 5) Descargar / localizar HAM10000 con KaggleHub
# ============================================================
print("\nDescargando/localizando HAM10000 con KaggleHub...")
dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
dataset_path = Path(dataset_path)

print("\n📁 Path to dataset files:")
print(dataset_path)


def find_metadata_csv(root):
    candidates = list(root.rglob("HAM10000_metadata.csv"))
    if len(candidates) == 0:
        raise FileNotFoundError("No se encontró HAM10000_metadata.csv.")
    return candidates[0]


METADATA_CSV = find_metadata_csv(dataset_path)

print("\n✅ Metadata CSV:")
print(METADATA_CSV)


# ============================================================
# 6) Clases HAM10000
# ============================================================
CLASS_CODE_TO_NAME = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesions",
}

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CODE_TO_ID = {code: idx for idx, code in enumerate(CLASS_ORDER)}
ID_TO_CODE = {idx: code for code, idx in CODE_TO_ID.items()}

CLASS_NAMES = {
    idx: CLASS_CODE_TO_NAME[code]
    for idx, code in ID_TO_CODE.items()
}

# Nombres abreviados usados SOLO en títulos, leyendas y barras de color
# de las figuras SHAP-GLCD. No modifican las etiquetas del modelo,
# la selección del par ni ningún cálculo.
HAM_DISPLAY_CLASS_NAMES = {
    0: "AKIEC",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "VASC",
}

print("\nClass names:")
for k, v in CLASS_NAMES.items():
    print(f"{k}: {v}")


# ============================================================
# 7) Funciones auxiliares
# ============================================================
def safe_name(text):
    text = str(text)
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def build_image_path_map(root):
    root = Path(root)
    path_map = {}

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for img_path in root.rglob(ext):
            path_map[img_path.stem] = str(img_path)

    return path_map


def prepare_ham10000_dataframe(metadata_csv, dataset_root):
    df = pd.read_csv(metadata_csv)

    required_columns = {"image_id", "lesion_id", "dx"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "El metadata debe contener image_id, lesion_id y dx. "
            f"Faltan: {sorted(missing)}"
        )

    image_path_map = build_image_path_map(dataset_root)

    print("\nTotal image files found:", len(image_path_map))

    df["path"] = df["image_id"].map(image_path_map)
    df = df.dropna(subset=["path"]).copy()

    df = df[df["dx"].isin(CLASS_ORDER)].copy()
    df["label"] = df["dx"].map(CODE_TO_ID).astype(int)

    if df["image_id"].duplicated().any():
        dup_n = int(df["image_id"].duplicated().sum())
        raise ValueError(f"Se encontraron {dup_n} image_id duplicados.")

    # Comprobación esencial para poder estratificar a nivel de lesión.
    dx_per_lesion = df.groupby("lesion_id")["dx"].nunique()
    inconsistent = dx_per_lesion[dx_per_lesion > 1]
    if len(inconsistent) > 0:
        raise ValueError(
            "Hay lesion_id asociados a más de una clase diagnóstica. "
            f"Cantidad: {len(inconsistent)}"
        )

    print("\n✅ Metadata loaded")
    print("Total images with valid paths:", len(df))
    print("Unique lesions:", df["lesion_id"].nunique())

    print("\nImage distribution by class:")
    print(df["dx"].value_counts())

    lesion_distribution = (
        df.drop_duplicates("lesion_id")["dx"]
        .value_counts()
    )
    print("\nLesion distribution by class:")
    print(lesion_distribution)

    df[["image_id", "lesion_id", "dx", "label", "path"]].to_csv(
        os.path.join(OUTPUT_DIR, "ham10000_loaded_metadata_030926.csv"),
        index=False,
    )

    return df


def split_by_lesion(df, test_size=0.20, val_fraction_of_trainval=0.20):
    """
    Divide HAM10000 por lesion_id, no por imagen.

    Con test_size=0.20 y val_fraction_of_trainval=0.20:
      - ~64% de las lesiones -> train
      - ~16%                -> validation
      - ~20%                -> test

    La estratificación se realiza a nivel de lesión usando su etiqueta dx/label.
    """
    lesion_df = (
        df.groupby("lesion_id", as_index=False)
        .agg(
            dx=("dx", "first"),
            label=("label", "first"),
            n_images=("image_id", "size"),
        )
    )

    lesions_trainval, lesions_test = train_test_split(
        lesion_df,
        test_size=test_size,
        random_state=SEED,
        stratify=lesion_df["label"],
    )

    lesions_train, lesions_val = train_test_split(
        lesions_trainval,
        test_size=val_fraction_of_trainval,
        random_state=SEED,
        stratify=lesions_trainval["label"],
    )

    train_ids = set(lesions_train["lesion_id"])
    val_ids = set(lesions_val["lesion_id"])
    test_ids = set(lesions_test["lesion_id"])

    # Comprobación explícita de ausencia de fuga por lesión.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    df_train = df[df["lesion_id"].isin(train_ids)].copy()
    df_val = df[df["lesion_id"].isin(val_ids)].copy()
    df_test = df[df["lesion_id"].isin(test_ids)].copy()

    # Conservamos orden reproducible en validación/test.
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("\n============================================================")
    print(" TRAIN / VALIDATION / TEST SPLIT BY lesion_id")
    print("============================================================")
    print("Seed:", SEED)

    print("\nLesions:")
    print("Train      :", len(train_ids))
    print("Validation :", len(val_ids))
    print("Test       :", len(test_ids))
    print("Total      :", len(train_ids) + len(val_ids) + len(test_ids))

    print("\nImages:")
    print("Train      :", len(df_train))
    print("Validation :", len(df_val))
    print("Test       :", len(df_test))
    print("Total      :", len(df_train) + len(df_val) + len(df_test))

    print("\nLeakage check by lesion_id:")
    print("Train ∩ Validation =", len(train_ids.intersection(val_ids)))
    print("Train ∩ Test       =", len(train_ids.intersection(test_ids)))
    print("Validation ∩ Test  =", len(val_ids.intersection(test_ids)))

    print("\nTrain image distribution:")
    print(df_train["dx"].value_counts())

    print("\nValidation image distribution:")
    print(df_val["dx"].value_counts())

    print("\nTest image distribution:")
    print(df_test["dx"].value_counts())

    print("\nTrain lesion distribution:")
    print(lesions_train["dx"].value_counts())

    print("\nValidation lesion distribution:")
    print(lesions_val["dx"].value_counts())

    print("\nTest lesion distribution:")
    print(lesions_test["dx"].value_counts())

    # Guardar manifiesto completo de partición.
    manifest = df[["image_id", "lesion_id", "dx", "label", "path"]].copy()
    split_map = {}
    split_map.update({lesion_id: "train" for lesion_id in train_ids})
    split_map.update({lesion_id: "validation" for lesion_id in val_ids})
    split_map.update({lesion_id: "test" for lesion_id in test_ids})
    manifest["split"] = manifest["lesion_id"].map(split_map)
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)

    print("\nSplit manifest saved in:")
    print(SPLIT_MANIFEST_PATH)

    return df_train, df_val, df_test


def decode_resize_image(path, label):
    """
    ÚNICO preprocesamiento base usado tanto por tf.data como por SHAP.
    La arquitectura aplica después internamente ResNet50 preprocess_input.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(df, shuffle=False):
    paths = df["path"].values
    labels = df["label"].values.astype("int64")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        decode_resize_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def load_array_from_df(df, max_samples=None):
    """
    Carga imágenes como arrays NumPy para SHAP-GLCD usando EXACTAMENTE
    decode_resize_image(), es decir, el mismo preprocesamiento de evaluación.

    NO filtra por j_sel/r_sel. La clase filtra internamente mediante
    etiqueta verdadera y eval_pair_only=True.
    """
    df_use = df.copy()

    if max_samples is not None and len(df_use) > max_samples:
        df_use = df_use.sample(
            n=max_samples,
            random_state=SEED,
        ).reset_index(drop=True)

    print("\nDistribución cargada para SHAP antes del filtrado interno de la clase:")
    print(df_use["label"].value_counts().sort_index())

    x_list = []
    y_list = []

    for i, row in enumerate(df_use.itertuples(index=False)):
        if i % 100 == 0:
            print(f"Loading SHAP image {i}/{len(df_use)}")

        image_tensor, _ = decode_resize_image(
            tf.convert_to_tensor(str(row.path)),
            tf.convert_to_tensor(int(row.label), dtype=tf.int64),
        )

        x_list.append(image_tensor.numpy())
        y_list.append(int(row.label))

    x_arr = np.stack(x_list).astype("float32")
    y_arr = np.asarray(y_list, dtype="int64")

    return x_arr, y_arr


def merge_histories(history_1, history_2):
    """Une los historiales de ambas fases y conserva la fase por época."""
    rows = []

    n1 = len(history_1.history.get("loss", []))
    n2 = len(history_2.history.get("loss", []))

    for i in range(n1):
        rows.append({
            "epoch_global": i + 1,
            "phase": "head",
            "phase_epoch": i + 1,
            "loss": history_1.history["loss"][i],
            "accuracy": history_1.history["accuracy"][i],
            "val_loss": history_1.history["val_loss"][i],
            "val_accuracy": history_1.history["val_accuracy"][i],
        })

    for i in range(n2):
        rows.append({
            "epoch_global": n1 + i + 1,
            "phase": "fine_tuning",
            "phase_epoch": i + 1,
            "loss": history_2.history["loss"][i],
            "accuracy": history_2.history["accuracy"][i],
            "val_loss": history_2.history["val_loss"][i],
            "val_accuracy": history_2.history["val_accuracy"][i],
        })

    return pd.DataFrame(rows)


# ============================================================
# 8) Preparar datos
# ============================================================
df_ham = prepare_ham10000_dataframe(
    metadata_csv=METADATA_CSV,
    dataset_root=dataset_path,
)

df_train, df_val, df_test = split_by_lesion(
    df_ham,
    test_size=0.20,
    val_fraction_of_trainval=0.20,
)

ds_train = make_dataset(df_train, shuffle=True)
ds_val = make_dataset(df_val, shuffle=False)
ds_test = make_dataset(df_test, shuffle=False)


# ============================================================
# 9) Modelo ResNet50 con transfer learning
# ============================================================
def make_resnet50_ham10000(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_image")

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.03, 0.03),
        layers.RandomZoom(0.05),
    ], name="data_augmentation")

    x = data_augmentation(inputs)

    # Las imágenes externas llegan en [0,1].
    x = layers.Rescaling(255.0, name="to_255")(x)

    x = layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input,
        name="resnet50_preprocess",
    )(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    base_model.trainable = False

    x = base_model(x, training=False)

    x = layers.Dropout(0.40, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_HAM10000",
    )

    return model, base_model


num_classes = len(CLASS_NAMES)
input_shape = (IMG_SIZE, IMG_SIZE, 3)

print("\nInput shape:", input_shape)
print("Number of classes:", num_classes)

model, base_model = make_resnet50_ham10000(
    input_shape=input_shape,
    num_classes=num_classes,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# 10) Entrenar o cargar pesos
# ============================================================
if TRAIN_MODEL:
    print("\nEntrenando modelo corregido con split por lesion_id.")

    # --------------------------------------------------------
    # 10.1) Fase 1: entrenar cabeza clasificadora
    # --------------------------------------------------------
    callbacks_head = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_HEAD_WEIGHTS_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nEntrenando fase 1: classifier head...")

    history_head = model.fit(
        ds_train,
        epochs=EPOCHS_HEAD,
        validation_data=ds_val,
        callbacks=callbacks_head,
        class_weight=None,
        verbose=1,
    )

    model.load_weights(BEST_HEAD_WEIGHTS_PATH)
    print(f"\n✅ Best head weights loaded from: {BEST_HEAD_WEIGHTS_PATH}")

    # --------------------------------------------------------
    # 10.2) Fase 2: fine-tuning parcial de ResNet50
    # --------------------------------------------------------
    print("\nEntrenando fase 2: fine-tuning parcial...")

    base_model.trainable = True

    fine_tune_at = len(base_model.layers) - 50

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # BatchNormalization permanece congelado para un fine-tuning estable.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_fine = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_FINE_WEIGHTS_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history_fine = model.fit(
        ds_train,
        epochs=EPOCHS_FINE,
        validation_data=ds_val,
        callbacks=callbacks_fine,
        class_weight=None,
        verbose=1,
    )

    # --------------------------------------------------------
    # 10.3) Seleccionar mejores pesos globales SOLO con validación
    # --------------------------------------------------------
    candidate_paths = []

    if os.path.exists(BEST_HEAD_WEIGHTS_PATH):
        candidate_paths.append(("head", BEST_HEAD_WEIGHTS_PATH))

    if os.path.exists(BEST_FINE_WEIGHTS_PATH):
        candidate_paths.append(("fine", BEST_FINE_WEIGHTS_PATH))

    if not candidate_paths:
        raise FileNotFoundError("No se generaron pesos candidatos.")

    best_candidate_name = None
    best_candidate_path = None
    best_val_acc = -np.inf

    print("\nEvaluando candidatos de mejores pesos sobre VALIDACIÓN:")

    for name, path in candidate_paths:
        model.load_weights(path)
        val_loss, val_acc = model.evaluate(ds_val, verbose=0)

        print(f"{name}: val_accuracy={val_acc:.4f} | val_loss={val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_candidate_name = name
            best_candidate_path = path

    model.load_weights(best_candidate_path)
    model.save_weights(BEST_OVERALL_WEIGHTS_PATH)
    model.save_weights(FINAL_WEIGHTS_PATH)

    print("\n✅ Mejor candidato global seleccionado SOLO con validación:")
    print("Fase:", best_candidate_name)
    print("Path:", best_candidate_path)
    print(f"val_accuracy={best_val_acc:.4f}")
    print("Guardado como:", BEST_OVERALL_WEIGHTS_PATH)
    print("Copia final:", FINAL_WEIGHTS_PATH)

    # --------------------------------------------------------
    # 10.4) Guardar historial y curvas Loss / Accuracy
    # --------------------------------------------------------
    history_df = merge_histories(history_head, history_fine)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)

    print("\nTraining history saved in:")
    print(HISTORY_CSV_PATH)

    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch_global"], history_df["loss"], label="Train loss")
    plt.plot(history_df["epoch_global"], history_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Loss curve - HAM10000 ResNet50", fontsize=22)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "ham10000_resnet50_loss_curve_lesion_split_030926.png"),
        dpi=DPI,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch_global"], history_df["accuracy"], label="Train accuracy")
    plt.plot(history_df["epoch_global"], history_df["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Accuracy curve - HAM10000 ResNet50", fontsize=22)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "ham10000_resnet50_accuracy_curve_lesion_split_030926.png"),
        dpi=DPI,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

else:
    if not os.path.exists(BEST_OVERALL_WEIGHTS_PATH):
        raise FileNotFoundError(
            "TRAIN_MODEL=False, pero no existen los pesos corregidos: "
            f"{BEST_OVERALL_WEIGHTS_PATH}\n"
            "Ejecuta primero con TRAIN_MODEL=True y SHAP_MODE='always'."
        )

    print("\nCargando pesos corregidos previamente entrenados.")
    print("Path:", BEST_OVERALL_WEIGHTS_PATH)
    model.load_weights(BEST_OVERALL_WEIGHTS_PATH)


# ============================================================
# 11) Evaluación FINAL en TEST independiente
# ============================================================
print("\n============================================================")
print(" FINAL MODEL EVALUATION - INDEPENDENT TEST SET")
print("============================================================")

# Loss/accuracy Keras
test_loss, keras_test_acc = model.evaluate(ds_test, verbose=0)

print("Test images:", len(df_test))
print(f"Keras test loss     : {test_loss:.6f}")
print(f"Keras test accuracy : {keras_test_acc:.6f}")

print("\nGenerando predicciones en test...")
y_test_true = df_test["label"].values.astype("int64")
y_pred_prob = model.predict(ds_test, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test_true, y_pred)
balanced_acc = balanced_accuracy_score(y_test_true, y_pred)
f1_macro = f1_score(y_test_true, y_pred, average="macro")
mcc = matthews_corrcoef(y_test_true, y_pred)

try:
    macro_auc_ovo = roc_auc_score(
        y_test_true,
        y_pred_prob,
        multi_class="ovo",
        average="macro",
        labels=np.arange(num_classes),
    )
except ValueError as e:
    print("⚠️ No fue posible calcular Macro-AUC OvO:", e)
    macro_auc_ovo = np.nan

print("\nFinal test metrics:")
print(f"Loss                : {test_loss:.6f}")
print(f"Accuracy            : {accuracy:.6f}")
print(f"Balanced accuracy   : {balanced_acc:.6f}")
print(f"F1-score macro      : {f1_macro:.6f}")
print(f"MCC                 : {mcc:.6f}")
print(f"Macro-AUC (OvO)     : {macro_auc_ovo:.6f}")

metrics_df = pd.DataFrame([{
    "test_loss": test_loss,
    "accuracy": accuracy,
    "balanced_accuracy": balanced_acc,
    "f1_macro": f1_macro,
    "mcc": mcc,
    "macro_auc_ovo": macro_auc_ovo,
    "n_test_images": len(df_test),
    "n_test_lesions": df_test["lesion_id"].nunique(),
    "seed": SEED,
}])
metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "ham10000_resnet50_test_metrics_lesion_split_030926.csv"),
    index=False,
)


# ============================================================
# 12) Classification report y matriz de confusión
# ============================================================
target_names = [CLASS_NAMES[i] for i in range(num_classes)]

report_text = classification_report(
    y_test_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=target_names,
    digits=4,
    zero_division=0,
)

report_dict = classification_report(
    y_test_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_test_true,
    y_pred,
    labels=np.arange(num_classes),
)

print("\nClassification report:")
print(report_text)

print("\nConfusion matrix:")
print(cm)

with open(
    os.path.join(OUTPUT_DIR, "ham10000_resnet50_classification_report_lesion_split_030926.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write(report_text)

pd.DataFrame(report_dict).T.to_csv(
    os.path.join(OUTPUT_DIR, "ham10000_resnet50_classification_report_lesion_split_030926.csv")
)

pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names,
).to_csv(
    os.path.join(OUTPUT_DIR, "ham10000_resnet50_confusion_matrix_lesion_split_030926.csv")
)

# Liberar probabilidades/predicciones antes de SHAP.
del y_pred_prob
del y_pred
gc.collect()


# ============================================================
# 13) SHAP-GLCD para HAM10000
# ============================================================
if RUN_SHAP:

    print("\nPreparando arrays para SHAP-GLCD...")

    # --------------------------------------------------------
    # Background pool: SOLO TRAIN
    # --------------------------------------------------------
    x_train_shap, y_train_shap = load_array_from_df(
        df_train,
        max_samples=SHAP_TRAIN_POOL,
    )

    # --------------------------------------------------------
    # Evaluation source: TODO EL TEST independiente.
    # La clase filtrará por true labels j_sel/r_sel.
    # --------------------------------------------------------
    x_test_shap, y_test_shap = load_array_from_df(
        df_test,
        max_samples=None,
    )

    print("x_train_shap:", x_train_shap.shape)
    print("x_test_shap: ", x_test_shap.shape)

    print("\nClases presentes en y_test_shap antes del filtrado interno:")
    print(pd.Series(y_test_shap).value_counts().sort_index())

    # --------------------------------------------------------
    # BALANCE ESTRICTO DEL PAR
    # --------------------------------------------------------
    n_available_j = int(np.sum(y_test_shap == j_sel))
    n_available_r = int(np.sum(y_test_shap == r_sel))

    if n_available_j == 0 or n_available_r == 0:
        raise ValueError(
            f"No hay suficientes imágenes en test para el par "
            f"{CLASS_NAMES[j_sel]} vs {CLASS_NAMES[r_sel]}."
        )

    n_each = min(
        n_available_j,
        n_available_r,
        SHAP_MAX_TEST // 2,
    )

    pair_max_test_balanced = 2 * n_each

    print("\n============================================================")
    print(" SHAP-GLCD PAIR SELECTION CONFIGURATION")
    print("============================================================")
    print("Pair              :", CLASS_NAMES[j_sel], "vs", CLASS_NAMES[r_sel])
    print("Selection source  : independent test set")
    print("Selection criterion: TRUE LABEL")
    print("Prediction filter : NO")
    print("Seed              :", SEED)
    print("Available class j :", n_available_j)
    print("Available class r :", n_available_r)
    print("Selected class j  :", n_each)
    print("Selected class r  :", n_each)
    print("Balanced pair N   :", pair_max_test_balanced)
    print("Requested max N   :", SHAP_MAX_TEST)

    pair_txt = f"{CLASS_NAMES[j_sel]}_vs_{CLASS_NAMES[r_sel]}"
    pair_txt_safe = safe_name(pair_txt)

    out_dir = os.path.join(
        OUTPUT_DIR,
        "GLCD_HAM_BCC_vs_BKL"
    )

    shap_pickle = (
            ARTIFACT_DIR
            / (
                "shap_values_ham10000_resnet50_"
                f"lesion_split_030926_{j_sel}_vs_{r_sel}.pickle"
            )
    )

    explainer = SHAP_GLCD_ImageExplainer_MapsOnly(
        model=model,
        x_train=x_train_shap,
        x_test=x_test_shap,
        y_test=y_test_shap,
        j_sel=j_sel,
        r_sel=r_sel,
        class_names=CLASS_NAMES,
        shap_pickle=shap_pickle,
        out_dir=out_dir,
        dpi=DPI,
        random_state=SEED,
        show_plots=SHOW_PLOTS,
        display_class_names=HAM_DISPLAY_CLASS_NAMES,
    )

    compute_kwargs = {
        "background_size": SHAP_BACKGROUND_SIZE,
        "max_test": pair_max_test_balanced,
        "explainer_type": "gradient",
        "eval_pair_only": True,
    }

    top_percent = 12.0

    print("\nEjecutando SHAP-GLCD con el modelo ResNet50 definitivo...")
    print("SHAP background size:", SHAP_BACKGROUND_SIZE)
    print("SHAP pickle         :", shap_pickle)
    print("SHAP mode           :", SHAP_MODE)
    print("Output dir          :", out_dir)
    print("x_test_shap loaded  :", len(x_test_shap))
    print("Balanced max_test   :", pair_max_test_balanced)
    print(
        "Display pair        :",
        HAM_DISPLAY_CLASS_NAMES[j_sel],
        "vs",
        HAM_DISPLAY_CLASS_NAMES[r_sel],
    )

    gc.collect()

    explainer.run_all_maps(
        shap_mode=SHAP_MODE,
        compute_kwargs=compute_kwargs,
        top_percent=top_percent,
        make_parallel=True,
    )

else:
    print("\nRUN_SHAP=False. No se ejecutó SHAP-GLCD.")
