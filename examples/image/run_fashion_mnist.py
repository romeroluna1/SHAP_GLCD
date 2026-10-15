# ============================================================
# main_fashion_glcd_v2_CORREGIDO_TRAIN_VAL_TEST_020926.py
# ============================================================
#
# Fashion-MNIST + CNN + SHAP-GLCD
#
# CORRECCIONES METODOLÓGICAS:
#
# 1) El test oficial de Fashion-MNIST (10,000 imágenes)
#    NO se usa para validación ni selección del mejor modelo.
#
# 2) Del conjunto oficial de entrenamiento (60,000 imágenes)
#    se separa un 10 % para validación:
#
#       54,000 entrenamiento
#        6,000 validación
#       10,000 test oficial
#
# 3) EarlyStopping y ModelCheckpoint usan exclusivamente
#    el conjunto de validación.
#
# 4) Las métricas finales se calculan sobre las 10,000
#    imágenes del test oficial.
#
# 5) SHAP-GLCD se calcula posteriormente sobre el test oficial.
#
# 6) Primera ejecución definitiva:
#
#       TRAIN_MODEL = True
#       SHAP_MODE   = "always"
#
#    Después de generar el modelo y los SHAP definitivos:
#
#       TRAIN_MODEL = False
#       SHAP_MODE   = "auto"
#
# IMPORTANTE:
# No usar TRAIN_MODEL=True junto con SHAP_MODE="auto",
# porque podría entrenarse una CNN nueva y reutilizarse
# un pickle SHAP correspondiente a un modelo anterior.
# ============================================================


import os
import random
import re
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



from shap_glcd.image import SHAP_GLCD_ImageExplainer_MapsOnly



# ============================================================
# Carpetas del repositorio
# ============================================================

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "fashion_mnist"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "fashion_mnist"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

# 10 % del conjunto oficial de entrenamiento se usa
# exclusivamente para validación.
VALIDATION_SIZE = 0.10

SHOW_PLOTS = True
DPI = 600


# ============================================================
# CONFIGURACIÓN DE EJECUCIÓN
# ============================================================
#
# PRIMERA EJECUCIÓN DEFINITIVA:
#
# TRAIN_MODEL = True
# SHAP_MODE   = "always"
#
# EJECUCIONES POSTERIORES:
#
# TRAIN_MODEL = False
# SHAP_MODE   = "auto"
#
# ============================================================

TRAIN_MODEL = False

RUN_SHAP = True

SHAP_MODE = "auto"


# ============================================================
# Par de clases SHAP-GLCD
# ============================================================
#
# Fashion-MNIST:
#
# 0 = T-shirt/Top
# 1 = Trouser
# 2 = Pullover
# 3 = Dress
# 4 = Coat
# 5 = Sandal
# 6 = Shirt
# 7 = Sneaker
# 8 = Bag
# 9 = Ankle Boot
#
# ============================================================

j_sel = 0     # T-shirt/Top
r_sel = 1     # Trouser


# ============================================================
# Parámetros SHAP
# ============================================================

SHAP_BACKGROUND_SIZE = 100

# No se filtra el par en el main.
# Se pasa el test oficial completo y la clase realiza
# internamente la selección del par.
SHAP_MAX_TEST = None


# ============================================================
# Nombres NUEVOS de archivos definitivos
# ============================================================
#
# Se usan nombres distintos de la corrida anterior para evitar
# mezclar el modelo corregido con archivos SHAP antiguos.
#
# ============================================================

BEST_MODEL_PATH = (
    ARTIFACT_DIR
    / "fashion_cnn_best_corrected_train_val_test.keras"
)

FINAL_MODEL_PATH = (
    ARTIFACT_DIR
    / "fashion_cnn_final_corrected_train_val_test.keras"
)


# ============================================================
# 3) Datos Fashion-MNIST
# ============================================================

(
    (x_train_full, y_train_full),
    (x_test, y_test)
) = tf.keras.datasets.fashion_mnist.load_data()


# ------------------------------------------------------------
# Normalización
# ------------------------------------------------------------

x_train_full = (
    x_train_full.astype("float32") / 255.0
)[..., np.newaxis]

x_test = (
    x_test.astype("float32") / 255.0
)[..., np.newaxis]


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
    9: "Ankle Boot"
}


print("\n============================================================")
print(" Fashion-MNIST loaded")
print("============================================================")

print(
    "Official training set:",
    x_train_full.shape,
    "|",
    y_train_full.shape
)

print(
    "Official test set    :",
    x_test.shape,
    "|",
    y_test.shape
)


# ============================================================
# 4) Separación TRAIN / VALIDATION
# ============================================================
#
# El test oficial NO participa en esta separación.
#
# 60,000 ->
#     54,000 entrenamiento
#      6,000 validación
#
# ============================================================

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


print("\n============================================================")
print(" TRAIN / VALIDATION / TEST SPLIT")
print("============================================================")

print(
    "Training   :",
    x_train.shape,
    "|",
    y_train.shape
)

print(
    "Validation :",
    x_val.shape,
    "|",
    y_val.shape
)

print(
    "Test       :",
    x_test.shape,
    "|",
    y_test.shape
)

print("Seed       :", SEED)


# ============================================================
# 5) Funciones auxiliares
# ============================================================

def safe_name(text):
    """
    Evita caracteres problemáticos en nombres de carpetas
    y archivos.
    """

    text = str(text)

    text = text.replace("/", "_")
    text = text.replace("\\", "_")

    text = re.sub(
        r"[^A-Za-z0-9_\-]+",
        "_",
        text
    )

    text = re.sub(
        r"_+",
        "_",
        text
    )

    return text.strip("_")


def print_distribution(y, title):

    print(f"\n{title}")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for cls, count in zip(
        unique,
        counts
    ):
        print(
            f"{int(cls)} "
            f"({CLASS_NAMES[int(cls)]}): "
            f"{count}"
        )


# ============================================================
# Distribuciones
# ============================================================

print_distribution(
    y_train,
    "Distribución del conjunto de entrenamiento:"
)

print_distribution(
    y_val,
    "Distribución del conjunto de validación:"
)

print_distribution(
    y_test,
    "Distribución del test oficial:"
)


# ============================================================
# 6) Modelo CNN
# ============================================================
#
# Se conserva EXACTAMENTE la arquitectura del main anterior.
#
# ============================================================

def make_fashion_cnn(
    input_shape=(28, 28, 1),
    num_classes=10
):

    model = models.Sequential([

        layers.Input(
            shape=input_shape
        ),

        layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dense(
            num_classes,
            activation="softmax"
        ),
    ])

    return model


# ============================================================
# 7) Protección modelo nuevo / SHAP antiguos
# ============================================================

if TRAIN_MODEL and SHAP_MODE == "auto":

    raise ValueError(
        "\nCONFIGURACIÓN NO PERMITIDA:\n\n"
        "TRAIN_MODEL=True y SHAP_MODE='auto'.\n\n"
        "Si vas a entrenar una CNN nueva, debes usar:\n\n"
        "    SHAP_MODE='always'\n\n"
        "para garantizar que los SHAP correspondan "
        "exactamente al modelo recién entrenado."
    )


# ============================================================
# 8) Entrenar o cargar modelo
# ============================================================

need_training = (
    TRAIN_MODEL
    or not os.path.exists(BEST_MODEL_PATH)
)


if need_training:

    print("\n============================================================")
    print(" Entrenando CNN para Fashion-MNIST")
    print("============================================================")

    print(
        "Training instances   :",
        len(x_train)
    )

    print(
        "Validation instances :",
        len(x_val)
    )

    print(
        "Official test        :",
        len(x_test),
        "(NO usado durante entrenamiento)"
    )


    model = make_fashion_cnn(
        input_shape=(28, 28, 1),
        num_classes=10
    )


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    model.summary()


    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            patience=5,

            restore_best_weights=True,

            mode="max",

            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(

            BEST_MODEL_PATH,

            monitor="val_accuracy",

            save_best_only=True,

            mode="max",

            verbose=1
        )
    ]


    history = model.fit(

        x_train,
        y_train,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        validation_data=(
            x_val,
            y_val
        ),

        callbacks=callbacks,

        verbose=1
    )


    # --------------------------------------------------------
    # Cargar explícitamente el mejor modelo guardado
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    print(
        f"\n✅ Mejor modelo cargado desde: "
        f"{BEST_MODEL_PATH}"
    )


    model.save(
        FINAL_MODEL_PATH
    )

    print(
        f"\n✅ Mejor modelo guardado como: "
        f"{FINAL_MODEL_PATH}"
    )


    # ========================================================
    # Curva LOSS
    # ========================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Train loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel(
        "Epoch",
        fontsize=18
    )

    plt.ylabel(
        "Loss",
        fontsize=18
    )

    plt.xticks(
        fontsize=14
    )

    plt.yticks(
        fontsize=14
    )

    plt.title(
        "Loss curve - Fashion-MNIST CNN",
        fontsize=22
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.legend(
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        "fashion_cnn_loss_curve_corrected.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS:

        plt.show()

    else:

        plt.close()


    # ========================================================
    # Curva ACCURACY
    # ========================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        history.history["accuracy"],
        label="Train accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation accuracy"
    )

    plt.xlabel(
        "Epoch",
        fontsize=18
    )

    plt.ylabel(
        "Accuracy",
        fontsize=18
    )

    plt.xticks(
        fontsize=14
    )

    plt.yticks(
        fontsize=14
    )

    plt.title(
        "Accuracy curve - Fashion-MNIST CNN",
        fontsize=22
    )

    plt.legend(
        fontsize=16
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        "fashion_cnn_accuracy_curve_corrected.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS:

        plt.show()

    else:

        plt.close()


else:

    print("\n============================================================")
    print(" Cargando CNN previamente entrenada")
    print("============================================================")

    print(
        "Path:",
        BEST_MODEL_PATH
    )

    model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    print(
        f"\n✅ Mejor modelo cargado desde: "
        f"{BEST_MODEL_PATH}"
    )


# ============================================================
# 9) EVALUACIÓN FINAL EN TEST OFICIAL
# ============================================================
#
# Este conjunto NO participó en:
#
# - entrenamiento;
# - EarlyStopping;
# - ModelCheckpoint;
# - selección de época.
#
# ============================================================

print("\n============================================================")
print(" FINAL MODEL EVALUATION - OFFICIAL TEST SET")
print("============================================================")

print(
    "Test size:",
    len(y_test)
)


# ------------------------------------------------------------
# Loss + accuracy de Keras
# ------------------------------------------------------------

test_loss, keras_test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


# ------------------------------------------------------------
# Probabilidades
# ------------------------------------------------------------

y_prob = model.predict(
    x_test,
    batch_size=BATCH_SIZE,
    verbose=0
)


# ------------------------------------------------------------
# Predicciones
# ------------------------------------------------------------

y_pred = np.argmax(
    y_prob,
    axis=1
)


# ------------------------------------------------------------
# Métricas
# ------------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

test_f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro"
)

test_mcc = matthews_corrcoef(
    y_test,
    y_pred
)

test_auc_ovo = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovo",
    average="macro"
)


test_cm = confusion_matrix(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# Mostrar resultados
# ------------------------------------------------------------

print(
    f"Loss                : "
    f"{test_loss:.6f}"
)

print(
    f"Accuracy            : "
    f"{test_accuracy:.6f}"
)

print(
    f"Balanced accuracy   : "
    f"{test_balanced_accuracy:.6f}"
)

print(
    f"F1-score macro      : "
    f"{test_f1_macro:.6f}"
)

print(
    f"MCC                 : "
    f"{test_mcc:.6f}"
)

print(
    f"Macro-AUC (OvO)     : "
    f"{test_auc_ovo:.6f}"
)

print(
    "\nConfusion matrix:"
)

print(
    test_cm
)

print(
    "\nKeras test accuracy : "
    f"{keras_test_acc:.6f}"
)

print(
    "============================================================"
)


# ============================================================
# 10) SHAP-GLCD
# ============================================================

if RUN_SHAP:

    print_distribution(
        y_test,
        (
            "Distribución completa de y_test antes "
            "del filtrado interno de la clase:"
        )
    )


    pair_txt = (
        f"{CLASS_NAMES[j_sel]}"
        f"_vs_"
        f"{CLASS_NAMES[r_sel]}"
    )

    pair_txt_safe = safe_name(
        pair_txt
    )


    out_dir = RESULTS_DIR


    # --------------------------------------------------------
    # Pickle NUEVO y específico para:
    #
    # - modelo corregido
    # - dataset
    # - par de clases
    #
    # --------------------------------------------------------

    shap_pickle = (
            ARTIFACT_DIR
            / (
                "shap_values_fashion_corrected_"
                f"{j_sel}_vs_{r_sel}.pickle"
            )
    )


    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # x_train es ahora SOLO el conjunto utilizado realmente
    # para ajustar los pesos del modelo: 54,000 imágenes.
    #
    # No usamos x_val ni x_test como background.
    # --------------------------------------------------------

    explainer = SHAP_GLCD_ImageExplainer_MapsOnly(

        model=model,

        x_train=x_train,

        x_test=x_test,

        y_test=y_test,

        j_sel=j_sel,

        r_sel=r_sel,

        class_names=CLASS_NAMES,

        shap_pickle=shap_pickle,

        out_dir=out_dir,

        dpi=DPI,

        random_state=SEED,

        show_plots=SHOW_PLOTS
    )


    if SHAP_MAX_TEST is None:

        max_test_value = len(
            x_test
        )

    else:

        max_test_value = SHAP_MAX_TEST


    compute_kwargs = {

        "background_size":
            SHAP_BACKGROUND_SIZE,

        "max_test":
            max_test_value,

        "explainer_type":
            "gradient",

        "eval_pair_only":
            True
    }


    top_percent = 12.0


    print("\n============================================================")
    print(" SHAP-GLCD - Fashion-MNIST")
    print("============================================================")

    print(
        "Pair:",
        CLASS_NAMES[j_sel],
        "vs",
        CLASS_NAMES[r_sel]
    )

    print(
        "Selection source : official test set"
    )

    print(
        "Pair selection   : true labels"
    )

    print(
        "Prediction filter: no"
    )

    print(
        "Seed             :",
        SEED
    )

    print(
        "SHAP background  :",
        SHAP_BACKGROUND_SIZE
    )

    print(
        "SHAP pickle      :",
        shap_pickle
    )

    print(
        "SHAP mode        :",
        SHAP_MODE
    )

    print(
        "Output dir       :",
        out_dir
    )

    print(
        "max_test before internal pair filtering:",
        max_test_value
    )


    explainer.run_all_maps(

        shap_mode=SHAP_MODE,

        compute_kwargs=compute_kwargs,

        top_percent=top_percent,

        make_parallel=True
    )


else:

    print(
        "\nRUN_SHAP=False. "
        "No se ejecutó SHAP-GLCD."
    )