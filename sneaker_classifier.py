"""
Sneaker Model Classifier
========================
Dataset: ~6.500 imagens, 50 classes, resolucao original ~140x125px
Modelos: CNN customizada (3 configs) + MobileNetV2 com fine-tuning em 2 fases
"""

import os
import sys
import argparse
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # sem display necessario

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print(f"TensorFlow: {tf.__version__}")
print(f"GPU disponivel: {bool(tf.config.list_physical_devices('GPU'))}")


# =============================================================================
# CONFIGURACAO
# =============================================================================

# Tamanho reduzido: imagens originais tem media de ~140x125px.
# Fazer upscale para 224x224 nao adiciona informacao real e aumenta o custo.
# 96x96 ja captura o que existe sem distorcao excessiva.
IMG_SIZE     = (96, 96)
BATCH_SIZE   = 32
EPOCHS_HEAD  = 20   # fase 1: treinar so o head (MobileNetV2 congelado)
EPOCHS_FINE  = 20   # fase 2: fine-tuning das ultimas camadas
FINE_TUNE_AT = 100  # descongelar camadas a partir deste indice (MobileNetV2 tem 154)

RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# ARGUMENTO DE ENTRADA
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sneaker classifier")
    parser.add_argument(
        "--zip",
        default="dataset_marca.zip",
        help="Caminho para o arquivo .zip do dataset (default: dataset_marca.zip)"
    )
    return parser.parse_args()


# =============================================================================
# EXTRACAO DO DATASET
# =============================================================================

def extract_dataset(zip_path: str, extract_to: str = "dataset_marca") -> str:
    if not os.path.exists(zip_path):
        sys.exit(f"[ERRO] Arquivo nao encontrado: {zip_path}\n"
                 f"Use --zip para especificar o caminho correto.")

    print(f"Extraindo {zip_path} -> {extract_to}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # Detecta o diretorio raiz real dentro do zip (pode ter subpastas)
    candidates = [
        os.path.join(extract_to, d)
        for d in os.listdir(extract_to)
        if os.path.isdir(os.path.join(extract_to, d))
    ]
    if len(candidates) == 1:
        inner = candidates[0]
        # Verifica se ja tem subpastas de classe diretamente
        sub = [os.path.join(inner, d) for d in os.listdir(inner) if os.path.isdir(os.path.join(inner, d))]
        if sub and os.path.isdir(sub[0]):
            # dataset_marca/sneakers-dataset/sneakers-dataset/<classes>
            sub2 = [os.path.join(sub[0], d) for d in os.listdir(sub[0]) if os.path.isdir(os.path.join(sub[0], d))]
            if sub2:
                return sub[0]
        return inner

    return extract_to


# =============================================================================
# GENERATORS
# =============================================================================

def build_generators(base_dir: str):
    """
    Cria generators com augmentation real no treino.
    Validacao: apenas rescale (sem augmentation, para avaliacao justa).
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        # Augmentation adequada para fotos de produto em fundo branco/neutro
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    common = dict(
        directory=base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=42,
    )

    train_gen = train_datagen.flow_from_directory(**common, subset="training")
    val_gen   = val_datagen.flow_from_directory(**common, subset="validation")

    num_classes = train_gen.num_classes
    print(f"\nClasses encontradas: {num_classes}")
    print(f"Amostras treino: {train_gen.samples} | Validacao: {val_gen.samples}")


    # Salva mapeamento index -> nome da classe para uso no predict.py
    index_to_class = {str(v): k for k, v in train_gen.class_indices.items()}
    labels_path = os.path.join(RESULTS_DIR, "class_labels.json")
    with open(labels_path, "w") as f:
        json.dump(index_to_class, f, indent=2, ensure_ascii=False)
    print(f"Labels salvos: {labels_path}")
    return train_gen, val_gen, num_classes


# =============================================================================
# CALLBACKS PADRAO
# =============================================================================

def get_callbacks(model_path: str):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# =============================================================================
# CNN CUSTOMIZADA
# =============================================================================

def build_custom_cnn(
    num_classes: int,
    num_conv_layers: int = 3,
    num_filters: int = 64,
    dropout_rate: float = 0.5,
) -> Sequential:
    """
    CNN customizada com BatchNormalization apos cada bloco conv.
    BatchNorm estabiliza o treino e frequentemente substitui a necessidade
    de LR muito baixo no inicio.
    """
    model = Sequential(name=f"CNN_{num_conv_layers}conv_{num_filters}f")

    for i in range(num_conv_layers):
        if i == 0:
            model.add(Conv2D(num_filters, (3, 3), activation="relu",
                             padding="same", input_shape=(*IMG_SIZE, 3)))
        else:
            model.add(Conv2D(num_filters * (2 ** min(i, 2)), (3, 3),
                             activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))

    return model


# =============================================================================
# MOBILENETV2 COM FINE-TUNING EM 2 FASES
# =============================================================================

def build_mobilenetv2_head(num_classes: int) -> Model:
    """
    Fase 1: base congelada, treina apenas o head.
    Retorna o modelo completo com base congelada.
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output, name="MobileNetV2")
    return model, base


def unfreeze_for_fine_tuning(model: Model, base_model, fine_tune_at: int) -> Model:
    """
    Fase 2: descongela camadas a partir de fine_tune_at.
    Usa LR menor para nao destruir os pesos pre-treinados.
    """
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\nFine-tuning: {trainable_count} camadas descongeladas (de {len(base_model.layers)} na base)")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # LR muito menor na fase 2
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# PLOT DE CURVAS
# =============================================================================

def plot_history(history, title: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="treino")
    axes[0].plot(history.history["val_accuracy"], label="validacao")
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="treino")
    axes[1].plot(history.history["val_loss"], label="validacao")
    axes[1].set_title(f"{title} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Curva salva: {save_path}")


# =============================================================================
# EXPERIMENTOS CNN
# =============================================================================

def run_cnn_experiments(train_gen, val_gen, num_classes: int) -> list:
    configs = [
        {"num_conv_layers": 2, "num_filters": 32},
        {"num_conv_layers": 3, "num_filters": 64},
        {"num_conv_layers": 4, "num_filters": 64},
    ]

    resultados = []

    for cfg in configs:
        nome = f"CNN_{cfg['num_conv_layers']}conv_{cfg['num_filters']}f"
        print(f"\n{'='*50}")
        print(f"Experimento: {nome}")
        print(f"{'='*50}")

        model = build_custom_cnn(num_classes=num_classes, **cfg)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary(print_fn=lambda x: None)  # suprimir output longo

        model_path = os.path.join(RESULTS_DIR, f"{nome}.keras")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_HEAD,
            callbacks=get_callbacks(model_path),
            verbose=1,
        )

        plot_history(
            history,
            title=nome,
            save_path=os.path.join(RESULTS_DIR, f"{nome}_curvas.png"),
        )

        best_val_acc = max(history.history["val_accuracy"])
        best_val_loss = min(history.history["val_loss"])
        epochs_run = len(history.history["loss"])

        print(f"\nResultado {nome}: val_acc={best_val_acc:.4f} | epochs={epochs_run}")

        resultados.append({
            "modelo": nome,
            "val_acc": round(best_val_acc, 4),
            "val_loss": round(best_val_loss, 4),
            "epochs_executadas": epochs_run,
            "early_stopped": epochs_run < EPOCHS_HEAD,
        })

    return resultados


# =============================================================================
# EXPERIMENTO MOBILENETV2
# =============================================================================

def run_mobilenetv2(train_gen, val_gen, num_classes: int) -> dict:
    print(f"\n{'='*50}")
    print("MobileNetV2 - Fase 1: treino do head")
    print(f"{'='*50}")

    # Nota: MobileNetV2 funciona bem com 96x96 (foi treinado com imagens de 96x96+).
    # Nao e necessario 224x224 para transfer learning funcionar.

    model, base = build_mobilenetv2_head(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = os.path.join(RESULTS_DIR, "MobileNetV2.keras")
    history_fase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(model_path),
        verbose=1,
    )

    plot_history(
        history_fase1,
        title="MobileNetV2 Fase 1 (head)",
        save_path=os.path.join(RESULTS_DIR, "MobileNetV2_fase1_curvas.png"),
    )

    # Fase 2: fine-tuning
    print(f"\n{'='*50}")
    print(f"MobileNetV2 - Fase 2: fine-tuning (camadas >= {FINE_TUNE_AT})")
    print(f"{'='*50}")

    model = unfreeze_for_fine_tuning(model, base, fine_tune_at=FINE_TUNE_AT)

    history_fase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks(model_path),
        verbose=1,
    )

    plot_history(
        history_fase2,
        title="MobileNetV2 Fase 2 (fine-tuning)",
        save_path=os.path.join(RESULTS_DIR, "MobileNetV2_fase2_curvas.png"),
    )

    best_val_acc  = max(history_fase2.history["val_accuracy"])
    best_val_loss = min(history_fase2.history["val_loss"])
    epochs_run    = len(history_fase2.history["loss"])

    print(f"\nResultado MobileNetV2 fase 2: val_acc={best_val_acc:.4f} | epochs={epochs_run}")

    return {
        "modelo": "MobileNetV2 (fine-tuning)",
        "val_acc": round(best_val_acc, 4),
        "val_loss": round(best_val_loss, 4),
        "epochs_executadas": epochs_run,
        "early_stopped": epochs_run < EPOCHS_FINE,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    base_dir = extract_dataset(args.zip)
    print(f"\nDataset base_dir: {base_dir}")

    train_gen, val_gen, num_classes = build_generators(base_dir)

    resultados = run_cnn_experiments(train_gen, val_gen, num_classes)
    resultado_mv2 = run_mobilenetv2(train_gen, val_gen, num_classes)
    resultados.append(resultado_mv2)

    # Tabela de resultados
    df = pd.DataFrame(resultados).sort_values("val_acc", ascending=False)
    csv_path = os.path.join(RESULTS_DIR, "resultados.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*50}")
    print("COMPARACAO FINAL")
    print(f"{'='*50}")
    print(df.to_string(index=False))
    print(f"\nResultados salvos em: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()