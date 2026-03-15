import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

print("TensorFlow:", tf.__version__)

# =========================
# EXTRAÇÃO DO DATASET
# =========================

with zipfile.ZipFile("dataset_marca.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset_marca")

base_dir = "dataset_marca"

# =========================
# DATA GENERATORS
# =========================

# Generator para categorical_crossentropy (one-hot)
datagen_cat = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator_cat = datagen_cat.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator_cat = datagen_cat.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Generator para sparse_categorical_crossentropy (índices inteiros)
datagen_sparse = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator_sparse = datagen_sparse.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_generator_sparse = datagen_sparse.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# =========================
# VISUALIZAÇÃO DO DATASET
# =========================

x, y = next(train_generator_cat)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x[i])
    label_index = np.argmax(y[i])
    label_name = list(train_generator_cat.class_indices.keys())[label_index]
    plt.title(label_name)
    plt.axis("off")

plt.show()

# =========================
# CNN CUSTOMIZADA
# =========================

def build_custom_cnn(input_shape, num_classes, num_conv_layers=3, num_filters=64, activation='relu'):
    model = Sequential()

    for i in range(num_conv_layers):
        if i == 0:
            model.add(Conv2D(num_filters, (3, 3), activation=activation, padding='same', input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, (3, 3), activation=activation, padding='same'))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# =========================
# EXPERIMENTOS CNN
# =========================

configs = [
    {'num_conv_layers': 2, 'num_filters': 32,  'activation': 'relu', 'optimizer': 'adam', 'loss_fn': 'categorical_crossentropy'},
    {'num_conv_layers': 3, 'num_filters': 64,  'activation': 'relu', 'optimizer': 'adam', 'loss_fn': 'categorical_crossentropy'},
    {'num_conv_layers': 4, 'num_filters': 128, 'activation': 'tanh', 'optimizer': 'sgd',  'loss_fn': 'categorical_crossentropy'},
]

resultados = []
melhor_acc = 0

for cfg in configs:
    # Seleciona o generator correto com base na função de perda
    if cfg['loss_fn'] == 'sparse_categorical_crossentropy':
        train_generator = train_generator_sparse
        val_generator = val_generator_sparse
    else:
        train_generator = train_generator_cat
        val_generator = val_generator_cat

    model = build_custom_cnn(
        input_shape=(224, 224, 3),
        num_classes=train_generator.num_classes,
        num_conv_layers=cfg['num_conv_layers'],
        num_filters=cfg['num_filters'],
        activation=cfg['activation']
    )

    model.compile(
        optimizer=cfg['optimizer'],
        loss=cfg['loss_fn'],
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )

    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]

    if val_acc > melhor_acc:
        melhor_acc = val_acc
        model.save("melhor_modelo.keras")

    resultados.append({
        "modelo": "CNN",
        "num_conv_layers": cfg['num_conv_layers'],
        "num_filters": cfg['num_filters'],
        "activation": cfg['activation'],
        "optimizer": cfg['optimizer'],
        "acc": acc,
        "val_acc": val_acc
    })

# =========================
# RESNET50 (TRANSFER LEARNING)
# =========================

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_generator_cat.num_classes, activation='softmax')(x)

model_resnet = Model(inputs=base_model.input, outputs=output)

model_resnet.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_resnet = model_resnet.fit(
    train_generator_cat,
    validation_data=val_generator_cat,
    epochs=10
)

model_resnet.save("melhor_modelo_resnet.keras")

# Adiciona resultado do ResNet à lista de comparação
resultados.append({
    "modelo": "ResNet50",
    "num_conv_layers": "-",
    "num_filters": "-",
    "activation": "relu",
    "optimizer": "adam",
    "acc": history_resnet.history['accuracy'][-1],
    "val_acc": history_resnet.history['val_accuracy'][-1]
})

# =========================
# COMPARAÇÃO FINAL
# =========================

print("\n=== Resultados Finais ===")
df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
