"""
predict.py
==========
Carrega um modelo treinado e classifica uma imagem de sneaker.

Uso:
    python predict.py --model resultados/MobileNetV2.keras --image foto.jpg
    python predict.py --model resultados/MobileNetV2.keras --image foto.jpg --top 5
"""

import os
import sys
import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


IMG_SIZE = (96, 96)


def parse_args():
    parser = argparse.ArgumentParser(description="Sneaker classifier - inferencia")
    parser.add_argument(
        "--model",
        required=True,
        help="Caminho para o arquivo .keras do modelo treinado"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Caminho para a imagem a classificar"
    )
    parser.add_argument(
        "--labels",
        default="resultados/class_labels.json",
        help="Caminho para o JSON com mapeamento de classes (default: resultados/class_labels.json)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Numero de predicoes a exibir (default: 3)"
    )
    return parser.parse_args()


def load_class_labels(labels_path: str) -> dict:
    if not os.path.exists(labels_path):
        sys.exit(
            f"[ERRO] Arquivo de labels nao encontrado: {labels_path}\n"
            f"Execute sneaker_classifier.py primeiro para gerar os labels."
        )
    with open(labels_path, "r") as f:
        index_to_class = json.load(f)
    return index_to_class


def preprocess_image(img_path: str) -> np.ndarray:
    if not os.path.exists(img_path):
        sys.exit(f"[ERRO] Imagem nao encontrada: {img_path}")

    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def predict(model, img_array: np.ndarray, index_to_class: dict, top_k: int):
    probs = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = [
        {"rank": i + 1, "class": index_to_class[str(idx)], "confidence": float(probs[idx])}
        for i, idx in enumerate(top_indices)
    ]
    return results


def main():
    args = parse_args()

    print(f"Carregando modelo: {args.model}")
    model = load_model(args.model)

    index_to_class = load_class_labels(args.labels)
    img_array = preprocess_image(args.image)

    results = predict(model, img_array, index_to_class, top_k=args.top)

    print(f"\nImagem: {args.image}")
    print(f"{'Rank':<6} {'Classe':<35} {'Confianca':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['rank']:<6} {r['class']:<35} {r['confidence']:>9.1%}")


if __name__ == "__main__":
    main()