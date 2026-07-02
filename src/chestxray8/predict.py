"""Run single-image inference with a trained ChestX-ray8 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from chestxray8.constants import DISEASE_LABELS
from chestxray8.training import WeightedBinaryCrossentropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ChestX-ray8 labels for one image.")
    parser.add_argument("--model-path", required=True, help="Path to a .keras model.")
    parser.add_argument("--image", required=True, help="Path to an X-ray image.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_image(image_path: Path, image_size: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def predict_image(
    model_path: Path,
    image_path: Path,
    image_size: int = 224,
    threshold: float = 0.5,
) -> dict[str, object]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedBinaryCrossentropy": WeightedBinaryCrossentropy},
        compile=False,
    )
    probabilities = model.predict(load_image(image_path, image_size), verbose=0)[0]
    labels = [
        {
            "label": label,
            "probability": float(probabilities[index]),
            "predicted": bool(probabilities[index] >= threshold),
        }
        for index, label in enumerate(DISEASE_LABELS)
    ]
    labels.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "image": image_path.name,
        "threshold": threshold,
        "labels": labels,
        "positive_labels": [item["label"] for item in labels if item["predicted"]],
    }


def main() -> None:
    args = parse_args()
    result = predict_image(
        model_path=Path(args.model_path),
        image_path=Path(args.image),
        image_size=args.image_size,
        threshold=args.threshold,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

