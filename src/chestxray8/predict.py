"""Run single-image inference with a trained ChestX-ray8 model."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
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


def _remove_newer_keras_fields(config: object) -> object:
    """Drop Keras config fields not understood by older local Keras builds."""
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        if config.get("class_name") == "GlorotUniform" and isinstance(config.get("config"), dict):
            config["config"].pop("input_axes", None)
            config["config"].pop("output_axes", None)
        if config.get("class_name") == "BatchNormalization" and isinstance(config.get("config"), dict):
            config["config"].pop("renorm", None)
            config["config"].pop("renorm_clipping", None)
            config["config"].pop("renorm_momentum", None)
        for value in config.values():
            _remove_newer_keras_fields(value)
    elif isinstance(config, list):
        for value in config:
            _remove_newer_keras_fields(value)
    return config


def _load_model_with_keras_compat(model_path: Path) -> tf.keras.Model:
    custom_objects = {"WeightedBinaryCrossentropy": WeightedBinaryCrossentropy}
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
    except TypeError as exc:
        if "input_axes" not in str(exc) and "output_axes" not in str(exc):
            raise

    with tempfile.TemporaryDirectory() as temp_dir:
        patched_path = Path(temp_dir) / model_path.name
        with zipfile.ZipFile(model_path, "r") as source, zipfile.ZipFile(
            patched_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            for item in source.infolist():
                payload = source.read(item.filename)
                if item.filename == "config.json":
                    config = json.loads(payload.decode("utf-8"))
                    payload = json.dumps(
                        _remove_newer_keras_fields(config),
                        ensure_ascii=False,
                    ).encode("utf-8")
                target.writestr(item, payload)

        return tf.keras.models.load_model(
            patched_path,
            custom_objects=custom_objects,
            compile=False,
        )


def predict_image(
    model_path: Path,
    image_path: Path,
    image_size: int = 224,
    threshold: float = 0.5,
    label_thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = _load_model_with_keras_compat(model_path)
    probabilities = model.predict(load_image(image_path, image_size), verbose=0)[0]
    labels = []
    for index, label in enumerate(DISEASE_LABELS):
        label_threshold = float(label_thresholds.get(label, threshold)) if label_thresholds else threshold
        labels.append(
            {
                "label": label,
                "probability": float(probabilities[index]),
                "threshold": label_threshold,
                "predicted": bool(probabilities[index] >= label_threshold),
            }
        )
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
