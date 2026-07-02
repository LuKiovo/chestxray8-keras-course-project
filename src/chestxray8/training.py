"""Keras training entrypoint for staged ChestX-ray8 shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from chestxray8.constants import DISEASE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChestX-ray8 multi-label classifier.")
    parser.add_argument("--train-csv", required=True, help="Training manifest or shard CSV.")
    parser.add_argument("--val-csv", required=True, help="Validation manifest CSV.")
    parser.add_argument("--image-root", required=True, help="Root directory for images.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--model", default="mobilenet_v2", choices=("mobilenet_v2", "tiny_cnn"))
    parser.add_argument("--weights", default="imagenet", choices=("imagenet", "none"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-from", default="", help="Optional .keras checkpoint to resume.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest(manifest_csv: Path, image_root: Path) -> tuple[list[str], np.ndarray]:
    image_paths: list[str] = []
    labels: list[list[float]] = []

    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [label for label in DISEASE_LABELS if label not in (reader.fieldnames or [])]
        if "relative_path" not in (reader.fieldnames or []):
            missing.append("relative_path")
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(missing)}")

        for row in reader:
            image_paths.append(str(image_root / row["relative_path"]))
            labels.append([float(row[label]) for label in DISEASE_LABELS])

    if not image_paths:
        raise ValueError(f"Manifest has no rows: {manifest_csv}")
    return image_paths, np.asarray(labels, dtype=np.float32)


def build_dataset(
    image_paths: list[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    if training:
        ds = ds.shuffle(min(len(image_paths), 4096), seed=seed, reshuffle_each_iteration=True)

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32) / 255.0
        if training:
            image = tf.image.random_flip_left_right(image, seed=seed)
        return image, label

    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


def positive_weights(labels: np.ndarray) -> np.ndarray:
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    return (negatives + 1.0) / (positives + 1.0)


@tf.keras.utils.register_keras_serializable(package="chestxray8")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """Binary cross-entropy with per-label positive weights."""

    def __init__(self, pos_weights: list[float], name: str = "weighted_binary_crossentropy"):
        super().__init__(name=name)
        self.pos_weights = [float(weight) for weight in pos_weights]

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        weights = tf.constant(self.pos_weights, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        positive_loss = -y_true * tf.math.log(y_pred) * weights
        negative_loss = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(positive_loss + negative_loss)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"pos_weights": self.pos_weights})
        return config


def weighted_binary_crossentropy(pos_weights: np.ndarray) -> WeightedBinaryCrossentropy:
    return WeightedBinaryCrossentropy([float(weight) for weight in pos_weights])


def build_model(model_name: str, image_size: int, learning_rate: float, weights: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    if model_name == "tiny_cnn":
        x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None if weights == "none" else "imagenet",
            input_shape=(image_size, image_size, 3),
        )
        base.trainable = weights == "none"
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(DISEASE_LABELS), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(DISEASE_LABELS)),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
        ],
    )
    return model


def train(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels = read_manifest(Path(args.train_csv), Path(args.image_root))
    val_paths, val_labels = read_manifest(Path(args.val_csv), Path(args.image_root))

    train_ds = build_dataset(
        train_paths, train_labels, args.image_size, args.batch_size, True, args.seed
    )
    val_ds = build_dataset(val_paths, val_labels, args.image_size, args.batch_size, False, args.seed)

    train_loss = weighted_binary_crossentropy(positive_weights(train_labels))
    if args.resume_from:
        model = tf.keras.models.load_model(args.resume_from, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=train_loss,
            metrics=[
                tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(DISEASE_LABELS)),
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            ],
        )
    else:
        model = build_model(args.model, args.image_size, args.learning_rate, args.weights)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=train_loss,
            metrics=[
                tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(DISEASE_LABELS)),
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            ],
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv"), append=bool(args.resume_from)),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(output_dir / "last_model.keras")

    history_data = {key: [float(v) for v in values] for key, values in history.history.items()}
    summary = {
        "model": args.model,
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "labels": DISEASE_LABELS,
        "history": history_data,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
