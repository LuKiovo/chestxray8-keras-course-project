"""Evaluate a trained ChestX-ray8 multi-label model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from chestxray8.constants import DISEASE_LABELS
from chestxray8.training import WeightedBinaryCrossentropy, build_dataset, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChestX-ray8 multi-label model.")
    parser.add_argument("--model-path", required=True, help="Path to .keras model.")
    parser.add_argument("--test-csv", required=True, help="Test manifest CSV.")
    parser.add_argument("--image-root", required=True, help="Root directory for test images.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation artifacts.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_roc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]] | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {
        "fpr": [float(v) for v in fpr],
        "tpr": [float(v) for v in tpr],
        "thresholds": [float(v) for v in thresholds],
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, y_true = read_manifest(Path(args.test_csv), Path(args.image_root))
    test_ds = build_dataset(
        image_paths,
        y_true,
        args.image_size,
        args.batch_size,
        training=False,
        seed=42,
    )

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"WeightedBinaryCrossentropy": WeightedBinaryCrossentropy},
        compile=False,
    )
    y_score = model.predict(test_ds, verbose=0)
    y_pred = (y_score >= args.threshold).astype(np.int32)

    per_label: dict[str, dict[str, float | None]] = {}
    roc_data: dict[str, dict[str, list[float]] | None] = {}
    for index, label in enumerate(DISEASE_LABELS):
        label_true = y_true[:, index]
        label_pred = y_pred[:, index]
        label_score = y_score[:, index]
        per_label[label] = {
            "auc": safe_auc(label_true, label_score),
            "f1": float(f1_score(label_true, label_pred, zero_division=0)),
            "accuracy": float(accuracy_score(label_true, label_pred)),
            "positive_count": int(label_true.sum()),
        }
        roc_data[label] = safe_roc(label_true, label_score)

    auc_values = [metrics["auc"] for metrics in per_label.values() if metrics["auc"] is not None]
    summary = {
        "model_path": str(args.model_path),
        "test_csv": str(args.test_csv),
        "test_images": len(image_paths),
        "threshold": args.threshold,
        "macro_auc": float(np.mean(auc_values)) if auc_values else None,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "per_label": per_label,
    }

    write_predictions(output_dir / "predictions.csv", image_paths, y_true, y_score, y_pred)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "roc_curves.json").write_text(
        json.dumps(roc_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_predictions(
    path: Path,
    image_paths: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    columns = ["image"]
    for label in DISEASE_LABELS:
        columns.extend([f"true_{label}", f"prob_{label}", f"pred_{label}"])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row_index, image_path in enumerate(image_paths):
            row: dict[str, str | int | float] = {"image": Path(image_path).name}
            for label_index, label in enumerate(DISEASE_LABELS):
                row[f"true_{label}"] = int(y_true[row_index, label_index])
                row[f"prob_{label}"] = float(y_score[row_index, label_index])
                row[f"pred_{label}"] = int(y_pred[row_index, label_index])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

