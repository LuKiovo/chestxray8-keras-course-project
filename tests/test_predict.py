import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import tensorflow as tf

from chestxray8.constants import DISEASE_LABELS
from chestxray8.predict import predict_image
from chestxray8.training import train


def write_png(path: Path, value: int) -> None:
    array = np.full((32, 32, 3), value, dtype=np.uint8)
    tf.io.write_file(str(path), tf.io.encode_png(array))


def write_manifest(path: Path, rows):
    fieldnames = ["image", "patient_id", "finding_labels", "split", "relative_path", *DISEASE_LABELS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            labels = {label: 0 for label in DISEASE_LABELS}
            labels[row["label"]] = 1
            writer.writerow(
                {
                    "image": row["image"],
                    "patient_id": row["patient_id"],
                    "finding_labels": row["label"],
                    "split": row["split"],
                    "relative_path": row["image"],
                    **labels,
                }
            )


class PredictSmokeTest(unittest.TestCase):
    def test_predict_image_returns_sorted_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            for i in range(8):
                write_png(image_root / f"img_{i}.png", value=25 + i * 20)

            train_csv = root / "train.csv"
            val_csv = root / "val.csv"
            write_manifest(
                train_csv,
                [
                    {"image": "img_0.png", "patient_id": "p0", "split": "train", "label": "Atelectasis"},
                    {"image": "img_1.png", "patient_id": "p1", "split": "train", "label": "Effusion"},
                    {"image": "img_2.png", "patient_id": "p2", "split": "train", "label": "Mass"},
                    {"image": "img_3.png", "patient_id": "p3", "split": "train", "label": "Nodule"},
                    {"image": "img_4.png", "patient_id": "p4", "split": "train", "label": "Pneumonia"},
                    {"image": "img_5.png", "patient_id": "p5", "split": "train", "label": "Edema"},
                ],
            )
            write_manifest(
                val_csv,
                [
                    {"image": "img_6.png", "patient_id": "p6", "split": "val", "label": "Atelectasis"},
                    {"image": "img_7.png", "patient_id": "p7", "split": "val", "label": "Effusion"},
                ],
            )

            model_dir = root / "model"
            train(
                Namespace(
                    train_csv=str(train_csv),
                    val_csv=str(val_csv),
                    image_root=str(image_root),
                    output_dir=str(model_dir),
                    model="tiny_cnn",
                    weights="none",
                    image_size=32,
                    batch_size=2,
                    epochs=1,
                    learning_rate=1e-3,
                    resume_from="",
                    seed=123,
                )
            )

            result = predict_image(
                model_path=model_dir / "best_model.keras",
                image_path=image_root / "img_6.png",
                image_size=32,
                threshold=0.5,
            )

            self.assertEqual(result["image"], "img_6.png")
            self.assertEqual(len(result["labels"]), len(DISEASE_LABELS))
            probabilities = [item["probability"] for item in result["labels"]]
            self.assertEqual(probabilities, sorted(probabilities, reverse=True))
            self.assertTrue(all(0.0 <= probability <= 1.0 for probability in probabilities))
            self.assertIn("positive_labels", result)

            tuned = predict_image(
                model_path=model_dir / "best_model.keras",
                image_path=image_root / "img_6.png",
                image_size=32,
                threshold=0.5,
                label_thresholds={"Atelectasis": 1.0},
            )
            atelectasis = next(item for item in tuned["labels"] if item["label"] == "Atelectasis")
            self.assertEqual(atelectasis["threshold"], 1.0)
            self.assertFalse(atelectasis["predicted"])


if __name__ == "__main__":
    unittest.main()
