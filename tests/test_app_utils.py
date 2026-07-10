import tempfile
import unittest
from pathlib import Path

from chestxray8.app_utils import (
    load_training_log,
    metrics_to_dataframe,
    prediction_to_dataframe,
    top_predictions,
)
from chestxray8.constants import DISEASE_LABELS


class AppUtilsTest(unittest.TestCase):
    def test_metrics_and_prediction_tables(self):
        metrics = {
            "per_label": {
                label: {"auc": 0.7, "f1": 0.5, "accuracy": 0.8, "positive_count": 3}
                for label in DISEASE_LABELS
            }
        }
        metrics_df = metrics_to_dataframe(metrics)
        self.assertEqual(len(metrics_df), len(DISEASE_LABELS))
        self.assertEqual(
            set(["label", "auc", "f1", "accuracy", "positive_count", "threshold", "validation_f1"]),
            set(metrics_df.columns),
        )

        prediction = {
            "labels": [
                {"label": "A", "probability": 0.2, "predicted": False},
                {"label": "B", "probability": 0.9, "predicted": True},
            ]
        }
        prediction_df = prediction_to_dataframe(prediction)
        self.assertEqual(list(prediction_df["label"]), ["A", "B"])
        self.assertEqual(top_predictions(prediction, 1)[0]["label"], "B")

    def test_load_training_log_adds_one_based_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_log.csv"
            path.write_text("epoch,loss,val_loss\n0,1.0,1.2\n1,0.8,1.0\n", encoding="utf-8")
            df = load_training_log(path)
        self.assertEqual(list(df["epoch"]), [1, 2])

    def test_load_training_log_accepts_string_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_log.csv"
            path.write_text("epoch,loss,val_loss\n0,1.0,1.2\n1,0.8,1.0\n", encoding="utf-8")
            df = load_training_log(path)
            df["epoch"] = df["epoch"].astype(str)
            path.write_text(df.to_csv(index=False), encoding="utf-8")

            reloaded = load_training_log(path)

        self.assertEqual(list(reloaded["epoch"]), [2, 3])


if __name__ == "__main__":
    unittest.main()
