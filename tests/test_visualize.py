import json
import tempfile
import unittest
from pathlib import Path

from chestxray8.constants import DISEASE_LABELS
from chestxray8.visualize import create_figures


class VisualizeTest(unittest.TestCase):
    def test_create_figures_from_training_and_eval_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_log = root / "training_log.csv"
            training_log.write_text(
                "epoch,loss,auc,binary_accuracy,val_loss,val_auc,val_binary_accuracy\n"
                "0,0.9,0.55,0.60,1.0,0.50,0.58\n"
                "1,0.7,0.70,0.72,0.8,0.66,0.68\n",
                encoding="utf-8",
            )

            metrics = {
                "per_label": {
                    label: {
                        "auc": 0.5 + index * 0.01,
                        "f1": 0.3 + index * 0.01,
                        "accuracy": 0.6 + index * 0.01,
                        "positive_count": index + 1,
                    }
                    for index, label in enumerate(DISEASE_LABELS)
                }
            }
            metrics_json = root / "metrics_summary.json"
            metrics_json.write_text(json.dumps(metrics), encoding="utf-8")

            roc_json = root / "roc_curves.json"
            roc_json.write_text(
                json.dumps(
                    {
                        "Atelectasis": {
                            "fpr": [0.0, 0.2, 1.0],
                            "tpr": [0.0, 0.8, 1.0],
                            "thresholds": [1.5, 0.6, 0.1],
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_dir = root / "figures"
            outputs = create_figures(output_dir, training_log, metrics_json, roc_json)

            expected = {
                "training_loss.png",
                "training_auc.png",
                "training_accuracy.png",
                "label_auc.png",
                "label_f1.png",
                "label_accuracy.png",
                "roc_curves.png",
            }
            self.assertEqual({path.name for path in outputs}, expected)
            for output in outputs:
                self.assertTrue(output.exists())
                self.assertGreater(output.stat().st_size, 0)

    def test_create_figures_accepts_string_epoch_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_log = root / "training_log.csv"
            training_log.write_text(
                "epoch,loss,auc,binary_accuracy\n"
                "0,0.9,0.55,0.60\n"
                "1,0.7,0.70,0.72\n",
                encoding="utf-8",
            )

            output_dir = root / "figures"
            outputs = create_figures(output_dir, training_log=training_log)

            self.assertEqual(
                {path.name for path in outputs},
                {"training_loss.png", "training_auc.png", "training_accuracy.png"},
            )
            for output in outputs:
                self.assertTrue(output.exists())
                self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
