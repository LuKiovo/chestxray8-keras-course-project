import json
import tempfile
import unittest
from pathlib import Path

from chestxray8.report_materials import build_report_materials


class ReportMaterialsTest(unittest.TestCase):
    def test_build_report_materials_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_summary = root / "summary.txt"
            manifest_summary.write_text("total_images: 10\ntrain_shards: 2\n", encoding="utf-8")

            training_summary = root / "training_summary.json"
            training_summary.write_text(
                json.dumps(
                    {
                        "model": "tiny_cnn",
                        "train_images": 6,
                        "val_images": 2,
                        "train_csv": "train.csv",
                        "val_csv": "val.csv",
                        "history": {"loss": [1.0, 0.7], "val_auc": [0.4, 0.6]},
                    }
                ),
                encoding="utf-8",
            )

            metrics_summary = root / "metrics_summary.json"
            metrics_summary.write_text(
                json.dumps(
                    {
                        "test_images": 2,
                        "threshold": 0.5,
                        "macro_auc": 0.7,
                        "macro_f1": 0.4,
                        "micro_f1": 0.5,
                        "subset_accuracy": 0.25,
                        "per_label": {
                            "Atelectasis": {
                                "auc": 0.8,
                                "f1": 0.5,
                                "accuracy": 0.75,
                                "positive_count": 1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            figures_dir = root / "figures"
            figures_dir.mkdir()
            (figures_dir / "training_loss.png").write_bytes(b"png")

            output = root / "report_assets" / "materials.md"
            content = build_report_materials(
                output,
                manifest_summary,
                training_summary,
                metrics_summary,
                figures_dir,
            )

        self.assertIn("ChestX-ray8 项目报告素材", content)
        self.assertIn("total_images: 10", content)
        self.assertIn("tiny_cnn", content)
        self.assertIn("Macro AUC", content)
        self.assertIn("Atelectasis", content)
        self.assertIn("training_loss.png", content)


if __name__ == "__main__":
    unittest.main()

