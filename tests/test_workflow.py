import json
import tempfile
import unittest
from pathlib import Path

from chestxray8.workflow import load_config, planned_commands, quote


class WorkflowTest(unittest.TestCase):
    def test_plan_stage_train_uses_previous_checkpoint(self):
        config = {
            "metadata_csv": "/data/Data_Entry_2017.csv",
            "raw_image_root": "/data/images",
            "manifest_dir": "/project/manifests",
            "stage_dir": "/stage/current",
            "output_dir": "/project/outputs",
            "shard_size": 10000,
            "image_size": 224,
            "batch_size": 32,
            "epochs": 5,
            "learning_rate": 0.0001,
            "model": "mobilenet_v2",
            "weights": "imagenet",
            "threshold": 0.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            loaded = load_config(config_path)
            commands = planned_commands(loaded, "stage-train", shard_id=2)

        self.assertEqual(len(commands), 2)
        self.assertIn("train_shard_002.csv", quote(commands[0]))
        train_line = quote(commands[1])
        self.assertIn("chestxray8.training", train_line)
        self.assertIn("/project/manifests/shards/train_shard_002.csv", train_line)
        self.assertIn("/project/outputs/shard_002", train_line)
        self.assertIn("/project/outputs/shard_001/best_model.keras", train_line)

    def test_plan_evaluate_uses_requested_shard_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "metadata_csv": "/data/Data_Entry_2017.csv",
                        "raw_image_root": "/data/images",
                        "manifest_dir": "/project/manifests",
                        "stage_dir": "/stage/current",
                        "output_dir": "/project/outputs",
                        "shard_size": 10000,
                        "image_size": 224,
                        "batch_size": 32,
                        "epochs": 5,
                        "learning_rate": 0.0001,
                        "model": "mobilenet_v2",
                        "weights": "imagenet",
                        "threshold": 0.35,
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_config(config_path)
            commands = planned_commands(loaded, "evaluate", shard_id=4)

        self.assertEqual(len(commands), 1)
        line = quote(commands[0])
        self.assertIn("shard_004", line)
        self.assertIn("test.csv", line)
        self.assertIn("0.35", line)


if __name__ == "__main__":
    unittest.main()
