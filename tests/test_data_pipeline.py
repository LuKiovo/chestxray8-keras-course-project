import csv
import tempfile
import unittest
from pathlib import Path

from chestxray8.prepare_data import (
    assign_patient_splits,
    read_metadata,
    write_manifests,
)
from chestxray8.stage_shard import stage_shard


class DataPipelineTest(unittest.TestCase):
    def test_manifest_sharding_and_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "raw_images"
            image_root.mkdir()
            metadata_csv = root / "Data_Entry_2017.csv"

            image_rows = [
                ("000001.png", "Atelectasis|Effusion", "p1"),
                ("000002.png", "No Finding", "p1"),
                ("000003.png", "Cardiomegaly", "p2"),
                ("000004.png", "Mass|Nodule", "p3"),
                ("000005.png", "Pneumonia", "p4"),
                ("000006.png", "Edema", "p5"),
            ]
            for image_name, _, _ in image_rows:
                (image_root / image_name).write_bytes(b"fake image")

            with metadata_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["Image Index", "Finding Labels", "Patient ID"]
                )
                writer.writeheader()
                for image_name, labels, patient_id in image_rows:
                    writer.writerow(
                        {
                            "Image Index": image_name,
                            "Finding Labels": labels,
                            "Patient ID": patient_id,
                        }
                    )

            rows = read_metadata(metadata_csv, image_root, require_images=True)
            assign_patient_splits(rows, 0.5, 0.25, 0.25, seed=7)

            output_dir = root / "manifests"
            write_manifests(rows, output_dir, shard_size=2)

            self.assertTrue((output_dir / "all.csv").exists())
            self.assertTrue((output_dir / "train.csv").exists())
            self.assertTrue((output_dir / "val.csv").exists())
            self.assertTrue((output_dir / "test.csv").exists())
            self.assertTrue((output_dir / "summary.txt").exists())

            shard_files = sorted((output_dir / "shards").glob("train_shard_*.csv"))
            self.assertGreaterEqual(len(shard_files), 1)
            self.assertTrue((output_dir / "shards" / "index.csv").exists())

            with (output_dir / "all.csv").open("r", encoding="utf-8", newline="") as f:
                manifest_rows = list(csv.DictReader(f))
            self.assertEqual(len(manifest_rows), len(image_rows))

            patients_by_split = {}
            for row in manifest_rows:
                patients_by_split.setdefault(row["patient_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in patients_by_split.values()))

            first_row = next(row for row in manifest_rows if row["image"] == "000001.png")
            self.assertEqual(first_row["Atelectasis"], "1")
            self.assertEqual(first_row["Effusion"], "1")
            self.assertEqual(first_row["Cardiomegaly"], "0")

            stage_dir = root / "stage"
            copied = stage_shard(shard_files[0], image_root, stage_dir, clean=True)
            self.assertGreaterEqual(copied, 1)
            self.assertEqual(len(list(stage_dir.glob("*.png"))), copied)


if __name__ == "__main__":
    unittest.main()

