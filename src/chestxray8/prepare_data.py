"""Build manifests and staged-training shards for ChestX-ray8.

The script expects the official NIH ChestX-ray8 metadata file
``Data_Entry_2017.csv``. It does not copy images; it creates compact CSV
manifests that later training scripts can consume one shard at a time.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from chestxray8.constants import DISEASE_LABELS, NO_FINDING


REQUIRED_COLUMNS = ("Image Index", "Finding Labels", "Patient ID")
MANIFEST_COLUMNS = (
    "image",
    "patient_id",
    "finding_labels",
    "split",
    "relative_path",
    *DISEASE_LABELS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test manifests and train shards for ChestX-ray8."
    )
    parser.add_argument("--metadata-csv", required=True, help="Path to Data_Entry_2017.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated manifests.")
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional root directory that contains ChestX-ray8 images.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Maximum number of train images per shard.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient split.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Fail if an image listed in the metadata is not found under --image-root.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total <= 1.001:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios cannot be negative")


def label_vector(finding_labels: str) -> dict[str, int]:
    labels = [label.strip() for label in finding_labels.split("|") if label.strip()]
    if labels == [NO_FINDING]:
        labels = []

    unknown = sorted(set(labels) - set(DISEASE_LABELS))
    if unknown:
        raise ValueError(f"Unknown ChestX-ray8 labels: {', '.join(unknown)}")

    present = set(labels)
    return {label: int(label in present) for label in DISEASE_LABELS}


def image_relative_path(image_name: str, image_root: Path | None) -> str:
    if image_root is None:
        return image_name

    direct = image_root / image_name
    if direct.exists():
        return image_name

    matches = list(image_root.glob(f"**/{image_name}"))
    if matches:
        return str(matches[0].relative_to(image_root))

    return image_name


def read_metadata(
    metadata_csv: Path, image_root: Path | None, require_images: bool
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {', '.join(missing)}")

        for source_row in reader:
            image_name = source_row["Image Index"].strip()
            relative_path = image_relative_path(image_name, image_root)
            if require_images and image_root is not None and not (image_root / relative_path).exists():
                raise FileNotFoundError(f"Image not found under image root: {image_name}")

            row: dict[str, str | int] = {
                "image": image_name,
                "patient_id": source_row["Patient ID"].strip(),
                "finding_labels": source_row["Finding Labels"].strip(),
                "split": "",
                "relative_path": relative_path,
            }
            row.update(label_vector(source_row["Finding Labels"]))
            rows.append(row)

    if not rows:
        raise ValueError("Metadata CSV does not contain any image rows")
    return rows


def assign_patient_splits(
    rows: list[dict[str, str | int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    validate_ratios(train_ratio, val_ratio, test_ratio)

    by_patient: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)

    patients = list(by_patient)
    random.Random(seed).shuffle(patients)

    total = len(rows)
    train_limit = round(total * train_ratio)
    val_limit = train_limit + round(total * val_ratio)

    current_count = 0
    for patient_id in patients:
        patient_rows = by_patient[patient_id]
        if current_count < train_limit:
            split = "train"
        elif current_count < val_limit:
            split = "val"
        else:
            split = "test"
        for row in patient_rows:
            row["split"] = split
        current_count += len(patient_rows)


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifests(rows: list[dict[str, str | int]], output_dir: Path, shard_size: int) -> None:
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: (str(row["split"]), str(row["patient_id"]), str(row["image"])))
    write_csv(output_dir / "all.csv", sorted_rows)

    split_rows: dict[str, list[dict[str, str | int]]] = {}
    for split in ("train", "val", "test"):
        split_rows[split] = [row for row in sorted_rows if row["split"] == split]
        write_csv(output_dir / f"{split}.csv", split_rows[split])

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    shard_index: list[dict[str, str | int]] = []
    train_rows = split_rows["train"]
    for shard_id, start in enumerate(range(0, len(train_rows), shard_size)):
        shard_rows = train_rows[start : start + shard_size]
        shard_name = f"train_shard_{shard_id:03d}.csv"
        write_csv(shards_dir / shard_name, shard_rows)
        shard_index.append(
            {
                "shard": shard_name,
                "split": "train",
                "num_images": len(shard_rows),
                "first_image": shard_rows[0]["image"],
                "last_image": shard_rows[-1]["image"],
            }
        )

    with (shards_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=("shard", "split", "num_images", "first_image", "last_image")
        )
        writer.writeheader()
        writer.writerows(shard_index)

    write_summary(output_dir / "summary.txt", rows, shard_index)


def write_summary(
    path: Path, rows: list[dict[str, str | int]], shard_index: list[dict[str, str | int]]
) -> None:
    split_counts = {
        split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")
    }
    label_counts = {
        label: sum(int(row[label]) for row in rows) for label in DISEASE_LABELS
    }

    lines = [
        "ChestX-ray8 manifest summary",
        f"total_images: {len(rows)}",
        f"train_images: {split_counts['train']}",
        f"val_images: {split_counts['val']}",
        f"test_images: {split_counts['test']}",
        f"train_shards: {len(shard_index)}",
        "",
        "label_counts:",
    ]
    lines.extend(f"- {label}: {count}" for label, count in label_counts.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root) if args.image_root else None
    rows = read_metadata(Path(args.metadata_csv), image_root, args.require_images)
    assign_patient_splits(rows, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    write_manifests(rows, Path(args.output_dir), args.shard_size)


if __name__ == "__main__":
    main()

