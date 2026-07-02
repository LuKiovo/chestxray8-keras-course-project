"""Copy one ChestX-ray8 shard into a compact staging directory."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images listed in a shard CSV into a staging directory."
    )
    parser.add_argument("--shard-csv", required=True, help="Path to train_shard_XXX.csv.")
    parser.add_argument("--image-root", required=True, help="Root directory containing raw images.")
    parser.add_argument("--stage-dir", required=True, help="Directory for staged shard images.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete files in the stage directory before copying this shard.",
    )
    return parser.parse_args()


def clean_stage_dir(stage_dir: Path) -> None:
    if not stage_dir.exists():
        return
    for child in stage_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def stage_shard(shard_csv: Path, image_root: Path, stage_dir: Path, clean: bool) -> int:
    if clean:
        clean_stage_dir(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    with shard_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "relative_path" not in (reader.fieldnames or []):
            raise ValueError("Shard CSV must contain a relative_path column")

        for row in reader:
            source = image_root / row["relative_path"]
            if not source.exists():
                raise FileNotFoundError(f"Shard image not found: {source}")

            destination = stage_dir / row["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1

    return copied


def main() -> None:
    args = parse_args()
    copied = stage_shard(
        Path(args.shard_csv),
        Path(args.image_root),
        Path(args.stage_dir),
        args.clean,
    )
    print(f"staged_images: {copied}")


if __name__ == "__main__":
    main()

