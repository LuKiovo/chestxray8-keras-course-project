"""Cloud GPU workflow planner for staged ChestX-ray8 training."""

from __future__ import annotations

import argparse
import json
import posixpath
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowConfig:
    metadata_csv: str
    raw_image_root: str
    manifest_dir: str
    stage_dir: str
    output_dir: str
    shard_size: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    model: str
    weights: str
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run the staged cloud GPU workflow.")
    parser.add_argument("--config", required=True, help="Path to workflow JSON config.")
    parser.add_argument(
        "--step",
        required=True,
        choices=("prepare", "stage", "train", "evaluate", "stage-train", "all"),
        help="Workflow step to plan or run.",
    )
    parser.add_argument("--shard-id", type=int, default=0, help="Train shard id for stage/train steps.")
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional checkpoint path. Defaults to previous shard best_model.keras when shard-id > 0.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute commands. Without this flag the script prints commands only.",
    )
    return parser.parse_args()


def load_config(path: Path) -> WorkflowConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = WorkflowConfig.__dataclass_fields__.keys()
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Workflow config missing keys: {', '.join(missing)}")
    return WorkflowConfig(**{key: raw[key] for key in required})


def quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def config_path_join(base: str, *parts: str) -> str:
    if "\\" in base or ":" in base:
        return str(Path(base, *parts))
    return posixpath.join(base, *parts)


def shard_csv(config: WorkflowConfig, shard_id: int) -> str:
    return config_path_join(config.manifest_dir, "shards", f"train_shard_{shard_id:03d}.csv")


def shard_output_dir(config: WorkflowConfig, shard_id: int) -> str:
    return config_path_join(config.output_dir, f"shard_{shard_id:03d}")


def default_resume_path(config: WorkflowConfig, shard_id: int) -> str:
    if shard_id <= 0:
        return ""
    return config_path_join(shard_output_dir(config, shard_id - 1), "best_model.keras")


def prepare_command(config: WorkflowConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "chestxray8.prepare_data",
        "--metadata-csv",
        config.metadata_csv,
        "--image-root",
        config.raw_image_root,
        "--output-dir",
        config.manifest_dir,
        "--shard-size",
        str(config.shard_size),
        "--require-images",
    ]


def stage_command(config: WorkflowConfig, shard_id: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "chestxray8.stage_shard",
        "--shard-csv",
        shard_csv(config, shard_id),
        "--image-root",
        config.raw_image_root,
        "--stage-dir",
        config.stage_dir,
        "--clean",
    ]


def train_command(config: WorkflowConfig, shard_id: int, resume_from: str = "") -> list[str]:
    checkpoint = resume_from or default_resume_path(config, shard_id)
    command = [
        sys.executable,
        "-m",
        "chestxray8.training",
        "--train-csv",
        shard_csv(config, shard_id),
        "--val-csv",
        config_path_join(config.manifest_dir, "val.csv"),
        "--image-root",
        config.stage_dir,
        "--output-dir",
        shard_output_dir(config, shard_id),
        "--model",
        config.model,
        "--weights",
        config.weights,
        "--image-size",
        str(config.image_size),
        "--batch-size",
        str(config.batch_size),
        "--epochs",
        str(config.epochs),
        "--learning-rate",
        str(config.learning_rate),
    ]
    if checkpoint:
        command.extend(["--resume-from", checkpoint])
    return command


def evaluate_command(config: WorkflowConfig, shard_id: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "chestxray8.evaluate",
        "--model-path",
        config_path_join(shard_output_dir(config, shard_id), "best_model.keras"),
        "--test-csv",
        config_path_join(config.manifest_dir, "test.csv"),
        "--image-root",
        config.raw_image_root,
        "--output-dir",
        config_path_join(config.output_dir, "evaluation"),
        "--image-size",
        str(config.image_size),
        "--batch-size",
        str(config.batch_size),
        "--threshold",
        str(config.threshold),
    ]


def planned_commands(config: WorkflowConfig, step: str, shard_id: int, resume_from: str = "") -> list[list[str]]:
    if shard_id < 0:
        raise ValueError("--shard-id cannot be negative")
    if step == "prepare":
        return [prepare_command(config)]
    if step == "stage":
        return [stage_command(config, shard_id)]
    if step == "train":
        return [train_command(config, shard_id, resume_from)]
    if step == "evaluate":
        return [evaluate_command(config, shard_id)]
    if step == "stage-train":
        return [stage_command(config, shard_id), train_command(config, shard_id, resume_from)]
    if step == "all":
        return [
            prepare_command(config),
            stage_command(config, shard_id),
            train_command(config, shard_id, resume_from),
            evaluate_command(config, shard_id),
        ]
    raise ValueError(f"Unsupported step: {step}")


def run_commands(commands: list[list[str]], execute: bool) -> None:
    for command in commands:
        print(quote(command))
        if execute:
            subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    commands = planned_commands(config, args.step, args.shard_id, args.resume_from)
    run_commands(commands, args.execute)


if __name__ == "__main__":
    main()
