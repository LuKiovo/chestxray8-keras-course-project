"""Create report-ready figures from training and evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from chestxray8.constants import DISEASE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures for reports and demos.")
    parser.add_argument("--training-log", help="Path to training_log.csv.")
    parser.add_argument("--metrics-json", help="Path to metrics_summary.json.")
    parser.add_argument("--roc-json", help="Path to roc_curves.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures.")
    return parser.parse_args()


def save_training_curves(training_log: Path, output_dir: Path) -> list[Path]:
    if not training_log.exists():
        raise FileNotFoundError(f"Training log not found: {training_log}")

    df = pd.read_csv(training_log)
    if df.empty:
        raise ValueError(f"Training log is empty: {training_log}")

    output_paths: list[Path] = []
    epoch = df["epoch"] + 1 if "epoch" in df else range(1, len(df) + 1)

    metric_pairs = [
        ("loss", "val_loss", "training_loss.png", "Loss"),
        ("auc", "val_auc", "training_auc.png", "AUC"),
        ("binary_accuracy", "val_binary_accuracy", "training_accuracy.png", "Binary Accuracy"),
    ]
    for train_col, val_col, filename, title in metric_pairs:
        if train_col not in df:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
        ax.plot(epoch, df[train_col], marker="o", label=train_col)
        if val_col in df:
            ax.plot(epoch, df[val_col], marker="o", label=val_col)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_label_metric_bars(metrics_json: Path, output_dir: Path) -> list[Path]:
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    per_label = metrics.get("per_label", {})
    if not per_label:
        raise ValueError("metrics_summary.json does not contain per_label metrics")

    output_paths: list[Path] = []
    for metric_name, filename, title in [
        ("auc", "label_auc.png", "Per-label AUC"),
        ("f1", "label_f1.png", "Per-label F1"),
        ("accuracy", "label_accuracy.png", "Per-label Accuracy"),
    ]:
        labels: list[str] = []
        values: list[float] = []
        for label in DISEASE_LABELS:
            value = per_label.get(label, {}).get(metric_name)
            if value is not None:
                labels.append(label)
                values.append(float(value))
        if not values:
            continue

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
        ax.bar(labels, values, color="#2f80ed")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric_name.upper())
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_roc_curves(roc_json: Path, output_dir: Path) -> list[Path]:
    roc_data = json.loads(roc_json.read_text(encoding="utf-8"))
    curves = {label: curve for label, curve in roc_data.items() if curve}
    if not curves:
        return []

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=140)
    for label, curve in curves.items():
        ax.plot(curve["fpr"], curve["tpr"], linewidth=1.4, label=label)
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    output_path = output_dir / "roc_curves.png"
    fig.savefig(output_path)
    plt.close(fig)
    return [output_path]


def create_figures(
    output_dir: Path,
    training_log: Path | None = None,
    metrics_json: Path | None = None,
    roc_json: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if training_log:
        outputs.extend(save_training_curves(training_log, output_dir))
    if metrics_json:
        outputs.extend(save_label_metric_bars(metrics_json, output_dir))
    if roc_json:
        outputs.extend(save_roc_curves(roc_json, output_dir))
    if not outputs:
        raise ValueError("No figures were generated. Provide at least one valid input artifact.")
    return outputs


def main() -> None:
    args = parse_args()
    outputs = create_figures(
        output_dir=Path(args.output_dir),
        training_log=Path(args.training_log) if args.training_log else None,
        metrics_json=Path(args.metrics_json) if args.metrics_json else None,
        roc_json=Path(args.roc_json) if args.roc_json else None,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

