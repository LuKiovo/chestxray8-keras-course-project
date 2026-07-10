"""Create report-ready figures from training and evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from chestxray8.constants import DISEASE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures for reports and demos.")
    parser.add_argument("--training-log", help="Path to training_log.csv.")
    parser.add_argument("--metrics-json", help="Path to metrics_summary.json.")
    parser.add_argument("--baseline-metrics-json", help="Optional baseline metrics for threshold comparison charts.")
    parser.add_argument("--roc-json", help="Path to roc_curves.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures.")
    return parser.parse_args()


def save_training_curves(training_log: Path, output_dir: Path) -> list[Path]:
    if not training_log.exists():
        raise FileNotFoundError(f"Training log not found: {training_log}")

    df = pd.read_csv(training_log)
    if df.empty:
        raise ValueError(f"Training log is empty: {training_log}")

    # Interrupted runs can append a second CSV header and restart epoch numbering.
    # Keep the latest complete record for each epoch so report curves stay readable.
    if "epoch" in df:
        df["_epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.dropna(subset=["_epoch"])
        if df.empty:
            raise ValueError(f"Training log does not contain numeric epochs: {training_log}")
        df = df.drop_duplicates(subset="_epoch", keep="last").sort_values("_epoch")
        epoch = df["_epoch"].astype(int) + 1
    else:
        epoch = range(1, len(df) + 1)

    output_paths: list[Path] = []
    metric_pairs = [
        ("loss", "val_loss", "training_loss.png", "Loss"),
        ("auc", "val_auc", "training_auc.png", "AUC"),
        ("binary_accuracy", "val_binary_accuracy", "training_accuracy.png", "Binary Accuracy"),
    ]
    for train_col, val_col, filename, title in metric_pairs:
        if train_col not in df:
            continue
        train_values = pd.to_numeric(df[train_col], errors="coerce")
        valid = train_values.notna()
        if not valid.any():
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
        ax.plot(epoch[valid], train_values[valid], marker="o", label=train_col)
        if val_col in df:
            val_values = pd.to_numeric(df[val_col], errors="coerce")
            valid_val = val_values.notna()
            ax.plot(epoch[valid_val], val_values[valid_val], marker="o", label=val_col)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_xticks(list(epoch))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
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


def save_threshold_comparison_figures(
    baseline_metrics_json: Path,
    tuned_metrics_json: Path,
    output_dir: Path,
) -> list[Path]:
    """Create before/after charts for validation-selected per-label thresholds."""
    baseline = json.loads(baseline_metrics_json.read_text(encoding="utf-8"))
    tuned = json.loads(tuned_metrics_json.read_text(encoding="utf-8"))
    output_paths: list[Path] = []

    metrics = [
        ("macro_auc", "Macro AUC"),
        ("macro_f1", "Macro F1"),
        ("micro_f1", "Micro F1"),
        ("subset_accuracy", "Subset Accuracy"),
    ]
    names = [name for _, name in metrics]
    baseline_values = [float(baseline.get(key, 0.0) or 0.0) for key, _ in metrics]
    tuned_values = [float(tuned.get(key, 0.0) or 0.0) for key, _ in metrics]
    positions = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=140)
    ax.bar(positions - width / 2, baseline_values, width, label="Baseline (threshold=0.5)", color="#94a3b8")
    ax.bar(positions + width / 2, tuned_values, width, label="Per-label threshold tuning", color="#2563eb")
    ax.set_title("Test Metrics Before and After Threshold Tuning")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(positions, names)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = output_dir / "threshold_metric_comparison.png"
    fig.savefig(output_path)
    plt.close(fig)
    output_paths.append(output_path)

    for metric_name, title, filename in [
        ("f1", "Per-label F1: Baseline vs Tuned Thresholds", "label_f1_comparison.png"),
        ("accuracy", "Per-label Accuracy: Baseline vs Tuned Thresholds", "label_accuracy_comparison.png"),
    ]:
        labels = [label for label in DISEASE_LABELS if label in tuned.get("per_label", {})]
        baseline_scores = [float(baseline.get("per_label", {}).get(label, {}).get(metric_name, 0.0) or 0.0) for label in labels]
        tuned_scores = [float(tuned.get("per_label", {}).get(label, {}).get(metric_name, 0.0) or 0.0) for label in labels]
        positions = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        ax.bar(positions - width / 2, baseline_scores, width, label="Baseline (0.5)", color="#94a3b8")
        ax.bar(positions + width / 2, tuned_scores, width, label="Tuned", color="#2563eb")
        ax.set_title(title)
        ax.set_ylabel(metric_name.upper())
        ax.set_ylim(0, 1)
        ax.set_xticks(positions, labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def create_figures(
    output_dir: Path,
    training_log: Path | None = None,
    metrics_json: Path | None = None,
    roc_json: Path | None = None,
    baseline_metrics_json: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if training_log:
        outputs.extend(save_training_curves(training_log, output_dir))
    if metrics_json:
        outputs.extend(save_label_metric_bars(metrics_json, output_dir))
    if roc_json:
        outputs.extend(save_roc_curves(roc_json, output_dir))
    if baseline_metrics_json and metrics_json:
        outputs.extend(save_threshold_comparison_figures(baseline_metrics_json, metrics_json, output_dir))
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
        baseline_metrics_json=Path(args.baseline_metrics_json) if args.baseline_metrics_json else None,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
