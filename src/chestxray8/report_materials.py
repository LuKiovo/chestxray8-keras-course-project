"""Collect experiment artifacts into a Markdown report appendix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Markdown summary from project artifacts.")
    parser.add_argument("--manifest-summary", help="Path to manifests/summary.txt.")
    parser.add_argument("--training-summary", help="Path to training_summary.json.")
    parser.add_argument("--metrics-summary", help="Path to metrics_summary.json.")
    parser.add_argument("--figures-dir", help="Directory containing generated PNG figures.")
    parser.add_argument("--output", required=True, help="Output Markdown path.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_manifest_section(lines: list[str], path: Path) -> None:
    if not path.exists():
        return
    lines.extend(["## 数据集与分片摘要", "", "```text", path.read_text(encoding="utf-8").strip(), "```", ""])


def append_training_section(lines: list[str], path: Path) -> None:
    if not path.exists():
        return
    summary = read_json(path)
    lines.extend(
        [
            "## 训练过程摘要",
            "",
            f"- 模型：{summary.get('model', 'N/A')}",
            f"- 训练图片数：{summary.get('train_images', 'N/A')}",
            f"- 验证图片数：{summary.get('val_images', 'N/A')}",
            f"- 训练清单：`{summary.get('train_csv', 'N/A')}`",
            f"- 验证清单：`{summary.get('val_csv', 'N/A')}`",
            "",
        ]
    )
    history = summary.get("history", {})
    if history:
        lines.extend(["| 指标 | 最后一个 epoch |", "| --- | ---: |"])
        for metric, values in history.items():
            if values:
                lines.append(f"| {metric} | {float(values[-1]):.4f} |")
        lines.append("")


def append_metrics_section(lines: list[str], path: Path) -> None:
    if not path.exists():
        return
    metrics = read_json(path)
    lines.extend(
        [
            "## 测试集评估摘要",
            "",
            f"- 测试图片数：{metrics.get('test_images', 'N/A')}",
            f"- 阈值：{metrics.get('threshold', 'N/A')}",
            f"- Macro AUC：{format_optional(metrics.get('macro_auc'))}",
            f"- Macro F1：{format_optional(metrics.get('macro_f1'))}",
            f"- Micro F1：{format_optional(metrics.get('micro_f1'))}",
            f"- Subset Accuracy：{format_optional(metrics.get('subset_accuracy'))}",
            "",
            "| 疾病标签 | AUC | F1 | Accuracy | 阳性样本数 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, item in metrics.get("per_label", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_optional(item.get("auc")),
                    format_optional(item.get("f1")),
                    format_optional(item.get("accuracy")),
                    str(item.get("positive_count", 0)),
                ]
            )
            + " |"
        )
    lines.append("")


def append_figures_section(lines: list[str], figures_dir: Path) -> None:
    if not figures_dir.exists():
        return
    figures = sorted(figures_dir.glob("*.png"))
    if not figures:
        return
    lines.extend(["## 图表素材", ""])
    for figure in figures:
        lines.append(f"- `{figure.name}`")
    lines.append("")


def format_optional(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def build_report_materials(
    output: Path,
    manifest_summary: Path | None = None,
    training_summary: Path | None = None,
    metrics_summary: Path | None = None,
    figures_dir: Path | None = None,
) -> str:
    lines = [
        "# ChestX-ray8 项目报告素材",
        "",
        "本文件由脚本自动汇总，用于撰写课程专题报告和答辩材料。",
        "",
    ]
    if manifest_summary:
        append_manifest_section(lines, manifest_summary)
    if training_summary:
        append_training_section(lines, training_summary)
    if metrics_summary:
        append_metrics_section(lines, metrics_summary)
    if figures_dir:
        append_figures_section(lines, figures_dir)

    content = "\n".join(lines).rstrip() + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    args = parse_args()
    build_report_materials(
        output=Path(args.output),
        manifest_summary=Path(args.manifest_summary) if args.manifest_summary else None,
        training_summary=Path(args.training_summary) if args.training_summary else None,
        metrics_summary=Path(args.metrics_summary) if args.metrics_summary else None,
        figures_dir=Path(args.figures_dir) if args.figures_dir else None,
    )


if __name__ == "__main__":
    main()

