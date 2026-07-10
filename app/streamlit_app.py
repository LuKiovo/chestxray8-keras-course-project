from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from chestxray8.app_utils import (
    load_json,
    load_training_log,
    metrics_to_dataframe,
    prediction_to_dataframe,
)
from chestxray8.constants import DISEASE_LABELS
from chestxray8.predict import predict_image


st.set_page_config(page_title="ChestX-ray8 多标签分类展示", layout="wide")

DEFAULT_ARTIFACT_ROOT = Path("chestxray8_gpu_results") if Path("chestxray8_gpu_results").exists() else Path(".")
DEFAULT_SHARD = "shard_015"


def path_input(label: str, default: Path) -> Path:
    return Path(st.sidebar.text_input(label, str(default)))


def exists(path: Path) -> bool:
    return bool(str(path)) and path.exists()


def show_metric(label: str, value, baseline=None) -> None:
    if value is None:
        st.metric(label, "N/A")
        return
    formatted = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
    delta = None
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        delta = f"{float(value) - float(baseline):+.3f} vs baseline"
    st.metric(label, formatted, delta=delta)


def load_threshold_map(path: Path) -> dict[str, float]:
    if not exists(path):
        return {}
    raw = load_json(path)
    return {
        label: float(item["threshold"])
        for label, item in raw.items()
        if isinstance(item, dict) and "threshold" in item
    }


st.title("ChestX-ray8 胸部 X 光多标签分类")
st.caption("TensorFlow/Keras 分阶段训练 | 14 类疾病 | 验证集分类别阈值优化")

st.sidebar.header("结果配置")
artifact_root = path_input("结果根目录", DEFAULT_ARTIFACT_ROOT)
model_path = path_input("模型文件 (.keras)", artifact_root / "outputs" / DEFAULT_SHARD / "best_model.keras")
baseline_metrics_path = path_input(
    "基线指标 JSON", artifact_root / "outputs" / "evaluation" / "metrics_summary.json"
)
tuned_metrics_path = path_input(
    "优化后指标 JSON", artifact_root / "outputs" / "threshold_tuning" / "metrics_summary_tuned.json"
)
thresholds_path = path_input(
    "分类别阈值 JSON", artifact_root / "outputs" / "threshold_tuning" / "thresholds.json"
)
roc_path = path_input("ROC 数据 JSON", artifact_root / "outputs" / "evaluation" / "roc_curves.json")
training_log_path = path_input(
    "训练日志 CSV", artifact_root / "outputs" / DEFAULT_SHARD / "training_log.csv"
)
figures_dir = path_input("图表目录", artifact_root / "outputs" / "figures_tuned")
image_size = st.sidebar.number_input("输入尺寸", min_value=32, max_value=512, value=224, step=32)
fallback_threshold = st.sidebar.slider("未配置标签的默认阈值", 0.0, 1.0, 0.5, 0.01)

baseline_metrics = load_json(baseline_metrics_path) if exists(baseline_metrics_path) else None
tuned_metrics = load_json(tuned_metrics_path) if exists(tuned_metrics_path) else None
active_metrics = tuned_metrics or baseline_metrics
threshold_map = load_threshold_map(thresholds_path)

tabs = st.tabs(["项目概览", "单图预测", "评估指标", "训练过程", "图表与产物"])

with tabs[0]:
    st.subheader("实验闭环")
    st.write(
        "项目使用 ChestX-ray8 完成 14 类胸部疾病多标签分类。训练阶段将训练集划分为 16 个 shard，"
        "在云 GPU 上连续 checkpoint 续训；测试阶段保留固定测试集，并将分类别阈值仅在验证集上选择后应用到测试集。"
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_metric("疾病标签", len(DISEASE_LABELS))
    with col2:
        show_metric("训练分片", 16)
    with col3:
        show_metric("主干网络", "MobileNetV2")
    with col4:
        show_metric("测试图片", active_metrics.get("test_images") if active_metrics else None)

    st.subheader("14 类疾病标签")
    st.dataframe(pd.DataFrame({"label": DISEASE_LABELS}), hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("单张胸部 X 光预测")
    use_tuned_thresholds = st.checkbox(
        "使用验证集选择的分类别阈值",
        value=bool(threshold_map),
        disabled=not bool(threshold_map),
    )
    if use_tuned_thresholds:
        st.caption("每个疾病标签采用在验证集上使 F1 最大的阈值；仅用于课程实验展示，不构成临床诊断。")
    else:
        st.caption(f"全部标签统一使用阈值 {fallback_threshold:.2f}；仅用于课程实验展示，不构成临床诊断。")

    uploaded = st.file_uploader("上传胸部 X 光图像", type=["png", "jpg", "jpeg"])
    if not exists(model_path):
        st.info("未找到模型文件，请在侧栏确认模型路径。")
    elif uploaded is None:
        st.info("请选择一张 PNG、JPG 或 JPEG 图像。")
    else:
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            file.write(uploaded.getbuffer())
            temp_image = Path(file.name)

        prediction = predict_image(
            model_path,
            temp_image,
            int(image_size),
            fallback_threshold,
            threshold_map if use_tuned_thresholds else None,
        )
        prediction_df = prediction_to_dataframe(prediction)
        left, right = st.columns([0.9, 1.2])
        with left:
            st.image(str(temp_image), caption=uploaded.name, use_column_width=True)
        with right:
            positive_labels = prediction["positive_labels"]
            st.write("预测阳性标签：", "、".join(positive_labels) if positive_labels else "无")
            st.dataframe(
                prediction_df.style.format({"probability": "{:.3f}", "threshold": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(prediction_df.set_index("label")["probability"])

with tabs[2]:
    st.subheader("测试集评估")
    if active_metrics is None:
        st.info("未找到评估指标文件，请在侧栏确认路径。")
    else:
        if tuned_metrics:
            st.success("当前展示：验证集分类别阈值优化后，在独立测试集上的结果。AUC 与阈值无关，F1 与准确率会随阈值改变。")
        else:
            st.warning("未找到阈值优化结果，当前仅展示统一阈值 0.5 的基线结果。")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_metric("Macro AUC", active_metrics.get("macro_auc"), baseline_metrics.get("macro_auc") if tuned_metrics else None)
        with col2:
            show_metric("Macro F1", active_metrics.get("macro_f1"), baseline_metrics.get("macro_f1") if tuned_metrics else None)
        with col3:
            show_metric("Micro F1", active_metrics.get("micro_f1"), baseline_metrics.get("micro_f1") if tuned_metrics else None)
        with col4:
            show_metric("Subset Accuracy", active_metrics.get("subset_accuracy"), baseline_metrics.get("subset_accuracy") if tuned_metrics else None)

        metrics_df = metrics_to_dataframe(active_metrics)
        if baseline_metrics and tuned_metrics:
            baseline_df = metrics_to_dataframe(baseline_metrics).rename(
                columns={"f1": "baseline_f1", "accuracy": "baseline_accuracy"}
            )[["label", "baseline_f1", "baseline_accuracy"]]
            comparison_df = metrics_df.merge(baseline_df, on="label", how="left")
            st.subheader("逐标签 F1 与 Accuracy 对比")
            st.dataframe(
                comparison_df.style.format(
                    {
                        "auc": "{:.3f}", "baseline_f1": "{:.3f}", "f1": "{:.3f}",
                        "baseline_accuracy": "{:.3f}", "accuracy": "{:.3f}",
                        "threshold": "{:.2f}", "validation_f1": "{:.3f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            chart_data = comparison_df.set_index("label")[["baseline_f1", "f1"]].rename(
                columns={"baseline_f1": "Baseline F1 (0.5)", "f1": "Tuned F1"}
            )
            st.bar_chart(chart_data)
        else:
            st.dataframe(
                metrics_df.style.format({"auc": "{:.3f}", "f1": "{:.3f}", "accuracy": "{:.3f}"}),
                hide_index=True,
                use_container_width=True,
            )

with tabs[3]:
    st.subheader("最后一个训练分片的训练过程")
    if not exists(training_log_path):
        st.info("未找到训练日志，请在侧栏确认路径。")
    else:
        log_df = load_training_log(training_log_path)
        st.dataframe(log_df, hide_index=True, use_container_width=True)
        curve_options = [column for column in log_df.columns if column not in {"epoch", "_epoch"}]
        selected = st.multiselect("选择曲线", curve_options, default=curve_options[: min(4, len(curve_options))])
        if selected:
            st.line_chart(log_df.set_index("epoch")[selected])

with tabs[4]:
    st.subheader("可复现实验产物")
    artifact_rows = [
        ("模型文件", model_path),
        ("基线评估", baseline_metrics_path),
        ("阈值优化评估", tuned_metrics_path),
        ("分类别阈值", thresholds_path),
        ("ROC 数据", roc_path),
        ("训练日志", training_log_path),
        ("图表目录", figures_dir),
    ]
    st.dataframe(
        pd.DataFrame(
            {
                "artifact": [row[0] for row in artifact_rows],
                "path": [str(row[1]) for row in artifact_rows],
                "exists": [exists(row[1]) for row in artifact_rows],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    if exists(figures_dir):
        for image_path in sorted(figures_dir.glob("*.png")):
            st.image(str(image_path), caption=image_path.name, use_column_width=True)
