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


st.set_page_config(
    page_title="ChestX-ray8 多标签分类展示",
    layout="wide",
)


def path_input(label: str, default: str) -> Path:
    return Path(st.sidebar.text_input(label, default))


def exists(path: Path) -> bool:
    return bool(str(path)) and path.exists()


def show_metric_card(label: str, value) -> None:
    if value is None:
        st.metric(label, "N/A")
    elif isinstance(value, float):
        st.metric(label, f"{value:.3f}")
    else:
        st.metric(label, value)


st.title("ChestX-ray8 胸部 X 光多标签分类")
st.caption("Keras/TensorFlow 训练、评估与交互式结果展示")

st.sidebar.header("运行配置")
model_path = path_input("模型文件 (.keras)", "outputs/shard_001/best_model.keras")
metrics_path = path_input("评估指标 JSON", "outputs/evaluation/metrics_summary.json")
roc_path = path_input("ROC 数据 JSON", "outputs/evaluation/roc_curves.json")
training_log_path = path_input("训练日志 CSV", "outputs/shard_001/training_log.csv")
figures_dir = path_input("图表目录", "outputs/figures")
image_size = st.sidebar.number_input("输入尺寸", min_value=32, max_value=512, value=224, step=32)
threshold = st.sidebar.slider("疾病概率阈值", 0.0, 1.0, 0.5, 0.01)

tabs = st.tabs(["项目概览", "单图预测", "评估指标", "训练过程", "文件产物"])

with tabs[0]:
    st.subheader("项目流程")
    st.write(
        "本项目使用 ChestX-ray8 官方数据集完成 14 类胸部疾病多标签分类，"
        "通过分阶段训练适配云 GPU 有限数据盘，并提供模型评估与交互式展示。"
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_metric_card("标签数量", len(DISEASE_LABELS))
    with col2:
        show_metric_card("训练方式", "分阶段")
    with col3:
        show_metric_card("模型框架", "Keras")
    with col4:
        show_metric_card("展示方式", "Streamlit")

    st.subheader("14 类疾病标签")
    st.dataframe(pd.DataFrame({"label": DISEASE_LABELS}), hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("单张 X 光预测")
    uploaded = st.file_uploader("上传胸部 X 光图片", type=["png", "jpg", "jpeg"])
    if not exists(model_path):
        st.info("模型文件未就绪。")
    elif uploaded is None:
        st.info("等待图片输入。")
    else:
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getbuffer())
            temp_image = Path(f.name)

        prediction = predict_image(model_path, temp_image, int(image_size), threshold)
        prediction_df = prediction_to_dataframe(prediction)

        left, right = st.columns([0.9, 1.2])
        with left:
            st.image(str(temp_image), caption=uploaded.name, use_container_width=True)
        with right:
            st.write("预测阳性标签：", "、".join(prediction["positive_labels"]) or "无")
            st.dataframe(
                prediction_df.style.format({"probability": "{:.3f}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(prediction_df.set_index("label")["probability"])

with tabs[2]:
    st.subheader("评估指标")
    if not exists(metrics_path):
        st.info("未找到评估指标文件。完成 evaluate 步骤后，这里会展示整体指标和逐疾病指标。")
    else:
        metrics = load_json(metrics_path)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_metric_card("Macro AUC", metrics.get("macro_auc"))
        with col2:
            show_metric_card("Macro F1", metrics.get("macro_f1"))
        with col3:
            show_metric_card("Micro F1", metrics.get("micro_f1"))
        with col4:
            show_metric_card("Subset Accuracy", metrics.get("subset_accuracy"))

        metrics_df = metrics_to_dataframe(metrics)
        st.dataframe(
            metrics_df.style.format({"auc": "{:.3f}", "f1": "{:.3f}", "accuracy": "{:.3f}"}),
            hide_index=True,
            use_container_width=True,
        )
        chart_metric = st.radio("柱状图指标", ["auc", "f1", "accuracy"], horizontal=True)
        st.bar_chart(metrics_df.set_index("label")[chart_metric])

with tabs[3]:
    st.subheader("训练过程")
    if not exists(training_log_path):
        st.info("未找到训练日志。完成 training 步骤后，这里会展示 loss、AUC 和准确率曲线。")
    else:
        log_df = load_training_log(training_log_path)
        st.dataframe(log_df, hide_index=True, use_container_width=True)
        curve_options = [column for column in log_df.columns if column != "epoch"]
        selected = st.multiselect("选择曲线", curve_options, default=curve_options[: min(4, len(curve_options))])
        if selected:
            st.line_chart(log_df.set_index("epoch")[selected])

with tabs[4]:
    st.subheader("产物检查")
    artifact_rows = [
        ("模型文件", model_path),
        ("评估指标", metrics_path),
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
            st.image(str(image_path), caption=image_path.name, use_container_width=True)
