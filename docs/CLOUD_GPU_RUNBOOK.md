# 云 GPU 运行手册

本手册用于在 RTX4090 云 GPU 上运行 ChestX-ray8 分阶段训练流程。目标是在有限数据盘下完成全量训练集轮流参与训练，并保留固定验证集/测试集用于统一评估。

## 1. 租用配置

建议配置：

- GPU：RTX4090
- CPU：16 核左右
- 内存：120 GB 左右
- 系统盘：30 GB
- 数据盘：至少 50 GB；若平台允许，建议扩容到 100-193 GB
- CUDA：平台提供即可，优先选择 TensorFlow 兼容镜像

## 2. 推荐目录

```text
/workspace/project                         # 本项目仓库
/workspace/data/ChestXray8                 # ChestX-ray8 元数据和图片
/workspace/data/ChestXray8/Data_Entry_2017.csv
/workspace/data/ChestXray8/images
/workspace/stage_current_shard             # 当前训练分片图片
/workspace/project/manifests               # 生成的清单和分片
/workspace/project/outputs                 # 模型、日志、评估结果
```

如果数据盘只有 50 GB，不建议同时保留太多冗余复制。`stage_current_shard` 只保存当前 shard，训练完一个 shard 后会清理并切换到下一个 shard。

## 3. 环境准备

```bash
cd /workspace
git clone https://github.com/LuKiovo/chestxray8-keras-course-project.git project
cd /workspace/project
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

## 4. 配置路径

```bash
cp configs/cloud_gpu.example.json configs/cloud_gpu.local.json
```

打开 `configs/cloud_gpu.local.json`，确认以下路径与云 GPU 实际目录一致：

- `metadata_csv`
- `raw_image_root`
- `manifest_dir`
- `stage_dir`
- `output_dir`

## 5. 生成清单和训练分片

先预览命令：

```bash
python scripts/run_cloud_workflow.py --config configs/cloud_gpu.local.json --step prepare
```

确认无误后执行：

```bash
python scripts/run_cloud_workflow.py --config configs/cloud_gpu.local.json --step prepare --execute
```

成功后应出现：

- `manifests/all.csv`
- `manifests/train.csv`
- `manifests/val.csv`
- `manifests/test.csv`
- `manifests/shards/index.csv`
- `manifests/shards/train_shard_000.csv` 等
- `manifests/summary.txt`

## 6. 分阶段训练

训练第 0 个 shard：

```bash
python scripts/run_cloud_workflow.py --config configs/cloud_gpu.local.json --step stage-train --shard-id 0 --execute
```

训练第 1 个 shard：

```bash
python scripts/run_cloud_workflow.py --config configs/cloud_gpu.local.json --step stage-train --shard-id 1 --execute
```

当 `shard-id > 0` 时，脚本默认使用上一个 shard 的 `best_model.keras` 继续训练。例如第 1 个 shard 会加载：

```text
outputs/shard_000/best_model.keras
```

每个 shard 的输出目录：

```text
outputs/shard_000
outputs/shard_001
...
```

每个目录应包含：

- `best_model.keras`
- `last_model.keras`
- `training_log.csv`
- `training_summary.json`

## 7. 测试集评估

假设最后训练到 `shard_001`：

```bash
python scripts/run_cloud_workflow.py --config configs/cloud_gpu.local.json --step evaluate --shard-id 1 --execute
```

评估输出：

```text
outputs/evaluation/metrics_summary.json
outputs/evaluation/roc_curves.json
outputs/evaluation/predictions.csv
```

## 8. 生成图表与报告素材

```bash
python -m chestxray8.visualize \
  --training-log outputs/shard_001/training_log.csv \
  --metrics-json outputs/evaluation/metrics_summary.json \
  --roc-json outputs/evaluation/roc_curves.json \
  --output-dir outputs/figures

python -m chestxray8.report_materials \
  --manifest-summary manifests/summary.txt \
  --training-summary outputs/shard_001/training_summary.json \
  --metrics-summary outputs/evaluation/metrics_summary.json \
  --figures-dir outputs/figures \
  --output reports/report_materials.md
```

## 9. 启动网页

```bash
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

如果云平台提供公网访问，打开对应端口即可查看网页。

## 10. 注意事项

- 不要把原始数据集、`outputs/`、模型权重打进最终 GitHub 提交。
- 云端训练完成后，建议只下载 `outputs/evaluation`、`outputs/figures`、最终 `best_model.keras`、报告素材和必要截图。
- 如果数据盘不足，减小 `shard_size`，例如从 10000 调整到 5000。
- 如果训练速度太慢，先用较少 shard 跑通流程，再扩大训练范围。
- 最终报告中要说明：该系统用于课程实验和辅助研究演示，不作为临床诊断工具。
