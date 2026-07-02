# ChestX-ray8 Keras Course Project

基于 Keras/TensorFlow 的 ChestX-ray8 胸部 X 光 14 类多标签分类课程项目。

本仓库将按验收项逐步实现：数据准备、分阶段训练、评估、交互式网页展示和课程报告。

> 注意：原始数据集、模型权重、训练输出和本地 `Agents.md` 不提交到 GitHub。

## 当前验收项：数据清单与分片

本阶段已支持从 ChestX-ray8 官方 `Data_Entry_2017.csv` 生成训练清单，并将训练集按 shard 切分，便于在云 GPU 的有限数据盘上分阶段训练。

```powershell
$env:PYTHONPATH="src"
python -m chestxray8.prepare_data `
  --metadata-csv D:\ChestXray8\Data_Entry_2017.csv `
  --image-root D:\ChestXray8\images `
  --output-dir manifests `
  --shard-size 10000 `
  --require-images
```

生成结果：

- `manifests/all.csv`
- `manifests/train.csv`
- `manifests/val.csv`
- `manifests/test.csv`
- `manifests/shards/train_shard_000.csv` 等训练分片
- `manifests/summary.txt`

训练某个分片前，可将该分片图片复制到云 GPU 工作目录：

```powershell
$env:PYTHONPATH="src"
python -m chestxray8.stage_shard `
  --shard-csv manifests\shards\train_shard_000.csv `
  --image-root D:\ChestXray8\images `
  --stage-dir D:\stage_current_shard `
  --clean
```

## 当前验收项：Keras 训练入口

训练入口支持读取一个训练 shard 和固定验证集，完成 14 类多标签训练，并输出 checkpoint、训练日志和训练摘要。

```powershell
$env:PYTHONPATH="src"
python -m chestxray8.training `
  --train-csv manifests\shards\train_shard_000.csv `
  --val-csv manifests\val.csv `
  --image-root D:\stage_current_shard `
  --output-dir outputs\shard_000 `
  --model mobilenet_v2 `
  --weights imagenet `
  --image-size 224 `
  --batch-size 32 `
  --epochs 5
```

继续训练下一个 shard 时，先用 `stage_shard` 切换图片目录，再加载上一步 checkpoint：

```powershell
$env:PYTHONPATH="src"
python -m chestxray8.training `
  --train-csv manifests\shards\train_shard_001.csv `
  --val-csv manifests\val.csv `
  --image-root D:\stage_current_shard `
  --output-dir outputs\shard_001 `
  --resume-from outputs\shard_000\best_model.keras `
  --model mobilenet_v2 `
  --weights imagenet `
  --image-size 224 `
  --batch-size 32 `
  --epochs 5
```

本机快速自测可使用测试里的 tiny CNN 路径，不需要真实 ChestX-ray8 数据。

## 当前验收项：模型评估入口

训练完成后，可在固定测试集上输出整体指标、逐疾病指标、ROC 曲线数据和样例预测 CSV。

```powershell
$env:PYTHONPATH="src"
python -m chestxray8.evaluate `
  --model-path outputs\shard_001\best_model.keras `
  --test-csv manifests\test.csv `
  --image-root D:\ChestXray8\images `
  --output-dir outputs\evaluation `
  --image-size 224 `
  --batch-size 32 `
  --threshold 0.5
```

生成结果：

- `outputs/evaluation/metrics_summary.json`
- `outputs/evaluation/roc_curves.json`
- `outputs/evaluation/predictions.csv`

## 当前验收项：云 GPU 工作流配置

复制并修改示例配置：

```powershell
copy configs\cloud_gpu.example.json configs\cloud_gpu.local.json
```

先预览命令，不实际执行：

```powershell
$env:PYTHONPATH="src"
python scripts\run_cloud_workflow.py --config configs\cloud_gpu.local.json --step prepare
python scripts\run_cloud_workflow.py --config configs\cloud_gpu.local.json --step stage-train --shard-id 0
python scripts\run_cloud_workflow.py --config configs\cloud_gpu.local.json --step stage-train --shard-id 1
python scripts\run_cloud_workflow.py --config configs\cloud_gpu.local.json --step evaluate --shard-id 1
```

确认路径无误后，加 `--execute` 真正运行：

```powershell
$env:PYTHONPATH="src"
python scripts\run_cloud_workflow.py --config configs\cloud_gpu.local.json --step stage-train --shard-id 0 --execute
```

`stage-train` 会先清理并复制当前 shard 图片，再启动训练；当 `shard-id > 0` 时，会默认从上一个 shard 的 `best_model.keras` 继续训练。

## 当前验收项：提交包构建

最终提交前可生成项目 ZIP。默认打包 Git 跟踪的代码和文档，并排除原始数据、模型权重、训练输出、压缩包和本地 `Agents.md`。

```powershell
$env:PYTHONPATH="src"
python scripts\build_submission_zip.py --output dist\chestxray8-keras-course-project.zip
```

如果要把最终报告加入提交包：

```powershell
$env:PYTHONPATH="src"
python scripts\build_submission_zip.py `
  --output dist\chestxray8-keras-course-project.zip `
  --include reports\final_report.docx
```

压缩包内会额外生成 `SUBMISSION_MANIFEST.txt`，列出实际包含的文件。
