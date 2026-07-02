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
