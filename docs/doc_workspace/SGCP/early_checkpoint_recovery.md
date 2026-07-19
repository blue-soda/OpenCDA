# SGCP Early Checkpoint Recovery Protocol

更新时间：2026-07-19

## 当前状态

远程 early-fusion checkpoint fine-tune 已经准备完成，但尚未开始训练。阻塞原因不是代码或 checkpoint 兼容性，而是 `mindspore-187` 上 8 张 GPU 均被占用约 22.2 GB，watcher 一直输出：

```text
no GPU below 6000 MiB; sleeping 300s
```

远程路径：

```text
mindspore-187:/data2/gzc/sgcp_early_train/
```

训练环境：

```text
conda env: opencood-gzc
code: /data2/gzc/code/OpenCOOD
```

## 已准备文件

| 文件 | 用途 |
| --- | --- |
| `/data2/gzc/sgcp_early_train/checkpoints/latest.pth` | 当前本地 `pointpillar_early_fusion` checkpoint 上传副本 |
| `/data2/gzc/sgcp_early_train/checkpoints/config.yaml` | checkpoint 配套配置 |
| `/data2/gzc/sgcp_early_train/configs/pointpillar_early_ckpt_compat_onecav.yaml` | 远程训练使用配置，已通过 1-step smoke 的 checkpoint shape 兼容检查 |
| `/data2/gzc/sgcp_early_train/runs/start_train_when_gpu_free.sh` | GPU 空闲 watcher |
| `/data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log` | watcher 日志 |

watcher PID 文件：

```text
/data2/gzc/sgcp_early_train/runs/train_gpu_waiter.pid
```

## Watcher 行为

脚本逻辑：

1. 每 300 秒查询 `nvidia-smi`；
2. 选择第一张显存使用低于 6000 MiB 的 GPU；
3. 执行：

```bash
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=/data2/gzc/code/OpenCOOD \
conda run -n opencood-gzc python opencood/tools/train.py \
  --hypes_yaml /data2/gzc/sgcp_early_train/configs/pointpillar_early_ckpt_compat_onecav.yaml \
  --checkpoint_path /data2/gzc/sgcp_early_train/checkpoints/latest.pth \
  --num_workers 2 \
  --max_steps 200 \
  --save_step_freq 50 \
  --keep_step_checkpoints 4
```

训练日志会写入：

```text
/data2/gzc/sgcp_early_train/logs/train_ckptcompat_200steps_<timestamp>.log
```

## 每轮轮询命令

```powershell
ssh mindspore-187 "tail -20 /data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
```

若 watcher 选中 GPU，会在日志中出现：

```text
selected GPU <id>
training log: /data2/gzc/sgcp_early_train/logs/train_ckptcompat_200steps_<timestamp>.log
```

随后查看训练日志：

```powershell
ssh mindspore-187 "ls -t /data2/gzc/sgcp_early_train/logs/train_ckptcompat_200steps_*.log | head -1 | xargs -r tail -80"
```

## 回收 checkpoint 后必须执行

训练完成后，先找最新 step checkpoint：

```powershell
ssh mindspore-187 "find /data2/gzc/sgcp_early_train -type f -name '*.pth' -printf '%T@ %p\n' | sort -n | tail -10"
```

将最新 fine-tuned checkpoint 下载到新的本地 artifact 目录，不覆盖旧结果：

```powershell
New-Item -ItemType Directory -Force docs\doc_workspace\SGCP\artifacts\early_checkpoint_ft_20260719 | Out-Null
scp mindspore-187:/path/to/latest_step_checkpoint.pth docs\doc_workspace\SGCP\artifacts\early_checkpoint_ft_20260719\
```

回收后必须重跑同一 checkpoint 下的关键实验：

1. 11 帧 smoke：
   - SGCP-PAPG mainline；
   - Pure late controlled singleton。
2. 41 帧主线：
   - SGCP-PAPG；
   - Pure late controlled singleton；
   - Figure 1 Pareto 中关键 SGCP/Pure late 点；
   - Figure 2/3 dependent artifacts。

所有新结果必须创建新的 artifact 目录并更新：

- `results.md`
- `status.md`
- `target.md`
- `paper_artifact_index.md`
- `artifacts/paper_artifact_index_20260719/paper_artifact_index.csv` 或新版本 index。

## 验收标准

fine-tuned checkpoint 只有在满足以下条件时才替换主文结果：

- SGCP-PAPG AP@0.3/AP@0.5 不低于当前 `0.81/0.78`；
- AP@0.7 明显改善，或至少能更好解释 high-IoU localization headroom；
- Pure late controlled 使用同一 checkpoint 重跑，确保 detector fairness 不被破坏；
- 若新 checkpoint 使 Pure late 同步提升且 SGCP 相对关系未改善，仍应保留旧主文口径，把 fine-tune 作为 negative/sensitivity artifact 记录。
