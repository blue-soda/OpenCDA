# OpenCOOD Multi-Scene Area Confidence Plan

本文档记录将 LGCP area confidence validation 扩展到 OpenCOOD OPV2V / V2XSet 多 seed、多场景评估时已经确认的资源、协议入口、可复用 checkpoint 和当前阻塞边界。

## 当前结论

- Windows 本地 `C:\Workspace\OpenCOOD` 没有 `dataset/` 目录，不能直接在本机完成 OPV2V / V2XSet 多场景推理。
- OpenCOOD 远端共享环境已经具备可用的数据集和运行约定；论文级多场景结果应在 `mindspore-186` 上跑。
- 仓库模板 YAML 多数仍使用 `dataset/OPV2V/{train,validate,test}`，适合本地软链接数据后的 smoke test。
- 与当前 OpenCOOD A 线论文可比协议对齐时，应优先使用 `OPV2V(Culver) regular-history` 入口：`my_opv2v/train`、`my_opv2v/validate`、`my_opv2v/test_culver_city`。
- LGCP area-level AP / recall 不应只读取 `eval.yaml` 的 whole-frame AP；需要保存或截取 postprocessed `pred_box_tensor`、`pred_score`、`gt_box_tensor`，再按 LGCP area 切片统计。

## Confirmed Environment

OpenCOOD 共享环境来自 `C:\Workspace\OpenCOOD\agent-doc\environment.md`：

| Item | Value |
| --- | --- |
| Local workspace | `C:\Workspace\OpenCOOD` |
| Remote workspace | `mindspore-186:/data1/wql/gzc/workspace/OpenCOOD` |
| Remote Python | `/data1/wql/yyq/anaconda3/envs/opencood-gzc/bin/python` |
| Remote OPV2V train | `/data1/wql/gzc/dataset/opv2v/train` |
| Remote OPV2V val | `/data1/wql/gzc/dataset/opv2v/val` |
| Remote OPV2V test | `/data1/wql/gzc/dataset/opv2v/test` |
| Remote logs symlink | `/data1/wql/gzc/workspace/OpenCOOD/opencood/logs` |
| Actual checkpoint/log store | `/data0/wql/gzc/workspace/OpenCOOD_checkpoint_store/opencood_logs` |

本地检查结果：

```text
C:\Workspace\OpenCOOD\dataset
```

当前不存在，因此本地只能做配置 / checkpoint / 脚本审计，不能作为论文级多场景 inference 环境。

## Protocol Choices

### Template OPV2V Path

多数 `C:\Workspace\OpenCOOD\opencood\hypes_yaml\opv2v\lidar_only\*.yaml` 使用：

```yaml
root_dir: "dataset/OPV2V/train"
validate_dir: "dataset/OPV2V/validate"
test_dir: "dataset/OPV2V/test"
```

用途：

- 本地或远端建立 `dataset/OPV2V` 软链接后的快速 smoke test。
- 检查 `inference.py`、`--save_npy`、`eval.yaml` 和 LGCP area slicer 兼容性。

边界：

- 这不是当前 A 线记录的 `OPV2V(Culver) regular-history` 论文可比入口。
- 若使用该入口，需要在结果中明确标注为 template OPV2V split。

### OPV2V(Culver) Regular-History Path

OpenCOOD `agent-doc/protocol_notes.md` 记录的论文可比入口：

```text
train: my_opv2v/train
validation: my_opv2v/validate
test: my_opv2v/test_culver_city
with_history_frames: true
num_sweep_frames: 2
binomial_n: 10
max_cav: 5
comm_range: 70
```

用途：

- 论文级多场景 area confidence validation。
- 与 CoBEVFlow / CoDynTrust 风格 regular-history 协议保持可解释的比较边界。

边界：

- 需要确认目标 checkpoint 的 `config.yaml` 是否已经切到 Culver split。
- 如果 checkpoint config 仍指向 `dataset/OPV2V/...`，应复制配置到 run 目录并显式记录 split 修改。

## Checkpoint Inventory

来自 `C:\Workspace\OpenCOOD\checkpoints\manifest.md` 和本地 `config.yaml` 审计：

| ID | Role | Local config | Local checkpoint | Path / protocol note |
| --- | --- | --- | --- | --- |
| A-D20 | promoted calibrated SpikeMem state | `checkpoints/abc_key_results/A-D20/config.yaml` | `net_step_e1_i1000.pth` | config uses `dataset/OPV2V`, `max_cav: 3` |
| A-D23 | reliable-read SpikeMem | `checkpoints/abc_key_results/A-D23/config.yaml` | `net_step_e1_i1000.pth` | config uses `dataset/OPV2V`, `max_cav: 3` |
| A-D24 | multi-timescale SpikeMem | `checkpoints/abc_key_results/A-D24/config.yaml` | `net_step_e1_i1003.pth` | config uses `dataset/OPV2V`, `max_cav: 3` |
| A-D25 | long-prior multi-timescale SpikeMem | `checkpoints/abc_key_results/A-D25/config.yaml` | `net_step_e1_i1000.pth` | config uses `dataset/OPV2V`, `max_cav: 3` |
| A-D26 | conservative multi-timescale SpikeMem | `checkpoints/abc_key_results/A-D26/config.yaml` | `net_step_e1_i1002.pth` | config uses `dataset/OPV2V`, `max_cav: 3` |
| B-D07 | promoted objectness region selector | `checkpoints/abc_key_results/B-D07/config.yaml` | `net_step_e12_i500.pth` | config uses `dataset/OPV2V`, `max_cav: 5` |
| C-D05 | promoted TTFS q2 latent | `checkpoints/abc_key_results/C-D05/config.yaml` | `net_step_e1_i1000.pth` | config uses remote `/data1/wql/gzc/dataset/opv2v`, `max_cav: 2` |

LGCP area confidence 的第一优先候选是 B-D07 或其他 `max_cav: 5` / `comm_range: 70` 的 checkpoint，因为它与 OPV2V 多 CAV setting 更接近。A-D20/A-D23-A-D26 的 `max_cav: 3` 可作为模型族补充，但不应混入同一主表。

## Proposed Run Matrix

第一轮建议只做小而稳的多场景验证：

| Axis | Values | Reason |
| --- | --- | --- |
| Dataset | OPV2V Culver test | 与 OpenCOOD 当前论文可比协议对齐 |
| Frames | 400-frame gate, then full set | 先得到可复核趋势，再扩展 full set |
| Seeds | `303`, `7`, `11`, `23` | 与已有 OpenCOOD / LGCP smoke seed 风格兼容 |
| Async | `p=0.0`, optional `p=0.3` | 先验证 clean area confidence，再看异步稳定性 |
| Checkpoint | one fixed model family | 避免模型差异污染 confidence-quality 相关性 |
| Metrics | Spearman / Pearson vs recall@0.5, AP@0.5, AP@0.7 | 对齐现有 `lgcp_area_confidence_eval.py` |

## Remote Command Skeleton

正式命令需要按实际 checkpoint 目录调整；日志和输出目录必须回填到 `log.md`。

```bash
cd /data1/wql/gzc/workspace/OpenCOOD
CUDA_VISIBLE_DEVICES=<gpu_id> \
PYTHONPATH=/data1/wql/gzc/workspace/OpenCOOD:$PYTHONPATH \
/data1/wql/yyq/anaconda3/envs/opencood-gzc/bin/python opencood/tools/inference.py \
  --model_dir opencood/logs/<ckpt-dir> \
  --fusion_method intermediate \
  --save_npy \
  --note _lgcp_area_seed<seed> \
  --max_frames 400 \
  --seed <seed> \
  --p 0.0 \
  --async_max_delay_step 0 \
  > opencood/logs/lgcp_area_confidence_<run_id>.log 2>&1 &
```

若现有 `inference.py` 不支持 `--max_frames` / `--seed` / `--p` / `--async_max_delay_step`，则应使用对应线路已经扩展过的 inference wrapper，或先补一个只导出 postprocessed prediction / GT 的 LGCP adapter。不能在结果表里把不支持的参数写成已生效。

## LGCP Area Slicing Adapter

从 OpenCOOD 多场景结果进入 LGCP area confidence validation，需要以下最小输出：

```text
frame_id
scenario_id
cav_id or ego id
pred_box_tensor
pred_score
gt_box_tensor
pose / transformation metadata
dataset split
checkpoint id
seed
async setting
```

然后复用 OpenCDA 侧已有统计口径：

- 按 LGCP grid / ROI 将 prediction 与 GT 切入 area。
- 对每个 area-frame 统计 `recall_05`、`ap_05`、`ap_07`。
- 比较 density / detector-score / noisy-or confidence 与 area quality 的 Spearman / Pearson。
- 输出到 `docs/doc_workspace/LGCP/experiments/opencood_area_confidence/<run_id>/`。

## Remaining Work

- 在 `mindspore-186` 上确认当前可用 checkpoint 目录是否含 Culver split config。
- 选择一个固定 checkpoint family，避免多模型混表。
- 确认推理 wrapper 是否支持 seed / max_frames / async 参数；不支持时先补 adapter。
- 跑 400-frame gate 多 seed，记录 `eval.yaml`、`npy` manifest、area summary 和命令日志。
- 只有多 seed 结果稳定后，才扩展 full-set 和 V2XSet。
