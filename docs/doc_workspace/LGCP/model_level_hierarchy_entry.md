# LGCP Model-Level Hierarchy Entry

## 目的

本文档记录 LGCP 从当前 hierarchy proxy 推进到真实 model-level hierarchy 的代码入口、可落地路线和论文边界。

当前已经完成：

- RSU area assignment / area-task group / leader selection。
- raw LiDAR area slice manifest。
- member-to-leader 与 leader-to-RSU upload plan。
- scheduled / multi-slot NS3 replay 与 request-level lifecycle diagnostics。
- leader local result / RSU global aggregation proxy。

当前仍未完成：

- neural feature tensor 的 area-specific slicing。
- leader 端基于 group members 的真实 local fusion。
- RSU 端基于多个 leader results 的真实 global perception result。

因此，现阶段不能把 hierarchy aggregation proxy 写成完整模型级 LGCP AP 结果。它只能用于说明数据接口、覆盖率、byte proxy、scheduler 和 control-plane 行为。

## 已确认代码入口

OpenCOOD manager:

```text
opencda/core/ml_libs/opencood_manager.py
```

关键能力：

- `OpenCOODManager.inference(...)` 已支持 `late`、`early`、`intermediate`。
- `OpenCOODManager.naive_late_fusion(...)` 可将多个 receiver/source 的预测框与 score 做 NMS 融合。
- `return_object_ids=True` 已在部分 LGCP/SGCP 离线评估中使用，可保留 object-level diagnostics。

OpenCOOD inference utility:

```text
opencood/opencood/tools/inference_utils.py
```

关键事实：

- `inference_late_fusion(...)` 对每个 CAV 独立跑模型，然后由 dataset post-process 融合输出。
- `inference_intermediate_fusion(...)` 当前只是复用 early-fusion 风格入口，对 `batch_data['ego']` 统一推理。

Intermediate fusion dataset:

```text
opencood/opencood/data_utils/datasets/intermediate_fusion_dataset.py
```

关键字段：

- `processed_lidar`
- `record_len`
- `pairwise_t_matrix`
- `spatial_correction_matrix`
- `infra`
- `velocity`
- `time_delay`

这些字段说明 neural feature slicing 不能只改 LGCP CSV。它需要在 OpenCOOD dataset / collate / model feature encoder 之间增加 area mask 或 BEV feature map crop 逻辑。

Offline dataset:

```text
opencda/core/common/offline_dataset.py
```

关键能力：

- `OPV2VFrameDataset.load_frame(...)` 可以按 `ego_cav_id` 和 `cav_ids` 载入一帧的 CAV 子集。
- 这已经支撑 `lgcp_subset_ablation_eval.py` 的 perception-only selective sharing。

SGCP late-fusion reference:

```text
opencda/tools/offline_inference.py
```

关键参考：

- `--sgcp_inter_cluster_late_fusion` 路径会对多个 receiver/source 分别推理。
- 最后调用 `OpenCOODManager.naive_late_fusion(...)` 合并预测框与 score。
- 该路径可作为 LGCP box-level hierarchy adapter 的直接实现参考。

## 推荐推进路线

### Phase A: Box-Level Hierarchy Adapter

目标：先实现一个真实调用 OpenCOOD 推理的 hierarchy adapter，但 fusion 粒度是 detection boxes，而不是 neural feature tensor。

输入：

- `area_assignment_plan.csv`
- `feature_slice_manifest.csv` 或 `area_records.csv`
- OpenCDA / OPV2V-style dump
- OpenCOOD coperception config

步骤：

1. 对每个 `(timestamp, area_id)` 读取 `leader_id` 和 `group_members`。
2. 以 `leader_id` 作为 ego，使用 `OPV2VFrameDataset.load_frame(..., ego_cav_id=leader_id, cav_ids=group_members)` 载入 local group frame。
3. 调用 `OpenCOODManager.inference(...)` 得到该 area group 的 leader local prediction。
4. 将 local prediction 按 area ROI 裁剪，避免把 leader 对其它 area 的预测误算为该 area 的 local result。
5. 对同一 timestamp 的多个 leader local predictions 调用 `OpenCOODManager.naive_late_fusion(...)` 得到 RSU global prediction proxy。
6. 对 RSU global prediction 计算 AP / recall，并输出 per-area、per-frame 和 global summary。

输出建议：

```text
leader_local_predictions.csv
rsu_global_predictions.csv
rsu_global_eval_summary.csv
object_diagnostics.csv
config.yaml
notes.md
```

论文边界：

- 可以称为 "box-level hierarchical fusion" 或 "late-fusion implementation of LGCP hierarchy"。
- 不能称为 neural feature slicing。
- 可用于区分 flat selective sharing 与 local-to-global hierarchy 的结构收益。

### Phase B: Neural Feature Slice Adapter

目标：把 raw LiDAR slice proxy 替换为 model feature map / BEV feature tensor slice。

需要改造点：

1. 在 OpenCOOD preprocessing 或 model encoder 后暴露 per-CAV BEV feature tensor。
2. 将 LGCP area cell 映射到 feature map index range。
3. 对每个 area-task group 只传输对应 feature slice。
4. 在 leader 端按 `pairwise_t_matrix` / `spatial_correction_matrix` 对齐并融合 group member slices。
5. 在 RSU 端聚合多个 leader result，处理跨 area duplicate boxes。

主要风险：

- `IntermediateFusionDataset` 当前将所有 selected CAV 的 `processed_lidar` merge 后进入模型，缺少 per-area tensor crop 接口。
- `inference_intermediate_fusion(...)` 当前没有返回中间 feature tensor。
- 不同模型的 feature stride、BEV range、post-process anchor 都可能不同；需要先限定 PointPillar intermediate fusion。

最小实现建议：

1. 先只支持 `point_pillar_intermediate_fusion`。
2. 固定 `proj_first=True`，减少坐标系分支。
3. 只支持 rectangular area cell 到 BEV feature map 的 axis-aligned crop。
4. 先离线导出 feature slice byte / shape / area coverage，不立即跑端到端 AP。
5. 再接入 leader local fusion 和 RSU global aggregation。

论文边界：

- 只有 Phase B 跑通后，才能把结果称为 LGCP neural feature slicing / model-level intermediate fusion。
- 在 Phase A 之前，当前 hierarchy aggregation 仍是 quality proxy。

## 与 target.md 的关系

该审计将 `实现 LGCP 专用 RSU area assignment、leader local fusion 和 RSU global aggregation 管线` 拆成三个层级：

| 层级 | 当前状态 | 可写入论文的口径 |
| --- | --- | --- |
| Control-plane hierarchy | 已完成 | RSU assignment、area-task group、leader upload、scheduler、overhead |
| Box-level hierarchy | 待实现 | late-fusion hierarchy ablation / structure benefit |
| Neural feature hierarchy | 待实现 | full LGCP model-level feature slicing and local-to-global fusion |

2026-07-18 已新增 Phase A 入口：

```text
opencda/tools/lgcp_hierarchy_late_fusion_eval.py
```

该脚本按 hierarchy assignment plan 中的 `(timestamp, area_id, leader_id, group_members)` 真实调用 OpenCOOD，先在 leader/group 上推理，再把 prediction / GT 转到 world 坐标并按 area 裁剪，最后对同一 timestamp 的多个 area-local leader prediction 做 RSU box-level late fusion 与 AP 统计。

下一步最小任务应是使用 Top-30 计划跑 1-3 帧 smoke，确认输出的 `leader_local_predictions.csv`、`rsu_global_frame_summary.csv` 和 `rsu_global_eval_summary.csv` 可以进入 local-to-global ablation 结果表。

## 2026-07-18 Box-Level Smoke

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_smoke_area2
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_smoke_area2 --max-frames 1 --max-areas-per-frame 2 --fusion-method late
```

结果：

| Metric | Value |
| --- | ---: |
| Frames | 1 |
| Assignment rows | 2 |
| Cached group inference calls | 2 |
| RSU fused pred boxes | 6 |
| RSU fused GT boxes | 6 |
| AP@0.3 | 1.000000 |
| AP@0.5 | 1.000000 |
| AP@0.7 | 0.833333 |

解释：

- 该 smoke 证明 adapter 已能真实调用 OpenCOOD late model，并完成 leader/group 推理、world 坐标 area 裁剪、RSU global late fusion 和 AP 统计。
- 该结果只覆盖 1 帧 2 个 area，不能作为论文数值；下一步应扩大到 Top-30 的 1 帧完整 area，再扩大到 3 帧 / 11 帧。

## 2026-07-18 Top-30 One-Frame Run

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_1f
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top30_1f --max-frames 1 --fusion-method late
```

结果：

| Metric | Value |
| --- | ---: |
| Frames | 1 |
| Assignment rows | 30 |
| Cached group inference calls | 23 |
| Leader local pred / GT boxes | 38 / 35 |
| RSU fused pred / GT boxes | 35 / 35 |
| AP@0.3 | 0.606851 |
| AP@0.5 | 0.606851 |
| AP@0.7 | 0.517668 |

解释：

- 这是首个完整 Top-30 area budget 的 box-level hierarchy run。
- 30 个 area 中有重复 leader/group，因此缓存后只需 23 次 OpenCOOD inference。
- 结果仍只有单帧，主要用于确认 Top-30 local-to-global adapter 可运行；下一步应扩大到 3 帧和 11 帧，并与 flat selective-sharing baselines 对齐。

## 2026-07-18 Top-30 Three-Frame Run

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_3f
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top30_3f --max-frames 3 --fusion-method late
```

结果：

| Metric | Value |
| --- | ---: |
| Frames | 3 |
| Assignment rows | 90 |
| Cached group inference calls | 68 |
| Mean planned areas / frame | 30.000000 |
| Mean RSU fused pred boxes / frame | 35.666667 |
| Mean RSU fused GT boxes / frame | 35.666667 |
| AP@0.3 | 0.584564 |
| AP@0.5 | 0.584564 |
| AP@0.7 | 0.508387 |
| GT total | 107 |

Per-frame summary:

| Timestamp | Groups | Local pred / GT | RSU pred / GT |
| --- | ---: | ---: | ---: |
| `000060` | 23 | 38 / 35 | 35 / 35 |
| `000062` | 23 | 38 / 35 | 36 / 35 |
| `000064` | 22 | 36 / 37 | 36 / 37 |

解释：

- 3 帧 Top-30 run 说明 box-level hierarchy adapter 在连续帧上可稳定运行。
- AP@0.5 从单帧的 `0.606851` 到 3 帧的 `0.584564`，未出现明显链路崩溃。
- 下一步应扩大到 11 帧，并将结果与 existing full / confidence_topk / comm_aware_topk / area_aware_union subset ablation 做同帧口径对齐。

## 2026-07-18 Top-30 Eleven-Frame Run

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_11f
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top30_11f --max-frames 11 --fusion-method late
```

结果：

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Assignment rows | 330 |
| Cached group inference calls | 245 |
| Mean planned areas / frame | 30.000000 |
| Mean RSU fused pred boxes / frame | 34.909091 |
| Mean RSU fused GT boxes / frame | 37.090909 |
| AP@0.3 | 0.602748 |
| AP@0.5 | 0.602748 |
| AP@0.7 | 0.506345 |
| GT total | 408 |
| Pred samples | 384 |

逐帧 `unique_group_inference_calls` 为 `21-23`，说明 Top-30 area rows 中存在稳定的 leader/group 复用。

解释：

- 这是当前本地 `lgcp_carla` dump 上第一个完整 11 帧 box-level hierarchy result。
- 该结果可作为 local-to-global hierarchy ablation 的模型调用版本，但仍属于 box-level late fusion，不是 neural feature slicing。
- 下一步应把它与 `lgcp_subset_ablation_eval.py` 已有 11 帧 full / confidence_topk / comm_aware_topk / area_aware_union 结果做同表对齐，并补充 byte proxy：Top-30 raw member upload `59.42 KB/frame`，scheduled latency proxy `50 ms/frame`。

## 2026-07-18 PointPillar Feature Geometry Probe

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_probe_area23_1f5a
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_feature_probe --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_probe_area23_1f5a --fusion-method intermediate_attentive --max-frames 1 --max-areas-per-frame 5 --grid-size-x 10 --grid-size-y 6
```

已确认 PointPillar intermediate attentive checkpoint 的特征几何：

| Tensor | Shape | Interpretation |
| --- | --- | --- |
| `model.scatter` output `spatial_features` | `N x 64 x 200 x 704` | per-CAV scatter BEV tensor，`N` 等于 group size |
| `model.backbone` output `spatial_features_2d` | `1 x 384 x 100 x 352` | attentive fusion 后的 ego/leader BEV tensor |

5 个首帧 area cell 均可映射到 leader-local lidar range 内；在 `10m x 6m` area cell、`0.4m` voxel、stride-2 fused BEV 下，fused feature slice 约覆盖 `126-225` cells。未压缩 float32 byte 估计如下：

| Area | Group size | Scatter slice bytes for group | Fused slice bytes |
| --- | ---: | ---: | ---: |
| `12_9` | 2 | 430592 | 345600 |
| `13_2` | 2 | 384000 | 299520 |
| `13_0` | 1 | 117504 | 193536 |
| `11_7` | 2 | 399360 | 344064 |
| `13_3` | 1 | 192000 | 299520 |

解释：

- 该 probe 已把 LGCP world-coordinate area cell 映射到 PointPillar leader-local BEV feature index range。
- 当前 byte 只是未压缩 float32 tensor 上界，不应用作最终通信成本；后续需要量化、稀疏化或选择更早/更轻的 feature representation。
- 下一步应把该映射接入可选 feature crop adapter：先导出 per-area tensor slice manifest，再实现 leader-local crop fusion 和 RSU global aggregation。
