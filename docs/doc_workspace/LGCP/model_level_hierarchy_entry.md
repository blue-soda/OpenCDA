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

## 2026-07-18 PointPillar Feature Slice Export Smoke

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_slice_export_area23_1f5a
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_feature_slice_export --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_slice_export_area23_1f5a --fusion-method intermediate_attentive --max-frames 1 --max-areas-per-frame 5 --grid-size-x 10 --grid-size-y 6 --slice-level both --dtype float16
```

输出：

| File | Meaning |
| --- | --- |
| `feature_slice_manifest.csv` | 每个 `(timestamp, area_id, leader)` 的 feature crop shape、bounds、byte count 和 `.npz` 路径 |
| `feature_slice_summary.csv` | 汇总压缩 / 未压缩 feature bytes |
| `slices/*.npz` | 真实裁剪后的 `scatter` 和 `fused` tensor slice，附带 bounds 与 group metadata |

结果：

| Metric | Value |
| --- | ---: |
| Rows | 5 |
| Scatter slice shapes | `1-2 x 64 x 26-30 x 17-30` |
| Fused slice shapes | `1 x 384 x 14-15 x 9-16` |
| Uncompressed float16 bytes | 1502848 |
| Compressed `.npz` bytes | 178855 |
| Mean compressed bytes / area | 35771 |

扩展到 Top-23 完整首帧：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_slice_export_area23_1f
```

| Metric | Value |
| --- | ---: |
| Rows | 23 |
| Slice files | 23 |
| Uncompressed float16 bytes | 6183680 |
| Compressed `.npz` bytes | 810688 |
| Mean compressed bytes / area | 35247.304348 |

解释：

- 这是第一个真实保存 neural feature crop 的 LGCP smoke，不再只是坐标映射。
- 当前导出的 `scatter` slice 是 per-CAV/group tensor，`fused` slice 是 attentive backbone 已融合后的 leader tensor；二者都保存是为了比较后续 local fusion 的切入层。
- 该 smoke 仍未实现 leader-local feature fusion 或 RSU global aggregation，因此不能作为最终 model-level AP 结果。

## 2026-07-18 Leader-Local Feature Fusion Smoke

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f
```

命令：

```powershell
python -m opencda.tools.lgcp_pointpillar_leader_feature_fusion --slice-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_slice_export_area23_1f --feature-slice-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_slice_export_area23_1f\feature_slice_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --fusion-methods mean,max --dtype float16 --keep-model-fused
```

输出：

| File | Meaning |
| --- | --- |
| `leader_feature_manifest.csv` | 每个 area leader-local feature slice 的 shape、source slice、fusion method 和 byte count |
| `leader_feature_summary.csv` | 汇总 leader-local fused feature bytes |
| `leader_slices/*.npz` | `leader_scatter_mean`、`leader_scatter_max` 和可选 `model_fused_reference` |

结果：

| Metric | Value |
| --- | ---: |
| Rows | 23 |
| Fusion methods | `mean,max` |
| Dtype | `float16` |
| Leader scatter shape examples | `1 x 64 x 22-30 x 16-30` |
| Model fused reference shape examples | `1 x 384 x 11-16 x 9-16` |
| Uncompressed bytes | 7189760 |
| Compressed `.npz` bytes | 936298 |
| Mean compressed bytes / area | 40708.608696 |

解释：

- 这是第一个 leader-local neural feature fusion smoke：group 内 per-CAV `scatter` slices 已被融合为 leader-side feature slice。
- 当前 `mean/max` 是机制占位的 deterministic fusion，不等价于训练好的 OpenCOOD attentive fusion；`model_fused_reference` 用于后续比较或作为 teacher/reference。
- 下一步应实现 RSU global feature assembly，把多个 leader-local area slices 放回全局/RSU feature canvas，并定义跨 area overlap 的融合规则。

## 2026-07-18 RSU Feature Assembly Smoke

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f
```

命令：

```powershell
python -m opencda.tools.lgcp_pointpillar_rsu_feature_assembly --leader-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --leader-feature-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f\leader_feature_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f --feature-key leader_scatter_mean --canvas-height 200 --canvas-width 704 --channels 64 --dtype float16
```

输出：

| File | Meaning |
| --- | --- |
| `rsu_feature_frame_manifest.csv` | 每帧 RSU canvas 的输入 area 数、coverage、overlap 和 byte count |
| `rsu_feature_summary.csv` | RSU assembly 汇总 |
| `rsu_frames/*.npz` | `rsu_canvas` 与 `coverage_count` |

结果：

| Metric | Value |
| --- | ---: |
| Frames | 1 |
| Input / used rows | 23 / 23 |
| Canvas shape | `1 x 64 x 200 x 704` |
| Covered cells | 4669 |
| Coverage ratio | 0.033161 |
| Overlap cells | 2835 |
| Max overlap | 16 |
| Compressed `.npz` bytes | 82974 |

解释：

- 这是第一个 RSU-side neural feature assembly smoke：多个 leader-local area slices 已能放回统一 PointPillar scatter canvas。
- 当前 overlap 使用简单 average，不是训练后的 RSU aggregation module。
- 当前 assembly 仍是 index-space smoke：各 leader slice 来自不同 leader-local coordinate frame，尚未做 world / RSU 坐标系重投影，因此只能说明 model-level data path，不能报告最终 AP。

## 2026-07-18 RSU Detection Head Probe

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_rsu_head_probe_area23_1f
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_head_probe --rsu-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f --rsu-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f\rsu_feature_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_head_probe_area23_1f --fusion-method intermediate_attentive --top-k 20
```

结果：

| Metric | Value |
| --- | ---: |
| Input canvas | `1 x 64 x 200 x 704` |
| Backbone output | `1 x 384 x 100 x 352` |
| `psm` shape | `1 x 2 x 100 x 352` |
| `rm` shape | `1 x 14 x 100 x 352` |
| Score max / mean | `0.220411 / 0.002679` |
| Postprocess threshold | `0.2` |
| Postprocess pred boxes | 2 |

解释：

- 该 probe 证明 assembled RSU feature canvas 可以技术上接回 PointPillar backbone、classification/regression heads 和 voxel postprocess。
- 这仍不是有效 AP：当前 RSU canvas 尚未做跨 leader 坐标统一，postprocess 只验证接口可运行和输出张量维度。
- 下一步若要形成论文级 model-level AP，必须把 area slices 重投影到统一 world/RSU coordinate canvas，或把论文结果限制为 feature-level path / byte / coverage proxy。

## 2026-07-18 Reference-Frame Alignment Diagnostic

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_reference_aligned_head_probe_area23_1f_ref1
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_reference_aligned_assembly --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --leader-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --leader-feature-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f\leader_feature_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1 --reference-cav-id 1 --feature-key leader_scatter_mean --grid-size-x 10 --grid-size-y 6 --dtype float16
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_head_probe --rsu-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1 --rsu-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1\reference_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_head_probe_area23_1f_ref1 --fusion-method intermediate_attentive --top-k 20 --frame-file-column reference_frame_file --canvas-key reference_canvas
```

结果：

| Metric | Value |
| --- | ---: |
| Reference CAV | 1 |
| Input / used leader slices | 23 / 23 |
| Reference canvas | `1 x 64 x 200 x 704` |
| Coverage cells / ratio | `9189 / 0.065263` |
| Overlap cells / max overlap | `293 / 3` |
| Mean / max abs yaw delta | `93.412838 / 175.817131 deg` |
| Mean resize area ratio | 0.637908 |
| Head `psm` / `rm` | `1 x 2 x 100 x 352` / `1 x 14 x 100 x 352` |
| Head score max / mean | `0.867036 / 0.003301` |
| Postprocess pred boxes | 18 |

解释：

- 将 world-coordinate area cells 映射到统一 reference CAV frame 后，coverage ratio 从 index-space assembly 的 `0.033161` 提高到 `0.065263`，head/postprocess 也出现更强响应。
- 但 leader 与 reference 的 yaw 差很大，平均约 `93.41 deg`，最大约 `175.82 deg`。当前实现只做 nearest resize，没有 feature rotation / affine warp / learned alignment。
- 因此这仍是 alignment diagnostic，不是 AP 结果。论文安全口径应是：model-level data path is feasible, while valid AP requires coordinate-aware feature warping or retrained aggregation.

## 2026-07-18 Coordinate-Warp Feature Assembly Smoke

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_head_probe_area23_1f_ref1
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_coordinate_warp_assembly --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --leader-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --leader-feature-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f\leader_feature_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1 --reference-cav-id 1 --feature-key leader_scatter_mean --grid-size-x 10 --grid-size-y 6 --dtype float16
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_head_probe --rsu-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1 --rsu-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1\coordinate_warp_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_head_probe_area23_1f_ref1 --fusion-method intermediate_attentive --top-k 20 --frame-file-column warped_frame_file --canvas-key warped_canvas
```

方法：

- 对 reference canvas 内的每个 target cell，计算 cell center 的 reference-local 坐标。
- 使用 reference pose 映射到 world coordinate。
- 再使用 leader pose 映射回 leader-local coordinate。
- 在 leader-local feature slice 内做 nearest-neighbor sampling，并放回 reference canvas。

结果：

| Metric | Value |
| --- | ---: |
| Reference CAV | 1 |
| Input / used leader slices | 23 / 23 |
| Target / sampled cells | `8550 / 8550` |
| Sample ratio | 1.000000 |
| Coverage cells / ratio | `8550 / 0.060724` |
| Overlap cells / max overlap | `0 / 1` |
| Mean / max abs yaw delta | `93.412838 / 175.817131 deg` |
| Head `psm` / `rm` | `1 x 2 x 100 x 352` / `1 x 14 x 100 x 352` |
| Head score max / mean | `0.893363 / 0.003926` |
| Postprocess pred boxes | 30 |

解释：

- 相比 nearest-resize diagnostic，coordinate warp 避免了 target bbox 内的重复 overlap，所有 world-area target cells 都能反查到 leader-local source slice。
- Head response 进一步增强，但当前仍是 nearest-neighbor feature sampling，没有双线性采样、feature rotation calibration 或训练后的 aggregation。
- 该结果可以作为 coordinate-aware model-level path 的 feasibility evidence；不能直接作为论文 AP，下一步才应做 GT/AP smoke 或定义 feature-level proxy。

## 2026-07-18 Coordinate-Warp AP Probe

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_ap_probe_area23_1f_ref1
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_warp_ap_probe --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --warped-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1 --warped-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1\coordinate_warp_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_ap_probe_area23_1f_ref1 --reference-cav-id 1 --fusion-method intermediate_attentive --frame-file-column warped_frame_file --canvas-key warped_canvas
```

结果：

| Metric | Value |
| --- | ---: |
| Frames | 1 |
| Pred boxes | 30 |
| GT boxes | 16 |
| AP@0.3 | 0.010000 |
| AP@0.5 | 0.010000 |
| AP@0.7 | 0.000000 |

解释：

- 该 probe 首次闭合了 coordinate-warp canvas -> PointPillar head/postprocess -> reference-frame GT/AP 的评价链路。
- 结果很低，说明当前 nearest-neighbor coordinate warp 虽然技术可运行，但不能支撑论文级 model-level detection claim。
- 短期论文安全选择是将 neural feature hierarchy 写作限制为 data-path feasibility / feature coverage / byte proxy；若要报告 AP，需要 bilinear/affine warp、feature normalization/calibration，甚至重新训练 aggregation head。
