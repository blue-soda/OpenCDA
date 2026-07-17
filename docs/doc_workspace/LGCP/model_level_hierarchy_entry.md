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

下一步最小任务应是实现 Phase A 的 `lgcp_hierarchy_late_fusion_eval.py`，复用 SGCP late-fusion 路径，先得到可评估 AP 的 local-to-global hierarchy ablation。
