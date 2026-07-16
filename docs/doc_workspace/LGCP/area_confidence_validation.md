# Area Confidence Validation Design

本文档对应 `target.md` 的 P0 任务：设计 area confidence 验证实验，统计 confidence 与 area-level AP / recall 的相关性。

## 目标

审稿人质疑 LGCP 将 area confidence 作为优化目标和 group selection 依据，但论文没有证明该值确实能代表局部感知质量。该实验要回答三个问题：

1. 单车 / RSU 对某个 area 的 confidence 是否与该 area 的 detection recall / AP 正相关。
2. Eq. (2) 的多 CAV confidence composition 是否比简单规则更能预测 fused perception quality。
3. confidence 作为调度依据时，是否能在相同通信预算下优先覆盖低质量或高价值 area。

## 当前代码基础

- 场景配置：`opencda/scenario_testing/config_yaml/lgcp_carla.yaml`
  - `lgcp.roi.center`
  - `lgcp.roi.size`
  - `lgcp.roi.grid_size`
  - `lgcp.area_confidence.delta_g`
- 在线导出：`opencda/scenario_testing/lgcp_carla.py --dump`
  - 已能导出 `D:\Data\Carla\<scenario_id>\<agent_id>\<timestamp>.yaml/.pcd`
  - RSU 目录为 `-1`
  - CAV 目录为 `1` 到 `20`
- 离线数据读取：`opencda/core/common/offline_dataset.py`
  - `OPV2VFrameDataset.load_frame()` 可加载每帧多 agent 的 YAML、PCD 和坐标变换。
- 离线推理：`opencda/tools/offline_inference.py`
  - 当前可得到 `pred_box_tensor`、`pred_score`、`gt_box_tensor`。
- AP 统计：`opencda/core/ml_libs/opencood_manager.py`
  - 使用 OpenCOOD `eval_utils.calculate_tp_fp()` 和 `calculate_ap()`。
- grid confidence 参考：`opencda/core/clustering/utils/common.py`
  - 当前 `density_score(density, rho_th)` 采用点云密度归一化。
  - `avg_grids_score(vid, grid_set)` 可作为已有 grid-score 风格的参考实现。

## 数据单元

建议以 `(scenario_id, timestamp, area_id, agent_id)` 为最小记录单元，导出 CSV / JSONL：

| 字段 | 含义 |
| --- | --- |
| `scenario_id` | 数据导出场景目录 |
| `timestamp` | 帧编号 |
| `area_id` | ROI 网格或合并后的 area 编号 |
| `agent_id` | CAV / RSU id，RSU 为 `-1` |
| `agent_pose` | agent LiDAR pose |
| `area_center` | area 中心坐标 |
| `distance` | agent 到 area 中心距离 |
| `point_count` | 该 agent 在 area 内 LiDAR 点数 |
| `density` | `point_count / area_size` |
| `confidence` | 单 agent area confidence |
| `pred_count` | 落在 area 内的预测框数量 |
| `gt_count` | 落在 area 内的 GT 框数量 |
| `tp_03/tp_05/tp_07` | area 内不同 IoU 阈值 TP 数 |
| `fp_03/fp_05/fp_07` | area 内不同 IoU 阈值 FP 数 |
| `recall_03/05/07` | area recall |
| `ap_03/05/07` | area AP，样本少时可使用 accumulated AP |

## Area 划分

第一版直接沿用 `lgcp_carla.yaml`：

- ROI center：`[0.0, 0.0]`
- ROI size：`[280.0, 80.0]`
- grid size：`[10.0, 6.0]`

`area_id` 建议使用 `ix_iy` 字符串。后续如果论文中的 area 比 grid 更粗，可把多个 grid 聚合为一个 area，但第一版应保持 grid-level，以便与 LiDAR density 和现有 cluster grid 字段对齐。

## Confidence 定义

第一版不急着改变论文公式，先同时导出几组候选 confidence：

| 名称 | 定义 | 用途 |
| --- | --- | --- |
| `density_linear` | `min(density / rho_th, 1.0)` | 对齐当前 `density_score()` |
| `distance_decay` | `exp(-distance / tau)` | 检查距离先验是否足够 |
| `density_distance` | `density_linear * distance_decay` | 论文式感知质量近似 |
| `score_mean` | area 内预测框 confidence 均值 | 检查 detector score 是否更相关 |
| `score_topk` | area 内 top-k score 均值 | 降低低分框噪声 |

多 CAV composition 对应下一项 P0 Eq. (2) 验证，但本实验应预留输出：

- `compose_product`
- `compose_max`
- `compose_mean`
- `compose_sum_clipped`
- `compose_topk_mean`

## Area-Level AP / Recall 计算

推荐第一版使用 box center 归属 area：

1. 将 `pred_box_tensor` 和 `gt_box_tensor` 从 ego 坐标系转到世界坐标，或统一在 ego 坐标下把 area 网格也投影过去。
2. 用 8 个 corner 的 XY 均值作为 box center。
3. 按 box center 落入的 `area_id` 划分预测框和 GT。
4. 对每个 area 分别调用 OpenCOOD 的 TP / FP 逻辑，或复用同样 IoU 函数实现 area 内匹配。
5. 对稀疏 area 同时记录 `gt_count`，避免用无 GT area 的 AP 噪声误导相关性。

统计时分两类：

- **Recall correlation**：只保留 `gt_count > 0` 的 area。
- **AP correlation**：按多个 timestamp 累积同一 area 的 TP / FP / score，再计算 AP。

## 相关性指标

每个 candidate confidence 都计算：

- Pearson correlation：衡量近似线性关系。
- Spearman correlation：衡量排序关系，更贴近调度选择。
- Kendall tau：样本较少时作为排序稳健性补充。
- Calibration bins：按 confidence 分桶，画出 confidence vs mean recall/AP。

论文中优先报告 Spearman 和 calibration curve，因为 LGCP 主要依赖 area ranking 和 group selection。

## 实验矩阵

第一阶段 smoke / design validation：

| 设置 | 数据 | 目标 |
| --- | --- | --- |
| S1 | `lgcp_carla` 1 个 scenario，20 CAV + RSU，约 50-100 帧 | 验证导出字段、相关性脚本、area AP 计算正确 |
| S2 | 同一 scenario，不同 ego / timestamps | 检查稳定性 |
| S3 | 只用 CAV、只用 RSU、CAV+RSU | 比较不同 sensor provider 的 confidence 相关性 |

论文补强阶段：

| 设置 | 数据 | 目标 |
| --- | --- | --- |
| P1 | 3-5 个随机种子 / traffic seeds | 报告主要相关性结果 |
| P2 | 不同 grid / area size | 与 sensitivity analysis 合并 |
| P3 | 不同模型：late / early / intermediate | 检查 confidence 是否模型无关 |

## 输出文件建议

统一放在：

```text
docs/doc_workspace/LGCP/experiments/area_confidence/
```

建议结构：

```text
area_confidence/
  20260715_lgcp_carla_smoke/
    config.yaml
    area_records.csv
    correlation_summary.csv
    calibration_bins.csv
    notes.md
```

后续可进入论文的结果再同步到 `results.md`。

## 实现计划

1. 新增离线脚本 `opencda/tools/lgcp_area_confidence_eval.py`。
2. 复用 `OPV2VFrameDataset` 逐帧加载已导出的 OPV2V-style 数据。
3. 对指定 agent 或 agent subset 单独运行 OpenCOOD inference，得到 per-agent / subset pred boxes。
4. 从 PCD 统计每个 agent 在每个 area 的点数和 density。
5. 按 area 切分 pred / GT，计算 recall、AP 和候选 confidence。
6. 写出 `area_records.csv` 和 `correlation_summary.csv`。
7. 在 `results.md` 中登记 smoke result，再决定是否扩大数据规模。

## 判定标准

可认为该 P0 实验补强有效的最低标准：

- 至少 50 个 `gt_count > 0` 的 area-frame 样本。
- 最佳 candidate confidence 与 area recall 的 Spearman 相关性为正，且显著优于随机排序。
- calibration bins 呈总体单调趋势。
- Eq. (2) product rule 若不优于 max/mean/top-k，需要在论文中替换或解释适用条件。

## 风险

- 单帧 area AP 样本过少，可能需要跨 frame 累积。
- 当前离线推理默认以 ego 为参考坐标，area 切分必须确保坐标系一致。
- RSU 作为 `-1` agent 可以导出数据，但 OpenCOOD loader 是否应把 RSU 当作普通 CAV sensor provider，需要在实现时单独验证。
- `density_score()` 只反映 LiDAR 点云密度，不一定等价于 detector quality；若相关性弱，需要引入 detector score 或 learned calibration。

## 2026-07-16 Grid Size Sensitivity Smoke

`opencda/tools/lgcp_area_confidence_eval.py` 已支持通过 `--grid-size-x` / `--grid-size-y` 在离线评估中覆盖 ROI grid size。

当前完成三组单场景 11 帧 smoke：

| Grid size | Area-frame noisy-or vs recall@0.5 Spearman | Area-acc noisy-or vs AP@0.5 Spearman |
| --- | ---: | ---: |
| `5m x 3m` | 0.475952 | 0.213836 |
| `10m x 6m` | 0.570407 | 0.411840 |
| `20m x 12m` | 0.458975 | 0.233766 |

该结果支持 `10m x 6m` 作为当前默认 grid，但仍需多 seed / 多场景复核。
