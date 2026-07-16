# LGCP 核心实验结果记录

本文档只记录已经复核、可复现、可解释，并可能进入论文正文、附录或 rebuttal 的核心结果。探索性过程和失败尝试请先记录到 `log.md`。

## 结果记录原则

- 每个结果必须能追溯到 `log.md` 中的一次或多次实验记录。
- 每个表格或图都应记录数据集、模型、场景、关键参数和日志 / 产物路径。
- 对论文主张有直接支撑的结果优先沉淀。
- 不确定或一次性观察不要写成结论。

## R1：Area Confidence 有效性

### 2026-07-15 Smoke：Area Confidence 导出链路

目标：验证 LGCP 的 area confidence 是否能作为 area-level AP / recall 的可靠代理。

本次结果只验证 ROI/grid/agent-area record 导出链路，不作为论文中 confidence-vs-AP 的有效性证据。

配置：

- 脚本：`opencda/tools/lgcp_area_confidence_eval.py`
- 数据目录：`D:\Data\Carla\2026_07_15_02_33_21`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_smoke`
- 帧：`000060`, `000062`, `000064`

输出：

| 指标 | 数值 |
| --- | --- |
| area-agent records | 5836 |
| summary rows | 3 |
| 每帧 ROI 内 GT objects | 14 |

Density 与 GT count 的探索性相关性：

| Timestamp | Rows | GT areas | GT objects | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| 000060 | 1945 | 12 | 14 | -0.028869 | -0.051536 |
| 000062 | 1949 | 12 | 14 | -0.029360 | -0.072819 |
| 000064 | 1942 | 12 | 14 | -0.028361 | -0.064157 |

解释：

- 该 smoke test 说明 `lgcp_carla` dump 可以被稳定切成 area-level records。
- 仅用 LiDAR density 与 GT object count 的相关性接近 0，不能替代审稿人要求的 confidence-vs-area AP / recall 验证。
- 下一步必须接入 detector prediction slicing，计算真正的 area-level recall / AP。

### 待补充：Area-Level AP / Recall

### 2026-07-15 Smoke：Prediction Slicing 与 Area-Level Recall

本次结果验证 OpenCOOD prediction / GT box 可以按 LGCP area 切分，并计算 per-frame area-level recall / precision。它仍然不是最终论文证据，因为尚未累计 area AP，也尚未统计 confidence-vs-AP/recall 相关性。

配置：

- 脚本：`opencda/tools/lgcp_area_confidence_eval.py --with-inference`
- 数据目录：`D:\Data\Carla\2026_07_15_02_33_21`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_inference_smoke_3f`
- 模型：`pointpillar_early_fusion`
- 帧：`000060`, `000062`, `000064`

| Timestamp | Area-agent rows | Area quality rows | GT boxes in sliced areas | Pred boxes in sliced areas | Recall@0.5 |
| --- | --- | --- | --- | --- | --- |
| 000060 | 1945 | 33 | 48 | 40 | 0.791667 |
| 000062 | 1949 | 32 | 47 | 42 | 0.851064 |
| 000064 | 1942 | 32 | 47 | 40 | 0.829787 |

总计：

| 项目 | 数量 |
| --- | --- |
| area-agent records | 5836 |
| area quality rows | 97 |

结论：

- OpenCOOD 输出的 `pred_box_tensor / pred_score / gt_box_tensor` 已能按 LGCP ROI/grid 进行 area slicing。
- 当前脚本可输出每个 area 的 `pred_count`、`gt_count`、`tp/fp`、`recall`、`precision`。
- 下一步需要把实验扩大到更多帧 / 多 seed，并补充 detector-score confidence。

### 2026-07-15 Smoke：Area AP 与 Confidence-Quality 相关性

本次结果验证脚本已经能累计 area-level AP，并输出 confidence-vs-recall / confidence-vs-AP 相关性。样本仍只有 3 帧，因此只能作为 smoke signal，不可作为最终论文结论。

配置：

- 脚本：`opencda/tools/lgcp_area_confidence_eval.py --with-inference`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_smoke_3f`
- 帧：`000060`, `000062`, `000064`

输出：

| 文件 | 行数 | 含义 |
| --- | --- | --- |
| `area_records.csv` | 5836 | per-agent area confidence records |
| `area_quality.csv` | 97 | per-frame area recall / precision |
| `area_ap_summary.csv` | 34 | accumulated area AP summary |
| `confidence_quality_records.csv` | 97 | joined confidence-quality records |
| `confidence_quality_correlation.csv` | 30 | correlation summary |

代表性相关性：

| Scope | Confidence | Quality | Samples | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| area_frame | confidence_max | recall_05 | 97 | 0.424462 | 0.647552 |
| area_frame | confidence_noisy_or | recall_05 | 97 | 0.385940 | 0.647552 |
| area_frame | density_distance_mean | recall_05 | 97 | 0.323188 | 0.272039 |
| area_accumulated | confidence_max | ap_05 | 33 | 0.430768 | 0.524064 |
| area_accumulated | confidence_noisy_or | ap_05 | 33 | 0.371881 | 0.524064 |
| area_accumulated | density_distance_mean | ap_05 | 33 | 0.419267 | 0.448864 |

解释：

- 3 帧 smoke 中，density-based confidence 与 area-level recall/AP 出现正相关信号。
- `confidence_max` 和 `confidence_noisy_or` 在排序相关性上较强，但两者 Spearman 相同，说明当前 density confidence 已接近饱和或排序粒度不足。
- 下一步应扩大样本，并加入 detector score / feature confidence，避免只验证点云密度 proxy。

### 2026-07-15：11 帧 Area Confidence Validation + Detector Score 对照

本次结果使用完整 `lgcp_carla` dump 的 11 帧，补充 detector-score confidence 作为对照。它比 3 帧 smoke 更稳定，但仍然只来自单个场景 / 单个 traffic seed，暂不作为最终论文结论。

配置：

- 脚本：`opencda/tools/lgcp_area_confidence_eval.py --with-inference`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_11f_detector_score`
- 帧：`000060` 到 `000080`，共 11 帧
- 模型：`pointpillar_early_fusion`

输出：

| 文件 | 行数 | 含义 |
| --- | --- | --- |
| `area_records.csv` | 21418 | per-agent area confidence records |
| `area_quality.csv` | 363 | per-frame area recall / precision |
| `area_ap_summary.csv` | 40 | accumulated area AP summary |
| `confidence_quality_records.csv` | 363 | joined confidence-quality records |
| `confidence_quality_correlation.csv` | 54 | correlation summary |

代表性相关性：

| Scope | Confidence | Quality | Samples | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| area_frame | confidence_max | recall_05 | 354 | 0.242796 | 0.570690 |
| area_frame | confidence_noisy_or | recall_05 | 354 | 0.229353 | 0.570407 |
| area_accumulated | confidence_max | ap_05 | 36 | 0.401529 | 0.411840 |
| area_accumulated | confidence_noisy_or | ap_05 | 36 | 0.380141 | 0.411840 |
| area_accumulated | score_mean | ap_05 | 36 | 0.299850 | 0.402059 |
| area_accumulated | score_top2_mean | ap_07 | 36 | 0.349922 | 0.401030 |

观察：

- Density-based `confidence_max` / `confidence_noisy_or` 与 area-frame Recall@0.5 的 Spearman 约为 `0.57`，排序信号比 3 帧 smoke 更稳定。
- Detector-score confidence 在逐帧 recall 上相关性较弱，但在 accumulated AP 上出现中等正相关，例如 `score_mean` vs AP@0.5 的 Spearman 为 `0.402059`。
- 当前结果支持继续推进 confidence validation，但还不足以回应审稿人要求的跨数据集 / 跨模型稳定性。

下一步：

- 扩展到多个 CARLA traffic seed / scenario dump。
- 对比 early / late / intermediate fusion。
- 若 detector score 仍弱，进一步导出模型 feature confidence 或 calibrated confidence。

建议记录字段：

| 数据集 | 模型 | 指标 | 相关性方法 | 相关系数 | 样本数 | 结论 | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | confidence vs AP / recall | Pearson / Spearman | TBD | TBD | TBD | TBD |

需要回答：

- 单 CAV area confidence 与真实 area-level detection quality 是否相关？
- 多 CAV 组合 confidence 与 fusion 后的 area-level detection quality 是否相关？
- Eq. (2) 的 product rule 是否优于 max / mean / sum / top-k？

## R2：Group Selection Optimality Gap

### 2026-07-15 Smoke：Group-Member Selection Exhaustive Gap

目标：在小规模场景中比较 greedy group selection 与 exhaustive search / ILP / DP 的差距。

本次结果只覆盖 group member selection，即论文 selection algorithm 中基于 `Delta_g` 构造每个 area CAV group 的第一阶段。Leader assignment / load balancing 尚未纳入。

配置：

- 脚本：`opencda/tools/lgcp_greedy_gap_eval.py`
- 输入：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_11f_detector_score`
- 输出目录：`docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260715_lgcp_carla_greedy_gap_density_distance`
- Confidence field：`density_distance`
- 每个 instance：最多 6 agents、5 areas、group size <= 4
- Instance 数：11
- Delta_g：`0.05`, `0.075`, `0.1`, `0.125`

结果：

| Objective | Delta_g | Mean relative gap | P90 relative gap | Max relative gap |
| --- | --- | --- | --- | --- |
| O1 confidence only | 0.05 | 0.049030 | 0.052639 | 0.053657 |
| O1 confidence only | 0.075 | 0.063452 | 0.066516 | 0.066665 |
| O1 confidence only | 0.1 | 0.068748 | 0.084891 | 0.093525 |
| O1 confidence only | 0.125 | 0.109864 | 0.117836 | 0.117895 |
| O2 confidence minus size | 0.05 | 0.034831 | 0.038233 | 0.039755 |
| O2 confidence minus size | 0.075 | 0.047666 | 0.051154 | 0.051216 |
| O2 confidence minus size | 0.1 | 0.052629 | 0.068291 | 0.075165 |
| O2 confidence minus size | 0.125 | 0.092049 | 0.100414 | 0.100445 |

对照：

- 使用 `density_linear` 时 gap 全为 0，说明该 confidence 在当前场景中较容易饱和，不能充分暴露 greedy 与 oracle 差异。
- 使用 `density_distance` 后，`Delta_g` 越大，greedy gap 越明显。

解释：

- 该结果可以支持 rebuttal 中 “we do not claim theoretical approximation guarantee; we provide empirical optimality-gap evidence” 的写法。
- 目前还不能覆盖 leader assignment / load balancing，因此不能说完整 selection algorithm 已做 optimality gap。

### 待补充：Leader Assignment / Load Balancing Gap

### 2026-07-15 Smoke：Leader Assignment / Load Balancing Gap

本次结果在 greedy-selected groups 上比较论文中的 leader greedy assignment 与 exhaustive min-max load oracle，覆盖 selection algorithm 的第二阶段。

配置：

- 脚本：`opencda/tools/lgcp_greedy_gap_eval.py`
- 输入：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_11f_detector_score`
- 输出目录：`docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260715_lgcp_carla_greedy_gap_with_leader`
- Confidence field：`density_distance`
- 每个 instance：最多 6 agents、5 areas、group size <= 4
- Instance 数：11

Group-member selection gap 与上一节一致；leader-load gap 如下：

| Delta_g | Mean relative gap | Median relative gap | P90 relative gap | Max relative gap | Mean absolute gap |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.128788 | 0.000000 | 0.250000 | 0.666667 | 0.454545 |
| 0.075 | 0.022727 | 0.000000 | 0.000000 | 0.250000 | 0.090909 |
| 0.1 | 0.022727 | 0.000000 | 0.000000 | 0.250000 | 0.090909 |
| 0.125 | 0.022727 | 0.000000 | 0.000000 | 0.250000 | 0.090909 |

解释：

- 在较小 `Delta_g=0.05` 时 group 更大 / 任务更多，leader greedy 的 max-load gap 更明显。
- 当 `Delta_g >= 0.075` 时，多数 instance 的 leader assignment 与 min-max oracle 一致，只有少数 instance 出现 25% relative gap。
- 这可以支持论文中将 selection algorithm 表述为 efficient heuristic，并用 empirical gap 而非 theoretical approximation guarantee 进行解释。

下一步：

- 扩大到更多 seed / 更大 instance。
- 将 group-member gap 和 leader-load gap 合并到 latency-aware `O3_confidence_latency_ratio`。

建议记录字段：

| 场景规模 | CAV 数 | Area 数 | 方法 | Objective | Latency | Gap | 运行时间 | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | Greedy / Optimal | TBD | TBD | TBD | TBD | TBD |

需要回答：

- Greedy threshold rule 离最优解有多远？
- `Δ_g` 对 optimality gap 和 latency constraint violation 的影响如何？

## R3：Local-to-Global Ablation

### 2026-07-15 Design：Local-to-Global Ablation

目标：区分 LGCP 的收益来自 partial sharing，还是来自 local-to-global hierarchy。

设计文档：`docs/doc_workspace/LGCP/local_to_global_ablation.md`

最低 rebuttal 组合：

| 方法 | Area partition | Group selection | Leader fusion | RSU aggregation | Selective sharing | AP | Volume | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full sharing baseline | No | No | No | Optional | No | TBD | TBD | TBD |
| Confidence selective sharing | No | No | No | Optional | Yes | TBD | TBD | TBD |
| LGCP without scheduling | Yes | Yes | Yes | Yes | Yes | TBD | TBD | TBD |
| Full LGCP | Yes | Yes | Yes | Yes | Yes | TBD | TBD | TBD |

关键原则：

- 必须加入 confidence selective sharing without hierarchy，不能只对比 full sharing。
- 主文建议使用 same packet budget，对比 AP / recall / latency。
- 第一阶段先做 offline perception-only ablation；第二阶段再实现真正 leader local fusion、RSU aggregation 和 scheduling。

### 2026-07-15 Smoke：Offline Subset Ablation

本次结果用于第一阶段验证 selective sharing 与 area-aware subset selection 的差异。它是 perception-only offline smoke，不包含真实 leader local fusion、RSU aggregation 或 NS3 scheduling，因此不能替代完整 local-to-global hierarchy ablation。

配置：

- 脚本：`opencda/tools/lgcp_subset_ablation_eval.py`
- 数据目录：`D:\Data\Carla\2026_07_15_02_33_21`
- Area records：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_11f_detector_score/area_records.csv`
- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_offline_subset_smoke`
- 帧数：`3`
- Budgets：`5`, `10`
- Confidence field：`density_distance`

结果：

| Method | Budget | Frames | AP@0.3 | AP@0.5 | AP@0.7 | GT total | Pred samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full sharing | 5 | 3 | 0.851542 | 0.828838 | 0.577169 | 214 | 185 |
| Full sharing | 10 | 3 | 0.851542 | 0.828838 | 0.577169 | 214 | 185 |
| Random selective | 5 | 3 | 0.449647 | 0.433443 | 0.222553 | 167 | 102 |
| Random selective | 10 | 3 | 0.557382 | 0.550719 | 0.391526 | 201 | 121 |
| Confidence top-k | 5 | 3 | 0.349202 | 0.332366 | 0.234146 | 157 | 76 |
| Confidence top-k | 10 | 3 | 0.610004 | 0.610004 | 0.482805 | 188 | 130 |
| Area-aware union | 5 | 3 | 0.484637 | 0.484637 | 0.350075 | 167 | 111 |
| Area-aware union | 10 | 3 | 0.678564 | 0.678564 | 0.591502 | 191 | 137 |

观察：

- 在相同 budget 下，`area_aware_union` 均高于 `confidence_topk`，尤其 budget=10 时 AP@0.7 为 `0.591502`，高于 `confidence_topk` 的 `0.482805`。
- `full sharing` 仍是 AP@0.5 的上界，但 budget=10 的 `area_aware_union` 在 AP@0.7 上略高于 full sharing；由于样本只有 3 帧，这只能视为 smoke signal，不能写成稳定结论。
- 该结果支持继续推进 “area-aware selective sharing” 方向，但仍需要完整 11 帧 / 多 seed，并最终接入真实 LGCP hierarchy。

下一步：

- 扩展到多 seed。
- 记录 selected agent count、packet count、volume proxy 和 latency proxy。
- 在机制层实现 leader local fusion 与 RSU global aggregation 后，复跑完整 R3 ablation。

### 2026-07-15：Offline Subset Ablation 11 帧扩展

本次结果将上一节 3 帧 smoke 扩展到完整 `lgcp_carla` dump 的 11 帧。它仍是 perception-only offline ablation，但比 3 帧结果更适合作为 rebuttal 中的初步证据。

配置：

- 脚本：`opencda/tools/lgcp_subset_ablation_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_offline_subset_11f`
- 帧：`000060` 到 `000080`，共 `11` 帧
- Budgets：`5`, `10`
- Confidence field：`density_distance`

结果：

| Method | Budget | Frames | AP@0.3 | AP@0.5 | AP@0.7 | GT total | Pred samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full sharing | 5 | 11 | 0.852487 | 0.841146 | 0.527546 | 782 | 675 |
| Full sharing | 10 | 11 | 0.852487 | 0.841146 | 0.526237 | 782 | 675 |
| Random selective | 5 | 11 | 0.331922 | 0.314671 | 0.165521 | 586 | 287 |
| Random selective | 10 | 11 | 0.611342 | 0.598993 | 0.380538 | 730 | 482 |
| Confidence top-k | 5 | 11 | 0.358450 | 0.345556 | 0.221396 | 573 | 263 |
| Confidence top-k | 10 | 11 | 0.629379 | 0.624088 | 0.435861 | 692 | 476 |
| Area-aware union | 5 | 11 | 0.405018 | 0.396807 | 0.251957 | 573 | 321 |
| Area-aware union | 10 | 11 | 0.678388 | 0.676678 | 0.538273 | 691 | 502 |

观察：

- 在完整 11 帧上，`area_aware_union` 继续稳定高于 `confidence_topk` 和 `random selective`。
- Budget=10 时，`area_aware_union` 相比 `confidence_topk` 的 AP@0.5 提升约 `0.052590`，AP@0.7 提升约 `0.102412`。
- Full sharing 的 AP@0.5 仍最高；`area_aware_union` 在 AP@0.7 上接近 full sharing，但该现象需要多 seed 验证。

结论：

- 该结果支持 “不是任意 selective sharing 都足够，area-aware selection 具有额外收益” 的 rebuttal 方向。
- 它仍不能证明完整 LGCP hierarchy 的收益；下一步必须补充多 seed，并实现 leader local fusion / RSU aggregation 后复跑完整 ablation。

### 2026-07-15：Offline Subset Ablation Packet-Budget 统计

本次结果复跑 11 帧 ablation，并在 `ablation_summary.csv` 中加入 selected-agent、non-ego packet 和 byte-volume proxy 统计。`feature_packet_bytes=10000` 仅作为统一 byte proxy，不代表最终论文中的真实 feature slice 大小。

配置：

- 脚本：`opencda/tools/lgcp_subset_ablation_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_offline_subset_11f_budget_stats`
- 帧数：`11`
- Feature packet byte proxy：`10000`

结果：

| Method | Budget | Non-ego packets | Byte proxy | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | --- | --- |
| Full sharing | 5 | 209 | 2,090,000 | 0.839868 | 0.526419 |
| Full sharing | 10 | 209 | 2,090,000 | 0.839868 | 0.526567 |
| Random selective | 5 | 44 | 440,000 | 0.314671 | 0.165521 |
| Random selective | 10 | 99 | 990,000 | 0.598993 | 0.380556 |
| Confidence top-k | 5 | 44 | 440,000 | 0.345556 | 0.221396 |
| Confidence top-k | 10 | 99 | 990,000 | 0.624088 | 0.435861 |
| Area-aware union | 5 | 44 | 440,000 | 0.396807 | 0.251957 |
| Area-aware union | 10 | 99 | 990,000 | 0.676678 | 0.538273 |

观察：

- 相同 non-ego packet budget 下，`area_aware_union` 仍优于 random selective 和 confidence top-k。
- Budget=10 时，`area_aware_union` 用约 `47.4%` 的 full-sharing non-ego packets（99/209），AP@0.5 达到 full-sharing 的约 `80.6%`，AP@0.7 与 full-sharing 接近。
- 该统计为 rebuttal 中的 same packet budget 表述提供了可复现字段，但 byte proxy 仍需在真实 feature slice / NS3 scheduling 接入后校准。

### 2026-07-15：Hierarchy Control-Plane Plan

本次结果实现 LGCP hierarchy 的第一阶段控制面：RSU 根据 area confidence 选择高优先级 area，为每个 area 构造可重叠 CAV group，选择 leader，并导出 member-to-leader 与 leader-to-RSU 上传计划。它仍不包含真实 feature slicing、leader local fusion 或 RSU global aggregation。

配置：

- 脚本：`opencda/tools/lgcp_hierarchy_plan_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f`
- 帧数：`11`
- `delta_g=0.05`
- `max_group_size=4`
- `max_areas=40`
- `feature_packet_bytes=10000`
- `leader_result_bytes=2000`

平均结果：

| 指标 | 数值 |
| --- | --- |
| covered areas / frame | 40 |
| average group size | 1.536364 |
| average group confidence | 0.908059 |
| member-to-leader packets / frame | 21.454545 |
| leader-to-RSU packets / frame | 40 |
| total byte proxy / frame | 299105.454545 |
| active leaders / frame | 15.090909 |
| leader max load / frame | 8.818182 |

解释：

- `area_assignment_plan.csv` 显式记录每个 area 的 `group_members` 和 `leader_id`，可证明 LGCP group 是可重叠 area-task group，而不是传统 disjoint clustering。
- `upload_plan.csv` 将 member-to-leader 和 leader-to-RSU 拆成两类上传，为后续 NS3 replay 和 leader upload scheduling 提供输入。
- 当前结果是 mechanism/control-plane prototype，不能作为完整 local-to-global perception-quality ablation。

下一步：

- 将 `upload_plan.csv` 接入 offline NS3 replay。
- 实现 area-specific feature slice 或可近似的 per-area prediction fusion。
- 让 RSU aggregation 输出可评估的 global perception result。

### 2026-07-15：LGCP Upload Plan Offline NS3 Dry-Run

本次结果将 `hierarchy_plan` 产生的 `upload_plan.csv` 接入 `offline_ns3_replay.py`，用于把 LGCP member-to-leader / leader-to-RSU 两级上传转换为 NS3 `transfer_requests`。本次只做 dry-run，不连接 ns-3。

配置：

- 脚本：`opencda/tools/offline_ns3_replay.py`
- 参数：`--lgcp-upload-plan .../upload_plan.csv --dry-run`
- 数据目录：`D:\Data\Carla\2026_07_15_02_33_21`
- 帧数：`11`
- 节点：`21`（20 CAV + RSU `-1`）

结果：

| Timestamp | Requests | Bytes |
| --- | --- | --- |
| 000060 | 62 | 300000 |
| 000062 | 62 | 300000 |
| 000064 | 62 | 300000 |
| 000066 | 62 | 300000 |
| 000068 | 60 | 280000 |
| 000070 | 60 | 280000 |
| 000072 | 61 | 290000 |
| 000074 | 61 | 290000 |
| 000076 | 60 | 280000 |
| 000078 | 63 | 310000 |
| 000080 | 63 | 310000 |

解释：

- Dry-run 已验证 replay 工具能逐帧加载 20 CAV + RSU，并从 LGCP `upload_plan.csv` 构造传输请求。
- 这一步使 hierarchy control-plane 可以进入 NS3 latency / delivery 验证。
- 尚未运行真实 ns-3，因此没有 latency、delivery ratio 或 packet delay 结果。

### 2026-07-15：LGCP Upload Plan Offline NS3 3 帧 Smoke

本次结果启动 WSL ns-3，并用 `offline_ns3_replay.py --lgcp-upload-plan` 回放 3 帧 LGCP hierarchy upload plan。

配置：

- NS3：`scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true --carlaHost=auto`
- Replay：`max-frames=3`
- RSU mapping：dump `-1` -> NS3 node `21`
- 日志目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_smoke_3f_rsu21`

Replay 输出：

| Timestamp | Nodes | Requests | Bytes |
| --- | --- | --- | --- |
| 000060 | 21 | 62 | 300000 |
| 000062 | 21 | 62 | 300000 |
| 000064 | 21 | 62 | 300000 |

解析到的 `cam_received`：

| 指标 | 数值 |
| --- | --- |
| cam received count | 5 |
| RSU received count | 3 |
| average delay | 16 ms |
| max delay | 31 ms |

解释：

- 联机 smoke 已验证 LGCP upload plan 能进入 ns-3，同步完成，并且 RSU 节点能收到 leader-to-RSU packet。
- 初次尝试中 RSU 使用负数 ID `-1`，ns-3 跳过 leader-to-RSU transfer；已修复为正整数节点 `21`。
- 当前 `cam_received` summary 只统计 ns-3 回传给 OpenCDA 的成功接收事件，还不是完整 request-level delivery ratio。

下一步：

- 扩展到 11 帧。
- 解析 request-level delivery ratio，区分 `member_to_leader` 和 `leader_to_rsu`。
- 将 NS3 delay 与 hierarchy byte proxy / leader load 汇总到同一张表。

### 2026-07-15：LGCP Upload Plan 11 帧 Offline NS3 Replay

本次结果将 hierarchy plan 的 `upload_plan.csv` 扩展到 11 帧联机 replay，并用 `opencda/tools/lgcp_ns3_log_eval.py` 解析 NS3 bridge 回传的 `cam_received`。

配置：

- 输入：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/upload_plan.csv`
- 输出：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_smoke_11f_rsu21`
- RSU node id：`21`
- Frames：`000060` 到 `000080`，共 11 帧

总览：

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| planned_bytes | 3,240,000 |
| observed_cam_received | 31 |
| matched_cam_received | 31 |
| bridge_observed_delivery_ratio | 0.045858 |
| observed_bytes | 174,000 |
| avg_delay_ms | 109.645 |
| p95_delay_ms | 209 |
| max_delay_ms | 211 |

按上传阶段：

| Upload type | Planned | Observed | Bridge-observed ratio | Avg delay ms | P95 delay ms |
| --- | --- | --- | --- | --- | --- |
| leader_to_rsu | 440 | 17 | 0.038636 | 75.471 | 115 |
| member_to_leader | 236 | 14 | 0.059322 | 151.143 | 209 |

解释：

- 该结果证明 LGCP hierarchy upload plan 可以完整进入 offline NS3 replay，并覆盖 20 CAV + RSU 的两阶段上传链路。
- 当前网络配置下，NS3 stdout 中出现大量 `PSCCH_DECODE_FAIL` / `reason=error_model`，bridge 可见交付率很低。
- 这里的 delivery ratio 是基于 `cam_received` 回调的 request-level lower-bound，不是完整 PHY/RLC trace；论文若报告严格链路层 delivery ratio，需要进一步接入 ns-3 侧 trace。

### 2026-07-15：NS3 PHY Decode-Failure Breakdown

本次在同一份 11 帧 replay 日志上解析 ns-3 PHY decode diagnostics，输出 `phy_decode_events.csv` 与 `phy_decode_summary.csv`。

总览：

| Metric | Value |
| --- | --- |
| PHY decode events | 12,740 |
| PHY decode failures | 7,779 |
| PSCCH OK | 4,709 |
| PSCCH FAIL | 7,491 |
| PSSCH OK | 252 |
| PSSCH FAIL | 288 |

Breakdown：

| Channel | Status | Reason | Count | Channel ratio | Avg SINR | Avg TBLER |
| --- | --- | --- | --- | --- | --- | --- |
| PSCCH | FAIL | decoded_overlap | 5,736 | 0.470164 | 0.171793 | 0.981760 |
| PSCCH | FAIL | error_model | 1,755 | 0.143852 | 0.308541 | 0.953855 |
| PSCCH | OK | - | 4,709 | 0.385984 | 52,207.608824 | 0.017849 |
| PSSCH | FAIL | decode_fail | 288 | 0.533333 | 3,581.433854 | 0.995913 |
| PSSCH | OK | - | 252 | 0.466667 | 373,591.447580 | 0.000912 |

解释：

- 当前 11 帧 replay 的主要 PHY 问题是 PSCCH control decode collision / overlap，而不仅仅是弱 SINR。
- 这为后续 LGCP scheduling 目标提供了直接证据：仅生成 hierarchy upload plan 不够，还需要控制同一时隙 / 子信道上的 overlap。
- 该 breakdown 尚未映射回 `upload_plan.csv` 的 request id；论文若要报告严格 request-level delivery ratio，还需要 ns-3 输出 RLC / application request-id trace。

### 2026-07-15：NS3 Application Request-ID Trace

本次扩展 ns-3 CAM header，将 replay 发送的 `pkt_id` 作为 `request_id` 透传到接收端，并在 `cam_received` JSON 中回传。解析器现在优先按 `(frame_index, request_id)` 匹配 `upload_plan.csv`。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_request_id_11f_rsu21/
```

验证结果：

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| observed_cam_received | 31 |
| matched_cam_received | 31 |
| frame_request_id matches | 31 |
| fallback endpoint matches | 0 |
| bridge_observed_delivery_ratio | 0.045858 |
| avg_delay_ms | 109.645 |
| p95_delay_ms | 209 |
| max_delay_ms | 211 |

按上传阶段：

| Upload type | Planned | Observed | Bridge-observed ratio | Avg delay ms |
| --- | --- | --- | --- | --- |
| leader_to_rsu | 440 | 17 | 0.038636 | 75.471 |
| member_to_leader | 236 | 14 | 0.059322 | 151.143 |

解释：

- 这是比上一版更可靠的 request-level bridge-observed delivery summary，因为每个 `cam_received` 都能直接映射到 `upload_plan.csv` 中的每帧 `pkt_id`。
- 该结果仍位于 application callback 层；RLC / HARQ / PHY event 到 request id 的严格映射仍需要 ns-3 侧继续暴露 RLC trace 或在 PHY tag 中携带 request id。

### 2026-07-15：NS3 RLC Request-ID Trace

本次将 request id 继续透传到 NR SL RLC TX / RX / DROP 日志，并在解析器中按 `(frame_index, request_id)` 映射回 `upload_plan.csv`。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_rlc_request_id_11f_rsu21/
```

RLC summary：

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| RLC TX events | 1,131 |
| RLC RX events | 252 |
| RLC DROP events | 0 |
| matched RLC TX events | 1,131 |
| matched RLC RX events | 252 |
| unique TX requests | 614 |
| unique RX requests | 164 |
| RLC request RX ratio | 0.242604 |

对比：

| Layer | Received requests/events | Ratio |
| --- | --- | --- |
| RLC unique RX requests | 164 / 676 | 0.242604 |
| Application `cam_received` | 31 / 676 | 0.045858 |

解释：

- RLC 层已有 request-level trace，可直接回答 “RLC / request-id trace 是否能映射回 LGCP upload request”。
- RLC RX 明显高于 application callback，说明一部分 request 已进入 RLC RX，但未完成上层 reassembly / callback。
- PHY decode / HARQ 仍只是 aggregate diagnostics；若要解释 RLC-to-application 损失，需要继续绑定 PHY TB / HARQ feedback 与 request id。

## R4：Communication-Aware Baseline

目标：补充更强 baseline，回应 baseline fairness 问题。

候选 baseline：

- 原生 Where2comm sparse sharing。
- Quality-aware sharing。
- Blind-spot-oriented scheduling。
- Compression / transmission / computation co-optimization。
- Selective sharing without local-to-global structure。

### 2026-07-15：Communication-Aware Top-K Offline Baseline

本次结果在 offline subset ablation 中新增 `comm_aware_topk`，作为不使用 LGCP hierarchy 的通信感知 selective-sharing baseline。它按 `confidence(v) / (1 + distance(v, ego) / 100)` 选择 top-k CAV；当前 distance cost 只是 offline proxy，还不是 NS3 链路质量。

配置：

- 脚本：`opencda/tools/lgcp_subset_ablation_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_comm_aware_baseline_11f`
- 帧数：`11`
- Budgets：`5`, `10`
- Feature packet byte proxy：`10000`

结果：

| Method | Budget | Non-ego packets | Byte proxy | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | --- | --- |
| Full sharing | 5 | 209 | 2,090,000 | 0.839868 | 0.525203 |
| Full sharing | 10 | 209 | 2,090,000 | 0.839868 | 0.526521 |
| Random selective | 5 | 44 | 440,000 | 0.314671 | 0.165521 |
| Random selective | 10 | 99 | 990,000 | 0.598993 | 0.380556 |
| Confidence top-k | 5 | 44 | 440,000 | 0.345556 | 0.221396 |
| Confidence top-k | 10 | 99 | 990,000 | 0.624088 | 0.436000 |
| Comm-aware top-k | 5 | 44 | 440,000 | 0.443572 | 0.296352 |
| Comm-aware top-k | 10 | 99 | 990,000 | 0.686146 | 0.545736 |
| Area-aware union | 5 | 44 | 440,000 | 0.396807 | 0.251957 |
| Area-aware union | 10 | 99 | 990,000 | 0.676678 | 0.538273 |

观察：

- `comm_aware_topk` 明显强于 `confidence_topk`，说明加入通信距离 proxy 后，selective-sharing baseline 更有竞争力。
- 在当前 11 帧 offline perception-only proxy 下，`comm_aware_topk` 还略高于 `area_aware_union`。
- 这是一条重要的边界结果：当前 offline subset ablation 不能单独证明 LGCP area-aware selection 优于所有强 selective-sharing baseline。

论文解释：

- Rebuttal 可以说已经补充了 stronger communication-aware baseline，而不是只对比 full/random。
- 但不能把当前 area-aware union 写成全面优于强 baseline；更稳妥的主张是：完整 LGCP 的贡献需要在 hierarchy、leader local fusion、RSU aggregation 和 scheduling latency 中一起体现。
- 下一步必须把 `comm_aware_topk` 纳入多 seed，并用 NS3 / link-quality proxy 替代简单距离成本。

## R5：Large-Scale Co-Simulation

目标：记录 5 到 30 CAV 的 OpenCDA + CARLA + NS3 联合仿真结果。

建议记录字段：

| CAV 数 | 方法 | Latency | Packet count | Control overhead | Delivery ratio | Quality proxy | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

需要特别说明：

- 如果只报告 latency，则论文中不能把该结果表述为 perception quality scalability。
- 如果报告 quality proxy，需要解释 proxy 与真实 AP / recall 的关系。

### 2026-07-15：Scalable Quality Proxy 校准

本次结果用于回应 “30 CAV large-scale co-simulation only reports latency” 的问题。短期口径应明确：大规模实验直接验证 latency / communication scalability；perception scalability 只能报告经过小规模 AP 校准的 proxy。

配置：

- 脚本：`opencda/tools/lgcp_quality_proxy_eval.py`
- 输入 subset：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_comm_aware_baseline_11f`
- 输入 area confidence：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_11f_detector_score/area_records.csv`
- 输出目录：`docs/doc_workspace/LGCP/experiments/large_scale_proxy/20260715_lgcp_carla_quality_proxy_11f`
- 样本：10 个 method / budget 点

Proxy 定义：

| Proxy | 含义 |
| --- | --- |
| `area_coverage_proxy_mean` | selected CAV 至少覆盖一个 area 的比例均值 |
| `confidence_max_proxy_mean` | selected CAV 在每个 area 的 max confidence 加权均值 |
| `confidence_noisy_or_proxy_mean` | selected CAV 在每个 area 的 noisy-or confidence 加权均值 |

代表性相关性：

| Proxy | Quality | Samples | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| area coverage | AP@0.5 | 10 | 0.863937 | 0.841463 |
| confidence max | AP@0.5 | 10 | 0.951439 | 0.926829 |
| confidence noisy-or | AP@0.5 | 10 | 0.966055 | 0.926829 |
| confidence noisy-or | AP@0.7 | 10 | 0.954195 | 0.802435 |

解释：

- 在 11 帧 offline AP 对照中，confidence-based proxy 与真实 AP 呈强正相关。
- 该结果支持将 `confidence_noisy_or_proxy_mean` 作为大规模 quality trend 的候选 proxy。
- 但样本仍是单场景 / 单 seed / 10 个方法点，不能替代真实 AP；论文中必须标注为 proxy。

建议论文表述：

```text
Large-scale co-simulation evaluates latency and communication scalability. Perception quality at large scale is reported using a calibrated area-confidence proxy, whose correlation with true AP is validated in the offline perception study.
```

## R7：LGCP 场景数据导出 / 离线推理 Smoke Test

### 2026-07-15

配置：`lgcp_carla`

数据目录：`D:\Data\Carla\2026_07_15_02_33_21`

导出结果：

| 项目 | 数量 |
| --- | --- |
| RSU agent | 1 (`-1`) |
| CAV agents | 20 (`1` 到 `20`) |
| 每个 agent 的 YAML/PCD 帧数 | 11 |
| ego CAV camera | 4 路 PNG |

离线推理结果：

| 测试 | 帧 | pred boxes | gt boxes | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| 单帧 | 000060 | 63 | 72 | - | - | - |
| 三帧 | 000060-000064 | 63/64/63 | 72/72/72 | 0.87 | 0.76 | 0.49 |

结论：LGCP 场景具备继续推进数据导出、离线评估和 RSU 参与式协同感知机制开发的基础。

## R6：Robustness / Sensitivity

### 待补充

目标：验证 LGCP 对关键系统参数和动态场景误差的稳定性。

建议实验：

- Localization error。
- Vehicle speed。
- Update frequency。
- Area / grid size。
- Subchannel count `Z`。
- CAV / edge compute capacity。
- Transmission rate / SINR threshold。
## 2026-07-16：Request Lifecycle Funnel Parser

在既有 11 帧 RLC request-id replay 上，`lgcp_ns3_log_eval.py` 已新增 request lifecycle 输出。

当前日志尚未包含 request-level PHY / HARQ event，因此 PHY / HARQ 计数为 0；该结果主要验证 OpenCDA 侧 parser 与 funnel 表结构已经就绪。

| Metric | Value |
| --- | --- |
| planned requests | 676 |
| requests with RLC TX | 614 |
| requests with RLC RX | 164 |
| requests with application `cam_received` | 31 |
| terminal application received | 31 |
| terminal RLC RX only | 133 |
| terminal RLC TX no RX | 450 |
| terminal planned only | 62 |
| request-level PHY/HARQ events | 0 |

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_rlc_request_id_11f_rsu21/
```

新增结果文件：

- `request_lifecycle.csv`
- `request_lifecycle_summary.csv`

解释边界：当前 funnel 可以报告 planned -> RLC TX -> RLC RX -> application received；PHY / HARQ 仍等待 ns-3 侧输出 `[NRSL_PHY_EVENT]` / `[NRSL_HARQ_EVENT]` request-level 日志。

## 2026-07-16：PSSCH Request-Level Trace Smoke

ns-3 侧已在 PSSCH decode OK/FAIL 处输出 `[NRSL_PHY_EVENT]`，OpenCDA parser 能将事件映射回 LGCP `upload_plan.csv`。

当前 3 帧 smoke：

| Metric | Value |
| --- | --- |
| planned requests | 186 |
| observed `cam_received` | 5 |
| RLC TX events | 228 |
| RLC RX events | 41 |
| aggregate PHY decode events | 2619 |
| aggregate PHY decode failures | 1653 |
| request-level PHY/HARQ events | 124 |
| matched request-level PSSCH OK events | 41 |
| matched request-level PSSCH FAIL events | 83 |
| requests with PSSCH OK | 31 |
| requests with PSSCH FAIL | 68 |
| terminal application received | 5 |
| terminal RLC RX only | 26 |
| terminal PSSCH fail | 50 |
| terminal RLC TX no RX | 43 |
| terminal planned only | 62 |

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_phy_harq_request_3f_rsu21/
```

解释边界：

- PSSCH decode OK/FAIL 已能 request-level 归因到 area / upload type。
- 当前 smoke 未观测到 HARQ ACK/NACK event，HARQ feedback 仍需继续确认配置和触发路径。

## 2026-07-16：HARQ Request-Level Trace Smoke

ns-3 侧已新增 HARQ/PSFCH 运行开关：

```text
--enableSlHarq=true --psfchPeriod=4
```

在该配置下，HARQ ACK/NACK 能通过 `[NRSL_HARQ_EVENT]` 输出，并由 OpenCDA parser 映射回 LGCP `upload_plan.csv`。

当前 3 帧 smoke：

| Metric | Value |
| --- | --- |
| planned requests | 186 |
| observed `cam_received` | 5 |
| RLC TX events | 228 |
| RLC RX events | 40 |
| aggregate PHY decode events | 2622 |
| aggregate PHY decode failures | 1660 |
| request-level PHY/HARQ events | 233 |
| matched PSSCH OK events | 40 |
| matched PSSCH FAIL events | 85 |
| matched HARQ ACK events | 40 |
| matched HARQ NACK events | 68 |
| requests with HARQ ACK | 30 |
| requests with HARQ NACK | 49 |
| terminal application received | 5 |
| terminal RLC RX only | 25 |
| terminal PSSCH fail | 50 |
| terminal RLC TX no RX | 44 |
| terminal planned only | 62 |

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_harq_request_3f_rsu21/
```

解释边界：

- HARQ event 现在可 request-level 归因。
- 该结果是 3 帧 smoke，用于验证 trace 链路；论文级结论仍需扩展到 11 帧 / 多 seed。

## 2026-07-16：11 帧 PSSCH / HARQ Request-Level Trace

使用：

```text
--enableSlHarq=true --psfchPeriod=4
```

完成 11 帧 LGCP upload plan replay，并解析 request-level PSSCH / HARQ trace。

| Metric | Value |
| --- | --- |
| planned requests | 676 |
| observed `cam_received` | 29 |
| bridge-observed delivery ratio | 0.042899 |
| average delay | 108.276 ms |
| p95 delay | 209 ms |
| RLC TX events | 1131 |
| RLC RX events | 251 |
| aggregate PHY decode events | 12736 |
| aggregate PHY decode failures | 7789 |
| request-level PHY/HARQ events | 1177 |
| matched PSSCH OK events | 251 |
| matched PSSCH FAIL events | 390 |
| matched HARQ ACK events | 251 |
| matched HARQ NACK events | 285 |
| requests with PSSCH OK | 167 |
| requests with PSSCH FAIL | 316 |
| requests with HARQ ACK | 167 |
| requests with HARQ NACK | 224 |
| requests with RLC RX | 167 |
| requests with application callback | 29 |
| terminal application received | 29 |
| terminal RLC RX only | 138 |
| terminal PSSCH fail | 222 |
| terminal RLC TX no RX | 225 |
| terminal planned only | 62 |

按 upload type 的 application callback：

| Upload type | Planned | Observed `cam_received` | Bridge-observed ratio | Avg delay |
| --- | --- | --- | --- | --- |
| leader_to_rsu | 440 | 17 | 0.038636 | 75.235 ms |
| member_to_leader | 236 | 12 | 0.050847 | 155.083 ms |

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_harq_request_11f_rsu21/
```

解释边界：

- 该结果已经可以做 request-level RLC / PSSCH / HARQ funnel。
- 仍需多 seed 或更多场景验证稳定性。

## 2026-07-16：Control-Plane Overhead Breakdown

新增离线统计工具：

```text
opencda/tools/lgcp_control_overhead_eval.py
```

在 11 帧 top-40 hierarchy plan 上显式拆分 CAV pose / direction report、area-confidence report、RSU assignment、RSU global-view broadcast 和 planned data upload。

| Metric | Mean | Max |
| --- | ---: | ---: |
| active CAVs / frame | 20.000000 | 20.000000 |
| confidence entries / frame | 1609.818182 | 1629.000000 |
| assignment entries / frame | 40.000000 | 40.000000 |
| pose report bytes / frame | 640.000000 | 640.000000 |
| confidence report bytes / frame | 25757.090909 | 26064.000000 |
| assignment bytes / frame | 2560.000000 | 2560.000000 |
| global-view bytes / frame | 2000.000000 | 2000.000000 |
| control-plane bytes / frame | 30957.090909 | 31264.000000 |
| planned data bytes / frame | 294545.454545 | 310000.000000 |
| total bytes with control / frame | 325502.545455 | 341264.000000 |
| control-plane ratio | 0.095202 | 0.099794 |

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/control_overhead_11f/
```

解释边界：

- 当前结果是 byte proxy，不是 PHY airtime 或 MAC scheduling overhead。
- 在 20 CAV / top-40 area / 11 frame 设置下，control-plane traffic 约为 30.96 KB/frame，占 planned data + control 的 9.52%。
- 该结果可回应 control-plane overhead 未量化的问题；论文级稳定性仍需多 seed / 多场景复核。

## 2026-07-16：Grid / Area Size Sensitivity Smoke

`opencda/tools/lgcp_area_confidence_eval.py` 新增 `--grid-size-x` / `--grid-size-y`，允许离线覆盖 ROI grid size，不修改 `lgcp_carla.yaml`。

共同设置：

- scenario：`2026_07_15_02_33_21`
- frames：11
- fusion method：`early`
- ROI：`280m x 80m`

| Grid size | Records | Active areas | Area-frame noisy-or vs recall@0.5 Spearman | Area-acc noisy-or vs AP@0.5 Spearman | Area-acc score_mean vs AP@0.5 Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `5m x 3m` | 46993 | 1101 | 0.475952 | 0.213836 | 0.363570 |
| `10m x 6m` | 21418 | 337 | 0.570407 | 0.411840 | 0.402059 |
| `20m x 12m` | 8386 | 94 | 0.458975 | 0.233766 | 0.472727 |

解释边界：

- default `10m x 6m` 在该 11 帧 smoke 中给出最强 area-frame recall ranking。
- `5m x 3m` 样本更多但 per-area AP 更稀疏；`20m x 12m` active areas 更少，accumulated AP 样本也更少。
- 当前只支持 default grid 的合理性，不能声称 LGCP 对 area size 完全不敏感。

## 2026-07-16：Localization Error Sensitivity Smoke

`opencda/tools/lgcp_area_confidence_eval.py` 新增 `--localization-noise-std` / `--localization-noise-seed`，在 CAV confidence report 路径中注入 deterministic xy pose noise。

共同设置：

- scenario：`2026_07_15_02_33_21`
- frames：11
- grid：`10m x 6m`
- fusion method：`early`
- noise seed：`7`

| Noise std | Records | Active areas | Area-frame noisy-or vs recall@0.5 Spearman | Area-acc noisy-or vs AP@0.5 Spearman | Area-acc score_mean vs AP@0.5 Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.0m` | 21418 | 337 | 0.570407 | 0.411840 | 0.402059 |
| `0.2m` | 21428 | 343 | 0.564515 | 0.411840 | 0.396911 |
| `0.5m` | 21408 | 347 | 0.546341 | 0.411840 | 0.396911 |
| `1.0m` | 21432 | 356 | 0.550885 | 0.314543 | 0.396911 |

解释边界：

- Area-frame confidence-to-recall ranking 在 1.0m xy pose noise 下仍保持约 `0.55` Spearman。
- Accumulated AP ranking 在 1.0m 噪声下降低，因此论文中只能作为 robustness diagnostic。
- 当前没有模拟 feature alignment 误差对 model-level fusion 的影响，后续完整 LGCP local fusion 仍需单独验证。

## 2026-07-16：Update Frequency / Stale Assignment Sensitivity Smoke

新增离线工具：

```text
opencda/tools/lgcp_stale_assignment_eval.py
```

该工具使用前 `lag_steps` 个 update 的 area confidence 预测当前 frame 的 area quality，近似较低 update frequency 或 stale assignment。

共同设置：

- scenario：`2026_07_15_02_33_21`
- frames：11
- quality：`recall_05`
- top-k：40 areas

| Lag steps | Samples | Noisy-or vs recall@0.5 Spearman | Top-40 Jaccard mean | Top-40 Jaccard min |
| --- | ---: | ---: | ---: | ---: |
| 0 | 354 | 0.584992 | 1.000000 | 1.000000 |
| 1 | 321 | 0.527720 | 0.911095 | 0.777778 |
| 2 | 289 | 0.529556 | 0.857818 | 0.777778 |
| 3 | 257 | 0.447925 | 0.805484 | 0.666667 |

解释边界：

- lag 1/2 帧仍保留较稳定 area ranking。
- lag 3 帧开始明显退化，支持论文中加入 assignment TTL 或 event-driven reassignment 机制。
- 当前是 temporal staleness proxy，没有显式改变车辆速度；真实 high-speed scenario 仍需后续多场景验证。

## 2026-07-16：Subchannel Count Z Sensitivity Proxy

新增离线工具：

```text
opencda/tools/lgcp_subchannel_sensitivity_eval.py
```

该工具基于 11 帧 `upload_plan.csv`，估算不同 `Z` 下每帧需要的顺序 scheduling slots。`member_to_leader` 和 `leader_to_rsu` 被视为两个顺序 stage。

| Z | Mean slots / frame | P95 slots / frame | Max slots / frame | Mean max stage packets / subchannel |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 12.727273 | 13.000000 | 13.000000 | 8.000000 |
| 10 | 6.727273 | 7.000000 | 7.000000 | 4.000000 |
| 15 | 5.000000 | 5.000000 | 5.000000 | 2.666667 |
| 20 | 3.727273 | 4.000000 | 4.000000 | 2.000000 |

解释边界：

- 该结果是 scheduling-capacity proxy，不是 ns-3 PHY delivery 多组重跑。
- `Z` 增大显著降低 slot proxy 和 per-subchannel pressure，可解释当前 NS3 aggregate PHY 中 PSCCH `decoded_overlap` 的敏感性。
- 后续若进入论文主实验，应补多组 NS3 replay 验证 RLC / PSSCH / HARQ delivery 是否随 `Z` 改善。

## 2026-07-17：CAV / Edge Computation Capacity Sensitivity Proxy

新增离线工具：

```text
opencda/tools/lgcp_compute_capacity_eval.py
```

该工具基于 11 帧 `hierarchy_frame_summary.csv`，使用 `leader_max_load` 估计 CAV leader local-fusion workload，使用 `covered_area_count` 估计 RSU aggregation workload。

| CAV capacity | RSU capacity | Local fusion mean ms | RSU aggregation mean ms | Compute mean ms | Compute max ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 4.409091 | 4.000000 | 8.409091 | 9.000000 |
| 4 | 20 | 2.204545 | 2.000000 | 4.204545 | 4.500000 |
| 8 | 40 | 1.102273 | 1.000000 | 2.102273 | 2.250000 |
| 16 | 80 | 0.551136 | 0.500000 | 1.051136 | 1.125000 |

解释边界：

- 该结果是 compute workload proxy，不是真实 OpenCOOD model-level runtime。
- 低 RSU capacity 时 RSU aggregation 会成为瓶颈；低 CAV capacity 时 leader local fusion 会成为瓶颈。
- 当前结果可支撑 CAV / edge capacity sensitivity 讨论，但完整 LGCP local fusion 实现后仍需测真实 runtime。

## 2026-07-17：Fig. 7 Axis and Low-Density Latency Audit

`C:\Workspace\icdcs-paper\LGCP\picture\num_latency_v2.pdf` 已渲染为 `docs/doc_workspace/LGCP/fig7_latency_audit.png` 进行人工核查。

| Item | Current state | Conclusion |
| --- | --- | --- |
| y-axis | `End-to-end latency (ms)` | 语义正确，单位明确 |
| x-axis | `Number of vehicles` | 建议改为 `Number of CAVs`，与 caption 和正文一致 |
| Low-CAV explanation | 正文当前解释不足 | 已形成 fixed overhead / sparse contention / edge compute 小规模优势解释 |

论文修改要点：

- 保留 y-axis 为 `End-to-end latency (ms)`。
- 使用原始绘图源重导出 Fig. 7，将 x-axis 改为 `Number of CAVs`。
- 在 Fig. 7 讨论中补充低密度解释：低 CAV 数时冗余和冲突尚小，LGCP 固定控制面开销未被充分摊薄，edge-assisted baseline 的边缘算力优势仍能覆盖集中式瓶颈；随 CAV 数增加，LGCP 的 area-task sharing、leader local fusion 和 RSU aggregation 才更明显体现 scalability。

## 2026-07-17：Greedy Gap O3 Latency-Aware Smoke

`opencda/tools/lgcp_greedy_gap_eval.py` 已接入 `--enable-o3`，使用 holistic exhaustive search 评估 `O3_confidence_latency_ratio`。

Run:

```text
docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260717_lgcp_carla_greedy_gap_o3_11f
```

| Objective | Delta_g | Instances | Mean relative gap | P90 relative gap | Max relative gap | Mean greedy packets | Mean optimal packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O3 | 0.050 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |
| O3 | 0.075 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |
| O3 | 0.100 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |
| O3 | 0.125 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |

Additional observations:

- O1 / O2 group-member gap remains `0.0` on the same 11 instances.
- Greedy leader min-max load gap remains `0.0`.
- O3 shows a small non-zero gap: the exhaustive optimum sometimes selects slightly more packets (`3.45` vs `3.00` mean) to improve confidence enough to offset the latency proxy.
- This supports an empirical small-gap claim for the smoke setup, but the paper should still avoid theoretical approximation wording.
