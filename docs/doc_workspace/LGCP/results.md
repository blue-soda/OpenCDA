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

## 2026-07-17：Offline Subset Ablation Multiseed Random Baseline

在 `20260715_lgcp_carla_comm_aware_baseline_11f` 的 11 帧结果基础上，补跑 random-only seeds `11 / 23 / 37`，并与原 seed `7` 汇总。Deterministic 方法复用原 11 帧结果。

Run:

```text
docs/doc_workspace/LGCP/experiments/ablation/20260717_lgcp_carla_offline_subset_multiseed_11f
```

| Method | Budget | Seeds | AP@0.3 mean | AP@0.5 mean | AP@0.7 mean | AP@0.7 std | Non-ego packets |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| random | 5 | 7,11,23,37 | 0.321267 | 0.302982 | 0.163843 | 0.026394 | 44 |
| confidence_topk | 5 | deterministic | 0.358450 | 0.345556 | 0.221396 | 0.000000 | 44 |
| area_aware_union | 5 | deterministic | 0.405018 | 0.396807 | 0.251957 | 0.000000 | 44 |
| comm_aware_topk | 5 | deterministic | 0.452105 | 0.443572 | 0.296352 | 0.000000 | 44 |
| random | 10 | 7,11,23,37 | 0.587821 | 0.571077 | 0.328993 | 0.038178 | 99 |
| confidence_topk | 10 | deterministic | 0.629379 | 0.624088 | 0.436000 | 0.000000 | 99 |
| area_aware_union | 10 | deterministic | 0.678388 | 0.676678 | 0.538273 | 0.000000 | 99 |
| comm_aware_topk | 10 | deterministic | 0.688703 | 0.686146 | 0.545736 | 0.000000 | 99 |

Interpretation:

- 多 seed random baseline 明显低于 confidence / area-aware / communication-aware selective baselines。
- `comm_aware_topk` 仍略高于当前 `area_aware_union`，说明 LGCP 论文不能只靠 offline area-aware subset AP 证明优于强 selective baseline。
- 更稳妥的 claim 是：当前 offline subset ablation 证明 strong selective baselines 必须纳入；完整 LGCP 增益需要由 hierarchy、leader local fusion、RSU aggregation 和 scheduling latency 共同支撑。

## 2026-07-17：Greedy Gap Larger O3 Instance

在原 5-agent O3 smoke 之后，进一步运行 `max_agents=6`、`max_areas=3`、`max_group_size=3`：

```text
docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260717_lgcp_carla_greedy_gap_o3_6agents_11f
```

| Objective | Delta_g | Instances | Mean relative gap | P90 relative gap | Max relative gap | Mean greedy packets | Mean optimal packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O3 | 0.050 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |
| O3 | 0.075 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |
| O3 | 0.100 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |
| O3 | 0.125 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |

Interpretation:

- O1 / O2 group-member gap and leader load gap remain `0.0` in this setting.
- O3 gap increases with candidate agents, which is expected because holistic optimal selection has more alternatives.
- This strengthens the rebuttal evidence by showing the heuristic gap under a larger enumerable setting, while preserving the limitation that it is still single-scenario / 11-frame evidence.

## 2026-07-17：Greedy Gap O3 Multiseed Sampled Smoke

新增 `--sample-seeds` 和 sampled candidate pool 后，在同一 `lgcp_carla` 11 帧 dump 上对 agents / areas 做 seed-controlled sampling。该实验不是多场景结果，但补足了 greedy O3 gap 的多 seed smoke。

Run:

```text
docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260717_lgcp_carla_greedy_gap_o3_multiseed_sampled_5agents_11f
```

Config:

```text
max_agents=5
max_areas=3
max_group_size=3
sample_seeds=7,11,23,37
candidate_pool_factor=2
instances=44
```

O3 summary:

| Objective | Delta_g | Instances | Mean relative gap | P90 relative gap | Max relative gap | Mean greedy packets | Mean optimal packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O3 | 0.050 | 44 | 0.060727 | 0.136821 | 0.187994 | 3.795455 | 3.227273 |
| O3 | 0.075 | 44 | 0.052953 | 0.116843 | 0.137931 | 3.568182 | 3.227273 |
| O3 | 0.100 | 44 | 0.047440 | 0.124867 | 0.137931 | 3.295455 | 3.227273 |
| O3 | 0.125 | 44 | 0.043650 | 0.124867 | 0.137931 | 3.159091 | 3.227273 |

O1 / O2 summary:

| Objective | Delta_g | Mean relative gap | P90 relative gap | Max relative gap |
| --- | ---: | ---: | ---: | ---: |
| O1 | 0.050 | 0.003516 | 0.014923 | 0.023611 |
| O1 | 0.075 | 0.008585 | 0.029008 | 0.048208 |
| O1 | 0.100 | 0.016768 | 0.037787 | 0.062837 |
| O1 | 0.125 | 0.022444 | 0.046197 | 0.102492 |
| O2 | 0.050 | 0.002552 | 0.012942 | 0.020422 |
| O2 | 0.075 | 0.007322 | 0.026460 | 0.043580 |
| O2 | 0.100 | 0.015180 | 0.035296 | 0.059038 |
| O2 | 0.125 | 0.020702 | 0.043054 | 0.098617 |

Interpretation:

- 多 seed sampled setting 下，O3 mean gap 约 `4.37%` 到 `6.07%`，仍属于可作为 online heuristic rebuttal 的经验 gap 证据。
- 低 `Delta_g` 下 greedy packet count 高于 optimal，说明 holistic O3 oracle 会在部分 sampled instances 中选择更紧凑的 groups。
- 6-agent multiseed sampled exhaustive run 超过本机 100s timeout；当前保留 6-agent deterministic larger-instance 证据和 5-agent multiseed sampled 证据，target 仍保持 open，等待多场景扩展。

## 2026-07-17：Hierarchy Aggregation Proxy

新增 `opencda/tools/lgcp_hierarchy_aggregation_eval.py`，将 hierarchy assignment plan 转换为 leader local result proxy 和 RSU global aggregation proxy。

Run:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_hierarchy_aggregation_top40_11f
```

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Quality areas / frame | 33.000000 |
| Selected hierarchy areas / frame | 40.000000 |
| Selected GT ratio | 1.000000 |
| Mean selected area recall@0.5 | 0.670455 |
| Mean confidence-weighted quality | 0.609181 |
| Active leaders / frame | 15.090909 |
| Leader max load / frame | 8.818182 |

Interpretation:

- Top-40 hierarchy plan covers all GT-bearing quality areas in this 11-frame dump.
- The new files materialize the missing hierarchy data products: `leader_local_results.csv`, `rsu_global_frame_summary.csv`, and `rsu_global_summary.csv`.
- This is still not real feature-slice local fusion; it is a proxy that makes the full LGCP data interface explicit before implementing model-level fusion.

## 2026-07-17：Feature Slice Manifest

新增 `opencda/tools/lgcp_feature_slice_manifest.py`，基于 hierarchy assignment plan 生成 raw LiDAR area-slice manifest。

Run:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_feature_slice_top40_11f
```

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Areas / frame | 40.000000 |
| Slice rows / frame | 61.454545 |
| Total slice points / frame | 34993.636364 |
| Member upload points / frame | 6199.181818 |
| Leader self points / frame | 28794.454545 |
| Member upload bytes / frame | 99186.909091 |
| Leader self bytes / frame | 460711.272727 |

Interpretation:

- This materializes the area-specific slicing interface required by LGCP hierarchy.
- The byte proxy is now variable and data-dependent, unlike the earlier fixed `10000 bytes` per feature packet.
- It remains a raw LiDAR proxy; model-level feature tensor slicing is still the next implementation step.

## 2026-07-17：Hierarchy Area-Budget Sweep

Run:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_hierarchy_budget_sweep_density_distance
```

Config:

```text
confidence_field=density_distance
delta_g=0.05
max_group_size=4
max_areas=10,20,30,40
```

| Max areas | Selected GT ratio | Mean area recall@0.5 | Weighted quality | Bytes / frame | Local packets / frame | Leader max load |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.472790 | 0.836364 | 0.766486 | 70821.818182 | 4.818182 | 3.000000 |
| 20 | 0.738288 | 0.827273 | 0.763475 | 138734.545455 | 9.545455 | 4.090909 |
| 30 | 0.953193 | 0.869697 | 0.795948 | 222101.818182 | 15.818182 | 6.181818 |
| 40 | 1.000000 | 0.670455 | 0.609181 | 299105.454545 | 21.454545 | 8.818182 |

Interpretation:

- Top-30 captures most GT-bearing area coverage with substantially less byte proxy and leader load than Top-40.
- Top-40 is useful as a coverage upper setting, but it includes lower-quality marginal areas; reporting only Top-40 can hide the budget tradeoff.
- This is a hierarchy control-plane / aggregation proxy result, not full model-level local fusion.

## 2026-07-17：Feature-Slice Budget Sweep

Run:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_feature_slice_budget_sweep_density_distance
```

| Max areas | Selected GT ratio | Fixed local bytes / frame | Raw member slice bytes / frame | Raw leader self bytes / frame | Raw total slice points / frame |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.472790 | 48181.818182 | 20933.818182 | 180315.636364 | 12578.090909 |
| 20 | 0.738288 | 95454.545455 | 39287.272727 | 283111.272727 | 20149.909091 |
| 30 | 0.953193 | 158181.818182 | 59415.272727 | 330554.181818 | 24373.090909 |
| 40 | 1.000000 | 214545.454545 | 99186.909091 | 460711.272727 | 34993.636364 |

Interpretation:

- Data-dependent raw member slice bytes are lower than the earlier fixed packet proxy in every budget setting.
- Top-30 gives a useful operating point: `95.32%` selected GT ratio with `59.42 KB/frame` member upload raw slice bytes.
- The result is a raw LiDAR proxy and should be described as a bridge toward neural feature slicing, not as completed model-level fusion.

## 2026-07-17：Raw-Slice-Aware Upload Plan

新增 `opencda/tools/lgcp_slice_upload_plan_eval.py`，将 Top-30 hierarchy upload plan 的 `member_to_leader` fixed bytes 替换为 raw LiDAR area-slice bytes。

Run:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_raw_slice_upload_plan_area30
```

| Upload type | Requests | New bytes total | Original bytes total | Ratio vs original |
| --- | ---: | ---: | ---: | ---: |
| member_to_leader | 174 | 653568 | 1740000 | 0.375614 |
| leader_to_rsu | 330 | 660000 | 660000 | 1.000000 |
| all | 504 | 1313568 | 2400000 | 0.547320 |

Dry-run with `offline_ns3_replay.py --lgcp-upload-plan` succeeded for 11 frames:

| Metric | Value |
| --- | ---: |
| Requests / frame | 45-48 |
| Bytes / frame | 105056-133680 |
| Unmatched member rows | 0 |

Interpretation:

- The hierarchy upload plan can now use data-dependent area-slice bytes without changing the replay bridge.
- This reduces Top-30 total planned bytes by about `45.27%` versus the fixed-byte proxy.
- This is still raw LiDAR slicing; neural feature tensor bytes and real leader fusion remain future work.

3-frame live ns-3 replay smoke also completed:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_raw_slice_upload_plan_area30/ns3_smoke_3f_rsu21
```

| Frame | Timestamp | Requests | Bytes |
| ---: | --- | ---: | ---: |
| 1 | `000060` | 46 | 125888 |
| 2 | `000062` | 46 | 121456 |
| 3 | `000064` | 45 | 105056 |

Boundary: this confirms live bridge acceptance and request emission, but does
not replace request-level delivery parsing because a complete ns-3 stdout log
was not captured for this smoke.

Request-level rerun with full stdout:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_raw_slice_upload_plan_area30/ns3_request_trace_3f_rsu21
```

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Planned bytes | 352400 |
| Observed `cam_received` | 6 |
| Bridge-observed delivery ratio | 0.043796 |
| Observed bytes | 12048 |
| Avg delay | 23.667 ms |
| P95 delay | 114.000 ms |
| RLC TX events | 106 |
| RLC RX events | 20 |
| Requests with PSSCH OK | 14 |
| Requests with PSSCH FAIL | 51 |

By upload type:

| Upload type | Planned | Observed app | Bridge ratio | Planned bytes | Observed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| member_to_leader | 47 | 1 | 0.021277 | 172400 | 2048 |
| leader_to_rsu | 90 | 5 | 0.055556 | 180000 | 10000 |

PHY decode breakdown:

| Channel | Status | Reason | Count | Channel ratio |
| --- | --- | --- | ---: | ---: |
| PSCCH | FAIL | decoded_overlap | 441 | 0.509827 |
| PSCCH | FAIL | error_model | 38 | 0.043931 |
| PSCCH | OK | - | 386 | 0.446243 |
| PSSCH | FAIL | decode_fail | 40 | 0.666667 |
| PSSCH | OK | - | 20 | 0.333333 |

Interpretation:

- The raw-slice-aware plan now has a complete 3-frame request-level trace.
- The network bottleneck remains severe under the current unscheduled replay setting; this supports the need for LGCP scheduling rather than weakening it.
- This is a smoke result, not a final paper row; use it to validate the trace path and motivate scheduled replay.

11-frame request-level trace:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_raw_slice_upload_plan_area30/ns3_request_trace_11f_rsu21
```

| Metric | Value |
| --- | ---: |
| Planned requests | 504 |
| Planned bytes | 1313568 |
| Observed `cam_received` | 55 |
| Bridge-observed delivery ratio | 0.109127 |
| Observed bytes | 116624 |
| Avg delay | 83.109 ms |
| P95 delay | 114.000 ms |
| Max delay | 212.000 ms |
| RLC TX events | 546 |
| RLC RX events | 118 |
| Requests with RLC TX | 446 |
| Requests with RLC RX | 94 |
| Requests with PSSCH OK | 94 |
| Requests with PSSCH FAIL | 250 |

By upload type:

| Upload type | Planned | Observed app | Bridge ratio | Planned bytes | Observed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| member_to_leader | 174 | 8 | 0.045977 | 653568 | 22624 |
| leader_to_rsu | 330 | 47 | 0.142424 | 660000 | 94000 |

PHY decode breakdown:

| Channel | Status | Reason | Count | Channel ratio |
| --- | --- | --- | ---: | ---: |
| PSCCH | FAIL | decoded_overlap | 1850 | 0.412670 |
| PSCCH | FAIL | error_model | 216 | 0.048182 |
| PSCCH | OK | - | 2417 | 0.539148 |
| PSSCH | FAIL | decode_fail | 193 | 0.620579 |
| PSSCH | OK | - | 118 | 0.379421 |

Interpretation:

- The 11-frame raw-slice-aware trace closes the request lifecycle path over the full local dump.
- Current unscheduled replay is network-limited; leader-to-RSU application visibility is higher than member-to-leader, but both are low.
- This result should be used as trace validation and motivation for scheduled replay, not as the final LGCP network row.

## LGCP Raw-Slice Scheduled NS3 Smoke

`opencda/tools/lgcp_schedule_upload_plan_eval.py` converts the raw-slice-aware plan into a single-slot, capacity-gated scheduled smoke input. `offline_ns3_replay.py` now preserves `sc_start/sc_num` from LGCP CSV rows, so the scheduled plan can drive ns-3 manual resource allocation.

Run directory:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_raw_slice_scheduled_smoke_z10
```

11-frame scheduled plan summary:

| Metric | Value |
| --- | ---: |
| Input requests | 504 |
| Scheduled requests | 110 |
| Capacity-gated requests | 394 |
| Scheduled request ratio | 0.218254 |
| Input bytes | 1313568 |
| Scheduled bytes | 543408 |
| Scheduled byte ratio | 0.413689 |

3-frame live ns-3 scheduled request-level trace:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_raw_slice_scheduled_smoke_z10/ns3_request_trace_3f_rsu21
```

| Metric | Value |
| --- | ---: |
| Planned requests | 30 |
| Observed `cam_received` | 24 |
| Bridge-observed delivery ratio | 0.800000 |
| Planned bytes | 146992 |
| Observed bytes | 129696 |
| Avg delay | 20.833 ms |
| P95 delay | 42.000 ms |
| RLC TX events | 415 |
| RLC RX events | 376 |
| Requests with PSSCH OK | 28 |
| Requests with PSSCH FAIL | 0 |

By upload type:

| Upload type | Planned | Observed app | Bridge ratio | Planned bytes | Observed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| member_to_leader | 21 | 15 | 0.714286 | 128992 | 111696 |
| leader_to_rsu | 9 | 9 | 1.000000 | 18000 | 18000 |

PHY decode breakdown:

| Channel | Status | Reason | Count | Channel ratio |
| --- | --- | --- | ---: | ---: |
| PSSCH | OK | - | 376 | 1.000000 |

Interpretation:

- The scheduled smoke sharply reduces visible decode failures compared with the unscheduled raw-slice replay.
- This validates the `sc_start/sc_num` path and motivates a full LGCP scheduler.
- It is a single-slot capacity-gated smoke; do not report it as final end-to-end LGCP throughput or perception performance.

## LGCP Multi-Slot Scheduling Proxy

`opencda/tools/lgcp_schedule_upload_plan_eval.py` now also supports `--schedule-mode multi_slot`. Unlike the single-slot NS3 smoke input, this mode schedules every raw-slice-aware request into sequential member-to-leader and leader-to-RSU slots. It produces a full-plan scheduling proxy with `slot_index`, `sc_start`, `sc_num`, `stage`, and `scheduled_delay_ms` for every request.

Run directory:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_raw_slice_multislot_schedule_z10
```

Configuration:

| Field | Value |
| --- | ---: |
| Subchannels per slot | 10 |
| Slot duration | 10 ms |
| Schedule mode | `multi_slot` |
| Priority mode | `hierarchy_bytes` |

Summary:

| Metric | Value |
| --- | ---: |
| Input requests | 504 |
| Scheduled requests | 504 |
| Capacity-gated requests | 0 |
| Scheduled request ratio | 1.000000 |
| Input bytes | 1313568 |
| Scheduled bytes | 1313568 |
| Scheduled byte ratio | 1.000000 |
| Mean slots / frame | 5.000000 |
| Max slots / frame | 5 |
| Mean frame scheduling latency | 50.000 ms |
| Max frame scheduling latency | 50.000 ms |

Per-frame structure is stable across the 11-frame local dump:

| Stage | Requests / frame | Slots / frame |
| --- | ---: | ---: |
| member-to-leader | 15-18 | 2 |
| leader-to-RSU | 30 | 3 |
| total | 45-48 | 5 |

Interpretation:

- The Top-30 raw-slice plan can be fully scheduled with `Z=10` in five sequential slots per frame.
- At the proxy value of `10 ms/slot`, the two-stage LGCP transfer plan adds `50 ms/frame` of scheduled communication latency before RSU aggregation.
- This is the full-plan scheduler proxy; the current live NS3 proof remains the 3-frame single-slot scheduled smoke.

## LGCP Multi-Slot Live Replay Smoke

`offline_ns3_replay.py` now supports `--respect-slot-index`. For LGCP upload plans that contain `slot_index`, the replay sends requests slot-by-slot and advances NS3 time by `--slot-duration-seconds` after each slot. The default replay behavior is unchanged.

Run directory:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_raw_slice_multislot_replay_dryrun_z10
```

Dry-run check:

| Frame | Timestamp | Requests | Slots | Slotted requests |
| ---: | --- | ---: | ---: | ---: |
| 1 | `000060` | 46 | 5 | 46 |
| 2 | `000062` | 46 | 5 | 46 |
| 3 | `000064` | 45 | 5 | 45 |

3-frame live multi-slot NS3 replay:

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Observed `cam_received` | 54 |
| Bridge-observed delivery ratio | 0.394161 |
| Planned bytes | 352400 |
| Observed bytes | 120176 |
| Avg delay | 68.333 ms |
| P95 delay | 202.000 ms |
| RLC TX events | 1013 |
| RLC RX events | 737 |
| Requests with RLC TX | 127 |
| Requests with RLC RX | 110 |
| Requests with PSSCH OK | 110 |
| Requests with PSSCH FAIL | 0 |

By upload type:

| Upload type | Planned | Observed app | Bridge ratio | Planned bytes | Observed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| member_to_leader | 47 | 2 | 0.042553 | 172400 | 16176 |
| leader_to_rsu | 90 | 52 | 0.577778 | 180000 | 104000 |

PHY decode breakdown:

| Channel | Status | Reason | Count | Channel ratio |
| --- | --- | --- | ---: | ---: |
| PSSCH | OK | - | 737 | 1.000000 |

Interpretation:

- Slot-indexed live replay preserves all 137 requests over 3 frames, rather than capacity-gating to 30 requests.
- Compared with the unscheduled raw-slice 3-frame trace, bridge-observed delivery improves from `0.043796` to `0.394161`, and request-level PSSCH failures drop from 51 requests to 0.
- Member-to-leader application callbacks are still low despite PSSCH/RLC success, so the next live-replay investigation should focus on application-level completion timing, fragmentation, or drain duration.

## LGCP Multi-Slot Lifecycle Diagnostics

`opencda/tools/lgcp_lifecycle_diagnostics.py` joins `request_lifecycle.csv` with the replayed upload plan, exposing stage / slot / endpoint lifecycle ratios. A second 3-frame live replay with `--drain-seconds 1.0` produced the same delivery summary as the 0.3s drain run, so the low member-to-leader application callback is not explained by too-short drain time.

Run directory:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_raw_slice_multislot_replay_drain1_z10
```

Stage summary:

| Stage | Planned | RLC TX | RLC RX | PSSCH OK | PSSCH fail | CAM ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| member-to-leader | 47 | 40 | 28 | 28 | 0 | 0.042553 |
| leader-to-RSU | 90 | 87 | 82 | 82 | 0 | 0.577778 |

Slot summary:

| Stage | Slot | Planned | RLC RX | PSSCH OK |
| --- | ---: | ---: | ---: | ---: |
| member-to-leader | 0 | 30 | 21 | 21 |
| member-to-leader | 1 | 17 | 7 | 7 |
| leader-to-RSU | 2 | 30 | 24 | 24 |
| leader-to-RSU | 3 | 30 | 29 | 29 |
| leader-to-RSU | 4 | 30 | 29 | 29 |

Terminal-state breakdown:

| Upload type | Terminal state | Requests |
| --- | --- | ---: |
| member-to-leader | application_received | 2 |
| member-to-leader | rlc_rx_only | 26 |
| member-to-leader | rlc_tx_no_rx | 12 |
| member-to-leader | planned_only | 7 |
| leader-to-RSU | application_received | 52 |
| leader-to-RSU | rlc_rx_only | 32 |
| leader-to-RSU | rlc_tx_no_rx | 6 |

Member-to-leader size-bin diagnostics:

| Planned bytes | Planned | RLC RX | PSSCH OK | CAM ratio |
| --- | ---: | ---: | ---: | ---: |
| 0-1000 | 7 | 4 | 4 | 0.000000 |
| 1000-2000 | 9 | 2 | 2 | 0.000000 |
| 2000-4000 | 21 | 12 | 12 | 0.047619 |
| 4000-8000 | 4 | 4 | 4 | 0.000000 |
| 8000-16000 | 6 | 6 | 6 | 0.166667 |

Interpretation:

- The scheduler path itself is working: no PSSCH failures are observed in either stage.
- The member-to-leader bottleneck is split between RLC/PSSCH non-arrival and application-level non-callback after RLC RX.
- The bottleneck is not simply caused by large raw slices: the largest member-to-leader bin (`8000-16000` bytes) reaches RLC/PSSCH for all 6 requests, while the `1000-4000` byte bins are weaker.
- Next debugging should inspect member slot timing / target receiver setup and CAM application completion for non-RSU receivers.

## LGCP Source-Unique Multi-Slot Sensitivity

`lgcp_schedule_upload_plan_eval.py` now supports `--enforce-source-unique`, which prevents a CAV from sending more than one request in the same slot. This tests whether member-to-leader losses are caused by a single source being scheduled to multiple leaders simultaneously.

Run directory:

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_raw_slice_multislot_source_unique_z10
```

Schedule proxy:

| Metric | Value |
| --- | ---: |
| Scheduled requests | 504 / 504 |
| Mean slots / frame | 7.363636 |
| Max slots / frame | 8 |
| Mean frame scheduling latency | 73.636 ms |
| Max frame scheduling latency | 80.000 ms |

3-frame live replay:

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Observed `cam_received` | 52 |
| Bridge-observed delivery ratio | 0.379562 |
| Avg delay | 59.769 ms |
| P95 delay | 111.000 ms |
| Requests with PSSCH FAIL | 0 |

By upload type:

| Upload type | Planned | Observed app | Bridge ratio | Requests with RLC RX |
| --- | ---: | ---: | ---: | ---: |
| member-to-leader | 47 | 5 | 0.106383 | 25 |
| leader-to-RSU | 90 | 45 | 0.500000 | 84 |

Interpretation:

- Source-unique packing avoids same-source same-slot transmissions and increases member-to-leader application callbacks from `2/47` to `5/47`.
- It does not improve total delivery versus the previous multi-slot replay (`52/137` vs `54/137`) and lowers member-to-leader RLC RX from `28/47` to `25/47`.
- Therefore source uniqueness is a useful half-duplex constraint to retain in the scheduler design, but it is not sufficient to solve the member-to-leader path by itself.
## Model-Level Hierarchy Boundary

2026-07-18 完成模型级 hierarchy 入口审计，结论如下：

| Item | Status | Paper-safe interpretation |
| --- | --- | --- |
| RSU assignment / upload plan / scheduler | Implemented | 可作为 LGCP control-plane 与 network scheduling 证据 |
| Leader local result / RSU aggregation proxy | Implemented as proxy | 可说明接口、coverage 和 byte/quality proxy，不能声称真实模型级融合 AP |
| Box-level hierarchy late fusion | Next implementation target | 可作为 local-to-global hierarchy ablation 的真实 OpenCOOD 推理版本 |
| Neural feature slicing | Pending | 只有完成 PointPillar intermediate feature slice 后，才能称为完整 LGCP model-level feature hierarchy |

该结论记录在 `model_level_hierarchy_entry.md`，用于约束后续论文表述和实现优先级。

## PointPillar Intermediate Feature Geometry Probe

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_feature_probe.py`，用于确认 LGCP area cell 到 PointPillar intermediate BEV feature tensor 的映射关系：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_probe_area23_1f5a
```

| Item | Value |
| --- | --- |
| Fusion method | `intermediate_attentive` |
| Frames / areas | `1 / 5` |
| Lidar range | `[-140.8, -40, -3, 140.8, 40, 1]` |
| Voxel size | `[0.4, 0.4, 4]` |
| Scatter tensor | `N x 64 x 200 x 704` |
| Fused backbone tensor | `1 x 384 x 100 x 352` |
| Fused slice cells | `126-225` |

Interpretation:

- This is the first validated neural-feature entry for LGCP: world-coordinate area cells can be mapped to leader-local PointPillar BEV feature index ranges.
- The recorded float32 byte estimates are uncompressed upper bounds, not final communication numbers.
- The result moves model-level hierarchy from code-entry audit to concrete feature-slice adapter work.

## PointPillar Feature Slice Export Smoke

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_feature_slice_export.py`，在 geometry probe 基础上实际裁剪并保存 feature tensor slices：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_slice_export_area23_1f5a
```

| Item | Value |
| --- | ---: |
| Rows | 5 |
| Slice level | `both` |
| Dtype | `float16` |
| Uncompressed bytes | 1502848 |
| Compressed `.npz` bytes | 178855 |
| Mean compressed bytes / area | 35771 |

Top-23 first-frame extension:

| Item | Value |
| --- | ---: |
| Rows / slice files | 23 |
| Uncompressed bytes | 6183680 |
| Compressed `.npz` bytes | 810688 |
| Mean compressed bytes / area | 35247.304348 |

Interpretation:

- The model-level hierarchy path now has a real feature crop / slice manifest smoke.
- The saved `.npz` files contain cropped `scatter` and `fused` tensors plus bounds and group metadata.
- This is still an interface result, not leader-local fusion or RSU global AP.

## Leader-Local Feature Fusion Smoke

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_leader_feature_fusion.py`，读取 Top-23 首帧 feature-slice manifest，并将 group 内 per-CAV `scatter` slices 融合为 leader-local feature slices：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f
```

| Item | Value |
| --- | ---: |
| Rows | 23 |
| Fusion methods | `mean,max` |
| Dtype | `float16` |
| Uncompressed bytes | 7189760 |
| Compressed `.npz` bytes | 936298 |
| Mean compressed bytes / area | 40708.608696 |

Interpretation:

- This moves the neural hierarchy path from feature crop/export to leader-local feature fusion.
- The current `mean/max` scatter fusion is a deterministic smoke, not a trained attentive fusion replacement.
- RSU global feature assembly and AP evaluation remain pending.

## RSU Feature Assembly Smoke

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_rsu_feature_assembly.py`，将 Top-23 首帧 leader-local feature slices 装配回统一 PointPillar scatter canvas：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f
```

| Item | Value |
| --- | ---: |
| Frames | 1 |
| Input / used leader slices | 23 / 23 |
| Canvas shape | `1 x 64 x 200 x 704` |
| Covered cells | 4669 |
| Coverage ratio | 0.033161 |
| Overlap cells | 2835 |
| Max overlap | 16 |
| Compressed `.npz` bytes | 82974 |

Interpretation:

- The neural hierarchy path now covers feature crop, leader-local fusion, and RSU-side canvas assembly.
- This is still not a trained RSU aggregation head and does not produce AP.
- The current canvas is an index-space assembly smoke; leader-local slices are not yet reprojected into a unified world/RSU coordinate frame.

## RSU Detection Head Probe

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_rsu_head_probe.py`，将 assembled RSU canvas 接回 PointPillar backbone、classification/regression heads 和 voxel postprocess：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_rsu_head_probe_area23_1f
```

| Item | Value |
| --- | ---: |
| Input canvas | `1 x 64 x 200 x 704` |
| Backbone output | `1 x 384 x 100 x 352` |
| `psm` | `1 x 2 x 100 x 352` |
| `rm` | `1 x 14 x 100 x 352` |
| Score max | 0.220411 |
| Score mean | 0.002679 |
| Postprocess threshold | 0.2 |
| Postprocess pred boxes | 2 |

Interpretation:

- The assembled canvas is technically compatible with the downstream PointPillar heads and voxel postprocessor.
- Because the current assembly is not world-aligned across leaders, these boxes are interface evidence, not valid model-level AP.
- The next publishable path is either world/RSU-frame feature reprojection before postprocess, or a clearly scoped feature-level coverage/byte proxy.

## Reference-Frame Alignment Diagnostic

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_reference_aligned_assembly.py`，以 CAV 1 lidar frame 作为统一 reference，将 Top-23 首帧 leader feature slices 按 world-coordinate area cell 映射到 reference canvas，并复用 head probe：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_reference_aligned_head_probe_area23_1f_ref1
```

| Item | Value |
| --- | ---: |
| Used leader slices | 23 |
| Coverage ratio | 0.065263 |
| Overlap cells | 293 |
| Max overlap | 3 |
| Mean abs yaw delta | 93.412838 deg |
| Max abs yaw delta | 175.817131 deg |
| Mean resize area ratio | 0.637908 |
| Head score max | 0.867036 |
| Postprocess pred boxes | 18 |

Interpretation:

- Reference-frame target bounds improve coverage and produce stronger head responses than the earlier index-space canvas.
- The large yaw deltas prove that nearest resize is not a valid geometric feature warp.
- This should be reported only as a diagnostic toward coordinate-aware feature alignment, not as model-level detection performance.

## Coordinate-Warp Feature Assembly Smoke

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_coordinate_warp_assembly.py`，对 reference-frame target cells 执行 `reference -> world -> leader` 反查采样，并复用 head probe：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_head_probe_area23_1f_ref1
```

| Item | Value |
| --- | ---: |
| Used leader slices | 23 |
| Target / sampled cells | `8550 / 8550` |
| Sample ratio | 1.000000 |
| Coverage ratio | 0.060724 |
| Overlap cells | 0 |
| Max overlap | 1 |
| Mean abs yaw delta | 93.412838 deg |
| Max abs yaw delta | 175.817131 deg |
| Head score max | 0.893363 |
| Head score mean | 0.003926 |
| Postprocess pred boxes | 30 |

Interpretation:

- Coordinate-warp assembly is a stronger feasibility result than nearest resize: every target cell is sampled through the leader-local coordinate frame.
- It is still nearest-neighbor feature sampling, not a trained or calibrated feature warp.
- Do not report AP from this path until a GT/AP smoke and geometric calibration are done.

## Coordinate-Warp AP Probe

2026-07-18 新增 `opencda/tools/lgcp_pointpillar_warp_ap_probe.py`，闭合 warped feature canvas 到 reference-frame GT/AP 的评价链路：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_ap_probe_area23_1f_ref1
```

| Item | Value |
| --- | ---: |
| Frames | 1 |
| Pred boxes | 30 |
| GT boxes | 16 |
| AP@0.3 | 0.010000 |
| AP@0.5 | 0.010000 |
| AP@0.7 | 0.000000 |

Interpretation:

- The evaluation chain is technically closed, but nearest-neighbor coordinate warp produces very weak AP.
- This should be treated as a negative smoke result and a boundary for paper claims.
- Near-term paper-safe usage is feature-level feasibility/coverage/byte evidence, unless a calibrated bilinear/affine warp or retrained aggregation head is implemented.

## Box-Level Hierarchy Late-Fusion Smoke

2026-07-18 新增 `opencda/tools/lgcp_hierarchy_late_fusion_eval.py`，并完成最小 model-calling smoke：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_smoke_area2
```

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

Interpretation:

- This smoke verifies the model-calling local-to-global path: leader/group OpenCOOD inference -> world-coordinate area slicing -> RSU global late fusion -> AP summary.
- It is not a paper result because it only covers 1 frame and 2 areas from the Top-30 plan.
- The next publishable step is Top-30 multi-frame evaluation and comparison against flat selective-sharing baselines.

## Box-Level Hierarchy Top-30 One-Frame

2026-07-18 将 `lgcp_hierarchy_late_fusion_eval.py` 扩大到 Top-30 的完整首帧：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_1f
```

| Metric | Value |
| --- | ---: |
| Frames | 1 |
| Assignment rows | 30 |
| Cached group inference calls | 23 |
| Leader local pred boxes | 38 |
| Leader local GT boxes | 35 |
| RSU fused pred boxes | 35 |
| RSU fused GT boxes | 35 |
| AP@0.3 | 0.606851 |
| AP@0.5 | 0.606851 |
| AP@0.7 | 0.517668 |

Interpretation:

- This verifies that the adapter can handle a complete Top-30 area budget for a frame, not only a two-area smoke.
- Caching reduced 30 area rows to 23 unique OpenCOOD group inference calls.
- Because this is one frame only, it should remain a method-validation result until expanded to 3/11 frames and compared with the existing flat selective-sharing baselines.

## Box-Level Hierarchy Top-30 Three-Frame

2026-07-18 将 Top-30 box-level hierarchy late-fusion 扩大到 3 帧：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_3f
```

| Metric | Value |
| --- | ---: |
| Frames | 3 |
| Assignment rows | 90 |
| Cached group inference calls | 68 |
| Mean RSU fused pred boxes / frame | 35.666667 |
| Mean RSU fused GT boxes / frame | 35.666667 |
| AP@0.3 | 0.584564 |
| AP@0.5 | 0.584564 |
| AP@0.7 | 0.508387 |
| GT total | 107 |

Per-frame:

| Timestamp | Planned areas | Groups | RSU pred / GT |
| --- | ---: | ---: | ---: |
| `000060` | 30 | 23 | 35 / 35 |
| `000062` | 30 | 23 | 36 / 35 |
| `000064` | 30 | 22 | 36 / 37 |

Interpretation:

- The 3-frame result confirms the adapter is not limited to a single timestamp and can cache repeated leader/group plans across a full Top-30 area budget.
- It remains a method-validation result until expanded to the full 11-frame local dump and compared against flat selective-sharing baselines under the same frame range.

## Box-Level Hierarchy Top-30 Eleven-Frame

2026-07-18 完成 Top-30 box-level hierarchy late-fusion 的完整 11 帧本地 dump 运行：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_11f
```

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Assignment rows | 330 |
| Cached group inference calls | 245 |
| Mean RSU fused pred boxes / frame | 34.909091 |
| Mean RSU fused GT boxes / frame | 37.090909 |
| AP@0.3 | 0.602748 |
| AP@0.5 | 0.602748 |
| AP@0.7 | 0.506345 |
| GT total | 408 |
| Pred samples | 384 |

Interpretation:

- This is the first complete 11-frame model-calling local-to-global hierarchy result on the local `lgcp_carla` dump.
- It supports the local-to-global ablation direction more directly than the previous quality proxy because OpenCOOD is actually invoked for each leader area-task group.
- It remains box-level late fusion. It should be described separately from future neural feature slicing / intermediate fusion.
- The next table should align this result with the existing 11-frame flat selective-sharing baselines and report both perception AP and communication byte proxy.

## Local-to-Global Ablation Alignment

2026-07-18 对齐 11 帧 Top-30 box-level hierarchy late-fusion 与既有 flat selective-sharing baselines：

```text
docs/doc_workspace/LGCP/experiments/ablation/20260718_lgcp_local_to_global_ablation_alignment
```

| Method | Structure | Budget | AP@0.5 | AP@0.7 | Bytes / frame | Byte proxy |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Full sharing | Flat | 20 agents | 0.839868 | 0.526521 | 190000.000000 | fixed 10KB / non-ego agent |
| Random | Flat | 10 agents | 0.598993 | 0.380556 | 90000.000000 | fixed 10KB / non-ego agent |
| Confidence top-k | Flat | 10 agents | 0.624088 | 0.436000 | 90000.000000 | fixed 10KB / non-ego agent |
| Comm-aware top-k | Flat | 10 agents | 0.686146 | 0.545736 | 90000.000000 | fixed 10KB / non-ego agent |
| Area-aware union | Flat | 10 agents | 0.676678 | 0.538273 | 90000.000000 | fixed 10KB / non-ego agent |
| LGCP Top-20 box late fusion | Local-to-global hierarchy | 20 areas | 0.538594 | 0.440331 | 79287.272727 | raw member slice + leader result estimate |
| LGCP Top-23 box late fusion | Local-to-global hierarchy | 23 areas | 0.554762 | 0.460461 | 93985.454545 | raw member slice + leader result estimate |
| LGCP Top-30 box late fusion | Local-to-global hierarchy | 30 areas | 0.602748 | 0.506345 | 119415.272727 | scheduled raw-slice plan |

Interpretation:

- LGCP Top-20 是低于 flat 10-agent `90KB/frame` proxy 的近 common-budget 点，但 AP@0.5 只有 `0.538594`，说明当前 box-level hierarchy 在严格低预算下质量下降明显。
- LGCP Top-23 是最接近 `90KB/frame` 的实测点，约 `93.99KB/frame`，AP@0.5 为 `0.554762`，仍低于所有 10-agent flat baselines。
- LGCP Top-30 的 AP@0.5 接近 random 10-agent flat selection，但低于 strong flat top-k baselines；AP@0.7 明显优于 random，并接近 area-aware / communication-aware flat baselines。
- Byte proxy 不是完全同口径：flat baselines 使用固定 selected-agent packet proxy，LGCP 使用 raw-slice-aware scheduled upload bytes。论文表格必须显式标注 proxy 类型，不能直接宣称通信量绝对公平。
- 当前结果支持谨慎结论：local-to-global mechanism path 已经被真实模型调用验证，但若要主张 perception AP 超过强 flat selective-sharing baseline，还需要 neural feature slicing 或更严格的统一 byte accounting。

### Unified Raw-Byte Accounting

2026-07-18 新增 flat baseline raw PCD byte accounting：

```text
docs/doc_workspace/LGCP/experiments/ablation/20260718_lgcp_flat_raw_byte_accounting_11f
docs/doc_workspace/LGCP/experiments/ablation/20260718_lgcp_local_to_global_ablation_alignment/unified_raw_byte_accounting_summary.csv
```

| Method | AP@0.5 | AP@0.7 | Raw bytes / frame | Byte ratio vs comm-aware top-k |
| --- | ---: | ---: | ---: | ---: |
| Comm-aware top-k, 10 agents | 0.686146 | 0.545736 | 741029.818182 | 1.000000 |
| Area-aware union, 10 agents | 0.676678 | 0.538273 | 743892.363636 | 1.003863 |
| LGCP Top-23 box late fusion | 0.554762 | 0.460461 | 93985.454545 | 0.126831 |
| LGCP Top-30 box late fusion | 0.602748 | 0.506345 | 119415.272727 | 0.161148 |

Interpretation:

- 若 flat baselines 按完整 selected-agent raw PCD 计算，10-agent communication-aware baseline 约为 `741.03KB/frame`。
- LGCP Top-30 约为 `119.42KB/frame`，仅为 comm-aware top-k raw bytes 的 `16.11%`，同时保留 `87.85%` 的 AP@0.5 和 `92.78%` 的 AP@0.7。
- 这不能证明当前 box-level LGCP 的 AP 超过强 baseline，但可以更公平地支撑“以较低通信量保留大部分感知质量”的 rebuttal 口径。

### Unified Area-Slice Accounting

2026-07-18 进一步新增 flat baseline area-slice byte accounting：flat 方法仍保留自己的 selected agents，但只统计同一组 LGCP planned area cells 内的 raw points。

```text
docs/doc_workspace/LGCP/experiments/ablation/20260718_lgcp_flat_area_slice_accounting_area23_11f
docs/doc_workspace/LGCP/experiments/ablation/20260718_lgcp_flat_area_slice_accounting_area30_11f
docs/doc_workspace/LGCP/experiments/ablation/20260718_lgcp_local_to_global_ablation_alignment/unified_area_slice_accounting_summary.csv
```

| Area plan | Method | AP@0.5 | AP@0.7 | Area-slice bytes / frame | Byte ratio vs comm-aware top-k |
| --- | --- | ---: | ---: | ---: | ---: |
| Top-23 | Comm-aware top-k, 10 agents | 0.686146 | 0.545736 | 253130.181818 | 1.000000 |
| Top-23 | LGCP Top-23 box late fusion | 0.554762 | 0.460461 | 93985.454545 | 0.371292 |
| Top-30 | Comm-aware top-k, 10 agents | 0.686146 | 0.545736 | 295755.636364 | 1.000000 |
| Top-30 | LGCP Top-30 box late fusion | 0.602748 | 0.506345 | 119415.272727 | 0.403765 |

Interpretation:

- 即使 flat baselines 也只传 LGCP planned area cells 的 raw slices，comm-aware top-k 在 Top-30 area plan 下仍需 `295.76KB/frame`。
- LGCP Top-30 用 `40.38%` 的 comm-aware area-slice bytes，保留 `87.85%` AP@0.5 和 `92.78%` AP@0.7。
- 这是比 fixed 10KB proxy 和 full selected-agent raw bytes 都更公平的本地通信口径；仍然应谨慎表述为 bounded quality loss under much lower communication，而不是 AP superiority。

## R4：Model-Level Feature Hierarchy Boundary

### 2026-07-22：Where2comm 4-Leader Feature-Packet Diagnostic

本次结果用于验证“少数 Leader 接管更多 members 与 areas”是否能缓解 Top-23 大范围下的 RSU feature fusion 退化。实验将原 Top-23 assignment 每帧重分配给最多 4 个 Leader，并让每个 Leader 只上传一个合并后的 feature packet。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_carla_hierarchy_plan_area23_4leaders
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_where2comm_leader_feature_top23_4leaders_11f_areaobj_dilate1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_where2comm_leader_feature_top23_4leaders_11f_areaobj_dilate1_thr001
```

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples | Valid leader packets/frame | Member upload KB/frame | Second-hop Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5 leader feature packet, threshold 0.05 | 0.708405 | 0.599561 | 0.038212 | 85 | 102 | 3.73 | 18.26 | 7.274124 |
| Top-23 per-area feature packet, threshold 0.05 | 0.157367 | 0.157367 | 0.000123 | 290 | 55 | 17.36 | 47.99 | 25.600930 |
| Top-23 4-leader feature packet, threshold 0.05 | 0.650372 | 0.412919 | 0.021884 | 290 | 305 | 3.73 | 227.50 | 24.865513 |
| Top-23 4-leader feature packet, threshold 0.01 | 0.578643 | 0.357752 | 0.016061 | 290 | 432 | 3.73 | 227.50 | 24.865513 |
| Top-23 5-leader feature packet, threshold 0.05 | 0.642838 | 0.409177 | 0.030923 | 290 | 305 | 4.27 | 191.92 | 24.886924 |
| Top-23 6-leader feature packet, threshold 0.05 | 0.658934 | 0.452508 | 0.026470 | 290 | 302 | 5.09 | 154.50 | 24.707258 |
| Top-23 7-leader feature packet, threshold 0.05 | 0.669542 | 0.484321 | 0.026363 | 290 | 289 | 5.91 | 132.88 | 24.756596 |
| Top-23 8-leader feature packet, threshold 0.05 | 0.623809 | 0.459635 | 0.035736 | 290 | 279 | 6.45 | 99.98 | 24.767767 |
| Top-23 13-leader feature packet, threshold 0.05 | 0.541581 | 0.381057 | 0.043158 | 290 | 249 | 8.27 | 56.06 | 25.210880 |

结论：

- 4-Leader reassignment 将 Top-23 AP@0.5 从 `0.157367` 提升到 `0.412919`，说明原 Top-23 的主要问题之一是 per-area packet 过多且偏离 checkpoint 的 `ego + limited CAVs` 语义。
- 5-Leader 的 AP@0.5 `0.409177` 与 4-Leader `0.412919` 基本持平，AP@0.7 略高，但没有解决 Top-23 质量差距。
- 6-Leader 的 AP@0.5 进一步升至 `0.452508`，同时第一跳 member-to-leader 上传降至 `154.50 KB/frame`；在当前启发式重分配下，它优于 4/5-Leader。
- 7-Leader 是 4-13 sweep 中 AP@0.5 最高点，达到 `0.484321`；继续增加到 8-13 后 AP@0.5 下降。
- 第一跳通信随 Leader 数增加下降：4-Leader `227.50 KB/frame`，7-Leader `132.88 KB/frame`，13-Leader `56.06 KB/frame`；第二跳 feature 仍基本稳定在 `24-25 Mbps`。
- checkpoint/YAML 严格声明的 total CAV cap 是 `5`，即保守 Leader cap 为 `4`；但 synthetic Where2comm fusion probe 证明当前 runtime 没有 5-CAV shape 硬上限，真实三尺度 feature 在当前 CUDA 环境下可跑到 total CAV `232`，`234` 开始 OOM。论文主实验不应使用 runtime OOM 上限，应把 7-Leader 作为当前 Top-23 的实用 sweep 上限/默认候选。

### 2026-07-22：All-20-CAV Point Cloud To RSU

本次结果回答“20 辆 CAV 的点云全部传到 RSU 后能得到多少 AP”。当前 `lgcp_carla` 数据为 `20 CAV + 1 RSU + 80 background vehicles`。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_where2comm_all20cav_to_rsu_11f_objectness_thr005
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_where2comm_all20cav_to_rsu_11f_centralized_raw_thr005
```

| Setting | Scope | AP@0.3 | AP@0.5 | AP@0.7 | GT | Raw upload KB/frame | Feature Mbps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 CAV per-CAV Where2comm | full scene | 0.590709 | 0.358292 | 0.021621 | 542 | 1603.89 | 512.565848 |
| 20 CAV centralized raw | full scene | 0.651840 | 0.487223 | 0.080560 | 542 | 1603.89 | 0.000000 |
| LGCP 7-Leader feature packet | planned areas | 0.669542 | 0.484321 | 0.026363 | 290 | 132.88 member upload | 24.756596 |

结论：

- 直接把 20 个 CAV 作为 20 个 Where2comm agents 融合，AP@0.5 只有 `0.358292`，说明更多 agent 并不自动更好。
- 更合理的 raw centralized upper bound 是先在 RSU 合并 20 CAV 点云再检测，AP@0.5 为 `0.487223`。
- LGCP 7-Leader 的 AP@0.5 `0.484321` 几乎追平 centralized raw upper bound，但第一跳点云上传约为 `132.88 KB/frame`，只有 all-raw upload `1603.89 KB/frame` 的约 `8.29%`。

### 2026-07-22：Attentive Early Centralized Raw Upper Bound

上面的 `20 CAV centralized raw` 使用的是 Where2comm checkpoint 的检测 route。为确认 SGCP 中 attentive checkpoint 移植的 early-fusion 权重是否更适合 centralized raw，本次新增同场景、同 20 CAV、同 RSU reference 的对照。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_attentive_early_all20cav_to_rsu_11f_centralized_raw_thr005
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_attentive_early_all20cav_to_rsu_11f_centralized_raw_thr020
```

| Setting | Detector checkpoint | Threshold | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples | Raw upload KB/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 CAV centralized raw | SGCP attentive-derived early | 0.05 | 0.799015 | 0.746692 | 0.424014 | 542 | 617 | 1603.89 |
| 20 CAV centralized raw | SGCP attentive-derived early | 0.20 | 0.816923 | 0.779641 | 0.470207 | 542 | 463 | 1603.89 |
| 20 CAV centralized raw | Where2comm route | 0.05 | 0.651840 | 0.487223 | 0.080560 | 542 | 798 | 1603.89 |
| LGCP 7-Leader feature packet | Where2comm leader packet | 0.05 | 0.669542 | 0.484321 | 0.026363 | 290 planned-area | 289 | 132.88 member upload |

结论：

- 20 CAV centralized raw 之前确实用的是 Where2comm checkpoint；SGCP attentive early 权重不是同一个 detector。
- attentive early detector 在当前 LGCP 11 帧场景上给出更合理的 all-raw centralized upper bound：AP@0.5 `0.779641`、AP@0.7 `0.470207`。
- 这会改变后续表述：LGCP 7-Leader 不能再被说成“几乎追平 all-raw centralized upper bound”；它只能说是接近 Where2comm 同检测 route 的 centralized raw 诊断。真正的 all-raw early-fusion 上界仍显著更高，但通信成本约 `1603.89 KB/frame`。

### 2026-07-22：SGCP Attentive-Derived Leader BEV To RSU Fusion

本次把 SGCP `pointpillar_early_from_attentive_weights/latest.pth` 放入 intermediate attentive model definition 中使用。由于 `AttFusion` 无可学习参数，该 checkpoint 与 `opencood/logs/pointpillar_attentive_fusion/latest.pth` 的 142 个 tensor 完全一致；差别在于本次显式以 `point_pillar_intermediate` / `AttBEVBackbone` 运行 `leader BEV feature -> RSU AttFusion -> detection`。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/model_dirs/pointpillar_intermediate_from_sgcp_attentive_early
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_sgcp_attentive_intermediate_rsu_bev_top5_11f_z2_thr005
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_sgcp_attentive_intermediate_rsu_bev_top23_11f_z2_thr005
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_sgcp_attentive_intermediate_rsu_bev_top23_7leaders_11f_z2_thr005_leaderpkt
```

| Setting | Packet granularity | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples | Leader packets/frame | Member upload KB/frame | Sparse feature KB/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5 first 5 areas | area | 0.940917 | 0.828329 | 0.534530 | 93 | 95 | 3.73 | 4.95 | 83.00 |
| Top-23 original assignment | area | 0.134185 | 0.134185 | 0.102179 | 313 | 42 | 17.36 | 26.81 | 340.80 |
| Top-23 7-Leader reassignment | leader | 0.663529 | 0.556226 | 0.252941 | 313 | 223 | 5.91 | 88.42 | 339.25 |
| Where2comm Top-23 7-Leader | leader | 0.669542 | 0.484321 | 0.026363 | 290 planned-area | 289 | 5.91 | 132.88 | feature Mbps 24.756596 |

结论：

- SGCP attentive-derived checkpoint 可以用于 leader BEV feature -> RSU attentive fusion，但不是通过拆出融合权重，而是通过 intermediate model definition 激活无参数 `AttFusion`。
- Top-5 的 AP@0.5 `0.828329` / AP@0.7 `0.534530` 是当前最强的 model-level hierarchy positive signal。
- Top-23 不能直接用 per-area packets；改成 7-Leader leader packets 后 AP@0.5 恢复到 `0.556226`，且 AP@0.7 `0.252941` 明显高于 Where2comm 7-Leader。
- Dense full BEV feature 仍不可作为通信量 claim；后续主线应围绕 sparse BEV cell packet、压缩 feature packet、metadata overhead 和多 seed validation。

### 2026-07-22：Small Town03 Diagnostic Dataset

为降低 100 车密集环岛对 model-level feature fusion 的难度，本次新增小规模 Town03 环岛数据集。它保留 LGCP 论文中的 RoI/grid 思路，但规模接近 OPV2V/V2XSet checkpoint 的训练分布。

场景与数据：

| Item | Value |
| --- | --- |
| Scenario | `lgcp_carla_small` |
| Config | `opencda/scenario_testing/config_yaml/lgcp_carla_small.yaml` |
| Dataset | `D:\Data\Carla\2026_07_22_20_04_41` |
| Agents | `8 CAV + 1 RSU` |
| Background | `28 vehicles` |
| Frames | `21` |
| RoI / grid | `120m x 60m` / `10m x 6m` |

LGCP Top-10 hierarchy plan:

| Frames | Areas/frame | Avg group size | Avg leader count | Avg leader max load | Byte proxy/frame |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 10.00 | 1.552381 | 5.714286 | 4.666667 | 77.88 KB |

SGCP attentive-derived leader-BEV route:

| Setting | Packet granularity | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples | Member upload KB/frame | Sparse feature KB/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small Town03 Top-10 | leader | 0.800311 | 0.755478 | 0.393690 | 294 | 247 | 12.67 | 234.69 |

结论：

- 小场景已完成 CARLA 导出、离线 area confidence、hierarchy plan 和 leader-BEV attentive evaluation。
- AP@0.5 `0.755478` / AP@0.7 `0.393690` 显示它适合作为 model-mechanism validation 和调参场景。
- 论文使用时应标注为 small-scale diagnostic，不替代 5-30 CAV co-simulation 的 scalability 证据。

### 2026-07-22：Ordinary Intersection Easy-Scene Attempts

| Dataset | Variant | Frames | Early model | AP@0.3 | AP@0.5 | AP@0.7 | Note |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `D:\Data\Carla\2026_07_22_20_26_17` | 4 CAV + 6 background, fixed spawn | 36 | default early, with RSU | 0.76 | 0.75 | 0.68 | Vehicle placement fixed; AP gate not met. |
| `D:\Data\Carla\2026_07_22_20_31_56` | 4 CAV + 0 background | 26 | default early, CAV only | 0.69 | 0.69 | 0.69 | Fewer vehicles did not improve AP. |
| `D:\Data\Carla\2026_07_22_20_37_27` | 2 CAV + 0 background | 21 | default early, CAV only | 0.52 | 0.52 | 0.52 | GT matched, but FP ranking suppresses AP. |
| `D:\Data\Carla\2026_07_22_20_44_26` | 2 CAV + 16 background, fixed spawn | 21 | default early, with RSU | 0.75 | 0.73 | 0.52 | Current ordinary-intersection config. |
| `D:\Data\Carla\2026_07_22_20_44_26` | 2 CAV + 16 background, fixed spawn | 21 | SGCP attentive-derived early, with RSU | 0.70 | 0.70 | 0.53 | Alternative early checkpoint did not improve AP. |
| `D:\Data\Carla\2026_07_22_22_00_04` | 10 CAV + 10 background, fixed spawn | 21 | default early, with RSU | 0.86 | 0.86 | 0.77 | Current best ordinary-intersection result. |
| `D:\Data\Carla\2026_07_22_22_00_04` | 10 CAV + 10 background, fixed spawn | 21 | SGCP attentive-derived early, with RSU | 0.86 | 0.86 | 0.78 | Slightly better AP@0.7. |
| `D:\Data\Carla\2026_07_22_22_00_04` | 10 CAV + 10 background, fixed spawn | 21 | default early, CAV only | 0.72 | 0.72 | 0.63 | RSU point cloud helps in this scene. |

Conclusion: ordinary-intersection placement is fixed by explicit spawn points, but AP@0.3 `>= 0.90` remains open. The next step is to search a cleaner non-roundabout junction / road geometry or calibrate detector postprocessing beyond score/NMS sweeps.

### 2026-07-22：Intersection10 LGCP Hierarchy Result

Baseline upper bound for this diagnostic scene:

| Scene | Method | Scope | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | --- | ---: | ---: | ---: |
| `D:\Data\Carla\2026_07_22_22_00_04` | Best full early fusion with RSU | full scene | 0.86 | 0.86 | 0.78 |

LGCP hierarchy result:

| Method | Scope | Areas/frame | Leaders/frame | Member upload KB/frame | Sparse BEV KB/frame | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP attentive leader-BEV -> RSU attentive | planned areas | 10 | 8 | 17.57 | 211.08 | 0.868668 | 0.797311 | 0.733363 | 252 | 294 |
| SGCP attentive leader-BEV -> RSU attentive | full scope | 10 | 8 | 17.57 | 211.08 | 0.813771 | 0.746923 | 0.687017 | 269 | 294 |

Limited-Leader sweep under the same Top-10 area budget:

| Max leaders | Scope | Leaders | Area load | Avg group | Member upload KB/frame | Sparse BEV KB/frame | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | planned areas | `6;9;8` | `6:5;9:4;8:1` | 2.10 | 68.28 | 218.57 | 0.809229 | 0.798743 | 0.660574 | 252 | 316 |
| 3 | full scope | `6;9;8` | `6:5;9:4;8:1` | 2.10 | 68.28 | 218.57 | 0.710024 | 0.700737 | 0.578433 | 269 | 340 |
| 4 | planned areas | `6;9;8;4` | `6:3;9:3;8:1;4:3` | 2.00 | 48.40 | 215.55 | 0.865886 | 0.798881 | 0.744764 | 252 | 294 |
| 4 | full scope | `6;9;8;4` | `6:3;9:3;8:1;4:3` | 2.00 | 48.40 | 215.55 | 0.811164 | 0.748394 | 0.697697 | 269 | 294 |
| 5 | planned areas | `6;9;8;4;3` | `6:3;9:2;8:1;4:2;3:2` | 1.80 | 36.37 | 214.58 | 0.865886 | 0.804755 | 0.752848 | 252 | 294 |
| 5 | full scope | `6;9;8;4;3` | `6:3;9:2;8:1;4:2;3:2` | 1.80 | 36.37 | 214.58 | 0.811164 | 0.753897 | 0.705270 | 269 | 294 |

K=5 checkpoint comparison:

| Checkpoint route | Scope | Mask / packet | Leaders | Member upload KB/frame | 2nd-hop feature KB/frame | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP attentive-derived | planned areas | leader sparse BEV | 5 | 36.37 | 214.58 | 0.865886 | 0.804755 | 0.752848 | 252 | 294 |
| SGCP attentive-derived | full scope | leader sparse BEV | 5 | 36.37 | 214.58 | 0.811164 | 0.753897 | 0.705270 | 269 | 294 |
| Where2comm | planned areas | `area_objectness+dilation1`, leader | 5 | 36.37 | 184.12 | 0.759775 | 0.743407 | 0.381554 | 252 | 344 |
| Where2comm | full scope | `area_objectness+dilation1`, leader | 5 | 36.37 | 184.12 | 0.643896 | 0.629983 | 0.323233 | 269 | 386 |

Interpretation:

- The full-scene upper bound is still the best early-fusion result, `0.86/0.86/0.78`.
- The planned-area AP is not directly comparable with full-scene AP because it evaluates only LGCP-selected areas.
- Full-scope LGCP retains most of the scene-level upper-bound AP, while using the local-to-global leader packet route.
- Limiting to 4 or 5 Leaders removes the earlier 8-Leader over-partitioning concern without hurting full-scope AP. K=5 is the current preferred setting under the `<=5` constraint; K=4 is the conservative checkpoint-friendly comparison point.
- Where2comm uses fewer second-hop feature bytes than SGCP attentive-derived sparse BEV on K=5 (`184.12` vs `214.58 KB/frame`), but its high-IoU AP drops sharply. It is useful as a communication-aware checkpoint baseline, not the current main-quality route.
- The remaining bottleneck is communication: SGCP attentive-derived sparse BEV feature upload is still about `214-219 KB/frame`, so compression / area budget / feature-cell selection are now the right next axes.

Coarse-area rerun:

| Setting | Total areas | Area size | Route | Scope | Member upload KB/frame | 2nd-hop feature KB/frame | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Old Top-10 selected from fine grid | 101 active / 108 theoretical | `10m x 6m` | SGCP attentive-derived K=5 | full scope | 36.37 | 214.58 | 0.811164 | 0.753897 | 0.705270 |
| Old Top-10 selected from fine grid | 101 active / 108 theoretical | `10m x 6m` | Where2comm K=5 | full scope | 36.37 | 184.12 | 0.643896 | 0.629983 | 0.323233 |
| New all-area coarse grid | 9 active / 9 theoretical | `30m x 24m` | SGCP attentive-derived K=5 | full scope | 94.60 | 916.46 | 0.607293 | 0.599637 | 0.572605 |
| New all-area coarse grid | 9 active / 9 theoretical | `30m x 24m` | Where2comm K=5 `dilation1` | full scope | 94.60 | 382.02 | 0.470654 | 0.470654 | 0.358164 |
| New all-area coarse grid | 9 active / 9 theoretical | `30m x 24m` | Where2comm K=5 `dilation0` | planned areas | 94.60 | 359.56 | 0.480319 | 0.480319 | 0.371061 |

Interpretation:

- The coarse grid fixes the semantic mismatch: the scene now has about ten LGCP areas in total (`9`), not Top-10 selected from over one hundred small cells.
- It increases first-hop raw point slices as expected (`36.37 -> 94.60 KB/frame`), but current second-hop features are still larger than first-hop raw slices.
- Where2comm remains more communication efficient than SGCP attentive-derived sparse BEV on coarse areas, but its AP is lower. This suggests the next communication fix should be feature-cell budget / thresholding / quantization, not merely larger areas.

Where2comm size-accounting interpretation:

| Payload type | Current formula / source | Typical size in intersection10 K=5 |
| --- | --- | ---: |
| Raw point slice | `points x 4 float32` | fine `36.37 KB/frame`, coarse `94.60 KB/frame` |
| SGCP attentive dense scatter BEV | full `leader x 64 x 200 x 704` feature payload | `85.94 MiB/frame` |
| Where2comm dense multiscale feature | full `64x96x352 + 128x48x176 + 256x24x88`, 16-bit values | `7392 KB/agent`, `36960 KB/5 agents` |
| Where2comm selected sparse feature | selected BEV cells after objectness mask | fine `184.12 KB/frame`, coarse `359.56-382.02 KB/frame` |

Break-even against raw point slices:

| Case | Raw points/frame | Selected feature KB/frame | Break-even raw points | Current relation |
| --- | ---: | ---: | ---: | --- |
| fine Top-10 K=5 Where2comm | 2328 | 184.12 | 11784 | raw is smaller |
| coarse 9-area K=5 Where2comm dilation1 | 6055 | 382.02 | 24449 | raw is smaller |
| coarse 9-area K=5 Where2comm dilation0 | 6055 | 359.56 | 23012 | raw is smaller |

Where2comm is therefore highly efficient relative to dense BEV feature transmission, retaining only about `0.5%-1.0%` of the full multiscale feature payload in these runs. It is not automatically smaller than raw area point slices. For the network-paper argument, this should be written as a conditional communication trade-off: sparse intermediate features dominate when the selected area contains enough raw points or when the feature mask/quantization budget is sufficiently tight; sparse raw point slices dominate in low-density areas.

Two-hop Where2comm intermediate-feature diagnostic:

| Route | Scope | Areas/frame | Leader packets/frame | First hop Mbps | Second hop Mbps | Total Mbps | Raw-equivalent member KB/frame | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-hop Where2comm, member feature -> leader feature -> RSU feature | planned areas | 5 | 4 | 7.617585 | 0.290621 | 7.908206 | 19.59 | 0.000000 | 0.000000 | 0.000000 | 210 | 0 |

Diagnostic controls:

- Same first frame / Top-10 one-hop Where2comm leader-feature route gives AP@0.3/0.5/0.7 `0.750000/0.657407/0.564815`, so the checkpoint and detection head are not globally broken.
- Two-hop Top-10 area-packet, two-hop Top-10 leader-packet, and leader-packet full-mask / low-threshold first-frame controls all failed to recover useful AP.
- The likely reason is distribution shift: the output of a Where2comm fusion module is not trained to become a normal per-CAV feature packet for a second Where2comm fusion pass. Two-hop intermediate feature sharing therefore needs a trained hierarchy adapter or dedicated leader/RSU fusion calibration before it can be a performance route.

Direct fused-feature aggregation at the RSU:

| Route | Scope | Areas/frame | Leader packets/frame | RSU fusion | First hop Mbps | Second hop Mbps | Total Mbps | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-hop fused feature, Top-10 first frame | planned areas | 10 | 5 | direct mean | 11.991040 | 3027.763200 | 3039.754240 | 0.000000 | 0.000000 | 0.000000 | 12 | 0 |
| Two-hop fused feature, Top-10 first frame | planned areas | 10 | 5 | direct max | 11.991040 | 3027.763200 | 3039.754240 | 0.000000 | 0.000000 | 0.000000 | 12 | 0 |
| Two-hop fused feature, Top-10 first frame | planned areas | 10 | 5 | direct attention | 11.991040 | 3027.763200 | 3039.754240 | 0.000000 | 0.000000 | 0.000000 | 12 | 0 |
| Two-hop fused feature, Top-10 first frame | planned areas | 10 | 5 | direct mean, threshold 0.001 | 11.991040 | 3027.763200 | 3039.754240 | 0.000000 | 0.000000 | 0.000000 | 12 | 1 |

This confirms that direct aggregation over first-hop fused features is implementable, but it does not solve the current route. Without a second-hop selector/compressor, dense fused-feature upload is about `3027.76 Mbps` in the first-frame diagnostic; without a trained RSU fused-feature adapter, AP remains zero.

Full-BEV first-hop feature diagnostic:

| Route | Scope | Areas/frame | Packets/frame | First-hop input | Second-hop fusion | Threshold | First hop Mbps | Second hop Mbps | Total Mbps | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-hop full-BEV mask, Top-10 first frame | planned areas | 10 | 5 leaders | full point cloud -> full BEV, area mask communication | Where2comm | 0.05 | 11.069440 | 3.512320 | 14.581760 | 0.000000 | 0.000000 | 0.000000 | 12 | 0 |
| Two-hop full-BEV mask, Top-10 first frame | planned areas | 10 | 5 leaders | full point cloud -> full BEV, area mask communication | Where2comm | 0.001 | 11.069440 | 3.512320 | 14.581760 | 0.000000 | 0.000000 | 0.000000 | 12 | 9 |
| Two-hop full-BEV mask, Top-5 21 frames | planned areas | 5 | 4 leaders | full point cloud -> full BEV, area mask communication | Where2comm | 0.001 | 7.112899 | 0.411550 | 7.524450 | 0.000000 | 0.000000 | 0.000000 | 210 | 21 |
| Two-hop full-BEV mask, Top-10 first frame | planned areas | 10 | 10 area packets | full point cloud -> full BEV, area mask communication | Where2comm | 0.001 | 11.069440 | 3.051520 | 14.120960 | 0.000000 | 0.000000 | 0.000000 | 12 | 4 |
| Two-hop full-BEV mask, Top-10 first frame | full scope | 10 | 10 area packets | full point cloud -> full BEV, area mask communication | Where2comm | 0.001 | 11.069440 | 3.051520 | 14.120960 | 0.000000 | 0.000000 | 0.000000 | 13 | 68 |

This variant is closer to the original Where2comm input distribution than point-crop encoding because every member uses its full point cloud to build BEV features, and LGCP areas only mask transmitted feature cells. It still fails to produce non-zero AP when the leader fused feature is passed into a second Where2comm stage. The full-scope row shows that the issue is not caused by planned-area filtering: even with 68 predicted boxes in the whole scene, none match GT at AP@0.3.

First-hop leader-side AP diagnostic for the Top-5 / 21-frame full-BEV route:

| Stage | Eval unit | Pred samples | GT boxes | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| First hop after leader Where2comm fusion | area-level leader reference | 356 | 147 | 0.470496 | 0.470496 | 0.470496 |
| Second hop after RSU Where2comm fusion | planned-area RSU reference | 21 | 210 | 0.000000 | 0.000000 | 0.000000 |

This isolates the failure point: first-hop leader fused features still support non-zero local area detection, while the second-hop RSU Where2comm stage collapses useful alignment / distribution for detection.

Coordinate-reference isolation for the same Top-10 first frame:

| Route | First-hop feature reference | Query | Pairwise mode | Pred | GT | AP@0.3 | AP@0.5 | AP@0.7 | Total Mbps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full-BEV leader packet | leader coordinate | zero | normal | 2 | 12 | 0.000000 | 0.000000 | 0.000000 | 14.581760 |
| full-BEV leader packet | leader coordinate | zero | inverse | 1 | 12 | 0.000000 | 0.000000 | 0.000000 | 14.981120 |
| full-BEV reference diagnostic | RSU/reference coordinate | zero | normal | 3 | 12 | 0.166667 | 0.166667 | 0.166667 | 28.979199 |

Interpretation:

- Reversing the pairwise transform direction does not recover AP, so the failure is not just a simple affine convention swap.
- Forcing first-hop packet features into the RSU/reference BEV coordinate frame recovers non-zero AP on the same frame. This supports the diagnosis that leader-local fused features cannot be consumed by the RSU second-hop Where2comm stage unless they are explicitly warped/generated into a common RSU/global reference.
- The recovered AP is still low, so coordinate alignment is necessary but not sufficient. The remaining gap is consistent with fused-feature distribution shift and an untrained second-hop selector / adapter / detection-head interface.

### 2026-07-18：Nearest vs Bilinear Coordinate-Warp AP Probe

本次结果用于判断当前 PointPillar neural feature hierarchy 是否能直接形成论文级 AP。实验复用 Top-23 首帧 leader-local feature slices，以 CAV 1 为 reference canvas，比较 nearest 与 bilinear 的 `reference -> world -> leader` coordinate warp。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_ap_probe_area23_1f_ref1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_bilinear_ap_probe_area23_1f_ref1
```

| Sampling | Sample ratio | Coverage ratio | Head score max | Pred boxes | GT boxes | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest | 1.000000 | 0.060724 | 0.893363 | 30 | 16 | 0.010000 | 0.010000 | 0.000000 |
| Bilinear | 0.998363 | 0.060625 | 0.815208 | 29 | 16 | 0.024457 | 0.011364 | 0.003472 |

结论：

- Bilinear 只带来极小 AP 改善，仍不能支撑 model-level LGCP AP claim。
- 当前问题不只是采样方式，而是跨 leader feature slice 的裁剪、均值融合、重投影 canvas 与预训练 PointPillar detection head 缺少校准。
- 论文近期可用口径应是 feature-level data-path / coverage / byte proxy，加上 box-level hierarchy late-fusion 的真实 OpenCOOD model-calling ablation；若要报告 neural feature AP，需要 affine warp calibration、feature normalization 或 retrained aggregation head。

### 2026-07-18：Neural Feature Proxy Summary

本次结果将 raw-slice reference、PointPillar feature crop、leader feature fusion、RSU canvas、nearest/bilinear coordinate warp 和 AP boundary 汇总到同一表，避免把低 AP 或未优化 feature bytes 误写成论文主张。

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_neural_feature_proxy_summary_area23
```

| Stage | Bytes / frame | Ratio vs raw member area23 | Ratio vs comm-aware area23 slice | Coverage | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw member area slice | 47,985.45 | 1.000000 | 0.189568 |  |  |  |
| Flat comm-aware area slice | 253,130.18 | 5.275144 | 1.000000 |  |  |  |
| PointPillar feature crop | 810,688.00 | 16.894453 | 3.202652 |  |  |  |
| Leader scatter fusion | 936,298.00 | 19.512121 | 3.698879 |  |  |  |
| RSU index canvas | 82,974.00 | 1.729149 | 0.327792 | 0.033161 |  |  |
| Coordinate warp nearest | 110,883.00 | 2.310763 | 0.438047 | 0.060724 | 0.010000 | 0.000000 |
| Coordinate warp bilinear | 149,700.00 | 3.119695 | 0.591395 | 0.060625 | 0.011364 | 0.003472 |

结论：

- 未优化 PointPillar feature crop 不比 raw area-slice 省通信；Top23 首帧压缩 feature crop 是 raw member area23 均值的 `16.89x`。
- RSU canvas / coordinate-warp canvas 较小，但它们是聚合后中间产物，不能当作 leader upload 负载。
- 当前 neural feature hierarchy 只能安全写作 data-path feasibility / coverage / byte boundary；感知质量主证据应继续使用 box-level hierarchy late-fusion 或等待校准 / 重训练后的 neural AP。
## 2026-07-28: OpenCOOD Original LGCP Reproduction

Goal: reproduce the original LGCP paper description on OPV2V with RSU/ego as the reference, area grouping by collaborative confidence gain, first-hop area intermediate-feature accounting, and second-hop perception-result upload plus RSU box NMS.

Implementation branch: `C:\Workspace\OpenCOOD`, branch `codex/lgcp-where2comm-area`.

Remote environment: `mindspore-186:/data1/wql/gzc/workspace/OpenCOOD`, conda env `opencood-gzc`, OPV2V test slice `start_frame=423`, `max_frames=20`, `max_cav_override=5`, `comm_range_override=200`.

New inference options:

```text
--lgcp_original
--lgcp_original_execution leader_box|accounting
--lgcp_original_delta_g
--lgcp_original_area_w_m 10
--lgcp_original_area_h_m 6
--lgcp_original_full_feature_mbit 2.16
--lgcp_original_box_payload_bytes 64
```

Two execution semantics:

- `leader_box`: physically executes leader-local intermediate fusion per selected group, filters leader boxes by assigned areas, and then performs RSU box NMS. This is closer to a literal executable hierarchy.
- `accounting`: keeps the original full intermediate model output for perception quality, while using LGCP area grouping to account first-hop area feature packets and second-hop detection-result packets. This matches the recovered original-code risk described by the original author: intermediate features are not actually spatially partitioned before inference; area partitioning is applied mainly to communication accounting and result grouping.

Current OPV2V 5-CAV slice results:

| Method | Execution | Delta_g | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Reduction vs edge-assisted full feature | Areas/frame | First-hop packets/frame | First-hop Mbps | Second-hop Mbps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Where2comm full intermediate | baseline | - | 0.906450 | 0.904430 | 0.851670 | 425.548300 | - | - | - | - | - |
| LGCP + Where2comm | accounting | 0.075 | 0.906450 | 0.904430 | 0.851670 | 4.112139 | 26.418x | 162 | 66 | 3.970059 | 0.142080 |
| LGCP + Where2comm | accounting | 0.100 | 0.906450 | 0.904430 | 0.851670 | 2.314377 | 43.721x | 162 | 39 | 2.172297 | 0.142080 |
| LGCP + Where2comm | accounting | 0.125 | 0.906450 | 0.904430 | 0.851670 | 1.370551 | 74.407x | 162 | 22 | 1.228471 | 0.142080 |
| LGCP + Where2comm | leader_box | 0.075 | 0.383235 | 0.371399 | 0.354552 | 4.161035 | 26.286x | 162 | 66 | 3.970059 | 0.190976 |
| LGCP + CoSDH | accounting | 0.075 | 0.842576 | 0.840305 | 0.775515 | 1.142095 | 177.806x | 88 | 8 | 1.009743 | 0.132352 |

Interpretation:

- The original-paper accounting route can reproduce the headline behavior on this slice: no AP loss relative to full intermediate inference, while `Delta_g=0.10` gives `43.721x` reduction, very close to the paper abstract's `44x` claim.
- The reduction is computed against the paper's edge-assisted paradigm: every CAV uploads a complete shared feature of `2.16 Mb` per frame. It is not a reduction against raw point cloud slices.
- Strict executable leader-local area perception is not yet strong enough: `leader_box` falls to AP@0.5 `0.371399`. Full-ROI and `Delta_g=0` diagnostics show the coordinate transform and subgroup inference are basically valid; the AP drop comes from fine-grained area assignment / area-box filtering and leader responsibility fragmentation.
- This creates a paper-writing boundary: if we choose the low-workload InfoCom submission path, the main table can use the `accounting` reproduction only if the manuscript honestly frames LGCP as a scheduling/accounting layer over existing intermediate perception, not as proof that spatially partitioned neural features were truly transmitted and fused.

Useful remote result files:

```text
opencood/logs/where2comm_10e/comm_stats_intermediate_baseline_same_slice_20f_epoch6.json
opencood/logs/where2comm_10e/comm_stats_intermediate_lgcp_original_accounting_dg075_10x6_20f_epoch6.json
opencood/logs/where2comm_10e/comm_stats_intermediate_lgcp_original_accounting_dg100_10x6_20f_epoch6.json
opencood/logs/where2comm_10e/comm_stats_intermediate_lgcp_original_accounting_dg125_10x6_20f_epoch6.json
opencood/logs/where2comm_10e/comm_stats_intermediate_lgcp_original_psm_dg075_10x6_20f_epoch6.json
opencood/logs/opv2v_cosdh_2026_05_15_11_17_12/comm_stats_intermediate_lgcp_original_accounting_dg075_10x6_20f_epoch21.json
```

### 2026-07-28 Follow-up: Leader-box Implementation Audit

The initial `leader_box` result was too low partly because of an implementation bug. `pairwise_t_matrix[i, j]` is generated as the transform from CAV `i` to CAV `j`, but the first implementation projected non-ego leader boxes with `[0, leader]` instead of `[leader, 0]`. The same direction issue also affected single-CAV box projection and PSM area-confidence projection. This has been fixed in OpenCOOD commit `2ff2ba31`.

Additional diagnostic switches:

```text
--lgcp_original_force_single_group
--lgcp_original_force_leader_idx
--lgcp_original_no_area_box_filter
```

Updated Where2comm OPV2V 5-CAV diagnostics:

| Diagnostic | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Areas/frame | First-hop packets/frame | Uploaded boxes/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full intermediate baseline | 0.906450 | 0.904430 | 0.851670 | 425.548300 | - | - | - |
| Accounting | 0.906450 | 0.904430 | 0.851670 | 4.112139 | 162 | 66 | 26 |
| Ego-coordinate area box filter | 0.906450 | 0.904430 | 0.851670 | 4.112139 | 162 | 66 | 26 |
| Leader-box, all CAVs, leader 0, no area filter | 0.906450 | 0.904430 | 0.851670 | 2.847713 | 176 | 42 | 26 |
| Leader-box, all CAVs, leader 1, no area filter, fixed direction | 0.835573 | 0.830341 | 0.776291 | 4.402825 | 113 | 63 | 26 |
| Leader-box, normal groups, fixed direction | 0.603480 | 0.594718 | 0.555254 | 4.355977 | 113 | 63 | 22 |
| Leader-box, normal groups, fixed direction, no area filter | 0.904729 | 0.902794 | 0.852588 | 4.842377 | 113 | 63 | 120 |
| Leader-box, all CAVs, leader 0, area filter | 0.906450 | 0.904430 | 0.851670 | 4.390793 | 113 | 63 | 26 |

Conclusions:

- AP statistics and full-batch baseline evaluation are correct.
- The CAV-subset construction is correct when the leader is CAV 0: all-CAV single-group leader-box reproduces the baseline exactly.
- Non-ego leader projection was previously wrong; after fixing transform direction, leader 1 all-CAV AP recovers from zero to `0.835573/0.830341/0.776291`. The remaining gap is consistent with changing the fusion reference/grid from ego to another CAV.
- The main remaining AP loss in normal `leader_box` is the hard per-leader area box filter. With normal multi-leader groups but no area filter, AP returns to `0.904729/0.902794/0.852588`; with area filter, AP is `0.603480/0.594718/0.555254`.
- Therefore the strictest leader responsibility rule is brittle for the current pretrained checkpoint. A more defensible executable variant is `leader_box + RSU NMS without per-leader area box filtering`, while communication accounting can remain area-based.

### 2026-07-28 Follow-up: Area-filter TP/FP Diagnosis and Fix

The strict area filter used center-cell ownership only. Because object boxes have spatial extent and OpenCOOD detections from different leaders may land in adjacent area cells, this hard boundary removed many true positives before the final RSU NMS. A one-cell area guard band was added through:

```text
--lgcp_original_area_filter_margin
```

Default is now `1` in OpenCOOD commit `950efbdf`.

TP/FP comparison on the same Where2comm OPV2V 5-CAV / 20-frame slice:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | IoU | TP | FP | GT | Pred | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict filter, margin 0 | 0.602126 | 0.593372 | 0.549994 | 0.5 | 361 | 57 | 600 | 418 | 0.601667 | 0.863636 |
| Guard-band filter, margin 1 | 0.901046 | 0.901046 | 0.845896 | 0.5 | 547 | 192 | 600 | 739 | 0.911667 | 0.740189 |
| Guard-band filter, margin 2 | 0.905105 | 0.903159 | 0.852491 | 0.5 | 551 | 293 | 600 | 844 | 0.918333 | 0.652844 |
| No per-leader filter | 0.904767 | 0.902836 | 0.852616 | 0.5 | 551 | 323 | 600 | 874 | 0.918333 | 0.630435 |

At IoU 0.7:

| Setting | TP | FP | GT | Pred | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict filter, margin 0 | 337 | 81 | 600 | 418 | 0.561667 | 0.806220 |
| Guard-band filter, margin 1 | 517 | 222 | 600 | 739 | 0.861667 | 0.699594 |
| Guard-band filter, margin 2 | 522 | 322 | 600 | 844 | 0.870000 | 0.618483 |
| No per-leader filter | 522 | 352 | 600 | 874 | 0.870000 | 0.597254 |

Conclusion:

- The AP drop was recall-driven. Strict filtering kept precision high but capped AP@0.5 recall at `0.601667`.
- A one-cell guard band recovers almost all missing TP while keeping fewer FP than no filter.
- This matches the intended LGCP semantics better: leader responsibility areas should form an overlapping or guard-banded cover of the scene before RSU aggregation, not a brittle non-overlapping center-cell ownership test.

### 2026-07-28 Follow-up: Multi-leader and Area-mask Results

All runs below use the same OPV2V 5-CAV slice: `start_frame=423`, `max_frames=20`, `comm_range_override=200`, Where2comm epoch 6 unless otherwise noted.

Multi-leader `leader_box` with default one-cell guard band:

| Delta_g | Leaders in first frame | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Edge-assisted reduction | Areas/frame | First-hop packets/frame | First-hop Mbps | Second-hop Mbps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.050 | 5 | 0.896900 | 0.894825 | 0.836048 | 6.765744 | 16.546x | 113 | 102 | 6.364080 | 0.401664 |
| 0.075 | 5 | 0.901734 | 0.901734 | 0.846506 | 4.640137 | 25.998x | 113 | 63 | 4.248713 | 0.391424 |
| 0.100 | 5 | 0.905693 | 0.902392 | 0.854416 | 2.579593 | 43.106x | 113 | 36 | 2.172297 | 0.407296 |
| 0.125 | 5 | 0.898105 | 0.892381 | 0.856666 | 1.493406 | 77.076x | 113 | 18 | 1.087646 | 0.405760 |

Interpretation:

- This is no longer a single-leader diagnostic. The first frame uses five leaders in all four runs.
- Increasing `Delta_g` reduces first-hop area packets and communication as expected. AP stays near the full Where2comm baseline `0.906450/0.904430/0.851670`.
- `Delta_g=0.10` is the best current paper-style operating point: AP@0.5 `0.902392` and `43.106x` edge-assisted reduction, close to the original abstract's `44x` communication claim.

Where2comm area mask, 3x4 BEV area grid, `constraint` mode:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Avg comm rate | Mbps | Reduction vs no area mask |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No area mask | 0.906450 | 0.904430 | 0.851670 | 0.191447 | 425.548300 | 1.00x |
| Top-1 area | 0.849722 | 0.848069 | 0.807164 | 0.034939 | 73.044482 | 5.83x |
| Top-2 areas | 0.893243 | 0.891220 | 0.833890 | 0.069437 | 146.446853 | 2.91x |
| Top-3 areas | 0.904974 | 0.902937 | 0.848870 | 0.097987 | 208.649222 | 2.04x |
| Top-4 areas | 0.904998 | 0.902964 | 0.850613 | 0.121738 | 261.686792 | 1.63x |
| Top-6 areas | 0.905343 | 0.903313 | 0.850865 | 0.159539 | 348.600844 | 1.22x |

CoSDH area mask, same 3x4 BEV area grid and `constraint` mode, epoch 21:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Avg comm rate | Mbps | Reduction vs no area mask |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No area mask | 0.842583 | 0.840308 | 0.775534 | 0.048765 | 7.403200 | 1.00x |
| Top-1 area | 0.814698 | 0.812288 | 0.718339 | 0.010475 | 1.522016 | 4.86x |
| Top-2 areas | 0.832293 | 0.829977 | 0.747939 | 0.019464 | 2.834528 | 2.61x |
| Top-3 areas | 0.837550 | 0.835279 | 0.766961 | 0.027341 | 4.011488 | 1.85x |
| Top-4 areas | 0.839244 | 0.836978 | 0.772112 | 0.034180 | 5.075936 | 1.46x |
| Top-6 areas | 0.840916 | 0.838647 | 0.773973 | 0.042234 | 6.322016 | 1.17x |

Interpretation:

- The area-mask mechanism works cleanly for Where2comm: Top-3 keeps nearly the same AP while halving communication.
- The same mask interface also works for another PointPillar BEV feature model, CoSDH. Its absolute communication is already much lower because the checkpoint uses `compression=16` and sparse selection, so the extra area-mask gain is smaller.
- These results support two paper routes: original LGCP `leader_box` with box-result upload for the 44x-style claim, and model-level area-mask selective BEV feature communication as the stronger neural-feature extension.

### 2026-07-28 Follow-up: 167-frame OPV2V 5-CAV Validation

The 20-frame smoke results above were expanded to the full OPV2V test scenario segment `2021_08_22_07_52_02`, dataset indices `423-589` (`167` frames), with `comm_range_override=200`.

Original-style multi-leader `leader_box`, Where2comm epoch 6:

| Method | Delta_g | Leaders in first frame | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Edge-assisted reduction | First-hop Mbps | Second-hop Mbps | IoU0.5 TP / FP / GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LGCP leader_box, guard-band margin 1 | 0.100 | 5 | 0.889357 | 0.886544 | 0.827220 | 2.200203 | 43.106x | 1.811757 | 0.388446 | 4236 / 2355 / 4623 |

Area-mask validation on the same 167-frame segment:

| Model | Setting | AP@0.3 | AP@0.5 | AP@0.7 | Avg comm rate | Mbps | Reduction vs model baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Where2comm | No area mask | 0.874003 | 0.872776 | 0.821696 | 0.139849 | 312.840226 | 1.00x |
| Where2comm | 3x4 Top-3 area mask | 0.846327 | 0.844633 | 0.789813 | 0.075207 | 160.382871 | 1.95x |
| CoSDH | No area mask | 0.869307 | 0.856879 | 0.759882 | 0.048852 | 7.416263 | 1.00x |
| CoSDH | 3x4 Top-3 area mask | 0.861586 | 0.848816 | 0.743772 | 0.031077 | 4.633623 | 1.60x |

Additional candidate audit:

- A local CoAlign checkpoint from `D:\Files\Recent\checkpoints\coAlign` was copied to the remote workspace and verified by SHA256.
- Its config requires missing stage-1 box-alignment pre-calculation: `opencood/logs/coalign_precalc/opv2v/test/stage1_boxes.json`.
- Running the checkpoint without `box_align` pre-calc, even with clean pose and `score_thresh=0.01`, produced zero predictions on the 5-frame smoke test. Therefore this checkpoint is not a valid current third-model AP/communication result.
- `point_pillar_infocom` checkpoints are available and are PointPillar BEV feature models with their own spatial communication path, but their mask is not yet an LGCP area-tile mask. They are good follow-up candidates only after a specific area-mask adapter is added.

Interpretation:

- The multi-leader box-result route remains numerically normal on 167 frames and preserves the paper-style `~44x` communication reduction.
- Where2comm area masking still reduces feature communication substantially on 167 frames, but AP loss is larger than on the 20-frame smoke slice. This is useful as a real Pareto point rather than a cherry-picked no-loss result.
- CoSDH confirms the mask mechanism extends beyond Where2comm. Its AP drop is small and communication is already low, so the extra reduction is moderate.

### 2026-07-29: Applying Area Mask Inside `leader_box`

User question: apply the same `3x4` Top-3 area mask to `leader_box, Delta_g=0.10` and observe AP / communication.

Command change:

```text
--lgcp_original
--lgcp_original_execution leader_box
--lgcp_original_delta_g 0.10
--lgcp_original_area_filter_margin 1
--lgcp_area_mask
--lgcp_area_grid_h 3
--lgcp_area_grid_w 4
--lgcp_area_topk 3
--lgcp_area_score mean
--lgcp_area_mode constraint
```

20-frame OPV2V 5-CAV slice:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Mbps recorded by `leader_box` accounting | First-hop Mbps | Second-hop Mbps | IoU0.5 TP / FP / GT / Pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `leader_box`, no area mask | 0.905693 | 0.902392 | 0.854416 | 2.579593 | 2.172297 | 0.407296 | 550 / 248 / 600 / 798 |
| `leader_box`, Where2comm area mask Top-3 | 0.907384 | 0.904078 | 0.854991 | 2.575241 | 2.172297 | 0.402944 | 551 / 238 / 600 / 789 |

167-frame OPV2V 5-CAV segment:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Mbps recorded by `leader_box` accounting | First-hop Mbps | Second-hop Mbps | IoU0.5 TP / FP / GT / Pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `leader_box`, no area mask | 0.889357 | 0.886544 | 0.827220 | 2.200203 | 1.811757 | 0.388446 | 4236 / 2355 / 4623 / 6591 |
| `leader_box`, Where2comm area mask Top-3 | 0.877374 | 0.874637 | 0.816406 | 2.179447 | 1.811757 | 0.367690 | 4173 / 2204 / 4623 / 6377 |

Interpretation:

- The mask is applied to the underlying Where2comm model during each leader-local intermediate fusion.
- AP is nearly unchanged on 20 frames and drops moderately on 167 frames: AP@0.5 `0.886544 -> 0.874637`.
- The recorded communication does not show a large first-hop reduction because `leader_box` currently reports the original LGCP communication accounting: first hop is still area-feature packet accounting from the paper, and second hop is uploaded detection boxes.
- Therefore the small communication change comes only from the number of uploaded leader boxes after NMS/filtering, not from Where2comm Top-3 feature-mask payload.
- If the paper needs the actual masked feature payload inside leader-local fusion, the next code change should aggregate each leader forward pass's `comm_mbps_meta` instead of using only `_lgcp_original_comm_meta`.

### 2026-07-29 修正：原文口径必须使用 `Delta_g` 选出的 LGCP group areas

用户指出：Top-K BEV area selection 不是 LGCP 原文机制，不能容忍“实际 mask”与
`Delta_g=0.10` 选出的 group areas 不一致。随后在 OpenCOOD 中新增
`--lgcp_original_area_feature_mask`，用于 `leader_box` 执行路径：

- 每一帧先按 LGCP 原文 collaborative confidence gain rule 生成 `area_groups`。
- 每个 leader 只为自己负责的 area 运行 leader-local fusion。
- 每个 member 的 `external_comm_mask` 只保留它在该 leader 下被 `Delta_g` 选中的那些 areas。
- leader 自身特征保持完整，因为它是本地计算，不计入 member-to-leader 通信。
- 该开关与 `--lgcp_area_mask` 互斥，防止 Where2comm Top-K 探索口径混入原文复现。

新的有效命令差异：

```text
--lgcp_original
--lgcp_original_execution leader_box
--lgcp_original_delta_g 0.10
--lgcp_original_area_filter_margin 1
--lgcp_original_area_feature_mask
```

20-frame OPV2V 5-CAV slice:

| Setting | Actual feature mask | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Edge-assisted reduction | First-hop Mbps | Second-hop Mbps | IoU0.5 TP / FP / GT / Pred |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `leader_box`, no actual feature mask | none; accounting only | 0.905693 | 0.902392 | 0.854416 | 2.579593 | 43.106x | 2.172297 | 0.407296 | 550 / 248 / 600 / 798 |
| `leader_box`, Where2comm Top-3 mask | Top-K exploration, not original LGCP | 0.907384 | 0.904078 | 0.854991 | 2.575241 | 43.106x | 2.172297 | 0.402944 | 551 / 238 / 600 / 789 |
| `leader_box`, LGCP group-area mask | `Delta_g=0.10` selected group areas | 0.711166 | 0.706329 | 0.636381 | 2.322569 | 47.794x | 2.172297 | 0.150272 | 428 / 64 / 600 / 492 |

167-frame OPV2V 5-CAV segment:

| Setting | Actual feature mask | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Edge-assisted reduction | First-hop Mbps | Second-hop Mbps | IoU0.5 TP / FP / GT / Pred |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `leader_box`, no actual feature mask | none; accounting only | 0.889357 | 0.886544 | 0.827220 | 2.200203 | 43.106x | 1.811757 | 0.388446 | 4236 / 2355 / 4623 / 6591 |
| `leader_box`, Where2comm Top-3 mask | Top-K exploration, not original LGCP | 0.877374 | 0.874637 | 0.816406 | 2.179447 | 43.106x | 1.811757 | 0.367690 | 4173 / 2204 / 4623 / 6377 |
| `leader_box`, LGCP group-area mask | `Delta_g=0.10` selected group areas | 0.795933 | 0.791619 | 0.694981 | 1.988045 | 47.794x | 1.811757 | 0.176287 | 3700 / 604 / 4623 / 4304 |

Mask diagnostic for the 167-frame strict LGCP group-area run:

| Metric | Value |
| --- | ---: |
| Mask source | `lgcp_original_delta_g_area_groups` |
| Average selected feature cells | 2804.31 |
| Average full non-leader cells | 368676.79 |
| Average feature-cell keep ratio | 0.7758% |
| First-frame leaders | 5 |

Interpretation:

- 原文口径现在已经统一：实际 feature mask 与通信统计都来自同一套
  `Delta_g=0.10` LGCP area/group selection；Top-K 结果只能保留为 Where2comm
  selective-feature 探索，不能作为原文 LGCP 机制复现证据。
- 严格对齐后，通信 reduction 可达到 `47.794x`，但 AP 明显下降，主要是召回下降：
  167 帧 IoU0.5 recall 为 `3700 / 4623 = 80.03%`，低于无实际 feature mask 的
  `4236 / 4623 = 91.63%`。
- 换言之，原文 area-feature 统计口径和真实神经特征裁剪一旦完全对齐，暴露出的核心问题是
  `10m x 6m` area packet 太窄，成员特征平均只保留约 `0.78%` 的 BEV cells。
- 论文若继续走“最少工作量修复原文硬伤”的路线，应避免把 Top-K 或 full-feature
  accounting 结果描述成真实 area-masked feature fusion；更稳妥的说法是：
  LGCP 的调度与通信统计可按 area/group 对齐，但严格空间裁剪的中期特征融合存在召回损失，
  需要 guard band、larger area、overlap 或 retraining 才能恢复检测质量。

### 2026-07-29 Follow-up: Larger Area and Overlap for Strict Group-area Mask

Goal: keep the original LGCP mechanism, i.e., feature masks are still generated
from `Delta_g=0.10` group areas, but test whether larger areas and small
overlap can recover the AP loss caused by strict `10m x 6m` feature slicing.

Implementation:

- Added `--lgcp_original_area_feature_overlap_w_m` and
  `--lgcp_original_area_feature_overlap_h_m` in OpenCOOD.
- The actual member feature mask expands each assigned area boundary by the
  configured overlap in meters.
- The first-hop area-feature accounting is also expanded by the same overlap:
  `(area_w + 2 overlap_w) * (area_h + 2 overlap_h)`, so implementation and
  communication statistics remain aligned.
- This is still not Top-K. The selected areas still come from LGCP
  `Delta_g=0.10` grouping.

20-frame OPV2V 5-CAV sweep:

| Area size | Overlap | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Reduction | First-hop Mbps | Second-hop Mbps | Mask keep ratio | Areas | First-hop packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10m x 6m` | `0m` | 0.711166 | 0.706329 | 0.636381 | 2.322569 | 47.794x | 2.172297 | 0.150272 | 0.7165% | 113 | 36 |
| `20m x 12m` | `0m` | 0.753144 | 0.750067 | 0.708645 | 5.991919 | 16.427x | 5.824751 | 0.167168 | 2.0117% | 52 | 27 |
| `20m x 12m` | `2m` | 0.761929 | 0.758884 | 0.715462 | 9.495730 | 10.328x | 9.319602 | 0.176128 | 3.1155% | 52 | 27 |
| `20m x 12m` | `4m` | 0.814971 | 0.809971 | 0.766587 | 13.784367 | 7.094x | 13.591087 | 0.193280 | 4.4147% | 52 | 27 |
| `20m x 12m` | `6m` | 0.827723 | 0.822795 | 0.779629 | 18.842725 | 5.184x | 18.639205 | 0.203520 | 5.9062% | 52 | 27 |
| `20m x 12m` | `8m` | 0.866437 | 0.864462 | 0.803893 | 24.689236 | 3.952x | 24.463956 | 0.225280 | 7.5613% | 52 | 27 |
| `30m x 18m` | `0m` | 0.750888 | 0.747976 | 0.692736 | 9.821836 | 10.974x | 9.627020 | 0.194816 | 3.7603% | 34 | 18 |
| `30m x 18m` | `3m` | 0.780450 | 0.777580 | 0.724284 | 15.611616 | 6.892x | 15.403232 | 0.208384 | 5.7474% | 34 | 18 |
| `30m x 18m` | `6m` | 0.810009 | 0.803496 | 0.752833 | 22.682694 | 4.731x | 22.463046 | 0.219648 | 8.0792% | 34 | 18 |

Best 167-frame validation point:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Reduction | First-hop Mbps | Second-hop Mbps | Mask keep ratio | IoU0.5 TP / FP / GT / Pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `10m x 6m`, no overlap | 0.795933 | 0.791619 | 0.694981 | 1.988045 | 47.794x | 1.811757 | 0.176287 | 0.7758% | 3700 / 604 / 4623 / 4304 |
| `20m x 12m`, `8m` overlap | 0.872071 | 0.868043 | 0.781997 | 19.907312 | 3.952x | 19.640612 | 0.266700 | 8.1619% | 4089 / 1097 / 4623 / 5186 |
| No actual feature mask | 0.889357 | 0.886544 | 0.827220 | 2.200203 | 43.106x | 1.811757 | 0.388446 | n/a | 4236 / 2355 / 4623 / 6591 |

Interpretation:

- Larger areas and overlap do rescue AP. On 167 frames, `20m x 12m + 8m overlap`
  recovers AP@0.5 from `0.791619` to `0.868043`, close to no-actual-mask
  `0.886544`.
- The recovery mainly comes from recall: IoU0.5 recall improves from
  `3700/4623 = 80.03%` to `4089/4623 = 88.45%`.
- The cost is large. The paper-style communication reduction drops from
  `47.794x` to `3.952x` because the effective feature packet area becomes
  `(20 + 16) * (12 + 16) = 1008 m^2`, much larger than the original
  `10 * 6 = 60 m^2`.
- Therefore this experiment answers the mechanism question clearly:
  area overlap can repair strict feature slicing, but it trades away most of
  the original `44x` communication advantage.
- Current paper implication: if we need to defend the original `44x` claim,
  strict real feature slicing is risky. If we instead present an honest
  implementation-performance trade-off, the best tested point is
  `20m x 12m + 8m overlap`, but it supports a moderate communication-reduction
  story rather than the original abstract-level reduction.

### 2026-07-29 Follow-up: Larger Area with Small Overlap Sweep

User focus: test more area sizes and overlaps, especially larger areas with
small overlap, and explicitly include `10m x 6m + 1m overlap`.

All runs still use the strict original mechanism:

- `Delta_g=0.10` selects LGCP group areas.
- Actual feature masks come from those group areas.
- No Top-K selection is used.
- Feature-mask overlap and communication accounting use the same meter-level
  expansion.

20-frame sweep:

| Area size | Overlap | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Reduction | Mask keep ratio | Areas | First-hop packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10m x 6m` | `0m` | 0.711166 | 0.706329 | 0.636381 | 2.322569 | 47.794x | 0.7165% | 113 | 36 |
| `10m x 6m` | `1m` | 0.731251 | 0.725809 | 0.663680 | 3.636955 | 30.213x | 1.1141% | 113 | 36 |
| `15m x 9m` | `0m` | 0.731939 | 0.728719 | 0.673358 | 4.039584 | 24.282x | 1.2642% | 70 | 32 |
| `15m x 9m` | `1m` | 0.752817 | 0.749612 | 0.693805 | 5.544026 | 17.692x | 1.7086% | 70 | 32 |
| `15m x 9m` | `2m` | 0.773223 | 0.771706 | 0.702051 | 7.276535 | 13.446x | 2.2234% | 70 | 32 |
| `20m x 12m` | `0m` | 0.753144 | 0.750067 | 0.708645 | 5.991919 | 16.427x | 2.0117% | 52 | 27 |
| `20m x 12m` | `1m` | 0.754954 | 0.751893 | 0.712172 | 7.646874 | 12.845x | 2.5231% | 52 | 27 |
| `20m x 12m` | `2m` | 0.761929 | 0.758884 | 0.715462 | 9.495730 | 10.328x | 3.1155% | 52 | 27 |
| `25m x 15m` | `0m` | 0.810998 | 0.809518 | 0.754794 | 7.279139 | 20.011x | 2.6432% | 34 | 14 |
| `25m x 15m` | `1m` | 0.817240 | 0.815762 | 0.757978 | 8.867838 | 16.434x | 3.1888% | 34 | 14 |
| `25m x 15m` | `2m` | 0.820025 | 0.818548 | 0.764200 | 10.605757 | 13.735x | 3.7615% | 34 | 14 |
| `25m x 15m` | `3m` | 0.823271 | 0.821802 | 0.768098 | 12.497248 | 11.642x | 4.3649% | 34 | 14 |
| `30m x 18m` | `0m` | 0.750888 | 0.747976 | 0.692736 | 9.821836 | 10.974x | 3.7603% | 34 | 18 |
| `30m x 18m` | `1m` | 0.753768 | 0.750917 | 0.699080 | 11.609225 | 9.287x | 4.3870% | 34 | 18 |
| `30m x 18m` | `2m` | 0.762773 | 0.759928 | 0.702585 | 13.537445 | 7.956x | 5.0453% | 34 | 18 |
| `35m x 21m` | `0m` | 0.821382 | 0.811700 | 0.753444 | 11.277793 | 14.093x | 4.4774% | 26 | 10 |
| `35m x 21m` | `1m` | 0.827331 | 0.817638 | 0.751528 | 13.019214 | 12.235x | 5.0961% | 26 | 10 |
| `35m x 21m` | `2m` | 0.836187 | 0.828029 | 0.755223 | 14.881255 | 10.730x | 5.7189% | 26 | 10 |
| `40m x 24m` | `0m` | 0.820531 | 0.812431 | 0.747371 | 12.918138 | 10.103x | 5.3859% | 21 | 11 |
| `40m x 24m` | `1m` | 0.820577 | 0.812496 | 0.747504 | 14.658884 | 8.903x | 5.9849% | 21 | 11 |

167-frame validation of representative small-overlap points:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Reduction | First-hop Mbps | Second-hop Mbps | Mask keep ratio | IoU0.5 TP / FP / GT / Pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `10m x 6m`, `0m` overlap | 0.795933 | 0.791619 | 0.694981 | 1.988045 | 47.794x | 1.811757 | 0.176287 | 0.7758% | 3700 / 604 / 4623 / 4304 |
| `10m x 6m`, `1m` overlap | 0.810602 | 0.806211 | 0.710632 | 3.087516 | 30.213x | 2.898812 | 0.188704 | 1.1928% | 3777 / 680 / 4623 / 4457 |
| `25m x 15m`, `1m` overlap | 0.855503 | 0.852906 | 0.753320 | 7.777682 | 16.434x | 7.535264 | 0.242418 | 3.7099% | 3988 / 773 / 4623 / 4761 |
| `25m x 15m`, `2m` overlap | 0.856875 | 0.854296 | 0.757346 | 9.291298 | 13.735x | 9.045600 | 0.245699 | 4.3600% | 4000 / 818 / 4623 / 4818 |
| `35m x 21m`, `2m` overlap | 0.866078 | 0.861531 | 0.767847 | 13.757980 | 10.730x | 13.493089 | 0.264891 | 6.6188% | 4029 / 827 / 4623 / 4856 |
| `20m x 12m`, `8m` overlap | 0.872071 | 0.868043 | 0.781997 | 19.907312 | 3.952x | 19.640612 | 0.266700 | 8.1619% | 4089 / 1097 / 4623 / 5186 |

Interpretation:

- `10m x 6m + 1m overlap` helps only modestly: 167-frame AP@0.5 rises from
  `0.791619` to `0.806211`, while reduction drops from `47.794x` to `30.213x`.
- The most useful larger-area/small-overlap region is currently
  `25m x 15m + 1m/2m overlap`: AP@0.5 reaches `0.852906/0.854296`, while still
  retaining `16.434x/13.735x` reduction.
- `35m x 21m + 2m overlap` improves AP@0.5 to `0.861531`, but reduction falls
  to `10.730x`. It is a stronger AP point but a weaker communication point.
- `40m x 24m` does not improve over `35m x 21m`; larger areas are not
  monotonically better because the group assignment becomes coarser and the
  selected leader/member structure changes.
- Recommended paper-facing Pareto points:
  - Conservative high-reduction: `10m x 6m + 1m`, AP@0.5 `0.806211`,
    `30.213x`.
  - Balanced small-overlap point: `25m x 15m + 1m`, AP@0.5 `0.852906`,
    `16.434x`.
  - Higher-AP small-overlap point: `35m x 21m + 2m`, AP@0.5 `0.861531`,
    `10.730x`.
  - Near-recovery but overlap-heavy point: `20m x 12m + 8m`, AP@0.5
    `0.868043`, `3.952x`.

### 2026-07-29 Follow-up: Apply the Same Area-Mask Protocol to Other Models

Protocol:

- Same OPV2V 5-CAV 167-frame segment: `start_frame=423`, `max_frames=167`.
- Same original-LGCP execution: `leader_box`, `Delta_g=0.10`.
- Same recommended strict group-area feature mask: `25m x 15m`, `1m` overlap.
- Actual mask uses LGCP group areas selected by `Delta_g`; no Top-K area
  selection is used.
- Communication statistics use the existing LGCP feature-packet accounting.

Main comparison:

| Model / checkpoint | Variant | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Reduction | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Where2comm `where2comm_10e` | `25m x 15m + 1m` mask | 0.855503 | 0.852906 | 0.753320 | 7.777682 | 16.434x | Best current strict executable LGCP-style reproduction point. |
| V2X-ViT `pointpillar_v2xvit_fusion` | no feature mask, same leader_box | 0.785449 | 0.762059 | 0.653262 | 4.935893 | 22.686x | Uses a compatibility fallback for missing `spatial_correction_matrix` / `prior_encoding`. |
| V2X-ViT `pointpillar_v2xvit_fusion` | `25m x 15m + 1m` mask | 0.763181 | 0.749789 | 0.676781 | 5.693814 | 19.047x | AP@0.5 drops only `0.012270`, but AP ceiling is below Where2comm. |
| CoAlign checkpoint from `D:\Files\Recent\checkpoints\coAlign` | no feature mask, no box-align | 0.956021 | 0.851802 | 0.528447 | 1.803286 | 44.118x | Stage-1 box-align files are missing on server, so this is CoAlign-w/o-box-align. |
| CoAlign checkpoint from `D:\Files\Recent\checkpoints\coAlign` | `25m x 15m + 1m` mask, no box-align | 0.933200 | 0.819052 | 0.497706 | 2.061053 | 39.151x | Promising communication point, but not strict full CoAlign due missing box-align precalc. |

Additional diagnostics:

- `pointpillar_attentive_fusion` can be loaded after config schema adaptation,
  but its 20-frame baseline AP is `0/0/0`. It outputs many boxes
  (`6321` predictions for 20 frames) with `TP=0`, suggesting checkpoint/config
  or training-domain mismatch. It should not be used as a paper-facing result
  until the correct matching code/config is recovered.
- `pointpillar_cobevt_fusion` in `C:\Workspace\OpenCDA\opencood\logs` is a
  camera BEV segmentation config: `CamIntermediateFusionDataset`,
  `CameraBevPostprocessor`, `model.core_method=corpbevt`. The current remote
  OPV2V test set has no `bev_dynamic.png` / `bev_static.png` /
  `bev_visibility_corp.png` files, and the current OpenCOOD workspace lacks the
  `corpbevt` model entry. Therefore it cannot be placed in the same 3D box AP
  table without preparing the camera BEV dataset/model stack.

Interpretation:

- Where2comm remains the cleanest strict reproduction candidate because its
  own objectness-driven BEV communication design matches the LGCP area-mask
  story most naturally.
- V2X-ViT can technically accept the area mask after adding a compatibility
  model entry, but its current baseline is weaker. It is useful as an
  "LGCP mask can generalize to another BEV intermediate model" result, not as
  the main performance claim.
- CoAlign-w/o-box-align is surprisingly strong on AP@0.3/AP@0.5 and gives
  high communication reduction under the same area-mask accounting. The caveat
  is important: without the missing stage-1 box-align files, this is not the
  full CoAlign protocol.
