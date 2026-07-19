# SGCP 核心实验结果

本文件只记录经过确认、可复现或准备进入论文/rebuttal 的核心结果。探索性现象先记录在 `log.md`，稳定后再整理到这里。

更新时间：2026-07-19

## 结果记录规范

每组结果至少应包含：

- 代码版本或 commit。
- 场景配置和随机种子。
- CAV 数量、背景车辆数量、速度范围。
- 通信配置：带宽、子信道数、发射功率、NS3 模型。
- 感知配置：backbone、fusion 方式、grid size、`rho_th`。
- SGCP 配置：`N_max`、`T_min^stab`、调度策略。
- 指标：mAP@0.3、mAP@0.5、mAP@0.7、通信开销、运行时耗时。
- 原始日志路径和结果文件路径。

## 主结果表

论文 `main.tex` 旧主表缺少原始日志、随机种子、代码提交和完整配置，不能作为已复现结果继续使用。当前可复现实验的版本、数据、命令和日志路径见 `reproducibility_manifest.md`；下表为当前 PAPG 主线的可复现主表候选。

| Method | mAP@0.3 | mAP@0.5 | mAP@0.7 | Comm. Overhead (Mbps) | Runtime / Cycle (ms) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Head-only | 0.26 | 0.22 | 0.09 | 0.00 | TBD | Reproducible lower reference; cluster heads detect alone with no point-cloud uploads, then late-fuse |
| Full 20-CAV early upper reference | 0.85 | 0.83 | 0.48 | 118.71 | TBD | Full point-cloud sharing AP upper bound; upload non-ego CAV payload 60,838,528 bytes |
| Built-in FullPerception PCS, tuned | 0.59 | 0.53 | 0.22 | 25.29 | TBD | Current formal FullPerception baseline; payload-based `c(q)`, real `sc_num`, schedulable blind-spot units |
| Global selective proxy | 0.84 | 0.80 | 0.46 | 109.71 | TBD | Virtual/global candidate pool, 3 members/head, 117 grid budget; not FullPerception PCS |
| EdgeCooper-style proxy | 0.75 | 0.70 | 0.32 | 109.53 | TBD | Virtual edge/global candidate pool, blind-spot complementarity proxy; preliminary, not strict paper reproduction |
| EdgeCooper-global network-aware proxy | 0.81 | 0.77 | 0.42 | 74.58 | TBD | Virtual edge/global assignment proxy with sender-load balancing and 35 m V2V feasibility; 11-frame NS3 73/110 complete |
| EdgeCooper-global-HD proxy | 0.81 | 0.78 | 0.42 | 65.40 | TBD | Virtual edge/global assignment proxy with sender-load balancing, 35 m V2V feasibility and half-duplex sender/receiver exclusion; 11-frame NS3 110/110 complete |
| Cluster-local selective proxy | 0.80 | 0.76 | 0.41 | 75.94 | TBD | CAV-side V2V only, cluster-local candidates, 3 members/head, 117 grid budget; NS3 110/110 complete |
| Full-cluster reference | 0.82 | 0.79 | 0.42 | 87.51 | TBD | Full intra-cluster upload reference |
| Selective V2V forced random | 0.77 | 0.73 | 0.38 | 61.68 | TBD | Same coalition path, 3 members/head, 117 grid budget |
| Selective V2V communication-aware | 0.78 | 0.75 | 0.40 | 58.97 | TBD | 2 members/head, 87 grid budget |
| Selective V2V density high-budget | 0.80 | 0.76 | 0.40 | 73.58 | TBD | 3 members/head, 117 grid budget |
| SGCP PAPG, 10ch, `rho_th=3`, `B_h=2` | 0.81 | 0.78 | 0.39 | 62.54 | TBD | Current main method; 110/110 PAPG NS3 replay complete |
| SGCP PAPG, 10ch, `rho_th=3`, `B_h=3` | 0.80 | 0.78 | 0.40 | 62.54 | TBD | Negative high-IoU probe; avg source CAVs drops to 2.67, so per-head RB relaxation is not enough |
| SGCP BPAPG source-balanced, 10ch, `rho_th=3` | 0.81 | 0.78 | 0.39 | 62.54 | TBD | Negative branch; source-diversity marginal term did not change PAPG per-CAV upload distribution |
| SGCP HUPAPG head-urgent, 10ch, `rho_th=3` | 0.81 | 0.78 | 0.39 | 62.54 | TBD | Safe but no-gain branch; receiver target urgency preserves PAPG but does not improve AP@0.7 |
| SGCP coverage-aware, 10ch, `rho_th=3` | 0.79 | 0.76 | 0.38 | 57.38 | TBD | PAPG predecessor/ablation |
| SGCP coverage-aware, 20ch | 0.80 | 0.76 | 0.41 | 73.98 | TBD | Resource-sensitivity row |

## Table 1 Protocol-Native Manifest

统一 source CSV：

`docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\protocol_native_manifest.csv`

| Method | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Evaluated Samples | Trace Rows | Receiver Policy | Late Fusion | Payload bytes | Mbps | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| Head-only | 0.26 | 0.22 | 0.09 | 41 | 246 | all-cluster-heads | yes | 0 | 0.00 | `mechanism_probe/head_only_41f_stdout.log` |
| Pure late, singleton 20-CAV | 0.82 | 0.76 | 0.37 | 41 | 820 | all-cluster-heads | yes | 0 | 0.00 | `table1_protocol_20260719/pure_late_singleton_41f.log` |
| FullPerception-PCS tuned | 0.59 | 0.53 | 0.22 | 41 | 281 | all-scheduled-receivers | yes | 12,959,840 | 25.29 | `pcs_tuning_20260718/pcs_41f_tuned_div12_ov0.log` |
| EdgeCooper-HD proxy | 0.81 | 0.78 | 0.42 | 41 | 246 | all-cluster-heads | yes | 33,519,040 | 65.40 | `repeat_check_20260718/edgecooper_hd_41f_r1.log` |
| SGCP-PAPG full | 0.81 | 0.78 | 0.39 | 41 | 246 | all-cluster-heads | yes | 32,049,872 | 62.54 | `repeat_check_20260718/papg_41f_r1.log` |
| Full 20-CAV early upper reference | 0.85 | 0.83 | 0.48 | 41 | 41 | full-20-cav | no | 60,838,528 | 118.71 | `table1_protocol_20260719/full20_early_41f.log` |

解释边界：Pure late 在 aggregate AP@0.3 上已经达到 0.82，说明 late-fusion 覆盖本身很强；主文不能把 SGCP 的 AP@0.3 优势只归因于 late fusion。若该行进入 protocol-native comparison，必须同时报告或估算 detection-box exchange overhead，或者标注为 prediction-sharing reference。SGCP 的主张应集中在：在不依赖 edge/global assignment 的 V2V 点云预算下，通过分簇 + PAPG 点云选择 + 两层融合，在 62.54 Mbps 达到接近 EdgeCooper-HD 的 AP@0.3/AP@0.5，同时保留 NS3 子信道可行性。

## Table 2 Fusion Scaffold Ablation

统一 source CSV：

`docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\fusion_scaffold_manifest.csv`

| Variant | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Evaluated Samples | Trace Rows | Late Fusion | Payload bytes | Mbps | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Head-only | 0.26 | 0.22 | 0.09 | 41 | 246 | yes | 0 | 0.00 | Lower reference for no point-cloud sharing |
| Pure late singleton 20-CAV | 0.82 | 0.76 | 0.37 | 41 | 820 | yes | 0 point-cloud bytes | 0.00 point-cloud Mbps | Prediction-sharing reference; needs box-overhead accounting |
| One-cluster full early-only | 0.85 | 0.83 | 0.48 | 41 | 41 | no | 60,838,528 | 118.71 | Full raw-LiDAR upper reference |
| Clustered early-only, PAPG | 0.38 | 0.36 | 0.20 | 246 | 246 | no | 32,049,872 | 62.54 | Shows early fusion alone has poor network coverage |
| One-cluster early+late | 0.85 | 0.83 | 0.48 | 41 | 41 | identity | 60,838,528 | 118.71 | Late fusion over one source is identity |
| Full SGCP, PAPG | 0.81 | 0.78 | 0.39 | 41 | 246 | yes | 32,049,872 | 62.54 | Same payload as clustered early-only, but late fusion restores coverage |

核心结论：同一 PAPG payload 下，clustered early-only 只有 `0.38/0.36/0.20`，加入簇间 late fusion 后 Full SGCP 达到 `0.81/0.78/0.39`，证明 two-layer fusion 的覆盖贡献非常明显。另一方面，Full SGCP 仍低于 one-cluster/full-sharing upper reference `0.85/0.83/0.48`，但只用约 52.7% 的 raw point-cloud Mbps；这为“通信受限下接近 full-sharing 上界”的叙事提供了更稳的主线。

## Table 3 SGCP-Compatible Scheduler Comparison

统一 source CSV：

`docs\doc_workspace\SGCP\artifacts\scheduler_comparison_20260719\scheduler_comparison_manifest.csv`

固定 scaffold：coalition clustering、raw LiDAR grid upload、all-cluster-head receiver policy、inter-cluster late fusion、20 MHz / 10 subchannels、41 frames。该表只比较同一 SGCP-compatible scaffold 中的 sender/grid scheduler，不代表 protocol-native baseline comparison。

| Scheduler | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. Source CAVs | Avg. Selected Grids | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random budgeted | 0.77 | 0.73 | 0.38 | 31,613,424 | 61.68 | 3.33 | 103.20 | Forced budget random baseline |
| Density-greedy | 0.80 | 0.76 | 0.40 | 37,710,864 | 73.58 | 3.33 | 102.18 | High-budget density-only scheduler |
| Link-aware density | 0.80 | 0.76 | 0.40 | 37,710,864 | 73.58 | 3.33 | 102.18 | Same result as density at this budget; link penalty did not change selected set |
| PACP-style LiDAR proxy | 0.81 | 0.79 | 0.42 | 44,361,424 | 86.56 | 3.33 | 104.93 | RGB/BEV PACP idea migrated to LiDAR grid priority; not strict PACP |
| EdgeCooper-HD proxy | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 | 3.00 | 89.02 | Edge/global half-duplex assignment proxy; stronger information condition |
| SGCP-PAPG | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 2.67 | 97.22 | Proposed scheduler: coverage layer + target layer |

解释边界：PAPG 不是 AP@0.7 最优；PACP-LiDAR 和 EdgeCooper-HD 的高 IoU 更强。但 PAPG 在同一 scaffold 下以更少 payload 达到最优 AP@0.3，并与 EdgeCooper-HD 持平 AP@0.5；相比 density/link-aware 少约 15.0% payload 且 AP@0.3/AP@0.5 更高，相比 PACP-LiDAR 少约 27.8% payload且 AP@0.3 持平。该表应写成 AP-Mbps scheduler tradeoff，而不是“SGCP 全面最高 AP”的单点主张。

## Figure 1 AP-Mbps Pareto Draft

统一 source CSV：

`docs\doc_workspace\SGCP\artifacts\pareto_20260719\pareto_source.csv`

图表草稿：

- AP@0.3 vs Mbps：`docs\doc_workspace\SGCP\artifacts\pareto_20260719\figure1_pareto_ap03.png`
- AP@0.7 vs Mbps：`docs\doc_workspace\SGCP\artifacts\pareto_20260719\figure1_pareto_ap07.png`

当前结论：PAPG 在 raw-LiDAR V2V/PPS 方法中处于中等通信量区间，`62.54 Mbps` 达到 `0.81/0.78/0.39`，相比 forced random 近似同 payload 提升 AP，相比 density/link-aware 少约 15% raw payload 并提升 AP@0.3/AP@0.5。Pure late broadcast 以很低 detection-box overhead 达到 AP@0.3 `0.82`，因此必须单独标为 prediction-sharing reference；EdgeCooper-HD 与 PACP-LiDAR proxy 的 AP@0.7 更强，应作为 edge/global 或 stronger-prior boundary，而不是写成 SGCP 的同类 V2V-only 失败。

## Figure 2/3 Protocol and Fusion Breakdown Drafts

统一 artifact 目录：

`docs\doc_workspace\SGCP\artifacts\figures_20260719\`

图表：

- Figure 2 protocol-native breakdown：`figure2_protocol_breakdown.png` / `.pdf`
- Figure 3 fusion contribution：`figure3_fusion_contribution.png` / `.pdf`

Figure 2 当前用途：用 Head-only、Pure late、FullPerception-PCS、EdgeCooper-HD、SGCP-PAPG 和 Full 20-CAV upper reference 说明完整系统语义层级，避免把 FullPerception baseline、full-sharing upper reference 和 scheduler proxy 混在一起。图内通信标注为 raw LiDAR Mbps；Pure late 的 detection-box overhead 仍需参考 `late_fusion_box_comm.md`。

Figure 3 当前用途：支撑 two-layer fusion 分工。Clustered early-only 与 Full SGCP raw payload 相同，前者为 `0.38/0.36/0.20`，后者为 `0.81/0.78/0.39`，说明 inter-cluster late fusion 对 coverage / low-IoU AP 贡献最大；Full 20-CAV early upper reference 为 `0.85/0.83/0.48`，说明 high-IoU localization 仍受 early checkpoint 和 raw point-cloud sharing 上界约束。

## Table 4 Parameter Sensitivity Candidate

统一 artifact 目录：

`docs\doc_workspace\SGCP\artifacts\parameter_sensitivity_20260719\`

源数据：

- `table4_parameter_sensitivity.csv`
- `table4_parameter_sensitivity.md`

主文建议只放证据最清晰的两组：

| Parameter | Setting | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `rho_th` | 1.0 | 0.76 | 0.72 | 0.34 | 51.31 | Lower payload, lower AP |
| `rho_th` | 2.0 | 0.79 | 0.75 | 0.37 | 56.08 | Low-budget candidate |
| `rho_th` | 3.0 | 0.79 | 0.76 | 0.38 | 57.38 | Better AP with modest payload increase |
| Channels | 5 | 0.56 | 0.53 | 0.27 | 28.91 | Strong resource bottleneck |
| Channels | 10 | 0.79 | 0.75 | 0.37 | 56.08 | Main low-budget channel setting |
| Channels | 20 | 0.80 | 0.76 | 0.41 | 73.98 | Higher localization AP, higher payload |

`N_max` 和 `T_min^stab` 已有 sweep，但当前 41 帧短序列结论较弱：`N_max` 证明容量约束真实生效但 AP 非单调；`T_min^stab=100--1000 ms` 完全不敏感。因此建议放附录或 rebuttal，正文只保留保守一句话。

## Paper Experiment Section Update

论文文件：

`C:\Workspace\icdcs-paper\SGCP\main.tex`

2026-07-19 已完成第一版实验章节同步：

- 主表 `tab:mAP` 加入 Pure late prediction-sharing reference，报告 `0.82/0.76/0.37` 和 80B/box broadcast overhead `0.74 Mbps`。
- 新增 `fig:protocol_breakdown`，引用 `fig/sgcp_protocol_breakdown.pdf`。
- 新增 `fig:fusion_contribution`，引用 `fig/sgcp_fusion_contribution.pdf`。
- 新增 `fig:pareto`，同时引用 `fig/sgcp_pareto_ap03.pdf` 和 `fig/sgcp_pareto_ap07.pdf`。
- 新增 `tab:scheduler_comparison`，明确它是 SGCP-compatible scheduler comparison，不是 protocol-native system ranking。
- 将旧 `tab:rho_sensitivity` 替换为 `tab:param_sensitivity`，主文只放 `rho_th` 与 channel count 两组强结论；`N_max/T_min^stab` 写成附录/边界说明。

验证：本机未检测到 `latexmk` / `pdflatex` / `bibtex`，无法编译 PDF。轻量结构检查通过：table/figure/tabular begin-end 配对正常，新增 label/ref 无缺失。

## 消融实验

| Variant | mAP@0.3 | mAP@0.5 | mAP@0.7 | Comm. Overhead (Mbps) | Reconfig. Count | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SGCP full | 0.77 | 0.73 | 0.35 | TBD | 11 | Offline constrained intra-cluster early fusion + inter-cluster late fusion, `potential_game` |
| w/o stability window | 0.77 | 0.73 | 0.35 | TBD | 11 | `T_min_stab=0`; identical to full SGCP on current 41-frame dump |
| w/o coalition formation - singleton | 0.82 | 0.76 | 0.37 | TBD | 0 | Each CAV is a singleton cluster; prediction-level late-fusion overhead is not counted |
| w/o PPS - random | 0.44 | 0.39 | 0.17 | TBD | 11 | Random scheduling, same SGCP late-fusion evaluation path |
| w/o PPS - MWS | 0.31 | 0.26 | 0.11 | TBD | 11 | Greedy scheduling, needs baseline-definition review |
| early fusion only | 0.85 | 0.83 | 0.48 | TBD | N/A | Full 20-CAV early fusion, no SGCP communication constraint |
| constrained early only | 0.36 | 0.34 | 0.17 | TBD | 11 | All cluster heads, no inter-cluster late fusion |
| late fusion only | 0.91 | 0.85 | 0.51 | TBD | N/A | OpenCOOD full 20-CAV late checkpoint; reference only, not a strict same-checkpoint SGCP ablation |

## FullPerception Heuristic Sanity

PCS tuned 后，MWS/RS 复用相同 blind-spot 粒度和 scheduled `sc_num` 口径做 11 帧 sanity check。结果说明它们仍是 heuristic/diagnostic ablation，不适合进入主公平表。

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FullPerception-MWS tuned | 11 | 0.36 | 0.32 | 0.15 | 4,289,344 | 39.00 | Greedy heuristic remains weak despite higher payload |
| FullPerception-RS tuned | 11 | 0.54 | 0.49 | 0.23 | 1,644,160 | 14.95 | Random heuristic remains far below SGCP/strong selective baselines |

## Baseline 公平性说明

详细口径见 `baseline_fairness.md`。当前结果应按以下层级解释：

| Layer | Method | Current Result | Main Fair Baseline? | Notes |
| --- | --- | --- | --- | --- |
| Upper reference | Full 20-CAV early fusion | 0.85 / 0.83 / 0.48 | No | 全点云共享，无 SGCP 通信约束；不是 FullPerception baseline；non-ego upload payload 60,838,528 bytes |
| Upper reference | Full 20-CAV late checkpoint | 0.91 / 0.85 / 0.51 | No | 使用独立 late checkpoint，不能直接作为同 checkpoint 消融 |
| Built-in FullPerception | PCS (`pcs.py`) tuned | 0.59 / 0.53 / 0.22 | No | 仓库内置 PCS 对应 FullPerception 论文调度算法；payload 12,959,840 bytes / 25.29 Mbps；正式 baseline 但不作为强 V2V 主对比 |
| RSU/edge-assisted | Global selective proxy | 0.84 / 0.80 / 0.46 | No | 虚拟/global candidate pool；当前 dump 无真实 RSU sensor，不作为 V2V-only 公平主对比，也不命名为 FullPerception |
| RSU/edge-assisted | EdgeCooper-style proxy | 0.75 / 0.70 / 0.32 | No | 虚拟 edge/global candidate pool；当前是 blind-spot complementarity proxy，不是严格原论文 MCF/coloring 复现 |
| RSU/edge-assisted | EdgeCooper-global network-aware proxy | 0.81 / 0.77 / 0.42 | No | 虚拟 edge/global assignment proxy；74.58 Mbps，11 帧 NS3 73/110 complete，说明离线高 AP 仍需 deadline-aware 调度补强 |
| RSU/edge-assisted | EdgeCooper-global-HD proxy | 0.81 / 0.78 / 0.42 | No | 虚拟 edge/global assignment proxy + 半双工约束；65.40 Mbps，11 帧 NS3 110/110 complete，是当前最强 edge-assisted baseline |
| V2V-only fair baseline | Cluster-local selective proxy | 0.80 / 0.76 / 0.41 | Yes | cluster-local candidate pool，3 members/head，117 grid budget；强 decentralized proxy；11 帧 NS3 replay 110/110 application/RLC complete |
| SGCP main | SGCP PAPG 10ch | 0.81 / 0.78 / 0.39 | Yes | 当前主方法，62.54 Mbps，PAPG NS3 110/110 complete |
| SGCP sensitivity | SGCP PAPG 10ch, `B_h=3` | 0.80 / 0.78 / 0.40 | No | 简单放宽 per-head RB 上限，未形成更多有效 source diversity；作为负面 sensitivity |
| SGCP negative branch | SGCP BPAPG source-balanced | 0.81 / 0.78 / 0.39 | No | source-diversity marginal term 不足以改变最终 upload distribution；history-credit 11 帧会伤 AP |
| SGCP negative branch | SGCP QG/HU-PAPG | 0.75 / 0.72 / 0.33 on 11f QG; 0.81 / 0.78 / 0.39 on 41f HU | No | QG source-history 仍伤 AP；HU receiver-urgency 安全但无增益 |
| SGCP ablation | SGCP potential_game | 0.77 / 0.73 / 0.35 | Yes | 原始 PPS 消融 |
| Same pipeline ablation | Random scheduler | 0.44 / 0.39 / 0.17 | No | payload 过低，只作 w/o-PPS 诊断 |
| Same pipeline ablation | MWS scheduler | 0.31 / 0.26 / 0.11 | No | payload 过低，只作 w/o-PPS 诊断 |
| Same-budget selective baseline | Nearest top-k grid sharing | 0.76 / 0.73 / 0.37 | Yes | CAV-only, same clustering + late-fusion path, grid budget 87 |
| Same-budget selective baseline | Density top-k grid sharing | 0.77 / 0.74 / 0.39 | Yes | Strong baseline; slightly higher AP@0.7 with higher payload |
| Same-budget selective baseline | Communication-aware density sharing | 0.78 / 0.75 / 0.40 | Yes | 2 members/head, 87 grid budget; density divided by distance cost |
| Same-budget selective baseline | Forced-budget random sharing | 0.77 / 0.73 / 0.38 | Yes | 3 members/head, 117 grid budget，61.68 Mbps |
| Same-budget selective baseline | Density/communication-aware high-budget | 0.80 / 0.76 / 0.40 | Yes | 3 members/head, 117 grid budget; payload-matched to SGCP 20ch |
| SGCP ablation | Coverage-aware spatial-diverse, 10ch/rho3 | 0.79 / 0.76 / 0.38 | Yes | PAPG 前身/消融，57.38 Mbps，NS3 110/110 complete |
| SGCP sensitivity | Coverage-aware spatial-diverse, 20ch | 0.80 / 0.76 / 0.41 | Yes | 高预算资源敏感性，73.98 Mbps，NS3 154/154 complete |
| Reference only | Singleton full late-fusion reference | 0.82 / 0.76 / 0.37 | No | late-fuse 全部 20 CAV，当前未计 detection-box exchange overhead |

论文写作建议：FullPerception baseline 使用 `pcs.py` / `fullperception_pcs`；full 20-CAV early/late fusion 只能作为 upper/reference，不应放入“同通信预算公平主对比”结论。公平主对比应使用同数据、同 backbone、同 AP 口径，并尽量匹配通信预算或显式报告 payload。旧 `RandomRA/MWS` 的 payload 只有约 9.7/9.9 MB，未充分利用 10 子信道资源，不宜作为“SGCP 降低通信量”的主证据；它们更适合作 w/o PCS/PPS 消融。主公平 baseline 使用 forced-budget random、density/communication-aware selective sharing；SGCP 主方法使用 PAPG。

### Explicit FullPerception Baselines

代码状态：仓库中的 FullPerception baseline 应指向 `opencda/core/clustering/algorithms/resource_allocation/pcs.py`，即 FullPerception 论文的 PCS 调度算法；`mws.py` 和 `random_ra.py` 是同一 PCS 问题上的 greedy/random baseline。正规入口为 `--resource-allocation fullperception_pcs|fullperception_mws|fullperception_random`。此前后补的两个 selective-sharing proxy 已重命名为 `global_selective_proxy` 和 `cluster_local_selective_proxy`，避免把 proxy 误写成 FullPerception 论文复现。

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_41f_tuned_div12_ov0_trace.csv
```

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\
```

| Baseline | Candidate Scope | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. Source CAVs | Avg. Selected Grids | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full 20-CAV early upper reference | all CAVs, full upload | 0.85 | 0.83 | 0.48 | 60,838,528 | 118.71 | 20.00 | N/A | AP upper bound, not budgeted scheduling |
| `fullperception_pcs` built-in, legacy cluster-head eval | PCS blind-spot scheduling | 0.44 | 0.39 | 0.17 | 12,684,880 | 24.75 | 1.66 | 630.66 | Pre-repair compatibility row; simplified `c(q)=1` |
| `fullperception_pcs` built-in, tuned PCS | PCS blind-spot scheduling | 0.59 | 0.53 | 0.22 | 12,959,840 | 25.29 | 2.00 | 27.00 | Current FullPerception baseline; blind spots split into schedulable units (`division=12`, `min_overlap=0`) |
| `global_selective_proxy` | global / virtual RSU | 0.84 | 0.80 | 0.46 | 56,224,736 | 109.71 | 4.00 | 117.00 | Infrastructure-assisted proxy, not FullPerception PCS |
| `cluster_local_selective_proxy` | cluster-local V2V | 0.80 | 0.76 | 0.41 | 38,920,592 | 75.94 | 3.33 | 103.20 | Strong V2V-only selective proxy; NS3 110/110 complete |
| `global_selective_proxy`, ego receiver probe | ego virtual receiver | 0.71 | 0.70 | 0.49 | 26,350,784 | 51.42 | 9.54 | 332.93 | Diagnostic only; several frames fell back to ego-only |

`cluster_local_selective_proxy` 的 11-frame true NS3 replay 已完成：110/110 scheduled requests 完成 application callback，110/110 RLC complete，RLC TX/RX events 2970/2970，PHY decode failures 0，avg/p95 callback delay 23.91/24.00 ms。tuned `fullperception_pcs` 已完成 11-frame dry-run：每帧 5 条 scheduled request，0 skipped unscheduled；41-frame perception run 为 `0.59/0.53/0.22`、25.29 Mbps。

### Same-Budget CAV-Only Selective Sharing

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。复用 SGCP coalition formation 和 inter-cluster late fusion 评价口径，但不使用 PPS；每个 cluster head 最多选择 2 个非 head 成员，总 grid budget 为 87，接近 SGCP 默认 `avg_selected_grids=87.32`。

算法定义：

- `Selective forced random`：复用 SGCP coalition 和 inter-cluster late fusion，但不使用 SGCP/PPS utility；每个 cluster head 随机选择固定预算内的 sender 和 grid。主表使用强制预算版 `3 members/head + 117 grids`，避免旧 RandomRA 因 payload 过低变成弱 baseline。
- `Selective communication-aware`：候选 sender 先按其可补充给 receiver weak/blind grids 的 density sum 打分，再除以距离代价 `1 + distance / 100`；如果提供 NS3 link-quality CSV，则再乘以 request-level delivery quality。主表低预算版使用 `2 members/head + 87 grids`。
- `Selective density high-budget`：候选 sender 只按 weak/blind grids 上的 density sum 贪心选择，grid 也按 sender 局部 density 从高到低选取；主表高预算版使用 `3 members/head + 117 grids`，用于和 PAPG/20ch 等高预算设置比较。

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline nearest --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --max-frames 0
```

| Baseline | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SGCP `potential_game` | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | Main method |
| Selective nearest | 0.76 | 0.73 | 0.37 | 113930.21 | 28026832 | 2.81 | 81.38 | CAV-only nearest member selection |
| Selective density | 0.77 | 0.74 | 0.39 | 124286.05 | 30574368 | 2.81 | 81.38 | Strong baseline; higher payload and AP@0.7 |
| Selective communication-aware | 0.78 | 0.75 | 0.40 | 122854.70 | 30222256 | 2.81 | 81.38 | Strongest current baseline; density score penalized by distance |
| Selective density high-budget | 0.80 | 0.76 | 0.40 | 153296.20 | 37710864 | 3.33 | 102.18 | 3 members/head, 117 grid budget; payload-matched to SGCP 20ch |
| Selective communication-aware high-budget | 0.80 | 0.76 | 0.40 | 153296.20 | 37710864 | 3.33 | 102.18 | Same result as density on this dump without external NS3 quality CSV |

观察：communication-aware selective-sharing 是强公平 baseline。低预算 2-member/87-grid 设置中，它的 AP@0.7 高于原始 SGCP；高预算 3-member/117-grid 设置中，它达到 `0.80/0.76/0.40`，与 SGCP spatial-diverse 20ch 的 `0.80/0.76/0.41` 接近但 AP@0.7 略低，payload 也接近。因此论文中不应依赖低通信 Random/MWS 来证明通信节省，而应报告 payload-matched selective baselines，并强调 SGCP 的 PPS 子信道可行性、NS3 完整交付和 coverage-aware grid selection。

### PAPG / EdgeCooper-HD Repeat Check

实验口径：同一 41 帧 dump、20 CAV、OpenCOOD early checkpoint、inter-cluster late fusion。PAPG 使用 `perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2`；EdgeCooper-HD 使用 `--selective-sharing-baseline edgecooper_global_hd --selective-member-budget 3 --selective-grid-budget 117`。本轮重跑用于确认两者 AP 接近是否为随机波动。

| Method | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Total Upload bytes | Mbps | Log / Trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PAPG repeat | 11 | 0.76 | 0.73 | 0.34 | 8,598,224 | 62.53 | `docs/doc_workspace/SGCP/artifacts/repeat_check_20260718/papg_11f_r1_trace.csv` |
| EdgeCooper-HD repeat | 11 | 0.77 | 0.73 | 0.37 | 9,097,008 | 66.16 | `docs/doc_workspace/SGCP/artifacts/repeat_check_20260718/edgecooper_hd_11f_r1_trace.csv` |
| PAPG repeat | 41 | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | `docs/doc_workspace/SGCP/artifacts/repeat_check_20260718/papg_41f_r1.log` |
| EdgeCooper-HD repeat | 41 | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 | `docs/doc_workspace/SGCP/artifacts/repeat_check_20260718/edgecooper_hd_41f_r1.log` |

结论：全 41 帧结果与既有主表完全一致，说明 EdgeCooper-HD 与 PAPG 的接近是稳定结果。论文中必须按能力边界解释：EdgeCooper-HD 有 edge/global assignment，因此 AP@0.7 更强；PAPG 是 V2V-only 去中心化主线，重点主张是低/中 IoU AP 与 payload tradeoff、子信道可行性和不依赖 RSU。

### NS3 Link-Quality-Aware Selective Sharing

实验口径：`D:\Data\Carla\2026_07_15_01_26_56` 前 11 帧，20 CAV，same-budget selective sharing，`communication_aware`，member budget 2，grid budget 87，inter-cluster late fusion。NS3 link-quality 使用 `docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\eval\rlc_by_request.csv` 中的 `rlc_complete`，即 `targetSubchannels=5` 受限暴露子信道回归。

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Distance proxy | 11 | 0.71 | 0.67 | 0.31 | 120873.94 | 7977680 | 2.80 | 80.85 | Old `density / distance` score |
| NS3 RLC-complete aware | 11 | 0.68 | 0.63 | 0.27 | 118129.70 | 7796560 | 2.80 | 80.85 | Uses `density * rlc_complete_ratio / distance` |

观察：NS3-aware cost 避开了受限 5 子信道下不可完整交付的链路，通信量略降，但 11 帧 AP 也下降。该结果不应被解释为 NS3-aware baseline 更强，而是说明真实链路可行性会改变 selective-sharing 的成员选择；后续主实验应在完整 41 帧或重新导出的网络受限场景上报告。

## Mechanism Probe

详细口径见 `mechanism_probe.md`。该 probe 使用同一 41 帧 dump、同一 coalition formation 和 inter-cluster late fusion，只改变每个 cluster head 接收的点云上传模式。

| Mode | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Bytes / Receiver | Avg. Uploaded Sources | Avg. Uploaded Points | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Head-only | 0.26 | 0.22 | 0.09 | 0 | 0.00 | 0.00 | 0.00 | Cluster heads detect alone, then late-fuse |
| SGCP grid-constrained | 0.77 | 0.73 | 0.35 | 26,916,208 | 109,415.48 | 1.67 | 6,838.47 | Current main SGCP constrained mode |
| Random grid, same scheduled links | 0.78 | 0.75 | 0.36 | 27,908,560 | 113,449.43 | 1.67 | 7,090.59 | Same PPS scheduled sender links and grid counts, deterministic random grid candidates |
| Raw-density score | 0.74 | 0.70 | 0.37 | 29,290,768 | 119,068.16 | 1.67 | 7,441.76 | Replaces saturated utility with sender grid density |
| Density-distance score | 0.74 | 0.71 | 0.37 | 29,219,088 | 118,776.78 | 1.67 | 7,423.55 | Sender density divided by receiver-grid distance cost |
| Spatial-diverse grid, same scheduled links | 0.79 | 0.75 | 0.37 | 28,743,280 | 116,842.60 | 1.67 | 7,302.66 | Density-aware spatial cover, same PPS scheduled links and grid counts |
| Full-cluster upload | 0.82 | 0.79 | 0.42 | 44,850,528 | 182,319.22 | 2.33 | 11,394.95 | Same clusters, upload all member point clouds |

观察：SGCP grid-constrained 使用约 60.0% 的 full-cluster payload，并保留大部分 AP@0.5，但 AP@0.7 损失明显。随机 grid selection 在相同 PPS scheduled links 和相同 grid 数量下略高于当前 utility selection，说明原始饱和 density utility 不足。`spatial_diverse` 进一步达到 `0.79/0.75/0.37`，高于 random-grid，说明 coverage-aware grid selection 是当前最有希望的主表修复方向；raw density / density-distance 虽提升 AP@0.7，但会损失 AP@0.3/0.5 且 payload 更高。

### Spatial-Diverse Channel Sweep

实验口径：同一 41 帧 dump、20 CAV、`potential_game`、SGCP inter-cluster late fusion，启用 `--sgcp-grid-selection-mode spatial_diverse`。该表用于评估 coverage-aware grid selection 在不同子信道预算下的通信-精度折中。

| Num. Channels | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Bytes / Receiver | Avg. Uploaded Sources | Avg. Uploaded Points | Avg. Selected Grids | Payload vs Full-Cluster | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0.56 | 0.53 | 0.27 | 14,815,408 | 60,225.24 | 0.83 | 3,764.08 | 45.58 | 33.0% | Strong channel bottleneck; AP governed by admitted links |
| 10 | 0.79 | 0.75 | 0.37 | 28,743,280 | 116,842.60 | 1.67 | 7,302.66 | 87.32 | 64.1% | Current best low-payload SGCP candidate |
| 20 | 0.80 | 0.76 | 0.41 | 37,912,544 | 154,116.03 | 2.33 | 9,632.25 | 117.18 | 84.5% | Near full-cluster AP@0.7 with lower payload |

观察：`spatial_diverse` 的 10 子信道版本在约 64.1% full-cluster payload 下达到 `0.79/0.75/0.37`，比原始 utility 和 random-grid 更稳；20 子信道版本把 AP@0.7 提升到 `0.41`，接近 full-cluster `0.42`，但 payload 升至 full-cluster 的 84.5%。论文主表可以考虑报告 10 子信道作为低通信主点，并用 20 子信道作为 high-budget sensitivity，而不是只给单一设置。

## 参数敏感性

### Stability Window

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，`potential_game`，SGCP inter-cluster late fusion。`T_min^stab` 命令行单位为秒；表中按论文写作习惯记录为 ms。

| `T_min^stab` (ms) | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Reconfig. Count | Vehicle-Head Changes | Avg. Cluster Lifetime (frames) | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 11 | 76 | 6.65 | Same as default on current dump |
| 300 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 11 | 76 | 6.65 | Same as default on current dump |
| 500 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 11 | 76 | 6.65 | Paper default candidate; no sensitivity observed here |
| 700 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 11 | 76 | 6.65 | Same as default on current dump |
| 1000 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 11 | 76 | 6.65 | Current implementation default |

### Max Cluster Size

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，`potential_game`，SGCP inter-cluster late fusion。`Comm. Overhead` 当前先记录为平均每个 cluster-head source 的点云 upload payload；尚未换算 Mbps。

| `N_max` | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Cluster Size | Avg. Clusters | Reconfig. Count | Avg. Cluster Lifetime (frames) | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.79 | 0.74 | 0.37 | 62198.64 | 26247824 | 1.95 | 10.29 | 16 | 7.28 | More clusters; smaller intra-cluster fusion groups |
| 3 | 0.75 | 0.71 | 0.34 | 82226.47 | 25572432 | 2.65 | 7.59 | 9 | 7.59 | Lower AP than `N_max=2/4` in current dump |
| 4 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 3.33 | 6.00 | 11 | 6.65 | Current default / paper candidate |
| 5 | 0.75 | 0.71 | 0.32 | 102582.76 | 25235360 | 3.33 | 6.00 | 8 | 10.70 | Same cluster count as `N_max=6`; different from default due to coalition search path |
| 6 | 0.75 | 0.71 | 0.32 | 102582.76 | 25235360 | 3.33 | 6.00 | 8 | 10.70 | Same result as `N_max=5` on current 20-CAV dump |

Capacity statistics for the same `N_max` sweep:

| `N_max` | Avg. Full Clusters | Max Full Clusters | Full Candidate Skips | Avg. Skips / Frame | Avg. Singleton Cluster Ratio | Avg. Small-Cluster Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 9.71 | 10 | 12534 | 305.71 | 0.053 | 1.000 |
| 3 | 6.00 | 6 | 7894 | 192.54 | 0.146 | 0.206 |
| 4 | 3.12 | 4 | 4065 | 99.15 | 0.000 | 0.187 |
| 5 | 1.00 | 1 | 1142 | 27.85 | 0.000 | 0.317 |
| 6 | 0.00 | 0 | 0 | 0.00 | 0.000 | 0.317 |

Observation: the default `N_max=4` creates no singleton clusters in this dump, but still has 3.12 full clusters per frame and 99.15 capacity-skipped candidate joins per frame. This supports the mechanism claim that `N_max` is an active hard capacity constraint; blocked vehicles are retained in feasible coalitions or small clusters and still enter inter-cluster late fusion.

### Density Threshold

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，`potential_game`，SGCP inter-cluster late fusion。`rho_th` 覆盖 lidar `density_threshold`，影响 high-density grid 判定、`Vehicle_Grid.rho_th`、cluster grid bits 和 PPS grid selection。

| `rho_th` | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Cluster-Head Sources | Avg. Clusters | Reconfig. Count | Vehicle-Head Changes | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.74 | 0.69 | 0.34 | 86658.74 | 21751344 | 251 | 6.12 | 10 | 60 | Lowest payload, lower AP |
| 1.0 | 0.75 | 0.71 | 0.33 | 96968.13 | 23854160 | 246 | 6.00 | 9 | 64 | Lower payload than default |
| 2.0 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 246 | 6.00 | 11 | 76 | Current implementation default / paper candidate |
| 3.0 | 0.77 | 0.73 | 0.37 | 113689.69 | 27967664 | 246 | 6.00 | 11 | 76 | Higher AP@0.7, higher payload |
| 4.0 | 0.77 | 0.74 | 0.37 | 115754.73 | 28475664 | 246 | 6.00 | 11 | 76 | Best AP@0.5/AP@0.7 in this dump, highest payload |

Coverage-aware spatial-diverse under the same `rho_th` sweep:

| `rho_th` | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 0.76 | 0.72 | 0.34 | 106896.20 | 26296464 | 2.67 | 80.75 | Lower payload, but AP drops |
| 2.0 | 0.79 | 0.75 | 0.37 | 116842.60 | 28743280 | 2.67 | 87.32 | Current low-budget candidate |
| 3.0 | 0.79 | 0.76 | 0.38 | 119533.72 | 29405296 | 2.67 | 89.72 | Slight AP gain with modest payload increase |
| 4.0 | 0.79 | 0.76 | 0.38 | 121291.64 | 29837744 | 2.67 | 90.62 | Similar AP to 3.0, higher payload |

Observation: `rho_th` is the main point-cloud threshold knob for this pipeline. For `spatial_diverse`, increasing `rho_th` from 2.0 to 3.0 improves AP@0.5/AP@0.7 from `0.75/0.37` to `0.76/0.38` with payload rising from 28.74 MB to 29.41 MB. This is a better paper parameter sweep than claiming Random/MWS reduce communication, because it shows an actual AP/payload threshold tradeoff inside the proposed method.

Target-aware potential-game scheduler:

| Method | Channels / BW | `rho_th` | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Mbps | Avg. Selected Grids | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `potential_game + spatial_diverse` | 10ch / 20 MHz | 3.0 | 0.79 | 0.76 | 0.38 | 29,405,296 | 57.38 | 89.72 | Former low-budget tuned row; grid action was replaced outside the allocator |
| `target_aware_potential_game` | 10ch / 20 MHz | 3.0 | 0.80 | 0.76 | 0.39 | 31,069,968 | 60.62 | 89.72 | New allocator: original PotentialGame sender/RB stage plus target-aware grid-action refinement |
| `perception_aware_potential_game`, `B_h=2` | 10ch / 20 MHz | 3.0 | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 97.22 | Two-layer allocator: coverage layer guarantees one external view per head, target layer assigns remaining RBs to object-prototype gains |

Object-level diagnostics show the new scheduler reduces full-reference-detected but SGCP-missed GT rows from 111 to 106. The main targeted bucket, “covered only by other cluster heads,” drops from 63 to 56, and nearest-head covering point mean rises from 69.4 to 79.0. This supports the mechanism change: AP gain comes from moving key target grids toward the relevant cluster head, not from adding more scheduled links.

Perception-aware PG is the current best coherent main-table candidate. It improves over target-aware PG on AP@0.3/AP@0.5 (`0.81/0.78` vs. `0.80/0.76`) while retaining AP@0.7 (`0.39`). Compared with the strong high-budget selective baseline (`0.80/0.76/0.40`, 37,710,864 bytes / 73.58 Mbps), PAPG uses about 15.0% less payload and improves AP@0.3/AP@0.5, with AP@0.7 lower by 0.01. It remains below the full 20-CAV early upper reference (`0.85/0.83/0.48`, 118.71 Mbps), which is the desired claim boundary.

PAPG object-level diagnostics reduce full-reference-detected but SGCP-missed rows from 106 under target-aware PG to 59, with 410 scheduled links over 41 frames (10 links/frame, no extra unscheduled source bypass). The dominant remaining missed grids are `0_-2`, `3_-1`, `0_1`, and `2_-2`; these should drive the next object-level paper figure or online validation, not another ad-hoc fallback.

PAPG NS3 request-level replay is now complete over the first 11 frames: 110 planned/scheduled requests, 110 matched `cam_received` callbacks, 110/110 RLC-complete requests, 2970/2970 RLC TX/RX events, 0 RLC drops, 0 PHY decode failures, average callback delay 23.91 ms and p95 delay 24.00 ms. Artifact path: `docs/doc_workspace/SGCP/artifacts/papg_ns3_20260717_210304/`.

Forced-budget random selective baseline now has the same 11-frame scheduled-only NS3 replay evidence: 110 planned/scheduled requests, 110 matched `cam_received` callbacks, 110/110 RLC-complete requests, 2970/2970 RLC TX/RX events, 0 RLC drops, 0 PHY decode failures, average callback delay 23.91 ms and p95 delay 24.00 ms. Artifact path: `docs/doc_workspace/SGCP/artifacts/forced_random_ns3_20260717_2304b/`. This confirms PAPG's AP gain over forced random is not caused by giving PAPG a more reliable NS3 path.

Forced-budget random selective baseline: using the same coalition and late-fusion path with 3 uploaded members per head and 117 grid budget, deterministic random member/grid selection reaches AP@0.3/0.5/0.7 = `0.77/0.73/0.38`, total payload `31,613,424` bytes (`61.68 Mbps`), avg source CAVs `3.33`, avg selected grids `103.20`. This replaces the old low-payload RandomRA row as the fair random baseline for main-table discussion.

### Routing Probe Boundary

These probes diagnose the residual gap between PAPG and edge/global assignment baselines. They should not be promoted to main-table methods unless a principled detector/proposal-level trigger is added.

| Probe | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload / Mbps | Key Evidence | Paper Use |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAPG main reference | 41 | 0.81 | 0.78 | 0.39 | 32,049,872 bytes / 62.54 Mbps | 410 scheduled links; first-11-frame NS3 replay 110/110 complete | Main V2V-only SGCP method |
| ISPG instance-support utility | 41 | 0.80 | 0.78 | 0.39 | 32,046,336 bytes / 62.53 Mbps | Instance-support term in intra-cluster utility is neutral/negative | Internal negative probe |
| CCISPG naive cross-cluster routing | 11 | 0.68 | 0.64 | 0.37 | 8,663,216 bytes / 62.99 Mbps | External links 104/110; high-IoU can move but low-threshold recall collapses | Shows global routing is risky without edge-level control |
| CCISPG layered/cap1 routing | 11 | 0.75/0.75 | 0.71/0.72 | 0.33/0.33 | about 62.6 Mbps | External links reduced to 44/110 or 11/110, but AP stays below PAPG | Negative mechanism probe |
| PAPG + object-grid routing hints | 11 | 0.75 | 0.71 | 0.35 | 8,563,440 bytes / 62.28 Mbps | Post-hoc GT comparison: 4 gained rows, 15 lost rows | Failure analysis only |

Interpretation: object-grid support can repair individual misses, but it can also remove context needed for other already-detected objects. The paper should therefore claim a stable decentralized AP/payload tradeoff for PAPG, while presenting EdgeCooper-HD as an edge-assisted/global-assignment capability that can retain an AP@0.7 advantage.

### CAV Count Scaling

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，`potential_game`，SGCP inter-cluster late fusion。该表使用同一 20-CAV dump 的数值排序前 `N` 个 CAV 子集，并固定 `ego_cav_id=1`；这是离线规模敏感性 smoke test，不等同于重新生成的不同交通密度场景。

| CAV Count | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Cluster-Head Sources | Avg. Clusters | Reconfig. Count | Vehicle-Head Changes | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0.33 | 0.32 | 0.18 | 113670.63 | 9320992 | 82 | 2.00 | 6 | 24 | Small CAV subset, limited spatial coverage |
| 10 | 0.63 | 0.59 | 0.31 | 165169.30 | 20315824 | 123 | 3.00 | 3 | 14 | Better AP, larger per-source payload |
| 15 | 0.69 | 0.66 | 0.34 | 130304.62 | 26712448 | 205 | 5.00 | 18 | 71 | More clusters and reconfiguration |
| 20 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 246 | 6.00 | 11 | 76 | Full current dump |

### Network Resource

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，`potential_game`，SGCP inter-cluster late fusion。`--num-channels` 覆盖 PPS 子信道数量；`--bandwidth-mhz` 覆盖 PPS 总带宽。当前离线口径不启动 NS3，通信开销记录为实际上传点云 payload。

| Num. Channels | Bandwidth (MHz) | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Avg. Clusters | Reconfig. Count | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 40 | 0.56 | 0.53 | 0.27 | 60225.24 | 14815408 | 1.83 | 45.58 | 6.00 | 11 | Fewer channels, PPS admits fewer members per cluster head |
| 10 | 40 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | 6.00 | 11 | Current default |
| 20 | 40 | 0.77 | 0.73 | 0.38 | 139299.64 | 34267712 | 3.33 | 117.18 | 6.00 | 11 | More channels increase payload and AP@0.7 |

Coverage-aware spatial-diverse selection under the same channel sweep:

| Num. Channels | Bandwidth (MHz) | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 40 | 0.56 | 0.53 | 0.27 | 60225.24 | 14815408 | 1.83 | 45.58 | Same as default under severe channel bottleneck |
| 10 | 40 | 0.79 | 0.75 | 0.37 | 116842.60 | 28743280 | 2.67 | 87.32 | Coverage-aware selection improves over utility/random |
| 20 | 40 | 0.80 | 0.76 | 0.41 | 154116.03 | 37912544 | 3.33 | 117.18 | Near full-cluster AP@0.7 with lower payload |

| Num. Channels | Bandwidth (MHz) | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Avg. Clusters | Reconfig. Count | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 20 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | 6.00 | 11 | Same as default in current offline PPS path |
| 10 | 40 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | 6.00 | 11 | Current default |
| 10 | 80 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | 6.00 | 11 | Same as default in current offline PPS path |

低带宽瓶颈触发实验：

| Num. Channels | Bandwidth (MHz) | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 0.1 | 0.26 | 0.22 | 0.09 | 0.00 | 0 | 1.00 | 0.00 | Bandwidth bottleneck; cluster heads only |
| 10 | 0.5 | 0.56 | 0.50 | 0.23 | 39694.05 | 9764736 | 2.44 | 4.32 | Partial recovery under severe bandwidth limit |
| 10 | 1.0 | 0.66 | 0.61 | 0.31 | 75639.67 | 18607360 | 2.61 | 9.66 | Higher bandwidth admits more grids |
| 10 | 20.0 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | Saturated for this dump |
| 10 | 40.0 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | Current default |
| 10 | 80.0 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 | Same as default |

观察：子信道数量会改变 PPS 选择的簇内上传成员数和 selected grids，并直接影响通信-精度折中。20/40/80 MHz 没有差异，是因为当前 41 帧 dump 下实际调度未受带宽上限约束；当带宽降至 0.1/0.5/1.0 MHz 后，`bandwidth_per_channel` 瓶颈被触发，selected grids 和 AP 随带宽提升而恢复。论文级网络资源实验可以保留两段式叙述：常规 DSRC/NR-V2X 带宽下该场景由子信道数量主导，极低带宽压力测试证明 PPS 吞吐约束可生效。

## `f(rho)` 标定结果

当前已新增 `opencda.tools.sgcp_density_calibration`，可从 dump 数据重建与 SGCP replay 相同的 LiDAR grid density，并输出 `f(rho)` 标定 CSV。详细协议见 `f_rho_calibration.md`。

命令：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_density_calibration --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --thresholds "0.5,1.0,2.0,3.0,4.0" --output-dir docs\doc_workspace\SGCP\artifacts\density_calibration_41f
```

数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，788,020 个 CAV-grid density samples。

| Metric | Value |
| --- | ---: |
| Nonzero grid ratio | 0.059794 |
| Mean density, all grids | 0.050816 |
| P99 density, all grids | 0.830000 |
| Mean density, nonzero grids | 0.849855 |
| P90 density, nonzero grids | 1.400000 |
| P95 density, nonzero grids | 3.600000 |
| P99 density, nonzero grids | 13.255600 |

| `rho_th` | High-Density Grids | Ratio / All Grids | Ratio / Nonzero Grids | Mean `f(rho)` |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 11,232 | 0.014253 | 0.238375 | 0.383800 |
| 1.0 | 6,481 | 0.008224 | 0.137545 | 0.275282 |
| 2.0 | 3,383 | 0.004293 | 0.071797 | 0.124640 |
| 3.0 | 2,587 | 0.003283 | 0.054904 | 0.051639 |
| 4.0 | 2,192 | 0.002782 | 0.046521 | 0.021290 |

观察：默认 `rho_th=2.0` 位于当前非零网格 density 的 p90 和 p95 之间，筛出约 7.18% 非零网格作为 high-density candidates。结合前述 `rho_th` AP/payload sweep，`rho_th=2.0` 可作为当前 detector / LiDAR / 10 m grid 设置下的经验折中点；不能写成跨场景通用常数。后续仍需补不同场景和 detector metadata 泛化。

## Control Overhead

详细估算口径见 `control_overhead.md`。当前 `opencda.tools.offline_replay` summary 会输出 SGCP 控制面开销，包括 beacon、density metadata、cluster membership 和 PPS schedule command。

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

| Component | Total Bytes | Avg. Bytes / Frame |
| --- | ---: | ---: |
| Beacon | 52,480 | 1,280.00 |
| Density metadata | 40,184 | 980.10 |
| Cluster control | 3,608 | 88.00 |
| PPS schedule | 90,840 | 2,215.61 |
| Total control | 187,112 | 4,563.71 |

观察：对应的 SGCP inter-cluster late-fusion 点云 payload 为 26,916,208 bytes；控制面估算为 payload 的约 0.70%。论文中应将控制信令作为单独轻量 overhead 报告，不应混入点云 payload，也不应忽略。

## PPS Convergence Diagnostics

当前 `opencda.tools.offline_replay` summary 会输出 `PotentialGame` / PPS 经验收敛统计。该统计用于支撑 “potential-guided constrained best-response scheduling” 的有限收敛叙述；它不是完整 exact-potential 证明。

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

结果：

| Metric | Value |
| --- | ---: |
| Frames converged before `max_iter=20` | 41 / 41 |
| Avg. iterations | 3.00 |
| Max iterations | 3 |
| Avg. cluster updates / frame | 10.00 |
| Avg. scheduled links / frame | 10.00 |
| Avg. selected grids / frame | 523.90 |
| Avg. used RBs / frame | 10.00 |
| Avg. reused RBs / frame | 0.00 |
| Max RB occupancy | 1 |

观察：当前默认 20-CAV / 10-subchannel dump 中，PPS 每帧 3 轮内停止，41/41 帧均在 `max_iter=20` 前收敛；10 条 scheduled links 使用 10 个不同 RB，因此没有触发 RB 复用。该结果也解释了修复后的 NS3 10-subchannel replay 为什么能做到 110/110 request complete：OpenCDA 侧 PPS 本身输出的是无冲突 manual subchannel allocation。

## SGCP 离线 NS3 Request-Level 统计

### Potential-game scheduled requests after manual subchannel fix

命令：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 90s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=2.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\eval --max-frames 11
```

说明：该结果使用修复后的 NS3 manual subchannel scheduler，并将 `offline_ns3_replay` 的 SGCP 资源分配从旧 `NaiveRA` 对齐为 `potential_game`。每帧 6 个 cluster，PPS 从 14 条 member-to-head 需求中调度 10 条有子信道的 request，另 4 条未调度需求不发送给 NS3，避免绕过 OpenCDA 调度进入 NS3 默认调度。

| NS3 Target Subchannels | Frames | Scheduled Requests | Skipped Unscheduled | Planned Bytes | CAM Received | CAM Delivery Ratio | Avg. Delay (ms) | P95 Delay (ms) | PHY Failures | RLC TX Events | RLC RX Events | RLC Complete | RLC Partial | RLC No TX/RX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 11 | 110 | 44 | 1,100,000 | 110 | 1.000000 | 23.909 | 24.000 | 0 | 2,970 | 2,970 | 110 | 0 | 0 |
| 5 | 11 | 110 | 44 | 1,100,000 | 55 | 0.500000 | 23.909 | 24.000 | 0 | 1,485 | 1,485 | 55 | 0 | 55 |

### Spatial-diverse high-budget scheduled requests

命令：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 90s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=2.5 --enableTimeSync=true --carlaHost=auto --targetSubchannels=20'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --num-channels 20 --sgcp-grid-selection-mode spatial_diverse --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\eval --max-frames 11
```

说明：`offline_ns3_replay` 已支持 `--num-channels`、`--bandwidth-mhz`、`--sgcp-grid-score-mode` 和 `--sgcp-grid-selection-mode`，用于让 NS3 replay 与离线感知主表候选保持同一 SGCP/PPS 资源窗口。`spatial_diverse` 改变的是每条已调度 link 内的 grid 选择，不改变 NS3 transfer request 的 source/target/subchannel；NS3 结果用于验证该高预算候选的 request-level 可交付性。

| SGCP Variant | NS3 Target Subchannels | Frames | Scheduled Requests | Skipped Unscheduled | Planned Bytes | CAM Received | CAM Delivery Ratio | Avg. Delay (ms) | P95 Delay (ms) | PHY Failures | RLC TX Events | RLC RX Events | RLC Complete | RLC Partial | RLC No TX/RX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial-diverse, 10-channel | 10 | 11 | 110 | 44 | 1,100,000 | 110 | 1.000000 | 23.909 | 24.000 | 0 | 2,970 | 2,970 | 110 | 0 | 0 |
| Spatial-diverse, 10-channel, `rho_th=3` | 10 | 11 | 110 | 44 | 1,100,000 | 110 | 1.000000 | 23.909 | 24.000 | 0 | 2,970 | 2,970 | 110 | 0 | 0 |
| Spatial-diverse, 10-channel, `rho_th=3`, `B_h=2` | 10 | 11 | 110 | 44 | 1,100,000 | 110 | 1.000000 | 23.909 | 24.000 | 0 | 2,970 | 2,970 | 110 | 0 | 0 |
| Spatial-diverse, 20-channel | 20 | 11 | 154 | 0 | 1,540,000 | 154 | 1.000000 | 23.909 | 24.000 | 0 | 4,158 | 4,158 | 154 | 0 | 0 |

Trace diagnostics：10 子信道 low-budget 候选使用 `sc_start=0..9`，每个子信道各 11 条 planned request；`rho_th=3` 的 10 子信道 tuned low-budget 候选保持同一 request-level 调度形态；`B_h=2,rho_th=3` 在 10 子信道全局窗口下同样形成 110 条 request、44 条 skipped unscheduled demand，并使用 `sc_start=0..9` 每个子信道各 11 条 planned request；20 子信道 high-budget 候选使用 `sc_start=0..13`，每个子信道各 11 条 planned request。10 子信道 replay 中 `MANUAL_RESOURCE_APPLY=2970`、`MANUAL_CMD_REJECT=0`、`PSCCH_DECODE_FAIL=0`、`PSSCH_DECODE_FAIL=0`；20 子信道 replay 中 `MANUAL_RESOURCE_APPLY=4158`、`MANUAL_CMD_REJECT=0`、`PSCCH_DECODE_FAIL=0`、`PSSCH_DECODE_FAIL=0`。该结果确认：`spatial_diverse` 的低通信、tuned low-budget、`B_h=2` high-IoU sensitivity 和高预算候选都在 NS3 暴露窗口内完整收发；10 子信道下的 44 条未调度需求在 OpenCDA replay 侧跳过，没有绕过 PPS 进入 NS3。

10 子信道结果：`sc_start=0..9` 每个子信道各 11 条 planned request；NS3 trace 中 `MANUAL_RESOURCE_APPLY=2970`、`MANUAL_CMD_REJECT=0`、`PSCCH_DECODE_FAIL=0`、`PSSCH_DECODE_FAIL=0`。该结果确认：在修复后的 NS3 中，SGCP PPS 已调度、带宽范围内、无冲突的 request 可以完整收发。

5 子信道结果：NS3 只向 OpenCDA 暴露 `targetSubchannels=5`，因此 `sc_start=0..4` 共 55 条 request 全部 complete，`sc_start=5..9` 共 55 条 request 全部 no_tx/no_rx；NS3 trace 中 `MANUAL_CMD_REJECT=55`、`MANUAL_RESOURCE_APPLY=1485`、无 PHY decode failure。该结果确认：超出暴露带宽/子信道窗口的 request 在 bridge 层被拒绝，不进入 CAM/RLC，也不会污染后续合法 request。

### Legacy all-member replay diagnostic

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --drain-seconds 0.5 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_eval --max-frames 11
```

说明：该实验是早期诊断口径，使用旧 `offline_ns3_replay` 行为：每帧 6 个 cluster、14 条 intra-cluster transfer request，每条 10,000 bytes，其中部分 request 没有 SGCP PPS 子信道分配，会落入 NS3 默认调度路径。该结果保留用于解释历史问题，不作为修复后的 SGCP-PPS NS3 主结果。

| Frames | Planned Requests | Planned Bytes | CAM Received | CAM Delivery Ratio | Avg. Delay (ms) | P95 Delay (ms) | RLC TX Events | RLC RX Events | Requests With Any RLC RX | Any RLC RX Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 154 | 1,540,000 | 86 | 0.558442 | 26.756 | 28.000 | 4,158 | 2,512 | 150 | 0.974026 |

观察：application callback delivery ratio 明显低于 any-RLC-RX ratio，和 LGCP 侧观察一致。因此后续论文叙事中应区分 bridge-observed application callback、RLC partial reception、RLC request completion、PHY decode diagnostics，不能用单一 `cam_received` 比例代表全部链路可靠性，也不能把 any-RLC-RX 解释为完整 request delivery。

## 实时性结果

详细写作口径见 `runtime_feasibility_revision.md`。当前结果来自 41 帧离线 replay，不启动 CARLA/NS3；`SGCP algorithm total` 不含离线文件 I/O、OpenCOOD detector inference 和真实传输等待。

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

日志：

```text
docs\doc_workspace\SGCP\artifacts\runtime_breakdown_41f\offline_replay_runtime.log
```

| Stage | Mean (ms) | Max (ms) | Online cycle? | Notes |
| --- | ---: | ---: | --- | --- |
| Dump frame loading | 448.40 | 513.31 | No | Offline PCD/YAML replay artifact |
| Offline world build | 151.33 | 199.34 | No/partial | Offline adapter rebuilds manager state |
| Coalition formation | 64.39 | 82.32 | Yes | `CoalitionGame.run()` |
| Post-cluster state update | 0.24 | 0.44 | Yes | Apply cluster state and topology state |
| PPS scheduling | 40.58 | 53.05 | Yes | `PotentialGame` resource allocation |
| Control overhead accounting | 0.03 | 0.05 | No | Paper accounting only |
| SGCP algorithm total | 105.24 | 127.58 | Yes | Control-plane prototype, excluding perception inference |
| Offline total | 704.97 | 789.68 | No | Includes replay file I/O and world rebuild |

观察：当前 Python 原型的 control-plane 平均 105.24 ms，接近但略高于 100 ms 协作周期，因此论文中应写为 near-real-time feasibility，而不是完整端到端 100 ms 保证。PPS 本身平均 40.58 ms，41/41 帧在 3 轮内收敛；主要优化空间在 coalition formation。已接入的 topology-trigger gate 可作为机制解释：在线执行时 cluster membership 不必每个 sensing cycle 重算，只有 topology/stability trigger 或 periodic guard 触发时才支付该成本。

## Appendix Support Summary

已将 runtime、control overhead、PPS convergence 和 NS3 request-level reliability 收束为附录证据包：

```text
docs\doc_workspace\SGCP\artifacts\appendix_support_20260719\runtime_control_ns3_appendix.md
docs\doc_workspace\SGCP\artifacts\appendix_support_20260719\runtime_control_ns3_summary.csv
docs\doc_workspace\SGCP\artifacts\appendix_support_20260719\qualitative_case_study.md
```

建议写作口径：

- 主文只保留短句：PAPG scheduled requests 在 11 帧 NS3 replay 中 110/110 application callback 与 RLC complete，0 PHY failures；控制面 Python prototype 平均 105.24 ms，接近 100 ms 周期。
- 附录表格报告详细分解：coalition formation 64.39 ms、PPS scheduling 40.58 ms、control metadata 187,112 bytes / 4,563.71 bytes/frame。
- 不写成 full end-to-end 100 ms guarantee；offline frame loading/world build、OpenCOOD detector inference 和在线 callback 开销不包含在 105.24 ms 中。
- 控制 metadata 相对 PAPG main raw payload `32,049,872 bytes` 约为 0.58%，相对旧 potential-game payload `26,916,208 bytes` 为 0.70%；论文中应统一写 “below 1% of perception payload”。
- Qualitative case study 第一版选取 `000068/438`、`000066/401`、`000062/337` 三个持续漏检案例，用表格说明 GT grid、cluster head、selected sender/grid、full reference detection 和 constrained method miss；下一步若放入论文，应渲染 BEV overlay。

## 数据集导出验证

| Dataset Path | CAVs | Frames / CAV | PCD Files | YAML Files | Offline Inference |
| --- | ---: | ---: | ---: | ---: | --- |
| `D:\Data\Carla\2026_07_15_01_26_56` | 20 | 41 | 820 | 821 | `000060`: 62 pred boxes, 71 GT boxes |

## 离线无 NS3 测试

| Dataset Path | Fusion | Frames | Ego CAV | AP@0.3 | AP@0.5 | AP@0.7 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `D:\Data\Carla\2026_07_15_01_26_56` | Early | 41 | 1 | 0.85 | 0.83 | 0.48 | No NS3, no online CARLA sensor stream |
| `D:\Data\Carla\2026_07_15_01_26_56` | Late | 41 | 1 | 0.91 | 0.85 | 0.51 | Full 20-CAV OpenCOOD late model; reference only |

## 在线 CARLA+NS3 短回归

说明：该表用于记录真实联仿 smoke/regression，不作为论文主表来源。论文主表仍采用 41 帧离线 mAP 和离线 NS3 request-level replay；在线短回归的价值是验证 CARLA tick、OpenCDA network slot、NS3 sync time、manual subchannel request 和真实接收链路没有明显协议漂移。

| Artifact | Ticks | Sync Req/Ack | Manual Add/Reject | CAM Callback Lines | Complete / Partial Episodes | PSCCH Fail | PSSCH Fail | Online AP@0.3 | Online AP@0.5 | Online AP@0.7 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `online_ns3_short_fixed_20260717_031703` | 35 | 38/38 | 158/0 | 137 | 14 / 8 | 1836 | 480 | 0.86 | 0.84 | 0.74 | Vehicle-registration gate fixed; scheduler stale strategy still present |
| `online_ns3_short_strategyclear_20260717_041313` | 35 | 38/38 | 156/0 | 150 | 21 / 6 | 95 | 10 | 0.88 | 0.88 | 0.79 | Scheduler strategy clear fixed; remaining partial episodes each miss one 10000-byte fragment |
| `opencda_20260717_161909.log` | 38 slots | observed to 1.90 s | N/A | N/A | 0 / 11 app episodes | N/A | N/A | 0.86 | 0.86 | 0.71 | User online run; AP is high but `cp counter=1`, parsed as 3 CP eval/submit frames and 185 CP wait frames. Total counted traffic 4,495,080 bytes = 9.46 Mbps over 3.8 s; try upload 3,367,776 bytes = 7.09 Mbps. Diagnostic only until fixed-tick rerun confirms stable CP count. |

## SGCP 约束感知评估

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --max-frames 0
```

说明：当前结果使用 `ego-cluster-head` receiver policy，即当 `ego_cav_id=1` 不是 cluster head 时，评估其所在 cluster 的 head；只包含 intra-cluster grid-constrained early fusion，尚未包含 inter-cluster late fusion。因此它是 SGCP 约束感知链路的工程基线，不直接等同论文完整 SGCP 主结果。

| Dataset Path | RA Algorithm | Receiver Policy | Frames | Samples | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/sample) | Total Upload (bytes) | Avg. Source CAVs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game` | `ego-cluster-head` | 41 | 41 | 0.35 | 0.35 | 0.21 | 106790.63 | 4378416 | 2.98 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game` | `all-cluster-heads` | 41 | 246 | 0.36 | 0.34 | 0.17 | 109415.48 | 26916208 | 2.67 |

## SGCP 跨簇晚期融合评估

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --max-frames 0
```

可用参数覆盖：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --n-max 2 --max-frames 0
```

说明：该口径更接近论文 SGCP。每帧先对所有 cluster head 执行 intra-cluster grid-constrained early fusion，并统一投影到 `ego_cav_id=1` 的 lidar pose，再对所有簇头预测框执行 simple late fusion/NMS，最终每帧提交一次 AP 统计。

| Dataset Path | RA Algorithm | Frames | Cluster-Head Sources / Frame | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs / Cluster Head |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game` | 41 | 6 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `fixed_first_frame clustering` | 41 | 6 | 0.73 | 0.70 | 0.33 | 107013.07 | 26325216 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `spatial_diverse`, `B_h=2`, `rho_th=2` | 41 | 6 | 0.75 | 0.72 | 0.41 | 110107.32 | 27086400 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `spatial_diverse`, `B_h=2`, `rho_th=3` | 41 | 6 | 0.76 | 0.72 | 0.42 | 113670.18 | 27962864 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `spatial_diverse`, `B_h=2`, `rho_th=3`, late NMS 0.05 | 41 | 6 | 0.73 | 0.70 | 0.40 | 113670.18 | 27962864 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `spatial_diverse`, `B_h=2`, `rho_th=3`, late NMS 0.30 | 41 | 6 | 0.75 | 0.71 | 0.41 | 113670.18 | 27962864 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `T_min_stab=0` | 41 | 6 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `singleton clustering` | 41 | 20 | 0.82 | 0.76 | 0.37 | 0.00 | 0 | 1.00 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `random` | 41 | 6 | 0.44 | 0.39 | 0.17 | 39534.05 | 9725376 | 1.51 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `mws` | 41 | 6 | 0.31 | 0.26 | 0.11 | 40284.68 | 9910032 | 1.50 |

说明：`random` 与 `mws` 当前作为 “w/o PPS / baseline scheduler” 第一版结果。两者通信开销显著低于 `potential_game`，但 mAP 也明显下降；当前 `mws` 结果低于 `random`，后续进入论文前需要复核 MWS 效用定义与论文 baseline 是否一致。

### SGCP coverage diagnostics

说明：以下表格不作为 AP 主表，而是解释 `B_h=2` 为什么不能直接替换 10ch 主行。诊断来自 `opencda.tools.sgcp_late_fusion_log_summary` 和 `opencda.tools.sgcp_trace_coverage_summary`。

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Fused CAVs / Frame | Avg. Uploaded CAVs / Frame | Avg. Fused GT | Avg. Fused Pred. | Avg. Uploaded Points / Frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial-diverse, 10ch, `rho_th=2`, `B_h=1` | 0.79 | 0.75 | 0.37 | 16.00 | 10.00 | 69.00 | 55.90 | 43,815.98 |
| Spatial-diverse, 10ch, `rho_th=3`, `B_h=2` | 0.76 | 0.72 | 0.42 | 16.00 | 10.00 | 64.83 | 53.71 | 42,626.32 |
| Spatial-diverse, 20ch, `rho_th=2`, `B_h=1` | 0.80 | 0.76 | 0.41 | 20.00 | 14.00 | 69.29 | 56.24 | 57,793.51 |

`B_h=2` 在 10ch 下没有增加 fused CAV 数，仍只融合 16/20 个 CAV；它主要替换了具体上传成员。最明显的是 CAV 6 从 `B_h=1` 的 41 帧上传降到 `B_h=2,rho_th=3` 的 7 帧，而 CAV 5 从 6 帧升到 31 帧。该替换与 fused GT 下降一致，因此 `B_h=2` 更适合写为 high-IoU sensitivity，而不是当前主表默认行。

Persistent coverage fallback 的 11 帧负面 probe 进一步说明，单纯按 CAV 历史欠覆盖替换成员不是有效修复：同一 11 帧上 `B_h=2,rho3` 无 fallback 为 AP `0.69/0.64/0.34`、7,416,720 bytes，persistent fallback 为 AP `0.67/0.62/0.34`、7,453,808 bytes，且 `missing_channel_rows=0`。因此后续 fallback 必须绑定 detector-quality 或 target-level coverage proxy。

Detector-quality proxy 也支持这个判断：41 帧 `B_h=2,rho3` 的 receiver-level average pred/GT ratio 为 0.4461，高于 `B_h=1` 10ch 的 0.3928，解释了高 IoU 提升；但 CAV 6 这一高质量长期贡献者的上传从 41 行降到 7 行，其 avg pred/GT ratio 为 0.6341/0.5746，明显高于被增加的 CAV 5 的 0.3129/0.3893。因此后续算法应采用 quality-weighted coverage，而不是 plain coverage fairness。

Quality-persistent fallback 的 11 帧 safety probe 表明质量门槛可以阻止有害替换，但还不能带来收益：`B_h=2,rho3,quality_persistent` 为 AP `0.69/0.64/0.34`、7,416,720 bytes、0 次 replacement，等同 no fallback。下一步需要 object/target-aware 候选生成。

### Target-grid case study / object-aware PG probe

该结果用于机制诊断，不作为当前主表。目标是确认漏检 GT 对应的 grid、最佳 CAV 和调度 action 是否能被新算法打通。

| Case | Frame | GT Grid | Original Failure | Object-aware PG Behavior |
| --- | --- | --- | --- | --- |
| Object 438 | `000068` | `3_0` | CAV12 有 424 点、rank=1，但 head4 调度 CAV9；CAV9 在该 grid 为 0 点 | 同 RB sender refinement 将 head4 sender 换为 CAV12，选中 `3_0` |
| Object 401 | `000066` | `2_0` | CAV4 有 891 点、rank=4，但 head12 调度 CAV7；CAV7 仅 7 点 | 调度 CAV4 并选中 `2_0` |
| Object 350 | `000084` | `1_-2` | CAV8 有 3371 点、rank=1，但 head1 只收到 CAV2/CAV11 的稀疏点 | 调度 CAV8 并选中 `1_-2` |
| Object 337 | `000062` | `0_-3` | head 自身高密度但 peer view 未作为 target candidate，近身/盲区目标缺少 multi-view confirmation | OAPG 将 head 高密度 + peer 中等密度 grid 纳入 candidate，但仍需继续调优 sender diversity |

11 帧快速检测结果：

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Avg. source CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `object_aware_potential_game`, 20MHz/10ch/rho3 | 11 | 0.74 | 0.69 | 0.30 | 8,209,376 | 2.64 | 73.48 |

结论：OAPG 机制上修复了若干明确的“最佳视角未调度”失败，但当前 AP 尚未超过 `target_aware_potential_game` / `spatial_diverse` 主表候选。后续应继续做 41 帧评估、sender replacement 限制和 detector-quality gate；暂不把 OAPG 写入主表。

## Online CARLA/NS3 Alignment Check

命令口径：

```powershell
$env:OPENCDA_ONLINE_TICKS = "80"
$env:OPENCDA_CLEAN_WORLD_ON_INIT = "1"
$env:OPENCDA_CARLA_CLIENT_TIMEOUT = "180"
$env:OPENCDA_USE_CURRENT_CARLA_WORLD = "1"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
```

NS3 口径：`targetSubchannels=10`、`enableTimeSync=true`。在线 Mbps 使用 `total_slots * time_slot`，并同时报告总计流量和 intra-cluster try upload。

| Online Variant | CP Submit | Complete / Partial Episodes | Sync Timeout | AP@0.3 | AP@0.5 | AP@0.7 | Total Payload Mbps | Try Payload Mbps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| User run before strict barrier | 4 | 26 / 9 | N/A | 0.51 | 0.48 | 0.41 | 18.51 | 15.83 | CP deadline consumption too sparse |
| Strict barrier + `min_upload_count=1` + one reupload | 10 | 55 / 2 | 0 | 0.70 | 0.68 | 0.58 | 25.48 | 17.85 | Current best online CARLA/NS3 alignment result |
| Strict barrier + `min_upload_count=1` + no reupload | 7 | 45 / 3 | 0 | 0.64 | 0.59 | 0.50 | 23.94 | 19.52 | Fewer PHY overlaps but worse deadline delivery |

结论：在线 CARLA/NS3 已经消除时间流速不一致导致的 sync timeout，并确认 OpenCDA 指定子信道真实落到 NS3 发送行为。在线 AP 仍不应直接与离线“最终 request complete”主表混用；论文中应额外声明 deadline-aware online CP delivery，即 request 必须在当前融合周期截止前完整或部分可用，才会影响该帧 AP。

## PACP-style LiDAR priority baseline

原 PACP 论文是 RGB/BEV 协作感知：使用多相机 perception、SinBEVT/CoBEVT BEV feature、BEV-match priority 和 adaptive autoencoder 压缩 camera data。当前结果不是 PACP 原方法严格复现，而是把 BEV-match priority-aware scheduling 思路迁移到当前 raw LiDAR grid sharing pipeline，代码入口为 `offline_inference --selective-sharing-baseline pacp_lidar`。

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline pacp_lidar --selective-member-budget 3 --selective-grid-budget 117 --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pacp_lidar_41f_trace.csv
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline pacp_lidar --selective-member-budget 2 --selective-grid-budget 87 --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pacp_lidar_2m87g_41f_trace.csv
```

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pacp_lidar`, 3 members/head, 117 grids/head | 41 | 0.81 | 0.79 | 0.42 | 44,361,424 | 86.56 | 3.33 | 104.93 |
| `pacp_lidar`, 2 members/head, 87 grids/head | 41 | 0.76 | 0.73 | 0.37 | 34,498,160 | 67.31 | 2.81 | 81.83 |

Dry-run NS3 plan:

- High-budget `pacp_lidar`: 11 frames, 110 scheduled requests, 44 skipped unscheduled demands.
- Low-budget `pacp_lidar`: 11 frames, 110 scheduled requests, 9 skipped unscheduled demands.

结论：PACP priority idea 可以迁移到点云通信场景，但 raw LiDAR grid payload 显著高于原 RGB feature/image-compression setting。高预算版本 AP@0.7 强，但 Mbps 高于 PAPG 与 EdgeCooper-HD；低预算版本接近公平通信量时 AP 低于 PAPG。因此它可作为近年 V2V priority-aware proxy baseline，不宜作为“严格 PACP”或 SGCP 主线替代。

## 离线 SGCP 回放稳定性与耗时

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --summary-only
```

说明：`potential_game` 是当前配置默认的 SGCP 资源分配算法；`naive` 保留为 baseline/fallback。当前表格是工程回放结果，尚未接入 OpenCOOD 的 SGCP 约束感知 mAP 评估，因此不直接作为论文主结果。

| Dataset Path | RA Algorithm | Frames | CAVs | Avg. Clusters | Avg. Cluster Size | Avg. Isolated CAVs | Reconfig. Events | Vehicle-Head Changes | Avg. Cluster Lifetime (frames) | Avg. Runtime (ms) | Avg. RA Runtime (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game` | 41 | 20 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 285.82 | 111.85 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `T_min_stab=0` | 41 | 20 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.99 | 37.39 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `singleton clustering` | 41 | 20 | 20.00 | 1.00 | 20.00 | 0 | 0 | 41.00 | 4.52 | 3.92 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `naive` | 41 | 20 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 169.94 | 0.50 |

## Topology Trigger 离线统计

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --summary-only --print-topology-events
```

说明：该统计从连续 dump 帧重建 CAV 位置、速度、邻居集合和 cluster head/member 关系。`offline_replay` 当前默认使用 `pose_delta`，即相邻帧位置差分速度，避免直接混用 dump 中以 km/h 表示的 `ego_speed`。

| Dataset Path | Frames | Transitions | Triggered | Actual Reconfig. | Matched | Reconfig. Without Trigger | Trigger Without Reconfig. | Trigger Type Counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `D:\Data\Carla\2026_07_15_01_26_56`, `dump ego_speed`, threshold 5 km/h | 41 | 40 | 40 | 11 | 11 | 0 | 29 | `relative_speed_risk`: 40; `neighbor_set_change`: 12 |
| `D:\Data\Carla\2026_07_15_01_26_56`, `pose_delta`, threshold 3 m/s | 41 | 40 | 40 | 11 | 11 | 0 | 29 | `relative_speed_risk`: 40; `neighbor_set_change`: 12 |
| `D:\Data\Carla\2026_07_15_01_26_56`, `pose_delta`, threshold 4 m/s | 41 | 40 | 40 | 11 | 11 | 0 | 29 | `relative_speed_risk`: 40; `neighbor_set_change`: 12 |
| `D:\Data\Carla\2026_07_15_01_26_56`, `pose_delta`, threshold 5 m/s | 41 | 40 | 37 | 11 | 9 | 2 | 28 | `relative_speed_risk`: 37; `neighbor_set_change`: 12 |

观察：`pose_delta` 速度源解决了单位歧义，但单靠 relative-speed trigger 仍偏敏感。在线 gate 不宜直接采用“relative speed 任意超阈即重构”，更适合与 neighbor-set change、utility drop 和 `T_min_stab` 滞回组合使用。

## Aggregate AP manifest protocol

用户已明确主文不引入 satisfaction rate，后续结果统一使用 aggregate AP + Mbps。Aggregate AP 是 OpenCOOD pooled evaluator AP，即把所有 evaluated receiver-frame samples 累计到同一个 evaluator 后计算 AP，不是 per-CAV AP 的简单平均。

新增 manifest 工具：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "PAPG=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1_trace.csv" --run "EdgeCooperHD=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1_trace.csv" --output-csv docs\doc_workspace\SGCP\artifacts\aggregate_ap_manifest_20260719\repeat_check_manifest.csv --notes "repeat check for aggregate AP manifest"
```

| Method | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Evaluated Samples | Trace Rows | Receiver Policy | Late Fusion | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| PAPG | 0.81 | 0.78 | 0.39 | 41 | 246 | all-cluster-heads | yes | 32,049,872 | 62.54 |
| EdgeCooper-HD | 0.81 | 0.78 | 0.42 | 41 | 246 | all-cluster-heads | yes | 33,519,040 | 65.40 |

生成的 manifest 路径：`docs\doc_workspace\SGCP\artifacts\aggregate_ap_manifest_20260719\repeat_check_manifest.csv`。后续 Table 1 / Table 2 / Table 3 / Pareto 图均应使用该 manifest 口径沉淀源数据。

## Pure late prediction-box communication budget

Pure late fusion 41 帧结果 `0.82/0.76/0.37` 不能继续只写作 `0 Mbps`，因为它不传 raw LiDAR，但需要传本地检测框。已新增 `opencda.tools.sgcp_late_box_comm_budget`，从 `pure_late_singleton_41f_trace.csv` 统计 detection-box payload、调度完成时间和无调度随机子信道冲突 proxy。

命令：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_late_box_comm_budget --trace-csv docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv --output-dir docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719 --box-bytes 80 --message-overhead-bytes 64 --packet-overhead-bytes 48 --mtu-bytes 1200 --total-bandwidth-mhz 20 --subchannels 10 --spectral-efficiency 6 --deadline-ms 100
conda run -n opencda python -m opencda.tools.sgcp_late_box_comm_budget --trace-csv docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv --output-dir docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719_box128 --box-bytes 128 --message-overhead-bytes 64 --packet-overhead-bytes 48 --mtu-bytes 1200 --total-bandwidth-mhz 20 --subchannels 10 --spectral-efficiency 6 --deadline-ms 100
```

| Assumption | Mode | Mean Mbps | Max Mbps | Mean Scheduled Completion | Deadline OK | Random-Access Full Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `80 B/box` | broadcast | 0.739 | 0.823 | 1.153 ms | 100% | 100% |
| `80 B/box` | all-to-all unicast | 14.043 | 15.638 | 19.102 ms | 100% | 0% |
| `128 B/box` | broadcast | 1.132 | 1.265 | 1.560 ms | 100% | 100% |
| `128 B/box` | all-to-all unicast | 21.515 | 24.028 | 27.336 ms | 100% | 0% |

结论：当前场景下，预测框交换不能靠 payload rate 或调度传输时延自然压低；有调度 all-to-all unicast 在 100 ms deadline 内也可完成。只有完全无调度的 all-to-all 随机抢信道会因碰撞失败。论文中应把 Pure late 写成 strong prediction-sharing reference，并显式说明它的低 payload 与信息内容限制，而不是声称 20-CAV late fusion 必然 broadcast storm。

### Actual late checkpoint sanity

补查发现：当前 Table 1 / Table 2 manifest 中的 Pure late 行 `fusion_method=early`，实现上是用 `pointpillar_early_fusion` 对 singleton CAV 做本地检测，再由 `OpenCOODManager.naive_late_fusion()` 做 box-level NMS。它不是 `pointpillar_late_fusion` checkpoint 的原生 late inference。

`pointpillar_late_fusion` 目录包含 `net_epoch30.pth`，`load_saved_model()` 会加载 epoch 30。使用真正 late checkpoint 的 sanity 结果如下：

| Variant | Frames | Fusion Method | AP@0.3 | AP@0.5 | AP@0.7 | Trace Rows | Raw LiDAR Mbps |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Pure late, early checkpoint singleton proxy | 11 | early | 0.78 | 0.72 | 0.32 | 220 | 0.00 |
| Pure late, actual late checkpoint | 11 | late | 0.90 | 0.84 | 0.46 | 220 | 0.00 |
| Pure late, actual late checkpoint | 41 | late | 0.89 | 0.83 | 0.49 | 820 | 0.00 |

Actual-late 41 帧 prediction-box overhead：

| Assumption | Broadcast Mean/Max Mbps | All-to-all Mean/Max Mbps | All-to-all Mean Scheduled Completion |
| --- | ---: | ---: | ---: |
| `80 B/box` | 1.068 / 1.148 | 20.298 / 21.815 | 25.072 ms |
| `128 B/box` | 1.654 / 1.782 | 31.431 / 33.853 | 38.906 ms |

结论：Pure late 过强不是因为误用了 early checkpoint；真正 late checkpoint 在当前场景下更强，甚至达到/略高于 full 20-CAV early upper reference `0.85/0.83/0.48`。这说明当前场景和模型组合下，box-level prediction sharing 是非常强的 reference。后续主表需要把 Pure late 从“baseline 被 SGCP 超越”的叙事中拆出来，作为 prediction-sharing upper/reference，或选择更能体现 raw LiDAR early fusion 恢复漏检能力的场景。

### Unified late-detector sanity

用户明确公平原则：SGCP 两层融合中，所有 “点云 -> 检测框” 过程应使用同一个 checkpoint；SGCP 的簇间 late fusion 和 Pure late fusion 都应使用 `naive_late_fusion()` 做 box-level NMS。

代码确认：

- SGCP inter-cluster late fusion：`offline_inference.py` 调用 `manager.naive_late_fusion()` 汇总每个 late source 的预测框。
- `naive_late_fusion()` 本质是 concatenation + torchvision NMS。
- 当前 SGCP 主线的第一层是 `fusion_method=early`，即 `pointpillar_early_fusion` raw point-cloud early fusion。

补跑 “all late detector” sanity：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --fusion-method late --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-inter-cluster-late-fusion --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\late_detector_unified_20260719\sgcp_papg_late_detector_41f_trace.csv
```

| Variant | Frames | Detector / First-stage Fusion | Box-level Fusion | AP@0.3 | AP@0.5 | AP@0.7 | Payload Mbps | Interpretation |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| Pure late actual late | 41 | `pointpillar_late_fusion` local detector | `naive_late_fusion()` | 0.89 | 0.83 | 0.49 | 0 raw LiDAR | Native prediction-sharing reference |
| SGCP PAPG forced late detector | 41 | `pointpillar_late_fusion` over scheduled source set | `naive_late_fusion()` | 0.87 | 0.81 | 0.48 | 62.54 | Sanity only; first stage is no longer raw point-cloud early fusion |
| SGCP PAPG mainline | 41 | `pointpillar_early_fusion` raw point-cloud early fusion | `naive_late_fusion()` | 0.81 | 0.78 | 0.39 | 62.54 | Actual SGCP protocol |

结论：如果都用 late checkpoint 的 local detector，SGCP 不能超过 Pure late；同时 forced SGCP late-detector row 不再代表论文中的 “簇内 early fusion + 簇间 late fusion”。因此主文公平策略应是：主线所有 raw point-cloud fusion baseline 统一使用 `pointpillar_early_fusion`；Pure late 作为 prediction-sharing reference 明确单列，或在 controlled ablation 中使用 `pointpillar_early_fusion` singleton detector + `naive_late_fusion()`。

## Pareto source second-pass consolidation

本轮未重跑新实验，而是将此前已经复现但未进入 Figure 1 源表的代表点补入 `docs\doc_workspace\SGCP\artifacts\pareto_20260719\pareto_source.csv`，用于支撑 AP-Mbps Pareto 曲线的 first-pass 参数覆盖。

新增源表点：

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| SGCPCoverage5ch20MHz | 0.56 | 0.53 | 0.27 | 28.91 | low-bandwidth stress |
| SGCPCoverage10chRho3Bh2 | 0.76 | 0.72 | 0.42 | 54.56 | SGCP B_h sensitivity |
| SelectiveCommunicationAwareLowBudget | 0.78 | 0.75 | 0.40 | 58.97 | V2V scheduler low-budget point |
| SGCP_PAPG_Bh3 | 0.80 | 0.78 | 0.40 | 62.54 | PAPG B_h sensitivity |
| PACP_LiDAR_LowBudget | 0.76 | 0.73 | 0.37 | 67.31 | PACP-style proxy low-budget point |

结论：当前 Pareto 源表已经足够支持“SGCP/PAPG 在 raw-LiDAR V2V 中等通信区间具有竞争力”的图形判断，但仍不能写成所有 baseline 全预算最优。正式论文 caption 需要分层解释 raw-LiDAR frontier、prediction-sharing Pure late reference 和 edge-assisted/global reference。

## Scheduler budget sweep for Pareto

为补齐 P4 中 `Random / Density / Link-aware` 的 member/grid budget sweep，本轮固定 41 帧、`rho_th=3`、all-cluster-heads、inter-cluster late fusion 和同一 clustered two-layer scaffold，补跑三组 selective-sharing baseline。

Manifest：`docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\scheduler_budget_sweep_manifest.csv`

| Method | Members/head | Grids/head | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomLowBudget | 2 | 87 | 0.75 | 0.70 | 0.34 | 24,772,192 | 48.34 |
| DensityLowBudget | 2 | 87 | 0.78 | 0.74 | 0.40 | 31,421,408 | 61.31 |
| CommunicationAwareHighBudget | 3 | 117 | 0.80 | 0.76 | 0.42 | 38,920,592 | 75.94 |

可读结论：random low-budget 是低通信/低 AP 端点；density low-budget 用接近 SGCP 的通信量获得较高 AP@0.7，但 AP@0.3/AP@0.5 仍低于 PAPG；communication-aware high-budget 在 AP@0.7 更强，但通信量高出 PAPG 约 21.4%。这支持论文中把 PAPG 写成中等通信区间的 raw-LiDAR V2V Pareto 候选，同时必须承认高预算 link-aware reference 的 AP@0.7 边界。

## EdgeCooper-HD budget sweep for Pareto

为补齐 P4 中 EdgeCooperV2V+ / EdgeCooper-inspired 的 sender cap、assignment budget、half-duplex constraint 扫描，本轮固定 41 帧、`rho_th=3`、all-cluster-heads、inter-cluster late fusion，使用 `edgecooper_global_hd` 作为 edge/global assignment + sender load cap + half-duplex proxy。

Manifest：`docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_budget_sweep_manifest.csv`

| Method | Members/head | Grids/head | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EdgeCooperHDLow | 1 | 58 | 0.65 | 0.61 | 0.30 | 18,501,232 | 36.10 |
| EdgeCooperHD | 3 | 117 | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 |

可读结论：EdgeCooper-HD 低预算端显著掉 AP，说明其高 AP@0.7 不是低通信量免费得到，而依赖 edge/global assignment + half-duplex sender cap 下仍保留较高 member/grid budget。与 PAPG 相比，EdgeCooper-HD 高预算在 AP@0.7 高 `+0.03`，但 payload 高约 `4.6%`，且属于 edge-assisted proxy/reference，而不是纯 V2V decentralized protocol。

## FullPerception-PCS parameter sweep

为补齐 P4 中 FullPerception-PCS 的原生参数扫描，本轮固定 20MHz/10ch、`fullperception_pcs`、`all-scheduled-receivers`，不改变主带宽或子信道数。41 帧 `div8/ov0` 与 `div12/ov1` 参数点均超过 10--15 分钟仍未产生日志/trace，说明 PCS 在更激进候选设置下存在明显运行时/候选规模边界；正式记录采用已经完成的 11 帧趋势 + 41 帧 tuned anchor。

Manifest：`docs\doc_workspace\SGCP\artifacts\pcs_parameter_sweep_20260719\pcs_parameter_sweep_manifest.csv`

| Variant | Frames | Division | Min overlap | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCS_11f_div4_ov1 | 11 | 4 | 1 | 0.58 | 0.51 | 0.24 | 3,668,304 | 26.68 |
| PCS_11f_div8_ov0 | 11 | 8 | 0 | 0.50 | 0.46 | 0.26 | 3,683,232 | 26.79 |
| PCS_11f_div12_ov0 | 11 | 12 | 0 | 0.57 | 0.54 | 0.30 | 4,513,424 | 32.82 |
| PCS_11f_div12_ov1 | 11 | 12 | 1 | 0.56 | 0.49 | 0.16 | 3,634,432 | 26.43 |
| PCS_11f_div16_ov1 | 11 | 16 | 1 | 0.42 | 0.40 | 0.20 | 540,160 | 3.93 |
| PCS_41f_div12_ov0 | 41 | 12 | 0 | 0.59 | 0.53 | 0.22 | 12,959,840 | 25.29 |

可读结论：`div12/ov0` 仍是当前 PCS baseline 的可写工作点。`min_overlap=1` 会显著降低 AP@0.7，`div16/ov1` 虽低通信但 AP 不可用；更细粒度/更宽候选的 41 帧运行不可承受。因此主表/Pareto 中保留 `PCS_41f_div12_ov0`，参数趋势放入附录或 rebuttal 支撑“PCS 已调参但受原机制限制”的边界。

## Detector / checkpoint fairness decision

已新增 `detector_checkpoint_fairness.md` 固化主文口径：SGCP raw-LiDAR 系列统一使用 `pointpillar_early_fusion` checkpoint。Pure late 主表行不切换到 `pointpillar_late_fusion` checkpoint，而是作为 controlled prediction-sharing reference：early-checkpoint singleton local inference + `naive_late_fusion()`。

| Variant | Frames | Detector / First-stage Fusion | Box-level Fusion | AP@0.3 | AP@0.5 | AP@0.7 | 用途 |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| Pure late controlled | 41 | `pointpillar_early_fusion` singleton local detector | `naive_late_fusion()` | 0.82 | 0.76 | 0.37 | 主表 prediction-sharing reference |
| Pure late actual-late sanity | 41 | `pointpillar_late_fusion` local detector | `naive_late_fusion()` | 0.89 | 0.83 | 0.49 | sensitivity/reference，不进公平 raw-LiDAR 主表 |
| SGCP PAPG mainline | 41 | `pointpillar_early_fusion` raw point-cloud early fusion | `naive_late_fusion()` | 0.81 | 0.78 | 0.39 | SGCP 主线 |
| SGCP PAPG forced-late sanity | 41 | `pointpillar_late_fusion` over scheduled source set | `naive_late_fusion()` | 0.87 | 0.81 | 0.48 | checkpoint sensitivity，不代表 SGCP 协议 |

结论：Pure late 仍是强 prediction-sharing reference，但不应和 raw-LiDAR SGCP/PAPG 写成同类点云通信 baseline。当前最大风险仍是 early checkpoint 偏弱；远程 fine-tune watcher 尚在等待 GPU 空闲。

## Paper number audit

已新增 `docs\doc_workspace\SGCP\artifacts\paper_number_audit_20260719\paper_number_audit.csv`。核查结论：

- `main.tex` Table 1、Table 3 和 Table 4 的 AP/Mbps 与当前 manifest 对齐。
- Pure late 例外已明确：manifest 中 raw-LiDAR payload 为 `0`，论文表格使用 `late_fusion_box_comm.md` 的 80 B/box one-hop broadcast estimate，即 `0.74 Mbps`。
- `table4_parameter_sensitivity.csv` 的 channel sweep 标签从 legacy `5/10/20 ch / 40 MHz` 修正为 `5/10/20 ch / 20 MHz`，对应当前复现实验命令中的 `--bandwidth-mhz 20`。
- `opencda.tools.sgcp_aggregate_ap_manifest` 已新增 `num_channels` / `bandwidth_mhz` 输出字段，后续新 trace 若包含网络元数据，manifest 不再丢失该口径。
- `protocol_native_manifest.csv` 和 `scheduler_comparison_manifest.csv` 已用更新后的 manifest 工具重生成，并通过 override 写入当前 paper scope 的 `10` subchannels / `20` MHz。

## Pareto claim audit

已新增 `pareto_claim_audit.md`，用 `artifacts/pareto_20260719/pareto_source.csv` 重新计算 raw-LiDAR V2V / SGCP-compatible 集合的 Pareto frontier。该集合只纳入 `proposed`、`sgcp_ablation`、`sgcp_sensitivity`、`scheduler_baseline` 和 `scheduler_baseline_proxy` 且 `scaffold=sgcp_compatible` 的结果；不把 Pure late prediction-box sharing、Edge/global reference 或 full-sharing upper reference 混入同一 frontier。

关键结论：

| Metric | SGCP-PAPG status | Boundary |
| --- | --- | --- |
| AP@0.3 | frontier point, 62.54 Mbps / 0.81 | 支撑 coverage / network-level recall claim |
| AP@0.5 | 同预算 frontier, 62.54 Mbps / 0.78 | 与 `B_h=3` sensitivity 同 AP@0.5；PACP-LiDAR high-budget 以 86.56 Mbps 达到 0.79 |
| AP@0.7 | not frontier | `B_h=2,rho3` sensitivity 以 54.56 Mbps 达到 0.42；说明 high-IoU localization 仍是 checkpoint/局部视角边界 |

可写结论：SGCP-PAPG 可在 raw-LiDAR V2V 中写 AP@0.3/AP@0.5 的中等通信 Pareto 优势；AP@0.7 不写全面最优，而写成 high-IoU sensitivity / early checkpoint headroom。

## Fusion scaffold claim audit

已新增 `fusion_scaffold_claim_audit.md`，用于收束 P2/P6 的两层融合叙事。

| Variant | Raw payload Mbps | AP@0.3 | AP@0.5 | AP@0.7 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Pure late controlled | 0.00 raw LiDAR | 0.82 | 0.76 | 0.37 | strong prediction-sharing reference |
| Full20Early / one-cluster early | 118.71 | 0.85 | 0.83 | 0.48 | full raw-sharing upper reference |
| Clustered early-only | 62.54 | 0.38 | 0.36 | 0.20 | cluster-local early-only coverage insufficient |
| Full SGCP | 62.54 | 0.81 | 0.78 | 0.39 | two-layer protocol |

Full SGCP 使用 52.7% full-sharing raw payload，保留 full-sharing AP@0.3/AP@0.5/AP@0.7 的 95.3%/94.0%/81.3%。这支持主文写 “low/mid-IoU coverage with much lower raw payload”；AP@0.7 仍作为 high-IoU localization/checkpoint headroom。

## Figure 2/3 readiness check

2026-07-19 视觉检查 Figure 2/3 后发现旧图中 Pure late 标为 `raw 0.0`，与 `late_fusion_box_comm.md` 和主文 prediction-box overhead 口径不一致。已修改 `artifacts/figures_20260719/plot_breakdowns.py`，为 Pure late 使用 `box 0.7` 的图内 communication label，并重生成：

- `artifacts/figures_20260719/figure2_protocol_breakdown.png/.pdf`
- `artifacts/figures_20260719/figure3_fusion_contribution.png/.pdf`

Figure 2 现在可以区分 Head-only、Pure late prediction-sharing、FullPerception-PCS、EdgeCooper-HD、SGCP-PAPG 和 Full20Early；Figure 3 可以清楚展示 clustered early-only 到 Full SGCP 的 coverage gain，以及 Full20Early 的 AP@0.7 上界。

## Scenario sufficiency audit

已新增 `scenario_sufficiency_audit.md`。当前 41 帧 `v2xp_cluster_carla` 离线场景足以支撑 first-pass 主文图表：

- Table 1 protocol-native comparison；
- Figure 1 AP-Mbps Pareto；
- Figure 2 protocol breakdown；
- Figure 3 fusion contribution；
- Table 3 scheduler comparison；
- Table 4 parameter sensitivity；
- runtime / NS3 appendix；
- qualitative case study draft。

当前不立即重新导出 CARLA 场景。后续只有在 early checkpoint 回收后仍无法支撑主张，或需要更强动态稳定性、不同 CAV 密度、在线端到端证据时，再按 `environment.md` 的 CARLA 约束启动新导出。

## Early checkpoint recovery protocol

已新增 `early_checkpoint_recovery.md`，把远程 fine-tune 从“等待 GPU”转成可执行回收流程：

- watcher：`mindspore-187:/data2/gzc/sgcp_early_train/runs/start_train_when_gpu_free.sh`
- 日志：`/data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log`
- 环境：`opencood-gzc`
- 配置：`/data2/gzc/sgcp_early_train/configs/pointpillar_early_ckpt_compat_onecav.yaml`
- 当前状态：8 张 GPU 均约 22.2GB used，watcher 仍在每 300 秒轮询。

训练完成后，必须先回收最新 step checkpoint，并在同一 checkpoint 下重跑 SGCP-PAPG 和 Pure late controlled baseline。只有在不破坏 AP@0.3/AP@0.5 且改善或解释 AP@0.7 时，才替换主文结果；否则作为 sensitivity/negative artifact 记录。

## Detector checkpoint probe: late / attentive / COSDH

本轮围绕“早期融合 checkpoint 偏弱”风险，固定 SGCP-PAPG 通信协议、41 帧 `2026_07_15_01_26_56` 场景、20MHz/10ch/rho3/`B_h=2`、all-cluster-heads 和 inter-cluster `naive_late_fusion()`，测试更强检测器权重能否作为 merged point-cloud detector。

Artifact：

- `docs\doc_workspace\SGCP\artifacts\checkpoint_sensitivity_20260719\detector_checkpoint_sensitivity_manifest.csv`
- `docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\`
- `docs\doc_workspace\SGCP\artifacts\cosdh_checkpoint_probe_20260719\`

| Variant | Frames | Detector source | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Raw Mbps | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original SGCP-PAPG smoke | 11 | `pointpillar_early_fusion` | 0.76 | 0.73 | 0.34 | 8,598,224 | 62.53 | baseline smoke |
| Late checkpoint as early detector | 11 | `pointpillar_late_fusion/net_epoch30.pth` copied into early config | 0.58 | 0.48 | 0.15 | 8,598,224 | 62.53 | reject |
| Attentive checkpoint as early detector | 11 | attentive intermediate checkpoint copied into early config | 0.85 | 0.77 | 0.32 | 8,598,224 | 62.53 | promising smoke |
| SGCP-PAPG attentive detector | 41 | attentive intermediate checkpoint copied into early config | 0.87 | 0.81 | 0.36 | 32,049,872 | 62.54 | sensitivity / candidate |
| Pure late attentive controlled | 41 | attentive checkpoint singleton local detector + `naive_late_fusion()` | 0.82 | 0.65 | 0.28 | 0 raw LiDAR | 0 raw LiDAR | prediction-sharing reference |
| Full20Early attentive detector | 41 | attentive intermediate checkpoint copied into early config | 0.88 | 0.85 | 0.45 | n/a | n/a | upper reference |
| COSDH compatible transplant | 11 | 140 compatible COSDH weights + original early fallback heads | 0.00 | 0.00 | 0.00 | 8,598,224 | 62.53 | reject |
| COSDH backbone + early heads | 11 | COSDH backbone, original early detection heads | 0.02 | 0.00 | 0.00 | 8,598,224 | 62.53 | reject |
| COSDH real model collapsed smoke | 1 | `point_pillar_comm_multiscale`, scheduled points collapsed to receiver cloud | n/a | n/a | n/a | 783,392 | n/a | runs, but 0 predictions |
| COSDH real model threshold diagnosis | 1 | same as above, `score_threshold=0.01/0.005/0.003` | n/a | n/a | n/a | 783,392 | n/a | logits too low; still 0 final boxes |

结论：

- 使用 attentive intermediate checkpoint 初始化/替换 early detector 对 AP@0.3/AP@0.5 很有帮助，说明当前主线的核心风险确实来自 early detector/checkpoint 强度。
- Attentive checkpoint 的 AP@0.7 仍低于原 PAPG 主线，因此不能简单替换全部主表；更合理的用法是 detector sensitivity、或在远程 fine-tune 失败时作为 AP@0.3/AP@0.5 strengthened candidate。
- 同 attentive checkpoint 下，Pure late controlled 只有 `0.82/0.65/0.28`，明显低于 SGCP-PAPG attentive 的 `0.87/0.81/0.36`。这说明 attentive detector 并非只同步增强 prediction-sharing reference；SGCP 的 scheduled raw point-cloud early fusion 在 AP@0.5/AP@0.7 上仍有实质贡献。
- Pure late attentive 的 prediction-box overhead：`80 B/box` broadcast mean/max `1.37/1.51 Mbps`，scheduled all-to-all mean/max `25.97/28.60 Mbps`；`128 B/box` broadcast mean/max `2.13/2.35 Mbps`，scheduled all-to-all mean/max `40.53/44.65 Mbps`。因此它仍应写作 prediction-sharing reference，而不是 raw-LiDAR baseline。
- COSDH checkpoint 不能直接迁移到 plain PointPillar early detector；真实 COSDH 模型虽然已在本仓库跑通加载与 forward path，但当前 CARLA dump 上 1 帧输出 0 个预测框。进一步诊断显示 6 个 receiver 的 `psm` sigmoid 最大值仅约 `0.0148--0.0224`；即使把 `score_threshold` 降到 `0.01/0.005/0.003`，正式 postprocess 仍返回 0 个最终框。该路线不是简单阈值校准问题，必须先解决 `proj_first`、feature-communication 输入语义、LiDAR range / postprocessor 约定与 collapsed raw point-cloud 输入不匹配的问题。

### Attentive comparison against key baselines

用户进一步关注：如果把 detector/checkpoint 统一替换为 attentive，SGCP 相对 Pure late 和 EdgeCooperHD 是否更好。已补跑 `EdgeCooperHD` attentive 41 帧，并生成 manifest：

- `docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719\attentive_comparison_manifest.csv`
- `docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719\edgecooper_hd_attentive_3m117g_41f.log`
- `docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719\edgecooper_hd_attentive_3m117g_41f_trace.csv`

| Variant | Frames | Detector | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Interpretation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Pure late attentive | 41 | attentive singleton detector + `naive_late_fusion()` | 0.82 | 0.65 | 0.28 | 0 raw LiDAR | prediction-sharing reference |
| EdgeCooperHD attentive | 41 | attentive raw-LiDAR detector | 0.85 | 0.74 | 0.35 | 65.40 | edge/global scheduler reference |
| SGCP-PAPG attentive | 41 | attentive raw-LiDAR detector | 0.87 | 0.81 | 0.36 | 62.54 | strengthened SGCP sensitivity |

结论：在同一 attentive detector 口径下，SGCP-PAPG 不再弱于 Pure late 或 EdgeCooperHD；它比 Pure late 高 `+0.05/+0.16/+0.08` AP，比 EdgeCooperHD 高 `+0.02/+0.07/+0.01` AP，同时比 EdgeCooperHD 少约 `2.87 Mbps` raw-LiDAR payload。该结果更有利于论文叙事，但由于它改变 detector 初始化，当前仍标为 checkpoint sensitivity / candidate；若要替换正式主表，必须把其他主表 baseline 也完整切到 attentive 后重跑并更新 Table 1/3/figures。

## Attentive table/figure rerun

用户要求立即重跑图表，弱化旧 `pointpillar_early_fusion` checkpoint 图表地位，避免后续论文写作被旧结果带偏。本轮已把关键 Table 1、Table 2、Table 3、Figure 1/2/3/4 全部重跑或重生成为 attentive candidate artifacts。

Artifacts：

- Table 1：`docs\doc_workspace\SGCP\artifacts\attentive_protocol_20260719\protocol_native_attentive_manifest.csv`
- Table 2：`docs\doc_workspace\SGCP\artifacts\attentive_fusion_ablation_20260719\fusion_scaffold_attentive_manifest.csv`
- Table 3：`docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719\scheduler_comparison_attentive_manifest.csv`
- Figure 1：`docs\doc_workspace\SGCP\artifacts\pareto_attentive_20260719\figure1_pareto_ap03_attentive.pdf` / `figure1_pareto_ap07_attentive.pdf`
- Figure 2/3/4：`docs\doc_workspace\SGCP\artifacts\figures_attentive_20260719\figure2_protocol_breakdown_attentive.pdf` / `figure3_fusion_contribution_attentive.pdf` / `figure4_scheduler_comparison_attentive.pdf`

### Attentive Protocol-Native Candidate

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Head-only attentive | 0.42 | 0.30 | 0.13 | 0.00 | lower reference |
| Pure late attentive | 0.82 | 0.65 | 0.28 | 1.37 box broadcast | prediction-sharing reference |
| FullPerception-PCS attentive | 0.43 | 0.29 | 0.14 | 16.38 | built-in PCS baseline remains weak |
| EdgeCooperHD attentive | 0.85 | 0.74 | 0.35 | 65.40 | edge/global scheduler reference |
| SGCP-PAPG attentive | 0.87 | 0.81 | 0.36 | 62.54 | new proposed candidate |
| Full20Early attentive | 0.88 | 0.85 | 0.45 | 118.71 | full raw-sharing upper reference |

### Attentive Scheduler Comparison

| Scheduler | AP@0.3 | AP@0.5 | AP@0.7 | Mbps |
| --- | ---: | ---: | ---: | ---: |
| RandomBudget attentive | 0.85 | 0.75 | 0.36 | 61.25 |
| DensityGreedy attentive | 0.86 | 0.78 | 0.38 | 75.94 |
| LinkAwareDensity attentive | 0.86 | 0.78 | 0.38 | 75.94 |
| PACP-LiDAR attentive | 0.88 | 0.79 | 0.37 | 86.56 |
| EdgeCooperHD attentive | 0.85 | 0.74 | 0.35 | 65.40 |
| SGCP-PAPG attentive | 0.87 | 0.81 | 0.36 | 62.54 |

### 结论

- 后续论文写作应默认使用 attentive candidate 图表；legacy early-checkpoint 表格降级为 checkpoint reference。
- SGCP-PAPG attentive 同时高于 Pure late attentive 与 EdgeCooperHD attentive，且比 EdgeCooperHD 少约 `2.87 Mbps`。
- PACP-LiDAR attentive 的 AP@0.3/AP@0.7 略高于 SGCP，但通信量高 `24.02 Mbps`；这适合作 Pareto tradeoff，而不是否定 SGCP。
- Full20Early attentive 仍是高 IoU 上界，SGCP 当前保留其 AP@0.3/AP@0.5 的 `98.9%/95.3%`，以 `52.7%` raw payload 运行。
