# SGCP Baseline Reproduction Plan

更新时间：2026-07-18

本文档记录审稿意见要求的额外 baseline 复现计划。目标是把 baseline 分成清晰层级：AP upper bound、RSU/edge-assisted、V2V-only decentralized、same-pipeline ablation，避免再把 full 20-CAV 上界、FullPerception-RSU 和公平 V2V baseline 混写。

## Reviewer Motivation

审稿意见集中指出两类问题：

- FullPerception baseline 不清楚：当前场景没有真实 RSU，若使用虚拟 RSU 或全局 oracle，需要明确其信息优势和通信边界。
- baseline 不够新也不够公平：Random/MWS 不能代表最新 decentralized collaborative perception；需要补充更接近 SOTA 的方法，并统一 backbone、数据帧、通信预算和评估口径。

## Current Code Audit

仓库中没有以 `FullPerception` 命名的算法分支，但存在对应 FullPerception 论文的 PCS/MWS/RS resource-allocation 实现：`opencda/core/clustering/algorithms/resource_allocation/pcs.py`、`mws.py` 和 `random_ra.py`。PCS 实现包含 blind-spot grouping、potential link generation、link utility、conflict graph、weight splitting、recursive PCS scheduling 和 subchannel assignment，对应论文 `FullPerception: Network-level Collaborative Perception for Eliminating Vehicular Blind Spots` 中的 Proactive Conflict-free Scheduling (PCS)。因此，后续文档应把 `pcs.py` 视为仓库内置 FullPerception-PCS baseline，而不是继续说“没有实现”。

已新增显式 baseline 名称：

| Name | Information Scope | Scheduling Scope | Current Status |
| --- | --- | --- | --- |
| `fullperception_pcs` / `fullperception` | base-station / RSU-side PCS scheduling | PCS blind-spot link scheduling from `pcs.py` | Alias added; first protocol repair complete; 41-frame legacy and repaired results available |
| `fullperception_mws` | base-station / RSU-side greedy baseline | MWS from `mws.py`, inherited from PCS | Tuned 11f sanity complete; weak diagnostic result |
| `fullperception_random` | base-station / RSU-side random schedule | RS from `random_ra.py`, inherited from PCS | Tuned 11f sanity complete; weak/low-payload diagnostic result |
| `global_selective_proxy` | virtual RSU / global CAV candidate pool | global density with mild distance cost, then grid-budgeted upload | 41-frame proxy result available; proxy/diagnostic, not FullPerception PCS |
| `cluster_local_selective_proxy` | CAV-side V2V only | cluster-local density with distance/link-quality cost | 41-frame result and 11-frame true NS3 replay available; V2V-only proxy |
| `edgecooper` | edge / virtual RSU | complementarity minus redundancy proxy | First proxy implemented |
| `edgecooper_global` | edge / virtual RSU with network-aware global assignment proxy | blind-spot complementarity + global sender-load balancing + 35 m V2V feasibility gate | Implemented; 41-frame offline result and 11-frame true NS3 replay available, but NS3 delivery incomplete |
| `edgecooper_global_hd` | edge / virtual RSU with network-aware half-duplex global assignment proxy | `edgecooper_global` plus sender/receiver half-duplex exclusion within each slot | Implemented; 41-frame offline result and 11-frame true NS3 replay available with full delivery |
| `pacp_lidar` | V2V-only priority-aware proxy | LiDAR BEV occupancy match + blind-grid complementarity + link/distance cost, then raw point-grid upload | Implemented; 41-frame offline result and 11-frame NS3 dry-run available |

PCS/MWS/RS 通过 `--resource-allocation fullperception_pcs|fullperception_mws|fullperception_random` 进入资源分配路径；`global_selective_proxy/cluster_local_selective_proxy/edgecooper/edgecooper_global` 是后补的 selective-sharing proxy，通过 `--selective-sharing-baseline <name>` 进入同一 OpenCOOD checkpoint、同一 41-frame dump 和同一 inter-cluster late-fusion evaluation path。

## FullPerception Baselines

实验配置：

- 数据：`D:\Data\Carla\2026_07_15_01_26_56`
- 帧数：41
- CAV 数：20
- 感知：OpenCOOD early-fusion checkpoint + SGCP inter-cluster late fusion
- 通信窗口：20 MHz / 10 subchannels
- budget：3 uploaded members per receiver，117 selected grids
- artifact：`docs\doc_workspace\SGCP\artifacts\fullperception_baselines_20260717\`

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full 20-CAV early upper reference | 0.85 | 0.83 | 0.48 | 60,838,528 | 118.71 | 20.00 | N/A | Full-sharing AP upper reference, not a fair baseline |
| `fullperception_pcs` / built-in PCS, legacy eval | 0.44 | 0.39 | 0.17 | 12,684,880 | 24.75 | 1.66 | 630.66 | Pre-repair compatibility result; simplified `c(q)=1` |
| `fullperception_pcs` / built-in PCS, tuned scheduled receivers | 0.59 | 0.53 | 0.22 | 12,959,840 | 25.29 | 2.00 | 27.00 | Current FullPerception PCS baseline; blind-spot units split with `division=12,min_overlap=0` |
| `global_selective_proxy` | 0.84 | 0.80 | 0.46 | 56,224,736 | 109.71 | 4.00 | 117.00 | Virtual RSU/global scheduler proxy, strong but infrastructure-assisted |
| `cluster_local_selective_proxy` | 0.80 | 0.76 | 0.41 | 38,920,592 | 75.94 | 3.33 | 103.20 | V2V-only selective proxy; NS3 110/110 complete |
| `global_selective_proxy`, ego receiver probe | 0.71 | 0.70 | 0.49 | 26,350,784 | 51.42 | 9.54 | 332.93 | Diagnostic only; candidate fallback is unreliable |

结论：`pcs.py` 是仓库内置 FullPerception PCS，本轮修复和调参后可作为正式但不强势的 FullPerception baseline：它比第一轮协议修复显著改善，但仍低于 SGCP/EdgeCooper 等主算法。两个 selective proxy 已改名为 `global_selective_proxy` / `cluster_local_selective_proxy`，只能作为 proxy/diagnostic，不能替代 FullPerception 论文 PCS。

## EdgeCooper Plan

本地论文：`C:\Users\sakakibara\OneDrive\Papers\Cooperative Perception\EdgeCooper_Network-Aware_Cooperative_LiDAR_Perception_for_Enhanced_Vehicular_Awareness.pdf`

论文机制摘要：

- edge server 聚合多车 LiDAR 视角；
- 调度目标强调 complementarity-enhanced 和 redundancy-minimized raw sensor sharing；
- multi-hop data sharing 被建模为带冲突约束的 minimum-cost flow；
- 使用二维图着色处理冲突。

当前 proxy：

- 假设存在虚拟 RSU/edge server；
- 对每个 receiver 从全局 CAV 候选池中迭代选择能补齐 receiver blind grids 的 sender；
- 候选 grid 限定为 sender 可观测且 receiver/head 低密度的 blind-spot grids，不再 fallback 到 sender 全视野；
- member score 使用 blind-spot complementarity、selected-sender redundancy 和 distance/network cost；
- 使用相同 grid budget 和 inter-cluster late-fusion 口径；
- naive 3-frame smoke test 为 `0.54/0.46/0.15`，说明原始 complementarity proxy 过度偏向少数高密度车辆，不能作为最终 EdgeCooper baseline。
- blind-spot-aware 3-frame smoke test 恢复到 `0.76/0.72/0.33`；41-frame full run 为 `0.75/0.70/0.32`，payload 56,134,048 bytes / 109.53 Mbps。
- `edgecooper_global` 在上述 proxy 基础上加入全局 sender-load balancing 和 35 m V2V feasibility gate，近似 edge-side global assignment，同时避免逐 receiver 贪心重复选择同一 sender。41-frame full run 为 `0.81/0.77/0.42`，payload 38,223,408 bytes / 74.58 Mbps，avg source CAVs 3.26，avg selected grids 98.75。
- `edgecooper_global` 11-frame true NS3 replay：73/110 application callback complete，73/110 RLC complete，RLC TX/RX events 2970/1971，37 个 request 为 `rlc_tx_no_rx`，PHY decode failures 0。该结果说明链路无 PHY 解码错误，但 deadline/调度层仍不能保证所有 request 及时完成。
- failure diagnosis 显示旧 `edgecooper_global` 的 NS3 failure 高度集中在 target 1 / target 4，同一 slot 中这些 receiver 又被调度为 sender，例如 `4->1` 与 `1->4` 同时存在；这些链路距离多在 7.8-32.6 m 内，因此不是距离超限，而是半双工 sender/receiver role conflict。
- `edgecooper_global_hd` 禁止本轮所有 cluster-head receivers 同时作为 sender。41-frame full run 为 `0.81/0.78/0.42`，payload 33,519,040 bytes / 65.40 Mbps，avg source CAVs 3.00，avg selected grids 89.02。11-frame true NS3 replay 为 110/110 application callback complete，110/110 RLC complete，RLC TX/RX events 2970/2970，PHY decode failures 0。

当前结论：EdgeCooper proxy 已经可复现并可进入 artifact 记录。`edgecooper_global_hd` 是当前最强 edge-assisted baseline proxy：AP@0.7 高于 PAPG，Mbps 接近 PAPG，且 NS3 delivery 完整。但它属于 virtual edge/RSU-assisted 口径，使用全局 receiver set 和全局 sender-load state，不是 V2V-only decentralized baseline。因此论文中应将其标注为 `EdgeCooper-global-HD proxy` 或 RSU/edge-assisted baseline，不宜把它和 PAPG 混写成同一信息条件下的公平 V2V 对比。

下一步实现方向：

- 若需要更贴近原论文，将当前 global sender-load/capacity + half-duplex proxy 进一步升级为 minimum-cost-flow-style global assignment；
- 若主表必须混放 RSU/edge-assisted 与 V2V-only，继续提升 PAPG AP@0.7 或显式说明信息条件差异；
- 若继续保持虚拟 RSU，表格中必须将其归入 RSU/edge-assisted baseline，而不是 V2V-only decentralized baseline。

## Additional Candidate Baselines

| Candidate | Type | Fit to Current Dump | Implementation Plan |
| --- | --- | --- | --- |
| Where2comm | V2V / learned communication | Medium | OpenCOOD 生态相关；优先评估是否可直接复用 pretrained/code，否则实现 confidence-map top-k proxy |
| PACP | V2V / priority-aware CP | Medium | 原论文是 RGB/BEV + CoBEVT/SinBEVT + adaptive autoencoder，不是点云原生；已实现 `pacp_lidar`，把 BEV-match priority 迁移为 LiDAR BEV occupancy/grid complementarity proxy |
| What2comm | V2V / what-to-communicate | Medium | 若公开实现可接入则复现；否则用 objectness/uncertainty grid selection proxy |
| CoBEVT | cooperative BEV transformer | Low-Medium | 更偏模型架构，需要 checkpoint/训练成本；适合作 related work，不一定适合短期主表 |
| V2VNet | V2V feature-message passing | Low-Medium | 与当前 early point-cloud checkpoint 不同；可作为 related work 或另起模型复现任务 |
| RACooper | RSU-assisted resource allocation | Low for main SGCP | 最新但 RSU-assisted，适合附加 edge/RSU baseline，不适合作 V2V-only 公平 baseline |

## Immediate Tasks

- `fullperception_pcs` 已完成 tuned baseline；`fullperception_mws/fullperception_random` 已完成 11-frame tuned sanity，结论为 heuristic diagnostics，不进入主公平表。
- `cluster_local_selective_proxy` 的 11-frame true NS3 replay 已完成：110/110 application callback complete、110/110 RLC complete、0 PHY failures。后续只需在表格中维护 artifact 路径和口径说明。
- 重构 EdgeCooper proxy：从当前 blind-spot-aware per-receiver greedy 改为 minimum-cost-flow/global assignment 风格。
- 选择一个 V2V-only SOTA proxy 优先实现，建议从 Where2comm-style confidence communication 或 PACP-style priority-aware sharing 开始。

## PACP LiDAR Adaptation

原 PACP 论文 `PACP: Priority-Aware Collaborative Perception for Connected and Autonomous Vehicles` 不是 raw LiDAR 点云方法。原文的 priority weight 基于 RGB 相机生成的 BEV feature/box overlap，backbone 使用 SinBEVT/CoBEVT，并用 adaptive autoencoder 压缩/重建 raw camera data。因此本文档和主表中不能把当前实现写成 PACP 原方法的严格复现，只能写为 `PACP-style LiDAR priority proxy`。

当前 `pacp_lidar` 迁移原则：

- 成员优先级：用 head/sender 的 LiDAR BEV grid 占据一致性近似 PACP 的 BEV-match；用 sender 对 head weak/blind grids 的互补密度近似 perception priority；再乘以距离或 NS3 link-quality cost。
- Grid 选择：在选中 sender 后，按 overlap-match、blind-grid complementarity、novelty 和 density 选择 raw point-cloud grids；实际上传仍是点云块，进入同一 OpenCOOD early-fusion + inter-cluster late-fusion pipeline。
- 公平边界：与 PACP 原论文共享 priority-aware / BEV-match resource scheduling idea，但没有复现 RGB encoder/decoder、CoBEVT feature fusion 或 adaptive image compression。

41-frame first results:

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | NS3 Plan |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pacp_lidar`, 3 members/head, 117 grids/head | 0.81 | 0.79 | 0.42 | 44,361,424 | 86.56 | 11f dry-run: 110 scheduled, 44 skipped unscheduled |
| `pacp_lidar`, 2 members/head, 87 grids/head | 0.76 | 0.73 | 0.37 | 34,498,160 | 67.31 | 11f dry-run: 110 scheduled, 9 skipped unscheduled |

结论：PACP 思路可迁移到点云通信场景，并能在高预算下达到较高 AP@0.7；但 raw LiDAR payload 偏高，低预算下 AP 低于 PAPG。因此它适合作为近年 V2V priority-aware proxy baseline 或附表 baseline，不宜声称为严格 PACP 复现，也不宜直接替代 SGCP 主线。
