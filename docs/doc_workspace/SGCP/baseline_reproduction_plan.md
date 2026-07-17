# SGCP Baseline Reproduction Plan

更新时间：2026-07-17

本文档记录审稿意见要求的额外 baseline 复现计划。目标是把 baseline 分成清晰层级：AP upper bound、RSU/edge-assisted、V2V-only decentralized、same-pipeline ablation，避免再把 full 20-CAV 上界、FullPerception-RSU 和公平 V2V baseline 混写。

## Reviewer Motivation

审稿意见集中指出两类问题：

- FullPerception baseline 不清楚：当前场景没有真实 RSU，若使用虚拟 RSU 或全局 oracle，需要明确其信息优势和通信边界。
- baseline 不够新也不够公平：Random/MWS 不能代表最新 decentralized collaborative perception；需要补充更接近 SOTA 的方法，并统一 backbone、数据帧、通信预算和评估口径。

## Current Code Audit

仓库中此前没有显式命名的 `FullPerception` 算法分支。历史文档和表格中的 FullPerception 多数对应 full 20-CAV early fusion 或 full-sharing upper reference，而不是一个可切换的 baseline scheduler。

已新增显式 baseline 名称：

| Name | Information Scope | Scheduling Scope | Current Status |
| --- | --- | --- | --- |
| `fullperception_rsu` | virtual RSU / global CAV candidate pool | global density with mild distance cost, then grid-budgeted upload | Implemented; 41-frame proxy result available |
| `fullperception_decentralized` | CAV-side V2V only | cluster-local density with distance/link-quality cost | Implemented; 41-frame result available |
| `edgecooper` | edge / virtual RSU | complementarity minus redundancy proxy | First proxy implemented; currently needs algorithm refinement |

这些分支使用 `opencda.tools.offline_inference --selective-sharing-baseline <name>` 进入同一 OpenCOOD checkpoint、同一 41-frame dump、同一 inter-cluster late-fusion evaluation path，并可通过 `opencda.tools.offline_ns3_replay --selective-sharing-baseline <name>` 生成 scheduled-only request plan。

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
| `fullperception_rsu` proxy | 0.84 | 0.80 | 0.46 | 56,224,736 | 109.71 | 4.00 | 117.00 | Virtual RSU/global scheduler, strong but infrastructure-assisted |
| `fullperception_decentralized` proxy | 0.80 | 0.76 | 0.41 | 38,920,592 | 75.94 | 3.33 | 103.20 | V2V-only decentralized FullPerception proxy |
| `fullperception_rsu`, ego receiver probe | 0.71 | 0.70 | 0.49 | 26,350,784 | 51.42 | 9.54 | 332.93 | Diagnostic only; candidate fallback is unreliable |

结论：`fullperception_rsu` 已不再等同 full 20-CAV early upper reference。它是一个 global/virtual-RSU scheduler proxy，通信量低于全量上传但仍拥有全局候选集合，因此只能放在 RSU/edge-assisted 层级。`fullperception_decentralized` 是更公平的 V2V-only FullPerception proxy，可作为强 decentralized baseline。

## EdgeCooper Plan

本地论文：`C:\Users\sakakibara\OneDrive\Papers\Cooperative Perception\EdgeCooper_Network-Aware_Cooperative_LiDAR_Perception_for_Enhanced_Vehicular_Awareness.pdf`

论文机制摘要：

- edge server 聚合多车 LiDAR 视角；
- 调度目标强调 complementarity-enhanced 和 redundancy-minimized raw sensor sharing；
- multi-hop data sharing 被建模为带冲突约束的 minimum-cost flow；
- 使用二维图着色处理冲突。

当前 proxy：

- 假设存在虚拟 RSU/edge server；
- 对每个 receiver 从全局 CAV 候选池中迭代选择 complementarity 高、与已覆盖 grid redundancy 低的 sender；
- 使用相同 grid budget 和 inter-cluster late-fusion 口径；
- 3-frame smoke test 为 `0.54/0.46/0.15`，说明 naive complementarity proxy 过度偏向少数高密度车辆，不能作为最终 EdgeCooper baseline。

下一步实现方向：

- 引入 per-receiver blind-spot target grids，而不是只按 sender 自身新增 grid 计 complementarity；
- 加入 global conflict/capacity term，避免所有 receiver 竞争同一高密度 sender；
- 若继续保持虚拟 RSU，表格中必须将其归入 RSU/edge-assisted baseline，而不是 V2V-only decentralized baseline。

## Additional Candidate Baselines

| Candidate | Type | Fit to Current Dump | Implementation Plan |
| --- | --- | --- | --- |
| Where2comm | V2V / learned communication | Medium | OpenCOOD 生态相关；优先评估是否可直接复用 pretrained/code，否则实现 confidence-map top-k proxy |
| PACP | V2V / priority-aware CP | Medium | 可 proxy 成 object/priority-aware selective sharing，与 PAPG 区分为 baseline priority scheduler |
| What2comm | V2V / what-to-communicate | Medium | 若公开实现可接入则复现；否则用 objectness/uncertainty grid selection proxy |
| CoBEVT | cooperative BEV transformer | Low-Medium | 更偏模型架构，需要 checkpoint/训练成本；适合作 related work，不一定适合短期主表 |
| V2VNet | V2V feature-message passing | Low-Medium | 与当前 early point-cloud checkpoint 不同；可作为 related work 或另起模型复现任务 |
| RACooper | RSU-assisted resource allocation | Low for main SGCP | 最新但 RSU-assisted，适合附加 edge/RSU baseline，不适合作 V2V-only 公平 baseline |

## Immediate Tasks

- 先把 FullPerception-RSU / FullPerception-Decentralized 两个显式 baseline 纳入 `results.md` 和 `main_table_candidate.md`。
- 将 `fullperception_decentralized` 做 11-frame true NS3 replay，确认 10ch scheduled requests application/RLC complete。
- 重构 EdgeCooper proxy：从 naive complementarity 改为 blind-spot-aware edge scheduling。
- 选择一个 V2V-only SOTA proxy 优先实现，建议从 Where2comm-style confidence communication 或 PACP-style priority-aware sharing 开始。
