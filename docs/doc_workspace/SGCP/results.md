# SGCP 核心实验结果

本文件只记录经过确认、可复现或准备进入论文/rebuttal 的核心结果。探索性现象先记录在 `log.md`，稳定后再整理到这里。

更新时间：2026-07-18

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
| NC | TBD | TBD | TBD | TBD | TBD | No cooperation |
| Full 20-CAV early upper reference | 0.85 | 0.83 | 0.48 | 118.71 | TBD | Full point-cloud sharing AP upper bound; upload non-ego CAV payload 60,838,528 bytes |
| Built-in FullPerception PCS, legacy cluster-head eval | 0.44 | 0.39 | 0.17 | 24.75 | TBD | Pre-repair compatibility result; `pcs.py` was already FullPerception PCS but simplified `c(q)=1` |
| Built-in FullPerception PCS, repaired scheduled receivers | 0.33 | 0.29 | 0.14 | 15.80 | TBD | Protocol-correct PCS pass: payload-based `c(q)`, real `sc_num`, 104 scheduled requests in NS3 dry-run; still under-scheduled |
| FullPerception-RSU proxy | 0.84 | 0.80 | 0.46 | 109.71 | TBD | Virtual RSU/global candidate pool, 3 members/head, 117 grid budget; not a V2V-only fair baseline |
| EdgeCooper-style proxy | 0.75 | 0.70 | 0.32 | 109.53 | TBD | Virtual edge/global candidate pool, blind-spot complementarity proxy; preliminary, not strict paper reproduction |
| EdgeCooper-global network-aware proxy | 0.81 | 0.77 | 0.42 | 74.58 | TBD | Virtual edge/global assignment proxy with sender-load balancing and 35 m V2V feasibility; 11-frame NS3 73/110 complete |
| FullPerception-Decentralized proxy | 0.80 | 0.76 | 0.41 | 75.94 | TBD | CAV-side V2V only, cluster-local candidates, 3 members/head, 117 grid budget; NS3 110/110 complete |
| Full-cluster reference | 0.82 | 0.79 | 0.42 | 87.51 | TBD | Full intra-cluster upload reference |
| Selective V2V forced random | 0.77 | 0.73 | 0.38 | 61.68 | TBD | Same coalition path, 3 members/head, 117 grid budget |
| Selective V2V communication-aware | 0.78 | 0.75 | 0.40 | 58.97 | TBD | 2 members/head, 87 grid budget |
| Selective V2V density high-budget | 0.80 | 0.76 | 0.40 | 73.58 | TBD | 3 members/head, 117 grid budget |
| SGCP PAPG, 10ch, `rho_th=3`, `B_h=2` | 0.81 | 0.78 | 0.39 | 62.54 | TBD | Current main method; 110/110 PAPG NS3 replay complete |
| SGCP coverage-aware, 10ch, `rho_th=3` | 0.79 | 0.76 | 0.38 | 57.38 | TBD | PAPG predecessor/ablation |
| SGCP coverage-aware, 20ch | 0.80 | 0.76 | 0.41 | 73.98 | TBD | Resource-sensitivity row |

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

## Baseline 公平性说明

详细口径见 `baseline_fairness.md`。当前结果应按以下层级解释：

| Layer | Method | Current Result | Main Fair Baseline? | Notes |
| --- | --- | --- | --- | --- |
| Upper reference | Full 20-CAV early / virtual FullPerception centralized | 0.85 / 0.83 / 0.48 | No | 全点云共享，无 SGCP 通信约束；non-ego upload payload 60,838,528 bytes |
| Upper reference | Full 20-CAV late checkpoint | 0.91 / 0.85 / 0.51 | No | 使用独立 late checkpoint，不能直接作为同 checkpoint 消融 |
| Built-in FullPerception | PCS (`pcs.py`) legacy eval | 0.44 / 0.39 / 0.17 | No | 仓库内置 PCS 对应 FullPerception 论文调度算法；旧评估 payload 为 12,684,880 bytes / 24.75 Mbps |
| Built-in FullPerception | PCS (`pcs.py`) repaired scheduled-receiver eval | 0.33 / 0.29 / 0.14 | No | `c(q)`、`sc_num` 和 scheduled links 已修复并进入 NS3 dry-run；payload 8,100,112 bytes / 15.80 Mbps，说明 PCS 当前仍 under-schedule |
| RSU/edge-assisted | FullPerception-RSU proxy | 0.84 / 0.80 / 0.46 | No | 虚拟 RSU/global candidate pool；当前 dump 无真实 RSU sensor，不作为 V2V-only 公平主对比 |
| RSU/edge-assisted | EdgeCooper-style proxy | 0.75 / 0.70 / 0.32 | No | 虚拟 edge/global candidate pool；当前是 blind-spot complementarity proxy，不是严格原论文 MCF/coloring 复现 |
| RSU/edge-assisted | EdgeCooper-global network-aware proxy | 0.81 / 0.77 / 0.42 | No | 虚拟 edge/global assignment proxy；74.58 Mbps，11 帧 NS3 73/110 complete，说明离线高 AP 仍需 deadline-aware 调度补强 |
| V2V-only fair baseline | FullPerception-Decentralized proxy | 0.80 / 0.76 / 0.41 | Yes | cluster-local candidate pool，3 members/head，117 grid budget；强 decentralized baseline；11 帧 NS3 replay 110/110 application/RLC complete |
| SGCP main | SGCP PAPG 10ch | 0.81 / 0.78 / 0.39 | Yes | 当前主方法，62.54 Mbps，PAPG NS3 110/110 complete |
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

论文写作建议：FullPerception-RSU 和 full 20-CAV early/late fusion 只能作为 upper/reference，不应放入“同通信预算公平主对比”结论。公平主对比应使用同数据、同 backbone、同 AP 口径，并尽量匹配通信预算或显式报告 payload。旧 `RandomRA/MWS` 的 payload 只有约 9.7/9.9 MB，未充分利用 10 子信道资源，不宜作为“SGCP 降低通信量”的主证据；它们更适合作 w/o PPS 消融。主公平 baseline 使用 forced-budget random、density/communication-aware selective sharing；SGCP 主方法使用 PAPG。

### Explicit FullPerception Baselines

代码状态：仓库没有以 `FullPerception` 命名的入口，但 `opencda/core/clustering/algorithms/resource_allocation/pcs.py` 是 FullPerception 论文 PCS 调度算法的内置实现；`mws.py` 和 `random_ra.py` 是同一 PCS 问题上的 greedy/random baseline。当前新增 `--resource-allocation fullperception_pcs|fullperception_mws|fullperception_random` alias，并保留 `--selective-sharing-baseline fullperception_rsu|fullperception_decentralized` 作为后补 proxy。proxy 分支使用 41 帧同一 dump、同一 OpenCOOD early checkpoint、同一 inter-cluster late-fusion evaluation path，并限制为 3 members/head 与 117 selected grids。

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline fullperception_rsu --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\fullperception_baselines_20260717\fullperception_rsu_trace.csv
```

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\fullperception_baselines_20260717\
```

| Baseline | Candidate Scope | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. Source CAVs | Avg. Selected Grids | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full 20-CAV early upper reference | all CAVs, full upload | 0.85 | 0.83 | 0.48 | 60,838,528 | 118.71 | 20.00 | N/A | AP upper bound, not budgeted scheduling |
| `fullperception_pcs` built-in, legacy cluster-head eval | PCS blind-spot scheduling | 0.44 | 0.39 | 0.17 | 12,684,880 | 24.75 | 1.66 | 630.66 | Pre-repair compatibility row; simplified `c(q)=1` |
| `fullperception_pcs` built-in, repaired scheduled receivers | PCS blind-spot scheduling | 0.33 | 0.29 | 0.14 | 8,100,112 | 15.80 | 2.00 | 57.71 | Payload-based `c(q)` and real `sc_num`; protocol-correct but weaker, under-scheduled |
| `fullperception_rsu` proxy | global / virtual RSU | 0.84 | 0.80 | 0.46 | 56,224,736 | 109.71 | 4.00 | 117.00 | Infrastructure-assisted global scheduler proxy |
| `fullperception_decentralized` proxy | cluster-local V2V | 0.80 | 0.76 | 0.41 | 38,920,592 | 75.94 | 3.33 | 103.20 | Strong V2V-only decentralized FullPerception proxy; NS3 110/110 complete |
| `fullperception_rsu`, ego receiver probe | ego virtual receiver | 0.71 | 0.70 | 0.49 | 26,350,784 | 51.42 | 9.54 | 332.93 | Diagnostic only; several frames fell back to ego-only |

`fullperception_decentralized` 已完成 11-frame true NS3 replay：110/110 scheduled requests 完成 application callback，110/110 RLC complete，RLC TX/RX events 2970/2970，PHY decode failures 0，avg/p95 callback delay 23.91/24.00 ms。`fullperception_rsu` 3-frame dry-run 每帧 10 scheduled / 8 skipped。修复后的 built-in `fullperception_pcs` 41 帧 dry-run 生成 104 条 scheduled request，`sc_num` 会按 required subchannels 写入 upload plan。下一步应优先校准 `fullperception_pcs` 的调度强度和 RSU/global receiver 口径，再进入主表。

### Same-Budget CAV-Only Selective Sharing

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。复用 SGCP coalition formation 和 inter-cluster late fusion 评价口径，但不使用 PPS；每个 cluster head 最多选择 2 个非 head 成员，总 grid budget 为 87，接近 SGCP 默认 `avg_selected_grids=87.32`。

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
