# SGCP 核心实验结果

本文件只记录经过确认、可复现或准备进入论文/rebuttal 的核心结果。探索性现象先记录在 `log.md`，稳定后再整理到这里。

更新时间：2026-07-15

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

当前尚未在本仓库中复现实验。下表预留用于记录论文主表或复现实验结果。

| Method | mAP@0.3 | mAP@0.5 | mAP@0.7 | Comm. Overhead (Mbps) | Runtime / Cycle (ms) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NC | TBD | TBD | TBD | TBD | TBD | No cooperation |
| RS | TBD | TBD | TBD | TBD | TBD | Random schedule |
| MUG | TBD | TBD | TBD | TBD | TBD | Maximum utility greedy |
| FullPerception-RSU | TBD | TBD | TBD | TBD | TBD | RSU-assisted reference |
| FullPerception-Decentralized | TBD | TBD | TBD | TBD | TBD | Same V2V information budget as SGCP |
| SGCP | TBD | TBD | TBD | TBD | TBD | Full method |

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
| Upper reference | Full 20-CAV early fusion | 0.85 / 0.83 / 0.48 | No | 全点云共享，无 SGCP 通信约束 |
| Upper reference | Full 20-CAV late checkpoint | 0.91 / 0.85 / 0.51 | No | 使用独立 late checkpoint，不能直接作为同 checkpoint 消融 |
| Upper reference | FullPerception-RSU | TBD | No | 集中式/RSU-assisted reference；`v2xp_cluster_carla` 当前未启用 RSU |
| SGCP main | SGCP constrained + inter-cluster late fusion | 0.77 / 0.73 / 0.35 | Yes | 当前完整 SGCP 离线主口径 |
| Same pipeline ablation | Random scheduler | 0.44 / 0.39 / 0.17 | Yes | 同 SGCP clustering + late-fusion path，替换 PPS |
| Same pipeline ablation | MWS scheduler | 0.31 / 0.26 / 0.11 | Pending | 需要复核 MWS baseline 定义 |
| Same-budget selective baseline | Nearest top-k grid sharing | 0.76 / 0.73 / 0.37 | Yes | CAV-only, same clustering + late-fusion path, grid budget 87 |
| Same-budget selective baseline | Density top-k grid sharing | 0.77 / 0.74 / 0.39 | Yes | Strong baseline; slightly higher AP@0.7 with higher payload |
| Same-budget selective baseline | Communication-aware density sharing | 0.78 / 0.75 / 0.40 | Yes | Strongest current baseline; density divided by distance cost |
| Reference only | Singleton full late-fusion reference | 0.82 / 0.76 / 0.37 | No | late-fuse 全部 20 CAV，当前未计 detection-box exchange overhead |

论文写作建议：FullPerception-RSU 和 full 20-CAV early/late fusion 只能作为 upper/reference，不应放入“同通信预算公平主对比”结论。公平主对比应使用同数据、同 backbone、同 AP 口径，并尽量匹配通信预算或显式报告 payload。

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

观察：communication-aware selective-sharing 是当前最强公平 baseline，AP@0.5/AP@0.7 均高于 SGCP，但 payload 也高于 SGCP。论文中不能声称 SGCP 在该 dump 上全面优于所有 selective-sharing baseline；更稳妥的结论是 SGCP 提供 coalition stability、PPS channel feasibility、动态/网络约束建模和可解释资源分配，而单帧 density/communication-aware top-k 在当前短序列上是强竞争 baseline。

### NS3 Link-Quality-Aware Selective Sharing

实验口径：`D:\Data\Carla\2026_07_15_01_26_56` 前 11 帧，20 CAV，same-budget selective sharing，`communication_aware`，member budget 2，grid budget 87，inter-cluster late fusion。NS3 link-quality 使用 `docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\eval\rlc_by_request.csv` 中的 `rlc_complete`，即 `targetSubchannels=5` 受限暴露子信道回归。

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Distance proxy | 11 | 0.71 | 0.67 | 0.31 | 120873.94 | 7977680 | 2.80 | 80.85 | Old `density / distance` score |
| NS3 RLC-complete aware | 11 | 0.68 | 0.63 | 0.27 | 118129.70 | 7796560 | 2.80 | 80.85 | Uses `density * rlc_complete_ratio / distance` |

观察：NS3-aware cost 避开了受限 5 子信道下不可完整交付的链路，通信量略降，但 11 帧 AP 也下降。该结果不应被解释为 NS3-aware baseline 更强，而是说明真实链路可行性会改变 selective-sharing 的成员选择；后续主实验应在完整 41 帧或重新导出的网络受限场景上报告。

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

### Density Threshold

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，`potential_game`，SGCP inter-cluster late fusion。`rho_th` 覆盖 lidar `density_threshold`，影响 high-density grid 判定、`Vehicle_Grid.rho_th`、cluster grid bits 和 PPS grid selection。

| `rho_th` | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Cluster-Head Sources | Avg. Clusters | Reconfig. Count | Vehicle-Head Changes | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.74 | 0.69 | 0.34 | 86658.74 | 21751344 | 251 | 6.12 | 10 | 60 | Lowest payload, lower AP |
| 1.0 | 0.75 | 0.71 | 0.33 | 96968.13 | 23854160 | 246 | 6.00 | 9 | 64 | Lower payload than default |
| 2.0 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 246 | 6.00 | 11 | 76 | Current implementation default / paper candidate |
| 3.0 | 0.77 | 0.73 | 0.37 | 113689.69 | 27967664 | 246 | 6.00 | 11 | 76 | Higher AP@0.7, higher payload |
| 4.0 | 0.77 | 0.74 | 0.37 | 115754.73 | 28475664 | 246 | 6.00 | 11 | 76 | Best AP@0.5/AP@0.7 in this dump, highest payload |

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

当前已完成 `rho_th` 阈值敏感性实验，但尚未复现完整 `f(rho)` 标定曲线。后续需要补充：

- 点云密度采样方式。
- mAP 与 `rho` 的拟合曲线。
- `rho_th = 2.0 points/m^2` 的选择依据。
- 不同 detector 或不同场景下的泛化结果。

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

当前尚未复现。后续需要补充：

| Stage | Mean (ms) | P95 (ms) | Max (ms) | Notes |
| --- | ---: | ---: | ---: | --- |
| Topology/change detection | TBD | TBD | TBD |  |
| Coalition formation | TBD | TBD | TBD |  |
| PPS scheduling | TBD | TBD | TBD |  |
| Point cloud transmission | TBD | TBD | TBD |  |
| Intra-cluster fusion | TBD | TBD | TBD |  |
| Detector inference | TBD | TBD | TBD |  |
| Inter-cluster late fusion | TBD | TBD | TBD |  |
| End-to-end cycle | TBD | TBD | TBD | Target <= 100 ms |

## 数据集导出验证

| Dataset Path | CAVs | Frames / CAV | PCD Files | YAML Files | Offline Inference |
| --- | ---: | ---: | ---: | ---: | --- |
| `D:\Data\Carla\2026_07_15_01_26_56` | 20 | 41 | 820 | 821 | `000060`: 62 pred boxes, 71 GT boxes |

## 离线无 NS3 测试

| Dataset Path | Fusion | Frames | Ego CAV | AP@0.3 | AP@0.5 | AP@0.7 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `D:\Data\Carla\2026_07_15_01_26_56` | Early | 41 | 1 | 0.85 | 0.83 | 0.48 | No NS3, no online CARLA sensor stream |
| `D:\Data\Carla\2026_07_15_01_26_56` | Late | 41 | 1 | 0.91 | 0.85 | 0.51 | Full 20-CAV OpenCOOD late model; reference only |

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
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `T_min_stab=0` | 41 | 6 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `singleton clustering` | 41 | 20 | 0.82 | 0.76 | 0.37 | 0.00 | 0 | 1.00 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `random` | 41 | 6 | 0.44 | 0.39 | 0.17 | 39534.05 | 9725376 | 1.51 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `mws` | 41 | 6 | 0.31 | 0.26 | 0.11 | 40284.68 | 9910032 | 1.50 |

说明：`random` 与 `mws` 当前作为 “w/o PPS / baseline scheduler” 第一版结果。两者通信开销显著低于 `potential_game`，但 mAP 也明显下降；当前 `mws` 结果低于 `random`，后续进入论文前需要复核 MWS 效用定义与论文 baseline 是否一致。

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
