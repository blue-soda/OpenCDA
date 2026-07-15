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
| w/o coalition formation | TBD | TBD | TBD | TBD | TBD | Distance/random cluster |
| w/o PPS - random | 0.44 | 0.39 | 0.17 | TBD | 11 | Random scheduling, same SGCP late-fusion evaluation path |
| w/o PPS - MWS | 0.31 | 0.26 | 0.11 | TBD | 11 | Greedy scheduling, needs baseline-definition review |
| early fusion only | 0.85 | 0.83 | 0.48 | TBD | N/A | Full 20-CAV early fusion, no SGCP communication constraint |
| constrained early only | 0.36 | 0.34 | 0.17 | TBD | 11 | All cluster heads, no inter-cluster late fusion |
| late fusion only | 0.91 | 0.85 | 0.51 | TBD | N/A | OpenCOOD full 20-CAV late checkpoint; reference only, not a strict same-checkpoint SGCP ablation |

## 参数敏感性

### Stability Window

| `T_min^stab` (ms) | mAP@0.5 | Comm. Overhead (Mbps) | Reconfig. Count | Avg. Cluster Lifetime (s) | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 100 | TBD | TBD | TBD | TBD |  |
| 300 | TBD | TBD | TBD | TBD |  |
| 500 | TBD | TBD | TBD | TBD | Paper default candidate |
| 700 | TBD | TBD | TBD | TBD |  |
| 1000 | TBD | TBD | TBD | TBD |  |

### Max Cluster Size

| `N_max` | mAP@0.5 | Comm. Overhead (Mbps) | Avg. Cluster Size | Fragmentation Rate | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | TBD | TBD | TBD | TBD |  |
| 3 | TBD | TBD | TBD | TBD |  |
| 4 | TBD | TBD | TBD | TBD | Paper default candidate |
| 5 | TBD | TBD | TBD | TBD |  |
| 6 | TBD | TBD | TBD | TBD |  |

## `f(rho)` 标定结果

当前尚未复现。后续需要补充：

- 点云密度采样方式。
- mAP 与 `rho` 的拟合曲线。
- `rho_th = 2.0 points/m^2` 的选择依据。
- 不同 detector 或不同场景下的泛化结果。

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

说明：该口径更接近论文 SGCP。每帧先对所有 cluster head 执行 intra-cluster grid-constrained early fusion，并统一投影到 `ego_cav_id=1` 的 lidar pose，再对所有簇头预测框执行 simple late fusion/NMS，最终每帧提交一次 AP 统计。

| Dataset Path | RA Algorithm | Frames | Cluster-Head Sources / Frame | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs / Cluster Head |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game` | 41 | 6 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| `D:\Data\Carla\2026_07_15_01_26_56` | `potential_game`, `T_min_stab=0` | 41 | 6 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
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
| `D:\Data\Carla\2026_07_15_01_26_56` | `naive` | 41 | 20 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 169.94 | 0.50 |
