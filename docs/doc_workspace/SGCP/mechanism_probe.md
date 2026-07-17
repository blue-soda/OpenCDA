# SGCP Mechanism Probe

更新时间：2026-07-17

本文档记录主表修复阶段的第二轮机制 probe，用同一套 cluster 和 inter-cluster late fusion 评估 head-only、SGCP grid-constrained 和 full-cluster upload，定位当前 AP 损失来自哪里。

## 代码更新

`opencda.tools.offline_inference` 新增：

```text
--clustering {coalition_game,fixed_first_frame,singleton,all_in_one}
--head-rb-budget <int>
--sgcp-late-nms-thresh <float>
--sgcp-upload-mode {grid,head_only,full_cluster}
--sgcp-grid-selection-mode {utility,random,spatial_diverse}
--sgcp-grid-score-mode {utility,raw_density,density_distance}
```

含义：

- `grid`：默认 SGCP 口径，只上传 PPS/grid selection 选中的 sender grid。
- `fixed_first_frame` clustering：首帧运行 coalition game 得到固定 head/member 模板，后续帧复用该模板；每帧仍重新计算 grid density、PPS resource allocation 和 OpenCOOD 融合，用于拆分 cluster 更新的贡献。
- `head_rb_budget`：覆盖 `PotentialGame` 中每个簇头最多使用的 RB 数 `B_h`。默认 `1`，保持原始 SGCP 协议；大于 1 用作 member/grid budget sensitivity。当前 `B_h=2,rho_th=3` 已完成 11 帧 NS3 request-level replay。
- `sgcp_late_nms_thresh`：覆盖 inter-cluster late fusion 的 NMS IoU threshold。默认 `0.15`，本轮只用于排查低 IoU AP 下降是否由 late NMS 过强/过弱导致。
- `head_only`：每个 cluster head 只使用自身点云，随后做 inter-cluster late fusion。
- `full_cluster`：每个 cluster head 接收本 cluster 所有成员的完整点云，随后做 inter-cluster late fusion。
- `random` grid selection：保留 SGCP/PPS 已调度 sender link 和每条 link 的 grid 数量，但将具体 grid 替换为确定性随机候选 grid，用于判断 utility 排序是否有效。
- `spatial_diverse` grid selection：同样保留 scheduled links 和 grid 数量，但在每条 link 内用 density-aware farthest-point cover 选择空间分散的候选 grid，用于测试覆盖多样性是否比单点 utility 更适合检测 AP。
- `raw_density` / `density_distance` grid score：只改变 `PotentialGame.grid_score()`，分别测试未饱和 sender density 和 density/distance cost 是否优于原始饱和 utility。

该开关只用于机制 probe。默认仍是 `grid`，不改变主实验命令。

## 命令

Head-only：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-upload-mode head_only --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_trace.csv
```

Full-cluster：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-upload-mode full_cluster --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\full_cluster_41f_trace.csv
```

默认 SGCP grid-constrained 结果来自：

```text
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_trace.csv
```

随机 grid selection：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode random --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\random_grid_41f_trace.csv
```

Spatial-diverse grid selection：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_trace.csv
```

Fixed first-frame cluster membership：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --clustering fixed_first_frame --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\protocol_audit\fixed_first_frame_41f_trace.csv
```

Per-head RB budget sensitivity：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --head-rb-budget 2 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_41f_trace.csv

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_trace.csv
```

Late NMS threshold probe：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --sgcp-late-nms-thresh 0.05 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_nms005_41f_trace.csv

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --sgcp-late-nms-thresh 0.30 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_nms030_41f_trace.csv
```

## 结果

| Mode | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Bytes / Receiver | Avg. Sources | Avg. Uploaded Sources | Avg. Uploaded Points | Avg. Selected Grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head-only | 0.26 | 0.22 | 0.09 | 0 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| Fixed first-frame cluster, SGCP grid-constrained | 0.73 | 0.70 | 0.33 | 26,325,216 | 107,013.07 | 2.67 | 1.67 | 6,688.32 | 88.09 |
| SGCP grid-constrained | 0.77 | 0.73 | 0.35 | 26,916,208 | 109,415.48 | 2.67 | 1.67 | 6,838.47 | 87.32 |
| Random grid, same scheduled links | 0.78 | 0.75 | 0.36 | 27,908,560 | 113,449.43 | 2.67 | 1.67 | 7,090.59 | 87.32 |
| Raw-density score | 0.74 | 0.70 | 0.37 | 29,290,768 | 119,068.16 | 2.67 | 1.67 | 7,441.76 | 88.55 |
| Density-distance score | 0.74 | 0.71 | 0.37 | 29,219,088 | 118,776.78 | 2.67 | 1.67 | 7,423.55 | 88.00 |
| Spatial-diverse grid, same scheduled links | 0.79 | 0.75 | 0.37 | 28,743,280 | 116,842.60 | 2.67 | 1.67 | 7,302.66 | 87.32 |
| Spatial-diverse, `B_h=2`, `rho_th=2` | 0.75 | 0.72 | 0.41 | 27,086,400 | 110,107.32 | 2.67 | 1.67 | 6,878.55 | 89.10 |
| Spatial-diverse, `B_h=2`, `rho_th=3` | 0.76 | 0.72 | 0.42 | 27,962,864 | 113,670.18 | 2.67 | 1.67 | 7,104.39 | 90.74 |
| Spatial-diverse, `B_h=2`, `rho_th=3`, late NMS 0.05 | 0.73 | 0.70 | 0.40 | 27,962,864 | 113,670.18 | 2.67 | 1.67 | 7,104.39 | 90.74 |
| Spatial-diverse, `B_h=2`, `rho_th=3`, late NMS 0.30 | 0.75 | 0.71 | 0.41 | 27,962,864 | 113,670.18 | 2.67 | 1.67 | 7,104.39 | 90.74 |
| Full-cluster upload | 0.82 | 0.79 | 0.42 | 44,850,528 | 182,319.22 | 3.33 | 2.33 | 11,394.95 | 0.00 |

## Spatial-Diverse Channel Sweep

| Num. Channels | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Bytes / Receiver | Avg. Uploaded Sources | Avg. Uploaded Points | Avg. Selected Grids | Payload vs Full-Cluster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.56 | 0.53 | 0.27 | 14,815,408 | 60,225.24 | 0.83 | 3,764.08 | 45.58 | 33.0% |
| 10 | 0.79 | 0.75 | 0.37 | 28,743,280 | 116,842.60 | 1.67 | 7,302.66 | 87.32 | 64.1% |
| 20 | 0.80 | 0.76 | 0.41 | 37,912,544 | 154,116.03 | 2.33 | 9,632.25 | 117.18 | 84.5% |

## 结论

- Cluster-head local-only perception 很弱，说明协同上传对 AP 有显著贡献。
- 固定首帧 cluster membership 的 AP 为 `0.73/0.70/0.33`，低于动态 coalition 的 `0.77/0.73/0.35`，而 payload 略低。这说明 cluster 更新本身有可观贡献，不能把所有周期都固定为首帧拓扑；但差距小于 grid-constrained 与 full-cluster 的差距，主表修复仍应优先改进 grid/PPS 选择质量。
- Full-cluster upload 在同一 cluster 和 inter-cluster late fusion 结构下达到 `0.82/0.79/0.42`，说明 cluster formation 和 late fusion 主体并没有崩。
- SGCP grid-constrained 使用约 60.0% 的 full-cluster payload，AP@0.5 保留约 92.4% 的 full-cluster AP，但 AP@0.7 损失明显。
- Random grid selection 在同一调度链路和相同 grid 数量下达到 `0.78/0.75/0.36`，略高于当前 utility selection 的 `0.77/0.73/0.35`。这说明当前 grid utility 对检测 AP 的排序能力不足，至少在该 dump 上没有优于简单随机候选。
- `raw_density` 和 `density_distance` 提升 AP@0.7 到 `0.37`，但明显损失 AP@0.3/0.5，并增加 payload；单纯追高密度或近距离高密度不是稳健解。
- `spatial_diverse` 在相同 scheduled links 和相同 grid count 下达到 `0.79/0.75/0.37`，高于原始 utility 和 random-grid，同时仍只使用 full-cluster payload 的约 64.1%。这说明覆盖多样性是比饱和密度 utility 更有希望的算法改造方向。
- `B_h=2` sensitivity 显著提升高 IoU：`rho_th=3` 时 AP@0.7 达到 `0.42`，等于 full-cluster upload 的 AP@0.7，且 payload 只有 27,962,864 bytes、约 54.56 Mbps。但 AP@0.3/0.5 降至 `0.76/0.72`，说明更灵活的 per-head RB budget 改善了定位质量/高置信局部几何，却可能损失召回分布。该结果适合作为 high-IoU sensitivity 或后续算法调参线索；11 帧 NS3 replay 已验证 110/110 request application/RLC complete。
- Late NMS threshold probe 中，默认 `0.15` 的 `0.76/0.72/0.42` 优于 `0.05` 的 `0.73/0.70/0.40` 和 `0.30` 的 `0.75/0.71/0.41`。因此 `B_h=2` 的 AP@0.3/0.5 下降不是简单由 inter-cluster late NMS 阈值导致；后续应优先检查 member/grid selection、box score distribution 和 per-cluster detection quality。
- 子信道 sweep 显示 20 子信道 `spatial_diverse` 可达到 `0.80/0.76/0.41`，AP@0.7 已接近 full-cluster `0.42`，payload 约为 full-cluster 的 84.5%。10 子信道仍是更强的低通信主点，20 子信道适合作为 high-budget sensitivity。
- 当前主表偏低的主要嫌疑从协议链路转移到 grid/PPS 选择质量：需要把 grid utility 从“密度饱和增益”改为“检测导向的覆盖/定位增益”，并继续处理 `B_h=1`、grid budget 和 AP@0.7 定位精度。

## 下一步

- 如需进一步拆 cluster 质量，可补 all-in-one 的同 upload-mode 对照；fixed-first-frame 与 singleton 已说明动态 coalition 不是主表异常的主要来源。
- 检查 `PotentialGame.best_response()` 中 `B_h=1` 和 `max_grids_per_rb` 的实际约束，评估是否需要更灵活的 per-head member budget。
- 将 `spatial_diverse` 进一步整理为论文可解释的 coverage-aware grid utility，并补 payload-matched sweep，目标是在约 60-65% full-cluster payload 下稳定超过 random-grid。
