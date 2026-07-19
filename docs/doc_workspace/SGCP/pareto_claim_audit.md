# SGCP Pareto Claim Audit

更新时间：2026-07-19

## 目的

本审计用于关闭 P4 中 “SGCP 是否在中低通信区间形成清晰 Pareto 优势” 的叙事风险。核心原则是先划分公平比较集合，再讨论 Pareto frontier：

- prediction-sharing reference：Pure late detection-box sharing，通信内容是检测框，不是 raw LiDAR point grids；
- edge/global reference：EdgeCooper-HD、global selective proxy、Full20Early 等，信息条件强于 RSU-free V2V SGCP；
- raw-LiDAR V2V / SGCP-compatible：同一 clustered two-layer scaffold 或同类 raw-LiDAR V2V selective sharing，只比较 sender/grid budget、scheduler 或 SGCP 参数。

论文主张只能落在第三类集合上；前两类作为 reference / boundary。

## Raw-LiDAR V2V Pareto Frontier

源数据：`artifacts/pareto_20260719/pareto_source.csv`。

纳入集合：

- `category in {proposed, sgcp_ablation, sgcp_sensitivity, scheduler_baseline, scheduler_baseline_proxy}`;
- `scaffold == sgcp_compatible`;
- 排除 prediction-box sharing、edge/global、full-sharing upper reference 和 negative probe。

### AP@0.3 frontier

| Method | Mbps | AP@0.3 | AP@0.5 | AP@0.7 | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| SGCPCoverage5ch20MHz | 28.91 | 0.56 | 0.53 | 0.27 | low-bandwidth SGCP sensitivity |
| SGCPCoverageCap3000 | 38.07 | 0.74 | 0.70 | 0.33 | point-cap SGCP sensitivity |
| RandomLowBudget | 48.34 | 0.75 | 0.70 | 0.34 | low-budget random |
| SGCPCoverage10chRho3Bh2 | 54.56 | 0.76 | 0.72 | 0.42 | high-IoU SGCP sensitivity |
| SGCPCoverage10chRho2 | 56.08 | 0.79 | 0.75 | 0.37 | coverage-aware SGCP |
| SGCPTargetAwarePG | 60.62 | 0.80 | 0.76 | 0.39 | target-aware PG ablation |
| SGCP_PAPG | 62.54 | 0.81 | 0.78 | 0.39 | proposed |

结论：在 raw-LiDAR V2V / SGCP-compatible 集合中，SGCP-PAPG 是 AP@0.3 frontier 的最右端最高 AP 点。它可以支撑 “coverage / network-level recall” 方向的 Pareto claim。

### AP@0.5 frontier

| Method | Mbps | AP@0.3 | AP@0.5 | AP@0.7 | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| SGCPCoverage5ch20MHz | 28.91 | 0.56 | 0.53 | 0.27 | low-bandwidth SGCP sensitivity |
| SGCPCoverageCap3000 | 38.07 | 0.74 | 0.70 | 0.33 | point-cap SGCP sensitivity |
| SGCPCoverage10chRho3Bh2 | 54.56 | 0.76 | 0.72 | 0.42 | high-IoU SGCP sensitivity |
| SGCPCoverage10chRho2 | 56.08 | 0.79 | 0.75 | 0.37 | coverage-aware SGCP |
| SGCPCoverage10chRho3 | 57.38 | 0.79 | 0.76 | 0.38 | coverage-aware SGCP |
| SGCP_PAPG_Bh3 | 62.54 | 0.80 | 0.78 | 0.40 | PAPG sensitivity |
| SGCP_PAPG | 62.54 | 0.81 | 0.78 | 0.39 | proposed |
| PACP_LiDAR | 86.56 | 0.81 | 0.79 | 0.42 | high-budget proxy |

结论：SGCP-PAPG 与 `B_h=3` sensitivity 在 62.54 Mbps 达到 AP@0.5=0.78，位于 raw-LiDAR V2V frontier；PACP-LiDAR 以更高通信和 stronger priority proxy 达到 0.79，是 high-budget boundary。

### AP@0.7 frontier

| Method | Mbps | AP@0.3 | AP@0.5 | AP@0.7 | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| SGCPCoverage5ch20MHz | 28.91 | 0.56 | 0.53 | 0.27 | low-bandwidth SGCP sensitivity |
| SGCPCoverageCap3000 | 38.07 | 0.74 | 0.70 | 0.33 | point-cap SGCP sensitivity |
| RandomLowBudget | 48.34 | 0.75 | 0.70 | 0.34 | low-budget random |
| SGCPCoverage10chRho3Bh2 | 54.56 | 0.76 | 0.72 | 0.42 | high-IoU SGCP sensitivity |

结论：PAPG 主点不是 AP@0.7 frontier；`B_h=2` / `rho=3` 的 SGCP sensitivity 以 54.56 Mbps 达到 AP@0.7=0.42，但 AP@0.3/AP@0.5 较低。论文应写成：PAPG 主点优化 network-level coverage/AP@0.3-0.5 tradeoff；AP@0.7 边界由更偏局部定位的 high-IoU sensitivity 或 high-budget/edge-assisted references 给出。

## 写作边界

可以写：

- 在 raw-LiDAR V2V / SGCP-compatible 集合中，SGCP-PAPG 位于 AP@0.3 frontier，并在 AP@0.5 上达到同预算 frontier；
- Pure late 是 prediction-sharing reference，不属于 raw-LiDAR point-grid Pareto；
- EdgeCooper-HD、PACP-LiDAR high-budget、Full20Early 是边界参考，分别对应 edge/global information、stronger proxy prior 或 full-sharing upper reference；
- AP@0.7 仍是 SGCP 的风险/边界，应解释为 early checkpoint 和 high-quality localization headroom，而不是硬写全面最优。

不应写：

- “SGCP 在所有方法和所有 IoU 上都 Pareto 最优”；
- “Pure late 是零通信 baseline”；
- “EdgeCooper-HD/PACP-LiDAR 与 SGCP 是完全同信息条件的 protocol-native baseline”。

## 对 target.md 的影响

P4 的 Pareto 验收可以按分层口径 first-pass 关闭：SGCP 在 raw-LiDAR V2V 的 AP@0.3/AP@0.5 维度有可解释 frontier 位置；AP@0.7 不关闭为强 claim，而作为 sensitivity/boundary 写入论文。剩余真正阻塞仍是 early-fusion checkpoint fine-tune 和可能的新场景。
