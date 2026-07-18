# SGCP Pareto Source Data

更新时间：2026-07-19

本目录是 Figure 1 AP-Mbps Pareto 曲线的第一版源数据，先汇总已复现的 41 帧结果，不重新跑实验。

## 口径

- AP：OpenCOOD pooled evaluator aggregate AP。
- Mbps：除 Pure late prediction-box rows 外，均为 raw LiDAR upload payload 按 41 帧、10 Hz 计算。
- Pure late：使用同 SGCP 主线一致的 early checkpoint singleton detector 加 `naive_late_fusion()`；CSV 中给出 80 B/box 的 broadcast 和 scheduled all-to-all 两种 detection-box overhead。它是 prediction-sharing reference，不是 0 Mbps baseline。
- `sgcp_compatible` rows 共享 clustering/two-layer late-fusion scaffold，只比较调度器或参数；`protocol_native` / `edge_assisted_reference` / `upper_reference` 不应和 V2V-only SGCP 直接写成同一公平 ranking。

## 当前可读结论

- SGCP-PAPG 位于中等 raw LiDAR payload 区间：`62.54 Mbps`，AP `0.81/0.78/0.39`。
- 同一 scaffold 下，PAPG 相比 forced random 只多 `0.86 Mbps`，AP 提升 `+0.04/+0.05/+0.01`。
- PAPG 相比 density/link-aware 少约 `15.0%` raw payload，并提高 AP@0.3/AP@0.5，但 AP@0.7 低 `0.01`。
- EdgeCooper-HD 和 PACP-LiDAR proxy 在 AP@0.7 更强，但使用更强信息条件或更高 payload，应作为 reference/proxy boundary 解释。
- Pure late prediction sharing 在 payload 上很强，因此主文必须把它写成 prediction-sharing reference，并强调其信息内容与 raw LiDAR early fusion 不同，不能把它当作 SGCP 需要“击败”的同类点云通信 baseline。

## 待补

- 用该 CSV 生成 AP@0.3-vs-Mbps 和 AP@0.7-vs-Mbps 图。
- 补 `rho_th`、channel count、point cap 的系统扫描，避免 Pareto 只依赖少数手工点。
- early checkpoint fine-tune 完成后，重跑 SGCP-PAPG 与 Pure late controlled baseline，并更新本 CSV。
