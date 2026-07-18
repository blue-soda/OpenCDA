# SGCP Pareto Source Data

更新时间：2026-07-19

本目录是 Figure 1 AP-Mbps Pareto 曲线的第一版源数据，汇总已复现的 41 帧结果，不重新跑实验。2026-07-19 第二轮更新已补入 5ch stress、`B_h=2/3`、communication-aware low-budget 与 PACP-LiDAR low-budget 点，使 SGCP 主线、低带宽压力和若干 scheduler proxy 至少具备 first-pass 扫描覆盖。

## 口径

- AP：OpenCOOD pooled evaluator aggregate AP。
- Mbps：除 Pure late prediction-box rows 外，均为 raw LiDAR upload payload 按 41 帧、10 Hz 计算。
- Pure late：使用同 SGCP 主线一致的 early checkpoint singleton detector 加 `naive_late_fusion()`；CSV 中给出 80 B/box 的 broadcast 和 scheduled all-to-all 两种 detection-box overhead。它是 prediction-sharing reference，不是 0 Mbps baseline。
- `sgcp_compatible` rows 共享 clustering/two-layer late-fusion scaffold，只比较调度器或参数；`protocol_native` / `edge_assisted_reference` / `upper_reference` 不应和 V2V-only SGCP 直接写成同一公平 ranking。

## 当前可读结论

- SGCP-PAPG 位于中等 raw LiDAR payload 区间：`62.54 Mbps`，AP `0.81/0.78/0.39`。
- SGCP sensitivity 目前覆盖 `5ch` stress、`10ch rho2/rho3`、`20ch rho2`、`cap=3000`、`B_h=2/3`。这些点显示通信下降会先损伤 AP@0.3/AP@0.5 覆盖，`B_h=3` 能把 AP@0.7 推到 `0.40`，但不改善 AP@0.3。
- 同一 scaffold 下，PAPG 相比 forced random 只多 `0.86 Mbps`，AP 提升 `+0.04/+0.05/+0.01`。
- PAPG 相比 density/link-aware 少约 `15.0%` raw payload，并提高 AP@0.3/AP@0.5，但 AP@0.7 低 `0.01`。
- EdgeCooper-HD 和 PACP-LiDAR proxy 在 AP@0.7 更强，但使用更强信息条件或更高 payload，应作为 reference/proxy boundary 解释。
- Pure late prediction sharing 在 payload 上很强，因此主文必须把它写成 prediction-sharing reference，并强调其信息内容与 raw LiDAR early fusion 不同，不能把它当作 SGCP 需要“击败”的同类点云通信 baseline。

## 图表草稿

- AP@0.3：`figure1_pareto_ap03.png` / `figure1_pareto_ap03.pdf`
- AP@0.7：`figure1_pareto_ap07.png` / `figure1_pareto_ap07.pdf`
- 绘图脚本：`plot_pareto.py`

当前图是 source-point draft：可以用于结果判断和论文版式设计，但正式论文图仍建议按主文分层语义重绘，避免把 prediction-sharing reference、edge-assisted reference 和 V2V-only raw-LiDAR Pareto 混成单一公平排名。若直接进入论文，caption 必须说明虚线 frontier 只连接 raw-LiDAR rows，Pure late prediction-box rows 是通信形态不同的 reference。

## 待补

- SGCP 已有 first-pass 扫描，但若 early checkpoint 回收后主线数值变化，需要重跑 `rho_th/channel/B_h/point cap`。
- Random/Density/Link-aware 仍需要更系统的 member/grid budget sweep；当前源表只有 forced random、density high-budget 与 communication-aware low-budget/high-budget 代表点。
- EdgeCooperV2V+ / EdgeCooper-inspired 仍需要 sender cap、assignment budget、half-duplex constraint 扫描；当前主点是 EdgeCooper-HD proxy。
- FullPerception-PCS tuning 已进入源表，但 PCS 原生参数尚未形成完整 Pareto sweep。
- early checkpoint fine-tune 完成后，重跑 SGCP-PAPG 与 Pure late controlled baseline，并更新本 CSV。
