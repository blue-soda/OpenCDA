# SGCP Pareto Source Data

更新时间：2026-07-19

本目录是 Figure 1 AP-Mbps Pareto 曲线的第一版源数据，汇总已复现的 41 帧结果。2026-07-19 第二轮更新已补入 5ch stress、`B_h=2/3`、communication-aware low-budget 与 PACP-LiDAR low-budget 点；第三轮更新补跑并纳入 Random/Density/Communication-aware budget sweep；第四轮更新补入 EdgeCooper-HD low/high budget sweep；第五轮完成 FullPerception-PCS 参数 first-pass 审计，Pareto 源表保留可复现的 41 帧 tuned PCS anchor，11 帧 granularity/overlap 趋势单独放在 `pcs_parameter_sweep_20260719`，避免混入 41 帧 Pareto。

## 口径

- AP：OpenCOOD pooled evaluator aggregate AP。
- Mbps：除 Pure late prediction-box rows 外，均为 raw LiDAR upload payload 按 41 帧、10 Hz 计算。
- Pure late：使用同 SGCP 主线一致的 early checkpoint singleton detector 加 `naive_late_fusion()`；CSV 中给出 80 B/box 的 broadcast 和 scheduled all-to-all 两种 detection-box overhead。它是 prediction-sharing reference，不是 0 Mbps baseline。
- `sgcp_compatible` rows 共享 clustering/two-layer late-fusion scaffold，只比较调度器或参数；`protocol_native` / `edge_assisted_reference` / `upper_reference` 不应和 V2V-only SGCP 直接写成同一公平 ranking。

## 当前可读结论

- SGCP-PAPG 位于中等 raw LiDAR payload 区间：`62.54 Mbps`，AP `0.81/0.78/0.39`。
- SGCP sensitivity 目前覆盖 `5ch` stress、`10ch rho2/rho3`、`20ch rho2`、`cap=3000`、`B_h=2/3`。这些点显示通信下降会先损伤 AP@0.3/AP@0.5 覆盖，`B_h=3` 能把 AP@0.7 推到 `0.40`，但不改善 AP@0.3。
- 同一 scaffold 下，PAPG 相比 high-budget forced random 只多 `0.86 Mbps`，AP 提升 `+0.04/+0.05/+0.01`；相比 low-budget random 多约 `14.20 Mbps`，AP 提升 `+0.06/+0.08/+0.05`。
- Density/communication-aware 的预算扫描显示：low-budget density `61.31 Mbps` 得到 `0.78/0.74/0.40`；high-budget communication-aware `75.94 Mbps` 得到 `0.80/0.76/0.42`。PAPG 在 AP@0.3/AP@0.5 上更高或持平且 payload 更低，但 AP@0.7 仍低于 high-budget link-aware reference。
- EdgeCooper-HD 预算扫描显示低预算 `36.10 Mbps` 时只有 `0.65/0.61/0.30`，高预算 `65.40 Mbps` 时达到 `0.81/0.78/0.42`。因此 EdgeCooper-HD 的 AP@0.7 优势依赖 edge/global assignment 信息条件和更高通信预算，应作为 reference/proxy boundary 解释。
- PACP-LiDAR proxy 在 AP@0.7 更强，但使用迁移自 RGB/BEV priority matching 的更强 prior 且 payload 更高，应作为 proxy boundary 解释。
- Pure late prediction sharing 在 payload 上很强，因此主文必须把它写成 prediction-sharing reference，并强调其信息内容与 raw LiDAR early fusion 不同，不能把它当作 SGCP 需要“击败”的同类点云通信 baseline。

## 图表草稿

- AP@0.3：`figure1_pareto_ap03.png` / `figure1_pareto_ap03.pdf`
- AP@0.7：`figure1_pareto_ap07.png` / `figure1_pareto_ap07.pdf`
- 绘图脚本：`plot_pareto.py`

当前图是 source-point draft：可以用于结果判断和论文版式设计，但正式论文图仍建议按主文分层语义重绘，避免把 prediction-sharing reference、edge-assisted reference 和 V2V-only raw-LiDAR Pareto 混成单一公平排名。若直接进入论文，caption 必须说明虚线 frontier 只连接 raw-LiDAR rows，Pure late prediction-box rows 是通信形态不同的 reference。

## 待补

- SGCP 已有 first-pass 扫描，但若 early checkpoint 回收后主线数值变化，需要重跑 `rho_th/channel/B_h/point cap`。
- Random/Density/Link-aware 已有 first-pass budget sweep：random low/high、density low/high、communication-aware low/high。若进入 camera-ready 附录，可继续补更密集的 member/grid grid。
- EdgeCooper-HD 已有 low/high budget sweep；若进入 camera-ready 附录，可继续补原版 RSU scheduling 与更多 sender-cap 点。
- FullPerception-PCS 已补原生参数 first-pass：`pcs_parameter_sweep_20260719` 记录 11 帧 blind-spot granularity / overlap threshold 趋势和 41 帧 `div12/ov0` anchor。更激进 41 帧 sweep 出现候选规模/运行时不可承受，因此不混入 Pareto 源图。
- early checkpoint fine-tune 完成后，重跑 SGCP-PAPG 与 Pure late controlled baseline，并更新本 CSV。
