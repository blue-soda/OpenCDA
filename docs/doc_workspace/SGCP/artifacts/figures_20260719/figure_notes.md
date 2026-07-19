# SGCP Figure 2/3 Drafts

更新时间：2026-07-19

本目录保存 P5/P6 的第一版图表草稿，数据来自已复现 manifest，不重新运行推理。

## Figure 2 - Protocol-Native Aggregate AP Breakdown

- 图：`figure2_protocol_breakdown.png` / `figure2_protocol_breakdown.pdf`
- 源数据：`../table1_protocol_20260719/protocol_native_manifest.csv`
- 目的：比较 Head-only、Pure late、FullPerception-PCS、EdgeCooper-HD、SGCP-PAPG 和 Full 20-CAV upper reference。
- 图注要点：所有 AP 均为 pooled aggregate AP；每个柱组底部标注 communication label 和 evaluated sample count；Pure late 是 prediction-sharing reference，图内标为 `box 0.7`，通信量按 detection-box overhead 解释；其他 raw-LiDAR 方法标为 `raw X.X`；Full 20-CAV 是 upper reference，不是 FullPerception baseline。

当前可写结论：SGCP-PAPG 在 62.54 Mbps raw LiDAR payload 下达到 `0.81/0.78/0.39`，接近 edge/global assignment reference 的 AP@0.3/AP@0.5，但 AP@0.7 仍低于 EdgeCooper-HD 与 full-sharing upper reference。Pure late 的 AP@0.3 很强，说明 coverage 不能只归功于 SGCP 调度器。2026-07-19 已重生成图表，修正 Pure late 从 `raw 0.0` 到 `box 0.7` 的图内标注。

## Figure 3 - Fusion Contribution by IoU Threshold

- 图：`figure3_fusion_contribution.png` / `figure3_fusion_contribution.pdf`
- 源数据：`../fusion_ablation_20260719/fusion_scaffold_manifest.csv`
- 目的：展示 two-layer fusion 的分工：簇间 late fusion 提供 coverage/AP@0.3，raw LiDAR early fusion/full sharing 提供 localization/AP@0.7 上界。
- 图注要点：Clustered early-only 和 Full SGCP 使用相同 raw LiDAR payload `62.54 Mbps`；二者差异来自 inter-cluster late fusion。Full 20-CAV early 是全点云共享上界，通信量为 `118.71 Mbps`。

当前可写结论：Clustered early-only 只有 `0.38/0.36/0.20`，加入 late fusion 后 Full SGCP 达到 `0.81/0.78/0.39`，覆盖收益明显；但 Full SGCP 的 AP@0.7 仍低于 full-sharing upper reference，支撑“通信受限下接近上界但 detector/early-fusion 仍是瓶颈”的保守叙事。2026-07-19 已重生成图表，Pure late 同样标为 `box 0.7`，避免误读为零通信。

## 复现命令

```powershell
conda run -n opencda python docs\doc_workspace\SGCP\artifacts\figures_20260719\plot_breakdowns.py
```
