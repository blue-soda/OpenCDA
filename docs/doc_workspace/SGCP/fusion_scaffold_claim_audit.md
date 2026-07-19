# SGCP Fusion Scaffold Claim Audit

更新时间：2026-07-19

## 目的

本审计用于关闭 P2/P6 中 fusion scaffold ablation 的叙事风险：证明 two-layer fusion 有贡献，但不把当前 checkpoint 下尚未充分成立的 AP@0.7 强 claim 写过头。

源数据：

- `artifacts/fusion_ablation_20260719/fusion_scaffold_manifest.csv`
- `artifacts/table1_protocol_20260719/protocol_native_manifest.csv`

## 核心结果

| Variant | Fusion / Sharing | Samples | Raw payload Mbps | AP@0.3 | AP@0.5 | AP@0.7 | 可写作用 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Head-only | no point sharing, no useful network fusion | 41 | 0.00 | 0.26 | 0.22 | 0.09 | lower reference |
| Pure late controlled | singleton local detector + box NMS | 41 | 0.00 raw LiDAR | 0.82 | 0.76 | 0.37 | prediction-sharing reference |
| One-cluster early-only / Full20Early | all 20 CAV raw point clouds | 41 | 118.71 | 0.85 | 0.83 | 0.48 | full raw-sharing upper reference |
| Clustered early-only | SGCP clusters + PAPG raw grids, no inter-cluster late | 246 | 62.54 | 0.38 | 0.36 | 0.20 | shows cluster-local early-only coverage is insufficient |
| Full SGCP | clustered raw grids + inter-cluster late | 41 | 62.54 | 0.81 | 0.78 | 0.39 | proposed two-layer protocol |

## 可量化结论

Full SGCP 使用 `62.54 / 118.71 = 52.7%` 的 full raw-sharing payload，保留了 full-sharing upper reference 的：

- AP@0.3：`0.81 / 0.85 = 95.3%`
- AP@0.5：`0.78 / 0.83 = 94.0%`
- AP@0.7：`0.39 / 0.48 = 81.3%`

与 clustered early-only 相比，Full SGCP 在相同 raw-LiDAR payload 下从 `0.38/0.36/0.20` 提升到 `0.81/0.78/0.39`。因此，inter-cluster late fusion 对 network-level coverage / AP@0.3/AP@0.5 的贡献非常明确。

与 controlled Pure late 相比，Full SGCP 为 `0.81/0.78/0.39`，Pure late 为 `0.82/0.76/0.37`。这说明 SGCP raw point-cloud sharing 在 AP@0.5/AP@0.7 上有小幅收益，但 AP@0.3 不超过 prediction-sharing reference。论文中应把 Pure late 写成不同通信内容的 strong reference，而不是 SGCP 必须全面击败的同类 baseline。

## 写作边界

可以写：

- cluster-local early-only 覆盖不足，必须通过 inter-cluster late fusion 传播检测结果；
- Full SGCP 在约 52.7% full-sharing payload 下保留了 full-sharing AP@0.3/AP@0.5 的 95.3%/94.0%；
- AP@0.7 仍落后 full raw-sharing upper reference，说明高 IoU localization 受 raw point-cloud availability、view selection 和 early checkpoint 限制；
- two-layer fusion 的主贡献是 coverage/AP@0.3-0.5，AP@0.7 是 sensitivity/headroom 叙事。

不应写：

- clustered early fusion alone already gives strong AP；
- current SGCP fully closes high-IoU localization gap；
- Pure late 是 0 通信或同类 raw-LiDAR baseline。

## 对 target.md 的影响

P2 的验收可以 first-pass 关闭，但带有明确边界：

- “late fusion 提升覆盖”是强结论；
- “early fusion 支撑 AP@0.7”只能写为 full raw-sharing upper reference 和 SGCP-vs-Pure-late small gain 所体现的 localization headroom，不能写成当前主线已经高 IoU 最优；
- early checkpoint fine-tune 仍作为 P1/P4 的核心风险保留。
