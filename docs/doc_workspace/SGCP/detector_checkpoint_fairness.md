# SGCP Detector / Checkpoint Fairness Audit

更新时间：2026-07-19

## 结论

SGCP 主文 raw-LiDAR 协同感知结果必须统一使用同一个“点云 -> 检测框”检测器口径。旧主线使用 `pointpillar_early_fusion` checkpoint；2026-07-19 起，forward-writing Table 1 / Figure 1/2/3/4 已切换到 attentive intermediate checkpoint candidate，并且所有 protocol-native baseline、scheduler comparison、fusion ablation、Pure late controlled reference 和 Full20Early upper reference 均使用同一 attentive detector 口径。

因此，当前 forward-writing 主表和 fusion ablation 中的 Pure late 行采用以下 controlled baseline：

- 每个 CAV 只用自己的点云做 singleton local inference；
- singleton local inference 使用与 SGCP raw-LiDAR pipeline 相同的 detector checkpoint。当前 forward-writing candidate 为 attentive checkpoint；legacy reference 为 `pointpillar_early_fusion` checkpoint；
- 多车预测框汇总使用与 SGCP 簇间 late fusion 相同的 `OpenCOODManager.naive_late_fusion()` box-level NMS；
- 通信量不记为 0，而用 `late_fusion_box_comm.md` 中的 prediction-box broadcast / all-to-all overhead 估算。

`pointpillar_late_fusion` checkpoint 的结果只作为 detector sensitivity / prediction-sharing reference，不进入 raw-LiDAR 主表公平比较。

## 为什么这样处理

用户已明确：SGCP 两层融合中，所有 “点云 -> 检测框” 过程应使用同一个 checkpoint；为了公平，所有 baseline 也应使用这个 checkpoint，包括纯晚期 baseline。

SGCP 的核心机制是 raw point-cloud region sharing + intra-cluster early fusion + inter-cluster box-level late fusion。如果 Pure late 使用独立的 `pointpillar_late_fusion` checkpoint，而 SGCP 使用 `pointpillar_early_fusion` checkpoint，那么主表会同时比较通信协议和 detector checkpoint，结论不可解释。

## 当前实验证据

| Variant | Frames | Detector / First-stage Fusion | Box-level Fusion | AP@0.3 | AP@0.5 | AP@0.7 | Raw-LiDAR Mbps | 用途 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| Pure late controlled legacy | 41 | `pointpillar_early_fusion` singleton local detector | `naive_late_fusion()` | 0.82 | 0.76 | 0.37 | 0.00 | legacy checkpoint reference |
| Pure late actual-late sanity | 41 | `pointpillar_late_fusion` local detector | `naive_late_fusion()` | 0.89 | 0.83 | 0.49 | 0.00 | detector sensitivity，不进公平主表 |
| SGCP PAPG mainline | 41 | `pointpillar_early_fusion` raw point-cloud early fusion | `naive_late_fusion()` | 0.81 | 0.78 | 0.39 | 62.54 | SGCP 主线 |
| SGCP PAPG forced-late sanity | 41 | `pointpillar_late_fusion` over scheduled source set | `naive_late_fusion()` | 0.87 | 0.81 | 0.48 | 62.54 | checkpoint sensitivity，不代表 SGCP 协议 |
| Pure late attentive controlled | 41 | attentive checkpoint singleton local detector | `naive_late_fusion()` | 0.82 | 0.65 | 0.28 | 0.00 | current forward-writing prediction-sharing reference |
| SGCP PAPG attentive | 41 | attentive checkpoint raw point-cloud early fusion | `naive_late_fusion()` | 0.87 | 0.81 | 0.36 | 62.54 | current forward-writing SGCP candidate |
| Full20Early attentive upper | 41 | attentive checkpoint full 20-CAV raw point-cloud early fusion | none | 0.88 | 0.85 | 0.45 | 118.71 | current full-sharing upper reference |

Prediction-box overhead 见 `late_fusion_box_comm.md`：

- 80 B/box one-hop broadcast：平均 0.739 Mbps，最大 0.823 Mbps；
- 80 B/box scheduled all-to-all unicast：平均 14.043 Mbps，最大 15.638 Mbps；
- actual-late checkpoint 会产生更多检测框，80 B/box broadcast 约 1.068/1.148 Mbps mean/max。

Attentive checkpoint sensitivity 的 prediction-box overhead 见
`artifacts/early_from_late_checkpoint_20260719/pure_late_attentive_box_comm_80`
和
`artifacts/early_from_late_checkpoint_20260719/pure_late_attentive_box_comm_128`：

- 80 B/box one-hop broadcast：平均 1.37 Mbps，最大 1.51 Mbps；
- 80 B/box scheduled all-to-all unicast：平均 25.97 Mbps，最大 28.60 Mbps；
- 128 B/box one-hop broadcast：平均 2.13 Mbps，最大 2.35 Mbps；
- 128 B/box scheduled all-to-all unicast：平均 40.53 Mbps，最大 44.65 Mbps。

该 attentive candidate 的关键结论是：当 detector/checkpoint 对所有方法统一换成 attentive 权重时，Pure late controlled 为 `0.82/0.65/0.28`，而 SGCP-PAPG attentive 为 `0.87/0.81/0.36`。这说明同 detector 条件下，SGCP 的 scheduled raw point-cloud early fusion 仍然带来 AP@0.5/AP@0.7 收益；它不是单纯被 late fusion coverage 或 detector checkpoint 盖住的结果。

## 论文写作口径

主文应这样写：

- Pure late 是 prediction-sharing reference，不是 raw-LiDAR point-cloud sharing baseline；
- 为避免 detector checkpoint 混淆，主表中的 Pure late 使用与 SGCP 一致的 checkpoint 进行 singleton local detection；当前 forward-writing 表采用 attentive checkpoint，legacy early-checkpoint 表只作 reference；
- actual late checkpoint 的结果作为 sensitivity study，说明当前场景中 box-level prediction sharing 很强，但不用于 raw-LiDAR 协议公平排名；
- attentive checkpoint 已补齐完整 Table 1 / Table 2 / Table 3 / Pareto / figures，因此可作为当前 forward-writing candidate。若远端 fine-tune 回收更好的 early checkpoint，需要新建 artifact 版本并重跑所有 baseline 后再替换；
- SGCP 的优势应写成“在 raw-LiDAR V2V 协议中，用中等点云 payload 获得接近/超过 controlled Pure late 的 AP@0.5/AP@0.7，同时保持多层融合叙事”，而不是声称全面击败所有 prediction-sharing detector 组合。

## 当前风险

`pointpillar_early_fusion` checkpoint 仍是 SGCP 最大实验风险。当前 remote watcher 已在 `mindspore-187:/data2/gzc/sgcp_early_train/` 等待 GPU 空闲进行 fine-tune。新 checkpoint 回收后，需要重跑：

- SGCP PAPG mainline；
- Pure late controlled singleton；
- fusion scaffold ablation；
- AP-Mbps Pareto 中的关键 SGCP/Pure late 点。
