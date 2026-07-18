# Pure Late Fusion Prediction-Box Communication

更新时间：2026-07-19

## 目的

Pure late fusion 在当前主表中只计 `0` raw-LiDAR payload，但它实际需要交换本地检测框。该文档把 `pure_late_singleton_41f_trace.csv` 中每帧每车的 `pred_boxes` 转成 prediction-box communication budget，用于判断“20 辆 CAV 全局 late fusion 是否会因 100 ms 通信 deadline、收发延迟或信道冲突被自然限制”。

工具入口：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_late_box_comm_budget --trace-csv docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv --output-dir docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719 --box-bytes 80 --message-overhead-bytes 64 --packet-overhead-bytes 48 --mtu-bytes 1200 --total-bandwidth-mhz 20 --subchannels 10 --spectral-efficiency 6 --deadline-ms 100
```

保守版本：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_late_box_comm_budget --trace-csv docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv --output-dir docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719_box128 --box-bytes 128 --message-overhead-bytes 64 --packet-overhead-bytes 48 --mtu-bytes 1200 --total-bandwidth-mhz 20 --subchannels 10 --spectral-efficiency 6 --deadline-ms 100
```

## 当前 trace 规模

- 帧数：41
- CAV 数：20
- 每帧预测框均值：约 102
- 每帧预测框最大值：115
- 每帧非空发送者均值：约 16.34
- 每帧非空发送者最大值：18

## 20 MHz / 10 ch / 100 ms 预算

默认假设：`80 B/box + 64 B/message + 48 B/packet`，`MTU=1200 B`，每子信道等效速率为 `20 MHz / 10 ch * 6 bps/Hz = 12 Mbps`。

| Communication Mode | Mean Mbps | Max Mbps | Mean Scheduled Completion | Deadline OK | Random-Access Full Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-hop broadcast, one message per sender | 0.739 | 0.823 | 1.153 ms | 100% | 100% |
| All-to-all unicast, sender-to-19 receivers | 14.043 | 15.638 | 19.102 ms | 100% | 0% |

保守 `128 B/box` 假设：

| Communication Mode | Mean Mbps | Max Mbps | Mean Scheduled Completion | Deadline OK | Random-Access Full Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-hop broadcast, one message per sender | 1.132 | 1.265 | 1.560 ms | 100% | 100% |
| All-to-all unicast, sender-to-19 receivers | 21.515 | 24.028 | 27.336 ms | 100% | 0% |

## 解释

1. 仅靠 payload rate 或有调度传输延迟，当前 20-CAV Pure late fusion 很难被 100 ms deadline 自然限制。即使 all-to-all unicast 且按 `128 B/box` 计，调度完成时间仍只有约 27 ms。
2. 如果预测框交换是 one-hop broadcast/multicast，通信开销远低于 SGCP raw LiDAR PPS；此时 Pure late 应作为强 prediction-sharing reference，而不是零通信 baseline。
3. 如果预测框交换被定义为完全无调度的 all-to-all unicast，同一 contention round 内大量消息随机抢 10 个子信道会发生严重碰撞。这个限制来自 unscheduled access assumption，而不是来自预测框 payload 本身。
4. 论文中不能简单声称 “20 CAV late fusion 会广播风暴”。更稳妥的写法是明确区分：prediction-box sharing has low payload but limited information content; raw LiDAR sharing costs more but can recover objects missed by local detectors and improve high-IoU localization.

## 对主文图表的影响

- Table 1 中 Pure late fusion 不应继续写成 `0 Mbps`，应至少标为 `0 raw-LiDAR Mbps / 0.74-1.13 prediction-box Mbps broadcast`，或作为 prediction-sharing reference 单独说明。
- 如果要用通信约束限制 Pure late，必须采用并说明 `unscheduled all-to-all unicast` 假设，最好用 NS3 synthetic request replay 再确认 deadline delivery，而不是只用理论估算。
- SGCP 的论文优势应落在 raw LiDAR early fusion + cluster-level protocol 的感知质量与可扩展性，而不是声称所有 late-fusion prediction sharing 都通信不可行。

## Artifact

- 默认估计：`docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719\late_box_summary.csv`
- 默认逐帧：`docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719\late_box_frame_budget.csv`
- 保守估计：`docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719_box128\late_box_summary.csv`
- 保守逐帧：`docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719_box128\late_box_frame_budget.csv`

## Actual late checkpoint update

2026-07-19 复核发现，原 Table 1 / Table 2 manifest 中的 Pure late 使用 `fusion_method=early`，即 `pointpillar_early_fusion` singleton local inference + custom box-level late NMS；它不是 `pointpillar_late_fusion` checkpoint。补跑真正 `fusion_method=late` 后：

- 11 帧：AP `0.90/0.84/0.46`
- 41 帧：AP `0.89/0.83/0.49`
- `80 B/box` actual-late broadcast mean/max：`1.068/1.148 Mbps`
- `80 B/box` actual-late all-to-all mean/max：`20.298/21.815 Mbps`
- `128 B/box` actual-late broadcast mean/max：`1.654/1.782 Mbps`
- `128 B/box` actual-late all-to-all mean/max：`31.431/33.853 Mbps`

该结果进一步说明 Pure late 是强 prediction-sharing reference；若主文保留它，必须明确 checkpoint、通信类型和 detection-box overhead。
