# FullPerception PCS Paper Audit

日期：2026-07-19

## 核查问题

用户指出 FullPerception-PCS 的 `4.99 Mbps` attentive 行通信量异常低，并要求对照原论文检查 PCS 实现，尤其是“同一个接收方的不同发送方”是否应归为 Class A 硬冲突。

## 论文结论

FullPerception 原文在 System Model 中将冲突分为 Class A 和 Class B：

- Class A：一个车辆同一时刻只能参与一条链路；若两条链路有共同节点，则存在 Class A conflict。
- Class B：接收方在同一子信道上同时接收多个传输包，并且处于干扰范围时构成干扰冲突。

因此，`L_i,j,k` 与 `L_x,j,z` 共享接收方 `N_j`，按原文属于 Class A conflict。当前 `pcs.py` 中把同 sender、同 receiver、sender/receiver 交叉复用都归为 A 类冲突是论文一致实现，不应为了提高通信量而放宽。

代码位置：

- `opencda/core/clustering/algorithms/resource_allocation/pcs.py`
- `_build_conflict_graph()`
- A 类判断包含 `sender_q == sender_p`、`receiver_q == receiver_p`、`sender_q == receiver_p`、`sender_p == receiver_q`

## 当前复现实验

所有实验均使用 20MHz/10ch、41 帧 `2026_07_15_01_26_56`、attentive detector checkpoint、`fullperception_pcs`、`all-scheduled-receivers`，并保持论文一致 Class A 冲突定义。

| Variant | Upload interpretation | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Trace |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PCS strict grid replay | selected blind-spot raw-LiDAR grids only | 0.56 | 0.41 | 0.18 | 11.22 | `attentive_pcs_budget_fix_20260719/pcs_grid_paperfaithful_div12_ov0_41f_trace.csv` |
| PCS raw-LiDAR adaptation | PCS-selected sender uploads full local point cloud | 0.63 | 0.49 | 0.17 | 32.06 | `attentive_pcs_budget_fix_20260719/pcs_fullsender_paperfaithful_div12_ov0_41f_trace.csv` |
| Rejected low-payload anchor | over-split blind spots, selected few grids | 0.59 | 0.46 | 0.22 | 4.99 | `attentive_pcs_adjust_20260719/fullperception_pcs_attentive_41f_div16_ov0_sched_trace.csv` |

## 写作口径

- 主文 Table 1 使用 `PCS raw-LiDAR adaptation`，因为它更接近本文其他 raw point-cloud baselines 的通信口径。
- 严格 PCS grid replay 作为边界说明：FullPerception 原文传输的是 blind-spot intermediate features，映射到当前 raw-LiDAR replay 会显得低 payload。
- 不使用放宽 Class A 的结果；该结果不符合原论文。
- 不再将 `4.99 Mbps` 写成 forward-writing anchor。它保留为诊断结果，说明过细 blind-spot splitting 会产生异常低通信量。
