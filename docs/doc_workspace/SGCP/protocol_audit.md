# SGCP Offline Protocol Audit

更新时间：2026-07-16

本文档记录主表修复阶段的第一轮离线协议审计，目标是确认分簇、点云选择和子信道分配是否真实影响 OpenCOOD 融合输入。

## 代码更新

新增 `opencda.tools.offline_inference --sgcp-trace-output <csv>`。

每个 receiver / cluster-head 样本输出一行 trace：

- `scenario_id`
- `timestamp`
- `receiver_id`
- `cluster_member_ids`
- `source_cav_ids`
- `uploaded_source_ids`
- `selected_grid_counts_json`
- `point_counts_json`
- `communication_bytes`
- `channel_allocation`
- `missing_channel_sources`
- `pred_boxes`
- `gt_boxes`

同时，`build_constrained_frame()` 的 metadata 现在记录：

- receiver 所在 cluster 的 member ids。
- receiver scheduler 中的 channel allocation。

该改动只增加观测信息，不改变融合输入和 AP 计算。

## 单帧 Probe

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_000060_trace.csv
```

日志：

```text
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_000060_stdout.log
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_000060_trace.csv
```

结果摘要：

| Receiver | Cluster Members | Uploaded Sources | Channel Allocation | Missing Channel Sources |
| ---: | --- | --- | --- | --- |
| 4 | 4;8;12 | 12 | 12>4:4 | none |
| 11 | 1;2;10;11 | 10;2 | 2>11:6;10>11:0 | none |
| 13 | 9;13;14;19 | 14;19 | 14>13:1;19>13:7 | none |
| 15 | 6;15 | 6 | 6>15:5 | none |
| 16 | 5;7;16;20 | 7;20 | 7>16:2;20>16:8 | none |
| 17 | 3;17;18 | 3;18 | 3>17:3;18>17:9 | none |

单帧结论：`000060` 帧中，进入 OpenCOOD early-fusion 输入的 uploaded sources 都有 PPS channel allocation；未发现未调度 sender 绕过 PPS 融合。

## 41 帧 Probe

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_trace.csv
```

日志：

```text
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_stdout.log
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_trace.csv
```

结果：

- Frames：41
- Trace rows：246
- Cluster-head receivers / frame：6
- `missing_channel_rows`：0
- Total communication bytes：26,916,208
- Avg communication bytes / receiver sample：109,415.48
- AP@0.3 / AP@0.5 / AP@0.7：0.77 / 0.73 / 0.35

结论：

- 分簇结果真实决定 receiver / cluster head 列表和每个 receiver 的 member set。
- `PotentialGame` 输出的 grid selection 真实裁剪 sender 点云，并进入 OpenCOOD early-fusion 输入。
- 离线融合样本中的 uploaded sender 均有对应 channel allocation，未发现“未调度 member 绕过 PPS 进入融合”的协议错误。

## 当前判断

第一轮审计暂未发现离线主结果低 AP 是由 cluster/grid/channel allocation 没有接入融合输入导致。后续应转向更细消融：

- full-cluster unconstrained upload vs grid-constrained upload，量化 grid selection 本身造成的 AP 损失。
- head-only local detection vs SGCP grid upload，量化每个 cluster 的实际增益。
- inter-cluster late fusion/NMS 的召回损失，检查 detection boxes 坐标变换和 NMS 阈值。
- `B_h=1` 每簇头单 RB 是否过于保守，导致每帧只调度 10 条 link、selected grids 不足。
- `rho_th` / candidate grid utility 是否选到了对 AP@0.7 不友好的稀疏或远距离 grid。
