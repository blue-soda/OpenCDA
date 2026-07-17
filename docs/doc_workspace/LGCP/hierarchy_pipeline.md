# LGCP Hierarchy Pipeline

## 当前目标

将 LGCP 从 offline subset proxy 推进到可执行的 hierarchy control-plane：

1. RSU 读取 area confidence。
2. RSU 选择高优先级 area。
3. 为每个 area 构造可重叠 CAV group。
4. 为每个 area 选择 leader。
5. 输出 member-to-leader upload 与 leader-to-RSU upload。
6. 统计 control-plane / upload / broadcast byte proxy。

当前实现仍不做真实 feature slicing、leader local fusion 或 RSU global aggregation，但已经给出了后续在线实现和 NS3 replay 所需的计划表结构。

## 当前工具

脚本：

```text
opencda/tools/lgcp_hierarchy_plan_eval.py
```

输入：

- `area_records.csv`
- 可选 `area_quality.csv`

输出：

- `area_assignment_plan.csv`：每个 frame / area 的 group、leader 和 confidence。
- `upload_plan.csv`：member-to-leader 与 leader-to-RSU 上传请求。
- `hierarchy_frame_summary.csv`：逐帧 packet / byte / leader load。
- `hierarchy_summary.csv`：跨帧均值和最大值。

## 当前 11 帧 Top-40 结果

配置：

- `delta_g=0.05`
- `max_group_size=4`
- `max_areas=40`
- `feature_packet_bytes=10000`
- `leader_result_bytes=2000`
- `assignment_bytes=64`
- `broadcast_bytes=2000`

平均结果：

| 指标 | 数值 |
| --- | --- |
| frames | 11 |
| covered areas / frame | 40 |
| average group size | 1.536364 |
| average group confidence | 0.908059 |
| member-to-leader packets / frame | 21.454545 |
| leader-to-RSU packets / frame | 40 |
| total byte proxy / frame | 299105.454545 |
| active leaders / frame | 15.090909 |
| leader max load / frame | 8.818182 |

## Leader Result / RSU Aggregation Proxy

2026-07-17 新增离线工具：

```text
opencda/tools/lgcp_hierarchy_aggregation_eval.py
```

该工具输入 `area_assignment_plan.csv` 和 `area_quality.csv`，输出：

- `leader_local_results.csv`：每个 area 的 leader、group members、group confidence、area-level quality 和 confidence-weighted quality。
- `rsu_global_frame_summary.csv`：每帧 RSU 聚合后的 selected GT coverage、mean area quality、leader load。
- `rsu_global_summary.csv`：跨帧 summary。

运行命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_aggregation_eval --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\area_assignment_plan.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_aggregation_top40_11f --quality-field recall_05
```

当前 11 帧 Top-40 proxy：

| 指标 | 数值 |
| --- | --- |
| frames | 11 |
| quality areas / frame | 33.000000 |
| selected hierarchy areas / frame | 40.000000 |
| selected GT ratio | 1.000000 |
| mean selected area recall@0.5 | 0.670455 |
| mean confidence-weighted quality | 0.609181 |
| active leaders / frame | 15.090909 |
| leader max load / frame | 8.818182 |

注意：

- `selected_area_ratio` 会大于 1，因为 hierarchy plan 固定选 Top-40 area，而 `area_quality.csv` 只包含有 prediction / GT quality 记录的 area；论文中应优先报告 `selected_gt_ratio`。
- 这仍是 proxy：leader local result 不是真实 feature slicing + OpenCOOD fusion，而是将现有 area-level quality 映射到 leader result 记录。
- 该步骤的价值是补齐完整 hierarchy 的数据接口：area assignment -> leader local result -> RSU global summary。

## Offline NS3 Replay 接入

`opencda/tools/offline_ns3_replay.py` 已支持：

```powershell
--lgcp-upload-plan <upload_plan.csv>
--dry-run
```

Dry-run 验证命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --dry-run
```

Dry-run 结果：

| Frame range | Nodes | Requests / frame | Bytes / frame |
| --- | --- | --- | --- |
| 11 frames | 21 | 60-63 | 280000-310000 |

说明：

- 21 个节点包括 20 个 CAV 和 RSU `-1`。
- `member_to_leader` 与 `leader_to_rsu` 都被转换为 NS3 `transfer_requests`。
- 这只是请求构建验证；真实 latency / delivery 仍需启动 ns-3 后运行非 dry-run。

## Offline NS3 联机 Smoke

3 帧联机 smoke 已完成。由于 ns-3 不接受负数节点 ID，replay 侧将 dump 中的 RSU `-1` 映射为正整数节点 `21`。

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --drain-seconds 0.3 --sync-timeout 10
```

结果：

| 指标 | 数值 |
| --- | --- |
| frames | 3 |
| nodes | 21 |
| requests / frame | 62 |
| bytes / frame | 300000 |
| parsed `cam_received` | 5 |
| parsed RSU receives | 3 |
| average parsed delay | 16 ms |
| max parsed delay | 31 ms |

日志目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_smoke_3f_rsu21/
```

注意：

- 初次联机时使用 RSU `-1`，ns-3 报告 `(leader, -1) skipped` 和 invalid vehicle payload；已修正为 RSU node `21`。
- 当前 summary 只统计 ns-3 回传给 OpenCDA 的 `cam_received`，不是完整 request-level delivery ratio。

## Offline NS3 11 帧 Replay

11 帧联机 replay 已完成，使用同一份 `upload_plan.csv` 和 RSU node `21`。

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10
```

解析：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_smoke_11f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_smoke_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

结果：

| 指标 | 数值 |
| --- | --- |
| frames | 11 |
| planned requests | 676 |
| planned bytes | 3240000 |
| observed `cam_received` | 31 |
| bridge-observed delivery ratio | 0.045858 |
| observed bytes | 174000 |
| average delay | 109.645 ms |
| p95 delay | 209 ms |
| max delay | 211 ms |

按阶段：

| Upload type | Planned | Observed | Bridge-observed ratio | Avg delay |
| --- | --- | --- | --- | --- |
| leader_to_rsu | 440 | 17 | 0.038636 | 75.471 ms |
| member_to_leader | 236 | 14 | 0.059322 | 151.143 ms |

日志目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_smoke_11f_rsu21/
```

注意：

- 该 summary 是 bridge-observed `cam_received` 统计，不是完整 PHY/RLC trace。
- stdout 中存在大量 `PSCCH_DECODE_FAIL` 和 `reason=error_model`，说明当前 NS3/V2X 参数下链路解码失败严重。
- 后续若要把 delivery ratio 写入论文，需要接入 ns-3 侧更细的 PHY/RLC trace，并区分调度拥塞、SINR/error model 和超时。

## Application Request-ID Trace

ns-3 侧已扩展 CAM header，将 OpenCDA replay 发送的 `pkt_id` 作为 `request_id` 透传到接收端。`cam_received` 现在包含：

```json
{"type":"cam_received","sender_id":1,"request_id":6,"receiver_id":21,...}
```

解析器优先按 `(frame_index, request_id)` 对齐 `upload_plan.csv`，旧日志无 `request_id` 时才回退到 `(frame, source, target, bytes)`。

当前 11 帧 request-id replay：

| 指标 | 数值 |
| --- | --- |
| output dir | `ns3_request_id_11f_rsu21` |
| planned requests | 676 |
| observed `cam_received` | 31 |
| matched `cam_received` | 31 |
| match method | `frame_request_id` |
| bridge-observed delivery ratio | 0.045858 |

含义：

- Bridge-observed delivery 已经可以精确回连到 LGCP area / upload type / leader-member request。
- 这仍不是完整链路 trace；RLC PDU 已另行绑定 request id，HARQ / PHY event 与 request id 的绑定还需要继续扩展 ns-3。

## RLC Request-ID Trace

ns-3 侧已新增 `LteRlcRequestIdTag`，OpenCDA replay 发送 CAM 时将 `request_id` 同时写入 packet tag / byte tag；RLC UM 日志现在输出：

```text
[NRSL_RLC_TX] ... request_id=...
[NRSL_RLC_RX] ... request_id=...
[NRSL_RLC_DROP] ... request_id=...
```

解析器会将 RLC event 按 `(frame_index, request_id)` 映射回 `upload_plan.csv`。

当前 11 帧 RLC request-id replay：

| 指标 | 数值 |
| --- | --- |
| output dir | `ns3_rlc_request_id_11f_rsu21` |
| planned requests | 676 |
| RLC TX events | 1131 |
| RLC RX events | 252 |
| RLC DROP events | 0 |
| matched RLC TX events | 1131 |
| matched RLC RX events | 252 |
| unique TX requests | 614 |
| unique RX requests | 164 |
| RLC request RX ratio | 0.242604 |
| application `cam_received` | 31 |
| bridge-observed delivery ratio | 0.045858 |

含义：

- RLC 层现在已经能按 LGCP upload request 做严格归因，可用于区分“已进入 RLC / RLC 接收 / application bridge 可见接收”。
- RLC RX 覆盖的 request 明显多于 `cam_received`，说明 application bridge 统计低估了链路层成功到达情况。
- 当前 RLC trace 还没有把 HARQ feedback 和 PHY decode event 绑定回 request id，因此 `PSCCH/PSSCH` breakdown 仍是 aggregate 诊断。

## PHY Decode Breakdown

`opencda/tools/lgcp_ns3_log_eval.py` 已能解析当前 ns-3 stdout 中的 PHY decode diagnostics：

- `PSCCH_DECODE_OK`
- `PSCCH_DECODE_FAIL`
- `PSSCH_DECODE_OK`
- `PSSCH_DECODE_FAIL`

输出：

```text
phy_decode_events.csv
phy_decode_summary.csv
```

当前 11 帧结果：

| Channel | Status | Reason | Count | Channel ratio |
| --- | --- | --- | --- | --- |
| PSCCH | FAIL | decoded_overlap | 5736 | 0.470164 |
| PSCCH | FAIL | error_model | 1755 | 0.143852 |
| PSCCH | OK | - | 4709 | 0.385984 |
| PSSCH | FAIL | decode_fail | 288 | 0.533333 |
| PSSCH | OK | - | 252 | 0.466667 |

含义：

- 当前链路失败的主要可见原因是 PSCCH `decoded_overlap`，直接指向 subchannel / timeslot scheduling 的必要性。
- PHY breakdown 仍不能替代 request-level RLC delivery ratio；application `cam_received` 和 RLC event 已通过 request id 映射到 `upload_plan.csv`，但 PSSCH / HARQ event 还没有绑定 request id。

## 论文含义

该结果可以支撑机制说明：

- LGCP area-task groups 是可重叠的，不是传统 disjoint clustering。
- 一个 CAV 可以在多个 area 中承担 leader 或 member 角色。
- Leader local fusion 前的 member upload 和 fusion 后的 leader-to-RSU upload 是两个不同阶段。
- RSU control-plane assignment 可以被显式统计为 bytes。

## 未完成部分

这还不是完整 LGCP：

- 还没有真正切分 area-specific feature slice。
- leader local fusion 现在已有离线 result proxy，但仍未调用 OpenCOOD / model-level fusion。
- RSU global aggregation 现在已有 coverage / quality proxy summary，但仍未生成真实 global perception result。
- scheduling 仍未接入 LGCP 专用 subchannel / latency 优化；当前只是将 hierarchy upload plan 转换为 NS3 transfer requests，解析 request-id `cam_received`、RLC TX/RX/DROP 和 PHY decode diagnostics。

下一步应把 PHY / HARQ event 继续绑定到 request id，或实现 leader local fusion 的离线近似。
