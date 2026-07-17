# NS3 PHY / HARQ Request-Level Trace Plan

## 目标

把当前已经完成的 LGCP request-level trace 从 application / RLC 层继续推进到 PHY / HARQ 层。

当前状态：

- OpenCDA `transfer_requests` 已带 `pkt_id`。
- ns-3 CAM header 已带 `request_id`。
- RLC TX / RX / DROP 已能输出 `request_id`。
- PHY decode diagnostics 仍是 aggregate 事件，能说明 PSCCH / PSSCH 失败原因，但不能映射回具体 LGCP upload request。

目标状态：

- 每条 PSSCH TB / HARQ feedback / decode failure 都能关联到 `request_id`。
- 每个 LGCP upload request 能形成跨层状态：
  - planned
  - RLC TX
  - PHY scheduled
  - PSCCH decoded / failed
  - PSSCH decoded / failed
  - HARQ ack / nack / timeout
  - RLC RX
  - application `cam_received`

## 为什么需要它

LGCP 论文需要解释 transmission scheduling 和两阶段 upload 的链路瓶颈。当前 11 帧 replay 已有三个层次：

| Layer | Current trace | Request-level |
| --- | --- | --- |
| OpenCDA plan | `upload_plan.csv` | yes |
| Application | `cam_received request_id` | yes |
| RLC | `[NRSL_RLC_TX/RX/DROP] request_id` | yes |
| PHY decode | `PSCCH/PSSCH_DECODE_*` | no |
| HARQ | not exported | no |

这意味着当前可以证明“很多 request 到达 RLC 但没有形成 application callback”，也可以证明“PHY aggregate 失败很多”，但还不能回答：

- 哪些 LGCP request 因 PSCCH overlap 失败？
- leader-to-RSU 和 member-to-leader 哪类 request 更容易出现 PSSCH decode failure？
- 某个 area / leader 的上传是否因为 HARQ NACK / timeout 失败？
- LGCP scheduling 改进后，是减少了 collision，还是只是改变了 application callback 可见性？

## 推荐 trace 字段

新增 PHY / HARQ 日志应至少包含：

| Field | Meaning |
| --- | --- |
| `event` | `PHY_SCHEDULE` / `PSCCH_DECODE_OK` / `PSCCH_DECODE_FAIL` / `PSSCH_DECODE_OK` / `PSSCH_DECODE_FAIL` / `HARQ_ACK` / `HARQ_NACK` / `HARQ_TIMEOUT` |
| `time_s` | ns-3 simulation time |
| `frame_index` | OpenCDA replay frame index, if available |
| `request_id` | OpenCDA transfer request id |
| `sender_l2_id` | source L2 id |
| `receiver_l2_id` | target L2 id |
| `tb_size` | transport block size or packet bytes |
| `subchannel_start` | scheduled subchannel start |
| `subchannel_num` | scheduled subchannel count |
| `slot` | slot or time resource index |
| `harq_id` | HARQ process id |
| `sinr` | observed SINR, when available |
| `tbler` | error model TBler, when available |
| `reason` | failure reason such as `decoded_overlap`, `error_model`, `decode_fail`, `nack`, `timeout` |

日志格式建议保持当前 stdout 可解析风格：

```text
[NRSL_PHY_EVENT] event=PSSCH_DECODE_FAIL time_s=1.234 frame_index=5 request_id=42 sender_l2_id=17 receiver_l2_id=21 harq_id=3 subchannel_start=8 subchannel_num=2 sinr=0.12 tbler=0.98 reason=decode_fail
```

## ns-3 落点建议

优先级从低风险到高风险：

1. **RLC -> MAC / PHY tag propagation check**
   - 确认 `LteRlcRequestIdTag` 作为 ByteTag 能随 packet copy / fragmentation / segmentation 到达 MAC / PHY 层。
   - 若 MAC / PHY 只处理 burst / TB，不保留 original packet tag，需要在 TB 构建处聚合 request id list。

2. **PHY schedule event**
   - 在 sidelink MAC/PHY 分配 TB 或 subchannel 时输出 request id、sender、receiver、subchannel。
   - 这是最关键的 bridge：只要能记录 scheduled request，后续 decode / HARQ 才能回连。

3. **PSSCH decode event**
   - PSSCH 是数据承载，应优先绑定 request id。
   - 如果一个 TB 包含多个 request，字段可写为 `request_ids=1,2,3`，OpenCDA 解析器再展开。

4. **PSCCH decode event**
   - PSCCH 是控制信息，不一定直接带 application packet tag。
   - 如果 PSCCH event 无法天然携带 request id，可通过 `(sender_l2_id, receiver_l2_id, slot, subchannel)` 与 PHY schedule event 回填。

5. **HARQ feedback**
   - HARQ ACK/NACK 通常绑定 TB / HARQ process，而不是 application packet。
   - 推荐用 `(sender_l2_id, receiver_l2_id, harq_id, slot)` 先关联到 PHY schedule，再回填 request id。

## OpenCDA 解析器扩展

`opencda/tools/lgcp_ns3_log_eval.py` 后续应新增：

- `parse_phy_request_event()`
- `parse_harq_request_event()`
- `phy_request_events.csv`
- `harq_request_events.csv`
- `request_lifecycle.csv`

`request_lifecycle.csv` 建议一行一个 LGCP upload request，字段包括：

| Field | Meaning |
| --- | --- |
| `frame_index` | replay frame |
| `request_id` | request id |
| `upload_type` | `member_to_leader` / `leader_to_rsu` |
| `source_id` | OpenCDA source |
| `target_id` | OpenCDA target |
| `bytes` | request bytes |
| `rlc_tx_count` | RLC TX event count |
| `phy_schedule_count` | scheduled PHY event count |
| `pscch_fail_count` | PSCCH failure count |
| `pssch_fail_count` | PSSCH failure count |
| `harq_nack_count` | HARQ NACK count |
| `rlc_rx_count` | RLC RX event count |
| `cam_received` | application callback observed |
| `terminal_state` | highest-confidence terminal state |

## 论文可用输出

完成后可形成三张表：

1. **Request lifecycle funnel**
   - planned -> RLC TX -> PHY scheduled -> PSSCH OK -> RLC RX -> application received。

2. **Failure reason by upload type**
   - `member_to_leader` vs `leader_to_rsu` 的 PSCCH overlap、PSSCH decode fail、HARQ NACK。

3. **Scheduling sensitivity**
   - subchannel count `Z` 或 LGCP scheduling 策略变化后，collision / HARQ / latency 的变化。

## 当前结论边界

在该 trace 完成前：

- 可以报告 request-level RLC delivery。
- 可以报告 aggregate PHY decode failure breakdown。
- 不应声称已经知道每个 LGCP request 的 PHY / HARQ 失败原因。
- 如果论文需要 PHY/HARQ 证据，应表述为“diagnostic trace shows severe aggregate PSCCH/PSSCH failures”，而不是“request-level PHY attribution”。

## 2026-07-17 Raw-Slice 3F Trace Update

Top-30 raw-slice-aware upload plan 已完成 3 帧 request-level trace smoke：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_raw_slice_upload_plan_area30/ns3_request_trace_3f_rsu21/
```

当前该 run 已包含：

- application `cam_received request_id`；
- RLC TX / RX request id；
- request-level `[NRSL_PHY_EVENT] event=PSSCH_DECODE_OK/FAIL request_ids=...`；
- aggregate PSCCH decode breakdown；
- `request_lifecycle.csv` 和 `request_lifecycle_summary.csv`。

关键结果：

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Requests with RLC TX | 88 |
| Requests with RLC RX | 14 |
| Requests with PSSCH OK | 14 |
| Requests with PSSCH FAIL | 51 |
| Application callbacks | 6 |

边界：

- PSSCH 已有 request-level attribution。
- PSCCH 仍主要作为 aggregate decode breakdown 使用。
- HARQ ACK/NACK 在该 3 帧 run 中未观测到；后续若需要 HARQ 证据，仍应使用启用 HARQ 的配置重新跑。
