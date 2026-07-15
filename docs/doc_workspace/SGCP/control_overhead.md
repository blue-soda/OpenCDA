# SGCP Control Overhead

本文档记录 SGCP 控制面开销的估算口径。该开销用于论文/回复中解释 coalition formation、density utility 和 PPS scheduling 需要交换的轻量元数据；它不同于 OpenCOOD 点云上传 payload。

## 估算入口

`opencda.tools.offline_replay` 已在 summary 中输出控制开销：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

输出字段：

```text
summary control_overhead total_bytes=187112 avg_bytes_per_frame=4563.71 beacon_bytes=52480 density_metadata_bytes=40184 cluster_control_bytes=3608 pps_schedule_bytes=90840 avg_high_density_grids=82.51 avg_scheduled_links=10.00 avg_selected_grids=523.90
```

## 估算模型

当前 first-order accounting 使用以下常数：

| Item | Bytes | Meaning |
| --- | ---: | --- |
| Beacon / CAV / frame | 64 | CAV id, pose, velocity, heading, timestamp, framing |
| Density header / CAV / frame | 16 | Density metadata header |
| Density grid entry | 8 | Compact `(grid_id, quantized density)` tuple |
| Cluster header | 8 | Cluster head id, version, frame/slot metadata |
| Cluster member entry | 2 | Compact member id |
| PPS schedule link header | 12 | sender id, receiver id, subchannel, slot, flags |
| PPS schedule grid entry | 4 | Compact selected grid id |

This is an engineering estimate, not an NS3 byte-level packet trace. It is intentionally conservative enough to compare the order of magnitude between SGCP control metadata and perception payload.

## 当前 41 帧结果

数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，`potential_game`，默认 `N_max=4`，默认 `rho_th=2.0`。

| Component | Total Bytes | Avg. Bytes / Frame |
| --- | ---: | ---: |
| Beacon | 52,480 | 1,280.00 |
| Density metadata | 40,184 | 980.10 |
| Cluster control | 3,608 | 88.00 |
| PPS schedule | 90,840 | 2,215.61 |
| Total control | 187,112 | 4,563.71 |

Associated counts:

| Metric | Avg. / Frame |
| --- | ---: |
| High-density grids advertised | 82.51 |
| Scheduled PPS links | 10.00 |
| Selected upload grids | 523.90 |

The corresponding SGCP inter-cluster late-fusion perception payload in `results.md` is 26,916,208 bytes over 41 frames. Under this estimate, control metadata is about 0.70% of the perception payload:

```text
187,112 / 26,916,208 = 0.00695
```

## 论文写作口径

建议写法：

> We report perception payload separately from SGCP control metadata. The control plane consists of periodic beacons, compact high-density grid summaries, cluster membership records, and PPS scheduling commands. In the 41-frame CARLA dump, this accounting gives 187 KB total control metadata, or 4.56 KB per frame, which is less than 1% of the point-cloud payload used by SGCP inter-cluster fusion. Therefore the main communication cost is still dominated by perception payload, while SGCP control messages should be reported as a separate lightweight overhead rather than hidden inside point-cloud bytes.

## 边界

- 当前结果是 offline accounting，不是 NS3 packet capture。若论文需要严格 MAC/RLC bytes，应在 NS3 bridge 中为 control messages 建立真实 packet trace。
- Density metadata 当前按 high-density grids 估算；如果在线实现广播 full grid density map，开销会更高。
- PPS schedule 当前按 selected grids 估算；若 schedule command 只传 grid ranges 或 bitmaps，开销可能更低。
