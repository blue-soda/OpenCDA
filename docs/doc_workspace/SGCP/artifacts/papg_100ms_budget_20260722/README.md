# SGCP-PAPG 100 ms Budget Probe - 2026-07-22

Purpose: test whether relaxing the communication budget from 60 ms to 100 ms changes the restored SGCP-PAPG main result, and measure real NS3 callback/RLC/PHY delay for the resulting frame-000060 upload plan.

Dataset: `D:\Data\Carla\2026_07_15_01_26_56`, 20 CAVs, 41 frames.

Detector: `docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/enable_coperception_early_from_attentive.yaml`.

SGCP parameters:

- `resource_allocation=perception_aware_potential_game`
- `clustering=coalition_game`
- `receiver_policy=all-cluster-heads`
- `inter_cluster_late_fusion=yes`
- `N_max=4`
- `rho_th=3`
- `head_rb_budget=2`
- no point cap

Communication estimator:

- `bandwidth_mhz=40`
- `num_channels=10`
- `communication_deadline_ms=100`
- `channel_estimator=ns3`
- `ns3_tb_size_bytes=899`
- `ns3_slot_duration_ms=0.5`
- `ns3_subchannel_prbs=10`
- `ns3_symbols_per_slot=12`
- `ns3_mcs=28`

## Offline AP Result

| Budget | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | Est. frame time mean / P95 / max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100 ms | 0.87 | 0.79 | 0.37 | 61.47 | 0.71 | 62.18 | 43.68 / 44.12 / 44.32 ms |

The 100 ms result is identical to the restored 60 ms result because the restored main schedule already fits under the 60 ms NS3 estimator. Relaxing the deadline to 100 ms does not admit additional senders or grids.

## Real NS3 Delay

Frame: `000060`.

Replay plan generated from the 100 ms trace:

- Scheduled source-to-head links: `10`
- CAM chunks: `80`
- Payload: `771,280 bytes`
- Chunking: OpenCDA-compatible `10,000 bytes` max packet size

NS3 executable parameters:

```text
--slBandwidthIn100kHz=400 --targetSubchannels=10 --slSubchannelSize=10 --slMcs=28 --slSymbolsPerSlot=12
```

Measured result:

| Planned chunks | Payload bytes | App callbacks | RLC complete | PHY failures | Callback delay mean / P95 / max |
| ---: | ---: | ---: | ---: | ---: | --- |
| 80 | 771,280 | 80/80 | 80/80 | 0 | 26.51 / 53.00 / 55.00 ms |

Interpretation: the 100 ms budget is feasible in real NS3, and the same plan also satisfies the stricter 60 ms communication window. The measured max callback delay is higher than the offline estimator max because the real replay includes 10 KB packet chunking and NS3 protocol processing, but it remains below both 60 ms and 100 ms.
