# Raw-Slice Upload Plan NS3 Smoke

This smoke validates that the raw-slice-aware Top-30 LGCP upload plan can be
accepted by `offline_ns3_replay.py` and sent through a live ns-3 bridge.

## Command

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10 --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_3f_rsu21\upload_plan_replayed.csv
```

## Result

| Frame | Timestamp | Requests | Bytes |
| ---: | --- | ---: | ---: |
| 1 | `000060` | 46 | 125888 |
| 2 | `000062` | 46 | 121456 |
| 3 | `000064` | 45 | 105056 |

`upload_plan_replayed.csv` was written with positive RSU node id `21` and
request `pkt_id` values.

## Request-Level Rerun

A follow-up run captured complete ns-3 stdout as `ns3_stdout_request.log` and
replayed request ids as `upload_plan_replayed_request.csv`.

Parsed output:

```text
../ns3_request_trace_3f_rsu21/
```

Key parser summary:

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| `cam_received` | 6 |
| RLC TX events | 106 |
| RLC RX events | 20 |
| Requests with PSSCH OK | 14 |
| Requests with PSSCH FAIL | 51 |

## Boundary

The first live run confirmed bridge/replay acceptance. The follow-up rerun
provides request-level RLC / PSSCH / application lifecycle evidence for three
frames. It remains a smoke validation rather than a final network-performance
row.
