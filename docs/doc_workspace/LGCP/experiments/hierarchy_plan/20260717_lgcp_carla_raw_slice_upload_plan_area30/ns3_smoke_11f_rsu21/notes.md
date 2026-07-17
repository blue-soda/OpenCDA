# Raw-Slice Upload Plan 11F Request Trace

This run validates the Top-30 raw-slice-aware LGCP upload plan over the full
11-frame dump with live ns-3 and request-level parsing.

## Replay

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10 --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_11f_rsu21\upload_plan_replayed_request.csv
```

## Parser

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_11f_rsu21\ns3_stdout_request.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_11f_rsu21\upload_plan_replayed_request.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_request_trace_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

## Summary

| Metric | Value |
| --- | ---: |
| Planned requests | 504 |
| Planned bytes | 1313568 |
| Application callbacks | 55 |
| Bridge-observed delivery ratio | 0.109127 |
| RLC TX events | 546 |
| RLC RX events | 118 |
| Requests with RLC TX | 446 |
| Requests with RLC RX | 94 |
| Requests with PSSCH OK | 94 |
| Requests with PSSCH FAIL | 250 |

## Boundary

This remains an unscheduled replay smoke. The low delivery ratio is evidence
that scheduling is needed, not a final LGCP performance row.
