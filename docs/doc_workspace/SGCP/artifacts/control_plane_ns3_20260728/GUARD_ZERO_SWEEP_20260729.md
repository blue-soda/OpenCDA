# Guard / Zero-Time Control Sweep

Date: 2026-07-29.

This probe tests whether SGCP control-plane reliability depends on the NR
sidelink bearer guard and the application-layer zero-time send delay.

Protocol:

- 20 CAVs.
- 70 compact aggregated control summaries.
- 400 B per summary.
- 10 target subchannels.
- 10 requests per batch.
- Batch step: 2 ms.
- Cast modes: unicast and broadcast.
- Timing modes:
  - `at_zero`: send immediately after NS3 connection.
  - `after_activation`: synchronize to the bearer activation boundary before
    sending.
- Guard values: 0 ms and 1 ms.
- Zero-time send delay values: 0 ms and 1 ms.

Raw result CSV:

```text
C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\guard_zero_sweep_20260729\guard_zero_sweep_results.csv
```

| Cast | Timing | Guard | Zero-delay | Scheduled requests | Expected callbacks | Observed callbacks | Unique requests | Max receive timestamp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unicast | at_zero | 1 ms | 1 ms | 70 | 70 | 60 | 60 | 13 ms |
| unicast | at_zero | 1 ms | 0 ms | 70 | 70 | 60 | 60 | 13 ms |
| unicast | at_zero | 0 ms | 1 ms | 70 | 70 | 60 | 60 | 13 ms |
| unicast | at_zero | 0 ms | 0 ms | 70 | 70 | 60 | 60 | 13 ms |
| unicast | after_activation | 1 ms | 1 ms | 70 | 70 | 70 | 70 | 15 ms |
| unicast | after_activation | 1 ms | 0 ms | 70 | 70 | 70 | 70 | 15 ms |
| unicast | after_activation | 0 ms | 1 ms | 70 | 70 | 70 | 70 | 14 ms |
| unicast | after_activation | 0 ms | 0 ms | 70 | 70 | 70 | 70 | 14 ms |
| broadcast | at_zero | 1 ms | 1 ms | 70 | 700 | 600 | 60 | 13 ms |
| broadcast | at_zero | 1 ms | 0 ms | 70 | 700 | 600 | 60 | 13 ms |
| broadcast | at_zero | 0 ms | 1 ms | 70 | 700 | 600 | 60 | 13 ms |
| broadcast | at_zero | 0 ms | 0 ms | 70 | 700 | 600 | 60 | 13 ms |
| broadcast | after_activation | 1 ms | 1 ms | 70 | 700 | 699 | 70 | 15 ms |
| broadcast | after_activation | 1 ms | 0 ms | 70 | 700 | 699 | 70 | 15 ms |
| broadcast | after_activation | 0 ms | 1 ms | 70 | 700 | 699 | 70 | 14 ms |
| broadcast | after_activation | 0 ms | 0 ms | 70 | 700 | 699 | 70 | 14 ms |

Interpretation:

- Sending at simulator time zero loses the first 10-request batch regardless of
  guard or zero-delay. This is a bearer-activation ordering issue, not a channel
  capacity issue.
- If the sender waits until after bearer activation, unicast is reliable
  (`70/70`) and broadcast is effectively reliable (`699/700`) under the same
  2 ms batch step.
- Guard 0/1 ms and zero-delay 0/1 ms do not materially change reliability after
  activation synchronization.
- The paper-facing SGCP control protocol should use compact broadcast/groupcast
  summaries for control metadata. Raw LiDAR grid payloads remain scheduled
  unicast transmissions.
- For the 2026-07-29 dense-LiDAR experiment series, use `guard=1 ms`,
  `zero-time send delay=0 ms`, and explicit pre-send activation synchronization.
