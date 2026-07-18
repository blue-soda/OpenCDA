# SGCP Appendix Support: Runtime, Control Overhead, and NS3 Reliability

Date: 2026-07-19

This note consolidates evidence that is useful for the paper appendix or rebuttal, but should not distract from the main aggregate-AP and Mbps figures. The source values come from `results.md`, `runtime_feasibility_revision.md`, `control_overhead.md`, and the PAPG NS3 replay artifacts.

## Recommended Paper Use

- Main text: one short sentence can state that SGCP's scheduled PAPG requests are fully delivered in the 11-frame NS3 replay, and that the Python control-plane prototype is near the 100 ms cycle.
- Appendix/rebuttal: include the runtime table, control metadata table, and request-level reliability table.
- Avoid overclaiming: this evidence supports near-real-time control-plane feasibility, not detector-inclusive end-to-end 100 ms closure.

## Runtime

Command:

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

Log:

```text
docs\doc_workspace\SGCP\artifacts\runtime_breakdown_41f\offline_replay_runtime.log
```

| Stage | Mean (ms) | Max (ms) | Online Cycle? | Interpretation |
| --- | ---: | ---: | --- | --- |
| Coalition formation | 64.39 | 82.32 | Yes | Main remaining optimization target. |
| PPS scheduling | 40.58 | 53.05 | Yes | Potential-guided scheduling cost. |
| SGCP algorithm total | 105.24 | 127.58 | Yes | Control-plane prototype only. |
| Dump frame loading | 448.40 | 513.31 | No | Offline replay artifact. |
| Offline world build | 151.33 | 199.34 | No/partial | Offline adapter artifact. |

Writing boundary: the current Python control plane is close to the 100 ms cooperation period, but the paper should call this near-real-time feasibility rather than a full end-to-end 100 ms guarantee. The topology-triggered update mechanism is important here because coalition formation need not be paid every sensing cycle.

## PPS Convergence

| Metric | Value |
| --- | ---: |
| Frames converged before `max_iter=20` | 41 / 41 |
| Avg. iterations | 3.00 |
| Max iterations | 3 |
| Avg. scheduled links / frame | 10.00 |
| Total scheduled links | 410 |
| Avg. selected grids / frame | 523.90 |

This supports an empirical finite-convergence claim for the implemented constrained best-response scheduler. It should not be phrased as a complete exact-potential proof unless the proof text is tightened separately.

## Control Metadata

| Component | Total Bytes | Avg. Bytes / Frame |
| --- | ---: | ---: |
| Beacon | 52,480 | 1,280.00 |
| Density metadata | 40,184 | 980.10 |
| Cluster control | 3,608 | 88.00 |
| PPS schedule | 90,840 | 2,215.61 |
| Total control metadata | 187,112 | 4,563.71 |

Against the current PAPG main payload of 32,049,872 bytes, this is about 0.58%. Against the older potential-game payload of 26,916,208 bytes, it is about 0.70%. The safest wording is therefore: control metadata is below 1% of raw perception payload in this 41-frame dump, and should be reported separately from point-cloud payload.

## NS3 Request-Level Reliability

PAPG artifact:

```text
docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\
```

| Scope | Scheduled Requests | Application Callback | RLC Complete | RLC TX/RX Events | Avg. / P95 Delay (ms) | PHY Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG 10ch, first 11 frames | 110 | 110 / 110 | 110 / 110 | 2,970 / 2,970 | 23.91 / 24.00 | 0 |

Stress result: when NS3 exposes only 5 subchannels, the 55 in-window requests complete and the 55 out-of-window requests are rejected at the bridge before CAM/RLC transmission. This verifies that OpenCDA-specified subchannels are actually enforced by NS3 and that out-of-band demand does not pollute legal transmissions.

## Appendix Pointers

- Runtime details: `docs/doc_workspace/SGCP/runtime_feasibility_revision.md`
- Control-overhead assumptions: `docs/doc_workspace/SGCP/control_overhead.md`
- Main result tables and NS3 replay summaries: `docs/doc_workspace/SGCP/results.md`
- Machine-readable summary: `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/runtime_control_ns3_summary.csv`
