# PCS Repeated-Round Communication Time Report

Date: 2026-07-21

Terminology note: this report uses **round** for repeated PCS scheduling inside
one perception frame. It does not use "multi-slot" because the project treats a
frame/cycle as the 100 ms unit.

## Implementation

- PCS raw-LiDAR adaptation default is `div4/radius4/min128`.
- New offline arguments:
  - `--pcs-frame-rounds N`
  - `--pcs-frame-deadline-ms M`
- Repeated PCS scheduling excludes receiver grids accepted by previous rounds.
- If a tentative PCS round exceeds the remaining deadline, deadline admission
  trims selected grids per link so that all parallel links fit within the
  remaining time.
- Trace CSV now includes:
  - `frame_comm_time_ms`
  - `pcs_rounds_requested`
  - `pcs_rounds_accepted`
  - `pcs_round_comm_time_ms_json`
  - `pcs_round_comm_bytes_json`

## PCS Results

Command:

```powershell
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference `
  --dataset-root D:\Data\Carla `
  --scenario-id 2026_07_15_01_26_56 `
  --ego-cav-id 1 `
  --max-frames 11 `
  --fusion-method early `
  --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml `
  --sgcp-constrained `
  --resource-allocation fullperception_pcs `
  --sgcp-receiver-policy all-scheduled-receivers `
  --clustering singleton `
  --num-channels 10 `
  --bandwidth-mhz 20 `
  --pcs-frame-rounds 6 `
  --pcs-frame-deadline-ms 60
```

11-frame result:

| Method | Frames | Receiver samples | AP@0.3 | AP@0.5 | AP@0.7 | Raw bytes | Generated Mbps | Avg / max comm time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCS div4/radius4/min128, repeated-round deadline admission | 11 | 70 | 0.22 | 0.17 | 0.07 | 1,648,368 | 11.99 | 60.00 / 60.00 ms |

For frame `000060`, the no-deadline single PCS round would take about
`245.696 ms` under the same offline resource model. Therefore the 60 ms result
is not a free expansion of PCS; it is a deadline-admitted/cropped version of the
first PCS round. In the 11-frame run, every frame accepted one cropped round,
because the first round already filled the 60 ms budget.

Artifacts:

- `pcs_div4_r6_d60_11f_trace.csv`
- `pcs_div4_r6_d60_11f_eval_stats.csv`
- `pcs_div4_r1_nodeadline_1f_trace.csv`
- `communication_time_summary.csv`

## SGCP and EdgeCooper Timing Context

The table below separates two notions:

- **Offline exact-payload time** estimates how long the trace payload would take
  using the current offline resource model and the exact raw-LiDAR bytes in the
  trace.
- **NS3 request replay delay** is the measured application callback delay from
  prior scheduled-request replay artifacts. Those NS3 replays used 10,000-byte
  request payloads, so they validate subchannel placement, conflicts, and
  delivery timing for scheduled requests, but they are not exact replay of every
  raw-LiDAR byte in the AP table.

| Method / trace | Raw Mbps | Offline exact-payload time | NS3 request replay |
|---|---:|---:|---|
| SGCP-PAPG attentive, 41 frames | 62.54 | avg 320.38 ms, max 323.84 ms | 110/110 application + RLC complete, avg/p95 callback 23.91/24.00 ms |
| EdgeCooperHD scaffold proxy, 41 frames | 65.40 | avg 327.02 ms, max 388.54 ms | 110/110 application + RLC complete, avg/p95 callback 23.91/24.00 ms |
| EdgeCooper V2V protocol adaptation, 41 frames | 282.20 | avg 1411.02 ms, max 1508.43 ms | no exact-payload NS3 admission result; demand is far beyond 60/100 ms under this model |

Interpretation:

- The protocol-native EdgeCooper V2V adaptation is not a feasible one-frame
  raw-LiDAR protocol at its current generated demand; it should remain a
  protocol-adaptation diagnostic unless an upload-once edge model or stronger
  admission control is implemented.
- SGCP and EdgeCooperHD have existing NS3 evidence that scheduled subchannel
  requests complete reliably, but their AP-table raw payloads still require
  either a calibrated PHY throughput model or exact-payload NS3 replay before
  claiming full-byte transmission completes in 60 ms.
- PCS can be forced under 60 ms by deadline admission, but AP remains weak
  (`0.22/0.17/0.07` on 11 frames), supporting the earlier conclusion that PCS
  blind-spot scheduling is poorly aligned with raw-LiDAR detector utility in
  this scenario.

