# EdgeCooper V2V Deadline-Constrained Protocol Adaptation

Date: 2026-07-21

This artifact replaces the infeasible EdgeCooper V2V protocol-adaptation row
that admitted each singleton receiver independently. The corrected admission
uses one shared per-frame V2V communication window:

- Configured sidelink bandwidth: `40 MHz`
- OpenCDA-visible target subchannels: `10`
- Perception cycle: `100 ms`
- Communication deadline: `60 ms`
- NS3 parameters: `--slBandwidthIn100kHz=400 --targetSubchannels=10 --slSubchannelSize=10 --slMcs=28 --slSymbolsPerSlot=12`
- OpenCDA estimator: `--channel-estimator ns3 --ns3-tb-size-bytes 899 --ns3-slot-duration-ms 0.5 --ns3-subchannel-prbs 10 --ns3-symbols-per-slot 12 --ns3-mcs 28`

## Admission Rule

The previous `--selective-frame-deadline-ms 60` implementation trimmed each
receiver independently, so 20 singleton receivers could consume 20 copies of
the 60 ms frame budget. The corrected rule is frame-level:

1. Generate EdgeCooper-style blind-spot/grid-priority candidate uploads for
   all singleton receivers.
2. Select a high-priority V2V matching: at most one active link per CAV endpoint
   and at most 10 active links, matching the 10 target subchannels.
3. Admit selected grids under one shared 60 ms NS3-calibrated byte budget.
4. Receivers without an admitted upload fall back to local-only inference.

This keeps the protocol-native setting: no SGCP coalition clustering and no
inter-cluster/global late fusion.

## Results

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | 41-frame payload bytes | NS3 callbacks | NS3 delay mean/max (ms) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EdgeCooper V2V, old per-receiver admission | 0.54 | 0.48 | 0.25 | 275.94 | 141,417,808 | 15/348 | 127.87 / 215.00 | Deadline-infeasible; kept only as diagnostic. |
| EdgeCooper V2V, global byte budget only | 0.32 | 0.26 | 0.11 | 86.30 | 44,230,800 | 22/132 | 139.14 / 244.00 | Payload-capped but still conflicted/overloaded in NS3. |
| EdgeCooper V2V, deadline-constrained matching | 0.32 | 0.26 | 0.10 | 50.91 | 26,091,536 | 68/68 | 25.90 / 54.00 | Paper-facing constrained EdgeCooper row. |

The constrained result should be used in protocol-native baseline comparisons
whenever the experiment requires the 60 ms communication deadline.

## Files

- `edgecooper_matching_d60_41f.log`: 41-frame AP run.
- `edgecooper_matching_d60_41f_trace.csv`: per-receiver trace.
- `edgecooper_matching_d60_41f_eval_stats.csv`: per-sample eval stats.
- `ns3_plans_matching/edgecooper_upload_plan.csv`: frame `000060` upload plan.
- `edgecooper_matching_ns3_40mhz/`: NS3 replay logs and parsed delivery CSVs.
- `edgecooper_global_frame_budget_d60_41f*`: byte-budget-only diagnostic.
- `edgecooper_ns3_40mhz_frame_budget/`: byte-budget-only NS3 diagnostic.
