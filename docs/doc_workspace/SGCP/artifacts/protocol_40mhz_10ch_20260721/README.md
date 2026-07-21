# Protocol-Native Baselines under 40 MHz / 10 Target Subchannels

Date: 2026-07-21

Paper-facing network setting:

- Configured sidelink bandwidth: `40 MHz`
- OpenCDA-visible target subchannels: `10`
- Perception cycle: `100 ms`
- Communication deadline target: `60 ms`
- NS3 parameters: `--slBandwidthIn100kHz=400 --targetSubchannels=10 --slSubchannelSize=10 --slMcs=28 --slSymbolsPerSlot=12`
- OpenCDA channel estimator: `--channel-estimator ns3 --ns3-tb-size-bytes 899 --ns3-slot-duration-ms 0.5 --ns3-subchannel-prbs 10 --ns3-symbols-per-slot 12 --ns3-mcs 28`

All AP results use the attentive detector config
`docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/enable_coperception_early_from_attentive.yaml`.

## Main Results

| Method | Protocol | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Offline frame time mean/max (ms) | NS3 callback delivery | NS3 delay mean/max (ms) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FullPerception-PCS | singleton receivers, no late fusion, one PCS round | 0.23 | 0.17 | 0.06 | 53.55 | 43.93 / 44.35 | 77/77 | 25.71 / 54.00 | Deliverable within the 60ms communication deadline. |
| EdgeCooper V2V | singleton receivers, no late fusion, global V2V sender selection | 0.54 | 0.48 | 0.25 | 275.94 | 9.59 / 12.92 | 15/348 | 127.87 / 215.00 | Deadline-infeasible diagnostic; superseded for paper-facing comparison by `../edgecooper_deadline_constrained_20260721/`. |

## PCS Multi-Round Diagnostic

PCS single-round communication was below 60ms, so repeated in-frame rounds were
tested with `--pcs-frame-rounds 6 --pcs-frame-deadline-ms 60`.

- 41-frame offline AP: `0.22 / 0.17 / 0.06`
- Raw payload: `65.77 Mbps`
- Offline admitted frame time: `59.996 ms`
- Simultaneous NS3 replay: `60/97` callbacks, max delay `214 ms`
- Sequential in-frame NS3 replay: `67/97` callbacks, frame-start completion max `242 ms`

Conclusion: repeated-round PCS can fill the offline estimator budget, but it is
not reliably deliverable under the current NS3 application/RLC timing. The
paper-facing PCS row should therefore use the single-round deliverable result
unless the PCS packet scheduler is redesigned.

## Files

- `pcs_single_41f.log`, `pcs_single_41f_trace.csv`, `pcs_single_41f_eval_stats.csv`
- `edgecooper_global_d60_41f.log`, `edgecooper_global_d60_41f_trace.csv`, `edgecooper_global_d60_41f_eval_stats.csv`
- `pcs_rounds6_d60_41f.log`, `pcs_rounds6_d60_41f_trace.csv`, `pcs_rounds6_d60_41f_eval_stats.csv`
- `ns3_plans/*.csv`: generated 10KB chunked upload plans
- `pcs_first_round_ns3_40mhz/`: deliverable PCS single-round NS3 replay
- `edgecooper_ns3_40mhz/`: EdgeCooper global concurrent replay
- `pcs_sequential_ns3_40mhz/`: PCS repeated-round sequential replay diagnostic
