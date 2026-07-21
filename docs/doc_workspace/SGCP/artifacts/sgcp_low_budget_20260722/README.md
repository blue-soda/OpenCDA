# SGCP Low-Budget Operating Point - 2026-07-22

Purpose: test whether SGCP can be reported at a payload level comparable to the deadline-constrained EdgeCooper V2V row without changing the paper-facing NS3 channel setting.

## Protocol

- Dataset: `D:\Data\Carla\2026_07_15_01_26_56`
- Frames: `000060` to `000140`, 41 frames
- Detector: attentive checkpoint config `docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/enable_coperception_early_from_attentive.yaml`
- Clustering: `coalition_game`
- Scheduler: `perception_aware_potential_game`
- Receiver policy: `all-cluster-heads`
- Late fusion: inter-cluster box-level NMS
- Channel setting: `40 MHz`, `10` OpenCDA target subchannels, `60 ms` communication deadline
- NS3 estimator: `tb_size=899 B`, `slot=0.5 ms`, `subchannel_prbs=10`, `symbols_per_slot=12`, `mcs=28`
- Low-budget knob: `--max-upload-points-per-source 4000`

The low-budget knob is a deterministic per-source point cap applied after SGCP's cluster, sender, subchannel and grid decisions. It should be described as a lower-payload operating point, not as a different scheduler.

## Results

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP-PAPG low-budget, cap=4000 | 0.86 | 0.77 | 0.33 | 51.20 | 0.70 | 51.90 |

For comparison, the current deadline-constrained EdgeCooper V2V row is `0.32/0.26/0.10` at `50.91 Mbps` raw LiDAR payload with `68/68` NS3 callbacks. The SGCP low-budget point has nearly the same communication level while retaining much higher aggregate AP in this dump.

## NS3 Frame `000060`

The upload plan is generated from the 41-frame trace, preserving the cap-adjusted bytes and the scheduler's subchannel assignment.

| CAM chunks | Bytes | Application callbacks | Avg delay | P95 delay | Max delay | PHY failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 70 | 640,000 | 70/70 | 23.714 ms | 46.000 ms | 46.000 ms | 0 |

This satisfies the 60 ms communication window under the paper-facing NS3 setting.

## Files

- `sgcp_papg_40mhz_10ch_bh2_cap4000_41f_trace.csv`: per-source perception/communication trace.
- `sgcp_papg_40mhz_10ch_bh2_cap4000_41f_eval.csv`: pooled AP evaluator inputs.
- `build_lowbudget_upload_plan.py`: converts the frame `000060` trace into NS3 chunked upload plan.
- `ns3_frame000060/upload_plan.csv`: generated chunk plan.
- `ns3_frame000060/ns3_stdout.log`: NS3 replay stdout.
- `ns3_frame000060/eval/`: application callback, RLC and PHY diagnostics.
- `summary.csv`: compact result row for paper table/figure ingestion.
