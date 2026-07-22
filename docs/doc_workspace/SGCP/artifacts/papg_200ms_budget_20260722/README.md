# SGCP PAPG 200 ms Budget Probe

Purpose: run SGCP-PAPG with a 200 ms scheduling budget after the deadline
propagation fix, report AP/communication, and verify the resulting frame
`000060` upload plan in NS3.

## Protocol

- Dataset: `D:\Data\Carla\2026_07_15_01_26_56`
- Frames: 41 (`000060` to `000140`)
- Detector: attentive-derived early checkpoint YAML
- Resource allocation: `perception_aware_potential_game`
- Clustering: `coalition_game`
- Receiver policy: `all-cluster-heads`
- Inter-cluster late fusion: enabled
- `N_max=4`, `rho_th=3`, `head_rb_budget=2`
- Channel estimator: `ns3`
- NS3 estimator: `40 MHz`, `10` target subchannels, `tb_size=899 bytes`,
  `slot=0.5 ms`, `MCS=28`, `12` PSSCH symbols
- Communication deadline used by PAPG grid admission: `200 ms`

## Offline AP and Communication

| Deadline | AP@0.3 | AP@0.5 | AP@0.7 | raw Mbps | late-box Mbps | total Mbps | avg selected grids | estimated frame time mean / P95 / max |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 200 ms | 0.87 | 0.81 | 0.36 | 62.54 | 0.71 | 63.25 | 97.22 | 43.47 / 44.67 / 45.03 ms |

Compared with the corrected 100 ms run, the 200 ms budget mainly raises
per-link grid admission (`61.67` to `97.22` avg selected grids) and restores the
previous high AP@0.5 operating point. Payload rises only slightly because the
selected raw points are ultimately bounded by available points in the chosen
grid regions and the same `head_rb_budget=2` link budget.

## Frame `000060` NS3 Replay

The upload plan was generated from
`papg_attentive_nmax4_bh2_ns3_200ms_trace.csv` with OpenCDA-compatible
`10,000 bytes` CAM chunking and the trace's scheduled subchannels.

- Source-to-head links: `10`
- CAM chunks: `82`
- Payload: `783,392 bytes`

Measured result under NS3:

| Planned chunks | Payload bytes | App callbacks | RLC complete requests | PHY PSSCH OK | PHY failures | Callback delay mean / P95 / max |
|---:|---:|---:|---:|---:|---:|---|
| 82 | 783,392 | 82/82 | 81/82 | 881/881 | 0 | 27.18 / 54.00 / 55.00 ms |

RLC note: the only request not marked RLC-complete is the final `48 bytes` tail
chunk. It has an application callback and no RLC drop or PHY failure, but NS3
did not emit matching RLC segment events for that tiny tail. The application
delivery result is therefore complete (`82/82`), with no observed PHY failure.

## Artifacts

- `papg_attentive_nmax4_bh2_ns3_200ms_trace.csv`: 41-frame SGCP trace.
- `papg_attentive_nmax4_bh2_ns3_200ms_eval_stats.csv`: evaluator stats.
- `build_upload_plan_from_trace.py`: deterministic trace-to-NS3-plan helper.
- `ns3_frame000060/upload_plan.csv`: exact chunked replay plan.
- `ns3_frame000060/eval/`: NS3 delivery/RLC/PHY summaries.
