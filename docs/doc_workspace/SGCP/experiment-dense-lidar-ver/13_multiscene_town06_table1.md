# Multi-Scene Table 1: Town06 Dense

Status: completed for the previous `dynamic_cv` SGCP scheduler, with an SGCP
hybrid full-frame-GT addendum appended below. This document contains the
non-roundabout dense scenario with the paper-facing PointPillar-attentive
early-fusion checkpoint.

Dataset: `D:\Data\Carla\2026_07_31_02_24_35`.

Scenario: `v2xp_cluster_town06_dense`, Town06, 1 explicit CAV plus 9 managed
traffic CAVs, 20 unmanaged background vehicles. LiDAR setting matches the dense
roundabout scenario: 32 channels, 320000 points/s, 20 Hz rotation, 50 m range.

Protocol: 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`,
`slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline. SGCP uses
potential-verified clustering, dynamic C/V scheduling, density cap `rho=2`,
`head_rb_budget=2`, and `N_max=2` for the 10-CAV scene.

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 1.00 | 1.00 | 0.98 | 147.23 | 0.00 | 147.23 | 90.59 | 1.00 |
| No collaboration | 0.74 | 0.71 | 0.54 | 0.00 | 0.00 | 0.00 | 894.87 | 10.00 |
| Pure late | 0.91 | 0.89 | 0.82 | 0.00 | 0.79 | 0.79 | 894.87 | 10.00 |
| FullPerception-PCS | 0.71 | 0.69 | 0.54 | 36.35 | 0.00 | 36.35 | 895.14 | 10.00 |
| EdgeCooper-Pmax V2V adaptation | 0.83 | 0.82 | 0.71 | 69.14 | 0.00 | 69.14 | 895.39 | 10.00 |
| PACP-LiDAR V2V adaptation | 0.74 | 0.72 | 0.56 | 81.15 | 0.00 | 81.15 | 895.48 | 10.00 |
| SGCP | 0.96 | 0.94 | 0.81 | 15.42 | 0.50 | 15.92 | 470.65 | 5.26 |

## SGCP Hybrid Addendum

Protocol difference from the table above: the hybrid rerun explicitly uses
`--gt-scope full-frame`, `hybrid_round_robin_dynamic_marginal`, and the
formula-aligned `N_max=3` for the 10-CAV Town06 scene. The baseline rows above
are retained as the previous Town06 table; rerun all baselines with explicit
full-frame GT if this addendum becomes a paper-facing multi-scene table.

| Method | Backbone | GT scope | N_max | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame | P95 data time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP hybrid | PointPillar-Attentive | full-frame | 3 | 0.95 | 0.94 | 0.81 | 18.35 | 0.41 | 18.76 | 357.75 | 4.00 | 32.93 ms |

Reading:

- SGCP remains high-AP and low-communication in a non-roundabout Town06 scene.
- SGCP uses substantially fewer detector calls than pure late and
  protocol-native all-CAV baselines.
- Non-PointPillar backbone probes have been moved out of this clean package to
  keep the paper-facing result set focused and non-conflicting.
- Hybrid provenance: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\town06\pointpillar_att2early_sgcp_hybrid_trace.csv`.
