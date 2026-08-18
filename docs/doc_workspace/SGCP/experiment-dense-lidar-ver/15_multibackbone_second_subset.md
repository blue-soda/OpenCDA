# SECOND Attentive-to-Early Diagnostic Probe

Status: diagnostic only, not the paper-facing main table. This document records
the SECOND attentive-to-early AP/Mbps/call-count probe on both dense scenes
under the previous `dynamic_cv` SGCP scheduler, with a Town06 SGCP hybrid
full-frame-GT addendum appended below.
GFLOPs are reported for SECOND Pure late and SGCP hybrid rows using
SECOND-specific singleton/full20 calibration JSONs. The remaining older
`dynamic_cv` baseline rows remain AP/Mbps/call-count diagnostics and do not
include GFLOPs.

Protocol: same offline Table 1 protocol as the dense package: 40 MHz / 10 target
subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`,
`mcs=28`, 60 ms data-plane deadline, and box-level NMS for late aggregation.
SGCP uses potential-verified clustering, dynamic C/V scheduling, density cap
`rho=2`, and raw-LiDAR block upload. Detector checkpoint is
`second_attentive_fusion.zip` migrated into the SECOND early-fusion model
definition, so all rows use the same raw-LiDAR early-fusion detector protocol.

## Dense Roundabout

Dataset: `D:\Data\Carla\2026_07_29_02_32_08`, 20 CAVs, 41 frames
(`000060-000140`).

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 0.96 | 0.96 | 0.92 | 337.08 | 0.00 | 337.08 | - | 1.00 |
| No collaboration | 0.56 | 0.52 | 0.37 | 0.00 | 0.00 | 0.00 | - | 20.00 |
| Pure late | 0.88 | 0.86 | 0.73 | 0.00 | 2.05 | 2.05 | 3127.11 | 20.00 |
| FullPerception-PCS | 0.55 | 0.51 | 0.37 | 70.49 | 0.00 | 70.49 | - | 20.00 |
| EdgeCooper-Pmax V2V adaptation | 0.53 | 0.50 | 0.39 | 86.30 | 0.00 | 86.30 | - | 20.00 |
| PACP-LiDAR V2V adaptation | 0.52 | 0.48 | 0.34 | 86.30 | 0.00 | 86.30 | - | 20.00 |
| SGCP | 0.92 | 0.90 | 0.78 | 27.84 | 0.92 | 28.75 | - | 6.63 |

### Dense Roundabout SGCP Hybrid Addendum

Protocol difference from the table above: the hybrid rerun explicitly uses
`--gt-scope full-frame`, `hybrid_round_robin_dynamic_marginal`, `rho_th=2`,
`upload_density_cap_rho=2`, and `N_max=5`. GFLOPs are computed with the SECOND
attentive-to-early singleton/full20 calibration JSONs.

| Method | Backbone | GT scope | N_max | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame | P95 data time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP hybrid | SECOND attentive-to-early | full-frame | 5 | 0.92 | 0.89 | 0.78 | 28.42 | 0.94 | 29.36 | 1068.30 | 6.63 | 37.45 ms |

## Town06

Dataset: `D:\Data\Carla\2026_07_31_02_24_35`, 10 CAVs, 20 background vehicles,
31 frames.

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 0.99 | 0.99 | 0.98 | 147.23 | 0.00 | 147.23 | - | 1.00 |
| No collaboration | 0.70 | 0.66 | 0.52 | 0.00 | 0.00 | 0.00 | - | 10.00 |
| Pure late | 0.92 | 0.91 | 0.86 | 0.00 | 0.81 | 0.81 | 1600.20 | 10.00 |
| FullPerception-PCS | 0.68 | 0.65 | 0.53 | 36.35 | 0.00 | 36.35 | - | 10.00 |
| EdgeCooper-Pmax V2V adaptation | 0.79 | 0.77 | 0.69 | 69.14 | 0.00 | 69.14 | - | 10.00 |
| PACP-LiDAR V2V adaptation | 0.71 | 0.68 | 0.55 | 81.15 | 0.00 | 81.15 | - | 10.00 |
| SGCP | 0.95 | 0.94 | 0.87 | 15.42 | 0.51 | 15.93 | - | 5.26 |

## Town06 SGCP Hybrid Addendum

Protocol difference from the Town06 table above: the hybrid rerun explicitly
uses `--gt-scope full-frame`, `hybrid_round_robin_dynamic_marginal`, and
`N_max=3`. GFLOPs are computed with the SECOND attentive-to-early
singleton/full20 calibration JSONs.

| Method | Backbone | GT scope | N_max | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame | P95 data time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP hybrid | SECOND attentive-to-early | full-frame | 3 | 0.96 | 0.96 | 0.87 | 18.35 | 0.41 | 18.76 | 643.39 | 4.00 | 32.93 ms |

Provenance: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\town06\second_att2early_sgcp_hybrid_trace.csv`.
GFLOPs provenance: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\town06\second_att2early_sgcp_hybrid_compute.csv`.
Pure-late GFLOPs provenance: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\second_late_compute\second_pure_late_compute.csv`.

Dense Roundabout hybrid provenance:
`C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\town03_second\second_att2early_sgcp_hybrid_trace.csv`.

Reading:

- SECOND attentive-to-early is a diagnostic additional-backbone AP/Mbps probe:
  SGCP exceeds pure late and all protocol-native baselines in both scenes and
  all AP thresholds, but this document should not be used as the GFLOPs source.
- SGCP uses far fewer detector calls than pure late and the all-CAV
  protocol-native baselines: `6.63` vs `20.00` calls/frame in the dense
  roundabout scene, and `5.26` vs `10.00` calls/frame in Town06.
- The centralized all-in-one row remains an upper reference, not a feasible
  protocol-native baseline, because it requires all raw LiDAR to be aggregated
  at one receiver.

Raw artifacts are in
`C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\second_att2early_dense_table1_full`
and
`C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\town06_second_att2early_table1_full`.
