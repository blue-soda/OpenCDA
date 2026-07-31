# Multi-Backbone SECOND Probe

Status: targeted probe. SECOND early fusion was tested first. Because pure late
remained stronger in the dense scene, the SECOND attentive checkpoint was then
migrated into the early-fusion model definition and rerun with the same SGCP
protocol.

Protocol: same offline Table 1 protocol as the dense multi-backbone experiments:
40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`,
`slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, and box-level
NMS for late aggregation. SGCP uses the same dense setting as the current clean
package: potential-verified clustering, dynamic C/V scheduling, density cap
`rho=2`, and raw-LiDAR block upload.

## SECOND Early Fusion

Checkpoint: `second_early_fusion.zip`, converted for the local spconv weight
layout and patched to the local OpenCOOD dataset paths.

| Scene | Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense roundabout | Pure late | 0.97 | 0.96 | 0.82 | 0.00 | 1.72 | 1.72 | 2951.96 | 19.51 |
| Dense roundabout | SGCP | 0.93 | 0.91 | 0.80 | 27.84 | 0.84 | 28.67 | 1003.67 | 6.63 |
| Town06 | Pure late | 0.95 | 0.95 | 0.83 | 0.00 | 0.75 | 0.75 | 1512.88 | 10.00 |
| Town06 | SGCP | 0.98 | 0.97 | 0.81 | 15.42 | 0.49 | 15.91 | 795.48 | 5.26 |

Reading: SECOND early fusion already supports the SGCP tradeoff, but dense
pure late is still AP-stronger because it runs local detection at almost every
CAV.

## SECOND Attentive-to-Early

Checkpoint: `second_attentive_fusion.zip`, loaded into the SECOND early-fusion
model definition. Only backbone-compatible weights are used; the experiment
keeps the input/output protocol as raw-LiDAR early fusion.

| Scene | Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense roundabout | Pure late | 0.88 | 0.86 | 0.73 | 0.00 | 2.05 | 2.05 | 2951.96 | 19.51 |
| Dense roundabout | SGCP | 0.92 | 0.90 | 0.78 | 27.84 | 0.92 | 28.75 | 1003.67 | 6.63 |
| Town06 | Pure late | 0.92 | 0.91 | 0.86 | 0.00 | 0.81 | 0.81 | 1512.88 | 10.00 |
| Town06 | SGCP | 0.95 | 0.94 | 0.87 | 15.42 | 0.51 | 15.93 | 795.48 | 5.26 |

Reading:

- The attentive-to-early migration resolves the pure-late dominance seen with
  VoxelNet, PIXOR, and vanilla SECOND early fusion.
- SGCP is higher than pure late in both scenes and all AP thresholds while
  using about one third to one half of the detector compute.
- This is the recommended SECOND evidence if the paper needs a second backbone
  that supports the SGCP narrative.

Raw artifacts are under
`C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\`.
