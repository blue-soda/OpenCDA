# Backbone Weight-Migration Probe

Status: targeted diagnostic. The goal is to test whether stronger
intermediate/attentive checkpoints can be transplanted into an early-fusion
raw-LiDAR input protocol, so SGCP and pure late use the same detector forward
interface while benefiting from stronger pretrained weights.

Protocol: same offline Table 1 protocol as the dense multi-backbone experiments:
40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`,
`slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, and box-level
NMS for late aggregation. SGCP uses potential-verified clustering, dynamic C/V
scheduling, density cap `rho=2`, and raw-LiDAR block upload.

## Results

| Migration source | Scene | Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SECOND early | Dense roundabout | Pure late | 0.97 | 0.96 | 0.82 | 0.00 | 1.72 | 1.72 | 2951.96 | 19.51 |
| SECOND early | Dense roundabout | SGCP | 0.93 | 0.91 | 0.80 | 27.84 | 0.84 | 28.67 | 1003.67 | 6.63 |
| SECOND attentive-to-early | Dense roundabout | Pure late | 0.88 | 0.86 | 0.73 | 0.00 | 2.05 | 2.05 | 2951.96 | 19.51 |
| SECOND attentive-to-early | Dense roundabout | SGCP | 0.92 | 0.90 | 0.78 | 27.84 | 0.92 | 28.75 | 1003.67 | 6.63 |
| VoxelNet early | Dense roundabout | Pure late | 0.97 | 0.94 | 0.71 | 0.00 | 1.40 | 1.40 | 3188.50 | 20.00 |
| VoxelNet early | Dense roundabout | SGCP | 0.88 | 0.86 | 0.62 | 27.84 | 0.69 | 28.52 | 1057.65 | 6.63 |
| VoxelNet attentive-to-early | Dense roundabout | Pure late | 0.88 | 0.86 | 0.69 | 0.00 | 1.46 | 1.46 | 3188.50 | 20.00 |
| VoxelNet attentive-to-early | Dense roundabout | SGCP | 0.83 | 0.80 | 0.64 | 27.84 | 0.69 | 28.53 | 1057.65 | 6.63 |
| VoxelNet attentive-compression-to-early | Dense roundabout | Pure late | 0.81 | 0.67 | 0.15 | 0.00 | 1.13 | 1.13 | 3188.50 | 20.00 |
| VoxelNet attentive-compression-to-early | Dense roundabout | SGCP | 0.72 | 0.61 | 0.14 | 27.84 | 0.55 | 28.39 | 1057.65 | 6.63 |
| SECOND early | Town06 | Pure late | 0.95 | 0.95 | 0.83 | 0.00 | 0.75 | 0.75 | 1512.88 | 10.00 |
| SECOND early | Town06 | SGCP | 0.98 | 0.97 | 0.81 | 15.42 | 0.49 | 15.91 | 795.48 | 5.26 |
| SECOND attentive-to-early | Town06 | Pure late | 0.92 | 0.91 | 0.86 | 0.00 | 0.81 | 0.81 | 1512.88 | 10.00 |
| SECOND attentive-to-early | Town06 | SGCP | 0.95 | 0.94 | 0.87 | 15.42 | 0.51 | 15.93 | 795.48 | 5.26 |
| VoxelNet early | Town06 | Pure late | 0.97 | 0.97 | 0.88 | 0.00 | 0.59 | 0.59 | 1594.25 | 10.00 |
| VoxelNet early | Town06 | SGCP | 0.96 | 0.96 | 0.83 | 15.42 | 0.40 | 15.83 | 838.27 | 5.26 |
| VoxelNet attentive-to-early | Town06 | Pure late | 0.96 | 0.95 | 0.89 | 0.00 | 0.66 | 0.66 | 1594.25 | 10.00 |
| VoxelNet attentive-to-early | Town06 | SGCP | 0.97 | 0.95 | 0.86 | 15.42 | 0.43 | 15.85 | 838.27 | 5.26 |
| VoxelNet attentive-compression-to-early | Town06 | Pure late | 0.58 | 0.51 | 0.25 | 0.00 | 0.68 | 0.68 | 1594.25 | 10.00 |
| VoxelNet attentive-compression-to-early | Town06 | SGCP | 0.61 | 0.52 | 0.24 | 15.42 | 0.43 | 15.85 | 838.27 | 5.26 |

## Reading

- SECOND attentive-to-early is the only tested migration that cleanly supports
  SGCP AP dominance over pure late in both scenes and all AP thresholds.
- VoxelNet attentive-to-early reduces pure-late AP relative to vanilla VoxelNet,
  but it also reduces SGCP AP in the dense scene. It is useful as a diagnostic,
  not as the primary robustness result.
- VoxelNet attentive-compression-to-early should not be used in the paper. The
  checkpoint contains extra compression modules that are ignored by the
  early-fusion model; although it runs with `strict=False`, the resulting
  detector is poorly calibrated.
- PIXOR has no attentive/intermediate source checkpoint in the available local
  zip set, so only the existing PIXOR early-fusion probe is available.

Recommended paper-facing choice: keep PointPillar-attentive as the main
backbone and use SECOND attentive-to-early as the additional backbone robustness
evidence.
