# PIXOR SGCP vs Pure-Late Subset

Status: completed targeted probe requested after VoxelNet pure late appeared too
strong. This file only compares `SGCP` and `Pure late` on the two existing
dense scenes; it is not a full Table 1 rerun.

Protocol: same offline Table 1 protocol as the multi-backbone experiments:
40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`,
`slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, PIXOR
early-fusion checkpoint from `pixor_early_fusion.zip`, and box-level NMS for
late aggregation. SGCP uses the same clustering/scheduler parameters as the
corresponding PointPillar/VoxelNet scene (`N_max=5` for dense roundabout,
`N_max=2` for Town06).

| Scene | Backbone | Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Detector calls/frame |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense roundabout | PIXOR | Pure late | 0.86 | 0.78 | 0.45 | 0.00 | 1.64 | 1.64 | 2159.84 | 20.00 |
| Dense roundabout | PIXOR | SGCP | 0.85 | 0.76 | 0.41 | 27.84 | 0.77 | 28.61 | 716.44 | 6.63 |
| Town06 | PIXOR | Pure late | 0.98 | 0.96 | 0.73 | 0.00 | 0.73 | 0.73 | 1079.92 | 10.00 |
| Town06 | PIXOR | SGCP | 0.97 | 0.95 | 0.72 | 15.42 | 0.46 | 15.88 | 567.83 | 5.26 |

Reading:

- PIXOR does not remove the pure-late concern under an AP-only comparison:
  pure late remains slightly higher than SGCP in both scenes.
- The gap is much smaller than the VoxelNet dense-roundabout gap. Dense PIXOR
  SGCP is within `0.01/0.02/0.04` AP of pure late while using about one third
  of the detector compute.
- Town06 PIXOR SGCP is nearly tied with pure late (`-0.01/-0.01/-0.01` AP) and
  uses about half the detector compute.
- Communication remains a tradeoff: pure late uses only box sharing, while SGCP
  spends raw LiDAR payload to reduce detector calls and maintain clustered
  early-fusion capability. Therefore this probe should be used only if the
  paper explicitly discusses AP/compute/communication jointly.

Raw artifacts:

- Dense roundabout:
  `docs/doc_workspace/SGCP/artifacts/multi_backbone_20260731/pixor_dense_table1_subset/`
- Town06:
  `docs/doc_workspace/SGCP/artifacts/multi_backbone_20260731/town06_pixor_table1_subset/`
- Compute summary:
  `docs/doc_workspace/SGCP/artifacts/multi_backbone_20260731/pixor_compute_subset.csv`
