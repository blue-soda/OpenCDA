# SGCP Profiled GFLOPs Summary

This file summarizes detector-side compute from offline trace CSVs. GFLOPs are calibrated from one real OpenCOOD forward when a calibration JSON is provided; otherwise the table still reports forward-equivalent compute and input-size diagnostics.

## Calibration

- fusion_method: `early`
- scenario/timestamp: `2026_07_29_02_32_08/000060`
- CAVs in calibration forward: `1`
- input points: `13948`
- profiled detector GFLOPs/forward: `89.437098`
- FLOP policy: Conv2d/ConvTranspose2d/Linear/BatchNorm/ReLU hooks plus PillarVFE elementwise estimate; multiply-add=2 FLOPs; voxelization/hash/scatter memory ops excluded

## Compute Table

| label | ap_03 | ap_05 | ap_07 | total_mbps | detector_calls_per_frame | mean_source_cavs_per_call | mean_input_points_per_frame | mean_pred_boxes_per_frame | input_adjusted_point_feature_gflops_per_frame | profiled_detector_gflops_per_frame | input_adjusted_detector_gflops_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pure_late | 0.81 | 0.73 | 0.52 | 1.950033 | 19.512 | 1.000 | 270566.8 | 289.707 | 1.681954 | 1745.114112 | 1745.107390 |
| pcs_global | 0.81 | 0.74 | 0.53 | 72.697350 | 19.561 | 1.506 | 326143.2 | 329.634 | 1.918367 | 1749.476897 | 1749.702366 |
| edgecooper_global | 0.81 | 0.74 | 0.53 | 88.323028 | 19.512 | 1.510 | 337991.7 | 300.293 | 1.967138 | 1745.114112 | 1745.392574 |
| sgcp | 0.84 | 0.78 | 0.56 | 60.372980 | 6.780 | 2.421 | 140679.9 | 118.878 | 0.781826 | 606.427154 | 606.622165 |
| centralized | 0.90 | 0.90 | 0.80 | 337.076948 | 1.000 | 20.000 | 277306.6 | 64.610 | 1.200461 | 89.437098 | 90.551014 |
| head_only | 0.80 | 0.72 | 0.51 | 0.674310 | 6.780 | 1.000 | 94134.4 | 100.171 | 0.584954 | 606.427154 | 606.425293 |
| pure_late_scaffold | 0.81 | 0.73 | 0.52 | 1.950033 | 19.512 | 1.000 | 270566.8 | 289.707 | 1.681954 | 1745.114112 | 1745.107390 |
| one_cluster_early | 0.90 | 0.90 | 0.80 | 337.076948 | 1.000 | 20.000 | 277306.6 | 64.610 | 1.200461 | 89.437098 | 90.551014 |
| clustered_early | 0.57 | 0.52 | 0.37 | 41.020940 | 6.780 | 2.000 | 126182.0 | 144.683 | 0.720505 | 606.427154 | 606.560844 |
| one_cluster_early_late | 0.90 | 0.90 | 0.80 | 337.076948 | 1.000 | 20.000 | 277306.6 | 64.610 | 1.200461 | 89.437098 | 90.551014 |
| full_sgcp | 0.84 | 0.78 | 0.56 | 60.372980 | 6.780 | 2.421 | 140679.9 | 118.878 | 0.781826 | 606.427154 | 606.622165 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
