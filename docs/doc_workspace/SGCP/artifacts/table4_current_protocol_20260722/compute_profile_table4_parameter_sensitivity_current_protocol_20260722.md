# SGCP Profiled GFLOPs Summary

This file summarizes detector-side compute from offline trace CSVs. GFLOPs are calibrated from one real OpenCOOD forward when a calibration JSON is provided; otherwise the table still reports forward-equivalent compute and input-size diagnostics.

## Calibration

- fusion_method: `early`
- scenario/timestamp: `2026_07_15_01_26_56/000060`
- CAVs in calibration forward: `1`
- input points: `4918`
- profiled detector GFLOPs/forward: `89.411751`
- FLOP policy: Conv2d/ConvTranspose2d/Linear/BatchNorm/ReLU hooks plus PillarVFE elementwise estimate; multiply-add=2 FLOPs; voxelization/hash/scatter memory ops excluded

## Compute Table

| label | ap_03 | ap_05 | ap_07 | total_mbps | detector_calls_per_frame | mean_source_cavs_per_call | mean_input_points_per_frame | mean_pred_boxes_per_frame | input_adjusted_point_feature_gflops_per_frame | profiled_detector_gflops_per_frame | input_adjusted_detector_gflops_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rho_th_1 |  |  |  | 37.356800 | 6.000 | 2.000 | 58483.0 | 53.024 | 0.643948 | 536.470508 | 536.747270 |
| rho_th_2 |  |  |  | 37.311688 | 6.000 | 2.000 | 58449.1 | 52.366 | 0.643625 | 536.470508 | 536.746947 |
| rho_th_3 |  |  |  | 36.688172 | 6.000 | 2.000 | 57962.0 | 51.463 | 0.638972 | 536.470508 | 536.742294 |
| N_max_2 |  |  |  | 59.525463 | 10.098 | 1.961 | 95926.7 | 87.195 | 1.059878 | 902.840612 | 903.282541 |
| N_max_3 |  |  |  | 39.066568 | 7.000 | 1.913 | 64837.9 | 60.732 | 0.718872 | 625.882260 | 626.172747 |
| N_max_4 |  |  |  | 36.688172 | 6.000 | 2.000 | 57962.0 | 51.463 | 0.638972 | 536.470508 | 536.742294 |
| N_max_5 |  |  |  | 36.693791 | 6.000 | 2.000 | 57980.7 | 50.244 | 0.639151 | 536.470508 | 536.742473 |
| N_max_6 |  |  |  | 36.693791 | 6.000 | 2.000 | 57980.7 | 50.244 | 0.639151 | 536.470508 | 536.742473 |
| channels_5 |  |  |  | 30.507582 | 6.000 | 1.833 | 53133.4 | 48.610 | 0.592851 | 536.470508 | 536.696173 |
| channels_10 |  |  |  | 36.688172 | 6.000 | 2.000 | 57962.0 | 51.463 | 0.638972 | 536.470508 | 536.742294 |
| channels_20 |  |  |  | 36.688172 | 6.000 | 2.000 | 57962.0 | 51.463 | 0.638972 | 536.470508 | 536.742294 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
