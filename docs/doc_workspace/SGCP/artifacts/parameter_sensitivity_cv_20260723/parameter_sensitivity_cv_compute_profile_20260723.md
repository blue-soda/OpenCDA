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
| rho0p01 |  |  |  | 32.168929 | 6.293 | 2.589 | 55900.8 | 84.293 | 0.623447 | 562.639801 | 562.878150 |
| rho0p03 |  |  |  | 33.057436 | 6.390 | 2.565 | 57069.4 | 88.341 | 0.635997 | 571.362899 | 571.607827 |
| rho0p05 |  |  |  | 34.043380 | 6.488 | 2.541 | 58317.7 | 91.927 | 0.649307 | 580.085997 | 580.338265 |
| rho0p1 |  |  |  | 38.526002 | 6.488 | 2.541 | 61823.6 | 97.805 | 0.682795 | 580.085997 | 580.371753 |
| rho0p3 |  |  |  | 53.171949 | 6.122 | 2.633 | 71463.7 | 97.732 | 0.769671 | 547.374381 | 547.769402 |
| rho0p5 |  |  |  | 54.693120 | 6.122 | 2.633 | 72644.7 | 100.756 | 0.780952 | 547.374381 | 547.780683 |
| rho1 |  |  |  | 57.154716 | 6.000 | 2.667 | 73950.1 | 105.561 | 0.791687 | 536.470508 | 536.895008 |
| rho2 |  |  |  | 59.647500 | 6.000 | 2.667 | 75899.0 | 107.878 | 0.810302 | 536.470508 | 536.913624 |
| base |  |  |  | 60.181354 | 6.000 | 2.667 | 76316.0 | 108.024 | 0.814286 | 536.470508 | 536.917608 |
| nmax2 |  |  |  | 52.877050 | 10.098 | 1.961 | 90732.6 | 142.756 | 1.010266 | 902.840612 | 903.232929 |
| nmax3 |  |  |  | 60.340667 | 7.000 | 2.429 | 81458.3 | 120.098 | 0.877626 | 625.882260 | 626.331501 |
| nmax5 |  |  |  | 57.114037 | 6.000 | 2.667 | 73934.0 | 104.317 | 0.791533 | 536.470508 | 536.894855 |
| nmax6 |  |  |  | 57.114037 | 6.000 | 2.667 | 73934.0 | 104.293 | 0.791533 | 536.470508 | 536.894855 |
| ch5 |  |  |  | 30.196823 | 6.000 | 1.833 | 52890.6 | 84.390 | 0.590532 | 536.470508 | 536.693854 |
| ch20 |  |  |  | 65.614829 | 6.000 | 2.813 | 80561.0 | 108.878 | 0.854832 | 536.470508 | 536.958154 |
| budget40 |  |  |  | 52.957752 | 6.000 | 2.667 | 70672.6 | 97.049 | 0.760381 | 536.470508 | 536.863703 |
| budget60 |  |  |  | 56.761819 | 6.000 | 2.667 | 73644.5 | 101.098 | 0.788768 | 536.470508 | 536.892090 |
| budget100 |  |  |  | 59.482006 | 6.000 | 2.667 | 75769.7 | 105.220 | 0.809067 | 536.470508 | 536.912389 |
| budget300 |  |  |  | 60.181354 | 6.000 | 2.667 | 76316.0 | 108.049 | 0.814286 | 536.470508 | 536.917608 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
