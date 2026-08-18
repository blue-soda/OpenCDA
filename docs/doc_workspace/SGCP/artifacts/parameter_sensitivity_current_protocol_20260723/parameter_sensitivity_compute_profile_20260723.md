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
| rho0p01 |  |  |  | 62.405807 | 6.293 | 2.589 | 79523.3 | 101.195 | 0.849084 | 562.639801 | 563.103787 |
| rho0p03 |  |  |  | 62.437557 | 6.317 | 2.583 | 79665.5 | 103.244 | 0.850789 | 564.820576 | 565.284774 |
| rho0p05 |  |  |  | 62.456039 | 6.488 | 2.541 | 80515.0 | 104.512 | 0.861331 | 580.085997 | 580.550289 |
| rho0p1 |  |  |  | 62.455321 | 6.488 | 2.541 | 80518.4 | 105.829 | 0.861363 | 580.085997 | 580.550321 |
| rho0p3 |  |  |  | 62.591969 | 6.122 | 2.633 | 78823.0 | 108.366 | 0.839967 | 547.374381 | 547.839697 |
| rho0p5 |  |  |  | 62.604581 | 6.122 | 2.633 | 78825.5 | 109.171 | 0.839990 | 547.374381 | 547.839721 |
| rho1 |  |  |  | 62.569647 | 6.000 | 2.667 | 78180.5 | 109.732 | 0.832095 | 536.470508 | 536.935416 |
| rho2 |  |  |  | 62.535899 | 6.000 | 2.667 | 78155.5 | 110.829 | 0.831856 | 536.470508 | 536.935178 |
| base |  |  |  | 62.536336 | 6.000 | 2.667 | 78155.9 | 111.000 | 0.831860 | 536.470508 | 536.935181 |
| nmax2 |  |  |  | 60.538942 | 10.098 | 1.961 | 96718.4 | 148.902 | 1.067441 | 902.840612 | 903.290104 |
| nmax3 |  |  |  | 62.390072 | 7.000 | 2.429 | 83059.4 | 119.390 | 0.892919 | 625.882260 | 626.346794 |
| nmax5 |  |  |  | 62.624968 | 6.000 | 2.667 | 78239.4 | 108.805 | 0.832658 | 536.470508 | 536.935979 |
| nmax6 |  |  |  | 62.624968 | 6.000 | 2.667 | 78239.4 | 108.805 | 0.832658 | 536.470508 | 536.935979 |
| ch5 |  |  |  | 31.119922 | 6.000 | 1.833 | 53611.8 | 84.122 | 0.597420 | 536.470508 | 536.700742 |
| ch20 |  |  |  | 68.070400 | 6.000 | 2.813 | 82479.4 | 112.049 | 0.873156 | 536.470508 | 536.976478 |
| budget40 |  |  |  | 54.343056 | 6.000 | 2.667 | 71754.9 | 96.951 | 0.770719 | 536.470508 | 536.874040 |
| budget60 |  |  |  | 58.441897 | 6.000 | 2.667 | 74957.1 | 102.220 | 0.801306 | 536.470508 | 536.904627 |
| budget100 |  |  |  | 61.471938 | 6.000 | 2.667 | 77324.3 | 106.171 | 0.823917 | 536.470508 | 536.927238 |
| budget300 |  |  |  | 62.536336 | 6.000 | 2.667 | 78155.9 | 111.024 | 0.831860 | 536.470508 | 536.935181 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
