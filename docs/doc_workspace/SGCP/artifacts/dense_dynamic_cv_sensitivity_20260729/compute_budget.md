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
| budget_rho1_mbps1 |  |  |  | 0.999680 | 6.780 | 1.155 | 94841.7 | 101.659 | 1.002339 | 606.255290 | 606.842678 |
| budget_rho1_mbps5 |  |  |  | 4.999680 | 6.780 | 1.590 | 97966.7 | 121.707 | 1.032188 | 606.255290 | 606.872527 |
| budget_rho1_mbps10 |  |  |  | 9.999360 | 6.805 | 2.011 | 102216.9 | 130.024 | 1.073132 | 608.436064 | 609.092753 |
| budget_rho1_mbps20 |  |  |  | 16.609967 | 6.780 | 2.475 | 107037.2 | 136.805 | 1.118827 | 606.255290 | 606.959166 |
| budget_rho1_mbps40 |  |  |  | 16.609967 | 6.780 | 2.475 | 107037.2 | 136.780 | 1.118827 | 606.255290 | 606.959166 |
| budget_rho1_mbps60 |  |  |  | 16.609967 | 6.780 | 2.475 | 107037.2 | 136.805 | 1.118827 | 606.255290 | 606.959166 |
| budget_rho1_mbps68 |  |  |  | 16.609967 | 6.780 | 2.475 | 107037.2 | 136.805 | 1.118827 | 606.255290 | 606.959166 |
| budget_rho1_mbps84 |  |  |  | 16.592109 | 6.805 | 2.470 | 107367.5 | 137.049 | 1.122329 | 608.436064 | 609.141950 |
| budget_rho2_mbps1 |  |  |  | 0.999555 | 6.634 | 1.283 | 92770.9 | 99.146 | 0.980477 | 593.170643 | 593.745126 |
| budget_rho2_mbps5 |  |  |  | 4.999618 | 6.634 | 1.518 | 95895.9 | 105.585 | 1.010327 | 593.170643 | 593.774975 |
| budget_rho2_mbps10 |  |  |  | 9.999298 | 6.634 | 1.801 | 99801.9 | 116.561 | 1.047636 | 593.170643 | 593.812285 |
| budget_rho2_mbps20 |  |  |  | 19.999376 | 6.634 | 2.151 | 107614.5 | 124.463 | 1.122260 | 593.170643 | 593.886908 |
| budget_rho2_mbps40 |  |  |  | 28.417374 | 6.634 | 2.507 | 114191.0 | 130.171 | 1.185078 | 593.170643 | 593.949726 |
| budget_rho2_mbps60 |  |  |  | 28.417374 | 6.634 | 2.507 | 114191.0 | 130.171 | 1.185078 | 593.170643 | 593.949726 |
| budget_rho2_mbps68 |  |  |  | 28.417374 | 6.634 | 2.507 | 114191.0 | 130.195 | 1.185078 | 593.170643 | 593.949726 |
| budget_rho2_mbps84 |  |  |  | 28.417374 | 6.634 | 2.507 | 114191.0 | 130.195 | 1.185078 | 593.170643 | 593.949726 |
| budget_rho5_mbps1 |  |  |  | 0.983883 | 6.634 | 1.151 | 92758.6 | 98.098 | 0.980360 | 593.170643 | 593.745009 |
| budget_rho5_mbps5 |  |  |  | 4.976765 | 6.634 | 1.283 | 95878.1 | 100.805 | 1.010157 | 593.170643 | 593.774805 |
| budget_rho5_mbps10 |  |  |  | 9.968890 | 6.634 | 1.478 | 99778.2 | 103.024 | 1.047409 | 593.170643 | 593.812058 |
| budget_rho5_mbps20 |  |  |  | 19.971216 | 6.634 | 1.809 | 107592.5 | 108.195 | 1.122050 | 593.170643 | 593.886698 |
| budget_rho5_mbps40 |  |  |  | 39.967469 | 6.634 | 2.386 | 123214.6 | 114.707 | 1.271268 | 593.170643 | 594.035916 |
| budget_rho5_mbps60 |  |  |  | 43.966220 | 6.634 | 2.507 | 126338.6 | 115.659 | 1.301108 | 593.170643 | 594.065756 |
| budget_rho5_mbps68 |  |  |  | 43.966220 | 6.634 | 2.507 | 126338.6 | 115.683 | 1.301108 | 593.170643 | 594.065756 |
| budget_rho5_mbps84 |  |  |  | 43.966220 | 6.634 | 2.507 | 126338.6 | 115.659 | 1.301108 | 593.170643 | 594.065756 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
