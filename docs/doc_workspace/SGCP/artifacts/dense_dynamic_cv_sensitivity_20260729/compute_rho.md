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
| rho_sweep_rho0p1 |  |  |  | 2.073943 | 6.756 | 2.480 | 95516.6 | 113.610 | 1.008438 | 604.074516 | 604.669496 |
| rho_sweep_rho0p2 |  |  |  | 3.966814 | 6.927 | 2.444 | 99277.3 | 123.561 | 1.046788 | 619.339937 | 619.962818 |
| rho_sweep_rho0p5 |  |  |  | 9.273413 | 6.927 | 2.444 | 103412.8 | 133.366 | 1.086289 | 619.339937 | 620.002319 |
| rho_sweep_rho1 |  |  |  | 16.609967 | 6.780 | 2.475 | 107037.2 | 136.780 | 1.118827 | 606.255290 | 606.959166 |
| rho_sweep_rho2 |  |  |  | 28.417374 | 6.634 | 2.507 | 114191.0 | 130.195 | 1.185078 | 593.170643 | 593.949726 |
| rho_sweep_rho3 |  |  |  | 35.574260 | 6.634 | 2.507 | 119782.4 | 123.561 | 1.238485 | 593.170643 | 594.003133 |
| rho_sweep_rho4 |  |  |  | 40.103305 | 6.634 | 2.507 | 123320.7 | 117.415 | 1.272282 | 593.170643 | 594.036930 |
| rho_sweep_rho5 |  |  |  | 43.966220 | 6.634 | 2.507 | 126338.6 | 115.659 | 1.301108 | 593.170643 | 594.065756 |
| rho_sweep_rho10 |  |  |  | 49.573651 | 6.634 | 2.507 | 130719.4 | 107.659 | 1.342952 | 593.170643 | 594.107601 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
