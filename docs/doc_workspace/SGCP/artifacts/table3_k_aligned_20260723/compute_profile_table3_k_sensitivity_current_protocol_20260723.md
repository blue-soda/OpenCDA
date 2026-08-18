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
| PCS SGCP-scaffold K1 | 0.57 | 0.42 | 0.21 | 11.161943 | 6.000 | 1.366 | 37611.8 | 77.195 | 0.444592 | 536.470508 | 536.547914 |
| PCS SGCP-scaffold K2 | 0.59 | 0.44 | 0.21 | 12.293994 | 6.000 | 1.407 | 38496.6 | 77.122 | 0.453044 | 536.470508 | 536.556365 |
| EdgeCooperHD SGCP-scaffold K1 | 0.71 | 0.59 | 0.29 | 30.620909 | 6.000 | 1.935 | 52762.8 | 87.244 | 0.589311 | 536.470508 | 536.692633 |
| EdgeCooperHD SGCP-scaffold K2 | 0.81 | 0.70 | 0.32 | 52.064406 | 6.000 | 2.549 | 69425.7 | 105.098 | 0.748471 | 536.470508 | 536.851792 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
