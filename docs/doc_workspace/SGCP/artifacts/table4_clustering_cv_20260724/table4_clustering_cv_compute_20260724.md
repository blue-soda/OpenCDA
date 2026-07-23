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
| sgcp_cv |  |  |  | 60.181354 | 6.000 | 2.667 | 76316.0 | 108.024 | 0.814286 | 536.470508 | 536.917608 |
| random_balanced |  |  |  | 55.077245 | 5.000 | 2.995 | 67505.9 | 110.244 | 0.715911 | 447.058757 | 447.468679 |
| distance_greedy |  |  |  | 56.323465 | 5.000 | 3.000 | 68486.7 | 86.439 | 0.725280 | 447.058757 | 447.478048 |
| density_greedy |  |  |  | 46.254704 | 5.000 | 3.000 | 60645.3 | 113.122 | 0.650380 | 447.058757 | 447.403148 |
| seac_social |  |  |  | 56.301487 | 5.000 | 3.000 | 68518.5 | 94.171 | 0.725583 | 447.058757 | 447.478351 |
| hho_vanet |  |  |  | 54.595840 | 5.000 | 3.000 | 67140.3 | 102.488 | 0.712420 | 447.058757 | 447.465188 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
