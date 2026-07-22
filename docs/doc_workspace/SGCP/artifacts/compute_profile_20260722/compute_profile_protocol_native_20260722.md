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
| Centralized all-in-one raw-LiDAR early fusion upper reference | 0.85 | 0.83 | 0.48 | 118.709323 | 1.000 | 20.000 | 97669.1 | 60.122 | 0.947134 | 89.411751 | 90.297687 |
| No collaboration | 0.23 | 0.17 | 0.06 | 0.000000 | 20.000 | 1.000 | 97669.1 | 259.927 | 1.217356 | 1788.235028 | 1788.228428 |
| Pure late | 0.82 | 0.76 | 0.37 | 0.739122 | 19.171 | 1.000 | 93608.5 | 102.415 | 1.166777 | 1714.088698 | 1714.082268 |
| FullPerception-PCS protocol adaptation | 0.23 | 0.17 | 0.06 | 53.550000 | 20.000 | 1.454 | 139506.2 | 299.927 | 1.616974 | 1788.235028 | 1788.628046 |
| EdgeCooper V2V deadline-constrained adaptation | 0.32 | 0.26 | 0.10 | 50.910000 | 20.000 | 1.423 | 137442.8 | 307.829 | 1.597265 | 1788.235028 | 1788.608337 |
| SGCP-PAPG | 0.87 | 0.81 | 0.36 | 63.246548 | 6.000 | 2.667 | 78155.9 | 111.024 | 0.831860 | 536.470508 | 536.935181 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
