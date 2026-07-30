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
| Centralized all-in-one raw-LiDAR upper ref |  |  |  | 147.233693 | 1.000 | 10.000 | 127946.9 | 27.839 | 1.236340 | 89.411751 | 90.586894 |
| No collaboration |  |  |  | 0.000000 | 10.000 | 1.000 | 127946.9 | 129.677 | 1.364341 | 894.117514 | 894.869877 |
| Pure late |  |  |  | 0.000000 | 10.000 | 1.000 | 127946.9 | 114.935 | 1.364341 | 894.117514 | 894.869877 |
| FullPerception-PCS |  |  |  | 36.352165 | 10.000 | 1.490 | 156347.1 | 136.452 | 1.635612 | 894.117514 | 895.141148 |
| EdgeCooper-Pmax V2V adaptation |  |  |  | 69.137961 | 10.000 | 3.384 | 181961.0 | 189.839 | 1.880270 | 894.117514 | 895.385806 |
| PACP-LiDAR V2V adaptation |  |  |  | 81.151092 | 10.000 | 1.535 | 191346.2 | 144.968 | 1.969915 | 894.117514 | 895.475451 |
| SGCP |  |  |  | 15.420655 | 5.258 | 1.902 | 79554.4 | 73.968 | 0.834666 | 470.132757 | 470.645641 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
