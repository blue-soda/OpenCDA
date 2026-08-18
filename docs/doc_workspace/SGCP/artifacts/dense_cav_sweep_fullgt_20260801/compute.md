# SGCP Profiled GFLOPs Summary

This file summarizes detector-side compute from offline trace CSVs. GFLOPs are calibrated from one real OpenCOOD forward when a calibration JSON is provided; otherwise the table still reports forward-equivalent compute and input-size diagnostics.

## Calibration

- fusion_method: `early`
- scenario/timestamp: `2026_07_29_02_32_08/000060`
- CAVs in calibration forward: `1`
- input points: `13948`
- profiled detector GFLOPs/forward: `89.437098`
- FLOP policy: Conv2d/Conv3d/ConvTranspose2d/Linear/BatchNorm/ReLU hooks, sparse-3D active-output proxy, and PillarVFE/MeanVFE elementwise estimate; multiply-add=2 FLOPs; voxelization/hash/scatter memory/index ops excluded

## Compute Table

| label | ap_03 | ap_05 | ap_07 | total_mbps | detector_calls_per_frame | mean_source_cavs_per_call | mean_input_points_per_frame | mean_pred_boxes_per_frame | input_adjusted_point_feature_gflops_per_frame | profiled_detector_gflops_per_frame | input_adjusted_detector_gflops_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N5_pure_late |  |  |  | 0.000000 | 5.000 | 1.000 | 69649.0 | 84.927 | 0.432339 | 447.185491 | 447.185107 |
| N5_edgecooper_pmax |  |  |  | 12.752297 | 8.000 | 1.800 | 121187.1 | 134.317 | 0.732975 | 715.496786 | 715.537404 |
| N5_sgcp |  |  |  | 9.085940 | 1.366 | 2.732 | 26139.1 | 42.317 | 0.148188 | 122.157988 | 122.187968 |
| N10_pure_late |  |  |  | 0.000000 | 10.000 | 1.000 | 138439.7 | 183.244 | 0.861046 | 894.370983 | 894.366582 |
| N10_edgecooper_pmax |  |  |  | 59.654743 | 18.000 | 3.200 | 296049.9 | 339.317 | 1.748079 | 1609.867769 | 1610.058043 |
| N10_sgcp |  |  |  | 17.647703 | 3.000 | 2.732 | 55510.1 | 85.341 | 0.317437 | 268.311295 | 268.369098 |
| N15_pure_late |  |  |  | 0.000000 | 14.512 | 1.000 | 200682.2 | 235.098 | 1.248619 | 1297.928621 | 1297.921287 |
| N15_edgecooper_pmax |  |  |  | 83.816554 | 27.927 | 3.154 | 451304.3 | 430.073 | 2.678229 | 2497.694573 | 2497.955885 |
| N15_sgcp |  |  |  | 22.969163 | 4.195 | 2.686 | 75602.4 | 95.976 | 0.435345 | 375.199534 | 375.271814 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
