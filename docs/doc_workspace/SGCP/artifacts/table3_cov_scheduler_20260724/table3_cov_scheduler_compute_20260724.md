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
| SGCP-CV | 0.87 | 0.80 | 0.36 | 60.903430 | 6.000 | 2.667 | 76316.0 | 108.024 | 0.814286 | 536.470508 | 536.917608 |
| Cluster-head late only | 0.42 | 0.30 | 0.13 | 0.452121 | 6.000 | 1.000 | 29299.4 | 66.293 | 0.365194 | 536.470508 | 536.468516 |
| FullPerception-PCS | 0.61 | 0.46 | 0.22 | 13.421893 | 6.000 | 1.439 | 39374.6 | 77.756 | 0.461430 | 536.470508 | 536.564752 |
| Random budget | 0.85 | 0.75 | 0.36 | 62.474427 | 6.000 | 3.333 | 77543.9 | 107.927 | 0.826014 | 536.470508 | 536.929335 |
| Density greedy | 0.86 | 0.78 | 0.38 | 76.687173 | 6.000 | 3.333 | 88629.5 | 111.537 | 0.931901 | 536.470508 | 537.035223 |
| Link-aware density | 0.86 | 0.78 | 0.38 | 76.685768 | 6.000 | 3.333 | 88629.5 | 111.317 | 0.931901 | 536.470508 | 537.035223 |
| PACP-LiDAR | 0.88 | 0.78 | 0.37 | 87.062384 | 6.000 | 3.333 | 96723.1 | 113.951 | 1.009209 | 536.470508 | 537.112531 |
| EdgeCooper-HD | 0.81 | 0.70 | 0.32 | 53.189807 | 6.000 | 2.561 | 70304.0 | 105.268 | 0.756861 | 536.470508 | 536.860182 |
| FullPerception-PCS K1 note | 0.58 | 0.43 | 0.21 | 11.365307 | 6.000 | 1.366 | 37771.0 | 77.122 | 0.446113 | 536.470508 | 536.549435 |
| EdgeCooper-HD K1 note | 0.72 | 0.60 | 0.29 | 31.099161 | 6.000 | 1.935 | 53142.1 | 86.146 | 0.592934 | 536.470508 | 536.696256 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
