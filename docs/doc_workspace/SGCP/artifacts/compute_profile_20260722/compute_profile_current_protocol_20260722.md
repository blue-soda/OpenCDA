# SGCP Profiled GFLOPs Summary

This file summarizes detector-side compute from offline trace CSVs. GFLOPs are calibrated from one real OpenCOOD forward when a calibration JSON is provided; otherwise the table still reports forward-equivalent compute and input-size diagnostics.

## Calibration

- fusion_method: `early`
- scenario/timestamp: `2026_07_15_01_26_56/000060`
- CAVs in calibration forward: `1`
- input points: `4918`
- profiled detector GFLOPs/forward: `89.302374`
- FLOP policy: Conv2d/ConvTranspose2d/Linear hooks; multiply-add=2 FLOPs

## Compute Table

| label | ap_03 | ap_05 | ap_07 | total_mbps | detector_calls_per_frame | mean_source_cavs_per_call | mean_input_points_per_frame | mean_pred_boxes_per_frame | profiled_detector_gflops_per_frame | input_adjusted_detector_gflops_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PureLate_current_protocol | 0.82 | 0.76 | 0.37 | 0.739122 | 19.171 | 1.000 | 93608.5 | 102.415 | 1711.991860 | 1711.986094 |
| HeadOnly_current_protocol | 0.26 | 0.22 | 0.09 | 0.247259 | 6.000 | 1.000 | 29299.4 | 34.244 | 535.814246 | 535.812459 |
| OneClusterEarlyOnly_current_protocol | 0.85 | 0.83 | 0.48 | 118.709323 | 1.000 | 20.000 | 97669.1 | 60.122 | 89.302374 | 90.096943 |
| ClusteredEarlyOnly_current_protocol | 0.31 | 0.29 | 0.14 | 36.688172 | 6.000 | 2.000 | 57962.0 | 70.829 | 535.814246 | 536.058003 |
| SGCP_PAPG_main_current_protocol | 0.87 | 0.81 | 0.36 | 63.246549 | 6.000 | 2.667 | 78155.9 | 111.024 | 535.814246 | 536.230997 |
| FullPerceptionPCS_current_protocol | 0.50 | 0.36 | 0.17 | 13.704304 | 6.000 | 1.386 | 39577.7 | 81.220 | 535.814246 | 535.900510 |
| RandomBudget_current_protocol | 0.78 | 0.74 | 0.39 | 62.487227 | 6.000 | 3.333 | 77747.5 | 69.220 | 535.814246 | 536.227499 |
| DensityGreedy_current_protocol | 0.80 | 0.76 | 0.41 | 76.441787 | 6.000 | 3.333 | 88629.5 | 73.195 | 535.814246 | 536.320721 |
| PACP_LiDAR_current_protocol | 0.81 | 0.79 | 0.42 | 86.804199 | 6.000 | 3.333 | 96723.1 | 73.610 | 535.814246 | 536.390056 |
| EdgeCooperHD_current_protocol | 0.60 | 0.55 | 0.25 | 30.873007 | 6.000 | 1.935 | 53142.1 | 50.927 | 535.814246 | 536.016712 |
| FullPerception-PCS + global box current-protocol | 0.83 | 0.77 | 0.38 | 54.542517 | 19.854 | 1.457 | 138793.5 | 140.976 | 1772.978848 | 1773.331394 |
| EdgeCooper V2V + global box current-protocol | 0.84 | 0.79 | 0.37 | 51.825701 | 19.220 | 1.440 | 133619.3 | 129.683 | 1716.348074 | 1716.683012 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE Linear FLOPs on top of fixed BEV Conv/Deconv FLOPs.
