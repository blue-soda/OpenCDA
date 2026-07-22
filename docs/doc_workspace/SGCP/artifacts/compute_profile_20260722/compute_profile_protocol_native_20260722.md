# SGCP Profiled GFLOPs Summary

This file summarizes detector-side compute from offline trace CSVs. GFLOPs are calibrated from one real OpenCOOD forward when a calibration JSON is provided; otherwise the table still reports forward-equivalent compute and input-size diagnostics.

## Calibration

- fusion_method: `early`
- scenario/timestamp: `2026_07_15_01_26_56/000060`
- CAVs in calibration forward: `1`
- input points: `4918`
- profiled detector GFLOPs/forward: `89.249203`
- FLOP policy: Conv2d/ConvTranspose2d/Linear hooks; multiply-add=2 FLOPs

## Compute Table

| label | ap_03 | ap_05 | ap_07 | total_mbps | detector_calls_per_frame | mean_source_cavs_per_call | mean_input_points_per_frame | mean_pred_boxes_per_frame | profiled_detector_gflops_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FullPerception-PCS | 0.23 | 0.17 | 0.06 | 53.551485 | 20.000 | 1.454 | 139506.2 | 299.927 | 1784.984064 |
| EdgeCooper V2V protocol adaptation | 0.54 | 0.48 | 0.25 | 275.937186 | 20.000 | 3.524 | 313245.0 | 493.976 | 1784.984064 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
