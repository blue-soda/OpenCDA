# SGCP Profiled GFLOPs Summary

This file summarizes detector-side compute from offline trace CSVs. GFLOPs are calibrated from one real OpenCOOD forward when a calibration JSON is provided; otherwise the table still reports forward-equivalent compute and input-size diagnostics.

## Calibration

- fusion_method: `early`
- scenario/timestamp: `2026_07_29_02_32_08/000060`
- CAVs in calibration forward: `1`
- input points: `13948`
- profiled detector GFLOPs/forward: `89.437098`
- FLOP policy: Conv2d/ConvTranspose2d/Linear/BatchNorm/ReLU hooks plus PillarVFE elementwise estimate; multiply-add=2 FLOPs; voxelization/hash/scatter memory ops excluded

## Compute Table

| label | ap_03 | ap_05 | ap_07 | total_mbps | detector_calls_per_frame | mean_source_cavs_per_call | mean_input_points_per_frame | mean_pred_boxes_per_frame | input_adjusted_point_feature_gflops_per_frame | profiled_detector_gflops_per_frame | input_adjusted_detector_gflops_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgcp | 0.84 | 0.78 | 0.56 | 60.372980 | 6.780 | 2.421 | 140679.9 | 118.878 | 0.781826 | 606.427154 | 606.622165 |
| random_balanced | 0.65 | 0.57 | 0.37 | 34.391477 | 4.000 | 2.000 | 82015.3 | 79.732 | 0.457094 | 357.748393 | 357.859308 |
| distance_greedy | 0.77 | 0.72 | 0.51 | 29.367945 | 4.000 | 2.000 | 78288.8 | 84.366 | 0.441332 | 357.748393 | 357.843547 |
| density_greedy | 0.66 | 0.56 | 0.35 | 34.900730 | 4.000 | 2.000 | 82447.5 | 85.561 | 0.458922 | 357.748393 | 357.861137 |
| seac_social | 0.67 | 0.61 | 0.41 | 31.570794 | 4.000 | 2.000 | 79907.2 | 80.683 | 0.448177 | 357.748393 | 357.850392 |
| hho_vanet | 0.67 | 0.59 | 0.38 | 34.553475 | 4.000 | 2.000 | 82225.8 | 80.024 | 0.457984 | 357.748393 | 357.860199 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
