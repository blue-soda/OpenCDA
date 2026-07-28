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
| SGCP | 0.84 | 0.78 | 0.56 | 60.372980 | 6.780 | 2.421 | 140679.9 | 118.878 | 0.781826 | 606.427154 | 606.622165 |
| Cluster-head late only | 0.80 | 0.72 | 0.51 | 0.674310 | 6.780 | 1.000 | 94134.4 | 100.171 | 0.584954 | 606.427154 | 606.425293 |
| FullPerception-PCS | 0.82 | 0.74 | 0.54 | 21.879259 | 6.780 | 1.399 | 110624.3 | 115.463 | 0.654701 | 606.427154 | 606.495040 |
| Random budget | 0.83 | 0.79 | 0.60 | 86.117058 | 6.780 | 2.917 | 160763.4 | 124.561 | 0.866772 | 606.427154 | 606.707111 |
| Density greedy | 0.83 | 0.79 | 0.58 | 87.085487 | 6.780 | 2.917 | 161559.4 | 116.683 | 0.870139 | 606.427154 | 606.710478 |
| Link-aware density | 0.83 | 0.79 | 0.58 | 87.085487 | 6.780 | 2.917 | 161559.4 | 116.683 | 0.870139 | 606.427154 | 606.710478 |
| PACP-LiDAR | 0.78 | 0.71 | 0.50 | 86.999727 | 6.780 | 2.781 | 161559.4 | 103.341 | 0.870139 | 606.427154 | 606.710478 |
| EdgeCooper-HD | 0.82 | 0.75 | 0.54 | 86.966384 | 6.780 | 2.442 | 161487.7 | 112.537 | 0.869836 | 606.427154 | 606.710175 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
