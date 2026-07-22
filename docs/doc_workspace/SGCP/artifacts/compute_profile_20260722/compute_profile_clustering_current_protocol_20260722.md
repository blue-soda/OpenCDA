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
| Singleton pure late reference | 0.82 | 0.76 | 0.37 | 0.739122 | 19.171 | 1.000 | 93608.5 | 102.415 | 1.166777 | 1714.088698 | 1714.082268 |
| Random balanced clusters | 0.52 | 0.47 | 0.24 | 31.156574 | 5.000 | 2.000 | 48570.2 | 45.659 | 0.535042 | 447.058757 | 447.287810 |
| Distance-greedy clusters | 0.55 | 0.53 | 0.31 | 31.246798 | 5.000 | 2.000 | 48673.6 | 40.561 | 0.536030 | 447.058757 | 447.288798 |
| Mobility-stability greedy clusters | 0.60 | 0.54 | 0.27 | 31.221011 | 5.000 | 2.000 | 48492.2 | 51.634 | 0.534297 | 447.058757 | 447.287065 |
| Density/quality-greedy clusters | 0.62 | 0.56 | 0.36 | 31.239024 | 5.000 | 2.000 | 48633.8 | 52.098 | 0.535649 | 447.058757 | 447.288417 |
| Fixed first-frame clusters | 0.63 | 0.56 | 0.22 | 37.105701 | 6.000 | 2.000 | 57955.1 | 50.244 | 0.638907 | 536.470508 | 536.742228 |
| Dynamic coalition clusters (SGCP) | 0.87 | 0.81 | 0.36 | 63.246548 | 6.000 | 2.667 | 78155.9 | 111.024 | 0.831860 | 536.470508 | 536.935181 |
| All-in-one full raw sharing | 0.85 | 0.83 | 0.48 | 118.709323 | 1.000 | 20.000 | 97669.1 | 60.122 | 0.947134 | 89.411751 | 90.297687 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
