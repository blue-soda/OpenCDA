# SGCP Profiled GFLOPs Summary

This file summarizes detector-side compute from offline trace CSVs. GFLOPs are calibrated from one real OpenCOOD forward when a calibration JSON is provided; otherwise the table still reports forward-equivalent compute and input-size diagnostics.

## Calibration

- fusion_method: ``
- scenario/timestamp: `/`
- CAVs in calibration forward: ``
- input points: ``
- profiled detector GFLOPs/forward: `89.437098`
- FLOP policy: user-provided calibrated GFLOPs/forward

## Compute Table

| label | ap_03 | ap_05 | ap_07 | total_mbps | detector_calls_per_frame | mean_source_cavs_per_call | mean_input_points_per_frame | mean_pred_boxes_per_frame | input_adjusted_point_feature_gflops_per_frame | profiled_detector_gflops_per_frame | input_adjusted_detector_gflops_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgcp_hybrid |  |  |  | 28.417780 | 6.634 | 2.507 | 114283.6 | 134.146 |  | 593.338796 |  |
| random_balanced |  |  |  | 39.635231 | 4.000 | 3.000 | 86526.6 | 129.780 |  | 357.748392 |  |
| distance_greedy |  |  |  | 30.733175 | 4.000 | 3.000 | 79793.2 | 108.220 |  | 357.748392 |  |
| density_greedy |  |  |  | 39.392687 | 4.000 | 3.000 | 86423.0 | 130.512 |  | 357.748392 |  |
| seac_social |  |  |  | 32.088757 | 4.000 | 3.000 | 80731.0 | 110.854 |  | 357.748392 |  |
| hho_vanet |  |  |  | 38.621596 | 4.000 | 3.000 | 85820.1 | 126.195 |  | 357.748392 |  |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
