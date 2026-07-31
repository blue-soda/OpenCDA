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
| no_collaboration |  |  |  | 0.000000 | 20.000 | 1.000 | 277196.3 | 359.333 | 1.723433 | 1788.741965 | 1788.734505 |
| pcs |  |  |  | 73.771520 | 26.333 | 1.483 | 422697.0 | 384.333 | 2.513331 | 2355.176921 | 2355.411242 |
| edgecooper_pmax |  |  |  | 86.304000 | 39.000 | 3.467 | 607500.0 | 469.667 | 3.643943 | 3488.046832 | 3488.315533 |
| pacp_lidar |  |  |  | 86.303147 | 27.000 | 1.500 | 441634.0 | 371.333 | 2.611794 | 2414.801653 | 2415.076741 |

## Notes

- `detector_calls_per_frame` is the number of OpenCOOD detector forwards represented by the trace in one 100 ms perception cycle.
- Pure late/global box baselines can have high AP because many CAVs run local detection; SGCP reduces this by evaluating only cluster heads while still ingesting selected member point clouds.
- `mean_input_points_per_frame` is trace-derived and therefore captures fused point-cloud size, not just the number of receivers.
- `input_adjusted_detector_gflops_per_frame`, when present, uses singleton and dense/full calibrations to model point-dependent VFE point-feature FLOPs on top of fixed BEV Conv/Deconv FLOPs.
- `input_adjusted_point_feature_gflops_per_frame` is the point-cloud-to-feature floating-point subtotal. It includes PillarVFE elementwise feature construction, PFN Linear, BatchNorm and simple activation FLOPs when visible to hooks; it excludes voxelization/hash/scatter memory/index operations.
