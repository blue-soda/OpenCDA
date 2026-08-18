# Dense-LiDAR Table 2. Global Box and Fusion Scaffold Diagnostics

Protocol: dense `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, and 60 ms data-plane deadline. SGCP rows use the rho-Pareto selected `rho_th=2`, `upload_density_cap_rho=2`, `dynamic_cv` dynamic C/V scheduler with density-capped deterministic random point upload, and 40 Mbps raw admission budget.

Status note: this fusion scaffold table appends SGCP hybrid rows while retaining the earlier `dynamic_cv` rows as provenance. The baseline/global-box AP and communication results are unchanged; their GFLOPs are updated to the detector-plus-NMS Table 2 compute convention.

## Global Box Aggregation

| Method | Late fusion | Clustering | Scheduler/protocol | Receiver policy | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure late | prediction_nms | singleton | local_detection + box_nms | all-cavs | 0.81 | 0.73 | 0.52 | 0.00 | 1.95 | 1.95 | 1745.12 | 0.00 |
| FullPerception-PCS + global box | global_box_nms | singleton | fullperception_pcs | all-cavs | 0.81 | 0.74 | 0.53 | 70.49 | 2.21 | 72.70 | 1790.48 | 6.82 |
| EdgeCooper V2V + global box | global_box_nms | singleton | selective_edgecooper_global | all-cavs | 0.81 | 0.74 | 0.53 | 86.30 | 2.02 | 88.32 | 1745.40 | 0.85 |
| EdgeCooper-Pmax + global box | global_box_nms | singleton | selective_edgecooper_global_pmax | all-cavs | 0.85 | 0.80 | 0.56 | 86.30 | 2.64 | 88.95 | 1790.62 | 27.75 |
| PACP-LiDAR + global box | global_box_nms | singleton | selective_pacp_lidar | all-cavs | 0.86 | 0.79 | 0.55 | 86.30 | 2.72 | 89.02 | 1753.50 | 34.71 |
| SGCP hybrid | inter_cluster_nms | potential_verified_cov_coalition_game | hybrid_round_robin_dynamic_marginal | all-cluster-heads | 0.88 | 0.82 | 0.61 | 28.42 | 0.89 | 29.31 | 593.34 | 49.12 |
| SGCP | inter_cluster_nms | potential_verified_cov_coalition_game | dynamic_cv | all-cluster-heads | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 47.89 |
| Centralized all-in-one raw-LiDAR upper ref | none | all_in_one | full raw upload | all-in-one receiver | 0.90 | 0.90 | 0.80 | 337.08 | 0.00 | 337.08 | 90.55 | 0.00 |

## Fusion Scaffold Ablation

| Variant | Late fusion | Clustering | Scheduler/protocol | Receiver policy | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cluster-head late only | inter_cluster_nms | potential_verified_cov_coalition_game | local_detection_head_only | all-cluster-heads | 0.79 | 0.71 | 0.49 | 0.00 | 0.66 | 0.66 | 593.34 | 0.00 |
| ClusteredEarlyOnly hybrid | none | potential_verified_cov_coalition_game | hybrid_round_robin_dynamic_marginal | all-cluster-heads | 0.38 | 0.35 | 0.26 | 28.42 | 0.00 | 28.42 | 593.34 | 49.12 |
| FullSGCP hybrid | inter_cluster_nms | potential_verified_cov_coalition_game | hybrid_round_robin_dynamic_marginal | all-cluster-heads | 0.88 | 0.82 | 0.61 | 28.42 | 0.89 | 29.31 | 593.34 | 49.12 |
| ClusteredEarlyOnly | none | potential_verified_cov_coalition_game | dynamic_cv | all-cluster-heads | 0.59 | 0.55 | 0.39 | 28.42 | 0.00 | 28.42 | 593.43 | 47.94 |
| FullSGCP | inter_cluster_nms | potential_verified_cov_coalition_game | dynamic_cv | all-cluster-heads | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 47.89 |

Notes:
- Global-box rows evaluate whether adding box-level aggregation can recover scene coverage. The strong Pure late row should be read together with its near-20 detector calls/frame compute cost.
- For rows with box/global/inter-cluster NMS, `GFLOPs/frame` reports detector forward GFLOPs plus an estimated NMS cost. The NMS estimate uses the trace-level number of predicted boxes per frame: `mean_t(n_t(n_t-1)/2*256)/1e9`, where `n_t` is the total pre-NMS box count in frame `t`. This keeps Table 2's compute metric distinct from Table 1's detector-only rows while remaining reproducible from saved traces.
- `ClusteredEarlyOnly hybrid` is rerun without inter-cluster late aggregation; its low full-frame-GT AP (`0.38/0.35/0.26`) confirms that raw-LiDAR early fusion inside clusters alone does not recover scene-wide coverage. The gap to `FullSGCP hybrid` isolates the benefit of inter-cluster box aggregation under the same hybrid scheduling trace and dense rho=2 upload protocol.
- Non-Pmax EdgeCooper rows are deprecated for paper-facing EdgeCooper comparison and retained only as diagnostic full-grid evidence. Use `EdgeCooper-Pmax + global box` when reporting the EdgeCooper adaptation because the original EdgeCooper protocol uses per-pillar partial point upload.
- `FullPerception-PCS + global box` is corrected to the same all-20-receiver detector scope as the no-local-preserving Table 1 PCS rerun, then adds the global NMS estimate.
- `EdgeCooper-Pmax + global box` uses the same Pmax-style partial point-upload adaptation as Table 1 with multi-batch all-receiver admission inside the shared 60 ms data-plane window. The trace has 820/820 active receiver samples and no `local-preserving` extra detector calls; its GFLOPs were recomputed from the trace as 20 detector calls/frame plus global NMS.
- `PACP-LiDAR + global box` uses the corrected singleton global candidate fallback, the same K=2 endpoint matching as EdgeCooper, PACP-style BEV-match plus perceptual-region scoring, and density-capped raw-LiDAR upload as the compression/rate-control proxy. Its AP improves over the earlier full-grid PACP proxy but still requires high raw payload and near-20 detector calls/frame.
- PACP rerun provenance: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\pacp_lidar_paper_fix_20260801\table2_global_box`.
- Hybrid provenance: `ClusteredEarlyOnly hybrid` trace is `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\table2\clustered_early_only_hybrid_trace.csv`; `FullSGCP hybrid` reuses `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f\trace.csv`.

## NMS Compute Audit

This audit is used only to make the Table 2 `GFLOPs/frame` column reflect the
extra box aggregation work. It is a deterministic estimate from saved traces,
not a CUDA kernel profiler measurement.

| Row | Active detector calls/frame | Pre-NMS boxes/frame mean | Pre-NMS boxes/frame P95 | Estimated NMS GFLOPs/frame | Detector + NMS GFLOPs/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure late | 19.51 | 289.76 | 302 | 0.0107 | 1745.12 |
| FullPerception-PCS + global box | 20.00 | 329.63 | 356 | 0.0139 | 1790.48 |
| EdgeCooper V2V + global box | 19.51 | 300.29 | 313 | 0.0115 | 1745.40 |
| EdgeCooper-Pmax + global box | 20.00 | 471.15 | 497 | 0.0284 | 1790.62 |
| PACP-LiDAR + global box | 19.59 | 409.39 | 436 | 0.0215 | 1753.50 |
| SGCP hybrid | 6.63 | 134.15 | 147 | 0.0023 | 593.34 |
| SGCP dynamic_cv | 6.63 | 128.73 | 143 | 0.0021 | 593.43 |
