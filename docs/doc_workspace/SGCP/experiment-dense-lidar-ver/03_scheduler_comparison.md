# Dense-LiDAR Table 3. SGCP-Compatible Scheduler Comparison

Protocol: dense `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames, attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, `potential_verified_cov_coalition_game` clustering, all cluster heads as receivers, grid raw-LiDAR upload, and inter-cluster box NMS. Only the scheduler/protocol changes. SGCP rows use `rho_th=2`, `upload_density_cap_rho=2`, density-capped deterministic random point upload, `head_rb_budget=2`, `N_max=5`, and 40 Mbps raw admission budget.

| Method | Late fusion | Clustering | Scheduler/protocol | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids | P95 data-plane time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGCP hybrid | inter_cluster_nms | potential_verified_cov_coalition_game | hybrid_round_robin_dynamic_marginal | 0.88 | 0.82 | 0.61 | 28.42 | 0.89 | 29.31 | 593.34 | 49.12 | 37.39 ms |
| SGCP previous scheduler | inter_cluster_nms | potential_verified_cov_coalition_game | dynamic_cv | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 47.89 | 36.79 ms |
| SGCP round-robin dynamic probe | inter_cluster_nms | potential_verified_cov_coalition_game | round_robin_dynamic_marginal | 0.86 | 0.81 | 0.61 | 27.54 | 0.90 | 28.44 | 593.34 | 48.81 | 37.39 ms |
| Cluster-head late only | inter_cluster_nms | potential_verified_cov_coalition_game | local_detection_head_only | 0.79 | 0.71 | 0.49 | 0.00 | 0.66 | 0.66 | 593.34 | 0.00 | 0.00 ms |
| FullPerception-PCS | inter_cluster_nms | potential_verified_cov_coalition_game | fullperception_pcs | 0.82 | 0.76 | 0.54 | 19.70 | 0.77 | 20.46 | 593.40 | 8.72 | 60.00 ms |
| Random budget | inter_cluster_nms | potential_verified_cov_coalition_game | selective_random | 0.82 | 0.77 | 0.53 | 86.30 | 0.73 | 87.04 | 593.62 | 8.84 | 18.89 ms |
| Density greedy | inter_cluster_nms | potential_verified_cov_coalition_game | selective_density | 0.81 | 0.76 | 0.52 | 86.30 | 0.71 | 87.01 | 593.62 | 6.00 | 20.52 ms |
| Link-aware density | inter_cluster_nms | potential_verified_cov_coalition_game | selective_communication_aware | 0.81 | 0.76 | 0.52 | 86.30 | 0.71 | 87.01 | 593.62 | 6.00 | 20.52 ms |
| PACP-LiDAR | inter_cluster_nms | potential_verified_cov_coalition_game | selective_pacp_lidar | 0.88 | 0.82 | 0.59 | 32.53 | 0.96 | 33.48 | 593.98 | 67.34 | 5.61 ms |
| EdgeCooper-HD | inter_cluster_nms | potential_verified_cov_coalition_game | selective_edgecooper_global_hd | 0.81 | 0.74 | 0.51 | 86.30 | 0.73 | 87.03 | 593.62 | 3.68 | 15.79 ms |
| EdgeCooper-HD-Pmax | inter_cluster_nms | potential_verified_cov_coalition_game | selective_edgecooper_global_hd_pmax | 0.88 | 0.81 | 0.56 | 32.40 | 1.06 | 33.46 | 593.98 | 60.66 | 5.66 ms |

Notes:
- This is a scheduler comparison under the same SGCP-compatible clustering and late-aggregation scaffold, not a protocol-native baseline table.
- `SGCP hybrid` first gives each cluster head one round-robin opportunity and then greedily assigns the remaining subchannels to the largest dynamic early-utility increments. It is currently the strongest SGCP scheduler candidate under this protocol.
- `SGCP previous scheduler` is retained for continuity with the earlier dense headline. `SGCP round-robin dynamic probe` is a diagnostic row for the pure round-robin variant and should not be treated as a separate baseline.
- Baseline schedulers use `rho_th=2` for density/visibility scoring. The paper-aligned PACP-LiDAR row uses density-capped raw-LiDAR upload as the LiDAR counterpart of PACP's compression/rate-control stage.
- `EdgeCooper-HD-Pmax` additionally applies the EdgeCooper paper-style partial point-upload cap (`edgecooper_pmax_density_cap_rho=2`) under the same SGCP-compatible clustering and late-aggregation scaffold. It is reported separately from full-grid `EdgeCooper-HD` because the upload truncation changes the payload model.
- The non-Pmax `EdgeCooper-HD` row is deprecated for paper-facing EdgeCooper comparison and retained only as a diagnostic full-grid scheduler result. Use `EdgeCooper-HD-Pmax` for the EdgeCooper-compatible scheduler adaptation.
- `Raw Mbps` is scheduled raw-LiDAR payload; `Box Mbps` is inter-cluster detection-box aggregation payload; `Total Mbps = Raw Mbps + Box Mbps`.
- `P95 data-plane time` is a receiver/link-side service statistic under the trace estimator, not a serial whole-frame transmission time. The protocol-native Table 1 EdgeCooper full-grid row reports the global 60 ms deadline-constrained admission boundary.
- PACP rerun provenance: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\pacp_lidar_paper_fix_20260801\table3_sgcp_compatible`.
