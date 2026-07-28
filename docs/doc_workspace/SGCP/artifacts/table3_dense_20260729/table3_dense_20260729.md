# Dense-LiDAR Table 3. SGCP-Compatible Scheduler Comparison

Protocol: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, `potential_verified_cov_coalition_game` clustering, all cluster heads as receivers, grid raw-LiDAR upload, and inter-cluster box NMS. Only the scheduler/protocol changes. SGCP uses `N_max=5`, `rho_th=1`, and a 60 Mbps raw-LiDAR frame cap with per-link deadline trimming.

| Method | Late fusion | Clustering | Scheduler/protocol | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids | P95 link time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGCP | inter_cluster_nms | potential_verified_cov_coalition_game | cov_potential_game | 0.84 | 0.78 | 0.56 | 59.58 | 0.79 | 60.37 | 606.62 | 25.31 | 60.00 ms |
| Cluster-head late only | inter_cluster_nms | potential_verified_cov_coalition_game | local_detection_head_only | 0.80 | 0.72 | 0.51 | 0.00 | 0.67 | 0.67 | 606.43 | 0.00 | 0.00 ms |
| FullPerception-PCS | inter_cluster_nms | potential_verified_cov_coalition_game | fullperception_pcs | 0.82 | 0.74 | 0.54 | 21.11 | 0.77 | 21.88 | 606.50 | 5.01 | 60.00 ms |
| Random budget | inter_cluster_nms | potential_verified_cov_coalition_game | selective_random | 0.83 | 0.79 | 0.60 | 85.29 | 0.83 | 86.12 | 606.71 | 29.88 | 17.97 ms |
| Density greedy | inter_cluster_nms | potential_verified_cov_coalition_game | selective_density | 0.83 | 0.79 | 0.58 | 86.30 | 0.78 | 87.09 | 606.71 | 15.51 | 19.04 ms |
| Link-aware density | inter_cluster_nms | potential_verified_cov_coalition_game | selective_communication_aware | 0.83 | 0.79 | 0.58 | 86.30 | 0.78 | 87.09 | 606.71 | 15.51 | 19.04 ms |
| PACP-LiDAR | inter_cluster_nms | potential_verified_cov_coalition_game | selective_pacp_lidar | 0.78 | 0.71 | 0.50 | 86.30 | 0.70 | 87.00 | 606.71 | 2.49 | 15.43 ms |
| EdgeCooper-HD | inter_cluster_nms | potential_verified_cov_coalition_game | selective_edgecooper_global_hd | 0.82 | 0.75 | 0.54 | 86.21 | 0.75 | 86.97 | 606.71 | 5.26 | 15.56 ms |

Notes:
- This is a scheduler comparison under the same SGCP-compatible clustering and late-aggregation scaffold, not a protocol-native baseline table.
- `Raw Mbps` is scheduled raw-LiDAR payload; `Box Mbps` is inter-cluster detection-box aggregation payload; `Total Mbps = Raw Mbps + Box Mbps`.
- Dense LiDAR makes the cluster-head-only lower reference much stronger than in the sparse package. The scheduler table should therefore be interpreted together with GFLOPs: SGCP improves AP over cluster-head late only while keeping detector compute far below all-CAV pure late.
- Baseline scheduler rows use K=2 receiver-side concurrent inbound links where applicable, aligned with SGCP receiver capability.
