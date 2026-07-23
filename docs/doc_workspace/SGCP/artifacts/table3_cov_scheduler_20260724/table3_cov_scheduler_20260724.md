# Table 3. SGCP-compatible Scheduler Comparison

Protocol: attentive detector, v2xp_cluster_carla 41-frame offline replay, 20 CAVs, 40 MHz / 10 target subchannels, NS3-calibrated channel estimator (`tb_size=899`, `symbols=12`, `mcs=28`), `cov_coalition_game` clustering, all cluster heads as receivers, grid raw-LiDAR upload, inter-cluster box NMS. SGCP-CV uses the selected main operating point from the Raw LiDAR Mbps Budget sweep (`200 Mbps` row). Baseline scheduler rows report K=2 receiver-side concurrent inbound links where applicable, aligning their receiver capability with SGCP.

| Method | Late fusion | Clustering | Scheduler/protocol | Receiver policy | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGCP-CV | inter_cluster_nms | cov_coalition_game | cov_potential_game | all-cluster-heads | 0.87 | 0.80 | 0.36 | 60.18 | 0.72 | 60.90 | 536.92 | 85.73 |
| Cluster-head late only | inter_cluster_nms | cov_coalition_game | local_detection_head_only | all-cluster-heads | 0.42 | 0.30 | 0.13 | 0.00 | 0.45 | 0.45 | 536.47 | 0.00 |
| FullPerception-PCS | inter_cluster_nms | cov_coalition_game | fullperception_pcs | all-cluster-heads | 0.61 | 0.46 | 0.22 | 12.90 | 0.53 | 13.42 | 536.56 | 13.29 |
| Random budget | inter_cluster_nms | cov_coalition_game | selective_random | all-cluster-heads | 0.85 | 0.75 | 0.36 | 61.75 | 0.72 | 62.47 | 536.93 | 103.20 |
| Density greedy | inter_cluster_nms | cov_coalition_game | selective_density | all-cluster-heads | 0.86 | 0.78 | 0.38 | 75.94 | 0.74 | 76.69 | 537.04 | 103.20 |
| Link-aware density | inter_cluster_nms | cov_coalition_game | selective_communication_aware | all-cluster-heads | 0.86 | 0.78 | 0.38 | 75.94 | 0.74 | 76.69 | 537.04 | 103.20 |
| PACP-LiDAR | inter_cluster_nms | cov_coalition_game | selective_pacp_lidar | all-cluster-heads | 0.88 | 0.78 | 0.37 | 86.30 | 0.76 | 87.06 | 537.11 | 94.98 |
| EdgeCooper-HD | inter_cluster_nms | cov_coalition_game | selective_edgecooper_global_hd | all-cluster-heads | 0.81 | 0.70 | 0.32 | 52.49 | 0.70 | 53.19 | 536.86 | 71.74 |

Notes:
- This is a scheduler comparison under the same SGCP-compatible clustering and late-aggregation scaffold, not a protocol-native baseline table.
- `Raw Mbps` is scheduled raw-LiDAR payload; `Box Mbps` is inter-cluster detection-box aggregation payload; `Total Mbps = Raw Mbps + Box Mbps`.
- K=2 is used in the main table to align baseline receiver capability with SGCP. FullPerception-PCS K=1: AP 0.58/0.43/0.21, Total 11.37 Mbps, GFLOPs/frame 536.55; EdgeCooper-HD K=1: AP 0.72/0.60/0.29, Total 31.10 Mbps, GFLOPs/frame 536.70.
- Admission-budget parameters are internal scheduler controls and are not reported as paper-facing table columns; feasibility is judged from the resulting payload and NS3-calibrated delay diagnostics.
