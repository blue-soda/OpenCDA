# Dense-LiDAR Table 2. Global Box and Fusion Scaffold Diagnostics

Protocol: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, and 60 ms data-plane deadline. Dense Table 2 reuses Table 1/Table 3 traces when the protocol is identical.

## No-Clustering Global Box Aggregation

| Method | Late fusion | Clustering | Scheduler/protocol | Receiver policy | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure late | prediction_nms | singleton | cov_potential_game | all-cavs | 0.81 | 0.73 | 0.52 | 0.00 | 1.95 | 1.95 | 1745.11 | 0.00 |
| FullPerception-PCS + global box | global_box_nms | singleton | fullperception_pcs | all-cavs | 0.81 | 0.74 | 0.53 | 70.49 | 2.21 | 72.70 | 1749.70 | 6.82 |
| EdgeCooper V2V + global box | global_box_nms | singleton | selective_edgecooper_global | all-cavs | 0.81 | 0.74 | 0.53 | 86.30 | 2.02 | 88.32 | 1745.39 | 0.86 |
| SGCP | inter_cluster_nms | potential_verified_cov_coalition_game | cov_potential_game | all-cluster-heads | 0.84 | 0.78 | 0.56 | 59.58 | 0.79 | 60.37 | 606.62 | 25.31 |
| Centralized all-in-one raw-LiDAR upper ref | none | all_in_one | cov_potential_game | all-cluster-heads | 0.90 | 0.90 | 0.80 | 337.08 | 0.00 | 337.08 | 90.55 | 0.00 |

## Fusion Scaffold Ablation

| Variant | Late fusion | Clustering | Scheduler/protocol | Receiver policy | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HeadOnly | inter_cluster_nms | potential_verified_cov_coalition_game | cov_potential_game | all-cluster-heads | 0.80 | 0.72 | 0.51 | 0.00 | 0.67 | 0.67 | 606.43 | 0.00 |
| PureLate | prediction_nms | singleton | cov_potential_game | all-cavs | 0.81 | 0.73 | 0.52 | 0.00 | 1.95 | 1.95 | 1745.11 | 0.00 |
| OneClusterEarlyOnly | none | all_in_one | cov_potential_game | all-cluster-heads | 0.90 | 0.90 | 0.80 | 337.08 | 0.00 | 337.08 | 90.55 | 0.00 |
| ClusteredEarlyOnly | none | potential_verified_cov_coalition_game | cov_potential_game | all-cluster-heads | 0.57 | 0.52 | 0.37 | 41.02 | 0.00 | 41.02 | 606.56 | 26.98 |
| OneClusterEarlyLate | identity_single_cluster | all_in_one | cov_potential_game | all-cluster-heads | 0.90 | 0.90 | 0.80 | 337.08 | 0.00 | 337.08 | 90.55 | 0.00 |
| FullSGCP | inter_cluster_nms | potential_verified_cov_coalition_game | cov_potential_game | all-cluster-heads | 0.84 | 0.78 | 0.56 | 59.58 | 0.79 | 60.37 | 606.62 | 25.31 |

Notes:
- `Pure late` and global-box rows share prediction boxes and therefore report box payload in addition to raw-LiDAR payload.
- `Centralized all-in-one` and `OneClusterEarlyOnly` are upper references; they are not feasible all-receiver V2V baselines under the 60 ms data-plane constraint.
- Dense LiDAR makes local/head-only perception much stronger than in the sparse package. The useful SGCP gain is now best read as a three-way tradeoff: AP improvement over cluster-head-only, lower raw payload than dense selective baselines, and much lower GFLOPs than all-CAV pure late/global-box rows.
