# Dense-LiDAR Table 4. Clustering Baselines under the SGCP Protocol

Protocol: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, `cov_potential_game` C->V raw-LiDAR scheduler, all cluster heads as receivers, grid upload, inter-cluster box NMS. Only the clustering algorithm changes.

| Method | Baseline type | Clustering | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg grids | P95 link time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGCP | proposed | potential_verified_cov_coalition_game | 0.84 | 0.78 | 0.56 | 59.58 | 0.79 | 60.37 | 606.62 | 2.42 | 25.31 | 60.00 ms |
| Random balanced | heuristic | random_balanced | 0.65 | 0.57 | 0.37 | 33.86 | 0.53 | 34.39 | 357.86 | 2.00 | 8.38 | 60.00 ms |
| Distance-greedy | heuristic | distance_greedy | 0.77 | 0.72 | 0.51 | 28.81 | 0.56 | 29.37 | 357.84 | 2.00 | 17.50 | 60.00 ms |
| Density/quality-greedy | heuristic | density_greedy_cluster | 0.66 | 0.56 | 0.35 | 34.33 | 0.57 | 34.90 | 357.86 | 2.00 | 6.28 | 60.00 ms |
| SeAC-inspired | paper baseline | seac_social_adaptive | 0.67 | 0.61 | 0.41 | 31.03 | 0.54 | 31.57 | 357.85 | 2.00 | 17.09 | 60.00 ms |
| HHOCNET-inspired | paper baseline | hho_vanet | 0.67 | 0.59 | 0.38 | 34.02 | 0.53 | 34.55 | 357.86 | 2.00 | 7.95 | 60.00 ms |

Notes:
- `Total Mbps = Raw Mbps + Box Mbps`; all rows include the same inter-cluster detection-box communication accounting.
- This table is a clustering comparison, not a protocol-native baseline table. Resource scheduling, late fusion, checkpoint, channel estimator and communication accounting are fixed.
- Dense LiDAR reduces the AP gap among clustering methods. SGCP keeps the best AP@0.7 among the tested clustering methods while using comparable detector compute.
