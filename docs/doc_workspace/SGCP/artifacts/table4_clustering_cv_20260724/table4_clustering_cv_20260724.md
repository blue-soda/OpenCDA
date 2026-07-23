# Table 4. Clustering Baselines under the SGCP-CV Protocol

Protocol: attentive detector, v2xp_cluster_carla 41-frame offline replay, 20 CAVs, 40 MHz / 10 target subchannels, NS3-calibrated estimator (`tb_size=899`, `symbols=12`, `mcs=28`), `cov_potential_game` C->V raw-LiDAR scheduler, all cluster heads as receivers, grid upload, inter-cluster box NMS. Only the clustering algorithm changes. SGCP-CV uses the selected main operating point from the Raw LiDAR Mbps Budget sweep; all non-SGCP rows are rerun under the same C/V scheduler and communication accounting.

| Method | Baseline type | Clustering | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg grids |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGCP-CV (ours) | proposed | cov_coalition_game | 0.87 | 0.80 | 0.36 | 60.18 | 0.72 | 60.90 | 536.92 | 2.67 | 85.73 |
| Random balanced | heuristic | random_balanced | 0.79 | 0.64 | 0.30 | 55.08 | 0.73 | 55.81 | 447.47 | 3.00 | 67.64 |
| Distance-greedy | heuristic | distance_greedy | 0.80 | 0.69 | 0.35 | 56.32 | 0.58 | 56.90 | 447.48 | 3.00 | 70.47 |
| Density/quality-greedy | heuristic | density_greedy_cluster | 0.73 | 0.62 | 0.29 | 46.25 | 0.75 | 47.00 | 447.40 | 3.00 | 61.00 |
| SeAC-inspired | paper baseline | seac_social_adaptive | 0.82 | 0.72 | 0.36 | 56.30 | 0.63 | 56.93 | 447.48 | 3.00 | 69.97 |
| HHOCNET-inspired | paper baseline | hho_vanet | 0.79 | 0.68 | 0.32 | 54.60 | 0.68 | 55.28 | 447.47 | 3.00 | 68.20 |

Paper-baseline sources and adaptation:
- SeAC-inspired maps Akbar et al., `SeAC: SDN-Enabled Adaptive Clustering Technique for Social-Aware Internet of Vehicles`, IEEE Transactions on Intelligent Transportation Systems, 24(5):4827-4835, 2023, DOI `10.1109/TITS.2023.3237321`, to CARLA-side direction, relative-speed, distance and sensing-overlap proxies.
- HHOCNET-inspired maps Ali et al., `Harris Hawks Optimization-Based Clustering Algorithm for Vehicular Ad-Hoc Networks`, IEEE Transactions on Intelligent Transportation Systems, 24(6):5822-5841, 2023, DOI `10.1109/TITS.2023.3257484`, to deterministic multi-start partition search over proximity, relative mobility and sensing coverage.

Notes:
- `Total Mbps = Raw Mbps + Box Mbps`; all rows include the same inter-cluster detection-box communication accounting.
- This table is a clustering comparison, not a protocol-native baseline table. Resource scheduling, late fusion, checkpoint, channel estimator and communication accounting are fixed.
- `Avg source CAVs` and `Avg grids` are diagnostic columns kept in the experiment package to explain payload/AP changes; they can be removed from the camera-ready table if space is tight.
