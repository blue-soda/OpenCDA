# Dense-LiDAR Table 4. Clustering Ablation under the SGCP Protocol

Protocol: dense `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames, attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, `rho_th=2`, `upload_density_cap_rho=2`, 40 Mbps raw admission budget, all cluster heads as receivers, grid upload, and inter-cluster box NMS. Only the clustering algorithm changes within each section.

Status note: hybrid scheduler rows have been added while retaining the earlier `dynamic_cv` rows as provenance. If the paper promotes hybrid as final SGCP, use the hybrid section as the primary clustering ablation.

## Hybrid Scheduler Clustering Ablation

Scheduler fixed to `hybrid_round_robin_dynamic_marginal`.

| Method | Baseline type | Clustering | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg grids | P95 data time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGCP hybrid | proposed | potential_verified_cov_coalition_game | 0.88 | 0.82 | 0.61 | 28.42 | 0.89 | 29.31 | 593.34 | 2.51 | 49.12 | 37.45 ms |
| Random balanced + hybrid | heuristic | random_balanced | 0.84 | 0.76 | 0.50 | 39.64 | 0.85 | 40.49 | 357.75 | 3.00 | 64.00 | 43.15 ms |
| Distance-greedy + hybrid | heuristic | distance_greedy | 0.83 | 0.79 | 0.57 | 30.73 | 0.71 | 31.45 | 357.75 | 3.00 | 65.82 | 40.85 ms |
| Density/quality-greedy + hybrid | heuristic | density_greedy_cluster | 0.76 | 0.68 | 0.47 | 39.39 | 0.86 | 40.25 | 357.75 | 3.00 | 65.12 | 43.43 ms |
| SeAC-inspired + hybrid | paper baseline | seac_social_adaptive | 0.80 | 0.76 | 0.54 | 32.09 | 0.73 | 32.82 | 357.75 | 3.00 | 65.64 | 40.88 ms |
| HHOCNET-inspired + hybrid | paper baseline | hho_vanet | 0.83 | 0.75 | 0.50 | 38.62 | 0.83 | 39.45 | 357.75 | 3.00 | 64.10 | 42.57 ms |

Interpretation:
- Under the final-candidate hybrid scheduler, SGCP has the best AP@0.3/AP@0.5/AP@0.7 while using less payload than most heuristic and paper-inspired clustering baselines.
- Heuristic clustering rows converge to four cluster heads in this setting, hence lower GFLOPs than SGCP's six to seven cluster heads. This is a compute-accuracy tradeoff rather than a contradiction: SGCP deliberately keeps more high-value heads for inter-cluster box aggregation.
- Several heuristic rows exceed the 60 ms data-plane reserve less tightly in the estimator (`P95` around 41-43 ms), still below the 60 ms raw-upload budget.

## Previous Dynamic-CV Scheduler Clustering Ablation

| Method | Baseline type | Clustering | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg grids |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGCP | proposed | potential_verified_cov_coalition_game | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 2.51 | 47.89 |
| Random balanced | heuristic | random_balanced | 0.82 | 0.76 | 0.52 | 37.20 | 0.95 | 38.14 | 447.31 | 3.00 | 57.89 |
| Distance-greedy | heuristic | distance_greedy | 0.79 | 0.75 | 0.54 | 29.88 | 0.78 | 30.66 | 447.28 | 3.00 | 59.63 |
| Density/quality-greedy | heuristic | density_greedy_cluster | 0.79 | 0.74 | 0.51 | 35.79 | 0.98 | 36.77 | 447.30 | 3.00 | 52.60 |
| SeAC-inspired | paper baseline | seac_social_adaptive | 0.83 | 0.78 | 0.55 | 30.93 | 0.81 | 31.74 | 447.29 | 3.00 | 58.99 |
| HHOCNET-inspired | paper baseline | hho_vanet | 0.81 | 0.74 | 0.50 | 35.05 | 0.90 | 35.95 | 447.30 | 3.00 | 58.03 |

Paper-baseline sources and adaptation:
- SeAC-inspired maps Akbar et al., `SeAC: SDN-Enabled Adaptive Clustering Technique for Social-Aware Internet of Vehicles`, IEEE Transactions on Intelligent Transportation Systems, 24(5):4827-4835, 2023, DOI `10.1109/TITS.2023.3237321`, to CARLA-side direction, relative-speed, distance and sensing-overlap proxies.
- HHOCNET-inspired maps Ali et al., `Harris Hawks Optimization-Based Clustering Algorithm for Vehicular Ad-Hoc Networks`, IEEE Transactions on Intelligent Transportation Systems, 24(6):5822-5841, 2023, DOI `10.1109/TITS.2023.3257484`, to deterministic multi-start partition search over proximity, relative mobility and sensing coverage.

Notes:
- This table is a clustering comparison, not a protocol-native baseline table. Resource scheduling, late fusion, checkpoint, channel estimator and communication accounting are fixed.
- `Total Mbps = Raw Mbps + Box Mbps`; all rows include the same inter-cluster detection-box communication accounting.
- Hybrid provenance: traces are stored in `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\table4\`.
