# Dense SGCP Dynamic C/V Parameter Sensitivity

Protocol: dense `v2xp_cluster_carla` dump `2026_07_29_02_32_08`, 20 CAVs, 41 frames, attentive checkpoint, `40 MHz / 10 ch`, NS3-calibrated estimator (`tb_size=899`, `slot=0.5 ms`, `symbols=12`, `MCS=28`), `potential_verified_cov_coalition_game` clustering, `dynamic_cv` resource allocation, all cluster heads as receivers, grid upload, inter-cluster box NMS. `rho_th` and upload density cap use the same value; receiver-side residual density is updated after each admitted grid upload.

No explicit `N_max` is passed; for this protocol it follows `N_max = ceil(N / floor(K / B_h)) = ceil(20 / floor(10 / 2)) = 4`.

## rho_th Sweep

| Setting | rho_th | Raw budget (Mbps) | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.1 | 68 | 0.79 | 0.71 | 0.48 | 2.07 | 0.76 | 2.84 | 604.67 | 2.48 | 33.55 |
| 0.2 | 0.2 | 68 | 0.80 | 0.73 | 0.50 | 3.97 | 0.83 | 4.79 | 619.96 | 2.44 | 39.44 |
| 0.5 | 0.5 | 68 | 0.83 | 0.76 | 0.55 | 9.27 | 0.89 | 10.16 | 620.00 | 2.44 | 49.47 |
| 1 | 1 | 68 | 0.85 | 0.79 | 0.58 | 16.61 | 0.91 | 17.52 | 606.96 | 2.47 | 58.13 |
| 2 | 2 | 68 | 0.86 | 0.81 | 0.58 | 28.42 | 0.87 | 29.28 | 593.95 | 2.51 | 47.94 |
| 3 | 3 | 68 | 0.85 | 0.80 | 0.58 | 35.57 | 0.82 | 36.40 | 594.00 | 2.51 | 32.84 |
| 4 | 4 | 68 | 0.83 | 0.78 | 0.58 | 40.10 | 0.79 | 40.89 | 594.04 | 2.51 | 24.05 |
| 5 | 5 | 68 | 0.83 | 0.78 | 0.58 | 43.97 | 0.77 | 44.74 | 594.07 | 2.51 | 19.54 |
| 10 | 10 | 68 | 0.80 | 0.75 | 0.55 | 49.57 | 0.72 | 50.30 | 594.11 | 2.51 | 8.96 |

