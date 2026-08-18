# SGCP Parameter Sensitivity

Protocol: attentive detector, v2xp_cluster_carla, 41 frames, 20 CAVs, 40 MHz total bandwidth, NS3-calibrated estimator, PAPG scheduler, coalition-game clustering, all cluster heads as receivers, grid upload, inter-cluster box NMS. Unless varied, N_max=4, rho_th=3, head_rb_budget=2, target subchannels=10, and scheduler communication budget=200 ms. The headline point is retained because exact NS3 replay measured sub-60 ms delivery for the selected payload.

Box-level communication for inter-cluster NMS is included in total Mbps.

`rho_th` is measured in points per square meter. With the current `10 m x 10 m` grid, `rho_th=0.01/0.03/0.05/0.10` roughly corresponds to `1/3/5/10` points in one grid. The low-density sweep is included because the offline point cloud is sparse at the grid level: in frame `000060`, representative CAVs have only about `49--60` nonzero grids out of `961`, with nonzero-density median `0.11--0.20`.

## rho_th

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.86 | 0.78 | 0.34 | 62.41 | 0.68 | 63.08 | 563.10 | 2.59 | 94.03 |
| 0.03 | 0.86 | 0.77 | 0.32 | 62.44 | 0.69 | 63.13 | 565.28 | 2.58 | 93.52 |
| 0.05 | 0.86 | 0.78 | 0.32 | 62.46 | 0.70 | 63.16 | 580.55 | 2.54 | 90.86 |
| 0.1 | 0.86 | 0.78 | 0.32 | 62.46 | 0.71 | 63.16 | 580.55 | 2.54 | 90.61 |
| 0.3 | 0.87 | 0.81 | 0.33 | 62.59 | 0.72 | 63.32 | 547.84 | 2.63 | 95.75 |
| 0.5 | 0.88 | 0.81 | 0.34 | 62.60 | 0.73 | 63.33 | 547.84 | 2.63 | 95.55 |
| 1 | 0.87 | 0.81 | 0.36 | 62.57 | 0.73 | 63.30 | 536.94 | 2.67 | 97.24 |
| 2 | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 | 2.67 | 97.25 |
| 3 | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 | 2.67 | 97.22 |

## N_max

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.85 | 0.73 | 0.32 | 60.54 | 1.00 | 61.54 | 903.29 | 1.94 | 55.04 |
| 3 | 0.88 | 0.81 | 0.36 | 62.39 | 0.80 | 63.19 | 626.35 | 2.31 | 76.55 |
| 4 | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 | 2.67 | 97.22 |
| 5 | 0.88 | 0.80 | 0.34 | 62.62 | 0.73 | 63.35 | 536.94 | 2.67 | 97.43 |
| 6 | 0.88 | 0.80 | 0.34 | 62.62 | 0.73 | 63.35 | 536.94 | 2.67 | 97.43 |

## Target Subchannels

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.74 | 0.61 | 0.24 | 31.12 | 0.57 | 31.69 | 536.70 | 1.83 | 49.38 |
| 10 | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 | 2.67 | 97.22 |
| 20 | 0.88 | 0.81 | 0.36 | 68.07 | 0.75 | 68.82 | 536.98 | 2.81 | 105.19 |

## Communication Budget

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.85 | 0.76 | 0.38 | 54.34 | 0.65 | 54.99 | 536.87 | 2.67 | 23.33 |
| 60 | 0.87 | 0.76 | 0.38 | 58.44 | 0.68 | 59.13 | 536.90 | 2.67 | 36.67 |
| 100 | 0.87 | 0.79 | 0.37 | 61.47 | 0.71 | 62.18 | 536.93 | 2.67 | 61.67 |
| 200 | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 | 2.67 | 97.22 |
| 300 | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 | 2.67 | 97.22 |

