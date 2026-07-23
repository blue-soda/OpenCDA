# SGCP Parameter Sensitivity

Protocol: attentive detector, v2xp_cluster_carla, 41 frames, 20 CAVs, 40 MHz total bandwidth, NS3-calibrated estimator, formal SGCP C/V algorithm, cov_coalition_game V-only clustering, cov_potential_game scheduler with C coverage stage followed by V target stage, all cluster heads as receivers, grid upload, inter-cluster box NMS. Unless varied, N_max=4, rho_th=3, head_rb_budget=2, target subchannels=10, and raw-LiDAR payload is not artificially capped. The headline point is retained because exact NS3-calibrated trace estimates stay below the 60 ms communication portion of the 100 ms perception cycle. Raw LiDAR Mbps Budget caps scheduled grid payload per 100 ms perception frame after scheduling; box-level NMS payload is still added separately in Total Mbps.

Box-level communication for inter-cluster NMS is included in total Mbps.

## rho_th

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.74 | 0.61 | 0.24 | 32.17 | 0.57 | 32.74 | 562.88 | 2.59 | 53.83 |
| 0.003 | 0.74 | 0.61 | 0.24 | 32.17 | 0.57 | 32.74 | 562.88 | 2.59 | 53.83 |
| 0.005 | 0.74 | 0.61 | 0.24 | 32.17 | 0.57 | 32.74 | 562.88 | 2.59 | 53.83 |
| 0.01 | 0.74 | 0.61 | 0.24 | 32.17 | 0.57 | 32.74 | 562.88 | 2.59 | 53.83 |
| 0.03 | 0.76 | 0.62 | 0.26 | 33.06 | 0.60 | 33.65 | 571.61 | 2.56 | 55.88 |
| 0.05 | 0.78 | 0.66 | 0.29 | 34.04 | 0.62 | 34.66 | 580.34 | 2.54 | 56.68 |
| 0.1 | 0.83 | 0.71 | 0.32 | 38.53 | 0.66 | 39.18 | 580.37 | 2.54 | 61.72 |
| 0.3 | 0.85 | 0.78 | 0.36 | 53.17 | 0.65 | 53.83 | 547.77 | 2.63 | 74.93 |
| 0.5 | 0.86 | 0.78 | 0.37 | 54.69 | 0.67 | 55.37 | 547.78 | 2.63 | 78.50 |
| 1 | 0.86 | 0.78 | 0.36 | 57.15 | 0.71 | 57.86 | 536.90 | 2.67 | 82.36 |
| 2 | 0.87 | 0.80 | 0.36 | 59.65 | 0.72 | 60.37 | 536.91 | 2.67 | 84.89 |
| 3 | 0.87 | 0.80 | 0.36 | 60.18 | 0.72 | 60.90 | 536.92 | 2.67 | 85.73 |

## N_max

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.83 | 0.72 | 0.31 | 52.88 | 0.96 | 53.84 | 903.23 | 1.94 | 52.33 |
| 3 | 0.88 | 0.79 | 0.33 | 60.34 | 0.80 | 61.14 | 626.33 | 2.31 | 68.72 |
| 4 | 0.87 | 0.80 | 0.36 | 60.18 | 0.72 | 60.90 | 536.92 | 2.67 | 85.73 |
| 5 | 0.85 | 0.77 | 0.33 | 57.11 | 0.70 | 57.81 | 536.89 | 2.67 | 85.32 |
| 6 | 0.85 | 0.77 | 0.33 | 57.11 | 0.70 | 57.81 | 536.89 | 2.67 | 85.32 |

## Target Subchannels

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.71 | 0.59 | 0.24 | 30.20 | 0.57 | 30.76 | 536.69 | 1.83 | 47.33 |
| 10 | 0.87 | 0.80 | 0.36 | 60.18 | 0.72 | 60.90 | 536.92 | 2.67 | 85.73 |
| 20 | 0.87 | 0.81 | 0.35 | 65.61 | 0.73 | 66.34 | 536.96 | 2.81 | 92.50 |

## Raw LiDAR Mbps Budget

| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.50 | 0.36 | 0.16 | 1.00 | 0.46 | 1.46 | 536.48 | 1.17 | 0.76 |
| 5 | 0.52 | 0.38 | 0.17 | 5.00 | 0.47 | 5.47 | 536.51 | 1.17 | 2.32 |
| 10 | 0.60 | 0.46 | 0.21 | 10.00 | 0.50 | 10.50 | 536.54 | 1.35 | 10.61 |
| 20 | 0.70 | 0.58 | 0.28 | 20.00 | 0.57 | 20.57 | 536.62 | 1.67 | 26.41 |
| 40 | 0.81 | 0.73 | 0.34 | 40.00 | 0.67 | 40.67 | 536.77 | 2.17 | 51.32 |
| 60 | 0.87 | 0.80 | 0.35 | 59.43 | 0.72 | 60.15 | 536.91 | 2.67 | 81.61 |
| 100 | 0.87 | 0.80 | 0.36 | 60.18 | 0.72 | 60.90 | 536.92 | 2.67 | 85.73 |
| 200 | 0.87 | 0.80 | 0.36 | 60.18 | 0.72 | 60.90 | 536.92 | 2.67 | 85.73 |

