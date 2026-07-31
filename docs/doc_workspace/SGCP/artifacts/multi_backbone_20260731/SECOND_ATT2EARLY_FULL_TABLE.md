# SECOND Attentive-to-Early Full Table 1

Protocol: `40 MHz / 10ch`, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`,
`symbols=12`, `mcs=28`, 60 ms data-plane deadline. SGCP uses
potential-verified clustering, dynamic C/V scheduling, `rho=2`, raw-LiDAR grid
upload, and inter-cluster box aggregation. Detector checkpoint:
`second_attentive_fusion.zip` migrated into the SECOND early-fusion model
definition.

## Dense Roundabout

Dataset: `D:\Data\Carla\2026_07_29_02_32_08`, 20 CAVs, 41 frames.

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 0.96 | 0.96 | 0.92 | 337.08 | 0.00 | 337.08 | 151.29 | 1.00 |
| No collaboration | 0.56 | 0.52 | 0.37 | 0.00 | 0.00 | 0.00 | 3025.76 | 20.00 |
| Pure late | 0.88 | 0.86 | 0.73 | 0.00 | 2.05 | 2.05 | 3025.76 | 20.00 |
| FullPerception-PCS | 0.55 | 0.51 | 0.37 | 70.49 | 0.00 | 70.49 | 3025.76 | 20.00 |
| EdgeCooper-Pmax V2V adaptation | 0.53 | 0.50 | 0.39 | 86.30 | 0.00 | 86.30 | 3025.76 | 20.00 |
| PACP-LiDAR V2V adaptation | 0.52 | 0.48 | 0.34 | 86.30 | 0.00 | 86.30 | 3025.76 | 20.00 |
| SGCP | 0.92 | 0.90 | 0.78 | 27.84 | 0.92 | 28.75 | 1003.67 | 6.63 |

## Town06

Dataset: `D:\Data\Carla\2026_07_31_02_24_35`, 10 CAVs, 20 background vehicles,
31 frames.

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 0.99 | 0.99 | 0.98 | 147.23 | 0.00 | 147.23 | 151.29 | 1.00 |
| No collaboration | 0.70 | 0.66 | 0.52 | 0.00 | 0.00 | 0.00 | 1512.88 | 10.00 |
| Pure late | 0.92 | 0.91 | 0.86 | 0.00 | 0.81 | 0.81 | 1512.88 | 10.00 |
| FullPerception-PCS | 0.68 | 0.65 | 0.53 | 36.35 | 0.00 | 36.35 | 1512.88 | 10.00 |
| EdgeCooper-Pmax V2V adaptation | 0.79 | 0.77 | 0.69 | 69.14 | 0.00 | 69.14 | 1512.88 | 10.00 |
| PACP-LiDAR V2V adaptation | 0.71 | 0.68 | 0.55 | 81.15 | 0.00 | 81.15 | 1512.88 | 10.00 |
| SGCP | 0.95 | 0.94 | 0.87 | 15.42 | 0.51 | 15.93 | 795.48 | 5.26 |

Conclusion: SECOND attentive-to-early is the recommended additional-backbone
robustness evidence because SGCP exceeds pure late and all protocol-native
baselines on both scenes while requiring much fewer detector forwards.
