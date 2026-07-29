# Dense 120 Mbps rho_th Diagnostic

Protocol: dense 41-frame replay, SGCP, `N_max=5`, 40 MHz / 10ch, NS3 estimator `tb=899`, raw-LiDAR frame budget `120 Mbps`.

Communication deadline: `200.0 ms`.

| rho_th | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Avg source CAVs | Avg selected grids | Max link time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.84 | 0.79 | 0.56 | 56.90 | 1.00 | 41.56 | 127.38 ms |
| 1 | 0.85 | 0.81 | 0.59 | 73.64 | 1.00 | 47.14 | 127.53 ms |
| 2 | 0.86 | 0.81 | 0.59 | 82.21 | 1.00 | 52.24 | 127.53 ms |
| 3 | 0.87 | 0.82 | 0.57 | 87.97 | 1.00 | 54.90 | 127.53 ms |
| 5 | 0.87 | 0.82 | 0.61 | 93.75 | 1.00 | 44.00 | 127.00 ms |

This is a diagnostic upper-budget sweep. It should not replace the deadline-feasible dense main table unless a later NS3 timing protocol is also changed.
