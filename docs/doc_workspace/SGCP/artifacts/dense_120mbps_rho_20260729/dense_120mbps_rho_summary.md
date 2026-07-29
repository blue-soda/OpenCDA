# Dense 120 Mbps rho_th Diagnostic

Protocol: dense 41-frame replay, SGCP, `N_max=5`, 40 MHz / 10ch, NS3 estimator `tb=899`, raw-LiDAR frame budget `120 Mbps`.

Communication deadline: `default scenario time_slot`.

| rho_th | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Avg source CAVs | Avg selected grids | Max link time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.84 | 0.79 | 0.56 | 52.54 | 1.00 | 31.52 | 100.00 ms |
| 1 | 0.85 | 0.80 | 0.59 | 67.51 | 1.00 | 30.87 | 100.00 ms |
| 2 | 0.84 | 0.79 | 0.57 | 73.36 | 1.00 | 28.34 | 100.00 ms |
| 3 | 0.85 | 0.80 | 0.57 | 77.60 | 1.00 | 21.95 | 100.00 ms |
| 5 | 0.83 | 0.77 | 0.56 | 81.02 | 1.00 | 14.42 | 100.00 ms |

This is a diagnostic upper-budget sweep. It should not replace the deadline-feasible dense main table unless a later NS3 timing protocol is also changed.
