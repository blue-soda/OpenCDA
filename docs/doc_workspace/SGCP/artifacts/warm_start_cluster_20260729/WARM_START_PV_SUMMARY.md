# PV Coalition Warm-Start Convergence

- dataset root: `D:\Data\Carla`
- scenario: `2026_07_15_01_26_56`
- frames: `41`
- parameters: `N_max=4`, `rho_th=3.0`, `T_min_stab=0.1 s`

## Summary

| Metric | mean | median | p95 | max | min |
|---|---:|---:|---:|---:|---:|
| Cold-start rounds | 2.80 | 3.00 | 3.00 | 3.00 | 2.00 |
| Warm-start rounds, steady-state | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Warm-start accepted migrations, steady-state | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Warm-start potential checks, steady-state | 52.00 | 52.00 | 52.00 | 52.00 | 52.00 |
| Warm-start elapsed ms, steady-state | 56.45 | 55.36 | 62.62 | 71.52 | 50.47 |

- steady-state frames with one round: `40/40`
- steady-state frames with zero accepted migration: `40/40`
- warm-start final partition equals cold-start final partition: `1/41` frames

Raw per-frame records are in `warm_start_pv_convergence.csv`.
