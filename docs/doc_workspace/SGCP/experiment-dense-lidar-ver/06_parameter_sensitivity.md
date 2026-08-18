# Dense SGCP Parameter Sensitivity

This file keeps the dense-LiDAR SGCP hyperparameter evidence that is useful for the paper. Channel-count sensitivity and `N_max` sensitivity are intentionally excluded: the paper-facing network setting is fixed to `40 MHz / 10ch`, and `N_max` is derived rather than tuned.

Protocol unless varied: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), attentive-derived detector, `potential_verified_cov_coalition_game` clustering, `dynamic_cv` dynamic C/V scheduler with density-capped deterministic random point upload, all cluster heads as receivers, raw-LiDAR grid upload, inter-cluster box NMS, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, Guard `1 ms`, zero-time send delay `0 ms`, and activation-synchronized communication.

The only retained hyperparameter sweep is the point-cloud density threshold `rho_th`. For each `rho_th`, the raw-LiDAR budget is increased until the row is either communication-saturated or AP-converged. The best saturated operating point under the previous `dynamic_cv` scheduler is `rho_th=2`, AP `0.86/0.82/0.59`, total payload `28.69 Mbps`.

Status note: Table 1 and Table 3 now include a stronger hybrid scheduler probe at the same `rho_th=2` point (`0.88/0.82/0.61`, `29.31 Mbps`). If hybrid is selected as the final SGCP scheduler, this rho-Pareto sweep should be rerun with hybrid rather than inherited from `dynamic_cv`.

## Rho Pareto Sweep

| rho_th | Raw LiDAR Mbps | Total Mbps | AP@0.3 | AP@0.5 | AP@0.7 | GFLOPs/frame | Avg source CAVs | Avg selected grids |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.68 | 0.80 | 0.72 | 0.50 | 606.84 | 1.15 | 1.76 |
| 1 | 5.00 | 5.81 | 0.83 | 0.77 | 0.55 | 606.87 | 1.59 | 18.76 |
| 1 | 10.00 | 10.87 | 0.82 | 0.75 | 0.54 | 609.09 | 2.01 | 33.47 |
| 1 | 16.61 | 17.52 | 0.85 | 0.79 | 0.57 | 606.96 | 2.47 | 58.13 |
| 2 | 1.00 | 1.67 | 0.79 | 0.71 | 0.48 | 593.75 | 1.28 | 0.91 |
| 2 | 5.00 | 5.71 | 0.82 | 0.74 | 0.51 | 593.77 | 1.52 | 9.22 |
| 2 | 10.00 | 10.78 | 0.83 | 0.78 | 0.55 | 593.81 | 1.80 | 17.80 |
| 2 | 20.00 | 20.83 | 0.83 | 0.78 | 0.56 | 593.89 | 2.15 | 30.99 |
| 2 | 27.84 | 28.69 | 0.86 | 0.82 | 0.59 | 593.43 | 2.51 | 47.89 |
| 3 | 1.00 | 1.66 | 0.79 | 0.71 | 0.48 | 593.75 | 1.22 | 0.56 |
| 3 | 4.99 | 5.68 | 0.80 | 0.72 | 0.49 | 593.77 | 1.35 | 4.22 |
| 3 | 9.99 | 10.71 | 0.83 | 0.77 | 0.54 | 593.81 | 1.54 | 8.82 |
| 3 | 19.99 | 20.76 | 0.84 | 0.79 | 0.56 | 593.89 | 1.93 | 17.13 |
| 3 | 35.57 | 36.40 | 0.85 | 0.80 | 0.58 | 594.00 | 2.51 | 32.84 |
| 4 | 0.97 | 1.64 | 0.79 | 0.71 | 0.48 | 593.74 | 1.17 | 0.33 |
| 4 | 4.99 | 5.66 | 0.78 | 0.71 | 0.48 | 593.77 | 1.30 | 2.73 |
| 4 | 9.99 | 10.68 | 0.79 | 0.72 | 0.50 | 593.81 | 1.49 | 6.32 |
| 4 | 19.98 | 20.71 | 0.81 | 0.76 | 0.55 | 593.89 | 1.82 | 11.35 |
| 4 | 39.35 | 40.13 | 0.83 | 0.78 | 0.58 | 594.03 | 2.49 | 23.31 |
| 4 | 40.10 | 40.89 | 0.83 | 0.78 | 0.58 | 594.04 | 2.51 | 24.05 |
| 5 | 0.98 | 1.64 | 0.79 | 0.71 | 0.49 | 593.75 | 1.15 | 0.30 |
| 5 | 4.98 | 5.65 | 0.79 | 0.71 | 0.48 | 593.77 | 1.28 | 2.06 |
| 5 | 9.97 | 10.66 | 0.79 | 0.73 | 0.49 | 593.81 | 1.48 | 4.61 |
| 5 | 19.97 | 20.70 | 0.81 | 0.76 | 0.53 | 593.89 | 1.81 | 8.74 |
| 5 | 39.97 | 40.74 | 0.82 | 0.78 | 0.57 | 594.04 | 2.39 | 16.72 |
| 5 | 43.97 | 44.74 | 0.83 | 0.78 | 0.58 | 594.07 | 2.51 | 19.54 |

Interpretation:

- `rho_th=2` gives the best dense Pareto point: it reaches the highest AP@0.3/AP@0.5 among deadline-feasible rows while using only `28.69 Mbps` total payload.
- Higher `rho_th` values continue to admit more raw payload, but AP converges below the `rho_th=2` saturated point. This supports using `rho_th=2` rather than treating the largest communication row as best.
- Lower `rho_th=1` saturates at a smaller payload because density-capped grid upload reaches its per-grid cap earlier. It remains close, but its best AP is lower than `rho_th=2`.
- `N_max` is not swept. SGCP derives it from channel concurrency:

```text
N_max = ceil(N / floor(K / B_h))
```

For the dense main protocol, `N=20`, `K=10`, and `B_h=2`, giving `N_max=4`.

## Plot Suggestion

Draw `Total Mbps` on the x-axis and AP on the y-axis, with one line per `rho_th`. Use AP@0.3 and AP@0.5 in the main figure; AP@0.7 can be retained in the source table for completeness. Mark the selected paper-facing point at `rho_th=2`, total `28.69 Mbps`.
