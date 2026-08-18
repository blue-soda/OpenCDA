# Dense-LiDAR SGCP Hyperparameter Results

Status: dense SGCP hyperparameter record. The paper-facing dense SGCP operating
point is selected by the rho Pareto sweep in `06_parameter_sensitivity.md`.

## Protocol

- Dataset: `D:\Data\Carla\2026_07_29_02_32_08`.
- Scenario: `v2xp_cluster_carla_dense`.
- CAVs: 20.
- Frames: 41 per CAV, `000060` to `000140`.
- LiDAR: 32 channels, 320000 points/s, 20 Hz rotation, 50 m range.
- Detector: attentive-derived early-fusion checkpoint, shared by all dense
  protocol rows for detector fairness.
- Proposed method candidate: SGCP with `potential_verified_cov_coalition_game`
  clustering and `hybrid_round_robin_dynamic_marginal` scheduling with
  density-capped deterministic random point upload. The previous `dynamic_cv`
  scheduler row is retained in Table 1/3 as provenance.
- Fusion: all cluster heads as receivers, grid raw-LiDAR upload, inter-cluster
  box NMS.
- 40 MHz setting: 10 target subchannels, NS3 estimator `tb_size=899 B`,
  `slot=0.5 ms`, `symbols=12`, `mcs=28`.
- Control-plane setting for the dense series: Guard=1 ms, zero-time send delay=0
  ms, and pre-send activation synchronization.

## Derived Coalition Capacity

`N_max` is no longer treated as a tunable hyperparameter. It is derived from the
number of CAVs and the number of available subchannels:

```text
M = floor(K / B_h)
N_max = ceil(N / M) = ceil(N / floor(K / B_h))
```

where `N` is the number of CAVs, `K` is the number of target subchannels, `B_h`
is the per-head receive budget, and `M` is the expected number of simultaneously
active coalitions. With the dense main setting `N=20`, `K=10`, and `B_h=2`, the
derived value is:

```text
M = floor(10 / 2) = 5
N_max = ceil(20 / 5) = 4
```

This removes the need for an `N_max` sweep. Existing dense measurements that
were already run with a fixed cap report that cap only as provenance; future
paper-facing reruns should use the derived rule.

## Dataset Check

| Metric | Value |
| --- | ---: |
| CAV folders | 20 |
| PCD frames per CAV | 41 |
| Total PCD files | 820 |
| Frame range | `000060` to `000140` |
| First-frame points per CAV | about 12973-14331 |

The dense export directly tests the effect of higher LiDAR density on SGCP's
coverage, precision, and communication behavior.

## Selected Dense Operating Point

The selected row uses density threshold `rho_th=2`. It is the best saturated
Pareto point among the retained dense rho sweep and remains below the 60 ms
data-plane deadline.

| Setting | Bandwidth | TB size | Deadline | rho_th | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | Mean link time | P95 link time | Max link time | Avg source CAVs | Avg selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP hybrid headline candidate | 40 MHz | 899 B | 60 ms | 2 | 0.88 | 0.82 | 0.61 | 28.42 | 0.89 | 29.31 | 35.68 ms | 37.39 ms | 37.47 ms | 2.51 | 49.12 |
| SGCP previous dynamic_cv row | 40 MHz | 899 B | 60 ms | 2 | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 35.76 ms | 37.46 ms | 37.47 ms | 2.51 | 47.89 |

Interpretation:

- Dense LiDAR substantially improves AP@0.7 while the density threshold keeps
  communication far below the 60 ms data-plane deadline.
- The retained rho Pareto sweep shows that `rho_th=2` gives the strongest
  deadline-feasible AP/communication tradeoff. Higher rho values admit more
  payload but do not exceed this AP point.
- The hybrid scheduler improves the selected rho-2 operating point without
  changing the density threshold or detector. If the hybrid row is promoted to
  the final method, the rho Pareto table should be rerun with the hybrid
  scheduler rather than inherited from `dynamic_cv`.
- `N_max` is now derived by formula and is no longer swept as a hyperparameter.
  The previous `N_max` diagnostic table has therefore been removed from this
  clean package.
  Note that the current hybrid probe and previous dense headline were measured
  with `N_max=5`; the formula-derived `N_max=4` row should be rerun if the paper
  keeps the derived-capacity rule as a strict protocol statement.

## Dense Table Status

- Table 1 protocol-native baselines and GFLOPs: `01_protocol_native_baselines.md`.
- Table 2/3/4 have been rerun at the same rho-2 operating point.
