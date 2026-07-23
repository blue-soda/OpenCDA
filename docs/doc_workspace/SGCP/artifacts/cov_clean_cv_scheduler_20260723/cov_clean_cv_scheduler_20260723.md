# Clean C/V Scheduler Result - 2026-07-23

## Why This Run Exists

The previous COV scheduler mixed the intended C/O/V components with extra
PAPG-style engineering priors, including top-utility and connected-component
terms. That made the scheduler-side O/V ablation ambiguous. This run replaces
that path with a clean two-stage scheduler:

- Coalition formation remains V-only.
- Candidate grids are exactly grids with `C + V > 0`.
- Coverage stage scores links only by selected-grid `C`.
- Target/quality stage scores links only by selected-grid `V`.
- Connected-component and top-utility priors are not used.

## Utility

Let `q_i(g)=min(1, rho_i(g)/rho_th)` be the sender quality and `q_h(g)` be the
cluster-head quality for grid `g`.

```text
C(i,g|h) = q_i(g) * (1 - q_h(g))
V(i,g|h) = q_i(g) if q_h(g) > 0 else 0
```

For each candidate sender-link, the scheduler first builds:

```text
G_cand = {g | C(i,g|h) + V(i,g|h) > 0}
```

Then:

```text
Coverage stage: G_sel = top-K grids in G_cand by C, score = sum C
Target stage:   G_sel = top-K grids in G_cand by V, score = sum V
```

`K = max_grids_per_rb` under the current NS3-calibrated channel estimator.

## Protocol

- Dataset: `D:\Data\Carla`, scenario `2026_07_15_01_26_56`, 41 frames
- Detector: attentive config default
- Clustering: `cov_coalition_game`, default V-only
- Scheduler: cleaned `cov_potential_game`
- Receiver policy: all cluster heads
- Fusion: raw-LiDAR early fusion per cluster head + inter-cluster NMS
- Network: 40 MHz, 10 target subchannels, NS3 estimator defaults `tb_size=899`, `symbols=12`, `MCS=28`
- SGCP: `N_max=4`, `rho_th=3`, `head_rb_budget=2`, scheduler admission budget `200 ms`

## Result

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Cluster-head samples | Uploaded senders/sample | Source CAVs/sample | Selected grids/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean C/V scheduler | 0.87 | 0.80 | 0.36 | 60.18 | 246 | 1.67 | 2.67 | 85.73 |

Interpretation: the cleaned scheduler remains close to the PAPG/COV headline
result while removing hidden connected-component and top-utility priors. It is
slightly weaker than the previous mixed COV implementation (`0.87/0.81/0.37`),
but the mechanism is now suitable for clean paper exposition and future
ablation.
