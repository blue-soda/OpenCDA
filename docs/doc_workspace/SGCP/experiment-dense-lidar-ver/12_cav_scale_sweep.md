# Active-CAV Scale Sweep

Status: updated with `--gt-scope full-frame`, using the previous `dynamic_cv` SGCP scheduler. This table is intended as dense-network scalability evidence for SGCP, but should be rerun if the hybrid scheduler becomes the final paper method.

Protocol: Dense Roundabout / `v2xp_cluster_carla_dense`, PointPillar-Attentive early checkpoint, 41 frames (`000060`-`000140`), 40 MHz / 10 target subchannels, NS3-calibrated channel estimator (`tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`), and explicit full-frame GT evaluation via:

```text
--gt-scope full-frame
```

For `N=5/10/15`, the evaluator uses `--cav-count N`. This creates an active-CAV subset: only the first `N` sorted CAV ids are loaded as cooperative/evaluated agents, while non-selected CAVs do not participate as receivers, helpers, schedulers, or detector calls. Within each loaded active-CAV frame, `full-frame` GT aggregates all loaded CAV annotations and projects them into the evaluated receiver/target coordinate system, so the AP denominator is independent of the selected helper set and receiver sample count for a fixed `N`.

For the newly remeasured `N=5/10/15` SGCP rows, SGCP uses the derived cluster-size cap

```text
N_max = ceil(N / floor(K_ch / B_h))
```

with `K_ch=10` target subchannels and `B_h=2` per-head receive budget. Therefore `N_max=1/2/3` for `N=5/10/15`, respectively. This prevents small-N settings from collapsing into too few cluster heads and keeps the number of late-fusion participants matched to the available orthogonal receive capacity.

`N=20` reuses the previous Dense Roundabout full-frame-GT Table-1 run instead of rerunning the same main-table protocol. That row uses `dynamic_cv` and `N_max=5`; a formula-consistency check with `N_max=4` is stored in the provenance directory but is not substituted into the main table pending a separate main-table protocol decision.

## Main Scale Table

| Active CAVs | Method | SGCP `N_max` | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Detector calls/frame |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | Pure late | - | 0.72 | 0.64 | 0.42 | 0.00 | 0.57 | 0.57 | 447.19 | 5.00 |
| 5 | EdgeCooper-Pmax | - | 0.42 | 0.39 | 0.27 | 12.75 | 0.00 | 12.75 | 715.54 | 8.00 |
| 5 | SGCP | 1 | 0.72 | 0.64 | 0.42 | 0.00 | 0.57 | 0.57 | 447.19 | 5.00 |
| 10 | Pure late | - | 0.85 | 0.78 | 0.54 | 0.00 | 1.22 | 1.22 | 894.37 | 10.00 |
| 10 | EdgeCooper-Pmax | - | 0.55 | 0.50 | 0.33 | 59.65 | 0.00 | 59.65 | 1610.06 | 18.00 |
| 10 | SGCP | 2 | 0.85 | 0.77 | 0.52 | 15.18 | 0.90 | 16.08 | 523.58 | 5.85 |
| 15 | Pure late | - | 0.86 | 0.77 | 0.52 | 0.00 | 1.57 | 1.57 | 1297.92 | 14.51 |
| 15 | EdgeCooper-Pmax | - | 0.48 | 0.44 | 0.29 | 83.82 | 0.00 | 83.82 | 2497.96 | 27.93 |
| 15 | SGCP | 3 | 0.84 | 0.77 | 0.56 | 25.75 | 0.87 | 26.62 | 536.70 | 6.00 |
| 20 | Pure late | - | 0.82 | 0.74 | 0.53 | 0.00 | 1.95 | 1.95 | 1745.11 | 19.51 |
| 20 | EdgeCooper-Pmax | - | 0.40 | 0.37 | 0.25 | 86.30 | 0.00 | 86.30 | 3497.04 | 39.10 |
| 20 | SGCP | 5 | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 6.63 |

## Interpretation

- SGCP uses the channel-derived `N_max` rule for the newly remeasured small-N settings, so small-N settings keep enough cluster heads instead of collapsing into one large cluster. For `N=5`, the rule gives `N_max=1`, and SGCP naturally degenerates to pure late with the same AP, Mbps, and detector calls.
- From `N=10` onward, SGCP matches or nearly matches pure-late AP@0.5 while using much lower compute and only modest communication. At `N=20`, SGCP exceeds pure late on AP@0.3/AP@0.5/AP@0.7 while reducing detector calls from `19.51` to `6.63` per frame.
- EdgeCooper-Pmax consumes substantially more raw-LiDAR payload under the all-receiver protocol but is not competitive in this full-frame-GT scale setting. This is consistent with the main dense Table 1 finding that raw-LiDAR receiver-local baseline adaptations can become precision-limited under dense point clouds.
- SGCP's compute grows sublinearly after `N=10`: detector calls stay close to the number of feasible cluster heads, while pure late grows roughly with the number of active CAVs.
- A separate `N=20, N_max=4` formula-consistency check gives `0.85/0.79/0.59`, `29.24` Total Mbps, `551.98` GFLOPs/frame, and `6.17` detector calls/frame. The current main-table row is kept at `N_max=5` because that is the already reported Table-1 protocol; changing it should be handled as a main-table protocol update, not as part of this small-N sweep.

## Plot Suggestion

Recommended figure: two-panel line plot.

- Left: AP@0.5 vs active CAV count.
- Right: GFLOPs/frame vs active CAV count, optionally with marker size or annotation for Total Mbps.

Use the same color for each method across both panels. This figure directly supports the scalability claim: SGCP preserves high AP in denser cooperative settings without the detector-call growth of pure late and without the high raw-LiDAR payload of EdgeCooper-Pmax.

## Provenance

| Quantity | Source |
| --- | --- |
| N=5/10/15 Pure late and EdgeCooper-Pmax AP/trace | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_cav_sweep_fullgt_20260801\N5`, `N10`, `N15` |
| N=5/10/15 SGCP formula-`N_max` AP/trace | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_cav_sweep_fullgt_nmax_formula_20260801\N5`, `N10`, `N15` |
| N=5/10/15 Pure late and EdgeCooper-Pmax compute | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_cav_sweep_fullgt_20260801\compute.csv` |
| N=5/10/15 SGCP formula-`N_max` compute | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_cav_sweep_fullgt_nmax_formula_20260801\compute_sgcp_formula.csv` |
| N=20 AP and trace | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_pointpillar_fullgt_20260731\table1_41f` |
| N=20 compute | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_pointpillar_fullgt_20260731\table1_41f\compute_all_41f.csv` |
| N=20 formula-`N_max=4` consistency check | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_cav_sweep_fullgt_nmax_formula_20260801\N20` |

Commands use the following template. Replace `N` with `5/10/15` and replace `N_MAX` with `1/2/3`, respectively.

```text
conda run --no-capture-output -n opencda python docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\run_table1_scene_backbone.py \
  --scenario-id 2026_07_29_02_32_08 \
  --max-frames 41 \
  --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml \
  --n-max N_MAX \
  --sgcp-budget-mbps 40 \
  --cav-count N \
  --local-preserving-output \
  --gt-scope full-frame \
  --only pure_late,edgecooper_pmax,sgcp
```
