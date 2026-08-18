# Utility Proxy vs AP@0.5

Status: supporting evidence, but currently based on the pre-hybrid Table-3 scheduler set. This file validates whether the density-derived SGCP utility is aligned with downstream AP, using existing dense Table-3 scheduler traces and AP results. It does not rerun OpenCOOD detector inference.

Hybrid note: the new `hybrid_round_robin_dynamic_marginal` row has not yet been added to this utility-correlation table. Rerun the utility extraction if hybrid is promoted as the final SGCP scheduler.

PACP note: the `PACP-LiDAR` utility row below predates the 2026-08-01 paper-aligned PACP rerun. Do not use the PACP point from this document in paper figures until the utility trace is recomputed from `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\pacp_lidar_paper_fix_20260801`.

Protocol: dense Table-3 SGCP-compatible scheduler comparison, `v2xp_cluster_carla_dense`, 20 CAVs, 41-frame AP evaluation (`000060-000140`), 11 utility validation frames sampled every 4 frames, attentive-derived detector, potential-verified SGCP clustering, inter-cluster box NMS, 40 MHz / 10 target subchannels, NS3-calibrated data plane, and 60 ms data-plane deadline.

The utility proxy follows the current paper formulation:

```text
U_early_r(g; x)
= 1 - (1 - q_r(g)) prod_{i != r} (1 - x_{i,r,g} q_i(g)).

Delta U_early_{i,r}(g | A)
= q_i(g) (1 - Q_r^A(g)).

U_late(x, z)
= sum_g max_r z_r U_early_r(g; x).
```

Here `q_i(g)` is a normalized LiDAR evidence/observability score, not detector confidence. The table only checks whether this lightweight communication-side proxy is positively aligned with the final detector AP.

| Method | Utility final | Utility gain | Dynamic marginal gain | AP@0.5 | Avg selected grids | Links/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP | 0.1078 | 0.0408 | 0.0660 | 0.81 | 317.55 | 10.00 |
| EdgeCooper-HD-Pmax | 0.0981 | 0.0312 | 0.0950 | 0.81 | 558.82 | 15.36 |
| Random budget | 0.0791 | 0.0121 | 0.0190 | 0.77 | 55.18 | 13.45 |
| FullPerception-PCS | 0.0692 | 0.0022 | 0.0096 | 0.76 | 43.73 | 3.18 |
| Density greedy | 0.0767 | 0.0097 | 0.0149 | 0.76 | 40.09 | 13.45 |
| Link-aware density | 0.0767 | 0.0097 | 0.0149 | 0.76 | 40.09 | 13.45 |
| EdgeCooper-HD | 0.0710 | 0.0040 | 0.0101 | 0.74 | 23.55 | 9.73 |
| Cluster-head late only | 0.0670 | 0.0000 | 0.0000 | 0.71 | 0.00 | 0.00 |
| PACP-LiDAR, pre-2026-08-01 fix, exclude from paper | 0.0722 | 0.0052 | 0.0070 | 0.70 | 20.00 | 10.00 |

Correlation with AP@0.5:

| Explanatory variable | Pearson with AP@0.5 | Spearman with AP@0.5 | Interpretation |
| --- | ---: | ---: | --- |
| Utility gain | 0.846 | 0.829 | Strong positive linear/rank alignment with AP. |
| Final utility | 0.846 | 0.829 | Same ordering because initial cluster-head evidence is shared across these scheduler rows. |
| Selected grid count alone | 0.782 | 0.966 | Communication volume is highly monotonic here, but explains continuous AP differences less well than utility gain. |

Interpretation: the proposed utility should be described as a communication-side evidence surrogate, not as a detector-quality proof. Its positive Pearson correlation with AP@0.5 is higher than selected-grid count alone (`0.846` vs `0.782`), showing that the density-derived utility captures more than raw communication volume. The selected-grid count has higher Spearman correlation because heavier schedulers are almost monotonically stronger in this small scheduler set; this should be reported carefully rather than overstated.

Suggested paper figure: a compact scatter plot with `Utility gain` on the x-axis and AP@0.5 on the y-axis, one labeled point per scheduler. If space is limited, the correlation table is sufficient.

Provenance:

| Artifact | Path |
| --- | --- |
| Utility script | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\build_utility_proxy_ap.py` |
| Raw method-level data | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\utility_proxy_vs_ap.csv` |
| Frame samples | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\utility_proxy_frame_samples.csv` |
| Summary JSON | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\utility_proxy_summary.json` |

## Rho-Pareto Sensitivity Check from Parameter Table

This additional check uses the rho-Pareto rows from `06_parameter_sensitivity.md`, not the scheduler-comparison rows above. It keeps SGCP fixed and varies the point-density threshold and raw-LiDAR admission budget, so it is a stronger within-method test of whether the utility proxy tracks AP changes when communication volume and selected grids both change.

Input table: `06_parameter_sensitivity.md`, SGCP rho-Pareto table. Replay artifact: `C:/Workspace/OpenCDA/docs/doc_workspace/SGCP/artifacts/utility_proxy_ap_20260731/rho_pareto_utility_proxy_vs_ap.csv`.

| rho_th | Raw budget (Mbps) | Raw LiDAR Mbps | Total Mbps | Utility gain | Final utility | AP@0.5 | Avg selected grids |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.00 | 1.68 | 0.0021 | 0.0867 | 0.72 | 1.76 |
| 1 | 5 | 5.00 | 5.81 | 0.0044 | 0.0890 | 0.77 | 18.76 |
| 1 | 10 | 10.00 | 10.87 | 0.0240 | 0.1086 | 0.75 | 33.47 |
| 1 | 20 | 16.61 | 17.52 | 0.0465 | 0.1311 | 0.79 | 58.13 |
| 2 | 1 | 1.00 | 1.67 | 0.0018 | 0.0688 | 0.71 | 0.91 |
| 2 | 5 | 5.00 | 5.71 | 0.0050 | 0.0720 | 0.74 | 9.22 |
| 2 | 10 | 10.00 | 10.78 | 0.0089 | 0.0759 | 0.78 | 17.80 |
| 2 | 20 | 20.00 | 20.83 | 0.0272 | 0.0942 | 0.78 | 30.99 |
| 2 | 40 | 27.84 | 28.69 | 0.0408 | 0.1078 | 0.82 | 47.89 |
| 3 | 1 | 1.00 | 1.66 | 0.0013 | 0.0584 | 0.71 | 0.56 |
| 3 | 5 | 4.99 | 5.68 | 0.0046 | 0.0617 | 0.72 | 4.22 |
| 3 | 10 | 9.99 | 10.71 | 0.0074 | 0.0645 | 0.77 | 8.82 |
| 3 | 20 | 19.99 | 20.76 | 0.0181 | 0.0751 | 0.79 | 17.13 |
| 3 | 40 | 35.57 | 36.40 | 0.0353 | 0.0924 | 0.80 | 32.84 |
| 4 | 1 | 0.97 | 1.64 | 0.0011 | 0.0508 | 0.71 | 0.33 |
| 4 | 5 | 4.99 | 5.66 | 0.0042 | 0.0540 | 0.71 | 2.73 |
| 4 | 10 | 9.99 | 10.68 | 0.0066 | 0.0563 | 0.72 | 6.32 |
| 4 | 20 | 19.98 | 20.71 | 0.0146 | 0.0643 | 0.76 | 11.35 |
| 4 | 40 | 39.35 | 40.13 | 0.0305 | 0.0802 | 0.78 | 23.31 |
| 4 | 60 | 40.10 | 40.89 | 0.0309 | 0.0806 | 0.78 | 24.05 |
| 5 | 1 | 0.98 | 1.64 | 0.0009 | 0.0452 | 0.71 | 0.30 |
| 5 | 5 | 4.98 | 5.65 | 0.0038 | 0.0481 | 0.71 | 2.06 |
| 5 | 10 | 9.97 | 10.66 | 0.0060 | 0.0504 | 0.73 | 4.61 |
| 5 | 20 | 19.97 | 20.70 | 0.0123 | 0.0566 | 0.76 | 8.74 |
| 5 | 40 | 39.97 | 40.74 | 0.0268 | 0.0711 | 0.78 | 16.72 |
| 5 | 60 | 43.97 | 44.74 | 0.0289 | 0.0732 | 0.78 | 19.54 |

| Explanatory variable | Pearson with AP@0.5 | Spearman with AP@0.5 |
|---|---:|---:|
| Utility gain | 0.834 | 0.908 |
| Final utility | 0.674 | 0.742 |
| Total Mbps | 0.758 | 0.868 |
| Avg selected grids | 0.814 | 0.890 |

Interpretation: across the 26 rho-budget operating points, marginal utility gain has stronger correlation with AP@0.5 than raw communication volume alone. This supports using the point-density utility as a scheduling surrogate rather than treating Mbps as the explanatory variable. Final utility is less correlated because receiver-side evidence is already partly saturated; the marginal gain is the quantity the scheduler actually optimizes.

## Combined Rho-2 Scheduler And Budget Check

This is the recommended controlled correlation check. It combines the scheduler-variant rows above with the `rho_th = 2` budget-sweep rows from `06_parameter_sensitivity.md`, because both use the formal SGCP density threshold. The saturated `rho_th = 2, budget = 40 Mbps` row is excluded here because it duplicates the SGCP scheduler-variant operating point while using a slightly different AP table rounding (`0.82` vs `0.81`).

Artifact: `C:/Workspace/OpenCDA/docs/doc_workspace/SGCP/artifacts/utility_proxy_ap_20260731/combined_rho2_scheduler_utility_proxy_vs_ap.csv`.

| Source | Row | Utility gain | Final utility | AP@0.5 |
|---|---|---:|---:|---:|
| Scheduler variant | SGCP | 0.0408 | 0.1078 | 0.81 |
| Scheduler variant | Cluster-head late only | 0.0000 | 0.0670 | 0.71 |
| Scheduler variant | FullPerception-PCS | 0.0022 | 0.0692 | 0.76 |
| Scheduler variant | Random budget | 0.0121 | 0.0791 | 0.77 |
| Scheduler variant | Density greedy | 0.0097 | 0.0767 | 0.76 |
| Scheduler variant | Link-aware density | 0.0097 | 0.0767 | 0.76 |
| Scheduler variant | PACP-LiDAR, pre-2026-08-01 fix, exclude from paper | 0.0052 | 0.0722 | 0.70 |
| Scheduler variant | EdgeCooper-HD | 0.0040 | 0.0710 | 0.74 |
| Scheduler variant | EdgeCooper-HD-Pmax | 0.0312 | 0.0981 | 0.81 |
| SGCP rho=2 budget sweep | 1 Mbps | 0.0018 | 0.0688 | 0.71 |
| SGCP rho=2 budget sweep | 5 Mbps | 0.0050 | 0.0720 | 0.74 |
| SGCP rho=2 budget sweep | 10 Mbps | 0.0089 | 0.0759 | 0.78 |
| SGCP rho=2 budget sweep | 20 Mbps | 0.0272 | 0.0942 | 0.78 |

| Explanatory variable | Pearson with AP@0.5 | Spearman with AP@0.5 |
|---|---:|---:|
| Utility gain | 0.823 | 0.826 |
| Final utility | 0.823 | 0.826 |

Interpretation: after fixing `rho_th = 2`, the utility proxy remains strongly and positively aligned with AP@0.5 across both scheduler alternatives and SGCP communication budgets. In this controlled subset, `Utility gain` and `Final utility` have the same ordering because the initial receiver evidence is effectively shared by the utility replay protocol; the paper should still emphasize `Utility gain`, since it is the marginal quantity optimized by scheduling.

## Fixed Rho-2 Utility Re-evaluation

This is a stricter version of the rho-Pareto check. The AP/Mbps rows still come from the original rho-Pareto experiments, but the utility values are recomputed on a fixed `rho_eval = 2` scale. This avoids the confound that changing `rho_th` also changes the numerical scale of `q_i(g)` and therefore the utility values.

Artifacts:

| Artifact | Path |
|---|---|
| Recompute script | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\build_rho_pareto_utility_proxy_evalrho2.py` |
| Fixed-rho Pareto CSV | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\rho_pareto_utility_proxy_evalrho2_vs_ap.csv` |
| Fixed-rho Pareto summary | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\rho_pareto_utility_proxy_evalrho2_summary.json` |
| Combined CSV | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\combined_scheduler_rhopareto_evalrho2_vs_ap.csv` |
| Combined summary | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\combined_scheduler_rhopareto_evalrho2_summary.json` |

### Fixed-Rho Pareto Rows

| rho_th used by experiment | Raw budget (Mbps) | Raw LiDAR Mbps | Total Mbps | Utility gain at rho_eval=2 | Final utility at rho_eval=2 | AP@0.5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.00 | 1.68 | 0.0021 | 0.0699 | 0.72 |
| 1 | 5 | 5.00 | 5.81 | 0.0050 | 0.0728 | 0.77 |
| 1 | 10 | 10.00 | 10.87 | 0.0204 | 0.0883 | 0.75 |
| 1 | 20 | 16.61 | 17.52 | 0.0386 | 0.1064 | 0.79 |
| 2 | 1 | 1.00 | 1.67 | 0.0018 | 0.0688 | 0.71 |
| 2 | 5 | 5.00 | 5.71 | 0.0050 | 0.0720 | 0.74 |
| 2 | 10 | 10.00 | 10.78 | 0.0089 | 0.0759 | 0.78 |
| 2 | 20 | 20.00 | 20.83 | 0.0272 | 0.0942 | 0.78 |
| 2 | 40 | 27.84 | 28.69 | 0.0408 | 0.1078 | 0.82 |
| 3 | 1 | 1.00 | 1.66 | 0.0013 | 0.0682 | 0.71 |
| 3 | 5 | 4.99 | 5.68 | 0.0044 | 0.0714 | 0.72 |
| 3 | 10 | 9.99 | 10.71 | 0.0066 | 0.0735 | 0.77 |
| 3 | 20 | 19.99 | 20.76 | 0.0180 | 0.0849 | 0.79 |
| 3 | 40 | 35.57 | 36.40 | 0.0358 | 0.1028 | 0.80 |
| 4 | 1 | 0.97 | 1.64 | 0.0010 | 0.0679 | 0.71 |
| 4 | 5 | 4.99 | 5.66 | 0.0036 | 0.0706 | 0.71 |
| 4 | 10 | 9.99 | 10.68 | 0.0051 | 0.0721 | 0.72 |
| 4 | 20 | 19.98 | 20.71 | 0.0124 | 0.0793 | 0.76 |
| 4 | 40 | 39.35 | 40.13 | 0.0283 | 0.0952 | 0.78 |
| 4 | 60 | 40.10 | 40.89 | 0.0287 | 0.0957 | 0.78 |
| 5 | 1 | 0.98 | 1.64 | 0.0007 | 0.0677 | 0.71 |
| 5 | 5 | 4.98 | 5.65 | 0.0029 | 0.0699 | 0.71 |
| 5 | 10 | 9.97 | 10.66 | 0.0043 | 0.0713 | 0.73 |
| 5 | 20 | 19.97 | 20.70 | 0.0094 | 0.0763 | 0.76 |
| 5 | 40 | 39.97 | 40.74 | 0.0230 | 0.0900 | 0.78 |
| 5 | 60 | 43.97 | 44.74 | 0.0248 | 0.0917 | 0.78 |

Correlation within the 26 fixed-rho Pareto rows:

| Explanatory variable | Pearson with AP@0.5 | Spearman with AP@0.5 |
|---|---:|---:|
| Utility gain at rho_eval=2 | 0.856 | 0.918 |
| Final utility at rho_eval=2 | 0.855 | 0.930 |
| Total Mbps | 0.758 | 0.868 |
| Avg selected grids | 0.814 | 0.890 |

### Combined Scheduler And Fixed-Rho Pareto Statistic

This combined statistic merges the scheduler-variant rows with all fixed-rho Pareto rows, excluding the saturated `rho_th = 2, budget = 40 Mbps` row because it duplicates the SGCP scheduler row under slightly different AP table rounding. This gives `n = 34` points on the same `rho_eval = 2` utility scale.

| Explanatory variable | Pearson with AP@0.5 | Spearman with AP@0.5 |
|---|---:|---:|
| Utility gain at rho_eval=2 | 0.829 | 0.869 |
| Final utility at rho_eval=2 | 0.828 | 0.880 |
| Avg selected grids | 0.504 | 0.729 |

Interpretation: after recomputing all rho-Pareto utility values on the formal `rho=2` scale and combining them with scheduler variants, the utility proxy remains strongly aligned with AP@0.5. This is stronger than using selected-grid count alone, especially in Pearson correlation (`0.829` vs `0.504`), and is the cleanest evidence table for the paper-facing claim that the proposed density-derived utility is directionally consistent with downstream perception quality.

## Hybrid Scheduler Addendum Re-evaluation

Status: latest algorithm-side utility check after updating SGCP scheduling to `hybrid_round_robin_dynamic_marginal`.

Input table: `06_parameter_sensitivity_hybrid_addendum.md`. The AP/Mbps rows come from the final hybrid rho-Pareto sweep. Utility is recomputed with the same replay protocol as above, but every row is evaluated on the formal `rho_eval = 2` utility scale to avoid changing the numerical scale of `q_i(g)` across rho settings.

Artifacts:

| Artifact | Path |
|---|---|
| Recompute script | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\build_hybrid_rho_pareto_utility_evalrho2.py` |
| Hybrid fixed-rho CSV | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\hybrid_rho_pareto_utility_evalrho2_vs_ap.csv` |
| Hybrid fixed-rho summary | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\hybrid_rho_pareto_utility_evalrho2_summary.json` |
| Combined hybrid CSV | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\combined_scheduler_hybrid_evalrho2_vs_ap.csv` |
| Combined hybrid summary | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\utility_proxy_ap_20260731\combined_scheduler_hybrid_evalrho2_summary.json` |

Correlation within the 28 hybrid rho-Pareto rows:

| Explanatory variable | Pearson with AP@0.5 | Spearman with AP@0.5 |
|---|---:|---:|
| Utility gain at rho_eval=2 | 0.903 | 0.960 |
| Final utility at rho_eval=2 | 0.905 | 0.971 |
| Total Mbps | 0.770 | 0.849 |
| Avg selected grids | 0.866 | 0.953 |

Combined statistic after merging scheduler variants with the hybrid rho-Pareto rows and excluding the duplicated saturated `rho_th = 2, budget = 40 Mbps` SGCP point:

| Explanatory variable | Pearson with AP@0.5 | Spearman with AP@0.5 |
|---|---:|---:|
| Utility gain at rho_eval=2 | 0.863 | 0.912 |
| Final utility at rho_eval=2 | 0.863 | 0.920 |
| Avg selected grids | 0.470 | 0.842 |

Interpretation: under the latest hybrid scheduler, the fixed-scale utility proxy is more strongly aligned with AP@0.5 than in the previous dynamic-CV Pareto table. For paper-facing use, the cleanest update is the 28-point hybrid addendum statistic: `Utility gain` Pearson/Spearman `0.903/0.960`, with `Final utility` showing a similar or slightly stronger monotonic trend. Selected-grid count is also correlated, but utility gain has the stronger Pearson correlation and is the scheduling objective.
