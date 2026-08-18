# Dense-Ver Manifest

This directory is the 2026-07-29 dense-LiDAR rerun package. All files below are dense-version evidence unless explicitly marked as diagnostic/provenance.

Current headline update: Table 1 and Table 3 now include the new
`hybrid_round_robin_dynamic_marginal` SGCP scheduler row
(`0.88/0.82/0.61`, `29.31 Mbps`, `593.34 GFLOPs/frame`). The previous
`dynamic_cv` SGCP row is retained as provenance. Derived tables that still say
`dynamic_cv` are internally valid for that earlier scheduler, but should be
rerun before claiming a fully hybrid-based ablation/sensitivity/scalability
story.

| File | Status | Purpose |
|---|---|---|
| `README.md` | updated dense | Dense package overview and valid-file guide. |
| `00_protocol_and_metrics.md` | updated dense | Dense protocol and metrics. |
| `01_dense_lidar_sgcp_hyperparameters.md` | updated dense | Dense SGCP operating point and derived coalition-capacity rule. |
| `01_protocol_native_baselines.md` | updated dense full-frame GT + hybrid row | Dense Table 1 protocol-native baselines with GFLOPs, using paper-facing full-frame GT and EdgeCooper-Pmax partial-upload adaptation. |
| `02_global_box_and_fusion.md` | dynamic_cv dense rho-2 | Dense Table 2 global-box and fusion scaffold diagnostics with GFLOPs; rerun if hybrid is promoted as final SGCP. |
| `03_scheduler_comparison.md` | updated dense rho-2 + hybrid row | Dense Table 3 SGCP-compatible scheduler comparison with GFLOPs. |
| `04_clustering_ablation.md` | dynamic_cv dense rho-2 | Dense Table 4 clustering ablation with heuristic and paper-inspired baselines; rerun if hybrid is promoted as final SGCP. |
| `05_baseline_reproduction_parameters.md` | updated dense | Dense baseline reproduction parameters and citations. |
| `06_parameter_sensitivity.md` | dynamic_cv dense | Dense rho Pareto sweep under fixed 40 MHz / 10ch; rerun if hybrid is promoted as final SGCP. |
| `07_plot_data_and_suggestions.md` | updated dense | Dense plot suggestions and caption guardrails. |
| `08_data_quality_and_remaining_work.md` | updated dense | Dense package audit, caveats, and optional future validation. |
| `09_parameters_and_algorithms.md` | algorithm current | SGCP algorithm definitions and pseudocode; dense-specific numeric parameters are in files 00/05. |
| `10_realtime_feasibility.md` | dense/control + hybrid payload note | Guard/zero-delay and broadcast/unicast NS3 realtime feasibility evidence; scheduler compute timing remains from dynamic_cv profiler unless rerun. |
| `11_utility_proxy_vs_ap.md` | dynamic_cv dense | Utility-surrogate vs AP@0.5 validation table for the density-derived C/V proxy; rerun if hybrid is promoted as final SGCP. |
| `12_cav_scale_sweep.md` | dynamic_cv dense full-frame GT | Active-CAV scalability sweep for N=5/10/15/20 with Pure late, EdgeCooper-Pmax, and SGCP; rerun if hybrid is promoted as final SGCP. |
| `13_multiscene_town06_table1.md` | dynamic_cv dense PointPillar | Town06 dense 10-CAV/20-background Table 1 for the paper-facing PointPillar-attentive checkpoint; rerun if hybrid is promoted as final SGCP. |
| `15_multibackbone_second_subset.md` | dynamic_cv dense SECOND | Full protocol-native Table 1 for SECOND attentive-to-early on dense roundabout and Town06; rerun if hybrid is promoted as final SGCP. |

Important boundary: the dense clean package no longer includes channel-count
sensitivity. The fixed paper-facing network setting is `40 MHz / 10ch`.

Raw traces, CSV files, compute profiles, and exploratory probes are kept
outside this clean Markdown package.
