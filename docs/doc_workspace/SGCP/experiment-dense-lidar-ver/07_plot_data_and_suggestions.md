# Dense Plot Data and Drawing Suggestions

No figures are generated in this directory. Use the Markdown tables as raw data. All rows refer to the dense-LiDAR dataset unless a table explicitly says otherwise.

| Figure purpose | Use data from | Suggested axes/encoding |
| --- | --- | --- |
| Protocol-native baseline comparison | `01_protocol_native_baselines.md` | grouped bars for AP@0.3/AP@0.5/AP@0.7; annotate Total Mbps and GFLOPs/frame |
| Global box aggregation effect | `02_global_box_and_fusion.md`, global box table | AP bars plus stacked Raw/Box Mbps bar |
| Fusion scaffold ablation | `02_global_box_and_fusion.md`, fusion scaffold table | grouped AP bars; annotate detector GFLOPs/frame to show the compute benefit of cluster-head-only late aggregation |
| SGCP-compatible scheduler comparison | `03_scheduler_comparison.md` | AP@0.5 vs Total Mbps scatter; point label = scheduler; marker size = GFLOPs/frame |
| Clustering ablation | `04_clustering_ablation.md` | AP@0.5 bar sorted by AP; show Total Mbps and GFLOPs/frame as text labels |
| Dense rho Pareto | `06_parameter_sensitivity.md` | line plot with Total Mbps on x-axis and AP on y-axis, one line per `rho_th`; current table is based on `dynamic_cv` and marks `rho_th=2`, Total `28.69 Mbps`. If `hybrid_round_robin_dynamic_marginal` becomes final, rerun this Pareto and mark the new headline point, currently `29.31 Mbps` from the fixed rho-2 probe. |
| Utility proxy validation | `11_utility_proxy_vs_ap.md` | small scatter plot with Utility gain on x-axis and AP@0.5 on y-axis; label each scheduler; optionally report Pearson/Spearman in caption |
| Active-CAV scalability | `12_cav_scale_sweep.md` | two-panel line plot: AP@0.5 vs active CAV count, and GFLOPs/frame vs active CAV count; optionally annotate Total Mbps |

Recommended caption rule: state `dense LiDAR`, `attentive-derived detector`,
`40 MHz`, `10 target subchannels`, `100 ms perception cycle`, `60 ms
data-plane deadline`, explicit `--gt-scope full-frame`, and whether the row
uses global/inter-cluster box aggregation.
