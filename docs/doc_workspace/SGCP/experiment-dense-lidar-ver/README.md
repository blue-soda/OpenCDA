# SGCP Dense-LiDAR Experiment Package (2026-07-29)

Status: dense dataset export, control-plane NS3 probes, SGCP parameter records,
Table 1/2/3/4 reruns, dense GFLOPs, utility-proxy validation, realtime
breakdown, active-CAV scalability sweep, PointPillar multi-scene validation, SECOND attentive-to-early
backbone validation, and dense package audit are complete.

Dataset: `D:\Data\Carla\2026_07_29_02_32_08`.

Dataset export command:

```powershell
conda run --no-capture-output -n opencda python opencda.py -t v2xp_cluster_carla_dense --dump
```

This command requires one running CARLA server and writes the dump under
`D:\Data\Carla`; the dense package uses the generated scenario directory
`2026_07_29_02_32_08`.

Scenario: `v2xp_cluster_carla_dense`, same 20-CAV Town03 layout as `v2xp_cluster_carla`; vehicle LiDAR changed to 32 channels, 320000 points/s, 20 Hz rotation, 50 m range.

Frame protocol: DataDumper skips ticks before `000060` and stores every 2 CARLA ticks; exported frame range is `000060`-`000140`, 41 frames per CAV.

Detector checkpoint: attentive-derived early-fusion protocol, same checkpoint policy as the clean SGCP package.

Network/control protocol for dense series: Guard `1 ms`, zero-time send delay `0 ms`, pre-send activation synchronization for control-plane NS3 probes; raw LiDAR data-plane uses scheduled unicast and NS3-calibrated channel estimation.

Paper-facing dense candidate: `40 MHz / 10ch / 60 ms data-plane deadline`, SGCP hybrid scheduler row `0.88/0.82/0.61`, `29.31 Mbps` including box sharing, and `593.34 GFLOPs/frame`. The previous `dynamic_cv` row `0.86/0.82/0.59`, `28.69 Mbps`, and `593.43 GFLOPs/frame` is retained in Table 1/3 as provenance.

Updated dense files:

- `00_protocol_and_metrics.md`
- `01_dense_lidar_sgcp_hyperparameters.md`
- `01_protocol_native_baselines.md`
- `02_global_box_and_fusion.md`
- `03_scheduler_comparison.md`
- `04_clustering_ablation.md`
- `05_baseline_reproduction_parameters.md`
- `06_parameter_sensitivity.md`
- `07_plot_data_and_suggestions.md`
- `08_data_quality_and_remaining_work.md`
- `09_parameters_and_algorithms.md`
- `10_realtime_feasibility.md`
- `11_utility_proxy_vs_ap.md`
- `12_cav_scale_sweep.md`
- `13_multiscene_town06_table1.md`
- `15_multibackbone_second_subset.md`

`09_parameters_and_algorithms.md` describes the SGCP algorithms. It currently documents the formal `dynamic_cv` scheduler; the hybrid round-robin/dynamic-marginal scheduler is recorded as a late scheduler probe in Table 1/3 and should be promoted into the algorithm document only if selected as the final paper method.

Raw traces, CSV files, compute-profile JSON/CSV files, and exploratory probes
are kept outside this clean Markdown package. Non-paper-facing backbone
diagnostics were moved out to the sibling archive directory
`experiment-dense-lidar-ver-extra-backbone-probes-20260731`. This clean
directory intentionally contains only paper-facing tables, protocol notes,
algorithm descriptions, and reproducibility parameters.
