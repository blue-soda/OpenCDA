# Round-Robin Dynamic Marginal Scheduler Probe

Status: completed 2026-08-01.

Purpose: test the requested SGCP scheduler variant where cluster heads take
turns selecting the strongest dynamic early-utility link

```text
Delta U_early(i,h,g | A_h) = q_hat_i,h(g) * (1 - Q_h^A(g)).
```

The implementation is intentionally isolated from the current SGCP mainline:

```text
C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\round_robin_dynamic_marginal.py
```

It is registered as `round_robin_dynamic_marginal`,
`round_robin_dynamic_marginal_pg`, `rr_dynamic_marginal`, and `rrdmpg`.

## Protocol

- Dataset: `D:\Data\Carla\2026_07_29_02_32_08`
- Scenario: dense Town03 / `v2xp_cluster_carla_dense`
- Frames: 41, `000060` to `000140`
- CAVs: 20
- GT scope: explicit full-frame GT, `--gt-scope full-frame`
- Detector: PointPillar attentive-to-early checkpoint
- Clustering: `potential_verified_cov_coalition_game`
- Scheduler: `round_robin_dynamic_marginal`
- Receiver policy: `all-cluster-heads`
- Late fusion: `inter_cluster_nms`
- Network: 40 MHz, 10 target subchannels
- NS3 estimator: `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`
- Data-plane deadline: 60 ms
- SGCP parameters: `rho_th=2`, `N_max=5`, `head_rb_budget=2`
- Raw-LiDAR budget: `--sgcp-frame-mbps-budget 40.0`
- Point upload: density-capped grid upload, `--upload-density-cap-rho 2`

## Result

| Scheduler | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame | Avg source CAVs/call | Avg selected grids/call | P95 data-plane time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round_robin_dynamic_marginal | 0.86 | 0.81 | 0.61 | 27.54 | 0.90 | 28.44 | 593.34 | 6.63 | 1.51 | 48.81 | 37.39 ms |

For comparison, the current dense Table 1 SGCP row with `dynamic_cv`,
`rho_th=2`, `N_max=5`, and the same full-frame-GT protocol reports
`0.86/0.82/0.59`, `27.84` Raw Mbps, `0.86` Box Mbps, `28.69` Total Mbps,
and `593.43` GFLOPs/frame.

## Interpretation

The round-robin dynamic marginal variant is feasible and stays within the same
communication envelope as the current SGCP row.  It slightly improves AP@0.7
but slightly lowers AP@0.5.  The result is close enough to the current
`dynamic_cv` scheduler that it should be treated as a scheduler probe rather
than a replacement unless the paper specifically wants the round-robin
distributed scheduling story.

## Reproduction Command

```powershell
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference `
  --dataset-root D:\Data\Carla `
  --scenario-id 2026_07_29_02_32_08 `
  --ego-cav-id 1 `
  --max-frames 41 `
  --fusion-method early `
  --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml `
  --bandwidth-mhz 40 `
  --num-channels 10 `
  --channel-estimator ns3 `
  --ns3-tb-size-bytes 899 `
  --ns3-slot-duration-ms 0.5 `
  --ns3-subchannel-prbs 10 `
  --ns3-symbols-per-slot 12 `
  --ns3-mcs 28 `
  --communication-deadline-ms 60 `
  --gt-scope full-frame `
  --sgcp-constrained `
  --clustering potential_verified_cov_coalition_game `
  --resource-allocation round_robin_dynamic_marginal `
  --sgcp-receiver-policy all-cluster-heads `
  --sgcp-upload-mode grid `
  --sgcp-inter-cluster-late-fusion `
  --n-max 5 `
  --rho-th 2 `
  --head-rb-budget 2 `
  --sgcp-frame-mbps-budget 40.0 `
  --upload-density-cap-rho 2 `
  --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\round_robin_dynamic_marginal_20260801\table1_41f\trace.csv `
  --eval-stats-output docs\doc_workspace\SGCP\artifacts\round_robin_dynamic_marginal_20260801\table1_41f\eval_stats.csv
```
