# COV Coalition + COV Potential Game - 2026-07-23

## Purpose

This artifact validates a new SGCP algorithm pair whose paper-facing utility is

```text
Delta U(a) = Delta C(a) + Delta O(a) + Delta V(a) - L(a)
```

where `C` is observability/coverage completion, `O` is object-relevant evidence,
`V` is multi-view complementary evidence, and `L` is communication/link cost.

New files:

- `opencda/core/clustering/algorithms/clustering/cov_coalition_game.py`
- `opencda/core/clustering/algorithms/resource_allocation/cov_potential_game.py`

Existing algorithm files are not modified.  The only existing-code edits are
factory/CLI registration points.

## Protocol

- Dataset: `D:\Data\Carla\2026_07_15_01_26_56`
- Frames: 41, `000060` to `000140`
- Detector: attentive checkpoint through `enable_coperception_early_from_attentive.yaml`
- Clustering: `cov_coalition_game`
- Resource allocation: `cov_potential_game`
- Receiver policy: `all-cluster-heads`
- Upload mode: grid-level raw LiDAR
- Late aggregation: inter-cluster box NMS enabled
- Network estimator: NS3-calibrated estimator, 40 MHz, 10 target subchannels,
  `tb_size=899 bytes`, `slot=0.5 ms`, `subchannel_prbs=10`, `MCS=28`,
  `PSSCH symbols=12`
- Scheduler admission budget: `200 ms`
- Other parameters: `N_max=4`, `rho_th=3`, `head_rb_budget=2`

## Command

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 41 --fusion-method early --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml --sgcp-constrained --clustering cov_coalition_game --sgcp-receiver-policy all-cluster-heads --sgcp-upload-mode grid --resource-allocation cov_potential_game --sgcp-inter-cluster-late-fusion --sgcp-grid-selection-mode utility --sgcp-grid-score-mode utility --n-max 4 --rho-th 3 --head-rb-budget 2 --num-channels 10 --bandwidth-mhz 40 --communication-deadline-ms 200 --channel-estimator ns3 --ns3-tb-size-bytes 899 --ns3-slot-duration-ms 0.5 --ns3-subchannel-prbs 10 --ns3-symbols-per-slot 12 --ns3-mcs 28 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\cov_game_current_protocol_20260723\cov_game_final_trace.csv --eval-stats-output docs\doc_workspace\SGCP\artifacts\cov_game_current_protocol_20260723\cov_game_final_eval_stats.csv
```

## Result

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG main point | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 |
| COV coalition + COV potential game | 0.87 | 0.81 | 0.36 | 62.55 | 0.74 | 63.29 | 536.94 |

## Trace Summary

| Metric | Value |
| --- | ---: |
| Evaluated frames | 41 |
| Late-fused detector sources/frame | 6.00 |
| Scheduled links/frame | 10.00 |
| Selected grids/frame | 582.27 |
| Raw LiDAR bytes/frame | 781819.0 |
| Frame communication time mean/max, estimator | 44.55 / 45.03 ms |
| Mean source CAVs/detector call | 2.667 |
| Mean input points/frame | 78163.0 |
| Detector-side GFLOPs/frame | 536.94 |

## Interpretation

The COV scheduler matches the current PAPG main-table AP while exposing the
utility terms needed by the revised paper narrative.  The scheduler keeps PAPG's
two-stage potential-game structure: the first stage is coverage-dominant and the
second stage is object/view-dominant.

The new coalition game keeps the coalition-formation mechanics but rewrites the
vehicle-level marginal utility in C/O/V/L terms.  In the validated setting,
coalition membership is intentionally dominated by the stable multi-view
complementarity term, because earlier tuning showed that using standalone
object/coverage terms at vehicle level over-fragmented the topology.  Coverage
and object relevance are therefore most strongly realized at the block-level
scheduler, while the coalition game provides stable local multi-view groups.
