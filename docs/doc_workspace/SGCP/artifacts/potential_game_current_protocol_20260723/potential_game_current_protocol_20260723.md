# Original PotentialGame Current-Protocol Check - 2026-07-23

## Protocol

- Dataset: `D:\Data\Carla\2026_07_15_01_26_56`
- Frames: 41, `000060` to `000140`
- Detector: attentive checkpoint through `enable_coperception_early_from_attentive.yaml`
- Clustering: `coalition_game`
- Receiver policy: `all-cluster-heads`
- Late aggregation: inter-cluster box NMS enabled
- Upload mode: grid-level raw LiDAR
- Resource allocation under test: `potential_game`
- Network estimator: NS3-calibrated estimator, 40 MHz, 10 target subchannels, `tb_size=899 bytes`, `slot=0.5 ms`, `subchannel_prbs=10`, `MCS=28`, `PSSCH symbols=12`
- Scheduler admission budget: `200 ms`
- Other parameters: `N_max=4`, `rho_th=3`, `head_rb_budget=2`

Command:

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 41 --fusion-method early --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml --sgcp-constrained --clustering coalition_game --sgcp-receiver-policy all-cluster-heads --sgcp-upload-mode grid --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --sgcp-grid-selection-mode utility --sgcp-grid-score-mode utility --n-max 4 --rho-th 3 --head-rb-budget 2 --num-channels 10 --bandwidth-mhz 40 --communication-deadline-ms 200 --channel-estimator ns3 --ns3-tb-size-bytes 899 --ns3-slot-duration-ms 0.5 --ns3-subchannel-prbs 10 --ns3-symbols-per-slot 12 --ns3-mcs 28 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\potential_game_current_protocol_20260723\potential_game_trace.csv --eval-stats-output docs\doc_workspace\SGCP\artifacts\potential_game_current_protocol_20260723\potential_game_eval_stats.csv
```

## Result

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG main point | 0.87 | 0.81 | 0.36 | 62.54 | 0.74 | 63.28 | 536.94 |
| Original PotentialGame | 0.81 | 0.74 | 0.36 | 54.56 | 0.71 | 55.27 | 536.88 |

## Trace Summary

| Metric | PAPG | Original PotentialGame |
| --- | ---: | ---: |
| Evaluated frames | 41 | 41 |
| Late-fused detector sources/frame | 6.00 | 6.00 |
| Scheduled links/frame | 10.00 | 10.00 |
| Unique uploaded source CAVs/frame | 10.00 | 10.00 |
| Selected grids/frame | 583.32 | 544.44 |
| Raw LiDAR bytes/frame | 781704.20 | 682021.07 |
| Frame communication time mean/max, estimator | 44.55 / 45.03 ms | 44.11 / 45.03 ms |
| Predicted boxes/sample | 18.50 | 17.64 |
| Input points/frame for detector profile | 78155.9 | 71925.7 |

## Interpretation

The original PotentialGame works under the current channel configuration and deadline estimator. It schedules the same number of links and the same number of cluster-head detector calls as PAPG, so the compute profile is almost identical. The performance gap comes from action quality: PotentialGame selects fewer and less object-targeted grids, resulting in lower raw payload and fewer useful detections after inter-cluster NMS.

The AP difference is concentrated at coverage and medium-IoU quality: PAPG improves AP@0.3 by `+0.06` and AP@0.5 by `+0.07`, while AP@0.7 is unchanged. This supports the paper narrative that PAPG adds perception-aware target/view selection on top of the original potential-game resource framework rather than merely increasing detector compute.
