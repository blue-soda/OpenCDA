# Unified COV Game Current-Protocol Result - 2026-07-23

## Protocol

- Detector/checkpoint: attentive early detector, `docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/enable_coperception_early_from_attentive.yaml`
- Dataset: `D:\Data\Carla`, scenario `2026_07_15_01_26_56`, 41 frames
- Fusion: cluster-head raw-LiDAR early fusion plus inter-cluster box NMS
- Receivers: all cluster heads
- Clustering/resource allocation: `cov_coalition_game` + `cov_potential_game`
- Network estimator: 40 MHz, 10 target subchannels, `tb_size=899 B`, `slot=0.5 ms`, `subchannel_prbs=10`, `symbols=12`, `MCS=28`
- SGCP parameters: `N_max=4`, `rho_th=3`, `head_rb_budget=2`, scheduler admission budget `200 ms`

## Unified Utility

For a sender-grid action, let `q_i(g)` be the sender's normalized grid density and `q_r(g)` be the receiver-side quality. For scheduling, the receiver is the cluster head. For coalition formation, it is the candidate coalition quality `max_j q_j(g)`.

```text
C(i,g|r) = q_i(g) * (1 - q_r(g))
O(i,g)   = q_i(g)
V(i,g|r) = q_i(g) if q_r(g) > 0 else 0
L(i,r,g) = normalized link/payload cost
U(i,g|r) = C + O + V - L
```

Coalition utility is the same grid utility aggregated over candidate grids. The default coalition objective uses `V` only, because coalition formation is intended to create stable multi-view early-fusion groups; the complete C/O/V/L utility is used by block-level scheduling.

## Main Result

| Method | Cluster terms | Scheduler terms | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Avg source CAVs | Avg selected grids |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP-COV unified | V | C+O+V-L | 0.87 | 0.81 | 0.37 | 62.55 | 2.67 | 96.92 |

This is not lower than the previous SGCP-PAPG headline point `0.87/0.81/0.36`.

## O/V Internal Ablation

All rows use the same protocol above and differ only in active COV terms.

| Variant | Cluster terms | Scheduler terms | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Avg source CAVs | Avg selected grids |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main COV | V | C+O+V-L | 0.87 | 0.81 | 0.37 | 62.55 | 2.67 | 96.92 |
| Cluster O only | O | C+O+V-L | 0.81 | 0.71 | 0.34 | 62.83 | 3.00 | 116.80 |
| Cluster O+V | O+V | C+O+V-L | 0.87 | 0.81 | 0.37 | 62.55 | 2.67 | 96.92 |
| Scheduler w/o V | V | C+O-L | 0.87 | 0.81 | 0.36 | 62.54 | 2.67 | 97.00 |
| Scheduler w/o O | V | C+V-L | 0.87 | 0.81 | 0.36 | 62.54 | 2.67 | 96.93 |

Interpretation: the coalition stage is sensitive to using `V`; `O` alone fragments the grouping and hurts AP. In the scheduler, either `O` or `V` alone is nearly sufficient under this dataset, while the complete C/O/V/L expression gives the best AP@0.7. This supports writing coalition formation as multi-view group formation and scheduling as full marginal perception utility.

## Reproduction Command

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 41 --fusion-method early --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml --sgcp-constrained --clustering cov_coalition_game --resource-allocation cov_potential_game --sgcp-receiver-policy all-cluster-heads --sgcp-upload-mode grid --sgcp-inter-cluster-late-fusion --sgcp-grid-selection-mode utility --sgcp-grid-score-mode utility --bandwidth-mhz 40 --num-channels 10 --channel-estimator ns3 --communication-deadline-ms 200 --ns3-tb-size-bytes 899 --ns3-slot-duration-ms 0.5 --ns3-subchannel-prbs 10 --ns3-symbols-per-slot 12 --ns3-mcs 28 --n-max 4 --rho-th 3 --head-rb-budget 2
```
