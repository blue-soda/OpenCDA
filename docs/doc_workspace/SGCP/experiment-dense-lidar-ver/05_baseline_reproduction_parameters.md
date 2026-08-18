# Dense Baseline Reproduction Parameters

This file records the parameters needed to reproduce the baseline rows in the dense-LiDAR experiment package. It is not a separate result table.

## Common Dense Protocol

| Item | Value |
| --- | --- |
| Dataset | `D:\Data\Carla\2026_07_29_02_32_08` |
| Scenario | `v2xp_cluster_carla_dense` |
| CAV count | 20 |
| Frames | 41 frames, `000060` to `000140` |
| LiDAR | 32 channels, 320000 points/s, 20 Hz rotation, 50 m range |
| Detector/checkpoint | attentive-derived raw point-cloud detector |
| Main bandwidth | 40 MHz |
| Target subchannels | 10 |
| NS3 estimator | `tb_size=899 bytes`, `slot=0.5 ms`, `subchannel_prbs=10`, `PSSCH symbols=12`, `MCS=28` |
| Data-plane deadline | 60 ms inside a 100 ms perception cycle |
| Table 1 AP GT scope | Full-frame GT for every receiver sample; the all-CAV frame annotations are projected to each receiver pose so the denominator is scheduler-independent |
| Control-plane setting | Guard `1 ms`, zero-time send delay `0 ms`, explicit activation synchronization |
| Box aggregation payload | Included whenever a row uses pure/global/inter-cluster late NMS |
| Global-box GFLOPs | Detector forward GFLOPs plus estimated global NMS cost, `mean_t(n_t(n_t-1)/2*256)/1e9`, from trace-level pre-NMS box counts |

Density note: PCS and EdgeCooper use the replay LiDAR density map with
`density_threshold=2.0` for their native blind-spot and grid-scoring logic.
PCS does not use SGCP's density-capped grid upload; selected grids transmit
their full raw point payload. FullPerception does not define a point-cloud
truncation mechanism in the original PCS algorithm, so adding such a cap would
no longer be protocol-native PCS. EdgeCooper's original paper does define a
voxel/pillar-wise `P_max` partial point-upload constraint. The
`EdgeCooper-Pmax` row implements that idea as an OpenCDA grid-level residual
density cap while keeping EdgeCooper's V2V link and grid-selection logic.

Paper-facing rule: non-Pmax EdgeCooper rows are deprecated and should be used
only as diagnostic full-grid evidence. EdgeCooper comparisons in the paper
should use the Pmax rows.

## FullPerception-PCS

| Protocol | Clustering | Receiver policy | Late/box aggregation | Main dense row |
| --- | --- | --- | --- | --- |
| Protocol-native Table 1 | `singleton` | `all-cavs` with unscheduled receivers evaluated as local-only | none | `0.33/0.29/0.21`, `70.49 Mbps raw`, `1790.47 GFLOPs/frame`; no local-preserving extra detector |
| Global box Table 2 | `singleton` | `all-cavs` | global box NMS | `0.81/0.74/0.53`, `72.70 Mbps total`, `1790.48 GFLOPs/frame` |
| SGCP-compatible Table 3 | `potential_verified_cov_coalition_game` | all cluster heads | inter-cluster box NMS | `0.82/0.76/0.54`, `20.46 Mbps total` |

PCS raw-LiDAR adaptation parameters:

| Parameter | Value |
| --- | --- |
| Resource allocation | `fullperception_pcs` |
| Blind-spot min division | 4 |
| Blind-spot adjacency radius | 4 |
| Minimum blind-spot grids | 128 |
| Communication range | 35 m |
| K, simultaneous senders per receiver | 2 where the table aligns K with SGCP |
| Density use | Receiver blind spots are `req_grids - high_density_grids` with `density_threshold=2.0`; no SGCP upload-density cap |

## EdgeCooper V2V Adaptation

| Protocol | Clustering | Receiver policy | Late/box aggregation | Main dense row |
| --- | --- | --- | --- | --- |
| Protocol-native Table 1, Pmax-style partial upload | `singleton` | `all-cavs` | none | `0.40/0.37/0.26`, `86.30 Mbps raw`, `1790.59 GFLOPs/frame`; no local-preserving extra detector |
| Global box Table 2 | `singleton` | `all-cavs` | global box NMS | `0.81/0.74/0.53`, `88.32 Mbps total` |
| Global box Table 2, Pmax-style partial upload | `singleton` | `all-cavs` | global box NMS | `0.85/0.80/0.56`, `88.95 Mbps total`, `1790.62 GFLOPs/frame` |
| SGCP-compatible Table 3 | `potential_verified_cov_coalition_game` | all cluster heads | inter-cluster box NMS | `0.81/0.74/0.51`, `87.03 Mbps total` |
| SGCP-compatible Table 3, Pmax-style partial upload | `potential_verified_cov_coalition_game` | all cluster heads | inter-cluster box NMS | `0.88/0.81/0.56`, `33.46 Mbps total` |

EdgeCooper V2V adaptation parameters:

| Parameter | Value |
| --- | --- |
| Resource allocation | `selective_edgecooper_global` or `selective_edgecooper_global_hd` as named in the table |
| Communication range | 35 m |
| Member budget | 3 |
| Grid budget | table-specific; constrained by the 60 ms data-plane deadline in final dense rows |
| K, simultaneous senders per receiver | 2 where the table aligns K with SGCP |
| Density use | Receiver blind spots and grid scores use raw grid density with `density_threshold=2.0`; Pmax rows use `edgecooper_pmax_density_cap_rho=2.0` as the grid-level residual point cap |

EdgeCooper-Pmax adaptation:

```text
For each receiver r:
    build EdgeCooper candidate sender-grid links within communication range
    split candidate links into endpoint-feasible orthogonal batches
    admit batches until the 60 ms frame-level data-plane budget is exhausted
    for each selected grid g:
        cap(g) = ceil(rho * grid_area)
        residual(g) = max(cap(g) - local_points_r(g), 0)
    for each admitted sender i -> r:
        upload at most residual(g) points from each selected grid g
        decrement residual(g) after each accepted upload
```

This is the OpenCDA raw-LiDAR counterpart of EdgeCooper's paper-level `P_max`
constraint. It is intentionally not applied to PCS. The all-receiver Table 1/2
rows use multi-batch admission so that endpoint constraints apply per
orthogonal resource batch while the shared 60 ms frame budget is used across
all 20 receiver samples.

## PACP-LiDAR V2V Adaptation

PACP is originally a camera/BEV-feature collaborative perception method, not a
raw point-cloud scheduler. The dense package therefore reports a paper-aligned
LiDAR adaptation: replay LiDAR grid density is used as the BEV-match priority
proxy, uncovered sender grids approximate PACP's perceptual-region union gain,
and density-capped raw-LiDAR upload is used as the LiDAR counterpart of PACP's
adaptive compression/rate-control stage.

In the protocol-native singleton protocol, every CAV is its own receiver
cluster. A strict cluster-local candidate pool would therefore contain no helper
vehicles and would collapse PACP-LiDAR into local-only inference. To keep the
singleton protocol consistent with PCS and EdgeCooper, the Table 1 and Table 2
PACP-LiDAR rows use a global V2V candidate pool: for each singleton receiver,
all other CAVs within the 35 m V2V range are candidate senders, and PACP ranks
their candidate raw-LiDAR grids by the LiDAR BEV-match plus perceptual-region
score. The same 40 MHz / 10ch channel estimator, K=2 receiver-side parallel
sender limit, endpoint-feasible matching, and 60 ms data-plane deadline are then
applied.

In the SGCP-compatible scheduler comparison, PACP-LiDAR remains cluster-local:
candidate senders are the members of the SGCP coalition, and the global
singleton fallback is not used.

| Protocol | Clustering | Receiver policy | Late/box aggregation | Main dense row |
| --- | --- | --- | --- | --- |
| Protocol-native Table 1 | `singleton` | `all-cavs` | none | `0.40/0.37/0.26`, `86.30 Mbps raw`, `1790.59 GFLOPs/frame` |
| Global box Table 2 | `singleton` | `all-cavs` | global box NMS | `0.86/0.79/0.55`, `89.02 Mbps total`, `1753.50 GFLOPs/frame` |
| SGCP-compatible Table 3 | `potential_verified_cov_coalition_game` | all cluster heads | inter-cluster box NMS | `0.88/0.82/0.59`, `33.48 Mbps total` |

PACP-LiDAR adaptation parameters:

| Parameter | Value |
| --- | --- |
| Resource allocation | `pacp_lidar` |
| Protocol-native singleton candidate pool | all non-receiver CAVs within 35 m |
| SGCP-compatible candidate pool | cluster-local non-head members |
| Member budget | 3 |
| Grid budget | 117 |
| K, simultaneous senders per receiver | 2 |
| Density use | Grid scoring uses replay LiDAR density with `density_threshold=2.0`; admitted uploads are density-capped with `rho=2` as PACP's raw-LiDAR compression/rate-control proxy |

PACP rerun provenance:
`C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\pacp_lidar_paper_fix_20260801`.

Table 1 EdgeCooper/PACP payload note:

The protocol-native EdgeCooper-Pmax and PACP-LiDAR rows both report
`86.30 Mbps` because they are evaluated under the same all-20-receiver
singleton protocol and both saturate the same shared 60 ms raw-LiDAR data-plane
budget. Their traces and compute are not identical: EdgeCooper-Pmax uploads
`2.50` helper CAVs and `27.17` selected grids per receiver-call on average and
requires `1790.59 GFLOPs/frame` after removing hidden local-preserving extra
detector calls, whereas PACP-LiDAR uploads `2.59` helper CAVs and `33.99`
selected grids per receiver-call after endpoint matching and requires
`1790.59 GFLOPs/frame`.

## SGCP

| Component | Value |
| --- | --- |
| Clustering | `potential_verified_cov_coalition_game` |
| Resource scheduling | Current headline candidate uses `hybrid_round_robin_dynamic_marginal`; previous retained row uses `dynamic_cv`. Both use density-capped deterministic random point upload. |
| Receiver policy | all cluster heads |
| Fusion | raw-LiDAR grid upload to cluster heads, then inter-cluster box NMS |
| Dense headline candidate | `0.88/0.82/0.61`, raw `28.42 Mbps`, box `0.89 Mbps`, total `29.31 Mbps`, `593.34 GFLOPs/frame` |
| Previous dense headline row | `0.86/0.82/0.59`, raw `27.84 Mbps`, box `0.86 Mbps`, total `28.69 Mbps`, `593.43 GFLOPs/frame` |

## Table 4 Clustering Baselines

All Table 4 rows use the same dense detector, dynamic C/V scheduler with density-capped deterministic random point upload, all-cluster-head receiver policy, 40 MHz / 10ch channel estimator, and inter-cluster box NMS. Only the clustering algorithm changes.

| Clustering row | Type | Mapping |
| --- | --- | --- |
| Random balanced | heuristic | deterministic random partition into capacity-limited clusters |
| Distance-greedy | heuristic | proximity-first grouping with center-nearest head election |
| Density/quality-greedy | heuristic | sensing-density head selection plus incremental coverage member selection |
| SeAC-inspired | paper baseline | Akbar et al., IEEE T-ITS 2023, DOI `10.1109/TITS.2023.3237321`; SDN/social cues are adapted to same-frame direction, relative speed, distance and sensing overlap |
| HHOCNET-inspired | paper baseline | Ali et al., IEEE T-ITS 2023, DOI `10.1109/TITS.2023.3257484`; HHO search is adapted to deterministic multi-start cluster partitioning over proximity, relative mobility and sensing coverage |
