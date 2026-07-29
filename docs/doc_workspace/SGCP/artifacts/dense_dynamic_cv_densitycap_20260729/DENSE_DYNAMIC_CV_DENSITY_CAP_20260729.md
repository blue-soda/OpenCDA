# Dense Dynamic C/V and Density-Capped Upload Probe

Date: 2026-07-29

Purpose: explore why dense LiDAR improves AP@0.7 but can reduce AP@0.3 at the
same raw-LiDAR Mbps, and test whether dynamic receiver evidence and density
capped point upload recover coverage/precision.

Common setting:

- Dataset: `D:\Data\Carla\2026_07_29_02_32_08`
- CAVs / frames: 20 CAVs, 41 frames (`000060` to `000140`)
- Detector: attentive-derived early-fusion checkpoint
- Clustering: `potential_verified_cov_coalition_game`
- Inter-cluster fusion: cluster-head predictions with box-level NMS
- NS3/data-plane setting: Guard `1 ms`, zero-time send delay `0 ms`,
  `40 MHz`, `10` target subchannels, `MCS=28`, `12` PSSCH symbols,
  `10 PRB/subchannel`, `RRI=5 ms`
- SGCP hyperparameters: `N_max=5`, `rho_th=5`, `head_rb_budget=2`

## 1. NS3 Capacity Probe

The previous `~62 Mbps / 55 ms` replay and the relaxed `93.75 Mbps / 136 ms`
diagnostic are not contradictory. Completion time is governed by the heaviest
parallel link, not only by total frame Mbps.

| Plan | Total load | Parallel links | Max link payload | App callbacks | Delay mean | Delay P95 | Delay max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old SGCP replay, frame `000060` | 61.70 Mbps | 10 | 79.68 KB | 80/80 | 26.51 ms | 53 ms | 55 ms |
| Dense relaxed `rho_th=5`, frame `000076` | 92.34 Mbps | 6 | 225.58 KB | 118/118 | 60.98 ms | 121 ms | 136 ms |
| Dense balanced 10ch, 80 KB/link | 64.00 Mbps | 10 | 80.00 KB | 80/80 | 27.25 ms | 55 ms | 55 ms |
| Dense balanced 10ch, 85 KB/link | 68.00 Mbps | 10 | 85.00 KB | 90/90 | 29.56 ms | 58 ms | 58 ms |
| Dense balanced 10ch, 90 KB/link | 72.00 Mbps | 10 | 90.00 KB | 90/90 | 29.89 ms | 61 ms | 61 ms |
| Dense balanced 10ch, 100 KB/link | 80.00 Mbps | 10 | 100.00 KB | 100/100 | 32.50 ms | 66 ms | 66 ms |
| Dense balanced 10ch, 105 KB/link | 84.00 Mbps | 10 | 105.00 KB | 110/110 | 34.91 ms | 69 ms | 69 ms |
| Dense balanced 10ch, 110 KB/link | 88.00 Mbps | 10 | 110.00 KB | 110/110 | 35.18 ms | 72 ms | 72 ms |
| Dense balanced 10ch, 120 KB/link | 96.00 Mbps | 10 | 120.00 KB | 120/120 | 37.83 ms | 77 ms | 77 ms |

Interpretation:

- With 10 orthogonal links/subchannels occupied, the empirical 60 ms payload
  point is `68 Mbps` (`85 KB/link`, max `58 ms`); `72 Mbps` slightly exceeds
  60 ms.
- The empirical 70 ms payload point is `84 Mbps` (`105 KB/link`, max `69 ms`);
  `88 Mbps` slightly exceeds 70 ms.
- The relaxed `93.75 Mbps` diagnostic is slower because it uses only 6 parallel
  links and has per-link payloads above 200 KB.

## 2-4. Dynamic C/V and Density-Capped Upload

Variants:

- `static_full`: current clean SGCP scheduler, `cov_potential_game`, no upload
  point clipping.
- `dynamic_full`: new `dynamic_cv` scheduler; it updates receiver-side grid
  evidence after each accepted upload.
- `static_cap5`: current clean SGCP scheduler plus per-grid upload cap
  `rho_cap=5 points/m^2`.
- `dynamic_cap5`: dynamic C/V scheduler plus the same per-grid upload cap.

The upload cap is deterministic: for every selected grid, at most
`rho_cap * grid_area = 5 * 100 = 500` points are randomly sampled with a stable
seed. Payload/deadline accounting uses the same capped point count.

| Variant | Budget/deadline | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Avg uploaded sources/sample | Avg selected grids/sample | Est. mean/P95/max time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_full_68mbps` | 68 Mbps / 60 ms | 0.80 | 0.74 | 0.54 | 67.97 | 1.39 | 6.22 | 49.09 / 59.96 / 60.00 ms |
| `dynamic_full_68mbps` | 68 Mbps / 60 ms | 0.80 | 0.74 | 0.54 | 67.97 | 1.39 | 6.19 | 49.06 / 59.96 / 60.00 ms |
| `static_cap5_68mbps` | 68 Mbps / 60 ms | 0.85 | 0.81 | 0.64 | 55.91 | 1.51 | 19.60 | 38.98 / 49.58 / 50.83 ms |
| `dynamic_cap5_68mbps` | 68 Mbps / 60 ms | 0.85 | 0.81 | 0.64 | 55.95 | 1.51 | 19.60 | 39.03 / 49.58 / 50.83 ms |
| `static_full_84mbps` | 84 Mbps / 70 ms | 0.81 | 0.74 | 0.53 | 83.91 | 1.49 | 7.69 | 58.56 / 69.97 / 70.00 ms |
| `dynamic_full_84mbps` | 84 Mbps / 70 ms | 0.81 | 0.74 | 0.52 | 83.91 | 1.49 | 7.60 | 58.55 / 69.97 / 70.00 ms |
| `static_cap5_84mbps` | 84 Mbps / 70 ms | 0.85 | 0.81 | 0.65 | 59.38 | 1.51 | 22.61 | 41.47 / 52.41 / 53.54 ms |
| `dynamic_cap5_84mbps` | 84 Mbps / 70 ms | 0.85 | 0.81 | 0.65 | 59.43 | 1.51 | 22.61 | 41.53 / 52.41 / 53.54 ms |

Interpretation:

- Dynamic C/V alone does not improve this dense scenario. The selected links
  and selected grid counts are nearly unchanged from static C/V.
- Density-capped upload is the useful change. It allows more selected grids
  under the same budget, lowers actual raw Mbps, and improves all AP thresholds.
- The best explored point is `static_cap5_84mbps` or `dynamic_cap5_84mbps`:
  AP `0.85/0.81/0.65` with about `59.4 Mbps` actual raw payload and max
  estimated data-plane time `53.5 ms`.
- Because static and dynamic C/V are effectively tied after density capping,
  the cleaner paper-facing mechanism is likely "SGCP + density-capped grid
  upload" rather than adding dynamic C/V unless further scenarios show a gain.

Artifacts:

- Capacity plans and NS3 logs: `balanced_capacity/`
- Experiment runs: `static_full_68mbps/`, `dynamic_full_68mbps/`,
  `static_cap5_68mbps/`, `dynamic_cap5_68mbps/`, `static_full_84mbps/`,
  `dynamic_full_84mbps/`, `static_cap5_84mbps/`, `dynamic_cap5_84mbps/`
- New scheduler file:
  `opencda/core/clustering/algorithms/resource_allocation/dynamic_cv_potential_game.py`
- Upload density cap implementation:
  `opencda/core/common/offline_replay.py`,
  `opencda/tools/offline_inference.py`
