# Dense Data Quality and Validation Notes

## Checks Passed

| Check | Status |
| --- | --- |
| Dense dataset exported with 20 CAVs and 41 frames per CAV | Passed |
| Dense LiDAR uses 32 channels, 320000 points/s, 20 Hz rotation, 50 m range | Passed |
| Table 1 protocol-native baselines rerun on dense data | Passed |
| Table 2 global box aggregation and fusion scaffold rerun on dense data | Passed |
| Table 3 SGCP-compatible scheduler comparison rerun on dense data | Passed |
| Table 4 clustering ablation rerun on dense data | Passed |
| Dense main tables include AP, raw/box/total communication, and GFLOPs where applicable | Passed |
| Table 1 AP uses scheduler-independent full-frame GT scope | Passed |
| SGCP dense headline uses per-link 60 ms deadline trimming | Passed |
| Guard/zero-time control-plane protocol documented as Guard `1 ms`, zero-time send delay `0 ms`, activation-synchronized send | Passed |
| Exact 40 MHz / 10ch NS3 TB sanity probe completed | Passed |
| Utility-surrogate vs AP@0.5 validation added for reviewer-facing density-proxy justification | Passed |
| Realtime breakdown added without unsupported detector-runtime CDF claims | Passed |
| Active-CAV scalability sweep rerun with `--gt-scope full-frame` | Passed |
| Hybrid round-robin dynamic scheduler probe added and recorded in Table 1/3 | Passed |

## Current Paper-Facing Dense Headline

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP dense headline candidate, hybrid scheduler | 0.88 | 0.82 | 0.61 | 28.42 | 0.89 | 29.31 | 593.34 |
| SGCP previous dense headline, dynamic_cv | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 |

Use the hybrid row as the current SGCP candidate when writing dense-LiDAR
results, while keeping the `dynamic_cv` row as continuity/provenance. Older
non-dense rows should not be mixed into dense tables.

## Caveats

| Item | Interpretation |
| --- | --- |
| 40 MHz TB probe median `912 B` vs estimator `899 B` | Same order and close enough for the current estimator; table rows keep the existing conservative `899 B` estimator. |
| NS3 probe return code | Direct TB probes terminate with timeout-related return code after the useful manual consume events are collected; the relevant evidence is `MANUAL_CMD_CONSUME` and allocated TB size. |
| Table 3 scheduler comparison | Interpret as AP/communication/compute tradeoff, not AP-only dominance: some dense schedulers spend more payload for small AP differences. |
| Channel-count sensitivity | Removed from the dense clean package; the fixed paper-facing setting is `40 MHz / 10ch`. |
| `N_max` sensitivity | Removed from the dense clean package; `N_max` is derived from `N`, `K`, and `B_h`. |

## Completed Late Additions

| Item | Result | Status |
| --- | --- | --- |
| EdgeCooper Pmax-style partial point-upload adaptation | Implemented and added as `EdgeCooper-Pmax` / `EdgeCooper-HD-Pmax`. Table 1 all-receiver full-GT row was rerun without `--local-preserving-output`: `0.40/0.37/0.26`, `86.30 Mbps`, `1790.59 GFLOPs/frame`. Table 2 global box row uses the same 820 active receiver-sample trace and recomputed GFLOPs plus estimated global NMS: `0.85/0.80/0.56`, `88.95 Mbps`, `1790.62 GFLOPs/frame`. Table 3 SGCP-compatible scaffold remains `0.88/0.81/0.56`, `33.46 Mbps`, `593.98 GFLOPs/frame` and should be treated as cluster-head receiver scope. | Complete for Table 1/2; Table 3 low payload is cluster-head scoped rather than all-20 receiver scoped |
| PACP-LiDAR Table 1/2/3 dense update | Aligned the LiDAR adaptation with PACP's BEV-match priority, perceptual-region utility, and compression/rate-control idea; added density-capped raw-LiDAR upload, K=2 endpoint matching, and reran 41 frames. Table 1 PACP-LiDAR full-GT row: `0.40/0.37/0.26`, `86.30 Mbps`, `1790.59 GFLOPs/frame`; Table 2 global box: `0.86/0.79/0.55`, `89.02 Mbps`, `1753.50 GFLOPs/frame` including estimated global NMS; Table 3 SGCP-compatible row: `0.88/0.82/0.59`, `33.48 Mbps`, `593.98 GFLOPs/frame`. | Complete |
| Utility-proxy validation | Utility gain has Pearson `0.846` with Table 3 AP@0.5, higher than selected utility-grid count alone (`0.780`). | Complete |
| Active-CAV scalability sweep | Reran Pure late, EdgeCooper-Pmax, and SGCP for `N=5/10/15` with `--gt-scope full-frame`; `N=20` reuses the full-frame GT main table. Results are in `12_cav_scale_sweep.md`. | Complete |
| Hybrid round-robin dynamic scheduler | Added independent scheduler implementation and 41-frame dense Town03 run. Result: `0.88/0.82/0.61`, `29.31 Mbps`, `593.34 GFLOPs/frame`; now recorded in Table 1/3 while retaining the old `dynamic_cv` row. | Complete |
| Fusion scaffold hybrid addendum | Added `ClusteredEarlyOnly hybrid` and `FullSGCP hybrid` to Table 2. Early-only hybrid gives `0.38/0.35/0.26`, `28.42 Mbps`; FullSGCP hybrid gives `0.88/0.82/0.61`, `29.31 Mbps`. | Complete |
| Table 4 clustering hybrid addendum | Reran random, distance-greedy, density-greedy, SeAC-inspired, and HHOCNET-inspired clustering under fixed hybrid scheduling; SGCP hybrid remains best AP row. | Complete |
| Town06 SGCP hybrid addendum | Reran PointPillar-Attentive and SECOND attentive-to-early SGCP rows with explicit full-frame GT and formula-aligned `N_max=3`. | Complete |
| SECOND Town03 SGCP hybrid full protocol | Reran SECOND attentive-to-early SGCP hybrid on Dense Roundabout and computed SECOND-calibrated GFLOPs. Result: `0.92/0.89/0.78`, `29.36 Mbps`, `1068.30 GFLOPs/frame`. | Complete |
| Hybrid rho Pareto sweep | Completed all `rho_th=1..5` and raw-budget `1/5/10/20/40 Mbps` points, plus extra `50 Mbps` points for `rho_th=3/4/5`. Best completed point remains `rho_th=2`, raw budget `40 Mbps`: `0.88/0.82/0.61`, `29.31 Mbps`, matching the hybrid main-table SGCP row. | Complete |

## Required Follow-Up If Hybrid Becomes Final SGCP

The new hybrid scheduler row is currently recorded in Table 1 and Table 3 as
the strongest SGCP candidate. The following experiments should be completed
before replacing `dynamic_cv` everywhere in the paper narrative.

| Priority | Experiment | Why it is needed |
| --- | --- | --- |
| P0 | Decide and lock the `N_max` rule for the 20-CAV main table | The current hybrid row uses `N_max=5`, while the derived capacity formula in the dense package gives `N_max=4` for `N=20`, `K=10`, `B_h=2`. Either rerun hybrid with the formula-derived `N_max=4`, or explicitly report `N_max=5` as the dense main-table protocol. |
| P0 | Lock final hybrid parameter-sensitivity wording | The rho Pareto sweep is complete and supports `rho_th=2`, raw budget `40 Mbps`; only final paper wording and figure/table placement remain. |
| P1 | Rerun realtime scheduler-solver profiling for hybrid | Payload/data-plane timing is measured for hybrid, but the scheduler solving mean/P95 currently uses the `dynamic_cv` profiler as a conservative proxy. |
| P1 | Recompute utility-proxy correlation with hybrid included | `11_utility_proxy_vs_ap.md` currently validates the pre-hybrid scheduler set. Add the hybrid point before using the correlation figure/table in the paper. |
| P1 | Rerun active-CAV scalability with hybrid | `12_cav_scale_sweep.md` uses `dynamic_cv` for SGCP. Rerun `N=5/10/15/20` if the scalability figure is paper-facing under hybrid. |
| P1 | Refresh any figure scripts that still read previous `dynamic_cv` SGCP rows | The markdown tables now include hybrid rows; plotting scripts, if used later, should explicitly select the hybrid SGCP candidate. |

## Optional Future Validation

| Item | Why it may help | Priority |
| --- | --- | --- |
| Exact NS3 replay for the final dense Table 1 SGCP rho-2 row | Would strengthen delay evidence beyond estimator trace; current trace max is already below 60 ms | Optional |
| Online CARLA/NS3 dense sanity run | Useful if reviewers demand online co-simulation evidence; expensive relative to offline replay | Optional |
