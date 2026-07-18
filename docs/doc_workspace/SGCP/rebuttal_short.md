# SGCP Short Rebuttal Draft

更新时间：2026-07-17

本文档是 `rebuttal_draft.md` 的压缩版，面向最终 rebuttal 粘贴和字数控制。详细证据、命令和结果仍以 `rebuttal_draft.md`、`main_table_candidate.md`、`results.md` 和 `reproducibility_manifest.md` 为准。

## Opening

We thank the reviewers for the constructive comments. In the revision, we substantially tightened the experimental protocol and the claim boundary. We now separate centralized full-sharing upper references from fair RSU-free V2V baselines, replace the old main table with reproducible CARLA/OpenCDA/NS3 results, report AP together with measured upload Mbps, and add NS3 request-level delivery diagnostics. We also clarify SGCP as a perception-driven, capacity-constrained coalition and perception-aware potential-guided PPS framework, add a reproducible density-calibration protocol for `f(rho)`, report `rho_th`, `N_max`, and `T_min^stab` sensitivity, and soften the real-time claim to near-real-time control-plane feasibility.

## R2: Coalition Design, Full Clusters, Novelty, FullPerception

**Coalition value and late-fusion baseline.** The `max` term represents the best view already available without intra-cluster raw point-cloud sharing. Thus, the coalition value estimates the additional gain of early fusion over the strongest existing view, rather than additively counting all observations. We clarified this interpretation and now report head-only, full-cluster, selective V2V, and SGCP variants separately.

**Full clusters and fragmentation.** `N_max` is a hard capacity constraint. A vehicle can migrate only to a non-full coalition whose marginal perception gain exceeds its current contribution by a hysteresis factor. If all beneficial coalitions are full, the vehicle keeps its current coalition or remains a singleton; it still participates through inter-cluster late fusion. We do not merge coalitions beyond `N_max`. In the 41-frame dump with `N_max=4`, the average cluster size is 3.33, the singleton ratio is 0, and 99.15 candidate joins per frame are skipped due to capacity, showing that the constraint is active without causing singleton fragmentation.

**Novelty.** We revised the novelty statement to avoid claiming that generic coalition formation alone is new. SGCP integrates LiDAR density and coverage-complementarity utility, vehicular stability and capacity constraints, topology-triggered cluster retention, subchannel-feasible PPS, and hierarchical early/late fusion in CARLA/OpenCDA/NS3.

**FullPerception fairness.** The current dump is RSU-free, so full 20-CAV early fusion is reported only as a centralized upper reference: `0.85/0.83/0.48` AP at `118.71 Mbps`. FullPerception itself is evaluated through the repository PCS implementation (`pcs.py`), now giving `0.59/0.53/0.22` at `25.29 Mbps`. For fair RSU-free comparison, we add CAV-only selective-sharing baselines under the same dump, backbone, cluster-head late-fusion path, and comparable budgets. The strongest high-budget selective baseline reaches `0.80/0.76/0.40` at `73.58 Mbps`. Our PAPG main setting reaches `0.81/0.78/0.39` at `62.54 Mbps`, improving AP@0.3/AP@0.5 while using about 15.0% less payload; the full-sharing reference remains the upper bound.

## R3: Calibration, Stability Window, Baselines, Ablations

**`f(rho)` calibration.** We added a reproducible calibration protocol using the same 10 m LiDAR grids as SGCP. The 41-frame dump contains 788,020 CAV-grid samples; non-empty grids account for 5.98%, with non-empty density p90/p95 of 1.40/3.60 points/m2. `rho_th=2.0` lies between these percentiles and selects 7.18% of non-empty grids. The coverage-aware 10ch `rho_th=1/2/3/4` sweep gives AP/Mbps of `0.76/0.72/0.34/51.31`, `0.79/0.75/0.37/56.08`, `0.79/0.76/0.38/57.38`, and `0.79/0.76/0.38/58.22`. We explicitly state that `rho_th` is sensor-, grid-, preprocessing-, and detector-dependent.

**`T_min^stab` and `N_max`.** We no longer claim that `T_min^stab=500 ms` is empirically optimal. It is a conservative hysteresis default corresponding to five 10 Hz perception cycles. A sweep over 100/300/500/700/1000 ms gives identical AP and reconfiguration metrics in the current sequence, showing that the main result is not fragile to this parameter. For `N_max`, we explain the choice as a capacity-control tradeoff rather than pure AP tuning; `N_max=4` avoids singleton fragmentation while keeping the channel budget explicit.

**Fair baselines.** The original RandomRA and MWS schedulers are now treated as w/o-PPS diagnostics because their payloads are only 18.98/19.34 Mbps and they do not fully use the channel budget. We additionally add a forced-budget random baseline using the same coalition path, 3 uploaded members per head and 117-grid budget; it reaches `0.77/0.73/0.38` at `61.68 Mbps`. PAPG improves this by `+0.04/+0.05/+0.01` AP at nearly the same traffic. The fair main comparison also includes nearest, density-based and communication-aware selective variants, avoiding mixed centralized, RSU-assisted and RSU-free settings.

**Ablations.** We added mechanism probes: head-only late fusion gives `0.26/0.22/0.09`, SGCP grid-constrained fusion gives `0.77/0.73/0.35`, full-cluster upload gives `0.82/0.79/0.42`, random-grid same-link selection gives `0.78/0.75/0.36`, coverage-aware 10ch `rho_th=3` gives `0.79/0.76/0.38` at `57.38 Mbps`, and PAPG improves to `0.81/0.78/0.39` at `62.54 Mbps`. Object-level diagnostics show PAPG reduces full-reference-detected but SGCP-missed rows from 106 to 59. These results identify grid selection, source coverage and channel budget as the main AP/payload drivers.

**Object-level failure analysis.** We further inspected missed ground-truth boxes, associated them with BEV grids and candidate CAV point support, and tested object-grid routing probes. These probes can recover several individual high-IoU misses, but the best routing-hint run gained 4 GT rows while losing 15 previously detected rows. This is why the revision does not present ad-hoc cross-cluster repair as the main algorithm; instead, it uses PAPG as the stable decentralized setting and separates edge/global schedulers as a different capability class.

## R4: Runtime, Topology Trigger, NS3 Reliability

**Runtime.** We softened the claim from guaranteed end-to-end 100 ms closure to near-real-time control-plane feasibility. In the 20-CAV replay, SGCP control-plane computation averages 105.24 ms per profiled update: 64.39 ms for coalition formation and 40.58 ms for PPS. PPS converges in all 41 frames within three iterations. Offline file loading and world reconstruction are replay artifacts and are not counted as online control latency. We also clarify that coalition formation is topology-triggered, while PPS and perception metadata are refreshed per cycle.

**Topology trigger.** We clarified that each cycle refreshes beacon state, density metadata, and PPS scheduling, but cluster membership is updated only when the coalition becomes infeasible or sufficiently suboptimal. Trigger conditions include neighbor-set changes, head/member disconnection, relative-motion risk, link-quality degradation, utility drop, and a periodic guard. The stability window acts as hysteresis.

**NS3 reliability.** We now distinguish application callback delivery, RLC request completion, and PHY diagnostics. For the PAPG main setting, the first 11 replayed frames contain 110 scheduled requests. All 110 scheduled requests are application-delivered and RLC-complete, with 2,970/2,970 RLC TX/RX events and zero PHY failures. In a five-subchannel exposed-bandwidth regression, requests on legal subchannels complete, while out-of-window requests are rejected before CAM/RLC creation rather than being miscounted as successful receptions.

## Claim Boundary

The revised claim is intentionally more conservative: SGCP does not claim to dominate the centralized full-sharing upper bound or edge-assisted global assignment at every IoU threshold. Instead, it provides a decentralized, stability-aware and channel-feasible perception-sharing pipeline. The PAPG main setting improves over forced-budget random at nearly identical traffic, improves AP@0.3/AP@0.5 over the high-budget density selective baseline while using less payload, substantially reduces communication relative to centralized full sharing, and has request-level NS3 evidence that scheduled transmissions obey the intended subchannel constraints.
