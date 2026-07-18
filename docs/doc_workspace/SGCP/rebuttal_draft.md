# SGCP Rebuttal Draft

更新时间：2026-07-18

本文档整理可直接迁移到 rebuttal 的答复素材。口径以当前可复现实验和 `C:\Workspace\icdcs-paper\SGCP\main.tex` 最新修订为准，避免继续使用旧主表中缺少日志支撑的 `0.84/0.69` 和 `22.33 Mbps` 叙事。

## 总体答复主线

We thank the reviewers for recognizing the motivation and the hierarchical design of SGCP. In the revision, we substantially strengthened the evaluation protocol and clarified several previously ambiguous assumptions. Specifically, we now distinguish centralized full-sharing upper references from fair RSU-free V2V baselines, add capacity-matched selective-sharing baselines, replace the old main table with reproducible CARLA/OpenCDA/NS3 results, report AP and measured upload Mbps together, and add NS3 request-level delivery diagnostics. We also clarify that PPS is a perception-aware potential-guided constrained scheduler with a coverage layer and an object-prototype target layer, add a reproducible density calibration protocol for `f(rho)`, report `rho_th` sensitivity, and explicitly describe topology-triggered cluster updates and runtime limits.

## R2: Coalition Baseline, Full Clusters, Merge/Split, Novelty, FullPerception

### Max-over-S baseline for coalition formation

**Concern:** Why use `max` as the baseline for coalition formation? Would this overestimate the benefit when late fusion already works across clusters?

**Draft response:**

Thank you for pointing out this ambiguity. The `max` term is intended to represent the best single-view or late-fusion quality already available without intra-cluster raw point-cloud sharing. The coalition value therefore estimates only the additional gain from early fusion beyond the strongest existing view, rather than counting all members' observations additively. We have clarified this interpretation in the formulation and now emphasize that late-fusion availability is the baseline being subtracted. In the revised experiments, this distinction is also reflected by reporting head-only, full-cluster, selective V2V, and SGCP coverage-aware variants separately.

### Full clusters and fragmentation

**Concern:** What happens when neighboring clusters are already at `N_max`? Does this create many small clusters? Is merge beyond `N_max` allowed?

**Draft response:**

We now explicitly define `N_max` as a hard coalition capacity constraint. A vehicle can migrate only to a non-full coalition whose marginal perception gain exceeds its current contribution by a hysteresis factor. If all beneficial coalitions are full, the vehicle keeps its current coalition or remains a singleton; it is not dropped from the perception pipeline, because singleton or small-cluster heads still participate in inter-cluster late fusion. We do not allow merges that exceed `N_max`. Split/merge behavior is instead handled through event-triggered coalition reformation under the same capacity constraint. In our 41-frame dump with `N_max=4`, the average cluster size is 3.33, the singleton ratio is 0, and the algorithm records 99.15 capacity-skipped candidate joins per frame, indicating that the capacity constraint is active without causing singleton fragmentation in the tested sequence.

### Recomputing member contribution

**Concern:** Does the algorithm revisit members after the leader density changes due to other uploads?

**Draft response:**

Yes. The coalition formation procedure is iterative. After a move, the cluster membership, sensing grids, request grids and leader state are updated; subsequent vehicles and subsequent iterations evaluate their current and marginal contributions using this updated coalition state. To avoid oscillation, a migration is accepted only when the gain exceeds the current contribution by the hysteresis factor. We clarified this update semantics in the mechanism description.

### Novelty vs. Smartform-like coalition formation

**Concern:** Coalition formation resembles methods from other domains such as smart-grid energy trading.

**Draft response:**

We agree that coalition formation as a general mathematical tool has appeared in other domains. The novelty of SGCP is not the generic use of coalition games alone. The contribution is the domain-specific integration of: (1) perception-density and coverage-complementarity utility derived from LiDAR grids; (2) motion-stability and capacity constraints for vehicular dynamics; (3) topology-triggered cluster retention rather than unconditional per-cycle reformation; (4) potential-guided PPS with hard subchannel, SINR and bandwidth feasibility; and (5) a hierarchical early/late fusion pipeline evaluated with CARLA/OpenCDA/NS3. We revised the wording to emphasize this integration and avoid implying that generic coalition formation itself is the sole novelty.

### FullPerception / RSU fairness

**Concern:** FullPerception is RSU-assisted, but the scenario has no RSU. Was a virtual RSU with perfect knowledge assumed? Would FullPerception degrade under decentralized information?

**Draft response:**

Thank you for identifying this important ambiguity. We revised the experimental protocol to separate centralized full-sharing references, the built-in FullPerception PCS baseline, and fair decentralized baselines. The current `v2xp_cluster_carla` dump is RSU-free and contains no RSU sensor stream. Full 20-CAV early fusion is therefore reported only as a centralized upper reference: it reaches 0.85/0.83/0.48 AP at 118.71 Mbps, assuming full point-cloud availability. FullPerception is evaluated through the repository PCS implementation (`pcs.py`), which now reaches 0.59/0.53/0.22 at 25.29 Mbps after wiring payload-based subchannel demand and scheduled links into the OpenCDA/NS3 protocol. For fair RSU-free comparison, we add CAV-only selective-sharing baselines under the same CARLA dump, OpenCOOD backbone, cluster-head late-fusion path and matched grid budgets. The strongest high-budget selective baseline reaches 0.80/0.76/0.40 at 73.58 Mbps, while SGCP with perception-aware potential-game scheduling reaches 0.81/0.78/0.39 at 62.54 Mbps. Thus, the revised comparison no longer relies on an RSU-assisted baseline as the main fairness claim, and the full-sharing result is kept as an upper reference rather than a decentralized competitor.

## R3: Calibration, Stability Window, Baselines, Ablations

### `f(rho)` calibration reproducibility

**Concern:** The density utility calibration is too brief.

**Draft response:**

We added a reproducible density-calibration protocol. We reconstruct the same 10 m global LiDAR grids used by SGCP and measure `rho` as points per square meter for each CAV and frame. In the 41-frame CARLA dump, this gives 788,020 CAV-grid density samples. Non-empty grids account for 5.98% of all samples, with non-empty density p90/p95 equal to 1.40/3.60 points/m². The default `rho_th=2.0` lies between these percentiles and selects 7.18% of non-empty grids as high-density candidates. We also added a coverage-aware 10ch `rho_th` sweep: `rho_th=1/2/3/4` gives AP/Mbps of `0.76/0.72/0.34/51.31`, `0.79/0.75/0.37/56.08`, `0.79/0.76/0.38/57.38`, and `0.79/0.76/0.38/58.22`, respectively. The revision explicitly states that `rho_th` depends on LiDAR resolution, grid size, preprocessing and detector backbone, and should be recalibrated when these components change.

### Stability window `T_min^stab=500 ms`

**Concern:** 500 ms appears arbitrary; add parameter study.

**Draft response:**

We agree and no longer claim that 500 ms is an empirically optimal value. We treat it as a conservative hysteresis default corresponding to five 10 Hz perception cycles. We added a sensitivity study over 100/300/500/700/1000 ms. In the current 41-frame dump, AP and reconfiguration metrics are unchanged across this range, indicating that the reported main result is not fragile with respect to this parameter in the tested sequence. We also clarified that more aggressive traffic dynamics require a separate tuning study.

### Fair and decentralized baselines

**Concern:** Baselines are centralized or too simple; add decentralized baselines.

**Draft response:**

We revised the baseline set. The original RandomRA and MWS schedulers are now treated as w/o-PPS diagnostics because they either under-utilize the payload budget or remain weak after sharing the tuned FullPerception blind-spot units. For fair main comparison, we add capacity-matched CAV-only selective-sharing baselines: forced-budget random, nearest, density-based and communication-aware variants. Forced-budget random uses the same coalition path with 3 uploaded members per head and a 117-grid budget, reaching 0.77/0.73/0.38 at 61.68 Mbps; PAPG reaches 0.81/0.78/0.39 at 62.54 Mbps. The communication-aware variant can incorporate NS3 request-level RLC completion as link-quality cost. We also distinguish centralized full-sharing references, infrastructure-assisted references and fair decentralized baselines. EdgeCooper-HD reaches 0.81/0.78/0.42 at 65.40 Mbps, but it uses edge/global assignment information, so it is reported as an infrastructure-assisted reference rather than a fully decentralized V2V baseline. This revision directly addresses the concern that the old comparison mixed RSU-assisted, centralized and RSU-free settings.

### Ablation studies

**Concern:** More ablations are needed.

**Draft response:**

We added mechanism probes and ablations to isolate the contribution of each component. Head-only late fusion gives 0.26/0.22/0.09; SGCP grid-constrained fusion gives 0.77/0.73/0.35; full-cluster upload gives 0.82/0.79/0.42; random-grid same-link selection gives 0.78/0.75/0.36; coverage-aware spatial-diverse grid selection improves to 0.79/0.76/0.38 at 57.38 Mbps under 10ch; and the final PAPG scheduler reaches 0.81/0.78/0.39 at 62.54 Mbps. Object-level diagnostics show that PAPG reduces full-reference-detected but SGCP-missed rows from 106 to 59. The channel sweep further shows 5/10/20ch AP of 0.56/0.53/0.27, 0.79/0.75/0.37, and 0.80/0.76/0.41 for the coverage-aware predecessor, respectively. These results identify grid selection, source coverage and channel budget as the main AP/payload tradeoff drivers.

## R4: Runtime, Generalization, Baselines, Topology Trigger

### 100 ms feasibility

**Concern:** Multiple game-theoretic iterations may be too expensive for a strict 100 ms cycle.

**Draft response:**

We added a millisecond-level runtime breakdown and softened the claim from guaranteed end-to-end 100 ms closure to near-real-time feasibility of the control-plane prototype. In the 20-CAV replay, SGCP control-plane computation averages 105.24 ms per profiled update: 64.39 ms for coalition formation and 40.58 ms for PPS scheduling. PPS itself converges in all 41 frames within three iterations. Offline file loading and world reconstruction add 599.73 ms but are replay artifacts and are not counted as online control latency. The revised mechanism also clarifies that coalition formation is topology-triggered rather than executed every cycle, so the 64.39 ms coalition cost is not paid in every sensing cycle; PPS and perception updates remain per-cycle.

### Generalization of density utility

**Concern:** The density utility is calibrated for PointPillars and may not generalize.

**Draft response:**

We agree and explicitly state this limitation in the revision. The density utility is detector/sensor/grid-size dependent. Our contribution is not a universal density threshold, but a reproducible calibration-and-sensitivity protocol. When the LiDAR resolution, grid size, point-cloud preprocessing or detector backbone changes, the density distribution and `rho_th` should be recalibrated. We added this boundary to avoid overstating the generality of the PointPillars-specific calibration.

### Topology-change trigger vagueness

**Concern:** The criteria for significant topology changes are vague and could cause instability.

**Draft response:**

We clarified the trigger semantics. Each cycle refreshes beacon state, density metadata and PPS scheduling, but cluster membership is updated only when the current coalition becomes infeasible or sufficiently suboptimal. Trigger conditions include neighbor-set changes, head/member disconnection, relative-motion risk, link-quality degradation, utility drop and a periodic guard. The stability window acts as hysteresis, so non-hard-failure triggers do not immediately force reconfiguration. This resolves the ambiguity between event-triggered cluster updates and per-cycle resource scheduling.

### NS3 reliability metrics

**Concern addressed proactively:** Link reliability should not be represented by one ambiguous `cam_received` count.

**Draft response:**

We revised the NS3 evaluation to distinguish application callback delivery, RLC request completion and PHY decode diagnostics. For the PAPG main setting with `rho_th=3.0`, the first 11 replayed frames contain 110 scheduled requests. All 110 scheduled requests are delivered at the application callback level and are RLC-complete, with 2,970/2,970 RLC TX/RX events and zero PHY decode failures. We also verified the bandwidth-window behavior: when only five target subchannels are exposed, requests mapped to `sc_start=0..4` complete and requests mapped to `sc_start=5..9` are rejected before CAM/RLC creation, rather than being miscounted as successful receptions.

## Short Consolidated Rebuttal Opening

We appreciate the reviewers' constructive comments. The main revision addresses three categories of concerns: experimental fairness, reproducibility and practical feasibility. We replaced the old main table with reproducible CARLA/OpenCDA/NS3 results, separated centralized full-sharing references from fair RSU-free V2V baselines, added capacity-matched selective-sharing baselines including forced-budget random, and reported AP together with measured upload Mbps. We added a reproducible `f(rho)` calibration protocol and `rho_th` sensitivity table, clarified that density thresholds are sensor/detector specific, and revised the scheduling mechanism as perception-aware potential-guided PPS with a coverage layer and an object-prototype target layer. We also added request-level NS3 diagnostics and a runtime breakdown, and clarified topology-triggered cluster updates. These changes make the claims more conservative but better aligned with the implemented protocol and the reviewers' concerns.
