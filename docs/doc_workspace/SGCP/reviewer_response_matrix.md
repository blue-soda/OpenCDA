# SGCP Reviewer Response Matrix

更新时间：2026-07-19

本文档把 `C:\Workspace\icdcs-paper\SGCP\SGCP-review.txt` 中的核心意见映射到当前证据、论文修改位置和剩余风险。用途是防止 rebuttal / `main.tex` 修订时遗漏审稿点，尤其是 baseline、公平性、参数依据和机制边界。

## 总体策略

- 不再恢复旧主表的强 claim。旧 `0.84/0.69, 22.33 Mbps` 缺少完整日志、种子和代码版本，当前修订采用可复现 41 帧 dump 与 PAPG 主线。
- 主表采用分组布局：full-sharing / infrastructure-assisted references 与 RSU-free V2V baselines 分开，避免把 EdgeCooper-HD、global selective proxy 和 Full 20-CAV upper reference 混成公平去中心化 baseline。
- SGCP 的论文主张收束为：在 RSU-free V2V 约束、明确子信道调度和 NS3 request-level delivery 下，PAPG 提供更好的低/中 IoU AP 与 payload tradeoff；不声称全面击败 centralized full sharing 或 edge-assisted global assignment。
- 所有主文 Table/Figure 进入论文前必须能追溯 artifact；当前 `paper_number_audit_20260719` 已核查 Table 1/3/4 数值与 manifest 一致，Table 1/3 manifests 已显式记录 `10 ch / 20 MHz`。
- Detector/checkpoint 风险按统一检测器口径处理：旧 `pointpillar_early_fusion` 主线保留为 legacy reference；当前 forward-writing Table 1/2/3/Figure 1/2/3/4 已统一切到 attentive checkpoint candidate。actual-late 和 COSDH 仍只作 sensitivity/negative probe，不替换主表。

## Reviewer 2

| Concern | Current Response | Evidence / Location | Remaining Risk |
| --- | --- | --- | --- |
| Why use `max` over members as coalition baseline? Does it overestimate gain when late fusion already helps? | `max` 表示不做 intra-cluster raw sharing 时已有的最佳单视角/late-fusion view，coalition value 衡量 early fusion 的边际增益，而不是累加所有成员贡献。 | `main.tex` coalition value 段；`paper_revision_plan.md`; `rebuttal_short.md` R2。 | 公式仍偏抽象，终稿可再加一句 “not an additive late-fusion gain”。 |
| What if surrounding clusters are full? How prevent fragmentation? | `N_max` 是硬容量约束；车辆只能加入未满且有正边际收益的 coalition。若都不满足，则保留原簇或 singleton，并通过 inter-cluster late fusion 参与全局感知。 | `cluster_capacity_policy.md`; `status.md`; `rebuttal_short.md` R2；41 帧 `N_max=4` 平均 cluster size 3.33、singleton ratio 0。 | 目前 replacement/repair 是机制说明而非主实验重点；如篇幅允许，附录列 capacity statistics。 |
| Does cluster merging exceed `N_max`? | 不允许超过 `N_max` 的 merge；split/merge 通过 topology trigger + coalition reformation 间接发生。 | `cluster_capacity_policy.md`; `target.md` P3; `rebuttal_short.md` R2。 | 无明显风险。 |
| Does algorithm revisit members after leader density increases? | coalition formation 迭代会在更新后的 coalition state 上重算边际贡献；PPS 每周期重算 density/grid utility。 | `target.md` P3; `status.md`; `main.tex` every-cycle PPS 描述。 | 若 reviewer 要求代码级日志，可补 Delta utility trace；当前 rebuttal 可先文字说明。 |
| Similarity to Smartform / generic coalition formation. | novelty 不再声称 coalition game 本身新，而是强调 perception-density utility、motion stability、capacity constraint、PPS subchannel feasibility 和 hierarchical fusion 的组合。 | `related_work_novelty_revision.md`; `main.tex` intro/related work; `rebuttal_short.md` R2。 | 需要确保 final related work 引用 Smartform 或至少在 rebuttal 直接回应。 |
| FullPerception is RSU-assisted; no RSU scenario how simulated? Virtual RSU? | full 20-CAV early fusion 改为 upper reference，不再命名为 FullPerception。Formal FullPerception baseline 使用仓库 `pcs.py` / `fullperception_pcs`；已对照原论文确认 Class-A common-node conflict，包括同 receiver 冲突。当前 attentive Table 1 使用 paper-faithful PCS scheduling + raw-LiDAR full-sender adaptation：`0.63/0.49/0.17` at `32.06 Mbps`；严格 blind-spot grid replay `0.56/0.41/0.18` at `11.22 Mbps` 作为边界结果。 | `fullperception_pcs_paper_audit.md`; `fullperception_baseline_revision.md`; `main.tex` 主表；`paper_artifact_index.md`; `artifacts/attentive_protocol_20260719`。 | FullPerception-PCS 不是强 SGCP-compatible scheduler baseline；文中已说明它不应与 full-sharing upper reference 或 Table 3 scaffold comparison 混淆。 |

## Reviewer 3

| Concern | Current Response | Evidence / Location | Remaining Risk |
| --- | --- | --- | --- |
| `f(rho)` calibration too brief. | 已补 788,020 CAV-grid samples、non-empty ratio、p90/p95、`rho_th` sensitivity，并明确 detector/sensor/grid-size dependent。 | `f_rho_calibration.md`; `parameter_calibration_revision.md`; `main.tex` parameter section; `rebuttal_short.md` R3。 | 仍缺跨传感器/跨 detector 泛化；应写作 limitation/future，不声称通用常数。 |
| `T_min^stab=500 ms` arbitrary. | 改为保守 hysteresis 默认，对应 5 个 10 Hz perception cycles；100/300/500/700/1000 ms sweep 在当前 dump 上不敏感。 | `parameter_calibration_revision.md`; `main.tex`; `rebuttal_short.md` R3。 | 该序列不够动态，不能证明 500 ms 最优；必须保守表述。 |
| Baselines insufficient; lack latest decentralized CP. | 增加 protocol-native Table 1、SGCP-compatible scheduler Table 3 和 Pareto 分层：forced-budget random、density/link-aware、PACP-style LiDAR proxy、EdgeCooper-HD proxy、FullPerception-PCS、Pure late prediction-sharing reference 与 Full20Early upper reference 均有 attentive artifact。 | `baseline_fairness.md`; `baseline_reproduction_plan.md`; `paper_artifact_index.md`; `artifacts/attentive_protocol_20260719`; `artifacts/attentive_scheduler_comparison_20260719`; `main.tex`。 | Where2Comm/PACP 等严格模型级复现未完成；当前写法必须强调 same-backbone raw-LiDAR / proxy baseline 边界，不写成严格复现所有模型。 |
| Need ablation experiments. | 已有 fusion scaffold ablation、scheduler comparison、Pareto、parameter sensitivity、checkpoint sensitivity、object-level failure diagnosis 和 NS3 replay。Clustered early-only 到 Full SGCP 证明 late fusion 覆盖贡献；Full20Early 展示 high-IoU localization 上界。 | `fusion_scaffold_claim_audit.md`; `pareto_claim_audit.md`; `detector_checkpoint_sensitivity_manifest.csv`; `failure_diagnostics.md`; `results.md`; `main.tex`。 | 正文需要保守写 AP@0.7：当前 SGCP 不是 AP@0.7 frontier，定位能力仍是 detector/raw-sharing headroom。 |
| Add parameter study for stability window. | 已做 `T_min^stab=100/300/500/700/1000 ms`，结果不敏感，作为鲁棒性而非最优性证据。 | `parameter_calibration_revision.md`; `paper_revision_plan.md`; `main.tex`; `rebuttal_short.md` R3。 | 同上，需强调当前 sequence 边界。 |

## Reviewer 4

| Concern | Current Response | Evidence / Location | Remaining Risk |
| --- | --- | --- | --- |
| Multiple game iterations may not fit strict 100 ms. | claim 已从 guaranteed end-to-end 100 ms 改为 near-real-time control-plane feasibility；报告 coalition/PPS profiling、PPS 3 iterations、control overhead。 | `runtime_feasibility_revision.md`; `paper_revision_plan.md`; `main.tex`; `rebuttal_short.md` R4。 | 当前不含 OpenCOOD detector runtime，不应写 full end-to-end 100 ms guarantee。 |
| Density utility may overfit PointPillars/sensor. | 明确 `rho_th` 是 detector/sensor/grid dependent，作为 system metadata 标定，不是通用常数。 | `f_rho_calibration.md`; `parameter_calibration_revision.md`; `rebuttal_short.md` R3/R4。 | 缺跨 detector 实验；以 limitation 处理。 |
| Evaluation mostly centralized/basic scheduling. | 主表分层并加入 capacity-matched V2V selective baselines；EdgeCooper-HD 放 infrastructure-assisted group；PAPG 与 forced random、communication-aware、density、PACP-style LiDAR proxy 在同一 clustered scaffold 下比较。 | `main_table_candidate.md`; `baseline_fairness.md`; `artifacts/scheduler_comparison_20260719/scheduler_comparison_manifest.csv`; `main.tex`; `rebuttal_short.md` R3/R4。 | EdgeCooper-HD / PACP-LiDAR AP@0.7 更强；当前叙事必须承认 edge/global 或 stronger-priority assumption 的高 IoU 优势。 |
| Topology trigger vague; may be unstable. | 已定义 neighbor-set change、head/member disconnection、relative-motion risk、link-quality degradation、utility drop、periodic guard；cluster membership event-triggered，PPS every cycle。 | `topology_trigger.md`; `online_topology_gate_regression.md`; `main.tex`; `rebuttal_short.md` R4。 | 在线 gate smoke 未显示 reduced reconfiguration，因为默认 35 m 范围持续触发 unreachable；正文不应过度声称在线稳定性收益。 |
| NS3 reliability / co-simulation realism. | 区分 application callback、RLC request completion、PHY diagnostics。PAPG 11 帧 110/110 application/RLC complete、0 PHY failures；5-subchannel regression 正确 reject out-of-window requests。Online CARLA/NS3 时间同步 bug 已修，但主表采用离线 final-delivery 口径。 | `online_ns3_short_regression.md`; `main_table_candidate.md`; `status.md`; `rebuttal_short.md` R4。 | 在线 CARLA+NS3 deadline-aware AP 与离线 final-delivery AP 仍不同；论文需明确两种口径，不用在线少量 CP 帧替换主表。 |

## Current Rebuttal Must-Say Points

1. We corrected the old FullPerception/full-sharing naming error.
2. We added capacity-matched V2V baselines and no longer rely on under-budget RandomRA/MWS as the main fairness evidence.
3. PAPG is the main SGCP method; routing hints / ISPG / CCISPG are diagnostic only.
4. EdgeCooper-HD is an infrastructure-assisted reference, not a fully decentralized V2V baseline.
5. NS3 reliability is reported at three layers: application callback, RLC request completion, and PHY diagnostics.
6. `T_min^stab=500 ms` and `rho_th` are calibrated/default engineering parameters with sensitivity evidence, not universal optima.
7. Runtime claim is control-plane near-real-time, not full detector-inclusive 100 ms guarantee.
8. Pure late is a prediction-sharing reference with detection-box overhead, not a zero-communication raw-LiDAR baseline.
9. Table 1 now keeps a common attentive checkpoint across SGCP, Pure late and baselines; legacy early-checkpoint results remain reproducibility references, while actual-late and COSDH stay sensitivity/negative probes.

## Remaining Before Final Paper Freeze

- Run full PDF compile and visual check once LaTeX toolchain is available.
- Decide whether to include a compact appendix table for `T_min^stab`, `N_max`, and `rho_th` sweeps.
- Check final `main.tex` for any remaining unconditional phrases such as “outperforms all baselines”, “guaranteed 100 ms”, or “exact potential game” outside the constrained wording.
- If space permits, add one sentence in limitations: density utility requires recalibration when sensor, grid size, preprocessing, or detector changes.
- If early-fusion fine-tune finishes on `mindspore-187`, create a new artifact directory and rerun SGCP/Pure late/Full20Early before changing any main-table number.
