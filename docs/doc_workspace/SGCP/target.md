# SGCP 任务清单

更新时间：2026-07-21

最终目标：完成 SGCP 论文审稿意见响应与实验重构，使所有图表有效、可解释、保护论文叙事，并能证明 SGCP 在大规模 V2V 协同感知中以合理通信量获得更好的 aggregate AP。

## 当前原则

- 主指标统一为 aggregate AP@0.3 / AP@0.5 / AP@0.7 和 Mbps。Aggregate AP 指把所有 evaluated receiver-frame samples 的预测框和 GT 框累计进同一个 evaluator 后统一计算 AP，不是 per-CAV AP 的简单平均；ego-only AP 只作为附录 sanity。
- SGCP 的叙事是大规模多 CAV 协议：分簇降低点云通信复杂度，簇内 early fusion 支撑 AP@0.7，簇间 late fusion 支撑覆盖率和 AP@0.3，PAPG/PPS 在通信受限时选择关键 sender/grid。
- FullPerception 与 EdgeCooperV2V+ 在本文复现中只允许 RSU/edge 做调度，不允许使用 RSU 点云；它们属于 V2V data-sharing baseline。
- `SGCP-compatible scheduler comparison` 只能说明“在同一 clustered two-layer fusion scaffold 中哪个调度器更好”，不能混写成完整系统比较。
- 若任何主文图表结论较差或无法解释，必须先诊断原因；必要时修改 SGCP 算法、调整合理测试场景或重新设计实验协议。不得用不可解释的单点结果硬写论文。
- 负面结果必须记录到 `log.md` / `results.md`，但只有可解释且支撑叙事的图表进入 `main.tex`。

## 已完成工作压缩摘要

- 已建立 `v2xp_cluster_carla` 离线数据导出/导入、OpenCOOD 推理、SGCP 离线 replay、offline NS3 replay 与 trace 记录能力。
- 已修复 NS3 manual subchannel、OpenCDA/NS3 时间同步、在线初始化顺序、scheduler 残留信道等关键 bug；细节见 `online_ns3_short_regression.md`、`protocol_audit.md`、`reproducibility_manifest.md`。
- 已确认 FullPerception baseline 对应 `pcs.py` / `fullperception_pcs`；full 20-CAV early fusion 只是 upper reference，不再命名为 FullPerception。
- 已实现并记录 forced random、density/greedy、communication-aware、PACP-style LiDAR、EdgeCooper-inspired V2V、EdgeCooper-HD proxy、PAPG 等 scheduler/baseline 结果；细节见 `baseline_reproduction_plan.md`、`baseline_fairness.md`、`main_table_candidate.md`、`results.md`。
- 已完成 PAPG 主线 first version：20MHz/10ch/rho3/`B_h=2`，41 帧结果 `0.81/0.78/0.39`、62.54 Mbps，11 帧 NS3 scheduled replay 110/110 application/RLC complete。
- 已完成多轮漏检诊断和负面算法 probe：OAPG、ISPG、CCISPG、routing hints 等；当前结论是 PAPG 主线可写，但仍需通过更清晰图表证明分簇、点云划分、两层融合和调度贡献。

## P0：实验协议重构与主图表落地

- [x] 建立统一评估脚本/清单第一版：`opencda.tools.sgcp_aggregate_ap_manifest` 从 stdout log 和 trace CSV 生成表格 manifest，记录 AP、evaluated samples、receiver policy、late fusion scaffold、payload/Mbps 和 artifact 路径。
- [x] 明确并固化 aggregate AP 统计口径：所有 protocol-native、fusion ablation、scheduler comparison 和 Pareto 图均使用 pooled evaluator AP；每个结果必须记录 evaluated samples 数、receiver policy、是否 inter-cluster late fusion，以及是否为 ego-only sanity。不再保留独立 aggregate AP 指标文档，口径直接写入 `target.md` / `status.md` / `results.md`。
- [x] 删除 satisfaction rate 作为主文指标：不再新增或使用 `satisfaction_metric.md` / `sgcp_satisfaction_summary`，后续图表只报告 aggregate AP、Mbps 和必要的辅助统计。
- [x] 所有实验在结果进入论文前必须生成 artifact：命令、stdout/log 路径、trace CSV、manifest/summary CSV、图表源数据和 git commit。当前论文草稿第一版 artifact 索引已建立：`paper_artifact_index.md` 与 `artifacts/paper_artifact_index_20260719/paper_artifact_index.csv`；后续任何新 checkpoint/新场景/新主表数值必须追加新 artifact。
- [x] 如果当前 41 帧场景无法支撑某张关键图表，重新导出更合适的 CARLA 场景；注意 CARLA 进程至多一个，数据路径仍遵循 `docs/doc_workspace/environment.md`。已新增 `scenario_sufficiency_audit.md`：当前 41 帧场景足以支撑 first-pass 主文图表和附录证据，暂不重新导出；仅在 checkpoint 回收后仍无法支撑主张、需要更强动态稳定性/密度实验或在线端到端证据时触发新场景。

## P1：Table 1 - Protocol-Native System Comparison

目的：比较完整系统协议，而不是单个调度器。每个方法按自己的原生机制运行；FullPerception/EdgeCooperV2V+ 只使用 RSU/edge 调度，不使用 RSU 点云。

必须包含：

- [x] Head-only：单车感知下界。
- [x] Pure late fusion：所有 CAV 本地检测框 late fusion，不做点云共享；用于展示 late fusion 覆盖能力。
- [x] FullPerception-PCS V2V：`pcs.py` / `fullperception_pcs`，按论文 PCS 口径运行，必要时继续修正 PCS 复现质量。
- [x] EdgeCooperV2V+：V2V 数据共享、edge/RSU 只调度；不能使用 SGCP 分簇作为原生设置，若使用 proxy 必须标注。
- [x] SGCP full：分簇 + PAPG/PPS 点云划分 + 簇内 early fusion + 簇间 late fusion。
- [x] Full 20-CAV early fusion upper reference：只作上界，不作为 baseline。
- [x] FullPerception-PCS deterministic rerun：2026-07-20 发现 `pcs.py` 存在 id 类型自环过滤和 blind-spot `set.pop()` 非确定性问题。已完成 deterministic PCS singleton 41-frame no-late 与 all-cavs global-box rerun；两者 295 条非零 scheduled links 完全一致，payload 均为 `10,779,344` bytes / `21.03 Mbps`。no-late AP 为 `0.14/0.13/0.06`，global-box AP 为 `0.83/0.77/0.38`。

指标：

- [x] Network AP@0.3 / AP@0.5 / AP@0.7。
- [x] Evaluated sample count and receiver-frame scope。
- [x] Mbps / payload bytes。
- [x] 必要时补 delivery sanity 一句话，不做单独 NS3 表格。

验收标准：

- [x] 表格必须清晰解释 SGCP 的完整系统优势，尤其是 aggregate AP@0.3 的大场景覆盖收益和 AP@0.7 的局部精度收益。已新增 `protocol_native_claim_audit.md`，确认当前 `main.tex` 将收益归因到 clustering、point-cloud selection、early/late two-layer fusion，而不是单一 scheduler。
- [x] 若 SGCP 只靠 late fusion 在 AP@0.3 大幅领先，需要在正文明确这是系统协议优势，不把它写成单纯 scheduler 优势。当前 `main.tex` 已将 late fusion 写成 network-level coverage / low-IoU recall 机制，并把 scheduler comparison 独立成 Table 3。
- [x] 若 FullPerception-PCS 或 EdgeCooperV2V+ 复现结果异常低，先核查算法、参数和场景，不直接拿弱结果进主表。当前审计结论：FullPerception-PCS 是 repaired/tuned built-in PCS baseline；EdgeCooper-HD 是 edge-assisted/global assignment reference，二者均在表格和正文中标注信息条件。
- [x] Pure late fusion 当前 `0.82/0.76/0.37` 且 0 点云 payload 很强；论文写作必须计入 detection-box exchange overhead 或将其明确为 prediction-sharing reference，否则会削弱 SGCP 通信优势叙事。已新增 `late_fusion_box_comm.md` 与 `opencda.tools.sgcp_late_box_comm_budget`：20MHz/10ch 下 detection-box broadcast 约 `0.74-1.13 Mbps`，scheduled all-to-all unicast 约 `14.04-21.52 Mbps` 且 100 ms 内可完成；只有 unscheduled all-to-all 随机抢信道模型会因碰撞失败。
- [x] 修正 Pure late baseline 口径：当前 P1/P2 manifest 中的 Pure late 使用 `fusion_method=early` + singleton CAV + box-level late NMS，不是 `pointpillar_late_fusion` checkpoint。已新增 `detector_checkpoint_fairness.md`，明确主表采用 early-checkpoint singleton controlled Pure late + `naive_late_fusion()`，actual-late checkpoint 只作 detector sensitivity / prediction-sharing reference。
- [x] 统一 detector/checkpoint 公平性：SGCP 论文主线的“点云 -> 检测框”统一使用 `pointpillar_early_fusion`；Pure late 使用同 checkpoint 的 singleton local inference + `naive_late_fusion()`。已补 “all late detector” sanity：Pure late actual late `0.89/0.83/0.49`，forced SGCP PAPG late-detector `0.87/0.81/0.48`、62.54 Mbps；后者不再是严格 SGCP early-fusion 协议，只能作为 checkpoint sensitivity。
- [x] 固定 detector/checkpoint 主线：不再等待远端 early-fusion fine-tune，当前论文和 rebuttal 暂定使用 attentive forward-writing candidate。此前 `mindspore-187` watcher 因 GPU 长期占用未启动训练，已按用户要求手动停止；若未来重新训练 checkpoint，必须作为新任务开启，并新建 artifact 版本重跑 SGCP/Pure late/Full20Early/Table 1/2/3/4/Figure 1/2/3。
  - [x] Late checkpoint 直接替换 early detector 已判负：11 帧 `0.58/0.48/0.15`。
  - [x] Attentive intermediate checkpoint 作为 early detector 已完成全图表 candidate 重跑：41 帧 SGCP-PAPG `0.87/0.81/0.36`，Pure late attentive controlled `0.82/0.65/0.28`，Full20Early attentive upper reference `0.88/0.85/0.45`。后续写作默认使用 attentive candidate；legacy `pointpillar_early_fusion` 结果降级为 checkpoint-reference artifacts。
  - [x] 补齐 attentive 下的关键 EdgeCooperHD 对照：EdgeCooperHD attentive 41 帧 `0.85/0.74/0.35`、`65.40 Mbps`，SGCP-PAPG attentive `0.87/0.81/0.36`、`62.54 Mbps`，说明同 detector/checkpoint 口径下 SGCP 可同时优于 Pure late attentive 和 EdgeCooperHD attentive。该结果已扩展为完整 Table/Figure artifact set，当前作为 forward-writing candidate。
  - [x] 重跑 attentive Table/Figure：Table 1 `attentive_protocol_20260719`，Table 2 `attentive_fusion_ablation_20260719`，Table 3 `attentive_scheduler_comparison_20260719`，Figure 1 `pareto_attentive_20260719`，Figure 2/3/4 `figures_attentive_20260719`。PACP-LiDAR attentive `0.88/0.79/0.37`、`86.56 Mbps`，可作为高通信参考；SGCP attentive 保持 AP@0.5 最强且通信更低。
  - [x] 重新核查 FullPerception-PCS attentive 异常行：对照原论文确认“同一个接收方的不同发送方”属于 Class A common-node conflict，不能放宽冲突图。已新增 `fullperception_pcs_paper_audit.md`；主文 Table 1 改用 paper-faithful PCS scheduling + raw-LiDAR full-sender adaptation，41 帧结果为 `0.63/0.49/0.17`、`32.06 Mbps`。严格 blind-spot grid replay 为 `0.56/0.41/0.18`、`11.22 Mbps`，`4.99 Mbps` 低 payload 点降级为诊断结果。
  - [x] COSDH compatible-weight transplant 已判负：11 帧 `0.00/0.00/0.00` 或 `0.02/0.00/0.00`。
  - [x] COSDH 实模型已复制必要代码并跑通 1 帧 collapsed smoke test；进一步用 `--debug-opencood-output` 和 `score_threshold=0.01/0.005/0.003` 诊断后确认 `psm` 置信度极低且正式 postprocess 仍无最终框。该路线完成 first-pass 判定：暂不扩展 11/41 帧，不进入主表；后续仅作为低优先级 calibration/debug。

## P2：Table 2 - Fusion Scaffold Ablation

目的：证明 SGCP 复杂机制必要性：early fusion 支撑精定位，late fusion 支撑覆盖，分簇和点云划分共同降低通信量。

必须包含：

- [x] Head-only：无共享、无 late fusion。
- [x] Pure late fusion：全局本地检测框 late fusion，无点云共享。
- [x] One-cluster early-only：1 个簇头 + 19 个簇成员，全局点云通信后 early fusion，无 late fusion；用于观察无分簇的 early fusion 上限/通信代价。
- [x] Clustered early-only：SGCP 分簇 + 簇内 early fusion，无簇间 late fusion；用于证明只做簇内 early fusion 覆盖不足。
- [x] One-cluster early + late：无分簇/全局通信设置下的 early + late 对照；用于区分两层融合和分簇本身的贡献。当前 one-cluster late fusion 为 identity，复用 one-cluster early artifact。
- [x] Full SGCP：分簇 + 点云划分 + early + late。

不做核心消融：

- [x] 不把 `Clustered late-only` 作为核心表格行；分簇主要服务点云通信和 early fusion，clustered late-only 机制不自然，若保留仅放附录诊断。

验收标准：

- [x] AP@0.7 的提升应能解释为点云 early fusion / 高质量局部融合贡献。已新增 `fusion_scaffold_claim_audit.md`：当前证据只能支持保守口径，即 full raw-sharing upper reference `0.48` 展示 early-fusion localization headroom，Full SGCP 相比 controlled Pure late 在 AP@0.5/AP@0.7 有小幅收益；不能写成当前 SGCP 已全面解决高 IoU。
- [x] AP@0.3 的提升应能解释为 late fusion / 多区域覆盖贡献：同 payload 下 clustered early-only `0.38/0.36/0.20` 到 Full SGCP `0.81/0.78/0.39`。
- [x] Full SGCP 必须在通信量远低于 one-cluster/full-sharing 设置时保持有竞争力的 aggregate AP。Full SGCP 使用 `62.54/118.71=52.7%` 的 full-sharing raw payload，保留 full-sharing AP@0.3/AP@0.5 的 `95.3%/94.0%`；AP@0.7 保留 `81.3%`，作为 localization/checkpoint headroom。
- [x] 若该表不能证明分簇或两层融合有效，优先修改算法、场景或评估口径，再进入论文。审计结论：该表可以证明 inter-cluster late fusion 的 coverage 贡献和 SGCP 的中等通信 AP@0.3/AP@0.5 竞争力；AP@0.7 只写边界，不写全面最优。

## P3：Table 3 - SGCP-Compatible Scheduler Comparison

目的：在相同 clustered two-layer fusion scaffold 下，只替换 sender/grid scheduler，证明 PAPG/PPS 的边际贡献。该表不得写成完整系统 baseline。

固定 scaffold：

- SGCP 分簇 / cluster heads。
- raw LiDAR grid upload 表示。
- 簇内 early fusion + 簇间 late fusion。
- 相同 20MHz/10ch 或主文指定带宽。

必须包含：

- [x] Random budgeted scheduler。
- [x] Density-greedy scheduler。
- [x] Link-aware density scheduler。
- [x] PACP-style LiDAR priority scheduler：明确为 proxy，PACP 原文是 RGB/BEV 方法。
- [x] EdgeCooper-inspired V2V complementarity scheduler：明确为 inspired/proxy，不写成原版 EdgeCooper。
- [x] SGCP-PAPG scheduler。

指标：

- [x] Network AP@0.3 / AP@0.5 / AP@0.7。
- [x] Mbps / payload bytes。
- [x] Avg selected CAVs、avg selected grids、avg fused CAVs。
- [x] Evaluated sample count、receiver policy、是否 inter-cluster late fusion。

验收标准：

- [x] 该表必须显示 PAPG 在同一 scaffold 中具有可解释的 AP-Mbps 优势，至少在 AP@0.3/AP@0.5 或 AP@0.7/Mbps 的某个叙事维度上形成优势：PAPG 与 EdgeCooper-HD 同 AP@0.3/0.5 但少 4.4% payload；相比 density/link-aware AP@0.3/0.5 更高且少 15.0% payload；相比 PACP-LiDAR AP@0.3 持平但少 27.8% payload。
- [x] 若 EdgeCooper-inspired 或 PACP-style 高预算超过 SGCP，必须通过 Pareto 曲线和信息条件说明边界，或继续改造 SGCP 调度：PACP-LiDAR 与 EdgeCooper-HD 在 AP@0.7 更高，已记录为高 IoU / stronger-prior boundary，需在 P4 Pareto 和正文边界中解释。

## P4：Figure 1 - AP-Mbps Pareto Curve

目的：避免单点预算争议，展示通信-精度前沿。

- [x] 已建立第一版 Pareto 源数据：`docs/doc_workspace/SGCP/artifacts/pareto_20260719/pareto_source.csv`，汇总已复现 41 帧 protocol-native、SGCP-compatible、edge-assisted reference、Pure late box-overhead 和 full-sharing upper reference 点。

必须绘制：

- [x] AP@0.3 vs Mbps：coverage / recall / large-scene utility。第一版草稿见 `docs/doc_workspace/SGCP/artifacts/pareto_20260719/figure1_pareto_ap03.png`。
- [x] AP@0.7 vs Mbps：localization / high-quality fusion。第一版草稿见 `docs/doc_workspace/SGCP/artifacts/pareto_20260719/figure1_pareto_ap07.png`。

必须扫描：

- [x] SGCP：`rho_th`、`B_h`、channel count、可选 point cap。当前 `pareto_source.csv` 已覆盖 5ch stress、10ch rho2/rho3、20ch rho2、cap=3000、B_h=2/3；early checkpoint 回收后需重跑同一组关键点。
- [x] Random / Density / Link-aware：member budget、grid budget。当前已补 `scheduler_budget_sweep_20260719`：random low/high、density low/high、communication-aware low/high first-pass 预算扫描；若进附录可继续补更密集预算曲线。
- [x] PACP-LiDAR：member/grid budget 或 priority threshold。当前已纳入 high-budget `86.56 Mbps` 和 low-budget `67.31 Mbps` 两个代表点，足够支撑 proxy baseline 边界；若进附录可继续补 priority threshold。
- [x] FullPerception-PCS：blind-spot granularity、candidate threshold 或 PCS 原生参数，不改主带宽。当前已补 `pcs_parameter_sweep_20260719`：11 帧 granularity/overlap 趋势 + 41 帧 tuned anchor `div12/ov0`。41 帧更激进 sweep 因候选规模超过 10--15 分钟未完成，作为 PCS 可扩展性边界记录；Pareto 图仍只使用可复现的 41 帧 anchor。
- [x] EdgeCooperV2V+ / EdgeCooper-inspired：sender cap、assignment budget、half-duplex constraint。当前已补 `edgecooper_budget_sweep_20260719`：`edgecooper_global_hd` 低预算 `1 member/head,58 grids/head` 与高预算 `3 members/head,117 grids/head`，并复现 EdgeCooper-HD 主点。
- [x] Pure late prediction-box reference：在 Pareto 图或附注中加入 broadcast/all-to-all detection-box overhead；若要声称通信受限，必须补 NS3 synthetic late-box deadline replay，而不是只按 raw-LiDAR payload 记为 0 Mbps。第一版已加入 80B/box broadcast 与 all-to-all 两个点。

验收标准：

- [x] SGCP 应在中低通信区间形成清晰 Pareto 优势，或至少在同等 Mbps 下获得更高 aggregate AP@0.3 / AP@0.5。已新增 `pareto_claim_audit.md`：在 raw-LiDAR V2V / SGCP-compatible 集合中，SGCP-PAPG 位于 AP@0.3 frontier，并在 AP@0.5 上达到同预算 frontier；Pure late 已作为 prediction-sharing reference 单独解释。
- [x] 若 SGCP 不在 Pareto frontier 上，优先分析瓶颈并修改算法；必要时选择更能体现大规模分簇优势的场景。审计结论：PAPG 主点不是 AP@0.7 frontier，AP@0.7 边界由 `B_h=2` SGCP sensitivity、高预算 proxy 或 full-sharing/edge reference 给出；论文只写 AP@0.3/AP@0.5 Pareto claim，AP@0.7 写成 attentive detector / localization headroom 边界。远端 fine-tune 已停止，不再作为当前论文 blocker。

## P5：Figure 2 - Aggregate AP Protocol/Fusion Breakdown

目的：用 aggregate AP 展示 protocol-native、two-layer fusion 和 scheduler contribution 的关系，避免额外引入 satisfaction 指标。

必须包含：

- [x] Aggregate AP@0.3 / AP@0.5 / AP@0.7 grouped bar 或折线图。第一版草稿见 `docs/doc_workspace/SGCP/artifacts/figures_20260719/figure2_protocol_breakdown.png`。
- [x] 至少比较 Head-only、Pure late、FullPerception-PCS、EdgeCooperV2V+、SGCP。
- [x] 明确每个方法的 evaluated samples 数、receiver policy 和 fusion scaffold。图内标注 raw Mbps 与 evaluated samples，完整字段见 `table1_protocol_20260719/protocol_native_manifest.csv`。

验收标准：

- [x] 图中必须能说明 SGCP 的 aggregate AP 优势不是 ego-only 偶然结果，而来自多 receiver-frame / 多 cluster 的 pooled evaluation。
- [x] 若该图无法区分方法，优先检查 protocol-native baseline 是否正确、late fusion 是否一致、场景是否足够大，而不是新增自定义指标。2026-07-19 已视觉检查并重生成 Figure 2：Pure late 标为 `box 0.7`，SGCP/EdgeCooper/Full20Early/FullPerception-PCS 差异清晰；图可按 protocol-native breakdown 使用。

## P6：Figure 3 - Fusion Contribution by IoU Threshold

目的：直观支撑论文叙事：late fusion 提升 AP@0.3/覆盖，early fusion 提升 AP@0.7/定位。

必须包含：

- [x] Head-only。
- [x] Pure late fusion。
- [x] Clustered early-only。
- [x] Full SGCP。
- [x] 可选 One-cluster early-only / one-cluster early+late。第一版草稿使用 Full 20-CAV early upper reference，见 `docs/doc_workspace/SGCP/artifacts/figures_20260719/figure3_fusion_contribution.png`。

图形：

- [x] Grouped bar：每个 variant 对应 AP@0.3、AP@0.5、AP@0.7。
- [x] 需要在 caption 中明确各 IoU 阈值对应 coverage / localization 的解释。caption 草稿见 `docs/doc_workspace/SGCP/artifacts/figures_20260719/figure_notes.md`。

验收标准：

- [x] AP@0.3 和 AP@0.7 的趋势必须可解释，能支撑 “early + late 的分工”：Full SGCP 相比 clustered early-only 的 AP@0.3/AP@0.5 大幅提升来自 inter-cluster late fusion；Full 20-CAV early 的 AP@0.7 上界说明 localization 仍受 raw point-cloud sharing/checkpoint 限制。
- [x] 若趋势不符合叙事，优先检查 late-fusion NMS、坐标变换、full reference 和 receiver 统计口径。2026-07-19 已视觉检查并重生成 Figure 3：clustered early-only 到 Full SGCP 的覆盖提升清楚，Full20Early AP@0.7 上界也清楚；趋势符合“coverage gain + localization headroom”的保守叙事。

## P7：Table 4 - Parameter Sensitivity

目的：回应参数选择和鲁棒性审稿意见。

必须包含主文最小集合：

- [x] `rho_th` sweep：体现点云划分阈值的 AP-Mbps tradeoff。attentive forward-writing 版本见 `docs/doc_workspace/SGCP/artifacts/parameter_sensitivity_attentive_20260719/table4_parameter_sensitivity_attentive.csv`；当前 41 帧中 `rho_th=1/2/3` 均为 `0.87/0.81/0.36`，约 `62.54-62.57 Mbps`，说明该参数不是脆弱调参点。
- [x] `N_max` sweep：体现分簇容量对 AP、cluster size、reconfiguration 的影响。当前建议放附录，因为 AP 非单调但容量约束确实生效。
- [x] `T_min^stab` sweep：体现稳定窗口，若当前场景不敏感，必须明确说明并补更动态场景或降为附录。当前 41 帧 dump 对 100--1000 ms 不敏感，建议附录或负面结果。
- [x] Channel count / bandwidth sweep：体现网络资源受限时算法行为。attentive 5/10/20 ch 为 `0.74/0.61/0.24`、`0.87/0.81/0.36`、`0.88/0.81/0.36`，对应 `31.12/62.54/67.33 Mbps`；5ch 明显受限，20ch 收益很小。

验收标准：

- [x] 每个参数必须有选择依据，而不是只报告最优点。第一版解释见 `table4_parameter_sensitivity.md`。
- [x] 若参数结论弱，正文写保守边界，更多细节放附录或 rebuttal。当前 `N_max` / `T_min^stab` 已明确为附录/弱结论。

## P8：可选图表与附录

- [x] Runtime/control overhead：可进入附录或主文短表；用于回应实时性，但不作为主贡献主表。已整理附录证据包 `artifacts/appendix_support_20260719/runtime_control_ns3_appendix.md`，明确 control-plane 平均 105.24 ms、控制 metadata 低于 1% payload，并标注 near-real-time 边界。
- [x] NS3 request-level reliability：不做主文 Table；只在正文一句话报告 SGCP scheduled replay sanity，完整数据放附录/rebuttal。已整理 PAPG 11 帧 110/110 application/RLC complete、0 PHY failures，以及 5-subchannel out-of-window reject stress。
- [x] Qualitative case study：选 2-3 帧展示 GT、selected CAV、selected grids、cluster heads、final detections；用于解释失败修复和两层融合。已整理文字/表格草稿 `artifacts/appendix_support_20260719/qualitative_case_study.md`，并生成 BEV overlay draft `qualitative_case_study_bev.png/.pdf`；若进入正式论文，可继续美化图例和补预测框 overlay。
- [x] Appendix raw results：完整列出所有已跑结果，避免主文表格过载。第一版通过 `results.md`、`reproducibility_manifest.md` 和 `artifacts/appendix_support_20260719/runtime_control_ns3_summary.csv` 索引；后续若补新场景需继续追加。

## P9：论文 LaTeX 修改

- [x] 阅读并参考 EdgeCooper 写作方式，特别是 network-level evaluation、V2V+ / edge scheduling 的叙事边界；不额外引入 satisfaction rate，统一使用 aggregate AP。已新增 `edgecooper_writing_reference.md`，将 EdgeCooper 的实验章节组织映射到 SGCP 的 protocol-native table、fusion ablation、Pareto、scheduler comparison 和 appendix support。
- [x] 修改 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的实验章节结构，使其按以下顺序组织：protocol-native comparison、fusion scaffold ablation、Pareto curve、scheduler-compatible comparison、parameter sensitivity。第一版已完成。
- [x] 将 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的 Table 1/3、Figure 1/2/3 和正文解释切换到 attentive candidate；旧 `pointpillar_early_fusion` 数值只保留为 checkpoint-reference artifacts。已新增 `artifacts/paper_number_audit_attentive_20260719/`。
- [x] 重写 baseline 说明，明确 FullPerception-PCS、EdgeCooperV2V+、PACP-style LiDAR proxy、SGCP-compatible scheduler comparison 的信息条件和公平性边界。当前已在主表、Pareto caption 和 scheduler comparison 中分层说明。
- [x] 收紧正文强 claim 并回应 Smartform 相似性质疑：intro/related work 不再写 “Nearly all SOTA rely on RSUs” 等过宽表述；`Reference.bib` 新增 SMARTFORM 引用；正文明确 SGCP novelty 不在 generic coalition formation，而在 perception-density utility、motion-stability hysteresis、capacity-constrained cluster maintenance、raw-LiDAR grid selection 和 explicit V2V subchannel scheduling 的组合。
- [x] 修改图表 caption：每个图表必须说明回答的问题、统一资源设置、是否 protocol-native、是否 SGCP-compatible。
- [x] 删除或降级此前容易误导的单一“主表”叙事；不要把 full 20-CAV upper reference、FullPerception baseline、scheduler proxy 混在同一语义层级。

## P10：2026-07-21 实验表可信度修复（最高优先级）

目的：修复当前实验数据中最容易被审稿人质疑的三个点：FullPerception-PCS 数值不稳定/通信量偏低、late fusion 行通信量漏算检测框共享、分簇算法 baseline 不足。该任务完成前，不再把新的 Table 1/Table 3/Table 6 数值写成最终论文结论。

### P10.1 FullPerception-PCS 合理性与可复现性

- [ ] 重新核查并修复 `pcs.py` 的 blind-spot 定义和分组参数，使 PCS 在 `singleton` receiver protocol 下得到稳定、可复现且通信量合理的调度结果。当前异常信号：PCS 平均每个 receiver 仅上传约 `2.66--22.79` 个 grids，远低于 EdgeCooper 的约 `103.33` grids；这很可能来自原生 blind-spot 面积过小或分割粒度过碎。
- [ ] 在不修改主实验带宽/子信道设置（20MHz/10ch）的前提下，扫描并记录 PCS blind-spot 相关参数：`blind_spot_min_division`、候选邻域半径、最小 overlap、spot 面积下限/上限、确定性 tie-break。目标不是人为抬高 PCS，而是让其 raw-LiDAR V2V adaptation 与论文机制一致、候选需求完整、每次重复完全相同。
- [ ] 对 PCS 输出做 trace-level 验收：同一命令重复运行的 scheduled links、selected grids、payload bytes、AP 必须一致；有/无 global late box aggregation 的 PCS 调度和 raw-LiDAR payload 必须一致。
- [ ] 重新生成 PCS no-late、PCS + global box aggregation、PCS 参数诊断结果，并更新 `C:\Workspace\2026-7-papers\infocom\SGCP\experiment` 中相关表格、manifest 和说明；旧的 under-scheduled PCS 结果只保留为诊断，不进入主文表。
- [x] 排查 2026-07-21 PCS sweep AP 全 0 的命令口径 bug：第一轮 sweep 误用 `intermediate_attentive`，已按正式 `early` fusion-method 口径重跑；default 11 帧为 `0.12/0.11/0.04`，div4/radius4/min128 为 `0.16/0.14/0.07`。
- [x] 新增 PCS object-grid diagnostics，确认 PCS 低 AP 的主要原因是 paper-style blind spot (`req_grids - high_density_grids`) 与 raw-LiDAR detector object utility 错位：前 3 帧 full-reference detected but PCS missed 的 30 个 GT 中，16 个被 nearest head 视为 high-density 而不请求，14 个属于 blind spot 但只有 5 个被 scheduled link 覆盖。
- [x] 基于上述结论修订 experiment 目录中的 PCS 表述：区分 paper-faithful PCS baseline、raw-LiDAR adaptation variant 和 SGCP-compatible scheduler comparison，避免把 object-aware 修补写成原版 FullPerception-PCS。

### P10.2 Late-fusion 检测框通信量计入总通信量

- [x] 对所有启用 late fusion / global box aggregation / prediction sharing 的数据行，统一在通信量统计中加入检测框共享 payload。表格至少同时保留 `raw_lidar_mbps`、`box_mbps`、`total_mbps` 三列；若正文只显示一个 Mbps，必须使用 `total_mbps`。2026-07-21 已在 `C:\Workspace\2026-7-papers\infocom\SGCP\experiment` 中完成 first pass，`mbps=total_mbps`。
- [x] Pure late、SGCP full、SGCP-compatible scheduler comparison、PCS + global box aggregation、EdgeCooper/FullPerception protocol adaptation 中凡是共享检测框的行，都必须标注 box-sharing mode（broadcast、scheduled all-to-all、或 global aggregation）和估算参数（box bytes、message overhead、deadline）。当前口径：`broadcast_boxes`，`80 bytes/box + 64 bytes/message`，`100 ms` cycle。
- [x] 更新实验目录中的 README/manifest，明确哪些行是 raw-LiDAR only，哪些行是 raw-LiDAR + late-box total，避免论文写作 agent 把 0 Mbps / raw-only Mbps 当成最终通信量。已更新 `README.md`、`table_guidance.md`、`experiment_update_summary.md` 和 `MANIFEST.csv`。

### P10.3 分簇算法 baseline 扩展

- [x] 在当前 `Fixed first-frame clusters` 之外，补充分簇算法 baseline：确定性随机分簇、距离/通信贪心分簇、密度/质量贪心分簇。所有 baseline 使用同一 attentive checkpoint、同一资源调度算法、同一 late-fusion 开关和同一 20MHz/10ch 设置，只替换 clustering 维度。2026-07-21 已完成 41 帧：random balanced `0.53/0.49/0.23, 31.79 Mbps`；distance-greedy `0.58/0.54/0.31, 31.83 Mbps`；density/quality-greedy `0.58/0.53/0.30, 31.98 Mbps`。
- [ ] 调研近年 V2X / cooperative perception / vehicular edge sensing 中可复现的分簇或 coalition baseline，优先选择 1--2 个能映射到当前 raw-LiDAR V2V 场景的方法，并记录论文来源、机制映射和不适配点。初版记录见 `clustering_baseline_literature_notes.md`。
- [ ] 新增分簇算法消融表：至少包含 Dynamic SGCP coalition、Fixed first-frame、Random balanced、Distance-greedy、Density/quality-greedy，以及 1--2 个 literature-inspired baseline。若某个 baseline 结果较差，必须解释是通信拓扑、感知覆盖还是稳定性导致，不能只给数字。当前 Table 5 已包含 Dynamic SGCP coalition、Fixed first-frame、Random balanced、Distance-greedy、Density/quality-greedy；剩余缺口是 1--2 个 literature-inspired baseline 或明确说明三类启发式已覆盖 random/proximity/sensing-aware 边界。
- [x] 论文修改完成后尝试编译 PDF；若本机缺少 LaTeX 工具，记录未编译原因和需要人工验证的图表编号。本机未检测到 `latexmk` / `pdflatex` / `bibtex`，已做轻量结构检查：table/figure/tabular begin-end 配对正常，新增 label/ref 无缺失。

## P11：自动任务执行规则

- [ ] 每轮自动任务先查看 `readme.md`、`target.md`、`status.md`、`results.md` 和 `../environment.md`。
- [ ] 每轮优先推进 P0-P4；只有主图表证据稳定后再扩展 P5-P8。
- [ ] 每次实验前写入 `log.md`，实验后更新 `results.md` / `status.md`。
- [ ] 代码或论文修改完成并验证后及时 git commit；不得提交无关脏文件。
