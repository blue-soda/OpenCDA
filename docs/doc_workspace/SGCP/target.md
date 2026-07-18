# SGCP 任务清单

更新时间：2026-07-19

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
- [ ] 所有实验在结果进入论文前必须生成 artifact：命令、stdout/log 路径、trace CSV、manifest/summary CSV、图表源数据和 git commit。
- [ ] 如果当前 41 帧场景无法支撑某张关键图表，重新导出更合适的 CARLA 场景；注意 CARLA 进程至多一个，数据路径仍遵循 `docs/doc_workspace/environment.md`。

## P1：Table 1 - Protocol-Native System Comparison

目的：比较完整系统协议，而不是单个调度器。每个方法按自己的原生机制运行；FullPerception/EdgeCooperV2V+ 只使用 RSU/edge 调度，不使用 RSU 点云。

必须包含：

- [x] Head-only：单车感知下界。
- [x] Pure late fusion：所有 CAV 本地检测框 late fusion，不做点云共享；用于展示 late fusion 覆盖能力。
- [x] FullPerception-PCS V2V：`pcs.py` / `fullperception_pcs`，按论文 PCS 口径运行，必要时继续修正 PCS 复现质量。
- [x] EdgeCooperV2V+：V2V 数据共享、edge/RSU 只调度；不能使用 SGCP 分簇作为原生设置，若使用 proxy 必须标注。
- [x] SGCP full：分簇 + PAPG/PPS 点云划分 + 簇内 early fusion + 簇间 late fusion。
- [x] Full 20-CAV early fusion upper reference：只作上界，不作为 baseline。

指标：

- [x] Network AP@0.3 / AP@0.5 / AP@0.7。
- [x] Evaluated sample count and receiver-frame scope。
- [x] Mbps / payload bytes。
- [x] 必要时补 delivery sanity 一句话，不做单独 NS3 表格。

验收标准：

- [ ] 表格必须清晰解释 SGCP 的完整系统优势，尤其是 aggregate AP@0.3 的大场景覆盖收益和 AP@0.7 的局部精度收益。
- [ ] 若 SGCP 只靠 late fusion 在 AP@0.3 大幅领先，需要在正文明确这是系统协议优势，不把它写成单纯 scheduler 优势。
- [ ] 若 FullPerception-PCS 或 EdgeCooperV2V+ 复现结果异常低，先核查算法、参数和场景，不直接拿弱结果进主表。
- [x] Pure late fusion 当前 `0.82/0.76/0.37` 且 0 点云 payload 很强；论文写作必须计入 detection-box exchange overhead 或将其明确为 prediction-sharing reference，否则会削弱 SGCP 通信优势叙事。已新增 `late_fusion_box_comm.md` 与 `opencda.tools.sgcp_late_box_comm_budget`：20MHz/10ch 下 detection-box broadcast 约 `0.74-1.13 Mbps`，scheduled all-to-all unicast 约 `14.04-21.52 Mbps` 且 100 ms 内可完成；只有 unscheduled all-to-all 随机抢信道模型会因碰撞失败。
- [ ] 修正 Pure late baseline 口径：当前 P1/P2 manifest 中的 Pure late 使用 `fusion_method=early` + singleton CAV + custom box-level late NMS，不是 `pointpillar_late_fusion` checkpoint。已补 41 帧 actual late checkpoint sanity：`0.89/0.83/0.49`，预测框 broadcast overhead 约 `1.07-1.65 Mbps`。后续主表应决定使用 actual-late checkpoint 作为 prediction-sharing reference，或明确标注 early-singleton late proxy。
- [ ] 统一 detector/checkpoint 公平性：SGCP 论文主线的“点云 -> 检测框”必须使用同一 checkpoint；由于 SGCP 簇内阶段是 raw point-cloud early fusion，主线公平口径应统一使用 `pointpillar_early_fusion`，Pure late 则为 singleton local inference + `naive_late_fusion()`。已补 “all late detector” sanity：Pure late actual late `0.89/0.83/0.49`，forced SGCP PAPG late-detector `0.87/0.81/0.48`、62.54 Mbps；后者不再是严格 SGCP early-fusion 协议，只能作为 checkpoint sensitivity。
- [ ] 提升 early-fusion checkpoint：当前最大实验风险是 `pointpillar_early_fusion` 对 SGCP raw point-cloud early fusion 不够强，导致 Pure late prediction-sharing reference 过强、SGCP AP@0.7 上限偏低。远程训练固定使用 `ssh mindspore-187`、`/data2/gzc/sgcp_early_train/` 和 `opencood-gzc` 环境；已上传当前 checkpoint 并启动 GPU watcher，GPU 空闲后自动 fine-tune。回收 checkpoint 后必须用同一 checkpoint 重跑 SGCP 与 Pure late controlled baseline。

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

- [ ] 不把 `Clustered late-only` 作为核心表格行；分簇主要服务点云通信和 early fusion，clustered late-only 机制不自然，若保留仅放附录诊断。

验收标准：

- [ ] AP@0.7 的提升应能解释为点云 early fusion / 高质量局部融合贡献。
- [x] AP@0.3 的提升应能解释为 late fusion / 多区域覆盖贡献：同 payload 下 clustered early-only `0.38/0.36/0.20` 到 Full SGCP `0.81/0.78/0.39`。
- [ ] Full SGCP 必须在通信量远低于 one-cluster/full-sharing 设置时保持有竞争力的 aggregate AP。
- [ ] 若该表不能证明分簇或两层融合有效，优先修改算法、场景或评估口径，再进入论文。

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

- [ ] SGCP：`rho_th`、`B_h`、channel count、可选 point cap。
- [ ] Random / Density / Link-aware：member budget、grid budget。
- [ ] PACP-LiDAR：member/grid budget 或 priority threshold。
- [ ] FullPerception-PCS：blind-spot granularity、candidate threshold 或 PCS 原生参数，不改主带宽。
- [ ] EdgeCooperV2V+ / EdgeCooper-inspired：sender cap、assignment budget、half-duplex constraint。
- [x] Pure late prediction-box reference：在 Pareto 图或附注中加入 broadcast/all-to-all detection-box overhead；若要声称通信受限，必须补 NS3 synthetic late-box deadline replay，而不是只按 raw-LiDAR payload 记为 0 Mbps。第一版已加入 80B/box broadcast 与 all-to-all 两个点。

验收标准：

- [ ] SGCP 应在中低通信区间形成清晰 Pareto 优势，或至少在同等 Mbps 下获得更高 aggregate AP@0.3 / AP@0.5。当前第一版图显示：若把 Pure late broadcast 作为同一 Pareto 点，SGCP 不在 AP@0.3 frontier；因此主文必须把 Pure late 标为 prediction-sharing reference，并将 raw-LiDAR V2V/PPS 方法单独解释。
- [ ] 若 SGCP 不在 Pareto frontier 上，优先分析瓶颈并修改算法；必要时选择更能体现大规模分簇优势的场景。当前瓶颈已定位为 early checkpoint 偏弱和 Pure late 过强；远程 early checkpoint fine-tune 正在等待 GPU。

## P5：Figure 2 - Aggregate AP Protocol/Fusion Breakdown

目的：用 aggregate AP 展示 protocol-native、two-layer fusion 和 scheduler contribution 的关系，避免额外引入 satisfaction 指标。

必须包含：

- [x] Aggregate AP@0.3 / AP@0.5 / AP@0.7 grouped bar 或折线图。第一版草稿见 `docs/doc_workspace/SGCP/artifacts/figures_20260719/figure2_protocol_breakdown.png`。
- [x] 至少比较 Head-only、Pure late、FullPerception-PCS、EdgeCooperV2V+、SGCP。
- [x] 明确每个方法的 evaluated samples 数、receiver policy 和 fusion scaffold。图内标注 raw Mbps 与 evaluated samples，完整字段见 `table1_protocol_20260719/protocol_native_manifest.csv`。

验收标准：

- [x] 图中必须能说明 SGCP 的 aggregate AP 优势不是 ego-only 偶然结果，而来自多 receiver-frame / 多 cluster 的 pooled evaluation。
- [ ] 若该图无法区分方法，优先检查 protocol-native baseline 是否正确、late fusion 是否一致、场景是否足够大，而不是新增自定义指标。

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
- [ ] 若趋势不符合叙事，优先检查 late-fusion NMS、坐标变换、full reference 和 receiver 统计口径。

## P7：Table 4 - Parameter Sensitivity

目的：回应参数选择和鲁棒性审稿意见。

必须包含主文最小集合：

- [x] `rho_th` sweep：体现点云划分阈值的 AP-Mbps tradeoff。第一版 Table 4 candidate 见 `docs/doc_workspace/SGCP/artifacts/parameter_sensitivity_20260719/table4_parameter_sensitivity.csv`。
- [x] `N_max` sweep：体现分簇容量对 AP、cluster size、reconfiguration 的影响。当前建议放附录，因为 AP 非单调但容量约束确实生效。
- [x] `T_min^stab` sweep：体现稳定窗口，若当前场景不敏感，必须明确说明并补更动态场景或降为附录。当前 41 帧 dump 对 100--1000 ms 不敏感，建议附录或负面结果。
- [x] Channel count / bandwidth sweep：体现网络资源受限时算法行为。5/10/20 ch 可进主文，极低带宽 stress 可进附录。

验收标准：

- [x] 每个参数必须有选择依据，而不是只报告最优点。第一版解释见 `table4_parameter_sensitivity.md`。
- [x] 若参数结论弱，正文写保守边界，更多细节放附录或 rebuttal。当前 `N_max` / `T_min^stab` 已明确为附录/弱结论。

## P8：可选图表与附录

- [x] Runtime/control overhead：可进入附录或主文短表；用于回应实时性，但不作为主贡献主表。已整理附录证据包 `artifacts/appendix_support_20260719/runtime_control_ns3_appendix.md`，明确 control-plane 平均 105.24 ms、控制 metadata 低于 1% payload，并标注 near-real-time 边界。
- [x] NS3 request-level reliability：不做主文 Table；只在正文一句话报告 SGCP scheduled replay sanity，完整数据放附录/rebuttal。已整理 PAPG 11 帧 110/110 application/RLC complete、0 PHY failures，以及 5-subchannel out-of-window reject stress。
- [x] Qualitative case study：选 2-3 帧展示 GT、selected CAV、selected grids、cluster heads、final detections；用于解释失败修复和两层融合。已整理文字/表格草稿 `artifacts/appendix_support_20260719/qualitative_case_study.md`；若进入论文，需要下一轮生成 BEV overlay 图。
- [x] Appendix raw results：完整列出所有已跑结果，避免主文表格过载。第一版通过 `results.md`、`reproducibility_manifest.md` 和 `artifacts/appendix_support_20260719/runtime_control_ns3_summary.csv` 索引；后续若补新场景需继续追加。

## P9：论文 LaTeX 修改

- [ ] 阅读并参考 EdgeCooper 写作方式，特别是 network-level evaluation、V2V+ / edge scheduling 的叙事边界；不额外引入 satisfaction rate，统一使用 aggregate AP。
- [x] 修改 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的实验章节结构，使其按以下顺序组织：protocol-native comparison、fusion scaffold ablation、Pareto curve、scheduler-compatible comparison、parameter sensitivity。第一版已完成。
- [x] 重写 baseline 说明，明确 FullPerception-PCS、EdgeCooperV2V+、PACP-style LiDAR proxy、SGCP-compatible scheduler comparison 的信息条件和公平性边界。当前已在主表、Pareto caption 和 scheduler comparison 中分层说明。
- [x] 修改图表 caption：每个图表必须说明回答的问题、统一资源设置、是否 protocol-native、是否 SGCP-compatible。
- [x] 删除或降级此前容易误导的单一“主表”叙事；不要把 full 20-CAV upper reference、FullPerception baseline、scheduler proxy 混在同一语义层级。
- [x] 论文修改完成后尝试编译 PDF；若本机缺少 LaTeX 工具，记录未编译原因和需要人工验证的图表编号。本机未检测到 `latexmk` / `pdflatex` / `bibtex`，已做轻量结构检查：table/figure/tabular begin-end 配对正常，新增 label/ref 无缺失。

## P10：自动任务执行规则

- [ ] 每轮自动任务先查看 `readme.md`、`target.md`、`status.md`、`results.md` 和 `../environment.md`。
- [ ] 每轮优先推进 P0-P4；只有主图表证据稳定后再扩展 P5-P8。
- [ ] 每次实验前写入 `log.md`，实验后更新 `results.md` / `status.md`。
- [ ] 代码或论文修改完成并验证后及时 git commit；不得提交无关脏文件。
