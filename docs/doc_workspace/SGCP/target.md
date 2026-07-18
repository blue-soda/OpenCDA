# SGCP 任务清单

更新时间：2026-07-18

最终目标：完成 SGCP 论文审稿意见响应与实验重构，使所有图表有效、可解释、保护论文叙事，并能证明 SGCP 在大规模 V2V 协同感知中以合理通信量获得更好的 network-level 感知效果。

## 当前原则

- 主指标不再只依赖 ego AP；主文优先使用 network-level AP、mean satisfaction rate / coverage satisfaction、Mbps、必要的 delivery sanity。
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

- [ ] 建立统一评估脚本/清单，保证所有图表使用同一数据集、CAV 数、OpenCOOD checkpoint、IoU 阈值、Mbps 换算、帧范围和随机种子记录。
- [ ] 明确 network-level AP 的统计口径：优先按所有 receiver / cluster head / active CAV-frame 汇总，而不是只报告 ego AP；若保留 ego AP，只作为附录 sanity。
- [ ] 定义并实现/确认 satisfaction rate：参考 EdgeCooper 的平均 satisfaction 思路，给出每个 CAV-frame 或区域是否满足 coverage / recall / AP 阈值的可复现定义。
- [ ] 所有实验在结果进入论文前必须生成 artifact：命令、stdout/log 路径、trace CSV、summary CSV、图表源数据和 git commit。
- [ ] 如果当前 41 帧场景无法支撑某张关键图表，重新导出更合适的 CARLA 场景；注意 CARLA 进程至多一个，数据路径仍遵循 `docs/doc_workspace/environment.md`。

## P1：Table 1 - Protocol-Native System Comparison

目的：比较完整系统协议，而不是单个调度器。每个方法按自己的原生机制运行；FullPerception/EdgeCooperV2V+ 只使用 RSU/edge 调度，不使用 RSU 点云。

必须包含：

- [ ] Head-only：单车感知下界。
- [ ] Pure late fusion：所有 CAV 本地检测框 late fusion，不做点云共享；用于展示 late fusion 覆盖能力。
- [ ] FullPerception-PCS V2V：`pcs.py` / `fullperception_pcs`，按论文 PCS 口径运行，必要时继续修正 PCS 复现质量。
- [ ] EdgeCooperV2V+：V2V 数据共享、edge/RSU 只调度；不能使用 SGCP 分簇作为原生设置，若使用 proxy 必须标注。
- [ ] SGCP full：分簇 + PAPG/PPS 点云划分 + 簇内 early fusion + 簇间 late fusion。
- [ ] Full 20-CAV early fusion upper reference：只作上界，不作为 baseline。

指标：

- [ ] Network AP@0.3 / AP@0.5 / AP@0.7。
- [ ] Mean satisfaction rate / coverage satisfaction。
- [ ] Mbps / payload bytes。
- [ ] 必要时补 delivery sanity 一句话，不做单独 NS3 表格。

验收标准：

- [ ] 表格必须清晰解释 SGCP 的完整系统优势，尤其是大场景 coverage / satisfaction。
- [ ] 若 SGCP 只靠 late fusion 在 AP@0.3 大幅领先，需要在正文明确这是系统协议优势，不把它写成单纯 scheduler 优势。
- [ ] 若 FullPerception-PCS 或 EdgeCooperV2V+ 复现结果异常低，先核查算法、参数和场景，不直接拿弱结果进主表。

## P2：Table 2 - Fusion Scaffold Ablation

目的：证明 SGCP 复杂机制必要性：early fusion 支撑精定位，late fusion 支撑覆盖，分簇和点云划分共同降低通信量。

必须包含：

- [ ] Head-only：无共享、无 late fusion。
- [ ] Pure late fusion：全局本地检测框 late fusion，无点云共享。
- [ ] One-cluster early-only：1 个簇头 + 19 个簇成员，全局点云通信后 early fusion，无 late fusion；用于观察无分簇的 early fusion 上限/通信代价。
- [ ] Clustered early-only：SGCP 分簇 + 簇内 early fusion，无簇间 late fusion；用于证明只做簇内 early fusion 覆盖不足。
- [ ] One-cluster early + late：无分簇/全局通信设置下的 early + late 对照；用于区分两层融合和分簇本身的贡献。
- [ ] Full SGCP：分簇 + 点云划分 + early + late。

不做核心消融：

- [ ] 不把 `Clustered late-only` 作为核心表格行；分簇主要服务点云通信和 early fusion，clustered late-only 机制不自然，若保留仅放附录诊断。

验收标准：

- [ ] AP@0.7 的提升应能解释为点云 early fusion / 高质量局部融合贡献。
- [ ] AP@0.3 / satisfaction 的提升应能解释为 late fusion / 多区域覆盖贡献。
- [ ] Full SGCP 必须在通信量远低于 one-cluster/full-sharing 设置时保持有竞争力的 AP 和 satisfaction。
- [ ] 若该表不能证明分簇或两层融合有效，优先修改算法、场景或评估口径，再进入论文。

## P3：Table 3 - SGCP-Compatible Scheduler Comparison

目的：在相同 clustered two-layer fusion scaffold 下，只替换 sender/grid scheduler，证明 PAPG/PPS 的边际贡献。该表不得写成完整系统 baseline。

固定 scaffold：

- SGCP 分簇 / cluster heads。
- raw LiDAR grid upload 表示。
- 簇内 early fusion + 簇间 late fusion。
- 相同 20MHz/10ch 或主文指定带宽。

必须包含：

- [ ] Random budgeted scheduler。
- [ ] Density-greedy scheduler。
- [ ] Link-aware density scheduler。
- [ ] PACP-style LiDAR priority scheduler：明确为 proxy，PACP 原文是 RGB/BEV 方法。
- [ ] EdgeCooper-inspired V2V complementarity scheduler：明确为 inspired/proxy，不写成原版 EdgeCooper。
- [ ] SGCP-PAPG scheduler。

指标：

- [ ] Network AP@0.3 / AP@0.5 / AP@0.7。
- [ ] Mbps / payload bytes。
- [ ] Avg selected CAVs、avg selected grids、avg fused CAVs。
- [ ] Satisfaction rate。

验收标准：

- [ ] 该表必须显示 PAPG 在同一 scaffold 中具有可解释的 AP-Mbps 优势，至少在 AP@0.3/AP@0.5 或 satisfaction/Mbps 上形成优势。
- [ ] 若 EdgeCooper-inspired 或 PACP-style 高预算超过 SGCP，必须通过 Pareto 曲线和信息条件说明边界，或继续改造 SGCP 调度。

## P4：Figure 1 - AP-Mbps Pareto Curve

目的：避免单点预算争议，展示通信-精度前沿。

必须绘制：

- [ ] AP@0.3 vs Mbps：coverage / recall / large-scene utility。
- [ ] AP@0.7 vs Mbps：localization / high-quality fusion。

必须扫描：

- [ ] SGCP：`rho_th`、`B_h`、channel count、可选 point cap。
- [ ] Random / Density / Link-aware：member budget、grid budget。
- [ ] PACP-LiDAR：member/grid budget 或 priority threshold。
- [ ] FullPerception-PCS：blind-spot granularity、candidate threshold 或 PCS 原生参数，不改主带宽。
- [ ] EdgeCooperV2V+ / EdgeCooper-inspired：sender cap、assignment budget、half-duplex constraint。

验收标准：

- [ ] SGCP 应在中低通信区间形成清晰 Pareto 优势，或至少在同等 Mbps 下获得更高 satisfaction / AP@0.3。
- [ ] 若 SGCP 不在 Pareto frontier 上，优先分析瓶颈并修改算法；必要时选择更能体现大规模分簇优势的场景。

## P5：Figure 2 - Satisfaction / Coverage Distribution

目的：对齐 EdgeCooper/FullPerception 等 network-level 论文指标，避免只看 ego AP。

必须包含：

- [ ] Per-CAV-frame satisfaction CDF 或分布图。
- [ ] Mean satisfaction vs Mbps 或方法柱状图。
- [ ] 至少比较 Head-only、Pure late、FullPerception-PCS、EdgeCooperV2V+、SGCP。

验收标准：

- [ ] 图中必须能说明 SGCP 不只是少数 ego 帧 AP 高，而是 road-level / CAV-level coverage 更稳定。
- [ ] 若 satisfaction 定义不能区分方法，重新定义 coverage/recovery 阈值或使用 per-region satisfaction。

## P6：Figure 3 - Fusion Contribution by IoU Threshold

目的：直观支撑论文叙事：late fusion 提升 AP@0.3/覆盖，early fusion 提升 AP@0.7/定位。

必须包含：

- [ ] Head-only。
- [ ] Pure late fusion。
- [ ] Clustered early-only。
- [ ] Full SGCP。
- [ ] 可选 One-cluster early-only / one-cluster early+late。

图形：

- [ ] Grouped bar：每个 variant 对应 AP@0.3、AP@0.5、AP@0.7。
- [ ] 需要在 caption 中明确各 IoU 阈值对应 coverage / localization 的解释。

验收标准：

- [ ] AP@0.3 和 AP@0.7 的趋势必须可解释，能支撑 “early + late 的分工”。
- [ ] 若趋势不符合叙事，优先检查 late-fusion NMS、坐标变换、full reference 和 receiver 统计口径。

## P7：Table 4 - Parameter Sensitivity

目的：回应参数选择和鲁棒性审稿意见。

必须包含主文最小集合：

- [ ] `rho_th` sweep：体现点云划分阈值的 AP-Mbps tradeoff。
- [ ] `N_max` sweep：体现分簇容量对 AP、cluster size、reconfiguration 的影响。
- [ ] `T_min^stab` sweep：体现稳定窗口，若当前场景不敏感，必须明确说明并补更动态场景或降为附录。
- [ ] Channel count / bandwidth sweep：体现网络资源受限时算法行为。

验收标准：

- [ ] 每个参数必须有选择依据，而不是只报告最优点。
- [ ] 若参数结论弱，正文写保守边界，更多细节放附录或 rebuttal。

## P8：可选图表与附录

- [ ] Runtime/control overhead：可进入附录或主文短表；用于回应实时性，但不作为主贡献主表。
- [ ] NS3 request-level reliability：不做主文 Table；只在正文一句话报告 SGCP scheduled replay sanity，完整数据放附录/rebuttal。
- [ ] Qualitative case study：选 2-3 帧展示 GT、selected CAV、selected grids、cluster heads、final detections；用于解释失败修复和两层融合。
- [ ] Appendix raw results：完整列出所有已跑结果，避免主文表格过载。

## P9：论文 LaTeX 修改

- [ ] 阅读并参考 EdgeCooper 写作方式，特别是 satisfaction rate、network-level evaluation、V2V+ / edge scheduling 的叙事边界。
- [ ] 修改 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的实验章节结构，使其按以下顺序组织：protocol-native comparison、fusion scaffold ablation、Pareto curve、scheduler-compatible comparison、parameter sensitivity。
- [ ] 重写 baseline 说明，明确 FullPerception-PCS、EdgeCooperV2V+、PACP-style LiDAR proxy、SGCP-compatible scheduler comparison 的信息条件和公平性边界。
- [ ] 修改图表 caption：每个图表必须说明回答的问题、统一资源设置、是否 protocol-native、是否 SGCP-compatible。
- [ ] 删除或降级此前容易误导的单一“主表”叙事；不要把 full 20-CAV upper reference、FullPerception baseline、scheduler proxy 混在同一语义层级。
- [ ] 论文修改完成后尝试编译 PDF；若本机缺少 LaTeX 工具，记录未编译原因和需要人工验证的图表编号。

## P10：自动任务执行规则

- [ ] 每轮自动任务先查看 `readme.md`、`target.md`、`status.md`、`results.md` 和 `../environment.md`。
- [ ] 每轮优先推进 P0-P4；只有主图表证据稳定后再扩展 P5-P8。
- [ ] 每次实验前写入 `log.md`，实验后更新 `results.md` / `status.md`。
- [ ] 代码或论文修改完成并验证后及时 git commit；不得提交无关脏文件。
