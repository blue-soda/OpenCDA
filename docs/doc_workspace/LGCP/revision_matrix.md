# LGCP Revision Matrix

本文档将 `C:\Workspace\icdcs-paper\LGCP\LGCP-review.txt` 中的审稿意见转化为可执行的 revision / rebuttal 任务。优先级含义：

- P0：直接影响论文是否可信，必须优先补强。
- P1：重要支撑项，建议主文或附录补充。
- P2：表达、定位、限制讨论和工程细节完善。

## 总体策略

当前审稿意见集中在四个高风险点：

1. **核心变量未验证**：area confidence 被用于优化目标和 grouping，但缺少与真实 area-level AP / recall 的相关性证明。
2. **算法贡献偏启发式**：group selection、leader assignment、transmission scheduling 都被认为缺少理论保证或 optimality gap 证据。
3. **baseline 不够强 / 不够公平**：LGCP 做 selective sharing，却主要对比 full sharing 的 vehicle-based / edge-assisted 方案。
4. **大规模质量证据不足**：30 CAV co-simulation 只报告 latency，不能支撑 perception quality scalability。

因此 revision 应按以下顺序推进：

1. 先补 area confidence validation 和 Eq. (2) composition validation。
2. 再补 adaptive sharing baseline 与 local-to-global ablation。
3. 然后补 small-scale optimality gap 和 sensitivity analyses。
4. 最后修正文稿定位、workflow 图、部署假设、局限性和 stage 编号。

## 审稿意见对照表

| ID | Reviewer | 问题类型 | Priority | 审稿人关注点 | 修改动作 | 产物 / 位置 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1-1 | R1, R4 | 机制 / 理论 | P0 | 算法只是 heuristic，没有 approximation guarantee 或理论分析。 | 增加小规模 optimality gap：对 group selection 用 exhaustive search / ILP 做最优参考；正文避免声称 approximate guarantee，改为 heuristic with empirical gap。 | `results.md` R2；`manuscript_language_audit.md`；论文 Solution / Experiment | 已有 smoke gap 与措辞草稿，仍需多 seed |
| R1-2 | R1 | 机制 / 写作 | P1 | Grouping 不是传统 clustering，一个 CAV 可参与多个 group，示例中 CAV2 重复参与且 packet 含义不清。 | 将 group 命名为 area-task group / hypergraph assignment；说明 packet granularity 是 area-specific feature slice；补充去重/复用策略。 | `workflow_and_group_semantics.md`；论文 Framework / Problem Formulation | 已形成草稿 |
| R1-3 | R1 | 机制 / 实验 | P1 | leader 到 RSU 的 aggregated result 更重要，为什么不用第一阶段方法保障上传？ | 明确 leader-to-RSU upload 的优先级和可靠性；补充 leader upload scheduling 或失败影响实验。 | `workflow_and_group_semantics.md`；论文 latency / scheduling 小节 | 已形成机制草稿，仍需实验 |
| R2-1 | R2, R3 | 写作 | P2 | 缺 workflow figure，Fig. 2 不能清楚展示 partition、grouping 等流程。 | 增加 LGCP workflow 图：RSU partition、CAV report、area group assignment、local fusion、leader upload、RSU global aggregation、broadcast。 | `workflow_and_group_semantics.md`；论文 Framework figure artifact | 已形成草稿 |
| R2-2 | R2 | 写作 / 实验解释 | P2 | Fig. 7 y-axis label 错误；低车数 latency 与 baseline 接近原因不清。 | 修正图轴；解释低密度下冲突少、固定控制开销占比高、调度优势尚未显现。 | `latency_figure_audit.md`；论文 Evaluation Fig. 7 / discussion | 已核查：y-axis 正确，x-axis 应改为 `Number of CAVs`；解释草稿已完成 |
| R2-3 | R2, R3 | 实验 / 机制 | P1 | >20 CAV latency 改善有限，需要瓶颈分析和 scalability 解释。 | 增加瓶颈分解：V2V scheduling、leader fusion、leader-to-RSU、control-plane overhead；若质量未测，收窄大规模 claim。 | `results.md` R5；论文 Evaluation / Limitation | 未开始 |
| R2-4 | R2, R3 | 实验 | P0 | 位置误差、车辆移动、update frequency、车速会影响 partition 和 redundancy removal。 | 增加 localization error、vehicle speed、update frequency / stale assignment 敏感性实验。 | `localization_error_sensitivity.md`；`stale_assignment_sensitivity.md`；`results.md` R6；论文 Robustness | localization 和 stale assignment smoke 已完成，显式车速仍需多场景 |
| R3-1 | R3 | 实验 / 机制 | P0 | area confidence 未验证；需要 correlation with area-level AP / recall。 | 导出 area id、CAV id、confidence、fusion output、GT；计算 Pearson / Spearman 与 area-level AP / recall。 | `results.md` R1；新增实验脚本 | 未开始 |
| R3-2 | R3 | 实验 / 机制 | P0 | Eq. (2) composition rule 是否合理未知。 | 比较 product rule、max、mean、sum、top-k、learned/calibrated composition。 | `results.md` R1；论文 ablation | 未开始 |
| R3-3 | R3, R4 | 实验公平性 | P0 | 与 full-sharing baseline 比较不公平；缺强 communication-aware baseline。 | 实现 native sparse sharing / selective sharing without LGCP hierarchy / quality-aware sharing baseline。 | `results.md` R4；论文 baseline 小节 | 未开始 |
| R3-4 | R3, R4 | 实验 | P0 | 需要区分 local-to-global structure 与 merely not sending everything。 | 增加 ablation：full sharing、selective sharing only、LGCP without scheduling、full LGCP。 | `results.md` R3 | 未开始 |
| R3-5 | R3 | 实验 | P1 | 大规模 30 CAV 只报告 latency，不是 end-to-end perception evaluation。 | 增加 scalable quality proxy 或明确大规模只验证 communication/computation latency。 | `results.md` R5；论文 claim / limitation | 未开始 |
| R3-6 | R3 | 实验 | P1 | key gains 可能依赖单一设置；缺 sensitivity analyses。 | 变化 area size、`Delta_g`、subchannel count `Z`、CAV/edge compute capacity、transmission threshold/rate。 | `results.md` R6 | 未开始 |
| R3-7 | R3 | 实验 / 系统 | P1 | control-plane overhead 未量化。 | 显式统计每 cycle 的 location、direction、confidence upload、assignment broadcast、global view broadcast 字节数和时延占比。 | `results.md` R5/R6；日志字段 | 未开始 |
| R3-8 | R3 | 写作 | P2 | 算法解释过度依赖伪代码，缺直觉；stage-number inconsistency。 | 增加算法直觉段；修正 “fifth stage” 和总 latency stage 编号。 | `manuscript_language_audit.md`；论文 Solution / Latency | 已形成修改清单 |
| R3-9 | R3 | 写作 / 限制 | P2 | 需要部署假设和 failure modes：RSU centralization、mobility、stale information、multi-RSU scaling。 | 新增 Deployment Assumptions and Limitations 小节。 | `deployment_assumptions.md`；论文 Discussion | 已形成草稿 |
| R4-1 | R4 | 写作 / 定位 | P0 | 贡献被认为只是经典 hierarchical task partitioning / aggregation 的应用。 | 重写 contribution：强调 area-confidence-driven perception-quality/resource co-design、可重叠 area-task groups、local-to-global global-awareness pipeline；明确与 clustering/task assignment 的区别。 | Introduction / Related Work / Discussion | 未开始 |
| R4-2 | R4 | 实验 / 论证 | P0 | 性能提升被认为是 expected：减少共享和并行化导致。 | 用 ablation 和 stronger baseline 证明收益不是简单减少传输；报告 quality preservation under comparable budget。 | `results.md` R3/R4 | 未开始 |

## Rebuttal 要点草稿

### 对 “heuristic only” 的回应

承认当前算法是 practical heuristic，不声称 approximation guarantee。补充小规模 optimal solver 对照，展示 greedy threshold 在目标值和 latency constraint 上的 empirical gap，并解释完整问题包含 area assignment、leader load balancing 和 interference-aware scheduling，难以直接给出简单闭式 guarantee。

### 对 “not clustering” 的回应

接受该表述问题。将 group 改称 area-task group，强调它不是 disjoint clustering，而是针对不同 area 的可重叠任务分配。一个 CAV 可服务多个 area，但上传的是 area-specific feature slice；若多个 area 可共享同一底层 feature map，则机制上应支持 packet reuse / batching，并在论文中说明。

### 对 “confidence proxy not validated” 的回应

这是最关键补强点。应新增跨数据集、跨模型的 correlation study，证明 confidence 与真实 area-level AP / recall 具有统计相关性；同时验证 Eq. (2) 的 product composition 是否合理，或替换为更稳健的 calibrated rule。

### 对 “baseline unfair” 的回应

不能只说 full-sharing 是传统范式。需要补充 adaptive/selective baseline，或者至少用 ablation 分离 selective sharing 与 local-to-global hierarchy 的贡献。

### 对 “large-scale only latency” 的回应

若短期无法跑 30 CAV 端到端 perception AP，应明确收窄 claim：large-scale co-simulation only validates communication/computation latency scalability。若要继续声称 quality scalability，应提供 quality proxy，并用小规模真实 AP 验证 proxy 的相关性。

## 当前仓库落地映射

| Revision 需求 | 当前仓库基础 | 下一步 |
| --- | --- | --- |
| RSU 场景 | `lgcp_carla` 已有 RSU、20 CAV、Town03、数据导出 | 实现 RSU 控制面与 LGCP assignment |
| 离线评估 | `offline_inference.py` 可读取导出数据并推理 | 扩展为 area-level AP / recall 导出 |
| RSU agent 数据 | data dump 已包含 `-1` 目录 | 决定 offline loader 是否把 RSU 作为 sensor provider 或 controller |
| area/grid 信息 | LiDAR grid 已开启，导出仍以完整 PCD 为主 | 导出 grid id、area id、grid density、confidence |
| baseline / ablation | 现有 cluster 管线可作为参考 | 新增 selective sharing only / LGCP variants |
