# SGCP 当前状态

更新时间：2026-07-19

## 运行环境

- 项目可用环境：`conda activate opencda`
- 脚本化执行建议：`conda run -n opencda python ...`

## 仓库理解

OpenCDA 是基于 CARLA/SUMO 的协同驾驶仿真框架，提供感知、定位、规划、控制、V2X 通信等完整模块。当前仓库还包含 OpenCOOD，用于多智能体协同感知模型训练和推理。

SGCP 工作与仓库中以下部分关系最紧密：

- `opencda/core/clustering/`：车辆聚类、资源分配、协同结构形成。
- `opencda/core/networking/`：V2X 网络模拟和 NS3 联动。
- `opencda/application/`：协同感知应用逻辑。
- `opencda/scenario_testing/config_yaml/`：场景、CAV 数量、通信和感知参数配置。
- `opencood/`：PointPillars、early/late/intermediate fusion 等感知后端与评估工具。

## 当前主线快照

- 已按用户最新要求移除 satisfaction rate 作为论文主指标，固定使用 aggregate AP + Mbps。Aggregate AP 口径直接写入核心状态/任务/结果文档，不再保留独立 metric 文档；`opencda.tools.sgcp_aggregate_ap_manifest` 把 OpenCOOD pooled evaluator AP 与 evaluated sample count、receiver policy、inter-cluster late fusion、scheduler、payload/Mbps 和 artifact 路径合并成 manifest CSV。已用 PAPG / EdgeCooper-HD 41 帧 repeat 日志完成 smoke：PAPG `0.81/0.78/0.39`、62.54 Mbps；EdgeCooper-HD `0.81/0.78/0.42`、65.40 Mbps。
- 已生成 P1 protocol-native manifest：`docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\protocol_native_manifest.csv`。六行核心结果为 Head-only `0.26/0.22/0.09`、Pure late singleton 20-CAV `0.82/0.76/0.37`、FullPerception-PCS tuned `0.59/0.53/0.22`、EdgeCooper-HD `0.81/0.78/0.42`、SGCP-PAPG `0.81/0.78/0.39`、Full 20-CAV early upper `0.85/0.83/0.48`。Pure late 当前只统计 0 point-cloud payload，进入论文前必须补 detection-box exchange overhead 或明确为 prediction-sharing reference。
- 已生成 P2 fusion scaffold manifest：`docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\fusion_scaffold_manifest.csv`。关键结论是同 payload 62.54 Mbps 下，clustered early-only 为 `0.38/0.36/0.20`，Full SGCP 加入簇间 late fusion 后为 `0.81/0.78/0.39`；one-cluster/full-sharing upper reference 为 `0.85/0.83/0.48`、118.71 Mbps。
- 已生成 P3 SGCP-compatible scheduler manifest：`docs\doc_workspace\SGCP\artifacts\scheduler_comparison_20260719\scheduler_comparison_manifest.csv`。PAPG `0.81/0.78/0.39`、62.54 Mbps；Random `0.77/0.73/0.38`、61.68 Mbps；Density/Link-aware `0.80/0.76/0.40`、73.58 Mbps；PACP-LiDAR `0.81/0.79/0.42`、86.56 Mbps；EdgeCooper-HD `0.81/0.78/0.42`、65.40 Mbps。结论是 PAPG 具备 AP@0.3/AP@0.5 与 payload tradeoff 优势，但 AP@0.7 不是最优，需由 P4 Pareto 和正文信息条件边界解释。
- 已生成 Figure 1 Pareto 第一版源数据：`docs\doc_workspace\SGCP\artifacts\pareto_20260719\pareto_source.csv`。该 CSV 汇总 Head-only、Pure late broadcast/all-to-all detection-box overhead、FullPerception-PCS、SGCP coverage/target-aware/PAPG 参数点、forced random、Density/Link-aware、EdgeCooper-HD、PACP-LiDAR、cluster-local/global selective proxy 与 Full20Early upper reference。当前结论仍是 PAPG 处于中等 payload 区间，具备 AP@0.3/AP@0.5 与 raw-LiDAR Mbps tradeoff 优势；Pure late 必须按 prediction-sharing reference 单独解释。
- 已生成 Figure 1 Pareto 第一版图表草稿：`docs\doc_workspace\SGCP\artifacts\pareto_20260719\figure1_pareto_ap03.png` 和 `figure1_pareto_ap07.png`，并保留 PDF 版本与绘图脚本 `plot_pareto.py`。当前图表进一步确认：Pure late broadcast 在 AP@0.3 上形成强 prediction-sharing reference，不能和 raw-LiDAR PPS 方法混成单一公平 Pareto；AP@0.7 维度上 EdgeCooper-HD / PACP-LiDAR 仍强于 PAPG，应解释为 edge/global 或 stronger-prior boundary。
- 已生成 Figure 2/3 第一版图表草稿：`docs\doc_workspace\SGCP\artifacts\figures_20260719\figure2_protocol_breakdown.png` 与 `figure3_fusion_contribution.png`，并保留 PDF、绘图脚本和 caption notes。Figure 2 支撑 protocol-native aggregate AP 比较；Figure 3 支撑 two-layer fusion 分工：late fusion 改善 coverage/低 IoU AP，full raw-LiDAR early sharing 仍是高 IoU localization 上界。
- 已生成 Table 4 参数敏感性第一版：`docs\doc_workspace\SGCP\artifacts\parameter_sensitivity_20260719\table4_parameter_sensitivity.csv` 和 `table4_parameter_sensitivity.md`。当前建议主文只放 `rho_th` 与子信道数两个证据最清晰的参数；`N_max` / `T_min^stab` 在 41 帧短序列上结论弱，放附录或 rebuttal 边界说明更稳。
- 已整理 P8 附录证据包：`docs\doc_workspace\SGCP\artifacts\appendix_support_20260719\runtime_control_ns3_appendix.md` 和 `runtime_control_ns3_summary.csv`。该包收束 runtime、control metadata、PPS convergence 和 NS3 request-level reliability：SGCP control-plane prototype 平均 105.24 ms，控制 metadata 为 187,112 bytes / 4,563.71 bytes/frame，约为 PAPG raw payload 的 0.58%；PAPG 11 帧 NS3 replay 为 110/110 application callback 与 RLC complete、0 PHY failures。论文中应作为附录/短句支撑，不改主指标。
- 已补 P8 qualitative case study：`docs\doc_workspace\SGCP\artifacts\appendix_support_20260719\qualitative_case_study.md`，并生成 `qualitative_case_study_bev.png/.pdf` 和 `qualitative_case_study_summary.csv`。当前选取 frame/object `000068/438`、`000066/401`、`000062/337`，分别说明 best-view sender 未调度、receiver/sender 选择不当、dense grid 但 object-level detector support 不足三类失败模式。当前图可用于 appendix/rebuttal draft，正式论文可进一步美化图例并补 predicted box overlay。
- 已完成 P9 EdgeCooper 写作参考核查：新增 `docs\doc_workspace\SGCP\edgecooper_writing_reference.md`。结论是可借鉴其系统化 evaluation structure：平台/通信设置、comparison algorithms、qualitative evaluation、quantitative AP-communication tradeoff；但 SGCP 不引入 satisfaction rate，且必须更严格地区分 prediction sharing、raw-LiDAR sharing、edge-assisted reference 和 fully decentralized SGCP。
- 已建立 P0 论文 artifact 索引：`docs\doc_workspace\SGCP\paper_artifact_index.md` 和 `docs\doc_workspace\SGCP\artifacts\paper_artifact_index_20260719\paper_artifact_index.csv`。当前 Table 1、Figure 1/2/3、scheduler comparison、parameter sensitivity、runtime/NS3 appendix、qualitative case study 和 EdgeCooper 写作参考均已映射到 source artifact、脚本/命令、log/trace、commit 和 claim boundary。后续新 checkpoint 或新场景必须追加新索引版本。
- 已完成 P1 protocol-native claim audit：新增并更新 `docs\doc_workspace\SGCP\protocol_native_claim_audit.md`。当前 `main.tex` 已把 Full20Early 写成 upper reference、FullPerception-PCS 写成 repaired/tuned built-in baseline、EdgeCooper-HD 写成 edge-assisted/global assignment reference、Pure late 写成 controlled prediction-sharing reference，并把 SGCP 的 AP@0.3 贡献明确归因到 system protocol / inter-cluster late fusion，而不是单纯 scheduler 优势。Pure late detector/checkpoint fairness 已由 `detector_checkpoint_fairness.md` 关闭；P1 剩余风险只剩 early-fusion checkpoint 强度。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 实验章节第一版重构：主表加入 Pure late prediction-sharing reference，新增 protocol breakdown、fusion contribution、Pareto、SGCP-compatible scheduler comparison 和合并后的 parameter sensitivity Table。图文件已复制到 `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_*.pdf`。本机未检测到 LaTeX 编译工具，已完成轻量结构检查：table/figure/tabular begin-end 配对正常，新增 label/ref 无缺失。
- 当前论文主方法为 `perception_aware_potential_game`（PAPG），主配置为 20 MHz / 10 subchannels / `rho_th=3` / `head_rb_budget=2` / inter-cluster late fusion。
- 当前可复现主结果：PAPG 41 帧 AP@0.3/0.5/0.7 = `0.81/0.78/0.39`，payload `32,049,872 bytes` / `62.54 Mbps`，410 scheduled links。
- 公平随机 baseline 已改为 forced-budget random：`0.77/0.73/0.38`，`31,613,424 bytes` / `61.68 Mbps`。旧 RandomRA/MWS payload 过低，只保留为 w/o-PPS 诊断。
- FullPerception/full 20-CAV early fusion 作为 centralized upper reference：`0.85/0.83/0.48`，`118.71 Mbps`，不作为同通信预算公平主对比。
- 已重新核查代码：`opencda/core/clustering/algorithms/resource_allocation/pcs.py` 对应 FullPerception 论文 PCS 调度算法，`mws.py` / `random_ra.py` 是同一问题上的 heuristic baseline。正规入口为 `fullperception_pcs`。本轮进一步修复 PCS 的 blind-spot cache key、grid mAP cache index、同一 blind-spot 粒度下的 utility/payload/grid-selection 对齐，并将默认 blind-spot split 调为 `division=12,min_overlap=0`。41 帧 scheduled-receiver 结果提升为 `0.59/0.53/0.22`、12,959,840 bytes / `25.29 Mbps`；11 帧 NS3 dry-run 每帧 5 条 scheduled request、0 skipped unscheduled。
- MWS/RS heuristic 已同步 tuned PCS blind-spot 粒度和 `sc_num` 口径。11 帧 sanity：MWS `0.36/0.32/0.15`、39.00 Mbps；RS/random `0.54/0.49/0.23`、14.95 Mbps。结论是它们继续作为 w/o-PCS heuristic 诊断，不进入主公平表。
- 另有后补 selective proxy 已改名为 `global_selective_proxy` 和 `cluster_local_selective_proxy`，不再占用 FullPerception 命名。重命名前结果为：global selective proxy 41 帧 `0.84/0.80/0.46`、`109.71 Mbps`；cluster-local selective proxy `0.80/0.76/0.41`、`75.94 Mbps`，并已补 11 帧真实 NS3 replay：110/110 application callback complete、110/110 RLC complete、0 PHY failures。这些是 proxy/diagnostic，不替代论文 PCS baseline。
- EdgeCooper 已从逐 receiver blind-spot proxy 升级为 `edgecooper_global_hd` network-aware / half-duplex proxy：诊断发现旧 `edgecooper_global` 的 73/110 delivery 主要来自同一 100 ms slot 内 receiver 同时作为 sender 的半双工冲突。新增半双工约束后，41 帧结果为 `0.81/0.78/0.42`、33,519,040 bytes / `65.40 Mbps`，11 帧真实 NS3 replay 为 110/110 application/RLC complete、0 PHY failures。该结果应归入 RSU/edge-assisted baseline；它对 PAPG 主线形成新的高 IoU 压力，论文需分层呈现 edge-assisted 与 V2V-only，或继续提升 PAPG AP@0.7。
- PAPG `B_h=3` 已完成 41 帧 probe：`0.80/0.78/0.40`、32,051,792 bytes / `62.54 Mbps`，avg source CAVs 降至 2.67。结论是简单提高 per-head RB 上限不能追平 EdgeCooper-HD 的 AP@0.7；下一步若继续提升 PAPG，应做高质量 source / target-grid coverage 保护。
- 已同步 `C:\Workspace\icdcs-paper\SGCP\main.tex`：主表加入 `EdgeCooper-HD (edge-assisted)` 行，并在正文明确它是 virtual edge-assisted reference，不属于 fully decentralized RSU-free V2V 公平 baseline。
- 已新增 `balanced_perception_aware_potential_game` / `bpapg` 独立分支，用 source-diversity marginal potential 和 source-history credit 验证“欠服务源车保护”方向。41 帧 source-balanced 结果为 `0.81/0.78/0.39`、62.54 Mbps，与 PAPG 主行 AP/通信量相同且 per-CAV upload distribution 未改变；11 帧 source-history credit 降至 `0.75/0.71/0.33`。结论：不能做朴素 source fairness，下一步必须用 detector-quality / target-quality 门控保护低频但关键的 source。
- 已新增 `quality_gated_perception_aware_potential_game` 和 `head_urgent_perception_aware_potential_game` 两个后续分支。QG-PAPG 11 帧为 `0.75/0.72/0.33`，说明有 quality gate 的 source-history 仍会伤 AP；HU-PAPG 41 帧为 `0.81/0.78/0.39`、62.54 Mbps，安全但无增益，`B_h=3` 11 帧也未改变结果。当前结论：不要继续调 source/head fairness 系数，下一步转向 detector/pre-NMS 级证据和目标点级关联。
- 已新增 `opencda.tools.sgcp_head_box_diagnostics`，对 PAPG dense-miss top40 导出每个 cluster head 在 inter-cluster late fusion 前的 detector best IoU/score。结果：nearest-head matched 0/40，any-head matched 0/40，late-fused matched 0/40，而 full-reference matched 40/40；nearest head 有正常数量 detector boxes 但目标 best IoU 全为 0。结论：dense-grid miss 不是 late fusion/NMS 抑制已有正确框，grid-level 点数过粗，下一步应做 GT box 内/周边的 object-level point association。
- 已新增 `opencda.tools.sgcp_object_point_association`，对 PAPG dense-miss top40 统计 SGCP constrained input 与 full-reference 中真正落入 GT BEV box/邻域的点数。结果：exact box 内仅 1/40 行完全无点，2m 邻域 0/40 行无点；但 31/40 行 nearest CAV 的 exact-box 点没有直接上传到 nearest head。Full-reference exact-box 平均 164.98 点，SGCP/full exact-box ratio 平均 0.62，18/40 行低于 0.5；nearest CAV 在 34/40 行是最佳 raw object-support source。结论：剩余漏检不是简单“没有点”，而是 coarse grid coverage 无法保证实例级形状支撑；下一步应设计 instance-support-aware scheduling。
- 已新增 `instance_support_potential_game` (`ispg`) 作为 PAPG 的实例支撑势函数 probe：在 coverage/target 两层中加入紧凑高密度 component、weak-head gain 和 unique-best-view gain。41 帧结果为 `0.80/0.78/0.39`、32,046,336 bytes / `62.53 Mbps`，与 PAPG 主参考 `0.81/0.78/0.39`、`62.54 Mbps` 基本持平但 AP@0.3 略低；trace 显示仍为 16/20 fused、10/20 uploaded、4/20 unscheduled。结论：实例支撑项只放在簇内 sender/grid utility 不够，下一步应推进到跨簇 receiver assignment / target-to-head routing。
- 已新增 `cross_cluster_instance_support_potential_game` (`ccispg`) 验证跨簇 target-to-head routing。三版 11 帧 probe：naive external 104/110 links，AP `0.68/0.64/0.37`；layered external 44/110 links，AP `0.75/0.71/0.33`；cap1 external 11/110 links，AP `0.75/0.72/0.33`。结论：跨簇路由能影响高 IoU，但自动 external sender 会伤 coverage；下一步应做 diagnostic-triggered / persistent-target-triggered cross-cluster routing，而不是全局放开。
- 已新增 `offline_inference --sgcp-routing-hints-csv` oracle/debug probe，用 object-level diagnostics 每帧最多替换 1 条 route 或重排 1 个已调度 sender 的 grids。11 帧 PAPG + routing hints 触发 9 次 frame-level replacement，external links 2/110，结果 `0.75/0.71/0.35`、8,521,936 bytes / `61.98 Mbps`。结论：精准 object-grid routing 可微弱提升 AP@0.7，但低阈值 AP 下降，说明 object-support gap 不是充分触发条件；下一步需要 detector-benefit / context-preservation proxy。
- 已将 routing hints 的已调度 sender 行为改为 context-preserving merge：最多插入 3 个 object-neighborhood grids，其余保留原高密度上下文 grids。11 帧结果仍为 `0.75/0.71/0.35`、8,563,440 bytes / `62.28 Mbps`，`full_detected_method_missed` 为 91。结论：低阈值下降不是单纯 whole-list replacement 破坏上下文造成；object-grid/object-box support 仍不足以预测 detector benefit。
- 已完成 detector-benefit post-hoc 对比：11 帧 PAPG 无 hint 基线为 `0.76/0.73/0.34`、8,598,224 bytes；merged routing hints 相比 PAPG 逐 GT 对比为 4 个 gained GT、15 个 lost GT，PAPG `full_detected_method_missed=82`，hint 为 91。结论：routing 可以修复少数诊断目标，但会破坏更多已覆盖目标；后续触发器必须保护已覆盖 object prototypes，而不是只看 object-support gap。
- 已将 ISPG、CCISPG 和 routing-hint 诊断结果收束为论文边界：这些结果解释 PAPG 与 EdgeCooper-HD 在 AP@0.7 上的结构性差距，但当前不进入主表。主表/回复口径保持 PAPG 为稳定 V2V-only 主算法，EdgeCooper-HD 为 edge-assisted/global-assignment reference；若继续算法改造，必须先有 proposal/objectness-level trigger，不能继续追加临时 routing 修补。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的 PAPG 主表一致性修订：PAPG 行不再把 AP@0.7/Mbps 误加粗为列最优；正文明确 PAPG 相比 communication-aware selective V2V 是 AP@0.3/0.5 提升、AP@0.7/通信量存在 tradeoff；结论弱化为低/中 IoU 改善和 NS3 子信道可行性，而不是无条件全面击败所有 heuristic。
- 已将 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的主表改成分组布局：Full-sharing / infrastructure-assisted references 与 RSU-free V2V baselines and SGCP variants 分开显示。EdgeCooper-HD 因使用 edge/global assignment 信息，被视觉上归入 reference 组，避免被误读为同类去中心化公平 baseline。
- 已对 `main.tex` 分组主表做轻量 LaTeX 结构检查：3 个 table/table* begin/end 配对、3 个 tabular begin/end 配对、`tab:mAP` 普通行均为 5 列、`\multicolumn{5}` 与表格列数匹配。仍未做 PDF 编译，因为本机未检测到 `latexmk/pdflatex`。
- 已根据最新审阅修正 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的 FullPerception 命名问题：`0.85/0.83/0.48, 118.71 Mbps` 行现在命名为 `Full 20-CAV early fusion` upper reference；FullPerception baseline 对应仓库 `pcs.py` / `fullperception_pcs`，当前 tuned 结果为 `0.59/0.53/0.22, 25.29 Mbps`。这避免再把 FullPerception baseline 和 full-sharing AP 上界混写。
- 已重跑 PAPG 与 EdgeCooper-HD：11 帧 repeat 为 PAPG `0.76/0.73/0.34`、EdgeCooper-HD `0.77/0.73/0.37`；41 帧 repeat 为 PAPG `0.81/0.78/0.39`、EdgeCooper-HD `0.81/0.78/0.42`。结论是两者接近为稳定结果，EdgeCooper-HD 作为 edge/global assignment reference 保留 AP@0.7 优势。
- PAPG 真实 NS3 replay 已完成 11 帧：110/110 scheduled requests application callback 和 RLC request complete，RLC drops=0，PHY decode failures=0。
- `main.tex`、`main_table_candidate.md`、`results.md`、`baseline_fairness.md`、`fullperception_baseline_revision.md`、`baseline_reproduction_plan.md`、`rebuttal_draft.md` 和 `rebuttal_short.md` 当前应以 PAPG 主线和显式 baseline 分层为准；早期 coverage-aware 10ch/20ch 只作为消融或资源敏感性结果。

## 论文当前主要问题

根据 SGCP 论文和审稿意见，当前最需要处理的问题集中在三类。

### 写作问题

- 相关工作和 novelty 对比不足，尤其需要区分本文与通用 coalition formation 方法、Smartform 类工作、已有去中心化协同感知方法的差异。
- FullPerception baseline 设置不清楚：无 RSU 场景下是否使用虚拟 RSU、全局信息或集中调度，需要明确。
- `f(rho)` 感知效用函数标定过程过短，缺少拟合方式、数据采样方式、复现流程和误差说明。
- `T_min^stab = 500 ms`、`N_max = 4`、`rho_th = 2.0` 等关键参数缺少依据。
- topology change trigger 已形成机制规格，但尚未接入在线/离线代码统计。
- 100 ms 协作周期内的实时性论证过于简略。

### 需要补充实验

- 核心模块消融：无稳定窗口、无 coalition formation、无 PPS、仅 early fusion、仅 late fusion 等。
- 参数敏感性：`T_min^stab`、`N_max`、`rho_th`、CAV 数量、带宽/子信道数量。
- 更公平的 baseline：FullPerception PCS、global/cluster-local selective proxy、其他 V2V-only/decentralized 方法。
- `f(rho)` 标定曲线和泛化实验。
- 运行时开销：聚类耗时、调度耗时、通信耗时、融合与检测耗时。
- 稳定性指标：cluster lifetime、reconfiguration 次数、fragmentation rate。

### 需要完善机制

- 周围 cluster 已满时车辆如何处理，避免大量小 cluster 或孤立车辆。
- 是否允许 cluster merge/split，以及是否允许超过 `N_max` 的临时重组。
- 成员加入后是否重新计算已有成员边际贡献，如何避免振荡。
- coalition value 中用 late-fusion `max` 作为 baseline 的合理性和适用边界。
- potential game 的 exact potential 条件需要更严谨。
- fully decentralized 场景下全局 density/utility 信息如何传播、同步和计入开销。

## 当前文档状态

- 已建立 SGCP 文档工作区。
- 已整理论文问题和任务方向。
- 已记录 SGCP 在线 CARLA 仿真命令：`python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug`，启用 NS3 时追加 `--network`。
- 已新增 SGCP 数据导出场景入口：`v2xp_cluster_carla_datadump`。
- 已新增 OPV2V 风格离线帧加载器基础能力。
- 已完成无需 CARLA 的 OpenCOOD 单帧推理 smoke test：`E:\data\opv2v\test` 第一帧 early fusion 输出 18 个预测框、19 个 GT 框。
- 已完成一次实际 SGCP 数据导出：`D:\Data\Carla\2026_07_15_01_26_56`，20 个 CAV，每个 41 帧 `.pcd/.yaml`。
- 已验证新导出的数据可离线进入 OpenCOOD：`000060` 帧 early fusion 输出 62 个预测框、71 个 GT 框。
- 已完成无 NS3 离线全量测试：41 帧 early fusion，AP@0.3/0.5/0.7 = 0.85/0.83/0.48。
- 已梳理 SGCP 离线回放工程接口，新增 `offline_replay.md`，明确从 dump 帧重建 `cav_world/vehicle_manager/v2x_manager/lidar grid` 的最小适配层。
- 已新增离线状态适配层 `opencda.core.common.offline_replay`，可从单帧 dump 重建 `OfflineCavWorld/OfflineVehicleManager/OfflineV2XManager/OfflineLidarGrid`。
- 已新增离线 SGCP 回放入口：`python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla`。
- 已验证 `000060` 单帧无需 CARLA 可运行 `CoalitionGame`，20 个 CAV 输出 6 个 cluster。
- 已验证单帧默认 `NaiveRA` 资源分配可输出 380 条 channel allocation。
- 已完成 3 帧离线 SGCP smoke test，确认逐帧状态重建可运行。
- 已扩展 `opencda.tools.offline_replay` 多帧汇总能力，支持 `--summary-only` 输出稳定性和运行时指标。
- 已完成 `D:\Data\Carla\2026_07_15_01_26_56` 全量 41 帧离线 SGCP 回放汇总：平均 6.00 个 cluster、平均 cluster size 3.33、0 个孤立 CAV、11 次 reconfiguration、76 次 vehicle-head change、平均 cluster lifetime 6.65 帧、平均总耗时 129.78 ms、平均 RA 耗时 1.13 ms。
- 已新增资源分配算法 factory，在线 `ClusteringScheduler` 与离线 `offline_replay` 均可按配置/参数选择 `potential_game`、`pcs`、`mws`、`random`、`naive`。
- 已将 SGCP 资源分配默认值统一为配置默认 `potential_game`；`naive` 保留为 baseline/fallback。
- 已完成 `potential_game` 全量 41 帧离线 SGCP 回放：平均 6.00 个 cluster、平均 cluster size 3.33、0 个孤立 CAV、11 次 reconfiguration、76 次 vehicle-head change、平均 cluster lifetime 6.65 帧、平均总耗时 285.82 ms、平均 RA 耗时 111.85 ms。
- 已将离线 SGCP 回放接入 OpenCOOD 约束感知评估：`--sgcp-constrained` 会先运行 `CoalitionGame + potential_game`，再按在线 `CoperceptionManager` 的 grid upload 语义裁剪 frame。
- 已完成 41 帧 SGCP constrained early-fusion 评估：AP@0.3/0.5/0.7 = 0.35/0.35/0.21，平均上传 106,790.63 bytes/frame，平均 2.98 个 source CAV/frame。
- 已新增 `all-cluster-heads` 约束评估口径，并完成 41 帧全量实验：246 个簇头样本，AP@0.3/0.5/0.7 = 0.36/0.34/0.17，平均上传 109,415.48 bytes/sample，平均 2.67 个 source CAV/sample。
- 已新增 SGCP inter-cluster late fusion 离线评估，并完成 41 帧全量实验：每帧融合 6 个 cluster head，AP@0.3/0.5/0.7 = 0.77/0.73/0.35，平均上传 109,415.48 bytes/source，总 payload 26,916,208 bytes。
- 已完成 w/o PPS 第一版调度消融：`random` late-fusion 41 帧 AP@0.3/0.5/0.7 = 0.44/0.39/0.17，总 payload 9,725,376 bytes；`mws` late-fusion 41 帧 AP@0.3/0.5/0.7 = 0.31/0.26/0.11，总 payload 9,910,032 bytes。
- 已修复 `PCS/MWS/RandomRA` 离线执行接口：补齐 `PCS.run()`、`self.cav_world` 和缺失导入，使 baseline scheduler 可通过统一 resource allocation builder 运行。
- 已修复 OpenCOOD late fusion 离线推理路径，并完成 41 帧 full 20-CAV late fusion reference：AP@0.3/0.5/0.7 = 0.91/0.85/0.51。
- 已新增 `--t-min-stab` 离线参数，并完成 w/o stability window 第一版消融：`T_min_stab=0` 时 41 帧 AP@0.3/0.5/0.7 = 0.77/0.73/0.35，重配置 11 次、vehicle-head change 76 次、平均 cluster lifetime 6.65 帧，与默认 `T_min_stab=1.0` 一致。
- 已完成 `T_min^stab=100/300/500/700/1000 ms` 参数敏感性第一版：所有设置在当前 41 帧 dump 上均为 AP@0.3/0.5/0.7 = 0.77/0.73/0.35，reconfiguration 11 次，vehicle-head change 76 次。
- 已新增 `--clustering` 离线参数，并完成 w/o coalition formation 的 singleton 第一版参考：20 个单车簇、0 次重配置、AP@0.3/0.5/0.7 = 0.82/0.76/0.37；当前通信统计未计入 prediction-level late-fusion 开销。
- 已新增 `--n-max` 离线参数，并完成 `N_max=2/3/4/5/6` 参数敏感性第一版：AP@0.5 分别为 0.74/0.71/0.73/0.71/0.71；`N_max=2` 当前 AP 最高但 cluster-head source 更多，`N_max=5/6` 在当前 dump 上结果完全一致。
- 已新增 `--rho-th` 离线参数，并完成 `rho_th=0.5/1.0/2.0/3.0/4.0` 参数敏感性第一版：低阈值降低 payload 但 AP 下降，高阈值提升 AP@0.7 到 0.37 且增加 payload；当前默认 `2.0` 是通信-精度折中点。
- 已新增 `--cav-count` / `--cav-ids` 离线参数，并完成 5/10/15/20 CAV 子集规模敏感性第一版：AP@0.5 分别为 0.32/0.59/0.66/0.73；该实验是同一 dump 的协同车辆子集，不等同真实交通密度重采样。
- 已新增 `--num-channels` / `--bandwidth-mhz` 离线参数，并完成网络资源敏感性第一版：5/10/20 子信道 AP@0.5 分别为 0.53/0.73/0.73，平均上传 payload 分别为 60,225/109,415/139,300 bytes/source，平均 selected grids 分别为 45.58/87.32/117.18。
- 已完成 PPS 带宽参数第一轮复核：`bandwidth_per_channel` 已进入 `PotentialGame` 的 max-grid、SINR 和 data-rate 计算；20/40/80 MHz 当前结果一致，是因为本 dump 的调度不由带宽上限主导，主要受子信道数量、每簇头 `B_h=1` RB 和候选网格集合约束。
- 已完成低带宽瓶颈触发实验：0.1/0.5/1.0 MHz 下 AP@0.5 分别为 0.22/0.50/0.61，平均 selected grids 分别为 0.00/4.32/9.66，证明带宽约束可观测生效。
- 已新增并更新 `baseline_fairness.md`，明确 FullPerception baseline 对应 `pcs.py` PCS；full 20-CAV early/late fusion 只作为 upper/reference，global selective proxy 不作为同通信预算公平主对比。
- 已新增 same-budget CAV-only selective-sharing baseline：`nearest`、`density`、`communication_aware` 三种成员选择，复用 SGCP clustering + inter-cluster late fusion，默认每簇头 2 个成员、87 个 grid budget；41 帧结果分别为 AP@0.3/0.5/0.7 = 0.76/0.73/0.37、0.77/0.74/0.39、0.78/0.75/0.40。
- 已新增 `topology_trigger.md`，定义 SGCP topology change trigger 的邻居变化、相对运动、链路质量、utility 下降、hard failure 和 periodic guard 条件，并明确 `NO_CHANGE/LOCAL_REPAIR/RECLUSTER` 三类输出。
- 已将 topology trigger 统计接入 `opencda.tools.offline_replay`：支持 summary 输出和 `--print-topology-events` 逐 transition 明细；当前支持 `dump` 与 `pose_delta` 两种速度源，默认使用相邻帧位置差分速度。
- 已完成 topology trigger 速度源复核：`ego_speed` 来自 `get_speed(vehicle)`，默认单位是 km/h；`pose_delta` threshold 3/4 m/s 可覆盖 11 次实际 reconfiguration，5 m/s 会漏掉 2 次。
- 已将 topology trigger gate 接入在线 `ClusteringV2XManager` first version：默认关闭；打开后按 hard failure、邻居集合变化和 periodic guard 决定是否重跑 `CoalitionGame`，否则沿用上一轮 cluster。
- 已新增 `cluster_capacity_policy.md`，明确 cluster 已满时采用硬 `N_max`、保留当前 cluster、加入未满更优 cluster、singleton fallback 和 inter-cluster late fusion 补偿；merge/split 不作为独立无限制原语。
- 已将 cluster capacity 统计接入 `opencda.tools.offline_replay`：默认 `N_max=4` 的 41 帧 dump 中，平均每帧 3.12 个满簇、99.15 次满簇候选跳过、singleton cluster ratio 为 0、small-cluster ratio 为 0.187；`N_max=2/3/5/6` sweep 已写入 `results.md`。
- 已新增 `opencda.tools.sgcp_density_calibration` 和 `f_rho_calibration.md`，完成 41 帧 dump 的 `f(rho)` 标定第一版：788,020 个 CAV-grid 样本中非零网格占 5.98%，非零 density p90=1.40、p95=3.60；默认 `rho_th=2.0` 筛出 3,383 个 high-density grid，约占非零网格 7.18%。
- 已将 SGCP control overhead 估算接入 `opencda.tools.offline_replay`，并新增 `control_overhead.md`。当前 41 帧默认 SGCP 的控制面总开销为 187,112 bytes，平均 4,563.71 bytes/frame，约为 inter-cluster late-fusion 点云 payload 的 0.70%。
- 已新增 `potential_game_conditions.md`，复核 PPS exact potential 成立条件。当前代码更准确的口径是 potential-guided constrained best-response scheduling；若论文继续使用 exact potential game，需要限定固定 cluster、固定候选 grid、硬 feasibility constraints，并补显式势函数/monotonicity 证明。
- 已将 PPS convergence diagnostics 接入 `PotentialGame` 与 `offline_replay`：当前 41 帧默认 SGCP 中 41/41 帧均在 `max_iter=20` 前收敛，平均 3.00 iterations、10.00 scheduled links/frame、523.90 selected grids/frame，最大 RB occupancy 为 1。
- 已新增 `paper_revision_plan.md`，把 topology trigger 表述矛盾、实时性、`f(rho)` 标定、baseline fairness 和 game-theoretic convergence 转成 `main.tex` 级别替换建议；其中 topology trigger 矛盾的 P4 写作项已完成。
- 已新增 `related_work_novelty_revision.md`，完成 P4 related work / novelty 写作口径：SGCP 的新意不写成 coalition game 本身，而是感知效用标定、稳定性约束、容量约束、PPS 子信道可行性和分层 early/late fusion 的组合。
- 已新增 `parameter_calibration_revision.md`，完成 P4 `f(rho)` 标定过程和 `T_min^stab/N_max/rho_th` 参数依据写作口径；特别明确当前短序列不能证明 `T_min^stab=500 ms` 最优，只能写为五个感知周期的保守默认。
- 已新增并更新 `fullperception_baseline_revision.md`，完成 P4 FullPerception baseline 写作口径：`pcs.py` 是正式 FullPerception PCS baseline；full 20-CAV early/late fusion 作为 centralized full-sharing upper reference；同通信预算主对比采用 CAV-only nearest/density/communication-aware selective sharing，并明确当前短序列上 SGCP 不能声称 AP 全面领先强 selective baseline。
- 已扩展 `opencda.tools.offline_replay` 输出 `runtime_breakdown_ms`，并新增 `runtime_feasibility_revision.md`。当前 41 帧控制面 profiling 中，SGCP algorithm total 平均 105.24 ms，coalition formation 64.39 ms，PPS scheduling 40.58 ms；离线读取和 world build 约 599.73 ms/frame，属于 replay artifact，不计入在线周期。论文中应写为 near-real-time feasibility，而不是完整端到端 100 ms 保证。
- 已新增 `reproducibility_manifest.md`，固定当前可复现实验的 OpenCDA commit、数据集、命令、结果和 artifact 路径；同时确认 `main.tex` 旧主表 `NC/RS/MUG/FullPerception/Ours = 0.13/0.31/0.37/0.81/0.85@AP0.3...` 尚未找到原始日志、随机种子和代码版本，论文修订应以当前复现结果替换或找回旧日志后再使用。
- 已完成真实 CARLA 在线 topology-trigger gate 短回归，并新增 `online_topology_gate_regression.md`。通过 `OPENCDA_CLUSTERING_CONFIG=networking_clustering_topology_gate.yaml` 打开 gate，通过 `OPENCDA_ONLINE_TICKS=35` 自动结束在线仿真；回归正常退出，CP counter=8，AP@0.3/0.5/0.7 = 0.84/0.82/0.69，cluster trigger 为 `initial=1`、`neighbor_set_change=1`、`head_member_unreachable=3`、`skip=0`。结论：gate 接入和日志生效，但当前 35 m 通信范围下 hard condition 持续触发，不能用该回归证明 skip 收益。
- 已进入主表修复阶段，新增 `opencda.tools.offline_inference --sgcp-trace-output` 和 `protocol_audit.md`。第一轮离线协议审计显示：41 帧 SGCP inter-cluster late-fusion 共有 246 条 cluster-head receiver trace，`missing_channel_rows=0`，总通信 26,916,208 bytes，AP@0.3/0.5/0.7 = 0.77/0.73/0.35；分簇、grid selection 和 channel allocation 均已真实进入融合输入，暂未发现未调度 sender 绕过 PPS 的协议 bug。
- 已新增 `--sgcp-upload-mode {grid,head_only,full_cluster}` 和 `mechanism_probe.md`。41 帧机制 probe 显示 head-only 为 0.26/0.22/0.09，SGCP grid-constrained 为 0.77/0.73/0.35，full-cluster upload 为 0.82/0.79/0.42；SGCP 使用约 60.0% 的 full-cluster payload，主要 AP 损失集中在 grid/PPS 选择，cluster formation 和 inter-cluster late fusion 主体可用。
- 已新增 `--sgcp-grid-selection-mode {utility,random}`。Random grid probe 保留同一 PPS scheduled links 和每条 link 的 grid 数量，仅将具体 grid 替换为确定性随机候选；41 帧结果为 AP@0.3/0.5/0.7 = 0.78/0.75/0.36，总 payload 27,908,560 bytes，`missing_channel_rows=0`。该结果略高于当前 utility selection，说明 grid utility 对检测 AP 的排序能力不足，是下一轮算法改造优先目标。
- 已扩展 grid scoring / selection 改造探针：`raw_density` 为 0.74/0.70/0.37，`density_distance` 为 0.74/0.71/0.37，`spatial_diverse` 为 0.79/0.75/0.37。三者均保持 SGCP/PPS protocol path；其中 `spatial_diverse` 保留同一 scheduled links 和 grid count，用 density-aware spatial cover 替换原始选格，当前高于 utility 与 random-grid，说明 coverage-aware grid selection 是可继续推进的主表修复方向。
- 已完成 `spatial_diverse` 子信道 sweep：5/10/20 子信道分别为 AP@0.3/0.5/0.7 = 0.56/0.53/0.27、0.79/0.75/0.37、0.80/0.76/0.41；payload 分别为 14,815,408、28,743,280、37,912,544 bytes。10 子信道是低通信主点，20 子信道 AP@0.7 接近 full-cluster 0.42 且 payload 低约 15.5%。
- 已将 `offline_ns3_replay` 对齐 coverage-aware 主表候选，支持 `--num-channels`、`--bandwidth-mhz`、`--sgcp-grid-score-mode` 和 `--sgcp-grid-selection-mode`。10 子信道 `spatial_diverse` 11 帧 NS3 replay 完成：110/110 scheduled request application/RLC complete，44 条未调度需求被 OpenCDA replay 跳过；20 子信道 `spatial_diverse` 完成：154/154 scheduled request application/RLC complete。两者 `MANUAL_CMD_REJECT=0`，PSCCH/PSSCH decode failure 均为 0。
- 已补齐 `offline_ns3_replay --rho-th` 参数，并完成 `spatial_diverse` 10 子信道 `rho_th=3.0` 的 11 帧 NS3 replay：110/110 scheduled request application/RLC complete，44 条未调度需求跳过，PHY failures 为 0。至此主表推荐的 10ch tuned low-budget row 不再是 NS3 pending。
- 已补全 FullPerception 口径：当前 RSU-free dump 无真实 RSU sensor；可复现 centralized full 20-CAV early reference 为 AP@0.3/0.5/0.7 = 0.85/0.83/0.48，non-ego CAV 上传 payload 60,838,528 bytes。FullPerception PCS 使用 `pcs.py`，当前 tuned 结果为 0.59/0.53/0.22、25.29 Mbps；高预算 density/communication-aware selective baseline 为 0.80/0.76/0.40，payload 37,710,864 bytes。
- 已确认旧 Random/MWS scheduler payload 过低（约 9.7/9.9 MB），未充分利用 10 子信道资源，不适合作“通信量减少”的主表证据。它们保留为 w/o PPS 消融；公平主对比改用 payload-matched selective baselines、random-grid same-link probe 和 SGCP spatial-diverse 10/20ch。
- 已完成 `spatial_diverse` 的 `rho_th` 点云阈值 sweep：`rho_th=1/2/3/4` 中，`rho_th=3.0` 达到 0.79/0.76/0.38，payload 29,405,296 bytes，是默认 10ch 低通信候选之外的更高 AP 阈值配置。
- 已新增 `main_table_candidate.md`，把可复现结果收束为论文主表候选：FullPerception centralized upper reference 118.71 Mbps、payload-matched selective high-budget 73.58 Mbps、SGCP coverage-aware 10ch `rho_th=3` 为 57.38 Mbps、SGCP coverage-aware 20ch 为 73.98 Mbps，并明确 Random/MWS 只作消融。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第一轮论文落地：替换旧主表、旧通信开销图引用和过强 FullPerception 对比，改为 centralized full-sharing upper reference、capacity-matched V2V baseline、SGCP coverage-aware 10/20ch、NS3 request-level delivery 和 near-real-time feasibility 叙事。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第二轮机制修订：PPS 机制改为 potential-guided constrained scheduling，加入 coverage-aware / density-aware spatial diversification grid selection；同时修正 topology trigger 与 every-cycle PPS 的关系，并弱化无条件 exact-potential/Nash guarantee。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第三轮参数标定修订：把 `f(rho)` / `rho_th` 标定写成可复现协议，并加入 coverage-aware 10ch `rho_th` sensitivity 表，明确 `rho_th` 依赖 LiDAR、grid size、预处理和 detector backbone，不作为通用常数。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第四轮参数依据修订：补入 `N_max=4` 的容量控制解释、99.15 次/frame 满簇候选跳过统计，以及 `T_min^stab=500 ms` 作为五个 10 Hz sensing cycle 的保守滞回默认；同时说明当前短序列中 `T_min^stab=100--1000 ms` 不改变 AP/reconfiguration。
- 已新增 `rebuttal_draft.md`，按 R2/R3/R4 concern 整理可直接迁移的 rebuttal 素材，覆盖 coalition baseline、满簇处理、FullPerception 公平性、`f(rho)` 标定、`T_min^stab`、公平 baseline、runtime、topology trigger 和 NS3 request-level delivery。
- 已新增 `rebuttal_short.md`，将长 rebuttal 素材压缩为最终可粘贴版本，保留 FullPerception 公平性、payload-matched V2V baseline、`f(rho)`/`rho_th`/`N_max`/`T_min^stab` 参数依据、runtime、topology trigger、NS3 三层可靠性和更保守的 claim boundary。
- 已新增 `online_ns3_short_regression.md`，明确真实 CARLA+NS3 短回归的三段启动顺序、日志保存位置、时间同步验收条件和 subchannel 语义检查项。本轮已通过 `test_network_time_sync` 轻量回归，且确认没有 CARLA/NS3 残留进程。
- 已执行真实 CARLA+NS3 35 tick 短回归并修复在线初始化时序 bug：原先 sender thread 在 1 车注册时就发送 `vehicles_num=1`，导致 NS3 之后收到 20 车位置帧时重新初始化并 address collision / SIGABRT。现在 `NetworkManager` 等待 `mark_vehicle_registration_complete()` 后再初始化 NS3，并过滤 `carla_id=None` 帧；修复后 NS3 20 车初始化、38/38 sync_ack、无 sync timeout、无 fatal、无 manual reject，在线 AP@0.3/0.5/0.7 = 0.86/0.84/0.74。
- 已诊断并修复在线 PHY failure 的第一类原因：`PotentialGame` 每轮只清空算法内部 strategies，没有清空各车辆 `ClusteringScheduler.channel_allocation`，导致在线多轮 CP 旧链路残留；修复后真实 CARLA+NS3 35 tick 短回归中 PSCCH/PSSCH decode failures 从 `1836/480` 降至 `95/10`，`MANUAL_CMD_REJECT=0`，在线 AP@0.3/0.5/0.7 从 `0.86/0.84/0.74` 升至 `0.88/0.88/0.79`。
- 已新增 `opencda.tools.online_ns3_log_eval`，将在线 OpenCDA 重复 incomplete 轮询行压缩为 upload episode。最新策略清空回归中 `184` 行 incomplete 只有 `6` 个真实 partial episode，且每个都缺少一个 `10000` bytes fragment；这将剩余在线问题收窄到 fragment-level loss 后的重传/重调度或失败裁剪，而不是时间同步或子信道语义。
- 已新增可配置的在线 NS3 timeout reupload：`upload_timeout_slots`、`re_upload_when_timeout`、`max_reupload_attempts`。first trial 中 complete/partial episode 从 `21/6` 改善到 `39/3`，但暴露 late CAM completion 引发的 `KeyError`；已修复为安全处理缺少 `uploading_cavs` start slot 的 late completion。修复后的 clean rerun 尚未完成：一次启动等待超时，一次 CARLA spawn collision，均未进入有效 NS3/reupload 验证。
- 已新增 `OPENCDA_CLEAN_WORLD_ON_INIT=1` 在线回归保护开关：ScenarioManager 在创建 CAV 前清理已有 `vehicle/sensor/walker/controller` 动态 actor 并 tick 3 次，以避免固定 spawn 点被上一轮残留 actor 占用。默认关闭，SGCP 在线自动回归命令显式打开。
- 已新增 `OPENCDA_CARLA_CLIENT_TIMEOUT`，允许在线回归把 CARLA client/load_world timeout 从默认 60 秒提高到 180 秒；最近两次 clean rerun 的 blocker 已从 spawn collision 转移为 `load_world('Town03')` 超时。
- 已新增 `OPENCDA_USE_CURRENT_CARLA_WORLD=1`，用于 CARLA 已经以目标地图启动时跳过 `client.load_world(town)`，避免本轮反复出现的 `load_world('Town03')` RPC 超时。
- 已确认当前在线 clean rerun 的最新 blocker 是 CARLA RPC readiness：直接用 Town03 参数启动 CARLA 并等待 120 秒后，`carla.Client(...).get_world()` 在 180 秒 timeout 下仍超时。2000 端口 ready 不代表 CARLA Python API 可用；继续 NS3/reupload 前必须先恢复 CARLA smoke test。
- 已新增 `opencda.tools.carla_rpc_probe`，在线 OpenCDA/NS3 前必须先通过 `conda run -n opencda python -m opencda.tools.carla_rpc_probe --expect-map Town03 --timeout 30 --wait 180`。
- 已将 CARLA RPC probe 和在线保护开关补充到全局 `docs/doc_workspace/environment.md`，作为 SGCP/LGCP 共用前置检查。
- 已新增 `opencda.tools.sgcp_protocol_trace_summary`，将 `offline_inference --sgcp-trace-output` 的 246 行 receiver trace 汇总为 41 行 frame summary。当前 SGCP 41 帧审计中每帧固定 6 个 cluster-head receiver、10 条 PPS channel links、平均 16 个 fused CAV、平均 10 个 uploaded sources、平均 payload 656,492.88 bytes/frame、平均 selected grids 523.90/frame，`missing_channel_rows=0`。AP 仍按 OpenCOOD 全局累计指标报告，避免把 pred/GT count 误当逐帧 AP。
- 已新增 `offline_inference --clustering fixed_first_frame` 机制 probe：首帧运行 coalition game 固定 head/member 模板，后续帧只复用 cluster membership，每帧仍重新计算 density、PPS 和融合。41 帧结果为 AP@0.3/0.5/0.7 = 0.73/0.70/0.33，总 payload 26,325,216 bytes，`missing_channel_rows=0`；动态 coalition 的 0.77/0.73/0.35 仍更好，说明 topology-aware cluster 更新有贡献，但主表主要损失仍来自 grid/PPS 选择质量。
- 已新增 `offline_inference --head-rb-budget` 和 `offline_ns3_replay --head-rb-budget`，用于覆盖 `PotentialGame` 每簇头最多使用 RB 数 `B_h`。默认仍为 1。41 帧 `spatial_diverse,B_h=2` 为 0.75/0.72/0.41、27,086,400 bytes；`spatial_diverse,rho_th=3,B_h=2` 为 0.76/0.72/0.42、27,962,864 bytes、54.56 Mbps。该设置达到 full-cluster AP@0.7，但 AP@0.3/0.5 下降。11 帧 NS3 replay 已验证 `B_h=2,rho_th=3` 为 110/110 application/RLC complete、PHY failures 0。
- 已新增 `offline_inference --sgcp-late-nms-thresh` 做 inter-cluster late fusion NMS 消融。`spatial_diverse,rho_th=3,B_h=2` 下，late NMS 0.05/0.15/0.30 分别为 0.73/0.70/0.40、0.76/0.72/0.42、0.75/0.71/0.41；默认 0.15 最好，说明 AP@0.3/0.5 下降不是简单 NMS 阈值问题。
- 已新增 `opencda.tools.sgcp_late_fusion_log_summary`，从 offline inference stdout 汇总 source/fused prediction boxes、suppressed boxes、fused GT 和 payload。诊断显示 `B_h=2,rho_th=3` 的 fused GT 平均为 64.83，低于 `B_h=1,rho_th=2` 10ch 的 69.00 和 20ch 的 69.29；fused prediction 平均也从 55.90 降到 53.71。该结果支持“`B_h=2` 改善高 IoU 定位但缩窄覆盖/召回面”的解释，后续应检查 cluster-head/member selection 与 target coverage，而不是继续调 late NMS。
- 已新增 `opencda.tools.sgcp_trace_coverage_summary`，从 SGCP receiver-level trace 汇总 per-CAV 和 per-frame 覆盖。诊断显示 `B_h=2` 在 10ch 下并没有增加 fused CAV 总数，仍为每帧 16/20 CAV、10 个 uploaded CAV、4 个 unscheduled member；但它把 CAV 6 从 41 帧上传降到 7 帧，把 CAV 5 从 6 帧升到 31 帧。该成员替换与 fused GT 下降一致，下一步算法应考虑 coverage fairness、persistent contributor protection 或 target coverage fallback。
- 已新增 `offline_inference --sgcp-coverage-fallback {none,persistent}` 作为默认关闭的 coverage repair probe。11 帧 `B_h=2,rho3` 对照中，无 fallback 为 0.69/0.64/0.34，persistent fallback 为 0.67/0.62/0.34；fallback 执行 7 次 frame-level replacement，`missing_channel_rows=0`，但 AP@0.3/0.5 轻微下降。结论是 CAV 级 history fairness 不足以作为替换准则，后续应转向 detector-quality / target-level coverage proxy。
- 已新增 `opencda.tools.sgcp_source_quality_summary`，从 receiver-level trace 汇总 pred/GT ratio、zero/low-ratio rows 和 uploaded-CAV quality proxy。41 帧 `B_h=2,rho3` 的 avg pred/GT ratio 为 0.4461，高于 `B_h=1` 10ch 的 0.3928；但 CAV 6 作为高质量长期贡献者，上传从 41 行降到 7 行，avg pred/GT ratio 约 0.63/0.57，明显高于被增加的 CAV 5。下一步应做 quality-weighted coverage，而不是 plain coverage fairness。
- 已将 `--sgcp-coverage-fallback` 扩展为 `quality_persistent`，使用历史 receiver-level pred/GT proxy 作为替换安全门。11 帧 `B_h=2,rho3` 上，quality-persistent fallback 为 0.69/0.64/0.34、0 次 replacement，等同 no fallback；plain persistent 为 0.67/0.62/0.34、7 次 replacement。结论：质量门槛能阻止有害替换，但 CAV 级 fallback 仍过于保守，下一步需要 object/target-aware 候选生成。
- 已排查用户 2026-07-17 在线 CARLA+NS3 结果 `0.86/0.86/0.71` 高于离线主表的原因：该 run stdout 显示 `cp counter=1`，日志解析显示只有 3 个 `CP_EVAL_FRAME/CP_SUBMIT_FRAME`、185 个 `CP_WAIT_FRAME`，ego=1 等待 34 次；11 个 upload episode 均为 application partial，没有 complete episode。结论：高 AP 是极少数在线 CP 统计帧的结果，不能直接替换 41 帧离线主表。
- 已对齐在线/离线通信量统计：`opencda.tools.online_ns3_log_eval` 现在解析 CP wait/submit 和通信报告；`NetworkManager.get_communication_report()` 新增 `duration_s`、`total_payload_mbps`、`try_payload_mbps`。用户 run 的 4,495,080 total bytes / 38 slots / 0.1 s 对应 9.46 Mbps；try upload 3,367,776 bytes 对应 7.09 Mbps。后续论文/主表应明确区分 total counted traffic 与 point-cloud upload payload。
- 已修复在线 CP 容易长期等待 NS3 incomplete fragments 的评价阻塞：新增 `CoperceptionManager.upload_wait_exhausted()`，当 pending upload 超过 timeout/re-upload 预算后允许用实际到达的 partial uploads 继续 CP；late-fusion submit 改为每次 ego late-fusion frame 均可计入统计，以对齐离线“每帧提交一次”的 AP 口径。需要用固定 tick 在线重跑验证 `cp_submit_frames` 是否随 tick 增长。
- 已完成 SGCP 11 帧离线 NS3 request-level replay：154 条 SGCP intra-cluster request，CAM callback delivery ratio 0.558442，request with any RLC RX event ratio 0.974026。后者仅表示 request_id 至少出现一个 RLC RX 片段/事件，不代表完整 request delivery。
- 已确认当前可运行环境版本快照，并记录到 `docs/doc_workspace/environment.md`：OpenCDA HEAD `fcc29fdc9ee9a9fe694c12e1fb6792b4d41bccac`，Python 3.7.10，CARLA Python API 0.9.11，PyTorch 1.10.0+cu113，Open3D 0.10.0.0，ns-3 wrapper `ns-3-dev-v2x-v1.1-dirty`。
- 已修复在线 CARLA-NS3 时间流速不一致问题：`NetworkManager.time_slot` 不再将 `world.fixed_delta_seconds` 除以 5，`current_sim_time`、`current_time_slot` 与 CARLA tick 统一推进。
- 已修复在线 NS3 初始化顺序：sender 线程等待车辆注册后先发送真实车辆数和第一帧 `vehicles_position`，再进入 `sync_request/sync_ack` 循环。
- 已新增 `opencda.tools.offline_ns3_replay`，可不启动 CARLA，直接从 dump 数据驱动 NS3 同步和传输请求。
- 已完成 3 帧离线 NS3 smoke test：帧时间 0.000/0.100/0.200 s，20 车、6 个 cluster、每帧 14 条 intra-cluster upload request，NS3 返回多条 `cam_received`。
- 已修复 NS3 manual subchannel 链路：NS3 默认按 OpenCDA 10 个子信道配置可用资源，manual scheduler 严格匹配 `physicalStart == sc_start` 和 `physicalLen == sc_num`，不再对越界子信道取模 wrap；长 JSON socket payload 改为累积解析；RLC command size 修正，避免最后残片落入默认随机调度。
- 已完成 `opencda.tools.ns3_link_probe` 四类链路回归：`success`、`edge_success(sc_start=9)` 均全量 CAM delivery；`conflict` 中两请求均落到 physicalStart=0，并产生 `PSCCH_DECODE_FAIL reason=decoded_overlap`，仅 1/2 CAM delivery；`out_of_band(sc_start=10)` 产生一次 `MANUAL_CMD_REJECT reason=out_of_band totalSubCh=10`，无 RLC TX/RX/CAM。
- 已将 `opencda.tools.offline_ns3_replay` 的 SGCP 资源分配从硬编码 `NaiveRA` 对齐为配置默认 `potential_game`，并默认跳过没有 `sc_start/sc_num` 的未调度需求，防止绕过 OpenCDA PPS 进入 NS3 默认调度。
- 已完成修复后 SGCP potential_game 11 帧 NS3 replay：110 条 PPS scheduled request、44 条 skipped unscheduled demand、CAM delivery 110/110、RLC RX 2970/2970、PHY failures 0、manual reject 0。
- 已补齐 NS3 request-level 三层统计：application callback、RLC request completion/partial/no_tx、PHY decode diagnostics。`offline_ns3_replay` 现使用全局唯一 `pkt_id`，避免 RLC 延迟事件跨帧错配。
- 已完成 NS3 暴露带宽/子信道窗口回归：`targetSubchannels=10` 时 110/110 PPS request application/RLC complete；`targetSubchannels=5` 时 `sc_start=0..4` 的 55 条全部 complete，`sc_start=5..9` 的 55 条全部在 bridge 层 `MANUAL_CMD_REJECT reason=bridge_out_of_band`，无 CAM/RLC/PHY 污染。
- 已将 NS3 request-level delivery 接入 communication-aware selective-sharing baseline：`offline_inference --ns3-link-quality-csv <rlc_by_request.csv>` 使用 `rlc_complete` 作为 link-quality cost。11 帧对照中，distance proxy 为 AP@0.3/0.5/0.7 = 0.71/0.67/0.31、总通信 7,977,680 bytes；NS3 RLC-complete aware 为 0.68/0.63/0.27、总通信 7,796,560 bytes。

## 当前阻塞项

- 论文旧主表已经确认缺少原始日志、随机种子和复现实验配置；当前处理策略是使用 `reproducibility_manifest.md` 中的已复现离线结果修订论文，或后续人工找回旧日志后再恢复旧表。
- 当前 OpenCDA 与 ns-3 工作区均存在未提交/dirty 状态；NS3 exposed-subchannel 修复与 request-level 统计已通过离线回归，但仍需要分别在 OpenCDA 仓库和 co-simulation 仓库提交。
- 离线 SGCP 回放已完成到“多帧 clustering + `potential_game` 资源分配 + 稳定性/运行时指标 + OpenCOOD 约束感知 mAP”层。
- 论文 SGCP 的 PPS/博弈调度当前按配置默认 `potential_game` 执行；已完成 `RandomRA/MWS` 初版对比，但 MWS 结果低于 random，需要结合论文文本复核 baseline 定义与效用函数。
- 早期 `ego-cluster-head` 与 `all-cluster-heads` constrained 评估只包含 intra-cluster early fusion；论文完整 SGCP 结果应优先采用 inter-cluster late fusion 口径。
- full 20-CAV late fusion reference 使用独立 OpenCOOD late checkpoint，不能直接等同于严格同通信约束的 SGCP late-only 消融。
- 当前 `T_min_stab=0` 消融未显示差异，说明当前 41 帧 dump 不足以证明稳定窗口贡献；需要更长序列或更强相对运动/topology change 场景。
- `T_min^stab=100-1000 ms` 参数实验同样未显示差异；当前只能作为“短序列无敏感性”的工程记录，不能作为论文中参数选择依据。
- singleton-cluster 结果会 late-fuse 全部 20 个 CAV 的检测框，但当前只统计点云 payload，不能直接作为零通信公平 baseline；需要补检测框交换开销或实现距离/随机固定簇对比。
- Cluster-local selective proxy / same-budget CAV-only selective baseline 已有 first version；communication-aware baseline 现在同时支持 distance proxy 与 NS3 RLC-complete cost。当前 dump 上 distance proxy AP@0.5/AP@0.7 高于 SGCP 且 payload 更高；NS3-aware 11 帧结果显示链路可行性约束会降低 AP 和通信量。论文主张需要谨慎转向稳定性、PPS channel feasibility 和动态网络约束，而不是简单宣称 AP 全面领先。
- `N_max` 参数实验显示非单调趋势；进入论文前需要补更长序列/不同密度场景，并计入 inter-cluster 检测框交换开销。
- `rho_th` 参数实验已显示通信-精度折中，`f(rho)` 密度分布标定已有第一版；论文级结论仍需要补跨场景/探测器泛化，必要时把 density bin 与 per-grid detection recall/IoU 绑定。
- CAV 数量规模实验目前只是固定场景子集实验；论文级“密度扩展”仍需重新导出不同 CAV/背景车密度的 CARLA 场景。
- 网络资源实验已证明子信道数量影响 PPS 结果；低带宽 stress test 已触发带宽瓶颈。论文中应谨慎区分“常规带宽下该 dump 已饱和”和“极低带宽下吞吐约束有效”。
- Dump 中速度字段目前使用 `ego_speed`；后续如需更严格动态稳定性，应确认单位并评估是否改为相邻帧差分速度。
- 在线 CARLA-NS3 时间同步修复已通过真实 80 tick 回归：`NetworkManager` 现在在每个 CARLA tick 内先发送位置和 transfer requests，再执行 `sync_request/sync_ack`，主循环等待 NS3 ack 后才进入下一 tick；严格模式 sync timeout 下限提升到 60 s。最新两轮在线回归均无 sync timeout，`MANUAL_RESOURCE_APPLY` 显示 `requestedStart == physicalStart`。`min_upload_count=1` 后 CP submit 从用户 run 的 4/5 次提升到 10 次，最佳在线 AP@0.3/0.5/0.7 = `0.70/0.68/0.58`，complete/partial episode = `55/2`，total/try payload = `25.48/17.85 Mbps`。无重传对照为 `0.64/0.59/0.50`、CP submit 7、complete/partial `45/3`，说明严格同步后仍需要 1 次受控 deadline 重传；后续论文应区分最终 request delivery 与 deadline-aware CP delivery。
- Topology trigger 已接入离线 replay 统计，但尚未接入在线 `ClusteringV2XManager` gate；单独 relative-speed trigger 仍偏敏感，在线 gate 应结合 neighbor-set change、utility drop 和 `T_min_stab` 滞回。
- 在线 topology trigger gate 已完成真实 CARLA 35 tick smoke regression；未观察到 skip，原因是当前默认 35 m 通信范围下持续触发 `head_member_unreachable`。若论文需要展示 reduced reconfiguration，应补一组更静态或更大通信范围的在线回归。
- Cluster capacity 策略已有离线统计支撑；optional replacement repair 尚未实现，进入论文前可明确为 future/optional enhancement。
- PPS `PotentialGame` 当前没有显式势函数、action replacement 和 `Delta Phi >= 0` 日志；论文中不宜无条件声称当前实现是完整 exact potential game。若要保留强理论表述，需要补代码诊断和证明。
- SGCP potential_game NS3 replay 已确认“PPS 已调度且无冲突的 request 全部成功”，且低暴露子信道场景能正确拒绝超出带宽窗口的 request；NS3 delivery/PDR 已先接入 selective-sharing baseline。下一步是把该 link-quality 反馈进一步接入 SGCP PPS 本身或 OpenCOOD mAP 的端到端丢包裁剪。
- Random-grid probe 已显示当前 utility selection 不优于随机候选；`spatial_diverse` 已超过 random-grid，且 20 子信道下 AP@0.7 接近 full-cluster upper reference。10ch `rho_th=2/3` 与 20ch `spatial_diverse` 候选的 NS3 request-level delivery 均已验证。最新 box-count、coverage 和 quality proxy 诊断显示：`B_h=2` 提升高 IoU，但低阈值 AP 下降来自覆盖对象替换、fused GT 减少，以及高质量 CAV 6 被系统性挤出；persistent coverage fallback 负面 probe 表明只按 CAV 历史欠覆盖替换会伤 AP，quality-persistent 只能阻止有害替换。主表候选、coverage-aware PPS 机制、`f(rho)` 参数标定、`N_max/T_min` 参数依据、rebuttal 长草稿和短版均已完成；下一步优先实现 object/target-aware candidate generation，再补真实在线 CARLA/NS3 短回归或压缩 rebuttal。
- 2026-07-17 新一轮主表重构确认：Full 20-CAV early 上界为 `0.85/0.83/0.48`；5ch/20MHz `spatial_diverse` stress 为 `0.56/0.53/0.27`、14,815,408 bytes，Random/MWS 分别为 `0.43/0.38/0.18`、9,531,504 bytes 和 `0.31/0.26/0.11`、9,989,952 bytes。Object-level 诊断显示 773 个 GT 为 full reference 可检出但 SGCP 5ch 漏检，且集中在少数持续目标和 ego 左侧/左后区域。新增 `object_clustered` 负面 probe 说明局部高密度聚集会伤覆盖；新增 `--max-upload-points-per-source` 点预算参数，`spatial_diverse,rho3,10ch,cap=3000` 为 `0.74/0.70/0.33`、19,510,848 bytes、38.07 Mbps。当前结论：SGCP 有可调通信旋钮，但还未满足“最高 AP 且最低 Mbps”的最终主表目标；下一步必须先把 Random/Greedy 改成强制使用统一带宽/统一 payload cap 的强 baseline，再进行公平主表比较。
- 已新增 `opencda.tools.sgcp_failure_diagnostics` 与 `failure_diagnostics.md`，将 GT world/ego 坐标、车辆坐标、cluster membership、PPS schedule、selected grid、object-grid 覆盖点数和 full-reference/method match 统一输出。10ch/rho3 诊断显示 111 个 full-reference 可检出但 SGCP 漏检的 GT 中，63 个主要是 target grid 只被其他 cluster head 覆盖，35 个是最近 head 已拿到较密点云但无最终匹配框，12 个是最近 head 仅得到稀疏点，1 个完全没有调度覆盖；这说明剩余 AP 损失的首要方向是 target-aware sender/receiver/grid 保护，晚期融合是 secondary debug。
- 已新增 `target_aware_potential_game` 资源调度算法，原 `potential_game` 保留为备份/消融基线。新算法在 allocator 内部采用两阶段机制：第一阶段沿用原 PotentialGame 的 sender/subchannel best-response，第二阶段用 target-aware multi-view utility 重选每条链路的 grid action。41 帧 20MHz/10ch/rho3 结果从旧 `spatial_diverse` 的 `0.79/0.76/0.38`、57.38 Mbps 提升到 `0.80/0.76/0.39`、60.62 Mbps；对象级漏检从 111 降到 106，`covered only by other cluster heads` 从 63 降到 56。11 帧 NS3 dry-run 已确认每帧 10 条 scheduled request、4 条 unscheduled demand skipped；真实 NS3 socket replay 待启动 NS3 后补。
- 已新增 `opencda.tools.sgcp_grid_miss_analysis` 与 `target_grid_case_study.md`，对 object 438/401/350/337 等持续漏检目标做 frame-level grid 诊断。结论是当前调度会让覆盖大量普通 grid 的 sender 挤掉最佳目标视角 sender，例如 frame 000068 中 CAV12 对 `3_0` 有 424 点且 rank=1，但 head4 原先调度 CAV9。已新增 `object_aware_potential_game` 分支，在同一 RB/subchannel 预算下引入 object-prototype utility、target-first grid selection 和 sender refinement；诊断层面可把 head4 sender 从 CAV9 换为 CAV12，并选中 `3_0`。11 帧快速 AP 为 `0.74/0.69/0.30`、8,209,376 bytes，尚未优于主表候选，暂作为下一代机制分支继续调优。
- 已新增 `perception_aware_potential_game`，将 OAPG 的 object-prototype 诊断收束为完整的两层调度机制：coverage layer 先为每个 cluster head 保留一个高质量外部视角，target layer 再用剩余 RB 追逐 object-prototype marginal gain。41 帧 20MHz/10ch/rho3/`B_h=2` 结果为 AP@0.3/0.5/0.7 = `0.81/0.78/0.39`，payload `32,049,872` bytes、`62.54 Mbps`、410 条 scheduled links。对象级 missed rows 从 target-aware PG 的 106 降到 59；该算法当前是最完整的 SGCP 主表候选，下一步补真实 NS3 socket replay/在线短回归和论文正文机制替换。
- 已补齐 PAPG 11 帧真实 NS3 socket replay：110/110 scheduled requests 完成 application callback，RLC complete requests 110/110，RLC TX/RX events 2970/2970，RLC drops 0，PHY decode failures 0，平均/p95 callback delay 为 23.91/24.00 ms。artifact 路径为 `docs/doc_workspace/SGCP/artifacts/papg_ns3_20260717_210304/`；PAPG 主表候选现在具备离线 AP、对象级诊断和 NS3 request-level delivery 三类证据。
- 已将 `C:\Workspace\icdcs-paper\SGCP\main.tex` 同步到 PAPG 主表口径：主行改为 `0.81/0.78/0.39, 62.54 Mbps`，旧 coverage-aware 10ch 保留为 ablation；正文机制改为 coverage layer + target layer 的 perception-aware potential-guided PPS；通信效率段补入 PAPG 110/110 NS3 delivery。当前机器未检测到 LaTeX 编译命令，尚未生成 PDF 验证版。
- 已补 forced-budget random selective baseline：`offline_inference --selective-sharing-baseline random` 复用同一 coalition/late-fusion 路径，强制 3 members/head + 117 grid budget。41 帧结果为 `0.77/0.73/0.38`、31,613,424 bytes、61.68 Mbps，通信量接近 PAPG 62.54 Mbps，但 AP 明显更低；因此旧 RandomRA 继续作为低 payload w/o-PPS 消融，forced-budget random 可进入公平 baseline 表。
- 已同步 `rebuttal_draft.md` 与 `rebuttal_short.md` 到 PAPG 主线：FullPerception 作为 centralized upper reference，forced-budget random 作为公平随机 baseline，PAPG 作为主算法行，并更新 NS3 reliability 为 PAPG 110/110 application/RLC complete。
- 已核实 PACP 原论文是 RGB/BEV 协作感知方法，使用 camera perception、SinBEVT/CoBEVT、BEV-match priority 和 adaptive autoencoder，不是点云原生方法。已新增 `offline_inference --selective-sharing-baseline pacp_lidar`，将 PACP 的 priority-aware / BEV-match 思路迁移为 LiDAR BEV occupancy match + blind-grid complementarity + distance/link-quality cost 的 raw point-grid selective sharing proxy。41 帧高预算 `3 members/head,117 grids/head` 为 `0.81/0.79/0.42`、44,361,424 bytes / `86.56 Mbps`；低预算 `2 members/head,87 grids/head` 为 `0.76/0.73/0.37`、34,498,160 bytes / `67.31 Mbps`。结论：可作为近年 V2V priority-aware proxy baseline，但不能写成 PACP 严格点云复现，也不宜替代 PAPG 主线。
- 已按最新论文口径弃用额外 satisfaction metric：主文只使用 aggregate AP@0.3/AP@0.5/AP@0.7 和 Mbps。Aggregate AP 口径应明确为 pooled evaluator AP：把所有 evaluated receiver-frame samples 的预测框和 GT 框累计后统一计算 AP，而不是 per-CAV AP 简单平均。后续优先固化各实验的 evaluated sample count、receiver policy 和 late-fusion scaffold。
- 已新增 Pure late prediction-box 通信预算工具 `opencda.tools.sgcp_late_box_comm_budget` 与 `late_fusion_box_comm.md`。基于 41 帧 Pure late trace，20MHz/10ch/100ms 下：`80 B/box` broadcast 为 0.739/0.823 Mbps mean/max，scheduled all-to-all unicast 为 14.043/15.638 Mbps、平均 19.10 ms；`128 B/box` broadcast 为 1.132/1.265 Mbps，scheduled all-to-all 为 21.515/24.028 Mbps、平均 27.34 ms。结论是 Pure late 不能靠 payload rate 或调度时延自然限制；只有 unscheduled all-to-all 随机抢信道模型会因冲突失败。主文应将 Pure late 标为 prediction-sharing reference 并计入 detection-box overhead。
- 已核查 early/late checkpoint 口径：`v2xp_cluster_carla.yaml`、`v2xp_cluster_carla_datadump.yaml` 与 `enable_coperception.yaml` 默认 `fusion_method=early`，实际加载 `opencood/logs/pointpillar_early_fusion`；该 checkpoint 为 `point_pillar_early_fusion_low_res`。当前 manifest 中 Pure late 行也是 `fusion_method=early` singleton proxy，而非 `pointpillar_late_fusion` checkpoint。补跑 actual late checkpoint：11 帧 `0.90/0.84/0.46`，41 帧 `0.89/0.83/0.49`；actual-late 预测框 broadcast overhead 约 `1.07-1.65 Mbps`。结论是 Pure late 强势不是由 early checkpoint 误用造成，后续必须重构主表叙事或测试场景。
- 已确认 SGCP 簇间晚期融合已经是 `naive_late_fusion()` box-level NMS；Pure late 也应使用同一个 NMS。补跑 “all late detector” sanity：Pure late actual late 为 `0.89/0.83/0.49`；forced SGCP PAPG `--fusion-method late` 为 `0.87/0.81/0.48`、62.54 Mbps。该 forced row 的第一层已变成 OpenCOOD late inference over scheduled source set，不再是 SGCP raw point-cloud early fusion，只能作为 checkpoint sensitivity。主线公平口径应统一使用 `pointpillar_early_fusion` 做所有 raw point-cloud-to-box 过程，Pure late 作为 prediction-sharing reference 或 early-singleton controlled ablation 单列。
- 已启动 early-fusion checkpoint 提升任务：远程固定使用 `mindspore-187:/data2/gzc/sgcp_early_train/` 和 conda 环境 `opencood-gzc`。当前 checkpoint 已上传到 `/data2/gzc/sgcp_early_train/checkpoints/latest.pth`，并用 checkpoint 配套 config 生成兼容训练配置 `/data2/gzc/sgcp_early_train/configs/pointpillar_early_ckpt_compat_onecav.yaml`。烟测已确认 checkpoint 可在远程 OpenCOOD 中 `strict=False` 加载且为 `0 missing keys, 0 unexpected keys`；当前阻塞是 8 张 3090 均被 `VLLM::Worker` 占用约 22.2GB，导致 cuDNN 无可用卷积算法。已启动后台 watcher `/data2/gzc/sgcp_early_train/runs/start_train_when_gpu_free.sh`，PID `1532887`，日志 `/data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log`；任一 GPU 显存低于 6000 MiB 后会自动用 `opencood-gzc` 训练 200 steps。回收 checkpoint 后，必须用同一 early checkpoint 重跑 SGCP 和 Pure late controlled baseline。
- 已推进 P4 Pareto 源表第二轮整理：`artifacts/pareto_20260719/pareto_source.csv` 新增 5 个已复现点，包括 5ch/20MHz stress `0.56/0.53/0.27, 28.91 Mbps`、coverage rho3 `B_h=2` `0.76/0.72/0.42, 54.56 Mbps`、communication-aware low-budget `0.78/0.75/0.40, 58.97 Mbps`、PAPG `B_h=3` sensitivity `0.80/0.78/0.40, 62.54 Mbps`、PACP-LiDAR low-budget `0.76/0.73/0.37, 67.31 Mbps`。P4 现已覆盖 SGCP rho/channel/B_h/point-cap first pass 与 PACP-LiDAR high/low budget；Random/Density/Link-aware、FullPerception-PCS、EdgeCooper-HD 仍缺系统预算/参数 sweep。
- 已完成 P4 Random/Density/Link-aware first-pass budget sweep：新增 artifact `artifacts/scheduler_budget_sweep_20260719/`，41 帧结果为 Random low-budget `0.75/0.70/0.34, 48.34 Mbps`、Density low-budget `0.78/0.74/0.40, 61.31 Mbps`、Communication-aware high-budget `0.80/0.76/0.42, 75.94 Mbps`。已生成 `scheduler_budget_sweep_manifest.csv` 并补入 Pareto 源表；P4 中 Random/Density/Link-aware 扫描项可按 first-pass 完成处理。当前 P4 仍缺 FullPerception-PCS 参数曲线和 EdgeCooper-HD sender/assignment sweep。
- 已完成 P4 EdgeCooper-HD first-pass budget sweep：新增 artifact `artifacts/edgecooper_budget_sweep_20260719/`，`edgecooper_global_hd` 低预算 `1 member/head,58 grids/head` 为 `0.65/0.61/0.30, 36.10 Mbps`，高预算 `3 members/head,117 grids/head` 复现主点 `0.81/0.78/0.42, 65.40 Mbps`。已生成 `edgecooper_budget_sweep_manifest.csv` 并补入 Pareto 源表；P4 中 EdgeCooperV2V+ / EdgeCooper-inspired 扫描项可按 first-pass 完成处理。当前 P4 剩余主要缺口是 FullPerception-PCS 参数曲线。
- 已完成 P4 FullPerception-PCS first-pass 参数扫描证据包：新增 artifact `artifacts/pcs_parameter_sweep_20260719/pcs_parameter_sweep_manifest.csv`，汇总 11 帧 blind-spot granularity / overlap threshold 趋势与 41 帧 tuned anchor。41 帧 anchor `div12/ov0` 为 `0.59/0.53/0.22, 25.29 Mbps`；11 帧趋势显示 `div12/ov0` 在 AP@0.5/AP@0.7 上最合理，`min_overlap=1` 与 `div16/ov1` 会明显伤 AP。41 帧 `div8/ov0` 和 `div12/ov1` 更激进点均超过 10--15 分钟未完成，记录为 PCS 候选规模/运行时边界；P4 中 FullPerception-PCS 扫描可按 first-pass 完成处理。
- 已完成 detector/checkpoint fairness 收束：新增 `detector_checkpoint_fairness.md`，明确 SGCP 主文 raw-LiDAR 系列统一使用 `pointpillar_early_fusion` checkpoint。Pure late 主表行是 early-checkpoint singleton local detector + `naive_late_fusion()` 的 controlled prediction-sharing reference；actual `pointpillar_late_fusion` checkpoint 的 `0.89/0.83/0.49` 只作为 sensitivity/reference，不进入公平 raw-LiDAR 主表。`Clustered late-only` 不作为核心消融行。
- 已完成 P4 Pareto claim audit：新增 `pareto_claim_audit.md`，按 prediction-sharing、edge/global reference、raw-LiDAR V2V/SGCP-compatible 三类拆分。审计显示 SGCP-PAPG 在 raw-LiDAR V2V 集合中位于 AP@0.3 frontier，并与 `B_h=3` sensitivity 同处 AP@0.5 frontier；但 PAPG 主点不是 AP@0.7 frontier，AP@0.7 应写成 localization/checkpoint headroom 与 high-IoU sensitivity 边界。P4 验收按“AP@0.3/AP@0.5 raw-LiDAR frontier + AP@0.7 boundary” first-pass 关闭。
- 已完成 P2/P6 fusion scaffold claim audit：新增 `fusion_scaffold_claim_audit.md`。Full SGCP 使用 52.7% full raw-sharing payload，保留 full-sharing AP@0.3/AP@0.5 的 95.3%/94.0%，AP@0.7 保留 81.3%；clustered early-only 到 Full SGCP 的提升证明 inter-cluster late fusion 覆盖收益非常强。AP@0.7 只写为 raw point-cloud availability / early checkpoint headroom，不写成当前 SGCP 已全面最优。
- 已完成 Figure 2/3 artifact 口径修正与视觉复核：`plot_breakdowns.py` 已支持 per-row communication label，Pure late 从容易误导的 `raw 0.0` 改为 `box 0.7`，其他 raw-LiDAR 方法保持 `raw X.X`。已重生成 `figure2_protocol_breakdown.png/.pdf` 和 `figure3_fusion_contribution.png/.pdf`；P5/P6 剩余图表验收项可关闭。
- 已完成当前场景充分性审计：新增 `scenario_sufficiency_audit.md`，结论是 41 帧 `v2xp_cluster_carla` 离线场景足以支撑 first-pass 主文图表和附录证据，暂不重新打开 CARLA 导出新场景。新场景触发条件改为：checkpoint 回收后仍无法支撑主张、需要更强动态稳定性/密度实验、或需要正式在线端到端证据。
- 已完成 early checkpoint recovery protocol：新增 `early_checkpoint_recovery.md`，记录 `mindspore-187:/data2/gzc/sgcp_early_train/` watcher、轮询命令、训练日志路径、checkpoint 回收命令和重跑验收标准。当前 blocker 仍是外部 GPU 占用：8 张 GPU 均约 22.2GB used，watcher 正常每 300 秒轮询。
- 已完成 detector checkpoint 快速验证：late checkpoint 直接替换 early detector 为负结果，11 帧 `0.58/0.48/0.15`；attentive intermediate checkpoint 作为 early detector 明显提升 AP@0.3/AP@0.5，41 帧 SGCP-PAPG 为 `0.87/0.81/0.36`，Full20Early attentive 上界为 `0.88/0.85/0.45`；COSDH compatible weight transplant 为 `0.00/0.00/0.00` 或 `0.02/0.00/0.00`，真实 COSDH `point_pillar_comm_multiscale` collapsed smoke test 能跑通但 1 帧 6 个 cluster head 均输出 0 个预测框。阈值诊断显示 COSDH `psm` sigmoid 最大值仅约 `0.0148--0.0224`，降到 `score_threshold=0.01/0.005/0.003` 仍无最终框。当前结论：attentive checkpoint 是有价值的 sensitivity / candidate；COSDH 不是简单调阈值可用的 detector，暂不进入主表。
- 已补齐 attentive checkpoint 的 Pure late controlled 公平对照：41 帧 `0.82/0.65/0.28`，低于 SGCP-PAPG attentive 的 `0.87/0.81/0.36`。这强化了“同 detector 下 SGCP raw point-cloud early fusion 仍带来 AP@0.5/AP@0.7 收益”的叙事。Pure late attentive 的 prediction-box overhead 为 `80 B/box` broadcast `1.37/1.51 Mbps` mean/max、all-to-all `25.97/28.60 Mbps`；`128 B/box` broadcast `2.13/2.35 Mbps`、all-to-all `40.53/44.65 Mbps`。
- 已将 attentive checkpoint sensitivity 写入 `C:\Workspace\icdcs-paper\SGCP\main.tex` 实验段落：明确它用于说明 detector/checkpoint 上界与 SGCP raw point-cloud sharing 贡献，不替换 Table 1 主表。轻量结构检查通过：`table/figure/tabular` begin-end 数均配平；本机仍无 `latexmk/pdflatex`，未生成 PDF。
