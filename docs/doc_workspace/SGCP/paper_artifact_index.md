# SGCP Paper Artifact Index

更新时间：2026-07-19

本文档是当前 SGCP 论文实验图表和附录材料的统一索引，用于满足 `target.md` P0 中“进入论文前必须有 artifact”的要求。机器可读版本：

```text
docs\doc_workspace\SGCP\artifacts\paper_artifact_index_20260719\paper_artifact_index.csv
```

## 索引原则

- 每个主文表/图必须能追溯到 source CSV/manifest、绘图脚本或生成命令、原始 stdout/log/trace。
- 每个 artifact 必须有对应的 git commit，或者明确说明不在 OpenCDA git 仓库中。
- 每个图表必须记录 claim boundary，尤其是 Pure late、Full 20-CAV upper reference、EdgeCooper-HD edge-assisted reference 和 SGCP-compatible scheduler comparison。
- 后续新跑场景或替换 checkpoint 后，必须新增新 artifact 目录，不覆盖旧目录。

## 当前主文/附录 Artifact

| Paper Item | Primary Artifact | Commit | Status | Boundary |
| --- | --- | --- | --- | --- |
| Attentive Table 1 Protocol-Native Candidate | `artifacts/attentive_protocol_20260719/protocol_native_attentive_manifest.csv` | `b9ccf50`--`5945dea` | preferred candidate | 后续写作默认入口；SGCP-PAPG attentive `0.87/0.81/0.36` at `62.54 Mbps`，高于 Pure late / EdgeCooperHD attentive；FullPerception-PCS 使用 paper-faithful scheduling + raw-LiDAR full-sender adaptation `0.63/0.49/0.17` at `32.06 Mbps`；Full20Early 仍是 upper reference。 |
| Attentive Figure 1 AP-Mbps Pareto | `artifacts/pareto_attentive_20260719/pareto_attentive_source.csv` | `b9ccf50`--`5945dea` | preferred candidate | 使用 attentive source points；FullPerception-PCS 改为 raw-LiDAR adaptation 点；PACP-LiDAR AP@0.3/AP@0.7 略高但通信明显更贵，SGCP 是更优中等通信点。 |
| Attentive Figure 2/3/4 Breakdowns | `artifacts/figures_attentive_20260719/` | `b9ccf50`--`5945dea` | preferred candidate | Figure 2 protocol、Figure 3 fusion、Figure 4 scheduler 均使用 attentive detector；Pure late 标为 `box 1.37`。 |
| Attentive Table 2 Fusion Scaffold | `artifacts/attentive_fusion_ablation_20260719/fusion_scaffold_attentive_manifest.csv` | `b9ccf50`--`5945dea` | preferred candidate | Clustered early-only `0.51/0.45/0.21` 到 Full SGCP `0.87/0.81/0.36`，支撑两层融合贡献。 |
| Attentive Table 3 Scheduler Comparison | `artifacts/attentive_scheduler_comparison_20260719/scheduler_comparison_attentive_manifest.csv` | `b9ccf50`--`5945dea` | preferred candidate | SGCP-compatible scheduler comparison；PACP-LiDAR 更高 AP@0.3/AP@0.7 但 `86.56 Mbps`，SGCP AP@0.5 最高且 `62.54 Mbps`。 |
| Attentive Paper Number Audit | `artifacts/paper_number_audit_attentive_20260719/paper_number_audit_attentive.csv` | `0cfc70c`--`5945dea` | preferred candidate | 核查当前 `main.tex` Table 1/3 与 attentive manifests 一致，并记录论文 fig 目录已替换为 attentive PDFs。 |
| Table 1 Protocol-Native System Comparison | `artifacts/table1_protocol_20260719/protocol_native_manifest.csv` | `4fee24e` | usable with caveats | Pure late 是 prediction-sharing reference；Full20Early 是 upper reference；EdgeCooper-HD 是 edge-assisted reference；manifest 已显式记录 `10 ch / 20 MHz`。 |
| Figure 2 Protocol Breakdown | `artifacts/figures_20260719/figure2_protocol_breakdown.pdf` | `5776727` | usable | 只报告 aggregate AP，不引入 satisfaction rate；Pure late 图内标为 `box 0.7` 而不是 raw 0。 |
| Figure 3 Fusion Contribution | `artifacts/figures_20260719/figure3_fusion_contribution.pdf` | `5776727` | usable | 解释 early/late 两层融合分工，不把 AP@0.3 全写成 scheduler 贡献；Pure late 图内标为 `box 0.7`。 |
| Fusion Scaffold Claim Audit | `fusion_scaffold_claim_audit.md` | `b504274` | usable | Full SGCP 用 52.7% full-sharing raw payload 保留 95.3%/94.0% AP@0.3/AP@0.5；AP@0.7 写成 localization/checkpoint headroom。 |
| Scenario Sufficiency Audit | `scenario_sufficiency_audit.md` | `9381d7c` | usable | 当前 41 帧场景足以支撑 first-pass 主文图表；新场景触发条件改为 checkpoint/动态性/密度/在线端到端需求。 |
| Early Checkpoint Recovery Protocol | `early_checkpoint_recovery.md` | `3766311` | ready | 记录远程 watcher、GPU blocker、checkpoint 回收命令和重跑验收标准；当前等待 GPU 空闲。 |
| Figure 1 AP-Mbps Pareto | `artifacts/pareto_20260719/pareto_source.csv` | `6693b45` | usable with caveats | Pure late prediction sharing 必须和 raw-LiDAR Pareto frontier 分开解释；当前源表已包含 SGCP/PACP、Random/Density/Link-aware 与 EdgeCooper-HD first-pass budget 点。 |
| Pareto Claim Audit | `pareto_claim_audit.md` | `9095b07` | usable | 按 prediction-sharing、edge/global reference、raw-LiDAR V2V 集合拆分 Pareto claim；SGCP-PAPG 只声明 AP@0.3/AP@0.5 raw-LiDAR frontier，不声明 AP@0.7 全面最优。 |
| Table 3 Scheduler Comparison | `artifacts/scheduler_comparison_20260719/scheduler_comparison_manifest.csv` | `4fee24e` | usable | 只比较同一 SGCP-compatible scaffold 内的 scheduler；manifest 已显式记录 `10 ch / 20 MHz`。 |
| P4 Scheduler Budget Sweep | `artifacts/scheduler_budget_sweep_20260719/scheduler_budget_sweep_manifest.csv` | `c63e0c2` | usable | 支撑 Pareto 中 Random/Density/Communication-aware low/high budget first-pass，不替代 protocol-native 主表。 |
| P4 EdgeCooper-HD Budget Sweep | `artifacts/edgecooper_budget_sweep_20260719/edgecooper_budget_sweep_manifest.csv` | `6693b45` | usable | 支撑 EdgeCooper-HD edge/global assignment + half-duplex proxy 的 low/high budget 边界。 |
| P4 FullPerception-PCS Parameter Sweep | `artifacts/pcs_parameter_sweep_20260719/pcs_parameter_sweep_manifest.csv` | `0053134` | usable with caveats | 11 帧 granularity/overlap 趋势 + 41 帧 tuned anchor；更激进 41 帧 sweep 运行不可承受，不混入 41 帧 Pareto。 |
| Detector/Checkpoint Fairness Audit | `detector_checkpoint_fairness.md` | `4e6e8e2` | usable | 主表 Pure late 使用 early-checkpoint singleton detector + `naive_late_fusion()`；actual late checkpoint 只作 sensitivity/reference。 |
| Detector Checkpoint Sensitivity | `artifacts/checkpoint_sensitivity_20260719/detector_checkpoint_sensitivity_manifest.csv` | `738d148` | usable sensitivity | 汇总 legacy mainline、actual-late、attentive 和 COSDH checkpoint probes；attentive 已升级为当前 forward-writing candidate，actual-late/COSDH 仍只作 sensitivity 或 negative probe。 |
| Attentive Table 4 Parameter Sensitivity | `artifacts/parameter_sensitivity_attentive_20260719/table4_parameter_sensitivity_attentive.csv` | `5945dea` | preferred candidate | attentive forward-writing Table 4；`rho_th=1/2/3` 稳定，5ch stress 明显降低 AP，20ch 额外收益有限。 |
| Legacy Table 4 Parameter Sensitivity | `artifacts/parameter_sensitivity_20260719/table4_parameter_sensitivity.csv` | `9cf102f` | legacy | `pointpillar_early_fusion` checkpoint-reference artifact；不再作为 forward-writing Table 4。 |
| Runtime-Control-NS3 Appendix | `artifacts/appendix_support_20260719/runtime_control_ns3_appendix.md` | `d75abc4` | usable | 支撑 near-real-time control-plane feasibility，不承诺 detector-inclusive 100 ms。 |
| Qualitative Case Study | `artifacts/appendix_support_20260719/qualitative_case_study_bev.pdf` | `73fdee0` | draft usable | 可用于 appendix/rebuttal；正式论文可补 legend 和 prediction-box overlay。 |
| EdgeCooper Writing Reference | `edgecooper_writing_reference.md` | `2efda0a` | usable | 借鉴系统评估结构，不引入 satisfaction rate。 |
| Paper main.tex sync | `C:/Workspace/icdcs-paper/SGCP/main.tex` | outside OpenCDA git; docs `e1a8f15` | static checked not compiled | Paper directory is outside OpenCDA git；Table 1/3、Figure 1/2/3 和正文已同步到 attentive forward-writing candidate；2026-07-19 claim audit 已收紧 RSU/SOTA/100ms 表述并补 SMARTFORM citation。 |
| Paper Freeze Static Check | `paper_freeze_check_20260719.md` | pending | ready for compile | 记录 citation/label/ref/figure/env 静态检查结果、外部 paper source 边界和 PDF 编译前剩余风险。 |
| Paper Number Audit | `artifacts/paper_number_audit_20260719/paper_number_audit.csv` | `4fee24e` | usable | 核查 Table 1/3/4 与 manifest 数值一致；修正 Table 4 channel sweep 的 legacy `40 MHz` 标签为当前复现实验命令口径 `20 MHz`；Table 1/3 manifest 已用工具重生成并显式记录网络元数据。 |

## Current Risks

- Legacy `pointpillar_early_fusion` checkpoint 不再作为后续论文默认主表；当前默认 forward-writing artifacts 为 attentive candidate。旧 Table 1/3/Figure 1/2/3 保留为 checkpoint-reference artifacts。
- 远程 fine-tune watcher 已启动但 GPU 尚未空闲；回收流程见 `early_checkpoint_recovery.md`。若回收到更好 checkpoint，必须生成新一版 Table/Figure artifacts，不覆盖当前 attentive 版本。
- Detector checkpoint sensitivity 已补齐 mainline、actual-late、attentive、COSDH 四类证据；attentive 已补齐 key baselines 和图表，当前可作为 candidate mainline；COSDH 仍为 negative probe。
- Pure late 口径已固定为 controlled prediction-sharing reference：early-singleton + `naive_late_fusion()`；actual late checkpoint 已作为 sanity 记录，不混入同一公平 raw-LiDAR baseline。
- 在 attentive candidate 中，EdgeCooperHD 不再强于 SGCP；PACP-LiDAR AP@0.3/AP@0.7 略高于 SGCP 但通信量显著更高，应按 Pareto tradeoff 写作。
- `main.tex` 位于 `C:\Workspace\icdcs-paper\SGCP\main.tex`，不在 OpenCDA git 仓库；本机缺少 `latexmk/pdflatex`，当前只做了结构检查，未完成 PDF 编译验证。
- 2026-07-19 静态 LaTeX 检查通过：43 个 citation / 29 个 unique citation keys 均在 `Reference.bib` 中存在，32 个 label 无重复，22 个 ref 均可解析，7 个 includegraphics 文件均存在，主要 table/figure/tabular/equation/algorithm 环境配平。
- 2026-07-19 已将 `main.tex` 切换到 attentive forward-writing candidate；该修改无法在 OpenCDA 仓库提交，只能通过本文档和外部 paper 目录状态追踪。
- 2026-07-19 number audit 已确认 Table 1/3/4 数值与 manifest 对齐；Pure late 的 `0.74 Mbps` 是检测框 broadcast overhead，不是 raw-LiDAR manifest Mbps。旧 trace 未记录 `bandwidth_mhz`，后续新 trace/manifest 已补工具字段。

## 下一次更新条件

以下任一事件发生时，必须更新本索引：

- early-fusion checkpoint fine-tune 完成并替换 SGCP/Pure late controlled baseline；
- 新导出 CARLA 场景或改变 CAV 数量/带宽主设置；当前 `scenario_sufficiency_audit.md` 结论是 first-pass 不需要新场景；
- Table 1 / Pareto / scheduler comparison 任一数值被替换；
- `main.tex` 图表编号、caption 或主结论发生变化；
- BEV qualitative figure 升级为正式论文图。
