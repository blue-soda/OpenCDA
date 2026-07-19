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
| Table 1 Protocol-Native System Comparison | `artifacts/table1_protocol_20260719/protocol_native_manifest.csv` | `60728bd` | usable with caveats | Pure late 是 prediction-sharing reference；Full20Early 是 upper reference；EdgeCooper-HD 是 edge-assisted reference。 |
| Figure 2 Protocol Breakdown | `artifacts/figures_20260719/figure2_protocol_breakdown.pdf` | `80a590a` | usable | 只报告 aggregate AP，不引入 satisfaction rate。 |
| Figure 3 Fusion Contribution | `artifacts/figures_20260719/figure3_fusion_contribution.pdf` | `80a590a` | usable | 解释 early/late 两层融合分工，不把 AP@0.3 全写成 scheduler 贡献。 |
| Figure 1 AP-Mbps Pareto | `artifacts/pareto_20260719/pareto_source.csv` | `6693b45` | usable with caveats | Pure late prediction sharing 必须和 raw-LiDAR Pareto frontier 分开解释；当前源表已包含 SGCP/PACP、Random/Density/Link-aware 与 EdgeCooper-HD first-pass budget 点。 |
| Pareto Claim Audit | `pareto_claim_audit.md` | `9095b07` | usable | 按 prediction-sharing、edge/global reference、raw-LiDAR V2V 集合拆分 Pareto claim；SGCP-PAPG 只声明 AP@0.3/AP@0.5 raw-LiDAR frontier，不声明 AP@0.7 全面最优。 |
| Table 3 Scheduler Comparison | `artifacts/scheduler_comparison_20260719/scheduler_comparison_manifest.csv` | `2a2e4b2` | usable | 只比较同一 SGCP-compatible scaffold 内的 scheduler。 |
| P4 Scheduler Budget Sweep | `artifacts/scheduler_budget_sweep_20260719/scheduler_budget_sweep_manifest.csv` | `c63e0c2` | usable | 支撑 Pareto 中 Random/Density/Communication-aware low/high budget first-pass，不替代 protocol-native 主表。 |
| P4 EdgeCooper-HD Budget Sweep | `artifacts/edgecooper_budget_sweep_20260719/edgecooper_budget_sweep_manifest.csv` | `6693b45` | usable | 支撑 EdgeCooper-HD edge/global assignment + half-duplex proxy 的 low/high budget 边界。 |
| P4 FullPerception-PCS Parameter Sweep | `artifacts/pcs_parameter_sweep_20260719/pcs_parameter_sweep_manifest.csv` | `0053134` | usable with caveats | 11 帧 granularity/overlap 趋势 + 41 帧 tuned anchor；更激进 41 帧 sweep 运行不可承受，不混入 41 帧 Pareto。 |
| Detector/Checkpoint Fairness Audit | `detector_checkpoint_fairness.md` | `4e6e8e2` | usable | 主表 Pure late 使用 early-checkpoint singleton detector + `naive_late_fusion()`；actual late checkpoint 只作 sensitivity/reference。 |
| Table 4 Parameter Sensitivity | `artifacts/parameter_sensitivity_20260719/table4_parameter_sensitivity.csv` | `859f5d5` | usable | `rho_th` 和 channel count 可进主文；`N_max/T_min` 更适合附录或 rebuttal。 |
| Runtime-Control-NS3 Appendix | `artifacts/appendix_support_20260719/runtime_control_ns3_appendix.md` | `d75abc4` | usable | 支撑 near-real-time control-plane feasibility，不承诺 detector-inclusive 100 ms。 |
| Qualitative Case Study | `artifacts/appendix_support_20260719/qualitative_case_study_bev.pdf` | `73fdee0` | draft usable | 可用于 appendix/rebuttal；正式论文可补 legend 和 prediction-box overlay。 |
| EdgeCooper Writing Reference | `edgecooper_writing_reference.md` | `2efda0a` | usable | 借鉴系统评估结构，不引入 satisfaction rate。 |

## Current Risks

- Early-fusion checkpoint 仍是最大实验风险：远程 fine-tune watcher 已启动，但 GPU 尚未空闲。回收新 checkpoint 后必须生成新一版 Table/Figure artifacts，不覆盖当前版本。
- Pure late 口径已固定为 controlled prediction-sharing reference：early-singleton + `naive_late_fusion()`；actual late checkpoint 已作为 sanity 记录，不混入同一公平 raw-LiDAR baseline。
- EdgeCooper-HD 与 PACP-LiDAR 在 AP@0.7 上强于 PAPG，应按信息条件边界解释为 edge/global 或 stronger-priority reference，不应硬写 SGCP 全面最优。
- `main.tex` 位于 `C:\Workspace\icdcs-paper\SGCP\main.tex`，不在 OpenCDA git 仓库；本机缺少 `latexmk/pdflatex`，当前只做了结构检查，未完成 PDF 编译验证。

## 下一次更新条件

以下任一事件发生时，必须更新本索引：

- early-fusion checkpoint fine-tune 完成并替换 SGCP/Pure late controlled baseline；
- 新导出 CARLA 场景或改变 CAV 数量/带宽主设置；
- Table 1 / Pareto / scheduler comparison 任一数值被替换；
- `main.tex` 图表编号、caption 或主结论发生变化；
- BEV qualitative figure 升级为正式论文图。
