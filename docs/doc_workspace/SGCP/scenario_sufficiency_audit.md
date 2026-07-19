# SGCP Scenario Sufficiency Audit

更新时间：2026-07-19

## 结论

当前 41 帧 `v2xp_cluster_carla` 离线场景足以支撑 SGCP 论文修订版的 first-pass 主文图表和附录证据；暂不需要立即重新打开 CARLA 导出新场景。

保留新场景触发条件：

- early-fusion checkpoint fine-tune 回收后，现有场景上 SGCP 仍无法支撑 raw-LiDAR V2V AP@0.3/AP@0.5 frontier；
- 审稿意见或最终论文叙事需要展示更强动态性、不同 CAV 密度或更明显的 topology stability 贡献；
- 当前场景无法支撑 AP@0.7 / high-IoU localization 的核心主张，而论文决定把 AP@0.7 提升作为主贡献；
- 需要正式展示 CARLA+NS3 在线端到端结果，而现有 CARLA RPC / spawn / NS3 reupload artifact 不足以覆盖。

## 已被当前场景支撑的图表

| Item | Artifact | 状态 | 场景是否足够 |
| --- | --- | --- | --- |
| Table 1 Protocol-Native System Comparison | `artifacts/table1_protocol_20260719/protocol_native_manifest.csv` | usable with caveats | 是；Pure late / EdgeCooper / Full20Early 边界已写清 |
| Figure 1 AP-Mbps Pareto | `artifacts/pareto_20260719/pareto_source.csv` + `pareto_claim_audit.md` | usable with caveats | 是；raw-LiDAR V2V AP@0.3/AP@0.5 frontier 可写 |
| Figure 2 Protocol Breakdown | `artifacts/figures_20260719/figure2_protocol_breakdown.pdf` | usable | 是；方法差异可见，Pure late box overhead 已标注 |
| Figure 3 Fusion Contribution | `artifacts/figures_20260719/figure3_fusion_contribution.pdf` | usable | 是；late fusion coverage gain 明确 |
| Table 3 Scheduler Comparison | `artifacts/scheduler_comparison_20260719/scheduler_comparison_manifest.csv` | usable | 是；限定为 SGCP-compatible scheduler comparison |
| Table 4 Parameter Sensitivity | `artifacts/parameter_sensitivity_20260719/table4_parameter_sensitivity.csv` | usable | 是；`rho_th` 和 channel count 可写，`N_max/T_min` 边界已降级 |
| Runtime / NS3 appendix | `artifacts/appendix_support_20260719/runtime_control_ns3_appendix.md` | usable | 是；仅支撑 near-real-time / request-level sanity |
| Qualitative case study | `artifacts/appendix_support_20260719/qualitative_case_study_bev.pdf` | draft usable | 基本足够；正式图可继续美化 |

## 当前场景的边界

- Pure late prediction-sharing 很强，不能把 SGCP 写成全面击败所有通信形态；
- AP@0.7 不是 PAPG 主点的 Pareto frontier，必须写成 localization/checkpoint headroom；
- `T_min^stab` 在 41 帧短序列上不敏感，不能作为强动态稳定性主图；
- 当前 CAV 数量/密度 sweep 是固定场景子集，不等同于真实重新采样交通密度；
- CARLA 在线回归已经有若干 artifact，但当前主文仍主要依赖离线 + NS3 request-level replay。

## 决策

本阶段不新导出 CARLA 数据集。优先等待 remote early-fusion checkpoint fine-tune，回收后在同一 41 帧场景重跑 SGCP/Pure late controlled baseline 和关键图表点；只有 checkpoint 回收后仍无法支撑主张，才按 `docs/doc_workspace/environment.md` 中的 CARLA 启动和数据导出流程重新采集场景。
