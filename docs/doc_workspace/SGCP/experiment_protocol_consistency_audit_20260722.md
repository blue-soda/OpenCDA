# SGCP Experiment Protocol Consistency Audit - 2026-07-22

对应外部实验包：

`C:\Workspace\2026-7-papers\infocom\SGCP\experiment`

当前论文前向通信协议已收束为：

- attentive checkpoint
- `40 MHz`
- `10` target subchannels
- `100 ms` perception cycle
- `60 ms` communication deadline
- NS3-calibrated estimator: `tb_size=899 bytes`, `slot=0.5 ms`,
  `subchannel_prbs=10`, `MCS=28`, `PSSCH symbols=12`

## 本轮已修正

- 删除外部实验包中误导性的粗糙审计 CSV：
  `data/experiment_consistency_audit_20260722.csv`。
- 新增外部审计入口：
  `protocol_consistency_audit_20260722.md`。
- 修正 `table2_fusion_scaffold_attentive.csv` 中
  `PureLate_attentive` 的协议字段，使其与 Table A/Table 5/Table 6 一致：
  `prediction_nms + singleton + local_detection + box`。
- 修正 `table5_clustering_ablation_attentive_20260720.csv` 中
  Dynamic SGCP coalition 的通信量：
  `62.536336 raw + 0.741276 box = 63.277612 Mbps`。
- 重新生成外部实验包 figures 和 `MANIFEST.csv`。
- 将 table/figure registry 中 Table A、Table 2/3/4/4b/5/6、
  Bootstrap、Figure 1-8 的状态降级为 legacy 20MHz scaffold 或需重跑。

## 当前判定

除 Table1 的 40MHz addendum、SGCP low-budget addendum、EdgeCooper singleton
budget probe 和 NS3 frame-level feasibility artifacts 外，外部实验包中的
20260720 表格和图大多仍是 `20 MHz / 10 ch / 100 ms` attentive scaffold 数据。
这些结果可以用于机制诊断和写作结构参考，但不能写成当前
`40 MHz / 10 target subchannels / 60 ms` 正式协议下的最终数值。

## 需重跑项目

优先级从高到低：

1. Table 3 / Figure 4：SGCP-compatible scheduler comparison。2026-07-22
   已新增 current-protocol diagnostic rerun：
   `data/table3_scheduler_comparison_current_protocol_20260722.csv` 和
   `figures/figure4_scheduler_comparison_current_protocol_20260722.*`。
   该 rerun 使用当前 40MHz/10ch/60ms 协议，但 PAPG 默认结果为
   `0.64/0.60/0.25, 37.05 Mbps`，不适合作为 paper-facing final Table3；
   后续需算法/参数或叙事决策。
2. Table 2 / Figure 3：fusion scaffold ablation。
3. Table 5 / Figure 7：clustering ablation。
4. Table 4 / Table 4b / Figure 5：parameter 与 Nmax sensitivity。
5. Table 6 / Figure 8：global box aggregation normalized baselines。
6. Table A / Figures 1-2：combined scaffold/Pareto summaries。
7. Figure 6：bootstrap uncertainty，待最终行冻结后重算。

## 通过的静态一致性检查

当前外部结果 CSV 中，所有同时含有
`raw_lidar_mbps`、`box_mbps`、`total_mbps`、`mbps` 的表均满足：

```text
mbps = total_mbps = raw_lidar_mbps + box_mbps
```

所有 `prediction_nms` / `inter_cluster_nms` / `global_box_nms` 行都有非零
box communication accounting 或明确的 identity/no-box 解释；不再存在
`missing_trace` 造成的 late-fusion 通信漏算。
