# SGCP Aggregate AP Protocol

更新时间：2026-07-19

本文档固定 SGCP 后续主表、消融和 Pareto 图的效果指标口径。用户已明确不额外引入 satisfaction rate，主文只使用 aggregate AP 和通信量。

## 指标定义

Aggregate AP 指 OpenCOOD evaluator 在一次实验中把所有 evaluated receiver-frame samples 的预测框与 GT 框累计后统一计算 AP@0.3 / AP@0.5 / AP@0.7。

这不是 per-CAV AP 的简单平均，也不是单 ego AP。对于 SGCP inter-cluster late fusion 设置，当前 41 帧实验通常是每个 timestamp 产生一次 late-fused network prediction，因此 `evaluated_samples=41`；trace 中仍会记录多个 cluster-head receiver rows，用于说明每帧的分簇、调度和融合来源。

## 每个结果必须记录

- `evaluated_samples`：实际进入 evaluator 的 sample 数。
- `receiver_policy`：例如 `all-cluster-heads`、`ego` 或其他 policy。
- `inter_cluster_late_fusion`：是否把多个 cluster head 的检测结果做簇间 late fusion。
- `fusion_method`：OpenCOOD 输出的 fusion method，例如 `early`。
- `resource_allocation` / `clustering` / `upload_mode`：调度、分簇和上传语义。
- `payload_bytes` / `Mbps`：论文主通信量口径。当前离线实验默认 10 Hz；通信时长按仿真帧数计算，优先使用 trace 中的 `unique_timestamps`，若无 trace 再退回 `evaluated_samples`。因此 Mbps = payload bytes * 8 / (`unique_timestamps` * 0.1 s) / 1e6。对于 no-late 的多 receiver-sample 消融，AP 可由 246 个 receiver samples 计算，但通信时长仍是 41 个 timestamp。
- 原始 artifact：stdout log、trace CSV、summary/manifest CSV、代码 commit。

## Manifest 工具

入口：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest `
  --run "PAPG=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1_trace.csv" `
  --output-csv docs\doc_workspace\SGCP\artifacts\aggregate_ap_manifest_20260719\repeat_check_manifest.csv
```

`--run` 可重复传入多个 `label=log_path,trace_path`。PowerShell 下应把整个 run spec 加引号，避免路径中的逗号被提前拆开。

该工具不重新计算 AP，只解析 `offline_inference` stdout 中的 OpenCOOD AP 输出，并与 trace CSV 中的 receiver policy、resource allocation、payload 等字段合并，形成可进入论文表格源数据的 manifest。

## 当前 smoke 结果

基于 `repeat_check_20260718` 的 PAPG 与 EdgeCooper-HD 41 帧日志，已生成：

`docs\doc_workspace\SGCP\artifacts\aggregate_ap_manifest_20260719\repeat_check_manifest.csv`

| Method | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Evaluated Samples | Trace Rows | Late Fusion | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| PAPG | 0.81 | 0.78 | 0.39 | 41 | 246 | yes | 32,049,872 | 62.54 |
| EdgeCooper-HD | 0.81 | 0.78 | 0.42 | 41 | 246 | yes | 33,519,040 | 65.40 |

下一步：用同一 manifest 口径补齐 Table 1、Table 2、Table 3 和 Pareto 曲线源数据。
