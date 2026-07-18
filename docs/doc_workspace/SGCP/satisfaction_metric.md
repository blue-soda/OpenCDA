# SGCP Satisfaction Metric

更新时间：2026-07-19

本文定义 SGCP 后续主表和 coverage 图使用的第一版 network-level satisfaction metric。目标是补充 AP 指标，回答每个 receiver-frame 是否获得了足够的 road-level 感知覆盖。

## 定义

输入来自：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference ... --object-diagnostics-output <objects.csv>
```

`objects.csv` 中每一行对应一个 GT object，并同时记录：

- `full_reference_matched`：full 20-CAV early reference 是否能检测该 GT；
- `method_matched`：当前方法是否检测该 GT；
- `sample_label` / `receiver_id` / `timestamp`：receiver-frame 样本。

对每个 receiver-frame 样本，定义：

```text
full-reference recovery =
  (# GTs matched by both full reference and the method)
  / (# GTs matched by full reference)
```

若该 receiver-frame 中 full reference 无可检出 GT，则该样本不参与 satisfaction 统计。

给定阈值 `tau`，定义：

```text
satisfied(receiver-frame) = 1 if full-reference recovery >= tau else 0
mean satisfaction rate = mean satisfied(receiver-frame)
```

该指标不是 AP 的替代，而是 coverage/recovery 指标：它衡量当前方法能恢复 full-sharing reference 中多少可检测目标，适合描述 road-level network satisfaction。

## 工具

新增工具：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_satisfaction_summary --object-csv LABEL=path\to\objects.csv --sample-output samples.csv --summary-output summary.csv
```

可重复传入多个 `--object-csv`。默认 `tau=0.70`，但当前 41 帧强方法均达到 100% satisfaction，区分度不足；主文建议使用 `tau=0.85` 或同时报告 `mean recovery`。

## First Sanity Results

数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。以下结果使用已有 object diagnostics，不重新运行 detector。

命令：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_satisfaction_summary --satisfaction-threshold 0.85 --object-csv Full=docs\doc_workspace\SGCP\artifacts\object_diag_full_41f.csv --object-csv Spatial10ch=docs\doc_workspace\SGCP\artifacts\object_diag_sgcp_spatial_rho3_10ch_41f.csv --object-csv TargetAware=docs\doc_workspace\SGCP\artifacts\object_diag_target_aware_pg_10ch_rho3_41f.csv --object-csv PAPG=docs\doc_workspace\SGCP\artifacts\object_diag_papg_bh2_rho3_41f.csv --summary-output docs\doc_workspace\SGCP\artifacts\satisfaction_p0_20260719\summary_existing_methods_thr085.csv
```

| Method | Samples | Mean Recovery | P10 Recovery | Satisfaction @0.85 | Payload bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full reference | 41 | 1.000 | 1.000 | 1.000 | 0 |
| Spatial-diverse 10ch | 41 | 0.877 | 0.807 | 0.707 | 29,405,296 |
| Target-aware PG | 41 | 0.885 | 0.825 | 0.756 | 31,069,968 |
| PAPG | 41 | 0.924 | 0.855 | 0.927 | 32,049,872 |

For `tau=0.90`:

| Method | Satisfaction @0.90 |
| --- | ---: |
| Full reference | 1.000 |
| Spatial-diverse 10ch | 0.366 |
| Target-aware PG | 0.415 |
| PAPG | 0.756 |

## Interpretation

PAPG does not merely improve a few aggregate AP points. Under a receiver-frame recovery metric, PAPG has substantially better tail coverage than earlier SGCP variants: its p10 recovery is above the 0.85 satisfaction threshold, while spatial-diverse and target-aware PG fall below it. This supports the paper narrative that PAPG improves road-level coverage stability through coverage + target-aware scheduling.

## Caveats

- The current full reference is full 20-CAV early fusion; it is an upper reference, not a deployable baseline.
- `object_diag_full_41f.csv` has zero payload in this summary because the CSV represents the reference matcher; communication bytes should be filled from the full-sharing accounting table when used in final paper tables.
- Final protocol-native comparison still needs object diagnostics for FullPerception-PCS, EdgeCooperV2V+, pure late fusion, and final SGCP rows under the exact final table settings.
