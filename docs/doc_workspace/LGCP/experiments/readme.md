# LGCP Experiment Output Convention

本目录用于保存 LGCP 文档工作区内的实验输出摘要、配置快照和结果索引。大型原始数据仍保存在 `../environment.md` 指定的数据目录，例如 `D:\Data\Carla`；本目录只保存轻量级、可进入论文修订流程的记录。

## 目录结构

```text
experiments/
  area_confidence/
    <run_id>/
      config.yaml
      area_records.csv
      correlation_summary.csv
      calibration_bins.csv
      notes.md
  eq2_composition/
    <run_id>/
      config.yaml
      subset_records.csv
      rule_correlation_summary.csv
      marginal_gain_summary.csv
      notes.md
  greedy_optimality_gap/
    <run_id>/
      config.yaml
      instance_records.csv
      gap_summary.csv
      runtime_summary.csv
      notes.md
  ablation/
    <run_id>/
      config.yaml
      ablation_summary.csv
      notes.md
  baselines/
    <run_id>/
      config.yaml
      baseline_summary.csv
      notes.md
```

## Run ID

`run_id` 使用：

```text
YYYYMMDD_<scenario>_<purpose>
```

示例：

```text
20260715_lgcp_carla_smoke
20260716_lgcp_carla_area_confidence_seed1
20260717_opv2v_eq2_cobevt
```

## 必备文件

每个 run 至少包含：

| 文件 | 含义 |
| --- | --- |
| `config.yaml` | 实验配置快照，记录数据源、模型、场景、参数和命令 |
| `notes.md` | 人类可读记录：目的、命令、观察、异常和结论 |
| `*_summary.csv` | 可直接汇总到 `results.md` 的结果表 |

原始 `.pcd`、`.png`、模型 checkpoint、CARLA dump 不放进本目录，只在 `config.yaml` 或 `notes.md` 中引用绝对路径。

## 状态流转

1. 新实验先在 `target.md` 挂任务。
2. 开跑前建立 `experiments/<type>/<run_id>/config.yaml` 和 `notes.md`。
3. 执行过程追加到 `log.md`。
4. 结果可复核后写入对应 `*_summary.csv`。
5. 确认可进论文的关键数值再同步到 `results.md`。

## Config Template

```yaml
run_id:
date:
task:
scenario:
dataset_root:
scenario_id:
command:
model:
fusion_method:
frames:
agents:
parameters:
  delta_g:
  grid_size:
  area_size:
outputs:
  summary:
  records:
notes:
```

## Notes Template

```markdown
# <run_id>

## Goal

## Command

## Inputs

## Outputs

## Observations

## Result Summary

## Issues

## Next Step
```
