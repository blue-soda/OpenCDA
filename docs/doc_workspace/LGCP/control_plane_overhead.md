# LGCP Control-Plane Overhead

## 目标

回应审稿意见中关于 control-plane overhead 未量化的问题，显式拆分每个 update cycle 的控制面字节量：

- CAV pose / direction / speed report；
- CAV area-confidence report；
- RSU area assignment；
- RSU global-view broadcast；
- planned member-to-leader / leader-to-RSU data upload。

## 当前统计工具

```text
opencda/tools/lgcp_control_overhead_eval.py
```

输入：

- `area_records.csv`
- `area_assignment_plan.csv`
- `upload_plan.csv`
- 可选 `hierarchy_frame_summary.csv`

输出：

- `control_overhead_by_frame.csv`
- `control_overhead_summary.csv`
- `config.yaml`
- `notes.md`

## 当前字节口径

| 项 | 默认值 | 说明 |
| --- | ---: | --- |
| pose / direction / speed report | 32 bytes / CAV / frame | CAV 的位置、方向、速度等基础状态 |
| area-confidence entry | 16 bytes / active CAV-area entry | area id + confidence value 等轻量字段 |
| area assignment entry | 64 bytes / area | RSU 下发 area-task group / leader assignment |
| global-view broadcast | 2000 bytes / frame | RSU 全局结果广播 proxy |

当前默认只统计正整数 CAV id；`agent_id <= 0` 被视为 RSU / reference record，不计入 CAV 上报。Confidence report 只统计 `confidence_field > min_confidence_report` 的 active CAV-area entry，默认阈值为 `0.0`。

## 11 帧 Top-40 结果

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/control_overhead_11f/
```

| Metric | Mean | Max |
| --- | ---: | ---: |
| active CAVs / frame | 20.000000 | 20.000000 |
| confidence entries / frame | 1609.818182 | 1629.000000 |
| assignment entries / frame | 40.000000 | 40.000000 |
| pose report bytes / frame | 640.000000 | 640.000000 |
| confidence report bytes / frame | 25757.090909 | 26064.000000 |
| assignment bytes / frame | 2560.000000 | 2560.000000 |
| global-view bytes / frame | 2000.000000 | 2000.000000 |
| control-plane bytes / frame | 30957.090909 | 31264.000000 |
| planned data bytes / frame | 294545.454545 | 310000.000000 |
| total bytes with control / frame | 325502.545455 | 341264.000000 |
| control-plane ratio | 0.095202 | 0.099794 |

## 论文使用边界

- 这是 control-plane byte proxy，不是完整 PHY airtime / MAC scheduling overhead。
- 当前结果可用于说明 LGCP control traffic 在该 20 CAV / top-40 area setting 下约占 total planned traffic 的 9.52%。
- 若论文需要更强结论，应在多 seed / 多场景下复用同一工具统计均值和方差。
