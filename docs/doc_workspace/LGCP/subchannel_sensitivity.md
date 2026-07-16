# LGCP Subchannel Count Z Sensitivity

## 目标

回应审稿意见中 subchannel count `Z` 可能影响 scheduling 和 latency 的问题。当前先基于 LGCP 11 帧 `upload_plan.csv` 做调度容量 proxy，不重新运行 ns-3。

## 工具

```text
opencda/tools/lgcp_subchannel_sensitivity_eval.py
```

该工具假设每个 slot 最多调度 `Z` 个 request，并将 `member_to_leader` 与 `leader_to_rsu` 视为两个顺序 stage，估算每帧需要的 slot 数和每个 stage 的 subchannel 压力。

## 实验设置

- input：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/upload_plan.csv`
- frames：11
- planned requests：676
- bytes / frame mean：294545.454545
- `Z`：`5 / 10 / 15 / 20`

## 当前结果

| Z | Mean slots / frame | P95 slots / frame | Max slots / frame | Mean max stage packets / subchannel |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 12.727273 | 13.000000 | 13.000000 | 8.000000 |
| 10 | 6.727273 | 7.000000 | 7.000000 | 4.000000 |
| 15 | 5.000000 | 5.000000 | 5.000000 | 2.666667 |
| 20 | 3.727273 | 4.000000 | 4.000000 | 2.000000 |

## Interpretation

- Increasing `Z` from 5 to 20 reduces the mean slot proxy from `12.73` to `3.73` slots/frame.
- The max stage packets per subchannel drops from `8.0` to `2.0`, indicating lower collision / overlap pressure.
- This directly matches the aggregate PHY observation that PSCCH `decoded_overlap` is a major failure reason in the current ns-3 trace.

## Output Directory

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260716_lgcp_carla_subchannel_sensitivity_top40_11f/
```

## Paper Boundary

可用于 rebuttal / revision 的保守说法：

```text
We first evaluate subchannel sensitivity with a scheduling-capacity proxy.
Increasing Z from 5 to 20 reduces the required slot proxy from 12.73 to 3.73
slots per frame and lowers per-subchannel stage pressure. This explains why
limited subchannels can amplify PSCCH overlap in the current ns-3 diagnostics.
```

不能直接声称：

```text
Increasing Z to 20 guarantees proportional PHY delivery improvement.
```

该结论仍需后续多组 ns-3 replay 复核 PHY/RLC/HARQ delivery。
