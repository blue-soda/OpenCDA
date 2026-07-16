# LGCP CAV / Edge Computation Capacity Sensitivity

## 目标

回应审稿意见中 CAV / edge computation capacity 可能影响 local fusion、RSU aggregation 和端到端 latency 的问题。

当前先基于 11 帧 hierarchy plan 做 compute latency proxy，不运行真实模型级 local fusion。

## 工具

```text
opencda/tools/lgcp_compute_capacity_eval.py
```

输入：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/hierarchy_frame_summary.csv
```

Proxy 定义：

- CAV leader local-fusion workload：`leader_max_load`
- RSU aggregation workload：`covered_area_count`
- CAV capacity：leader fusion workload units / ms
- RSU capacity：area aggregation units / ms

## 当前结果

完整矩阵：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_compute_capacity_top40_11f/compute_capacity_summary.csv
```

代表性组合：

| CAV capacity | RSU capacity | Local fusion mean ms | RSU aggregation mean ms | Compute mean ms | Compute max ms | CAV bottleneck ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 4.409091 | 4.000000 | 8.409091 | 9.000000 | 0.909091 |
| 4 | 20 | 2.204545 | 2.000000 | 4.204545 | 4.500000 | 0.909091 |
| 8 | 40 | 1.102273 | 1.000000 | 2.102273 | 2.250000 | 0.909091 |
| 16 | 80 | 0.551136 | 0.500000 | 1.051136 | 1.125000 | 0.909091 |

Cross-bottleneck examples:

| CAV capacity | RSU capacity | Compute mean ms | Dominant bottleneck |
| ---: | ---: | ---: | --- |
| 16 | 10 | 4.551136 | RSU aggregation |
| 2 | 80 | 4.909091 | CAV leader local fusion |

## Interpretation

- Compute proxy scales roughly with both CAV leader capacity and RSU aggregation capacity.
- When RSU capacity is low, RSU aggregation dominates even if CAV leaders are fast.
- When CAV capacity is low, leader local fusion dominates even if RSU aggregation is fast.
- This supports reporting CAV / edge capacity as a sensitivity axis rather than assuming computation is free.

## Paper Boundary

可用于 rebuttal / revision 的保守说法：

```text
We estimate computation sensitivity using the hierarchy plan. Leader local
fusion load is approximated by the maximum assigned leader workload, while RSU
aggregation load is approximated by the number of covered areas. The proxy shows
that both CAV leader capacity and RSU aggregation capacity can become the
bottleneck, motivating a capacity-aware sensitivity analysis.
```

不能直接声称：

```text
The measured model-level fusion runtime is below X ms under all capacities.
```

当前结果是 workload proxy；真实 OpenCOOD local fusion / RSU aggregation runtime 仍需后续模型级实现后测量。
