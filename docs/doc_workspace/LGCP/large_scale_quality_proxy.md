# Large-Scale Quality Proxy

## 背景

审稿意见指出，30 CAV co-simulation 如果只报告 latency，不能支撑 perception quality scalability。短期内若无法在 30 CAV 上完成端到端 OpenCOOD AP 评估，论文必须收窄 claim，或提供经过小规模真实 AP 校准的 scalable quality proxy。

## 建议论文口径

大规模 co-simulation 只直接证明 communication / computation latency scalability。若报告 quality，应明确它是 proxy，而不是真实 AP：

```text
Large-scale experiments evaluate latency and communication scalability. For perception scalability, we report a calibrated area-confidence proxy and validate its correlation with true AP in smaller offline perception experiments.
```

## 当前 Proxy

对每个 frame / method / budget，读取 selected CAV set，并在每个 area 上聚合 selected CAV 的 confidence：

- `area_coverage_proxy`：有至少一个 selected CAV 覆盖的 area 比例。
- `confidence_max_proxy`：每个 area 内 selected CAV confidence 最大值的加权均值。
- `confidence_noisy_or_proxy`：每个 area 内 selected CAV confidence 的 noisy-or 组合加权均值。

权重使用 `1 + gt_count`；大规模无 GT 时可退化为均匀权重或 traffic-density 权重。

## 当前校准结果

在 `lgcp_carla` 11 帧 offline AP 对照中，proxy 与真实 AP 的相关性如下：

| Proxy | Quality | Samples | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| area coverage | AP@0.5 | 10 | 0.863937 | 0.841463 |
| confidence max | AP@0.5 | 10 | 0.951439 | 0.926829 |
| confidence noisy-or | AP@0.5 | 10 | 0.966055 | 0.926829 |
| confidence noisy-or | AP@0.7 | 10 | 0.954195 | 0.802435 |

这说明 area-confidence proxy 可以作为大规模 quality trend 的候选指标，但样本仍来自单 seed / 单场景，不能替代真实 AP。

## 下一步

1. 多 seed 上复核 proxy-AP 相关性。
2. 将 proxy 接入 offline NS3 / large-scale replay 输出。
3. 大规模图中将 proxy 与 latency 分开命名，避免写成 end-to-end AP。
