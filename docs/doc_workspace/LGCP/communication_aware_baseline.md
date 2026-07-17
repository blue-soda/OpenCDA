# Communication-Aware Baseline

## 目标

回应审稿人关于 baseline fairness 的问题：LGCP 不能只对比 full sharing 或 random partial sharing，必须包含不使用 LGCP hierarchy 的强 selective-sharing baseline。

## 当前已落地 Baseline

### Confidence top-k

不使用 area partition、leader fusion 或 RSU aggregation。每帧对每个 CAV 聚合 area confidence，选择 top-k CAV 与 ego 融合。

用途：

- 作为 selective sharing without hierarchy 的基础强 baseline。
- 验证收益是否只是来自减少共享车辆数。

局限：

- 只按感知 confidence 排序，没有考虑通信距离或链路代价。

### Communication-aware top-k

不使用 LGCP hierarchy。每帧先聚合每个 CAV 的 area confidence，再除以到 ego 的距离成本：

```text
utility(v) = confidence(v) / (1 + distance(v, ego) / 100)
```

选择 utility 最高的 top-k CAV。当前距离成本是轻量 proxy，用于 offline ablation；真实论文结果仍需结合 NS3 链路质量、packet size、SINR、delivery ratio 或 latency。

用途：

- 作为 communication-aware selective sharing without hierarchy 的更强 baseline。
- 检查 LGCP 的 area-aware selection 是否真的优于简单通信感知策略。

## 当前 11 帧结果含义

在 `lgcp_carla` 11 帧 offline perception-only ablation 中，`comm_aware_topk` 在 budget=5/10 下均强于 `confidence_topk`，并略高于 `area_aware_union`。

这说明：

- 论文必须保留强 communication-aware baseline，不能只报告 random / confidence top-k。
- 当前 offline proxy 不能证明 LGCP area-aware selection 已优于所有强 selective-sharing baseline。
- LGCP 的核心主张应继续通过完整 hierarchy、leader local fusion、RSU aggregation 和 scheduling latency 来支撑，而不是只依赖当前 area-aware union subset AP。

## Multiseed Random Baseline

2026-07-17 已补跑 random-only seeds `11 / 23 / 37`，并与原 seed `7` 汇总：

| Method | Budget | AP@0.7 mean | AP@0.7 std |
| --- | ---: | ---: | ---: |
| random | 5 | 0.163843 | 0.026394 |
| random | 10 | 0.328993 | 0.038178 |
| comm_aware_topk | 5 | 0.296352 | 0.000000 |
| comm_aware_topk | 10 | 0.545736 | 0.000000 |

结论保持不变：随机 partial sharing 不是足够强的 baseline；communication-aware top-k 是当前必须保留的强 baseline。

## 下一步

1. 用 NS3 或离线链路模型替换距离 proxy。
2. 在完整 LGCP 管线完成后，对比：
   - communication-aware top-k without hierarchy；
   - LGCP without scheduling；
   - full LGCP。
