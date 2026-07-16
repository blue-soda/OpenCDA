# LGCP Deployment Assumptions and Limitations

## 目的

本文档为论文新增 `Deployment Assumptions and Limitations` 小节准备素材，重点回应审稿意见中的：

- RSU centralization 是否过强；
- 车辆移动、定位误差和 stale information 是否会破坏 assignment；
- leader / member failure 如何处理；
- 多 RSU 场景如何扩展；
- 大规模实验若只报告 latency，论文 claim 应如何收窄。

## Core Deployment Assumptions

| Assumption | Paper wording | Risk | Mitigation |
| --- | --- | --- | --- |
| RSU-assisted road segment | LGCP targets infrastructure-assisted intersections / roundabouts / urban road segments where at least one RSU has regional coordination responsibility. | 纯 V2V 场景不适用。 | 将 LGCP 定位为 RSU-assisted CP，而不是 universal decentralized CP。 |
| Periodic CAV state report | CAVs periodically report pose, heading, speed, and compact area-confidence summaries. | 控制面开销和过期状态。 | 已统计 control-plane overhead；通过 update interval 和 stale threshold 控制。 |
| Shared spatial reference | CAVs and RSU share map frame / calibrated coordinate transform. | 定位误差会影响 area assignment。 | 增加 localization error sensitivity；使用 enlarged area margin / confidence decay。 |
| Bounded mobility during one cycle | CAV displacement within one update cycle is bounded. | 高速车辆会导致 assignment stale。 | assignment TTL、motion prediction、reassignment trigger。 |
| RSU compute availability | RSU can run assignment and aggregate leader results within cycle budget. | RSU central bottleneck。 | computation capacity sensitivity；multi-RSU partition。 |

## RSU Centralization Boundary

LGCP 的 RSU 不是替代所有 CAV 感知和融合，而是负责三类轻量控制 / 聚合任务：

1. 维护 spatial area partition；
2. 基于 CAV area confidence 构建 overlapping area-task groups；
3. 聚合 leader 上传的 local fused area result，生成 global view。

论文推荐保守表述：

```text
LGCP assumes RSU-assisted road segments, such as intersections and roundabouts,
where infrastructure can coordinate spatial tasks. The RSU does not require
raw sensor streams from all CAVs; it mainly collects compact confidence reports,
assigns area-task groups, and aggregates leader-level fused results.
```

该表述可以避免被理解为“所有计算都集中到 RSU”。

## Mobility and Stale Assignment

### Staleness Sources

- CAV 在 assignment 下发后移动；
- CAV heading / speed 改变，导致可观测 area 变化；
- confidence report 与实际传输之间存在 network delay；
- member-to-leader 或 leader-to-RSU 失败导致 RSU global view 落后。

### Recommended Mechanisms

| Mechanism | Trigger | Effect |
| --- | --- | --- |
| Assignment TTL | 每个 assignment 只在 `K` 个 frame 内有效 | 防止长期 stale group |
| Motion-aware confidence decay | CAV displacement 或 heading change 超阈值 | 降低旧 confidence 的权重 |
| Enlarged area margin | 构建 area slice 时加入边界 buffer | 缓解小定位误差 |
| Event-driven reassignment | CAV missing heartbeat / large pose jump | 下一周期重选 group / leader |
| Stale fallback | 当前 frame 缺少 fused area result | 使用上一帧 area result 或 best single-CAV observation |

论文可将这些机制作为 deployment mitigation，并把实际 sensitivity 放入 future work 或附录实验计划。

## Localization Error

定位误差主要影响两处：

1. CAV-to-area confidence assignment；
2. area-specific feature slice cropping / alignment。

建议论文中承认该风险，并说明两种缓解：

- **robust area margin**：area crop 增加空间 margin，牺牲少量通信量换取对齐鲁棒性；
- **confidence smoothing**：对相邻 area 的 confidence 做轻量平滑，避免边界处误差导致 group 震荡。

后续 P1 实验可按 `0m / 0.2m / 0.5m / 1.0m` Gaussian noise 注入 CAV pose，评估 area-confidence correlation、subset ablation AP 和 NS3 delivery / latency 是否稳定。

## Failure Modes

| Failure mode | Observable signal | Immediate fallback | Longer-term mitigation |
| --- | --- | --- | --- |
| CAV misses confidence report | no report in current cycle | exclude CAV from current assignment | use last valid report with decay for one cycle |
| member-to-leader packet loss | leader missing area slice before timeout | leader fuses available members only | retry only for high-priority area |
| leader CAV failure | no heartbeat or no leader result | RSU uses best member single-CAV result | reassign leader next cycle |
| leader-to-RSU packet loss | RSU missing fused `(frame, area)` | stale area result or low-confidence placeholder | prioritize leader result retransmission |
| RSU overload | assignment / aggregation exceeds cycle budget | reduce top areas or max group size | multi-RSU partition / edge offload |
| RSU outage | no assignment broadcast | CAVs fall back to local perception or last assignment for one TTL | handover to neighboring RSU |

## Multi-RSU Scaling

LGCP 可以自然扩展为多 RSU road segments，但论文不应暗示当前实验已经验证多 RSU。

推荐扩展口径：

1. 每个 RSU 管理一个 spatial partition；
2. RSU 边界区域允许 overlapping ownership；
3. CAV 根据位置和信号质量选择 serving RSU；
4. 相邻 RSU 只交换 boundary area 的 fused result 或 compact global-view summary；
5. 多 RSU handover 以 assignment TTL 为边界，避免同一 frame 内频繁切换。

论文推荐表述：

```text
This paper evaluates a single-RSU road segment, which is the basic deployment
unit of LGCP. Multi-RSU deployment can be built by assigning each RSU a spatial
partition and exchanging only boundary-area fused results. We leave full
multi-RSU scheduling and handover optimization to future work.
```

## Large-Scale Claim Boundary

当前大规模方向应收窄为：

- communication / computation scalability；
- latency and delivery diagnostics；
- calibrated perception-quality proxy。

避免写成：

```text
LGCP improves end-to-end perception AP in 30-CAV large-scale settings.
```

除非后续真实跑通 30 CAV model-level AP。推荐写法：

```text
For large-scale scenarios, we focus on communication and computation
scalability because full end-to-end perception evaluation with 30 CAVs is
computationally expensive. We therefore report latency, delivery diagnostics,
and a calibrated quality proxy, while using smaller-scale settings for
model-level AP validation.
```

## Paper Placement

- Main text Discussion：RSU-assisted assumption、single-RSU boundary、failure handling。
- Evaluation Limitations：large-scale latency-only / proxy-only claim boundary。
- Appendix：localization error、stale assignment、multi-RSU expansion details。

## Follow-Up Experiments

建议与 `target.md` 的 P1 项对应：

1. localization error sensitivity；
2. vehicle speed / update frequency / stale assignment sensitivity；
3. CAV / RSU computation capacity sensitivity；
4. multi-seed control-plane overhead and NS3 delivery diagnostics；
5. leader failure or leader-to-RSU packet loss stress test。
