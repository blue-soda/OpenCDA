# SGCP Baseline Fairness Notes

更新时间：2026-07-18

本文档用于澄清 SGCP 实验中各类 baseline 的公平性口径，避免把集中式 upper reference、全量共享 reference 和同通信约束 baseline 混在同一主结论中。

## 公平性原则

SGCP 的核心设定是去中心化 CAV 协同感知：车辆先形成 coalition/cluster，簇内执行受 PPS 约束的 early fusion，簇间通过 cluster head 的检测结果做 late fusion。因此 baseline 应明确四个维度：

- 信息来源：只使用 CAV V2V，还是允许 RSU/全局观察者。
- 通信预算：是否与 SGCP 使用相同子信道、带宽、grid upload 和 cluster-head exchange 口径。
- 融合后端：是否使用同一 OpenCOOD checkpoint/backbone。
- 调度控制：是否使用 centralized oracle、RSU controller，还是 fully decentralized CAV-only control。

## Baseline 分层

| 层级 | 方法 | 当前结果 | 是否公平主对比 | 用途 |
| --- | --- | --- | --- | --- |
| Upper reference | Full 20-CAV early fusion | AP@0.3/0.5/0.7 = 0.85/0.83/0.48 | 否 | 估计无通信约束、全点云共享上界；non-ego upload payload 60,838,528 bytes |
| Upper reference | Full 20-CAV late fusion checkpoint | AP@0.3/0.5/0.7 = 0.91/0.85/0.51 | 否 | 估计全 CAV prediction-level late fusion 上界；checkpoint 不同 |
| Built-in FullPerception | PCS (`pcs.py`) tuned | AP@0.3/0.5/0.7 = 0.59/0.53/0.22 | 否 | 对应 FullPerception 论文 PCS 调度；payload 12,959,840 bytes / 25.29 Mbps；正式 baseline 但不是强 V2V 主对比 |
| RSU/edge-assisted | Global selective proxy | AP@0.3/0.5/0.7 = 0.84/0.80/0.46 | 否 | 当前 dump 无真实 RSU sensor；虚拟/global candidate pool，payload 56,224,736 bytes / 109.71 Mbps |
| V2V-only baseline | Cluster-local selective proxy | AP@0.3/0.5/0.7 = 0.80/0.76/0.41 | 是 | cluster-local CAV candidates，payload 38,920,592 bytes / 75.94 Mbps |
| SGCP main | SGCP PAPG 10ch | AP@0.3/0.5/0.7 = 0.81/0.78/0.39 | 是 | 当前主方法，payload 32,049,872 bytes / 62.54 Mbps，NS3 110/110 complete |
| SGCP ablation | SGCP potential_game | AP@0.3/0.5/0.7 = 0.77/0.73/0.35 | 是 | 原始 PPS 消融 |
| Same pipeline ablation | Random scheduler | AP@0.3/0.5/0.7 = 0.44/0.39/0.17 | 否 | payload 过低，作为 w/o-PPS 诊断，不用于证明通信节省 |
| Same pipeline ablation | MWS scheduler | AP@0.3/0.5/0.7 = 0.31/0.26/0.11 | 否 | payload 过低，作为 w/o-PPS 诊断 |
| Same-budget selective baseline | Nearest grid sharing | AP@0.3/0.5/0.7 = 0.76/0.73/0.37 | 是 | 每簇头最多 2 个成员、87 个 grid budget |
| Same-budget selective baseline | Density grid sharing | AP@0.3/0.5/0.7 = 0.77/0.74/0.39 | 是 | 当前强 baseline，payload 高于 SGCP |
| Same-budget selective baseline | Communication-aware density sharing | AP@0.3/0.5/0.7 = 0.78/0.75/0.40 | 是 | 当前最强 baseline，density score 加入距离成本 |
| Same-budget selective baseline | Forced-budget random sharing | AP@0.3/0.5/0.7 = 0.77/0.73/0.38 | 是 | 3 members/head, 117 grid budget, payload 31,613,424 bytes / 61.68 Mbps |
| High-budget selective baseline | Density / communication-aware sharing | AP@0.3/0.5/0.7 = 0.80/0.76/0.40 | 是 | 3 members/head, 117 grid budget, payload 37,710,864 bytes |
| SGCP ablation | Coverage-aware spatial-diverse, 10ch/rho3 | AP@0.3/0.5/0.7 = 0.79/0.76/0.38 | 是 | PAPG 前身/消融，payload 29,405,296 bytes / 57.38 Mbps，NS3 110/110 complete |
| SGCP sensitivity | Coverage-aware spatial-diverse, 20ch | AP@0.3/0.5/0.7 = 0.80/0.76/0.41 | 是 | 高预算资源敏感性，payload 37,912,544 bytes / 73.98 Mbps，NS3 154/154 complete |
| Reference only | Singleton-cluster full late-fusion reference | AP@0.3/0.5/0.7 = 0.82/0.76/0.37 | 否 | late-fuse 全部 20 CAV，当前未计 prediction exchange overhead |

## FullPerception / RSU 口径

建议将真实 RSU-sensor reference 明确写作 centralized/RSU-assisted upper reference，而不是 SGCP 的公平主 baseline。

定义建议：

- RSU 或虚拟 RSU 可以接收所有 CAV 的点云或检测结果。
- RSU 拥有全局融合与调度视角。
- 通信预算不与 SGCP 的 decentralized V2V PPS 严格相同。
- 结果用于说明“如果存在集中式基础设施和更强通信条件，理论上可达到的参考性能”。

当前 `v2xp_cluster_carla` 没有启用真实 RSU，离线 dump 也没有 RSU 目录。因此真实 RSU sensor 版 reference 仍需重新导出带 RSU 的场景。不过仓库内已有 FullPerception 论文 PCS 调度代码：`opencda/core/clustering/algorithms/resource_allocation/pcs.py`，正规入口为 `--resource-allocation fullperception_pcs`。41 帧当前 tuned 结果为 AP@0.3/0.5/0.7 = 0.59/0.53/0.22，payload = 12,959,840 bytes / 25.29 Mbps；该结果可作为正式 FullPerception PCS baseline。

当前已实现的 `global_selective_proxy` 是另一个虚拟/global scheduler proxy：虚拟调度器拥有全局 CAV 候选池，但每个 receiver 仍受 3 members/head 和 117 grid budget 限制；重命名前 41 帧结果为 AP@0.3/0.5/0.7 = 0.84/0.80/0.46，payload = 56,224,736 bytes / 109.71 Mbps。它不应与 V2V-only SGCP 放入同一公平 ranking，也不应命名为 FullPerception。full 20-CAV early fusion仍是更高一级 full-sharing upper reference：AP@0.3/0.5/0.7 = 0.85/0.83/0.48，non-ego CAV upload payload = 60,838,528 bytes。

## Cluster-Local Selective Proxy 口径

Cluster-local selective proxy 如果作为公平 baseline，不能等同于 full 20-CAV early fusion。更合理的定义是：

- 只使用 CAV-side V2V 信息，不使用 RSU 或 oracle global controller。
- 使用与 SGCP 相同的 CAV 集合、数据帧、OpenCOOD backbone、grid upload payload 统计。
- 每帧使用与 SGCP 相同或显式匹配的通信预算，例如相同 selected-grid 上限、相同 source CAV 数、相同 total upload bytes，或相同子信道/带宽约束。
- 可作为 “same-budget full-perception attempt”，但需要实现具体调度策略，例如 nearest-neighbor sharing、top-k density sharing、communication-aware top-k sharing。

当前已实现 nearest/density/communication-aware selective-sharing first version，并保留 `cluster_local_selective_proxy` 作为强 V2V-only proxy：它们共享 SGCP cluster/head-wise evaluation path，不使用 RSU/全局 oracle，按固定 grid budget 进行 CAV-only V2V sharing。其中 communication-aware 默认使用 `density_sum / (1 + distance / 100)` 作为离线 proxy；若传入 `--ns3-link-quality-csv <rlc_by_request.csv>`，则扩展为 `density_sum * rlc_complete_ratio / (1 + distance / 100)`，用于体现 NS3 request-level 完整交付约束。`cluster_local_selective_proxy` 在 3 members/head、117 grid budget 下得到 AP@0.3/0.5/0.7 = 0.80/0.76/0.41，payload = 38,920,592 bytes / 75.94 Mbps，可作为强 V2V-only proxy baseline。

注意：旧 RandomRA/MWS scheduler payload 只有约 9.7/9.9 MB，远低于 SGCP 10ch/20ch 和 high-budget selective baseline，说明它们没有充分利用通信资源。它们应保留为 w/o PPS 消融，不应作为“SGCP 降低通信量”的主公平 baseline。公平主表应优先使用 forced-budget random、payload-matched selective sharing、communication-aware selective sharing 和 SGCP PAPG。

## 建议进入论文的表述

- 主结论表：SGCP PAPG 对比 forced-budget random、density/communication-aware selective sharing、cluster-local selective proxy 和 FullPerception PCS，要求同 backbone、同数据集、同 AP 统计口径，并报告通信开销。
- 上界参考表或附表：full early fusion、full late fusion、true RSU-sensor reference（若未来导出）。明确这些不是公平主 baseline。
- 消融表：w/o stability、w/o coalition、w/o PPS、only early、only late。对 singleton/full late 这类 reference 标注额外通信或不同 checkpoint。

## 后续任务

- 若需要真实 RSU-sensor reference，重新导出带 RSU sensor 的 CARLA dump；当前 `global_selective_proxy` 只能作为 RSU/edge-assisted proxy。
- `cluster_local_selective_proxy` 的 11-frame true NS3 replay 已完成，request-level delivery 与 PAPG/forced random 对称。
- 已将 communication-aware selective-sharing baseline 扩展为可选 NS3 RLC-complete cost；PAPG 主方法和 forced-budget random baseline 均已完成 11 帧真实 NS3 replay，二者 scheduled requests 均为 110/110 application/RLC complete、0 PHY failures。
- 为 singleton full late-fusion reference 估算 prediction-level box exchange overhead。
- 在 `results.md` 的主结果表旁保留 baseline fairness 说明，避免审稿回复中被理解为不公平对比。
