# SGCP Baseline Fairness Notes

更新时间：2026-07-15

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
| Upper reference | Full 20-CAV early fusion | AP@0.3/0.5/0.7 = 0.85/0.83/0.48 | 否 | 估计无通信约束、全点云共享上界 |
| Upper reference | Full 20-CAV late fusion checkpoint | AP@0.3/0.5/0.7 = 0.91/0.85/0.51 | 否 | 估计全 CAV prediction-level late fusion 上界；checkpoint 不同 |
| Upper reference | FullPerception-RSU | TBD | 否 | 若启用 RSU/虚拟 RSU，可作为集中式 upper reference |
| SGCP main | SGCP constrained + inter-cluster late fusion | AP@0.3/0.5/0.7 = 0.77/0.73/0.35 | 是 | 当前完整 SGCP 离线主口径 |
| Same pipeline ablation | Random scheduler | AP@0.3/0.5/0.7 = 0.44/0.39/0.17 | 是 | 同 SGCP clustering + late-fusion 口径，替换 PPS |
| Same pipeline ablation | MWS scheduler | AP@0.3/0.5/0.7 = 0.31/0.26/0.11 | 暂定 | 需要复核 MWS 定义和效用函数 |
| Same-budget selective baseline | Nearest grid sharing | AP@0.3/0.5/0.7 = 0.76/0.73/0.37 | 是 | 每簇头最多 2 个成员、87 个 grid budget |
| Same-budget selective baseline | Density grid sharing | AP@0.3/0.5/0.7 = 0.77/0.74/0.39 | 是 | 当前强 baseline，payload 高于 SGCP |
| Reference only | Singleton-cluster full late-fusion reference | AP@0.3/0.5/0.7 = 0.82/0.76/0.37 | 否 | late-fuse 全部 20 CAV，当前未计 prediction exchange overhead |

## FullPerception-RSU 口径

建议将 FullPerception-RSU 明确写作 centralized/RSU-assisted upper reference，而不是 SGCP 的公平主 baseline。

定义建议：

- RSU 或虚拟 RSU 可以接收所有 CAV 的点云或检测结果。
- RSU 拥有全局融合与调度视角。
- 通信预算不与 SGCP 的 decentralized V2V PPS 严格相同。
- 结果用于说明“如果存在集中式基础设施和更强通信条件，理论上可达到的参考性能”。

当前 `v2xp_cluster_carla` 没有启用 RSU，离线 dump 也没有 RSU 目录。因此本路线暂不把 FullPerception-RSU 填入主实验表；如需真实复现，应重新导出带 RSU 的场景，或明确使用虚拟 RSU 聚合全 CAV 数据。

## FullPerception-Decentralized 口径

FullPerception-Decentralized 如果作为公平 baseline，不能等同于 full 20-CAV early fusion。更合理的定义是：

- 只使用 CAV-side V2V 信息，不使用 RSU 或 oracle global controller。
- 使用与 SGCP 相同的 CAV 集合、数据帧、OpenCOOD backbone、grid upload payload 统计。
- 每帧使用与 SGCP 相同或显式匹配的通信预算，例如相同 selected-grid 上限、相同 source CAV 数、相同 total upload bytes，或相同子信道/带宽约束。
- 可作为 “same-budget full-perception attempt”，但需要实现具体调度策略，例如 nearest-neighbor sharing、top-k density sharing、communication-aware top-k sharing。

当前已实现 nearest/density selective-sharing first version：它们共享 SGCP cluster/head-wise evaluation path，不使用 PPS，按固定 grid budget 进行 CAV-only V2V sharing。后续仍建议新增 communication-aware selective-sharing baseline，将距离、链路质量或 payload cost 纳入选择。

## 建议进入论文的表述

- 主结论表：SGCP 对比 NC、RS/random、MUG/MWS 或 communication-aware selective sharing，要求同 backbone、同数据集、同 AP 统计口径，并报告通信开销。
- 上界参考表或附表：full early fusion、full late fusion、FullPerception-RSU。明确这些不是公平主 baseline。
- 消融表：w/o stability、w/o coalition、w/o PPS、only early、only late。对 singleton/full late 这类 reference 标注额外通信或不同 checkpoint。

## 后续任务

- 为 FullPerception-RSU 决定实现路线：真实 RSU dump、虚拟 RSU 聚合，或只作为 upper reference 不复现。
- 为 FullPerception-Decentralized 补充 communication-aware selective-sharing baseline。
- 为 singleton full late-fusion reference 估算 prediction-level box exchange overhead。
- 在 `results.md` 的主结果表旁保留 baseline fairness 说明，避免审稿回复中被理解为不公平对比。
