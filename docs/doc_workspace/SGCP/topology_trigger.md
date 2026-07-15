# SGCP Topology Change Trigger

更新时间：2026-07-15

本文档定义 SGCP 中 “topology change triggers reconfiguration” 的可执行机制口径，用于补齐论文机制说明，并作为后续在线/离线代码接入依据。

## 当前实现观察

- `CoalitionGame` 已包含 `T_min_stab` 和 `stability_cost()`，可表达稳定窗口内频繁切换的惩罚。
- `ClusteringV2XManager.run_algorithm()` 当前按 `cluster_interval` 周期调用 clustering 算法，没有显式判断 “是否发生 topology change”。
- 因此当前工程更接近周期性重算，而论文表述更接近事件触发重构。论文需要明确：每个周期都接收状态/信标，但不一定每个周期都重构 cluster。

## 触发条件

建议把 topology change trigger 定义为以下任一条件成立：

| Trigger | 判定信号 | 说明 |
| --- | --- | --- |
| Neighbor-set change | 新 CAV 进入通信范围、已有邻居离开范围、当前 head/member 不再可达 | 基于 beacon 中的位置和 `communication_range` 判断 |
| Relative-motion risk | 相对速度或预测位置导致稳定性收益低于阈值 | 可复用 `stability_cost()` 中的预测稳定性项 |
| Link-quality drop | SINR、data rate、PDR 或 NS3 返回链路质量低于阈值 | 在线 NS3/离线 NS3 replay 可逐步接入 |
| Utility degradation | 当前 coalition utility 相比上次接受状态下降超过阈值 | 避免只因小幅波动频繁重构 |
| Hard failure | cluster head 丢失、成员车辆消失、链路断开 | 绕过最小稳定时间，立即局部修复或重构 |
| Periodic guard | 超过最大保鲜周期仍未重评估 | 防止长期保持过期 cluster |

## 滞回策略

为避免振荡，trigger 不应直接等价于全局重聚类。建议增加两层滞回：

1. 最小稳定时间：若距离上次重构时间小于 `T_min_stab`，仅 hard failure 可立即触发。
2. 收益阈值：候选新 coalition 的 utility improvement 必须大于 `epsilon_u`，或当前稳定性低于 `beta_min`，才允许替换当前结构。

初始参数建议：

| Parameter | 初始值 | 说明 |
| --- | ---: | --- |
| `beta_min` | 0.5 | 预测稳定性低于该值时允许重构 |
| `epsilon_u` | 0.05 | utility 相对提升至少 5% |
| `utility_drop_eps` | 0.10 | 当前 utility 下降超过 10% 触发检查 |
| `min_reconfig_interval` | `T_min_stab` | 与论文稳定窗口保持一致 |
| `periodic_guard` | `cluster_interval` 或 1 s | 保底重评估周期 |

这些值不是论文最终参数，只是工程起点。进入论文前应通过更长动态序列做敏感性验证。

离线 replay 中的速度源建议优先使用相邻帧 pose 差分速度，而不是 dump 的 `ego_speed`。当前 dump 中 `ego_speed` 来自 `get_speed(vehicle)`，默认单位为 km/h；若与 m/s 阈值混用，会导致 relative-speed trigger 过敏。

## 触发结果

Trigger module 不直接输出最终 cluster，而输出重构级别：

| Result | 动作 |
| --- | --- |
| `NO_CHANGE` | 保持上一帧 cluster，仅更新车辆状态和 PPS 输入 |
| `LOCAL_REPAIR` | 只处理离开/加入/链路失败的局部车辆 |
| `RECLUSTER` | 调用 `CoalitionGame.run()` 全局或区域重构 |

推荐优先使用 `LOCAL_REPAIR`，只有 utility degradation、多个 cluster 同时失稳或 periodic guard 到期时才进入 `RECLUSTER`。

## 在线接入位置

建议在 `opencda/core/clustering/managers/clustering_v2x_manager.py` 中接入：

1. 每个周期先更新车辆 beacon、位置、速度和链路状态缓存。
2. 调用 topology trigger，比较当前状态与上次 accepted cluster state。
3. 若返回 `NO_CHANGE`，跳过 coalition formation，保留 cluster membership。
4. 若返回 `LOCAL_REPAIR`，只对失效 member/head 和候选邻居做局部处理。
5. 若返回 `RECLUSTER`，再调用 `CoalitionGame.run()`。
6. PPS/resource allocation 仍可按感知周期执行，因为即使 cluster 不变，点云密度和链路质量也会变化。

该设计可以修正文稿中 “topology change 才触发” 与 “每个周期重复” 的表述矛盾：每个周期执行状态观测和资源调度，但 cluster 结构只在触发条件满足时更新。

## 离线接入位置

建议在 `opencda.tools.offline_replay` 中先实现统计版本：

1. 从连续 dump 帧读取每个 CAV 的 pose、speed、可达邻居集合。
2. 计算 neighbor-set change、head/member 可达性、relative-motion risk。
3. 复用已有 cluster 输出，统计每帧 trigger type、是否实际发生 reconfiguration、vehicle-head change。
4. 将结果追加到 replay summary，形成论文稳定性指标。

后续再把相同 trigger 接入在线 `ClusteringV2XManager`。

## 论文表述建议

可将机制写成：

> SGCP continuously monitors beacon-level topology, relative motion, and link feasibility. A cluster reconfiguration is triggered only when the current coalition becomes unstable, infeasible, or sufficiently suboptimal; otherwise the previous coalition is retained and only perception resource scheduling is updated.

注意论文中不要声称当前实现已经使用 NS3 PDR/SINR 作为 trigger，除非后续完成在线或离线 NS3 link-quality 接入。

## 待实现

- 在线 `ClusteringV2XManager` 接入 trigger gate。
- 离线 replay 输出 trigger type 统计。
- 将 NS3/link-quality 结果接入 `Link-quality drop` trigger。
- 对 `beta_min`、`epsilon_u`、`periodic_guard` 做敏感性实验。
