# SGCP Cluster Capacity Policy

更新时间：2026-07-15

本文档定义 SGCP 中 cluster 已满、merge/split 和成员边际贡献重算的机制口径，用于回应审稿意见中关于 `N_max`、动态加入和振荡控制的问题。

## 当前实现

`CoalitionGame.coalition_formation()` 当前采用硬容量约束：

- 每个 cluster 的最大成员数由 `Params.N_max` 控制。
- 当候选 cluster `c.size() >= N_max` 时，新车辆不能加入该 cluster。
- 当前实现不会临时超过 `N_max`。
- 当前实现不会主动 split/merge cluster。
- 如果车辆无法找到收益更高且未满的 cluster，则保持在原 cluster；若它本来是 singleton，则继续作为 singleton cluster。

该行为是保守且安全的，但论文中需要显式说明，否则容易被理解为遗漏了满簇处理。

## 机制口径

建议将 cluster 已满时的策略定义为分层处理：

| Priority | Policy | 动作 | 说明 |
| ---: | --- | --- | --- |
| 1 | Keep current cluster | 若当前 cluster 可行，车辆保持原 cluster | 避免因满簇导致频繁迁移 |
| 2 | Join non-full better cluster | 只允许加入 `size < N_max` 且收益提升超过阈值的 cluster | 对应当前代码主路径 |
| 3 | Replacement repair | 若目标 cluster 已满，只在新车辆收益显著高于某个现有成员时考虑替换 | 后续可实现，当前不启用 |
| 4 | Singleton fallback | 若没有可行 cluster，车辆保持或形成 singleton | 避免强行超载 |
| 5 | Inter-cluster late fusion compensation | singleton 或小 cluster 仍通过簇头 late fusion 进入全局检测 | 降低孤立车辆损失 |

当前代码已经覆盖 1、2、4、5 的基本语义：不能加入满簇时车辆不会被丢弃，而是保留在原有 cluster 或 singleton；最终仍可通过 inter-cluster late fusion 汇总 cluster head 检测结果。

## Replacement Repair

后续若需要增强机制，可加入替换策略，但不建议作为默认策略直接打开。候选规则：

1. 对目标满簇 `C` 中每个成员 `j` 计算当前贡献 `u_j(C)`。
2. 临时移除 `j`，计算新车辆 `i` 加入后的贡献 `u_i(C - {j})`。
3. 只有当 `u_i - u_j > epsilon_replace` 且不会破坏稳定窗口时，允许替换。
4. 被替换车辆优先回到原 cluster；若原 cluster 不可行，则形成 singleton。

该策略可以提升局部 utility，但也会增加控制开销和振荡风险。论文中若没有实验支持，应写为 future extension 或 optional local repair。

## Merge/Split

建议论文主机制采用以下口径：

- 不允许任意 merge 后超过 `N_max`。
- 若两个 cluster 合并后 `size <= N_max` 且 utility 提升超过阈值，可视为普通 coalition move 的结果，而不是单独的 merge primitive。
- 若某个 cluster 因车辆离开或 link failure 变小，不需要显式 split。
- 若一个 cluster 内部出现 head/member 不可达，topology trigger 触发 re-cluster；这等价于事件驱动 split，而不是允许 cluster 在稳定状态下主动 split。

也就是说，`N_max` 是硬上限；merge/split 不是独立的无限制操作，而是通过 coalition formation 和 topology trigger 间接实现。

## 成员加入后的边际贡献重算

当前 `CoalitionGame` 的迭代流程已经具备重算语义：

1. 每轮遍历车辆。
2. 对车辆 `vid` 先计算其在当前 cluster 中的 `current_contribution`。
3. 对每个未满候选 cluster 重新计算 `marginal_contribution`。
4. 车辆移动后，cluster 的 `members`、`sens_grids`、`req_grids`、`head_id` 通过 `Cluster.add_member/remove_member` 更新。
5. 下一辆车或下一轮迭代会基于更新后的 cluster 状态重新计算贡献。

因此“成员加入后是否重新计算已有成员边际贡献”的答案是：不会在同一原子操作中立即对所有成员同步重算，但后续迭代会使用更新后的 coalition state 重新评估每个车辆的 current/marginal contribution。为了避免振荡，只有当 `best_delta > current_contribution * ita` 时才移动。

## 论文建议表述

可写为：

> SGCP treats `N_max` as a hard coalition capacity constraint. A vehicle can only migrate to a non-full coalition if its marginal perception utility exceeds its current contribution by a hysteresis factor. If all beneficial coalitions are full, the vehicle keeps its current coalition or remains a singleton; singleton/small-coalition predictions are still included through inter-cluster late fusion. Cluster split/merge is not a separate unconstrained primitive, but emerges from event-triggered coalition reformation under the same `N_max` constraint.

注意不要声称当前实现已有 replacement repair，除非后续补代码和实验。

## 待实现

- 在离线 replay 中统计每帧满簇数量、因满簇被跳过的候选 move 数。
- 如需增强机制，增加默认关闭的 replacement repair。
- 将 singleton/small cluster 的 detection-box exchange overhead 纳入通信统计。
