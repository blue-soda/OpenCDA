# SGCP Potential Game Conditions

本文档复核 SGCP PPS / resource allocation 中 “potential game” 表述的成立条件、当前代码实现与论文写作边界。

## 代码位置

当前实现入口：

```text
opencda/core/clustering/algorithms/resource_allocation/potential_game.py
```

离线调用入口：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

## 当前实现摘要

当前 `PotentialGame` 做的是 sequential best-response scheduling：

1. 每个 cluster head `h` 计算自身 requested grids `J_req` 和当前已覆盖的 effective grids `J_eff`。
2. 候选网格为 `J_req - J_eff`。
3. 对每个 member/grid 计算边际感知收益 `grid_score()`。
4. 第一轮选择未占用 RB 上收益最高的 member/grid subset。
5. 第二轮允许受保护 RB 复用，但需要通过 SINR / capacity 检查。
6. 多轮迭代直到没有 cluster 更新，或者达到 `max_iter=20`。

代码中的收益核心：

```text
gain = max(late_score, early_score_if_upload) - max(late_score, early_score)
```

这表达的是某个 member 上传某个 grid 后，对 “cluster early utility 与 inter-cluster late utility 取较大者” 的边际提升。

## Exact Potential 成立所需条件

若论文要严格称为 exact potential game，需要至少满足：

1. 每个 player 的 action 是明确定义的独立策略，例如 `(member, subchannel, timeslot, selected_grids)`。
2. 存在全局势函数 `Phi(a)`，使任意单个 player 单边改变策略时：

```text
u_i(a_i', a_-i) - u_i(a_i, a_-i)
= Phi(a_i', a_-i) - Phi(a_i, a_-i)
```

3. 每个 player 的局部 utility 必须等于其 action 对全局 grid-level utility 的边际贡献。
4. 资源冲突、干扰、带宽约束要么是 action feasibility constraints，要么以同样的 penalty 同时进入局部 utility 和全局 potential。
5. best response 更新必须允许替换旧 action，并且每次 accepted move 都不降低 `Phi`。
6. 若使用 inter-cluster late utility，其他 cluster 对同一 grid 的贡献必须在 `Phi` 和 `u_i` 中一致计入。

## 当前实现与严格条件的差异

| 条件 | 当前状态 | 影响 |
| --- | --- | --- |
| 明确 player/action | 部分满足。实现以 cluster head 顺序选择 member/grid/RB，但 action set 没有独立抽象。 | 可以称为 scheduling heuristic，但数学定义需要补。 |
| 全局 `Phi` | 未显式实现。 | 不能直接声称代码验证 exact potential。 |
| 局部收益等于全局边际 | 部分满足。`grid_score()` 是 grid utility 的边际提升，但只覆盖被考虑的候选 grid。 | 需要限定在固定 cluster、固定 candidate set 和固定 late utility 状态下。 |
| 资源冲突一致计入 | 部分满足。RB 占用、capacity、SINR 是 feasibility gate，不是 utility penalty。 | 可写为 constrained potential game / feasible best response。 |
| 单边替换旧 action | 当前主要是追加 schedule；替换逻辑被注释。 | 收敛是有限追加过程，不是完整 best-response dynamics。 |
| inter-cluster late utility | 部分满足。`get_participating_clusters()` 当前遇到第一个 cluster 后 `break`，不是完整多簇聚合。 | 严格 late-fusion potential 需要修正或在论文中弱化。 |
| 收敛保证 | 工程上有限迭代、有限 RB/grid、无更新停止。 | 可声称 empirical convergence / finite monotone scheduling，不宜声称 exact-potential 定理已由当前代码完整实现。 |

## 当前可安全写法

建议论文将当前实现写成：

> PPS is implemented as a constrained best-response scheduler guided by a grid-level potential utility. For a fixed clustering result and a fixed candidate grid set, each accepted upload is selected by its marginal improvement to the shared perception utility under RB, capacity, and SINR feasibility checks. The implementation converges in a finite number of iterations because it only accepts feasible non-empty schedules over a finite action set and stops when no cluster can add a beneficial schedule. We therefore use the potential utility as the scheduling objective and report empirical convergence/runtime, while leaving a fully general exact-potential proof with replacement dynamics as a formal extension.

如果论文必须保留 exact potential game 表述，应加上限定：

> Under fixed cluster membership, fixed candidate grids, additive grid utility, and feasibility constraints treated as hard action constraints, the grid-level sum utility can serve as an exact potential for unilateral upload-grid changes whose local utility is defined as the corresponding marginal global utility.

## 建议修正路径

进入论文最终版前，建议二选一：

1. **保守写法**：把算法命名为 potential-guided constrained best-response scheduling，不再强称完整 exact potential game。当前代码和实验已经能支撑该口径。
2. **严格写法**：补一个显式 `Phi` 计算器、action replacement、monotonicity assert，并修复 `get_participating_clusters()` 的多簇 late utility 聚合，再用离线 replay 输出每次 accepted move 的 `Delta Phi >= 0` 统计。

## 当前结论

当前代码可以支撑 “potential-guided PPS / constrained best-response scheduling” 和 “finite empirical convergence”。它尚不能无条件支撑 “完整 exact potential game 已严格实现并证明”。论文中应把 exact potential 的成立条件写成受限假设，或者将强表述改为 potential-guided heuristic，以免审稿人抓住理论和代码之间的缝隙。

## 已接入的经验收敛诊断

`PotentialGame` 当前已记录每帧 PPS convergence statistics，并由 `opencda.tools.offline_replay` 汇总输出：

```text
summary pps_convergence avg_iterations=3.00 max_iterations=3 converged_frames=41 avg_cluster_updates=10.00 avg_scheduled_links=10.00 avg_selected_grids=523.90 avg_used_rbs=10.00 avg_reused_rbs=0.00 max_rb_occupancy=1
```

当前 41 帧 dump 的结果：

| Metric | Value |
| --- | ---: |
| Frames converged before `max_iter=20` | 41 / 41 |
| Avg. iterations | 3.00 |
| Max iterations | 3 |
| Avg. cluster updates / frame | 10.00 |
| Avg. scheduled links / frame | 10.00 |
| Avg. selected grids / frame | 523.90 |
| Avg. used RBs / frame | 10.00 |
| Avg. reused RBs / frame | 0.00 |
| Max RB occupancy | 1 |

解释：该结果支持 “finite empirical convergence” 和 “当前默认场景下 PPS 使用 10 个 non-conflicting RB 完成调度”。它仍不是 exact-potential 的数学证明；若论文写作需要严格证明，应继续补显式 `Phi` 和每次 accepted move 的 `Delta Phi` 记录。
