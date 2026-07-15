# Related Work and Novelty Revision

本文档面向 P4 写作修订，专门回应两个审稿问题：

1. Related work 缺少 decentralized / V2V-only collaborative perception 与 coalition-game 方法的清晰对比。
2. SGCP 的 novelty 容易被理解为把已有 coalition formation 方法简单套到车辆网络。

## 审稿意见映射

Reviewer 2:

> The coalition formation algorithm seems very similar to existing approaches applied in different areas other than the vehicular networks like Smartform.

Reviewer 3/4:

> The baseline methods selected for the experiments do not constitute a fair and comprehensive comparison framework, particularly due to the absence of direct comparisons with the latest, similar decentralized collaborative perception methods.

> It would be more convincing to see how it performs against other decentralized state-of-the-art methods that do not rely on roadside units.

## Related Work 缺口

当前 `main.tex` 的 related work 已分为：

- Collaboration granularity: late / intermediate / early fusion。
- Communication and scheduling architectures: RSU-centric designs。

但它还不够突出：

- V2V-only / infrastructure-free CP 的已有路线。
- Decentralized scheduling 与 learned communication selection 的区别。
- Coalition formation 在其他网络/energy trading 场景中的相似性与 SGCP 的差异。
- 为什么 SGCP 不是泛化 coalition game，而是 perception-utility / hierarchical fusion / wireless feasibility 的组合。

## 建议新增小节

建议在 `Communication and Scheduling Architectures` 后新增一个短小节：

```tex
\subsection{Decentralized V2V Coordination and Coalition Formation}

Recent V2V-based collaborative perception studies reduce infrastructure dependence by learning when, where, or with whom to communicate. Methods such as When2Com and Where2Comm select informative partners or spatial regions from neural features, and graph-based late-fusion methods aggregate object-level predictions over vehicle-to-vehicle links. These approaches improve communication efficiency, but they usually assume a fixed neighbor set, idealized communication, or homogeneous detector features, and they do not jointly decide coalition structure, physical-channel allocation, and hierarchical fusion under explicit resource constraints.

Coalition formation has also been studied in other distributed systems, such as smart-grid energy trading and self-managed resource sharing. Those methods typically optimize economic utility, load balance, or social welfare under relatively static graph constraints. Directly applying them to vehicular collaborative perception is insufficient because the value of a coalition is not only a function of member count or link existence, but depends on spatially varying perception complementarity, LiDAR density saturation, vehicle motion stability, and the feasibility of transmitting selected point-cloud regions before a 100 ms sensing deadline.

SGCP differs from these lines of work in three aspects. First, the coalition utility is perception-driven: it is computed from grid-level sensing complementarity and calibrated point-cloud density rather than generic connectivity or payoff. Second, coalition updates are stability-aware and capacity-constrained, using a minimum stability window and a hard cluster-size bound to avoid oscillation and channel congestion. Third, SGCP couples coalition formation with PPS, a potential-guided constrained scheduler that selects high-value point-cloud grids under subchannel, SINR, and payload constraints, followed by inter-cluster late fusion. This combination targets infrastructure-free dense urban perception rather than generic coalition formation.
```

## Existing Related Work 段落微调建议

当前 `Communication and Scheduling Architectures` 最后一段：

```tex
In contrast, most existing methods either depend on RSUs or assume idealized communication conditions, limiting their scalability in real-world vehicular networks. To the best of our knowledge, no prior work optimizes collaboration structure and resource-constrained data sharing in a fully decentralized manner under dynamic, large-scale settings. Our framework addresses this gap by integrating stability-aware cluster formation with game-theoretic scheduling to maximize perception gains without roadside infrastructure.
```

建议改为更克制、可防守：

```tex
In contrast, existing CP systems either rely on RSU-assisted global scheduling, learn communication selection without explicit physical-channel feasibility, or use decentralized fusion over a fixed neighbor graph. SGCP targets the missing intersection of these requirements: infrastructure-free collaboration, adaptive coalition structure, perception-utility-driven point-cloud selection, and explicit V2V resource constraints. Rather than treating decentralized CP as only a partner-selection problem, SGCP jointly models who should cooperate, which regions should be uploaded, and which subchannels can support the upload within the sensing deadline.
```

## Novelty 强化口径

建议在 Introduction contributions 中避免只写 “game-theoretic framework”，改为强调组合贡献：

```tex
Our novelty is not the use of coalition games alone, but the integration of perception utility, stability-aware coalition control, and resource-feasible point-cloud scheduling for RSU-free collaborative perception. Specifically, SGCP introduces: (i) a grid-level perception utility calibrated from LiDAR density and used consistently for coalition formation and upload selection; (ii) a stability-aware, capacity-constrained cluster formation mechanism that avoids frequent reconfiguration while preserving inter-cluster late fusion; and (iii) a potential-guided PPS scheduler that maps selected point-cloud grids to feasible V2V subchannels and can be validated with NS3 request-level traces.
```

## Rebuttal 答法

针对 Smartform / coalition game 相似性：

> We agree that coalition formation has been used in other distributed systems. We clarified that SGCP does not claim coalition formation itself as new. The novelty lies in adapting coalition formation to collaborative perception with a perception-specific utility: coalition value depends on grid-level sensing complementarity, calibrated LiDAR density, motion stability, and downstream V2V scheduling feasibility. Unlike smart-grid coalition formation, SGCP must jointly handle 100 ms sensing cycles, cluster-size bounds, point-cloud payload selection, subchannel allocation, and hierarchical early/late fusion.

针对 decentralized baseline 不足：

> We added V2V-only selective-sharing baselines that use the same dump, detector backbone, SGCP cluster-head evaluation path, and matched grid/member budgets. These baselines include nearest, density-based, communication-aware, and NS3 RLC-complete-aware variants. The strongest communication-aware baseline achieves slightly higher AP but with higher payload and without SGCP's stability/PPS feasibility guarantees; we therefore present SGCP as a stability- and channel-feasible decentralized framework rather than claiming AP dominance over every selective-sharing heuristic.

## 与当前实验结果的绑定

可引用的 SGCP 文档：

- `baseline_fairness.md`：FullPerception-RSU 与 same-budget V2V-only baseline 的公平性说明。
- `results.md`：nearest / density / communication-aware / NS3-aware selective baseline 表格。
- `potential_game_conditions.md`：PPS 不宜过强声称 exact potential game。
- `control_overhead.md`：控制开销约为 perception payload 的 0.70%。
- `paper_revision_plan.md`：整体 rebuttal / main.tex 替换建议。

## Target 状态

本文件完成 P4 中两项：

- 重写 related work 的 decentralized CP 和 coalition game 对比。
- 增强 novelty：突出感知效用驱动、稳定性约束、分层 fusion、分布式资源调度的组合贡献。
