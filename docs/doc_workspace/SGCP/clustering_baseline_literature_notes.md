# Clustering Baseline Literature Notes

更新时间：2026-07-21

用途：为 `target.md` P10.3 选择分簇 baseline。本文只记录可映射到当前 SGCP raw-LiDAR V2V 离线实验的候选；不把 unrelated feature-fusion detector 直接写成 clustering baseline。

## 当前可立即复现的启发式 baseline

| Baseline | Implementation | Rationale | Limitation |
| --- | --- | --- | --- |
| Random balanced clustering | `offline_inference --clustering random_balanced` | 固定 seed 的随机均衡分簇，检验 SGCP coalition 是否优于无感知/无拓扑先验分组。 | 不是论文算法，只作 lower heuristic。 |
| Distance-greedy clustering | `offline_inference --clustering distance_greedy` | 按空间距离形成紧凑簇，近似 V2V/vehicular-network clustering 中常见的 proximity-aware 思路。 | 不考虑 perception coverage 和 density。 |
| Density/quality-greedy clustering | `offline_inference --clustering density_greedy_cluster` | 选择高感知密度车辆作为种子，并用新增感知覆盖扩展簇，贴近 cooperative perception 的 sensing-aware grouping。 | 当前是 SGCP 数据结构上的启发式，还不是外部论文严格复现。 |

## Literature-informed candidates

1. **MASS: Mobility-aware sensor scheduling**  
   来源：`Model-Assisted Learning for Adaptive Cooperative Perception` 的 related-work 明确引用 Jia et al. 的 raw-level cooperative perception scheduling 与 MASS（IEEE TVT 2023）。它更偏 sensor scheduling 而不是完整 clustering，但可映射为 mobility/distance-aware clustering + sender selection baseline。适合作为“mobility-aware heuristic”实现。检索入口：https://arxiv.org/html/2401.10156v1

2. **When2com / communication graph grouping**  
   V2X cooperative perception survey 将 When2com 归为 multi-agent perception via communication graph grouping。它是学习式 communication graph，不适合直接复现到当前无训练 pipeline，但可以作为“graph-connectivity grouping”参考，转化为基于距离/coverage 相似度的 graph clustering baseline。检索入口：https://arxiv.org/html/2310.03525v5

3. **Recent V2X cooperative perception surveys**  
   2023/2024 V2X cooperative perception survey 将 agent selection、communication constraints、fusion strategy 作为关键问题，支持本文把 clustering baseline 与 scheduler comparison 分开报告，而不是把所有 baseline 强行塞入一个“主表”。检索入口：https://arxiv.org/abs/2310.03525

## Decision

P10.3 先完成三类 deterministic heuristic 的 41 帧消融；随后从 MASS/graph-grouping 思路中选择 1--2 个 literature-inspired mapping，优先实现：

- mobility-aware distance/stability clustering：按距离、相对速度、稳定窗口形成簇；
- graph coverage clustering：按 sensing-grid overlap / blind-grid complementarity 构图后做 greedy cluster cover。

二者必须使用同一 attentive checkpoint、同一 PAPG resource allocation、同一 inter-cluster NMS 和 20MHz/10ch；只替换 clustering。
