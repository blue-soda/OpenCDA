# FullPerception Baseline Revision

更新时间：2026-07-18

本文档用于把 FullPerception baseline 的实现细节、公平性边界和论文/rebuttal 写法收束成可直接迁移到 SGCP 论文的材料。核心目标是避免把集中式 upper reference、全量共享 reference 和同通信预算 V2V baseline 混成同一个比较对象。

## 问题定位

审稿意见中对 FullPerception baseline 的质疑主要来自三个不清楚：

- `v2xp_cluster_carla` 当前是 RSU-free 场景，论文若写 RSU-assisted reference，必须说明 RSU 是真实传感器、虚拟聚合点，还是只作为 upper reference。
- FullPerception 若允许接收所有 CAV 点云或检测框，它拥有的信息和通信能力强于 SGCP，不能作为同通信预算公平主对比。
- 如果 FullPerception 被写成 decentralized baseline，则需要明确它只使用 CAV-side V2V 信息，并且匹配 SGCP 的通信预算、backbone、数据帧和评价口径。

因此，论文中应把相关结果拆成三个层级：`Full 20-CAV sharing upper reference`、`FullPerception PCS (pcs.py)` 正式 baseline、以及后补的 selective-sharing proxy。后补 proxy 不再使用 FullPerception 名称。

## 代码核查修正

2026-07-18 重新核查后确认：仓库中虽然没有以 `FullPerception` 命名的入口，但 `opencda/core/clustering/algorithms/resource_allocation/pcs.py` 对应 FullPerception 论文中的 Proactive Conflict-free Scheduling (PCS) 实现。该实现包含：

- blind spot grouping：`_get_vehicle_blind_spots()`；
- potential link generation：`_generate_potential_links()`；
- link utility / grid mAP：`_precompute_grid_mAP()` 和 `_calculate_link_utilities()`；
- conflict graph：`_build_conflict_graph()`；
- weight splitting：`_weight_splitting()`；
- recursive PCS scheduling：`_pcs_recursion()`；
- subchannel assignment and grid selection：`_resource_allocation()`。

同时，`mws.py` 和 `random_ra.py` 继承 `PCS`，对应 FullPerception 论文中的 MWS 和 RS heuristic baselines。已新增 alias：`--resource-allocation fullperception_pcs|fullperception_mws|fullperception_random`。

当前 `fullperception_pcs` 是正式 FullPerception baseline。旧的 legacy cluster-head evaluation 为 `0.44/0.39/0.17`，payload 12,684,880 bytes / 24.75 Mbps；它来自 `_get_link_required_subchannels()` 直接返回 1 的简化实现。第一轮协议修复后，`c(q)` 已由 sender 覆盖 blind grids 的点云 payload、20 MHz/10ch 和 0.1 s slot 计算，`sc_num` 会写入 OpenCDA scheduler 与 NS3 replay，receiver policy 也可以只评估实际 scheduled receivers。第二轮继续修复 blind-spot cache key、grid mAP cache index、utility/payload/grid-selection 的 blind-spot 粒度对齐，并将 PCS blind-spot unit 调为 `division=12,min_overlap=0`。当前 41 帧 tuned scheduled-receiver 结果为 `0.59/0.53/0.22`，payload 12,959,840 bytes / 25.29 Mbps；11 帧 NS3 dry-run 每帧 5 条 scheduled request、0 skipped unscheduled。该结果可作为正式但不强势的 FullPerception PCS baseline。

## Full Sharing / RSU 口径

真实 RSU-sensor reference 应定义为 centralized or RSU-assisted upper reference，而不是主公平 baseline。

可接受实现有三种：

| 实现方式 | 是否适合主对比 | 说明 |
| --- | --- | --- |
| 真实 RSU dump | 否 | 需要重新导出包含 RSU sensor/pose 的 CARLA 数据；代表 infrastructure-assisted 上界。 |
| 虚拟 RSU 聚合全 CAV 数据 | 否 | 当前 20-CAV full early/late fusion 更接近这个口径；应标注为 centralized oracle/reference。 |
| 只在表中说明不复现 | 是，作为说明 | 若主贡献是 RSU-free decentralized CP，可以不把 RSU baseline 放入主结果表。 |

当前 `D:\Data\Carla\2026_07_15_01_26_56` dump 不包含 RSU 目录，因此不能声称复现了真实 RSU sensor 版 FullPerception。已完成的 full 20-CAV early fusion `0.85/0.83/0.48` 和 full 20-CAV late checkpoint `0.91/0.85/0.51` 只能写成 full-sharing reference。若把 full 20-CAV early fusion 解释为 centralized upper reference，其 41 帧 non-ego CAV 点云上传 payload 为 60,838,528 bytes。

此前实现的虚拟 RSU/global candidate pool 分支已重命名为 `global_selective_proxy`：每个 cluster head 可以从全局 CAV 中选择 sender，但仍受 3 members/head、117 grid budget 和 20MHz/10ch request plan 约束。重命名前 41 帧结果为 `0.84/0.80/0.46`，payload 56,224,736 bytes / 109.71 Mbps。该行应写为 RSU/edge-assisted proxy，不作为 V2V-only 公平主对比，也不再命名为 FullPerception。

## Cluster-Local Selective Proxy 口径

若 reviewer 要求 decentralized FullPerception baseline，建议用 same-budget CAV-only selective sharing 表示，而不是无约束 full early fusion。

当前已实现的可复现口径：

- 数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。
- 感知：复用 OpenCOOD checkpoint 和 SGCP inter-cluster late-fusion evaluation path。
- 结构：复用 SGCP coalition/cluster head 结构，不使用 RSU/全局 oracle。
- 通信：主公平随机 baseline 强制每个 cluster head 最多选 3 个非 head 成员，grid budget 为 117，使 payload 与 PAPG 主方法接近；强 selective baseline 同样使用 3 members/head + 117 grid budget。
- 策略：`random`、`nearest`、`density`、`communication_aware`、`cluster_local_selective_proxy`。
- NS3 扩展：`communication_aware` 可读取 `rlc_by_request.csv`，用 request-level `rlc_complete` 作为链路质量权重。

主表可采用如下解释：

| Method | AP@0.3 / AP@0.5 / AP@0.7 | Payload | 公平性 |
| --- | --- | --- | --- |
| SGCP potential_game | 0.77 / 0.73 / 0.35 | 26,916,208 bytes | 原始 PPS 消融 |
| Selective nearest | 0.76 / 0.73 / 0.37 | 28,026,832 bytes | 同数据、同评估口径、略高 payload |
| Selective density | 0.77 / 0.74 / 0.39 | 30,574,368 bytes | 强 V2V baseline，高 payload |
| Selective communication-aware | 0.78 / 0.75 / 0.40 | 30,222,256 bytes | 当前最强 V2V baseline，高 payload |
| Selective forced random | 0.77 / 0.73 / 0.38 | 31,613,424 bytes | 3 members/head, 117 grid budget, payload 接近 PAPG |
| Selective density high-budget | 0.80 / 0.76 / 0.40 | 37,710,864 bytes | 3 members/head, 117 grid budget |
| Cluster-local selective proxy | 0.80 / 0.76 / 0.41 | 38,920,592 bytes | formerly `fullperception_decentralized`; cluster-local V2V candidates, 3 members/head, 117 grid budget; NS3 110/110 complete |
| SGCP coverage-aware 10ch | 0.79 / 0.76 / 0.38 | 29,405,296 bytes | PAPG 前身/消融，NS3 110/110 complete |
| SGCP PAPG 10ch | 0.81 / 0.78 / 0.39 | 32,049,872 bytes | 当前主方法，NS3 110/110 complete |
| SGCP coverage-aware 20ch | 0.80 / 0.76 / 0.41 | 37,912,544 bytes | 资源敏感性/高预算附表 |

这个结果要求论文叙事保持诚实：Full 20-CAV early fusion 是 centralized upper reference，不是公平主对比；旧 RandomRA/MWS 因 payload 过低只能作为 w/o-PPS 消融；公平随机 baseline 应使用 forced-budget random。当前可写入主表的 SGCP 主方法是 PAPG 10ch：它在接近 forced-budget random 的通信量下提升 AP，并且相对 high-budget density baseline 在 AP@0.3/AP@0.5 更高、通信量更低。coverage-aware 10ch/20ch 应保留为消融或资源敏感性结果，而不再作为主算法行。

## 论文正文建议

建议在 Experiment Setup 的 baseline 段替换为：

```tex
We separate full-sharing upper references, the built-in FullPerception PCS baseline, and fair decentralized baselines. Full 20-CAV early fusion assumes unconstrained complete point-cloud sharing and is reported only as an upper reference. FullPerception is evaluated through the repository's PCS implementation (`pcs.py`) with payload-based subchannel demand and explicit scheduled links. The main fair comparisons use CAV-only selective sharing under the same CARLA dump, OpenCOOD backbone, evaluation protocol, cluster-head late fusion path, and matched source/grid budgets. We instantiate forced-random, density-based, and communication-aware selective-sharing baselines, where the communication-aware variant penalizes distant or NS3-unreliable links using request-level RLC completion traces.
```

建议在主结果表注释中增加：

```tex
Full 20-CAV fusion is not included in the fair-budget ranking because it relies on unconstrained sharing. The built-in FullPerception PCS baseline is listed separately from selective-sharing proxies to avoid conflating the paper's PCS scheduler with virtual/global candidate-pool diagnostics.
```

## Rebuttal 建议

可直接回应 reviewer：

```text
Thank you for pointing out the ambiguity of FullPerception. In the revision, we explicitly distinguish the full-sharing upper reference, the built-in FullPerception PCS baseline, and decentralized fair baselines. Full 20-CAV early fusion is reported only as an upper reference. FullPerception is evaluated through the repository's PCS implementation (`pcs.py`), whose subchannel demand and scheduled links are now explicitly connected to the OpenCDA/NS3 protocol. For the fair comparison, we add CAV-only selective-sharing baselines that use the same CARLA dump, OpenCOOD backbone, cluster-head late-fusion evaluation path, and matched source/grid budgets. We include a forced-budget random baseline to avoid under-utilized random scheduling, and a communication-aware variant that incorporates NS3 request-level RLC completion traces, so the baseline is constrained by link feasibility instead of only selecting perceptually dense grids.
```

如果 reviewer 追问为什么不把 full 20-CAV early fusion 当作主 baseline：

```text
Full 20-CAV early fusion assumes all vehicles can upload complete point clouds to a common fusion center, which violates the RSU-free and subchannel-limited setting considered by SGCP. We therefore report it as an upper reference and compare SGCP against V2V-only selective-sharing baselines under matched communication budgets.
```

## 后续补强

- 若论文版面允许，保留一个 small upper-reference table，单列 full early fusion、full late fusion 和 true RSU-sensor reference（若未来重新导出）。
- 主表使用 SGCP PAPG、forced-budget random、density/communication-aware selective sharing、FullPerception-PCS baseline 和 full 20-CAV early-fusion upper reference；旧 random/MWS scheduler 只进入 w/o-PPS 消融或诊断表。
- PAPG 已完成 11 帧真实 NS3 replay：110/110 scheduled request application/RLC complete，RLC drops=0，PHY decode failures=0。
- 若后续重新打开 CARLA，可导出带 RSU sensor 的真实 RSU-assisted 数据，但它仍应标注为 infrastructure-assisted upper reference。
- 当前 selective proxy 代码已接入 `offline_ns3_replay --selective-sharing-baseline`。`cluster_local_selective_proxy` 的链路证据来自重命名前的 11-frame true NS3 replay：110/110 application callback complete、110/110 RLC complete、RLC drops=0、PHY decode failures=0，使其链路证据与 PAPG/forced random 对称。
- 对 built-in `fullperception_pcs`，当前 tuned 结果已经可进入 baseline 表；下一步只需决定 `fullperception_mws/fullperception_random` 是否也采用相同 blind-spot 粒度，或保留原论文 heuristic 默认参数后作为 w/o-PCS 诊断。
