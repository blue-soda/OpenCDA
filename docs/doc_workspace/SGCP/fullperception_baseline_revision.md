# FullPerception Baseline Revision

更新时间：2026-07-16

本文档用于把 FullPerception baseline 的实现细节、公平性边界和论文/rebuttal 写法收束成可直接迁移到 SGCP 论文的材料。核心目标是避免把集中式 upper reference、全量共享 reference 和同通信预算 V2V baseline 混成同一个比较对象。

## 问题定位

审稿意见中对 FullPerception baseline 的质疑主要来自三个不清楚：

- `v2xp_cluster_carla` 当前是 RSU-free 场景，论文若写 FullPerception-RSU，必须说明 RSU 是真实传感器、虚拟聚合点，还是只作为 upper reference。
- FullPerception 若允许接收所有 CAV 点云或检测框，它拥有的信息和通信能力强于 SGCP，不能作为同通信预算公平主对比。
- 如果 FullPerception 被写成 decentralized baseline，则需要明确它只使用 CAV-side V2V 信息，并且匹配 SGCP 的通信预算、backbone、数据帧和评价口径。

因此，论文中应把 FullPerception 拆成两个层级：`FullPerception-RSU / full-sharing upper reference` 和 `FullPerception-Decentralized / same-budget V2V selective sharing`。

## FullPerception-RSU 实现口径

`FullPerception-RSU` 建议定义为 centralized or RSU-assisted upper reference，而不是主公平 baseline。

可接受实现有三种：

| 实现方式 | 是否适合主对比 | 说明 |
| --- | --- | --- |
| 真实 RSU dump | 否 | 需要重新导出包含 RSU sensor/pose 的 CARLA 数据；代表 infrastructure-assisted 上界。 |
| 虚拟 RSU 聚合全 CAV 数据 | 否 | 当前 20-CAV full early/late fusion 更接近这个口径；应标注为 centralized oracle/reference。 |
| 只在表中说明不复现 | 是，作为说明 | 若主贡献是 RSU-free decentralized CP，可以不把 RSU baseline 放入主结果表。 |

当前 `D:\Data\Carla\2026_07_15_01_26_56` dump 不包含 RSU 目录，因此不应把 FullPerception-RSU 填成真实主实验结果。已完成的 full 20-CAV early fusion `0.85/0.83/0.48` 和 full 20-CAV late checkpoint `0.91/0.85/0.51` 只能写成 full-sharing reference。

## FullPerception-Decentralized 实现口径

若 reviewer 要求 decentralized FullPerception baseline，建议用 same-budget CAV-only selective sharing 表示，而不是无约束 full early fusion。

当前已实现的可复现口径：

- 数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。
- 感知：复用 OpenCOOD checkpoint 和 SGCP inter-cluster late-fusion evaluation path。
- 结构：复用 SGCP coalition/cluster head 结构，但不使用 PPS。
- 通信：每个 cluster head 最多选 2 个非 head 成员，grid budget 为 87，接近 SGCP `avg_selected_grids=87.32`。
- 策略：`nearest`、`density`、`communication_aware`。
- NS3 扩展：`communication_aware` 可读取 `rlc_by_request.csv`，用 request-level `rlc_complete` 作为链路质量权重。

主表可采用如下解释：

| Method | AP@0.3 / AP@0.5 / AP@0.7 | Payload | 公平性 |
| --- | --- | --- | --- |
| SGCP potential_game | 0.77 / 0.73 / 0.35 | 26,916,208 bytes | 主方法 |
| Selective nearest | 0.76 / 0.73 / 0.37 | 28,026,832 bytes | 同数据、同评估口径、略高 payload |
| Selective density | 0.77 / 0.74 / 0.39 | 30,574,368 bytes | 强 V2V baseline，高 payload |
| Selective communication-aware | 0.78 / 0.75 / 0.40 | 30,222,256 bytes | 当前最强 V2V baseline，高 payload |

这个结果要求论文叙事保持诚实：在当前短 41 帧 dump 上，SGCP 不是 AP 全面领先。更稳妥的贡献表述是 SGCP 在接近通信预算下提供 cluster stability、PPS 子信道可行性、NS3 可验证的无冲突传输、控制开销可解释和分层 early/late fusion，而 naive density-based selective sharing 是强竞争基线。

## 论文正文建议

建议在 Experiment Setup 的 baseline 段替换为：

```tex
We separate centralized full-sharing references from fair decentralized baselines. Full early/late fusion and FullPerception-RSU assume either infrastructure support or an oracle aggregator that can access all CAV observations; therefore, they are reported only as upper references. The main fair comparisons use CAV-only selective sharing under the same CARLA dump, OpenCOOD backbone, evaluation protocol, cluster-head late fusion path, and a matched grid budget. We instantiate nearest-neighbor, density-based, and communication-aware selective-sharing baselines, where the last variant penalizes distant or NS3-unreliable links using request-level RLC completion traces.
```

建议在主结果表注释中增加：

```tex
FullPerception-RSU and full 20-CAV fusion are not included in the fair-budget ranking because they rely on centralized/infrastructure-assisted information and unconstrained sharing. They indicate the perception upper bound rather than a deployable RSU-free V2V protocol.
```

## Rebuttal 建议

可直接回应 reviewer：

```text
Thank you for pointing out the ambiguity of FullPerception. In the revision, we explicitly distinguish centralized full-sharing references from decentralized fair baselines. Since our evaluated v2xp_cluster_carla setting is RSU-free, FullPerception-RSU is treated as an infrastructure-assisted upper reference rather than the main fair comparison. For the fair comparison, we add CAV-only selective-sharing baselines that use the same CARLA dump, OpenCOOD backbone, cluster-head late-fusion evaluation path, and a matched grid budget. We further include a communication-aware variant that incorporates NS3 request-level RLC completion traces, so the baseline is constrained by link feasibility instead of only selecting perceptually dense grids.
```

如果 reviewer 追问为什么不把 full 20-CAV early fusion 当作主 baseline：

```text
Full 20-CAV early fusion assumes all vehicles can upload complete point clouds to a common fusion center, which violates the RSU-free and subchannel-limited setting considered by SGCP. We therefore report it as an upper reference and compare SGCP against V2V-only selective-sharing baselines under matched communication budgets.
```

## 后续补强

- 若论文版面允许，保留一个 small upper-reference table，单列 full early fusion、full late fusion 和 FullPerception-RSU。
- 主表使用 SGCP、random/MWS scheduler ablation、nearest/density/communication-aware selective sharing。
- 完整 41 帧 NS3-aware selective-sharing 仍需补做；当前只有 11 帧受限 5-subchannel 诊断结果。
- 若后续重新打开 CARLA，可导出带 RSU sensor 的真实 FullPerception-RSU 数据，但它仍应标注为 infrastructure-assisted upper reference。
