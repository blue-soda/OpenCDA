# Local-to-Global Ablation Design

本文档对应 `target.md` 的 P0 任务：增加 local-to-global ablation，区分 partial sharing 与 LGCP 层次结构本身带来的收益。

## 背景

审稿意见认为 LGCP 的收益可能只是因为传输数据更少、并行度更高，而不是 local-to-global hierarchy 本身带来的结构性优势。因此实验必须拆开两个因素：

1. **Selective / partial sharing**：少传哪些数据。
2. **Local-to-global structure**：area partition、area-task group、leader local fusion、leader-to-RSU upload、RSU global aggregation、global view broadcast 是否带来额外收益。

如果只对比 full sharing baseline，审稿人仍会认为 baseline unfair。Ablation 必须加入 “selective sharing without LGCP hierarchy”。

## 实验问题

1. 在相同通信预算下，LGCP hierarchy 是否比简单 selective sharing 有更高 AP / recall。
2. 在相近 AP / recall 下，LGCP hierarchy 是否比 selective sharing 有更低 latency / packet count。
3. Leader local fusion 与 RSU global aggregation 分别贡献多少。
4. Scheduling 是否只是降低 latency，还是也影响 perception quality。

## Ablation 变体

| ID | 方法 | Area partition | Area group selection | Leader local fusion | RSU aggregation | Scheduling | Selective sharing | 目的 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | Full sharing | No | No | No | Optional | No | No | 上界质量 / 最大通信量 |
| A1 | Random selective sharing | No | No | No | Optional | No | Yes | 随机 partial sharing 下界 |
| A2 | Confidence selective sharing | No | No | No | Optional | No | Yes | 强 communication-aware baseline |
| A3 | Area partition only | Yes | No | No | RSU direct fusion | No | Yes | 只看 area decomposition |
| A4 | Area group without leader | Yes | Yes | No | RSU direct aggregation | No | Yes | 区分 group selection 与 local fusion |
| A5 | LGCP without scheduling | Yes | Yes | Yes | Yes | No | Yes | 只看 hierarchy，不看调度 |
| A6 | Full LGCP | Yes | Yes | Yes | Yes | Yes | Yes | 完整方法 |

最低可进入 rebuttal 的组合是：

- A0 Full sharing
- A2 Confidence selective sharing without hierarchy
- A5 LGCP without scheduling
- A6 Full LGCP

这四个变体可以直接回应 “merely not sending everything” 的质疑。

## 评价指标

### Perception Quality

- AP@0.3 / AP@0.5 / AP@0.7
- Area-level Recall@0.5 / Recall@0.7
- Area-level AP from `lgcp_area_confidence_eval.py`
- Quality preservation under same packet budget

### Communication / System

- Total transmitted bytes
- Packet count
- Feature slice count
- Control-plane bytes
- End-to-end latency
- Leader max-load / load imbalance
- RSU aggregation load

### Fairness Constraints

必须至少提供一种公平对比口径：

1. **Same packet budget**：A2 / A5 / A6 使用相同 packet count 或 byte budget。
2. **Same AP target**：比较达到同等 AP@0.7 所需的通信量。
3. **Same latency budget**：比较相同 latency 约束下的 AP / recall。

推荐主文报告 Same packet budget，附录报告 Same AP target。

## 当前仓库可复用基础

| 需求 | 当前基础 | 缺口 |
| --- | --- | --- |
| Full sharing / early fusion quality | `offline_inference.py` 可跑 full set early fusion | 需要固定 agent subset / packet budget |
| Area slicing quality | `lgcp_area_confidence_eval.py --with-inference` | 已能输出 area AP / recall |
| Confidence score | `area_records.csv` 有 density / detector score | 还需要 feature-level confidence 更贴近论文 |
| Greedy group selection | `lgcp_greedy_gap_eval.py` 有 offline greedy / oracle | 还未生成 actual LGCP subset inference |
| RSU entity | `lgcp_carla` 已有 RSU dump `-1` | RSU global aggregation 机制未实现 |
| Network / latency | cluster / NS3 管线已有基础 | LGCP area-task scheduling 未实现 |

## Offline 第一阶段实现方案

第一阶段先做 “perception-only ablation”，不依赖 NS3：

1. 输入：`D:\Data\Carla\2026_07_15_02_33_21`
2. 固定 ego：`1`
3. 固定帧：优先 11 帧 `000060` 到 `000080`
4. 生成候选 agent subsets：
   - A0：所有 20 CAV + RSU
   - A1：随机选 K 个 agent
   - A2：按全局 confidence top-K
   - A3/A4/A5：按 area confidence 为每个 area 选 K 或 `Delta_g` group，再合并成 subset
5. 对每个 subset 调用 OpenCOOD inference。
6. 使用 `lgcp_area_confidence_eval.py` 的 area slicing 统计 area-level quality。
7. 记录 estimated packet count：`sum(area selected agents)` 与 unique agent count 两种口径。

该阶段无法完整证明 local fusion / RSU aggregation，但能先回答：area-aware group selection 是否比 non-hierarchical selective sharing 更会选 agent。

### Current Multiseed Offline Status

2026-07-17 已完成 random-only multiseed 扩展：

```text
docs/doc_workspace/LGCP/experiments/ablation/20260717_lgcp_carla_offline_subset_multiseed_11f
```

该结果主要用于稳定 random partial-sharing baseline。Deterministic 的 confidence / communication-aware / area-aware 方法仍复用同一 11 帧结果。当前观察：

- random AP@0.7 mean/std：budget=5 为 `0.163843 ± 0.026394`，budget=10 为 `0.328993 ± 0.038178`。
- `comm_aware_topk` 仍高于 `area_aware_union`。
- 因此 offline 第一阶段已经证明 random baseline 不够强，但尚未证明 LGCP hierarchy 本身优于 strong communication-aware selective sharing。

## Online / Mechanism 第二阶段实现方案

第二阶段需要真正实现 LGCP pipeline：

1. RSU 收集 CAV pose / direction / confidence。
2. RSU 对 ROI 做 area partition。
3. RSU 为每个 area 选择可重叠 CAV group。
4. 每个 area 指定 leader。
5. group members 将 area-specific feature slice 发送给 leader。
6. leader local fusion 后上传 area result 给 RSU。
7. RSU aggregation 得到 global view。
8. RSU broadcast global view / selected perception result。

此阶段才能完整比较 A5 / A6 的 hierarchy 与 scheduling。

## 输出目录

```text
docs/doc_workspace/LGCP/experiments/ablation/
  20260715_lgcp_carla_l2g_design/
    config.yaml
    ablation_plan.csv
    notes.md
```

后续真实实验建议：

```text
docs/doc_workspace/LGCP/experiments/ablation/
  202607xx_lgcp_carla_offline_subset_ablation/
    config.yaml
    subset_quality.csv
    budget_quality_summary.csv
    area_quality_summary.csv
    notes.md
```

## 表格模板

### Main Ablation Table

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Area Recall@0.5 | Bytes | Packets | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full sharing | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Confidence selective | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LGCP w/o scheduling | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Full LGCP | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Same Budget Table

| Budget | Selective AP@0.7 | LGCP AP@0.7 | Gain | Selective latency | LGCP latency |
| --- | --- | --- | --- | --- | --- |
| 25% packets | TBD | TBD | TBD | TBD | TBD |
| 50% packets | TBD | TBD | TBD | TBD | TBD |
| 75% packets | TBD | TBD | TBD | TBD | TBD |

## 判定标准

可认为 local-to-global ablation 有效的最低标准：

- A2 confidence selective sharing 是强 baseline，而非 random / full-sharing-only。
- A5 或 A6 在相同 packet budget 下 AP / recall 高于 A2。
- A6 在相近 AP 下 latency 或 packet count 低于 A5 / A2。
- 若 A5/A6 质量不优于 A2，则论文必须收缩 claim：LGCP 主要是 system scheduling / latency optimization，而非 perception-quality improvement。

## 风险

- Offline subset inference 只能近似 selective sharing，不能完全模拟 feature-slice local fusion。
- OpenCOOD early fusion 要求 ego reference，非 ego subset 可能仍需 ego as coordinate anchor。
- RSU `-1` 是否应作为 sensor provider 需要单独验证；否则 RSU 先只作为 controller。
- 如果 A2 已经非常强，LGCP 的增益可能主要体现在 latency / control structure，而不是 AP。
