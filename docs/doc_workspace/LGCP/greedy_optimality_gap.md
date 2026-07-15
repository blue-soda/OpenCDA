# Greedy Group Selection Optimality Gap Design

本文档对应 `target.md` 的 P0 任务：增加 greedy group selection 的 small-scale optimality gap 实验，用 exhaustive search 或 ILP 做小规模最优参考。

## 背景

论文 `C:\Workspace\icdcs-paper\LGCP\conference_101719.tex` 的 selection algorithm 包含两部分：

1. 对每个 perception area `a_i`，按 Eq. (4) 的 confidence incremental threshold `Delta_g` 贪心选择 CAV group。
2. 对所有 group 按 group size 降序分配 leader，每次选择当前 load 最小的 CAV。

审稿人质疑该算法只是 heuristic，没有 approximation guarantee。当前最稳妥的 revision 路径不是强行声称理论保证，而是补一个 small-scale optimality gap：在可枚举的小规模问题上比较 greedy 与 oracle / ILP 的目标值差距。

## 实验目标

1. 量化 greedy group selection 相比 small-scale optimal 的 objective gap。
2. 分离两个误差来源：
   - group member selection gap；
   - leader assignment / load balancing gap。
3. 说明论文中 “approximate solution” 应改写为 “efficient heuristic with empirical optimality gap”。
4. 为 rebuttal 提供可复现表格，而不是只用直觉解释。

## 问题形式

对一个小规模 instance：

- area set：`A = {a_1, ..., a_N}`
- CAV set：`V = {v_1, ..., v_M}`
- 单车 area confidence：`c[i,j] = F_i({v_j})`
- group confidence：`F_i(S)`，默认使用 Eq. (2) noisy-or
- area-specific feature size：`B`
- leader decision：`y[i,j] = 1` 表示 `v_j` 是 area `a_i` 的 leader

论文中的主目标可近似写成：

```text
maximize avg_i F_i(S_i) / (t_delta + schedule_cost(S, y))
```

small-scale 实验建议拆成三个 objective，降低一次性实现难度：

| Objective | 作用 |
| --- | --- |
| `O1_confidence_only` | 只比较 group selection 对 avg confidence 的影响 |
| `O2_confidence_minus_size` | 加入 group size / packet count penalty |
| `O3_confidence_latency_ratio` | 近似论文目标，加入 leader load 和 schedule cost |

其中：

```text
O1 = mean_i F_i(S_i)
O2 = mean_i F_i(S_i) - lambda * sum_i |S_i|
O3 = mean_i F_i(S_i) / (t_delta + max_load(y) + packet_count(S, y))
```

第一版优先实现 `O1` 和 `O2`，随后接入 `O3`。

## Small-Scale Instance 构造

从 `eq2_composition_validation.md` 的 subset records 或 area confidence records 中抽样：

| 设置 | 值 |
| --- | --- |
| `M` CAV 数 | 4, 5, 6, 7 |
| `N` area 数 | 3, 5, 8 |
| confidence 来源 | `area_records.csv` 的 per-agent confidence |
| group confidence | Eq. (2) noisy-or，也可替换为 best validated rule |
| Delta_g | `[0.05, 0.075, 0.1, 0.125]` |
| max group size | 可选，`K in {2,3,4}`，用于控制枚举规模 |

instance 采样规则：

1. 只保留 `gt_count > 0` 或有预测框的 area，避免空 area 主导结果。
2. 每个 timestamp 取局部候选 CAV，优先包含对这些 area confidence 非零的 CAV。
3. 对每组 `(M, N)` 采样至少 30 个 instance。

## Greedy Baseline

严格复现论文算法：

1. 对每个 area `a_i`，按 `c[i,j]` 降序遍历 CAV。
2. 如果加入 `v_j` 后 `F_i(S_i union {v_j}) - F_i(S_i) >= Delta_g`，则加入 group。
3. 得到所有 `S_i` 后，按 `|S_i|` 降序处理 area。
4. 对每个 group，在 `S_i` 中选择当前 load 最小的 CAV 作为 leader。
5. 更新 `L_j += |S_i| * B`。

记录：

- selected groups；
- leader assignment；
- avg confidence；
- total selected links；
- max leader load；
- objective value。

## Oracle / Optimal Reference

### Exhaustive Search

适合 `M <= 6, N <= 5`。

1. 为每个 area 枚举所有非空 subset，必要时限制 `|S_i| <= K`。
2. 枚举所有 area 的 subset combination。
3. 对每个 combination，求最优 leader assignment：
   - 如果 area 数小，直接枚举每个 group 内 leader；
   - 或用 dynamic programming / branch-and-bound 最小化 max load。
4. 计算 objective，取最优。

### ILP / MILP

适合稍大规模，后续再做。

变量：

- `x[i,j] in {0,1}`：CAV `j` 是否属于 area `i` 的 group。
- `y[i,j] in {0,1}`：CAV `j` 是否为 area `i` 的 leader。
- `L[j] >= 0`：leader load。
- `Lmax >= L[j]`。

约束：

```text
y[i,j] <= x[i,j]
sum_j y[i,j] = 1
L[j] = sum_i y[i,j] * group_size_i * B
Lmax >= L[j]
```

由于 Eq. (2) 对 subset 是非线性的，第一版 ILP 可以采用预枚举 subset 的 set-packing formulation：

- 对每个 area `i` 预枚举 candidate subset `s`。
- 变量 `z[i,s]` 表示 area `i` 选择 subset `s`。
- `sum_s z[i,s] = 1`。
- objective 中使用预计算的 `F_i(s)` 和 `|s|`。

这样比直接线性化 noisy-or 更简单，也更适合 small-scale evidence。

## Gap Metrics

对每个 instance 计算：

```text
absolute_gap = objective_opt - objective_greedy
relative_gap = (objective_opt - objective_greedy) / max(abs(objective_opt), eps)
quality_gap = mean_conf_opt - mean_conf_greedy
cost_gap = cost_greedy - cost_opt
load_gap = max_load_greedy - max_load_opt
```

论文中建议报告：

- mean / median relative gap；
- p90 relative gap；
- worst-case gap；
- greedy runtime vs exhaustive / ILP runtime；
- 不同 `M, N, Delta_g` 下的趋势。

## 输出文件

统一放在：

```text
docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/
```

建议结构：

```text
greedy_optimality_gap/
  20260715_design/
    config.yaml
    instance_records.csv
    gap_summary.csv
    runtime_summary.csv
    notes.md
```

`instance_records.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `instance_id` | instance 编号 |
| `scenario_id` | 来源场景 |
| `timestamp` | 来源帧 |
| `M` | CAV 数 |
| `N` | area 数 |
| `Delta_g` | greedy threshold |
| `objective_name` | O1/O2/O3 |
| `greedy_value` | greedy objective |
| `optimal_value` | oracle objective |
| `relative_gap` | 相对 gap |
| `greedy_mean_conf` | greedy 平均 confidence |
| `optimal_mean_conf` | oracle 平均 confidence |
| `greedy_cost` | greedy cost |
| `optimal_cost` | oracle cost |
| `greedy_runtime_ms` | greedy 时间 |
| `optimal_runtime_ms` | oracle 时间 |

## 实现计划

1. 先实现纯 Python 离线工具 `opencda/tools/lgcp_greedy_gap_eval.py`。
2. 输入优先使用 `area_records.csv`，若已有 `subset_records.csv` 则直接复用 subset quality。
3. 第一版只做 confidence surrogate objective：
   - `O1_confidence_only`
   - `O2_confidence_minus_size`
4. 第二版接入 leader assignment 和 `O3_confidence_latency_ratio`。
5. 通过 `results.md` 记录 smoke result，再决定是否扩大 instance 数。

## 论文写法建议

如果结果显示 gap 较小：

```text
Although the proposed group selection is heuristic, a small-scale exhaustive comparison shows that it achieves near-optimal objective values with substantially lower runtime.
```

如果 gap 不稳定：

```text
We do not claim a theoretical approximation guarantee. Instead, LGCP uses the greedy rule as a practical online heuristic, and we report its empirical optimality gap under small-scale settings.
```

避免继续使用未证明的 “approximate solution” 或 “near optimal” 表述，除非 gap 实验真的支持。

## 风险

- 如果 confidence surrogate 本身相关性弱，optimality gap 只能证明优化了 surrogate，不能证明提升 detection quality。
- O3 的 schedule cost 和真实 NS3 latency 之间可能仍有差距，需要在 latency 实验中单独校准。
- 全枚举随 `M` 和 `N` 指数增长，必须限制 small-scale 范围并报告规模。
