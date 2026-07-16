# LGCP Manuscript Language Audit

## 目的

本文档记录当前论文 TeX 中需要修正的 stage 编号和 heuristic / approximate 表述问题。目标是让正文语言与当前证据一致：

- 不暗示尚未证明的 approximation guarantee；
- 明确 selection 和 scheduling 是 practical online heuristics；
- 用 empirical optimality gap 支撑启发式算法，而不是理论保证；
- 修正 latency section 中 stage 编号不一致。

源文件：

```text
C:\Workspace\icdcs-paper\LGCP\conference_101719.tex
```

本文档只记录建议，不直接修改论文源文件。

## Stage Numbering Audit

### 问题定位

| TeX line | Current text / issue | Suggested fix |
| --- | --- | --- |
| 268 | `In the first stage...` initiation and CAV information report | 保留为 Stage 1 |
| 273 | `In the second stage...` task assignment broadcast | 保留为 Stage 2 |
| 277 | `In the third stage...` member-to-leader upload, leader fusion, leader-to-RSU upload | 可保留为 Stage 3，但建议拆清楚 member upload / local fusion / leader upload 是同一 local-to-global aggregation stage 的子步骤 |
| 292 | `In the fifth stage...` RSU broadcasts global view | 改为 `In the fourth stage...` |
| 297 | `\sum_{k=1}^{4} t_i` | 改为 `\sum_{k=1}^{4} t_k`，使求和索引与 latency term 一致 |
| 305-308 | denominator uses `\sum_{i=1}^{4} t_i` | 改为 `\sum_{k=1}^{4} t_k`，避免与 area index `i` 混淆 |
| 491 | `two-stage process` means algorithmic process, not latency stages | 保留，但建议写成 `two algorithmic modules` 避免和 latency stage 混淆 |

### 推荐四阶段命名

| Stage | Name | Latency term | Description |
| --- | --- | --- | --- |
| Stage 1 | Coordination initialization and CAV report | `t_1` | RSU initiation broadcast + CAV pose / confidence report |
| Stage 2 | Area-task assignment broadcast | `t_2` | RSU broadcasts area-task groups, leaders, and scheduling control |
| Stage 3 | Local area fusion and leader upload | `t_3` | members upload area-specific feature slices, leaders fuse, leaders upload area results |
| Stage 4 | Global-view broadcast | `t_4` | RSU broadcasts global perception view / update |

### Suggested Replacement Text

Replace the current “fifth stage” sentence with:

```text
In the fourth stage, the RSU broadcasts the aggregated global view to all CAVs,
and the expected transmission latency is denoted as
$t_4 = D_G / R_t$, where $D_G$ is the length of the global-view message.
```

Replace “two-stage process” in the overall algorithm paragraph with:

```text
The algorithm consists of two algorithmic modules: Algorithm 1 constructs the
area-task groups and selects leaders, while Algorithm 2 schedules the
transmission and fusion order.
```

## Heuristic / Approximate Language Audit

### High-Risk Current Phrases

| TeX line | Current phrase | Risk | Suggested replacement |
| --- | --- | --- | --- |
| 145 | `We propose a greedy algorithm...` | Acceptable, but add evidence boundary | `We propose an efficient greedy heuristic... and evaluate its empirical gap against exhaustive references.` |
| 245 | `approximately estimate the value of P_acc` | Acceptable if validated | `use area confidence as a measurable proxy for P_acc`; cite correlation study |
| 259 | `P_acc ... \approx ... F_i` | Acceptable as model assumption | Add sentence: `We validate this proxy empirically in Sec. X.` |
| 269, 279, 294 | `latency is approximately...` | Acceptable analytical model | Prefer `modeled as` / `estimated as` to distinguish from algorithm approximation |
| 355 | `derive an approximate solution` | High risk: implies approximation guarantee | `derive an efficient heuristic solution` |
| 491 | `two-stage process` | Conflicts with four latency stages | `two algorithmic modules` |

### Recommended Terminology

Use:

- `heuristic solution`
- `efficient online heuristic`
- `empirical optimality gap`
- `small-scale exhaustive reference`
- `no theoretical approximation guarantee is claimed`
- `surrogate objective`
- `area-confidence proxy`

Avoid unless proven:

- `approximate solution`
- `near-optimal`
- `approximation algorithm`
- `performance guarantee`
- `bounded approximation ratio`
- `optimal scheduling`

## Suggested Paragraph for Algorithm Section

```text
The joint area-task grouping, leader assignment, and transmission scheduling
problem is combinatorial and must be solved online within each perception
cycle. Therefore, LGCP adopts an efficient greedy heuristic rather than a
theoretical approximation algorithm. The heuristic first selects CAVs whose
incremental area-confidence gain exceeds a threshold and then assigns leaders
to balance local fusion load. We do not claim a formal approximation ratio;
instead, we quantify the empirical optimality gap on small-scale instances
using exhaustive references.
```

## Suggested Paragraph for Evaluation Section

```text
To assess the quality of the heuristic, we compare it with exhaustive
references on small-scale instances. For group-member selection, the mean
relative gap is 4.90% when Delta_g is 0.05 and 6.35% when Delta_g is 0.075
under the density-distance confidence proxy. For leader load balancing, the
mean relative gap is 12.88% at Delta_g=0.05 and 2.27% at Delta_g>=0.075.
These results provide empirical evidence for the online heuristic, but do not
constitute a theoretical approximation guarantee.
```

## Rebuttal Wording

```text
We agree that the previous wording could be interpreted as claiming an
approximation guarantee. In the revision, we replace "approximate solution"
with "efficient heuristic solution" and explicitly state that no theoretical
approximation ratio is claimed. We further add a small-scale exhaustive
comparison to report empirical optimality gaps for both group-member selection
and leader assignment.
```

## Required Paper Edits Checklist

- [ ] Change `In the fifth stage` to `In the fourth stage`.
- [ ] Rename the four latency stages consistently in text and figure.
- [ ] Replace `\sum_{i=1}^{4} t_i` with `\sum_{k=1}^{4} t_k` where `i` also indexes area.
- [ ] Replace `derive an approximate solution` with `derive an efficient heuristic solution`.
- [ ] Add an explicit sentence that no approximation guarantee is claimed.
- [ ] Add empirical gap numbers from `results.md`.
- [ ] Change `two-stage process` in the algorithm overview to `two algorithmic modules`.
- [ ] Ensure `approximately` is used only for analytical latency/proxy modeling, not algorithmic optimality.

## Evidence Already Available

- `docs/doc_workspace/LGCP/greedy_optimality_gap.md`
- `docs/doc_workspace/LGCP/results.md`
- `docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260715_lgcp_carla_greedy_gap_density_distance/`
- `docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260715_lgcp_carla_greedy_gap_with_leader/`
