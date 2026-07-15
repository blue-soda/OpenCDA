# Eq. (2) Composition Validation Design

本文档对应 `target.md` 的 P0 任务：设计 Eq. (2) 组合规则验证实验，比较 product rule、max、mean、sum、top-k 等组合方式。

## 背景

论文 `C:\Workspace\icdcs-paper\LGCP\conference_101719.tex` 中 Eq. (2) 为：

```text
F_i(V_i) = 1 - product_{v_k in V_i}(1 - F_i({v_k}))
```

该公式等价于 noisy-or / independent success probability composition：如果多个 CAV 对同一区域的感知成功事件近似独立，则组合 confidence 会随 CAV 数量增加而单调提升，并呈现边际收益递减。

审稿意见的关键质疑是：论文没有证明这个独立性近似合理，也没有证明 Eq. (2) 相比更简单的 max / mean / sum / top-k 规则更能预测真实 area-level AP / recall。

## 要回答的问题

1. Eq. (2) 与真实 fused area-level recall / AP 的相关性是否为正。
2. Eq. (2) 是否比 max、mean、sum-clipped、top-k mean 更适合作为 group selection 的目标函数。
3. Eq. (2) 的边际收益递减趋势是否与实际融合收益一致。
4. 如果 Eq. (2) 不是最优规则，是否可以用 calibrated noisy-or 或 learned calibration 替代，并在论文中收缩原公式的 claim。

## 输入数据

该实验依赖 `area_confidence_validation.md` 中设计的中间表：

```text
docs/doc_workspace/LGCP/experiments/area_confidence/<run_id>/area_records.csv
```

最低需要字段：

| 字段 | 含义 |
| --- | --- |
| `scenario_id` | 场景 id |
| `timestamp` | 帧编号 |
| `area_id` | area / grid id |
| `agent_id` | CAV / RSU id |
| `confidence` | 单 agent 对该 area 的 confidence |
| `gt_count` | area 内 GT 数 |
| `recall_03/05/07` | 单 agent 或 subset 的 area recall |
| `ap_03/05/07` | 单 agent 或 subset 的 accumulated area AP |

如果只导出 per-agent 记录，还需要额外运行 subset inference，得到 `(scenario_id, timestamp, area_id, subset_id)` 的 fused quality。

## 候选组合规则

给定某个 area `a_i` 和 CAV subset `S = {v_1, ..., v_m}`，单车 confidence 为 `c_k = F_i({v_k})`。

| 名称 | 定义 | 解释 |
| --- | --- | --- |
| `product_noisy_or` | `1 - prod(1 - c_k)` | 当前 Eq. (2) |
| `max` | `max(c_k)` | 只信任最强 CAV |
| `mean` | `mean(c_k)` | 线性平均质量 |
| `sum_clipped` | `min(sum(c_k), 1)` | 简单累加并截断 |
| `top2_mean` | `mean(top 2 c_k)` | 降低弱 CAV 噪声影响 |
| `top3_mean` | `mean(top 3 c_k)` | 检查更大局部组 |
| `softmax_weighted` | `sum softmax(beta*c_k) * c_k` | 可调的偏强者规则 |
| `calibrated_noisy_or` | `1 - prod(1 - alpha*c_k)` | 校准 Eq. (2) 的增长速度 |

第一阶段不引入复杂学习模型，只网格搜索 `alpha` 和 `beta`：

- `alpha`: `[0.25, 0.5, 0.75, 1.0]`
- `beta`: `[1, 2, 4, 8]`

## Subset 采样策略

完全枚举 20 CAV 的所有 subset 不现实。第一版按 area-frame 采样：

1. 对每个 `(timestamp, area_id)`，选出对该 area 有非零 confidence 的 CAV。
2. 如果候选 CAV 数量 `M <= 6`，枚举所有非空 subset。
3. 如果 `M > 6`，采样：
   - top-1 / top-2 / top-3 / top-4 by confidence；
   - random subsets，每个 size 采样 10 个；
   - diversity subsets，优先选择空间位置分散的 CAV。
4. 对每个 subset 记录组合 confidence 和 fused area-level quality。

这样可以同时支持两个目标：

- 验证 composition rule 是否预测 fusion quality。
- 为后续 greedy group selection optimality gap 提供小规模 subset oracle。

## Fused Quality 计算

建议复用 `OPV2VFrameDataset.load_frame(..., cav_ids=subset)`，让 OpenCOOD 在指定 subset 上推理：

1. 固定 ego CAV，例如 `1`。
2. 对 subset 加入 ego；如果 ego 不在 subset 中，保留 ego 作为坐标参考，但不把 ego features 计入组合规则时需显式记录。
3. 得到 `pred_box_tensor`、`pred_score`、`gt_box_tensor`。
4. 按 `area_confidence_validation.md` 的 area 切分方法，计算 area-level recall/AP。
5. 写出 `subset_records.csv`。

注意：如果 OpenCOOD 的 early/intermediate fusion 必须包含 ego raw data，则需要在 `subset_policy` 字段中说明 ego 是否为 mandatory sensor provider，避免论文解释时混淆。

## 输出文件

统一放在：

```text
docs/doc_workspace/LGCP/experiments/eq2_composition/
```

建议结构：

```text
eq2_composition/
  20260715_lgcp_carla_smoke/
    config.yaml
    subset_records.csv
    rule_correlation_summary.csv
    marginal_gain_summary.csv
    selected_examples.csv
    notes.md
```

`subset_records.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `scenario_id` | 场景 id |
| `timestamp` | 帧编号 |
| `area_id` | area id |
| `subset_id` | CAV subset 标识 |
| `subset_size` | CAV 数量 |
| `agent_ids` | CAV ids |
| `product_noisy_or` | Eq. (2) 分数 |
| `max` | max 分数 |
| `mean` | mean 分数 |
| `sum_clipped` | sum-clipped 分数 |
| `top2_mean/top3_mean` | top-k 分数 |
| `calibrated_noisy_or_alpha_*` | 校准 noisy-or 分数 |
| `gt_count` | area GT 数 |
| `recall_03/05/07` | fused recall |
| `ap_03/05/07` | fused AP |

## 评价指标

### 预测质量相关性

对每个规则计算：

- Spearman correlation with `recall_05`
- Spearman correlation with `ap_05`
- Pearson correlation with `recall_05`
- Calibration bin monotonicity

优先报告 Spearman，因为 group selection 主要依赖排序。

### Group Selection Regret

对每个 area-frame，给定 subset size 或通信预算：

1. 用某个组合规则选择最佳 subset。
2. 用真实 fused recall/AP 选择 oracle subset。
3. 计算 regret：

```text
regret = quality(oracle_subset) - quality(rule_selected_subset)
relative_regret = regret / max(quality(oracle_subset), eps)
```

该指标比纯相关性更贴近审稿人关心的 “Eq. (2) 是否能驱动正确 group selection”。

### Marginal Gain Consistency

Eq. (2) 隐含边际收益递减。验证方式：

1. 按 confidence 从高到低逐步加入 CAV。
2. 记录 Eq. (2) marginal gain。
3. 记录真实 recall/AP marginal gain。
4. 比较两者的 Kendall tau 或趋势一致率。

## 论文判定标准

建议采用如下决策：

- 如果 `product_noisy_or` 在 Spearman / regret 上接近最优规则，则保留 Eq. (2)，并补充 “empirically validated surrogate”。
- 如果 `calibrated_noisy_or` 明显更好，则将 Eq. (2) 改为带校准项的组合规则，或把原公式作为默认近似。
- 如果 `max/top-k` 明显更好，则论文应避免声称多 CAV 独立增益，改为 quality-aware top-k group selection。
- 如果所有规则相关性都弱，则必须回到 area confidence 定义本身，优先修正 confidence surrogate，而不是继续强化 group selection。

## 实现计划

1. 在 `opencda/tools/lgcp_area_confidence_eval.py` 中预留 per-agent area confidence 输出。
2. 新增 `opencda/tools/lgcp_eq2_composition_eval.py`，读取同一数据集并采样 subsets。
3. 对每个 subset 调用 OpenCOOD inference，输出 `subset_records.csv`。
4. 汇总 `rule_correlation_summary.csv` 和 `marginal_gain_summary.csv`。
5. 将 smoke result 写入 `results.md`，再决定是否扩大到多 seed / 多模型。

## 风险

- subset inference 成本较高，第一版必须限制 frame 数、area 数和 subset 数。
- area 内 GT 太少时 AP 波动大，需要按 area 或 frame 累积。
- 如果 ego 必须参与 inference，需要把 “ego as coordinate anchor” 和 “ego as sensor provider” 分开记录。
- Eq. (2) 的独立性假设在空间高度重叠的 CAV 上可能过强，calibrated noisy-or 是较稳妥的替代路径。
