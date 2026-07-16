# LGCP Update Frequency / Stale Assignment Sensitivity

## 目标

回应审稿意见中 vehicle mobility、update frequency 和 stale assignment 可能影响 partition / redundancy removal 的问题。

本实验不重新运行 CARLA，而是复用已有 11 帧 area confidence / quality CSV。对每个 target frame，使用前 `lag_steps` 个 update 的 area confidence 去预测当前 frame 的 area quality，从而近似较低 update frequency 或 stale RSU assignment。

## 工具

```text
opencda/tools/lgcp_stale_assignment_eval.py
```

输入：

- `area_records.csv`
- `area_quality.csv`

输出：

- `stale_assignment_records.csv`
- `stale_topk_overlap.csv`
- `stale_assignment_summary.csv`
- `config.yaml`
- `notes.md`

## 实验设置

- scenario：`2026_07_15_02_33_21`
- frames：11
- quality：`recall_05`
- confidence：`confidence_noisy_or` from `density_linear`
- lags：`0 / 1 / 2 / 3` frame steps
- top-k：`40` areas

## 当前结果

| Lag steps | Samples | Noisy-or vs recall@0.5 Spearman | Top-40 Jaccard mean | Top-40 Jaccard min |
| --- | ---: | ---: | ---: | ---: |
| 0 | 354 | 0.584992 | 1.000000 | 1.000000 |
| 1 | 321 | 0.527720 | 0.911095 | 0.777778 |
| 2 | 289 | 0.529556 | 0.857818 | 0.777778 |
| 3 | 257 | 0.447925 | 0.805484 | 0.666667 |

## Interpretation

- One- or two-frame stale confidence still preserves much of the area ranking in this 11-frame smoke.
- Three-frame stale assignment shows visible degradation: Spearman drops from `0.584992` to `0.447925`, and top-40 overlap mean drops to `0.805484`.
- This supports using a short assignment TTL or event-driven reassignment in dynamic scenes.

## Output Directory

```text
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_stale_assignment_11f/
```

## Paper Boundary

可用于 rebuttal / revision 的保守说法：

```text
We evaluate stale area-confidence reports by using confidence from previous
frames to rank current-frame areas. In the current 11-frame smoke, one- and
two-frame stale reports preserve most of the ranking, while three-frame stale
assignment starts to degrade the confidence-to-recall correlation. This
motivates a short assignment TTL or event-driven reassignment in dynamic scenes.
```

不能直接声称：

```text
LGCP is insensitive to arbitrary update intervals or vehicle speed.
```

当前结果只验证已有 11 帧 dump 上的 temporal staleness proxy，尚未显式改变车辆速度或交通密度。
