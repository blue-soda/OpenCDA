# SGCP Failure Diagnostics

更新时间：2026-07-17

本文档记录 SGCP 主表 AP 偏低时的对象级诊断。目标不是替代 `results.md` 的主表，而是把漏检原因拆成可操作的工程问题：带宽、分簇、区块选择和跨簇晚期融合。

## Diagnostic Entry

诊断工具：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_failure_diagnostics --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --object-diagnostics-csv docs\doc_workspace\SGCP\artifacts\object_diag_sgcp_spatial_rho3_10ch_41f.csv --output-dir docs\doc_workspace\SGCP\artifacts\failure_diag_spatial_rho3_10ch_41f
```

输出文件：

- `vehicles.csv`：每帧 CAV 坐标、yaw、速度、cluster head/member、原始点数、感知 grid 数。
- `clusters.csv`：每帧 cluster head、成员、成员距离。
- `schedules.csv`：每条 PPS 上传链路的 receiver、sender、subchannel、selected grid、点数、bytes。
- `gt_objects.csv`：每个 GT 的 world/ego 坐标、最近 CAV、最近 head、所在 grid、能感知该 grid 的 CAV、覆盖该 grid 的调度链路、覆盖点数、full reference / SGCP match 标记。
- `summary.json`：聚合统计。

诊断列说明：

- `object_grid_id`：GT center 所在全局 10 m grid。
- `scheduled_covering_links`：上传链路中包含该 `object_grid_id` 的 sender > receiver 列表。
- `scheduled_covering_point_count`：这些 covering links 在该 grid 中上传的点数总和。
- `nearest_head_covering_point_count`：发给最近 cluster head 的 covering 点数。
- `nearest_cav_object_grid_points`：最近 CAV 在该 grid 中实际拥有的点数。若该值高而 `nearest_head_covering_point_count` 低，说明资源调度没有把关键局部点云送到最相关 receiver。

注意：covering link 是 grid-level proxy，不等于目标物体点级分割。一个 10 m grid 内可能同时含有道路、遮挡物和多个目标，因此后续若要进一步精确，应补 object-aware point association。

## Current Evidence

对照结果：

| Variant | Channels / BW | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP `spatial_diverse` | 5ch / 20 MHz | 0.56 | 0.53 | 0.27 | 14,815,408 | 28.91 |
| SGCP `spatial_diverse,rho_th=3` | 10ch / 20 MHz | 0.79 | 0.76 | 0.38 | 29,405,296 | 57.38 |
| SGCP `spatial_diverse` | 20ch / 20 MHz | 0.80 | 0.76 | 0.41 | 37,912,544 | 73.98 |
| Full 20-CAV early reference | unconstrained | 0.85 | 0.83 | 0.48 | 60,838,528 | 118.71 |

10ch / `rho_th=3` 对象级结果：

| Metric | Value |
| --- | ---: |
| GT rows in diagnostic CSV | 653 |
| Full-reference detected but SGCP missed rows | 111 |
| CAV-body-like missed rows | 61 |
| Non-CAV missed rows | 50 |
| Missed rows where nearest CAV uploaded anywhere | 21 |
| Missed rows where nearest CAV did not upload | 90 |
| Missed rows with any scheduled covering link | 110 |
| Missed rows with covering link to nearest head | 47 |
| Missed rows covered only by other cluster heads | 63 |
| Missed rows with no scheduled covering grid | 1 |

Persistent missed objects:

| Object | Type proxy | Missed frames | Main position / area | Diagnostic |
| --- | --- | ---: | --- | --- |
| `337` | CAV body, CAV 1 | 34 | around `(8.00, -30.31)`, grid `0_-3` | nearest head is usually `1`; object grid has many points at nearest CAV, but `to_nearest_head=0` in all 34 missed rows |
| `444` | non-CAV | 17 | around `(7.55, -17.80)`, grid `0_-2` | covering reaches nearest head, but nearest CAV is not selected; likely sender/point quality issue |
| `401` | CAV body, CAV 12 | 12 | around `(25-32, 3-4)`, grids `2_0/3_0` | often covered only by other heads; one frame has no covering grid |
| `374` | CAV body, CAV 7 | 11 | around `(8-9, 18)`, grid `0_1` | nearest head receives sparse points despite nearest CAV having many local points |
| `406` | non-CAV | 10 | around `(30.6, -8.1)`, grid `3_-1` | mixed; sometimes correct head gets points, sometimes only other heads |

Point-count buckets for 111 missed rows:

| Bucket | Rows | Interpretation |
| --- | ---: | --- |
| Covered only by other cluster heads | 63 | The target grid appears in some upload, but not at the nearest/relevant head. This is mainly resource scheduling / receiver assignment, not raw bandwidth alone. |
| Nearest head got dense points but no final box | 35 | The correct head receives at least 30 points in the target grid, but no matching final detection appears. This points to detector quality, object-level point association, or late fusion/NMS as secondary causes. |
| Nearest head got sparse object-grid points | 12 | The correct head receives the target grid but with too few points. This suggests sender choice / view quality is poor even when the grid id is selected. |
| No scheduled covering grid | 1 | Pure grid-selection miss is rare in 10ch/rho3, but likely much more severe in 5ch. |

## Four-Cause Diagnosis

### 1. Bandwidth Too Low

Evidence strength: strong for the 5ch stress setting.

5ch / 20 MHz drops from 10ch `0.79/0.76/0.38` to `0.56/0.53/0.27`. Payload is roughly halved, and missed GT count is higher in the earlier 5ch diagnostic. This means low bandwidth is a real AP limiter.

However, bandwidth is not the full explanation for 10ch/rho3. In 10ch, 110/111 missed rows have at least one covering link somewhere, so the remaining misses are mostly about which receiver/sender gets the useful points and how final boxes are fused.

### 2. Poor Clustering

Evidence strength: medium.

Misses concentrate around nearest heads `1`, `12`, `4`, `11`, and `20`. Some CAV-body objects are assigned to clusters where the nearest/relevant head does not receive the object grid, even though another cluster head does. This means cluster topology and receiver assignment contribute to failures.

But full-cluster upload and 20ch results are substantially better, so clustering itself is not completely broken. The current interpretation is: coalition formation gives a workable hierarchy, but the downstream scheduling does not always preserve target coverage for the most relevant head.

### 3. Resource Scheduling / Block Selection

Evidence strength: strongest remaining bottleneck.

The decisive pattern is that `nearest_cav_object_grid_points` is often high while `nearest_head_covering_point_count` is zero or tiny. Examples:

- Object `337`, frame `000062`: GT at `(8.000, -30.314)`, grid `0_-3`, nearest CAV `1`, nearest head `1`, nearest CAV grid points `1453`, but nearest-head covering points `0`.
- Object `401`, frame `000062`: GT at `(25.433, 4.033)`, grid `2_0`, nearest CAV `12`, nearest head `12`, nearest CAV grid points `2057`, but nearest-head covering points `0`.
- Object `374`, frame `000080`: GT at `(8.372, 18.164)`, grid `0_1`, nearest CAV `7`, nearest head `4`, nearest CAV grid points `1487`, but nearest-head covering points `7`.

This points to resource scheduling needing target-aware receiver/sender protection rather than only density/diversity over grids. The next algorithmic step should prioritize object/target-aware candidate generation or quality-weighted target coverage:

- protect persistent high-contribution CAVs when they uniquely cover missed regions;
- score sender-view quality for target grids, not just grid density;
- add a fallback that forces at least one high-quality same-cluster upload for persistent missed target grids;
- keep the subchannel budget fixed so the improvement remains fair.

### 4. Inter-Cluster Late Fusion

Evidence strength: secondary.

35/111 missed rows have dense points delivered to the nearest head but still no final matched box. That cannot be solved purely by channel count or grid selection. It may involve:

- detector failing with partial / single-view object points;
- CAV-body GT protocol issues, where the object is a CAV body and self/peer observation semantics matter;
- late-fusion/NMS suppressing or failing to preserve a head-local prediction.

Earlier NMS sweeps showed default `0.15` is better than `0.05/0.30`, so this is not a simple NMS-threshold fix. The next useful debug is to dump per-head pre-NMS boxes around the persistent missed GT objects and verify whether a box exists before inter-cluster fusion.

## Next Actions

1. Add target-grid diagnostics to `offline_inference --object-diagnostics-output` or a companion tool: for every missed GT, record per-head pre-NMS best IoU, score, and source CAVs.
2. Implement target-aware scheduling probe: persistent missed grids get a same-cluster, high-quality covering sender if one exists, replacing the lowest utility scheduled grid/link under the same channel budget.
3. Add CAV-body GT audit: mark GT objects whose centers overlap a CAV body, and decide whether paper AP should include them under the same convention as OPV2V/OpenCOOD.
4. Re-run 10ch/rho3 and 20ch with the target-aware probe; accept only if AP improves without increasing Mbps.

## Target-Aware PG Follow-Up

已新增 `target_aware_potential_game`，不再作为离线后处理修补，而是资源调度器内的两阶段算法：

1. 保留原 PotentialGame 的 sender/subchannel best-response。
2. 在 allocator 内部用 target-aware multi-view utility 重选每条链路的 grid action。

41 帧 20MHz/10ch/rho3 对照：

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Payload | Missed Rows | Covered Only By Other Heads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `potential_game + spatial_diverse` | 0.79 | 0.76 | 0.38 | 29,405,296 | 111 | 63 |
| `target_aware_potential_game` | 0.80 | 0.76 | 0.39 | 31,069,968 | 106 | 56 |

结论：target-aware PG 确实减少了此前诊断出的主失败桶，但代价是 payload 增加。下一步应在该算法内部加入 byte-aware grid utility 或 target-aware point cap，目标是在保住 `0.80/0.76/0.39` 附近 AP 的同时降低 Mbps。
