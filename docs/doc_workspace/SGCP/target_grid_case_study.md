# SGCP Target-Grid Case Study

更新时间：2026-07-17

本文记录 20MHz / 10ch / `rho_th=3` 下若干持续漏检 GT 的 grid-level 诊断。目标是回答三个问题：漏检真值框落在哪个 grid，为什么当前调度没有选到该 grid，下一版算法应该如何修改而不脱离 SGCP 的 potential-game 叙事。

## 诊断工具与输入

新增工具：

```text
opencda/tools/sgcp_grid_miss_analysis.py
```

输入来自当前 target-aware PG 的对象级漏检诊断：

```text
docs/doc_workspace/SGCP/artifacts/failure_diag_target_aware_pg_10ch_rho3_41f/gt_objects.csv
```

代表性命令：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_grid_miss_analysis --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --failure-gt-csv docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f\gt_objects.csv --resource-allocation target_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --max-objects 8 --max-rows-per-object 3 --output-csv docs\doc_workspace\SGCP\artifacts\grid_miss_analysis_target_aware_pg_top8.csv
```

对比新算法：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_grid_miss_analysis --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --failure-gt-csv docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f\gt_objects.csv --resource-allocation object_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --max-objects 8 --max-rows-per-object 3 --output-csv docs\doc_workspace\SGCP\artifacts\grid_miss_analysis_object_aware_pg_fill_top8.csv
```

## 代表性漏检

### Object 438, Frame 000068

- GT grid：`3_0`
- nearest CAV：12
- nearest head：4
- cluster：`4;9;12`
- target-aware PG：CAV12 在 `3_0` 有 424 点，candidate=1，rank=1，score=2.790442，但未被调度；实际调度 CAV9，CAV9 在 `3_0` 为 0 点。
- 原因：旧 sender utility 更偏向累计覆盖面积，CAV9 覆盖大量普通 grid，挤掉了对关键目标 grid 更强的 CAV12。
- object-aware PG：同 RB sender refinement 后，head4 的 link 从 CAV9 切换为 CAV12，选中 grid 列表前缀包含 `3_0`、`2_0` 等目标区域。

### Object 401, Frame 000066

- GT grid：`2_0`
- nearest CAV/head：12/12
- cluster：`4;7;12`
- target-aware PG：CAV4 在 `2_0` 有 891 点，candidate=1，rank=4，但未被调度；实际调度 CAV7，CAV7 在该 grid 只有 7 点且不是 candidate。
- 原因：调度只比较 member 总收益，未保护能为 head 的近身/盲区目标提供强外部视角的 member。
- object-aware PG：CAV4 被调度并选中 `2_0`，说明 target-grid 覆盖链路被打通。

### Object 350, Frame 000084

- GT grid：`1_-2`
- nearest CAV：8
- nearest head：1
- cluster：`1;2;8;11`
- target-aware PG：CAV8 在 `1_-2` 有 3371 点，candidate=1，rank=1，但未被调度；实际调度 CAV2/CAV11，仅提供 29/27 点。
- 原因：最佳视角 CAV8 被低质量但覆盖更多普通 grid 的 sender 挤出，导致送到 head 的 target-grid 点云稀疏。
- object-aware PG：CAV8 被调度并选中 `1_-2`，同一 RB 预算下提高了目标 grid 的视角质量。

### Object 337, Frame 000062

- GT grid：`0_-3`
- nearest CAV/head：1/1
- cluster：`1;2;8;11`
- target-aware PG：head 自身 CAV1 在该 grid 有 1453 点，但其他外部 CAV 的该 grid 没被当作 target-like candidate；CAV8 有 138 点但未被调度，CAV2/CAV11 被调度但未选择该 grid。
- 原因：原逻辑把 head 已经高密度的 grid 视为足够覆盖，忽略了自车近身/盲区目标仍可能需要 peer view 的情况。
- object-aware PG：将 head 高密度且 peer 有中等密度的 grid 纳入 multi-view confirmation candidate；当前版本可以让 CAV11/CAV8 等 peer view 进入候选，但仍需继续调优 sender diversity，避免某些帧中最佳 CAV8 被其他 sender 替代。

## 新算法设计：Object-Aware Potential Game

新增入口：

```text
object_aware_potential_game
object_aware_pg
oapg
```

设计原则：

- 保留 SGCP 原有 cluster head、RB/subchannel、deadline 和 channel conflict 约束，不绕过 PPS。
- action utility 不再只累计 grid density，而是优先近似目标原型：connected high-density grids、head weak grids、head 已高密度但仍需要 peer view 的 multi-view confirmation grids。
- sender 选择从“覆盖更多 grid”改为“关键目标原型峰值 + 原型组件 + 适度 coverage fill”。
- grid selection 采用 target-first + background-fill：先选择每个 object prototype 的代表 grid，再用普通正密度 grid 填满 RB 容量，避免只发局部目标块导致 AP 下降。
- sender refinement 在不增加 RB/subchannel 的前提下，用同一 RB 替换为 object-prototype utility 更高的 member，并避免同一 head 内重复选择同一个 sender。

当前代码：

```text
opencda/core/clustering/algorithms/resource_allocation/object_aware_potential_game.py
opencda/core/clustering/algorithms/resource_allocation/builder.py
```

## 当前实验结论

11 帧快速实验命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-constrained --resource-allocation object_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\object_aware_pg_diverse_10ch_rho3_11f_trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\object_diag_object_aware_pg_diverse_10ch_rho3_11f.csv
```

结果：

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Avg. source CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Object-aware PG, diverse sender guard | 11 | 0.74 | 0.69 | 0.30 | 8,209,376 | 2.64 | 73.48 |

解释：

- 机制层面已经修复了若干明确的“最佳视角 rank 很高但未被调度”的失败链路。
- 11 帧 AP 尚未超过当前主表候选，因此不能直接写入论文主表。
- AP@0.7 下降说明仅让目标 grid 进入融合还不够，后续需要检查 detector pre-NMS 输出、late fusion/NMS 和目标周边上下文 grid 是否被保留。

## 下一步

- 对 object-aware PG 做 41 帧完整评估和对象级诊断，确认 11 帧现象是否泛化。
- 将 sender refinement 改成每个 head 最多替换一个 RB，或加入 detector-quality proxy，防止目标峰值过强时牺牲 source diversity。
- 对已选中 target grid 仍漏检的对象 dump per-head pre-NMS boxes，区分 detector 未出框与 late fusion/NMS 丢框。
- 继续保留 `target_aware_potential_game` 作为当前较稳主表候选；`object_aware_potential_game` 暂作为下一代机制分支。
