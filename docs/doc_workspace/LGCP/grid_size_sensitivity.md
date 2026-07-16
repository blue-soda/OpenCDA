# LGCP Grid / Area Size Sensitivity

## 目标

回应审稿意见中 “key gains may depend on a single area/grid setting” 的问题，先在已有 `lgcp_carla` 11 帧 dump 上做单场景 grid-size sensitivity smoke。

本实验不重新运行 CARLA；只复用已导出的 OPV2V-style dump 和 OpenCOOD offline inference。

## 工具更新

`opencda/tools/lgcp_area_confidence_eval.py` 新增：

```text
--grid-size-x
--grid-size-y
```

用于在离线评估中覆盖 `lgcp_carla.yaml` 的 ROI grid size，而不修改场景配置。

## 实验设置

共同配置：

- scenario：`2026_07_15_02_33_21`
- frames：11
- fusion method：`early`
- ROI：`280m x 80m`
- density threshold：`2.0`

测试 grid：

| Grid size | 说明 |
| --- | --- |
| `5m x 3m` | finer grid |
| `10m x 6m` | paper / LGCP default |
| `20m x 12m` | coarser grid |

## 当前结果

| Grid size | Records | Active areas | Area-frame noisy-or vs recall@0.5 Spearman | Area-acc noisy-or vs AP@0.5 Spearman | Area-acc score_mean vs AP@0.5 Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `5m x 3m` | 46993 | 1101 | 0.475952 | 0.213836 | 0.363570 |
| `10m x 6m` | 21418 | 337 | 0.570407 | 0.411840 | 0.402059 |
| `20m x 12m` | 8386 | 94 | 0.458975 | 0.233766 | 0.472727 |

## Interpretation

- Default `10m x 6m` grid gives the strongest area-frame recall ranking in this smoke: noisy-or confidence vs recall@0.5 Spearman `0.570407`.
- Finer `5m x 3m` grid produces many more active areas, but accumulated AP ranking becomes weaker, likely because per-area GT / prediction samples are sparse.
- Coarser `20m x 12m` grid has fewer active areas and fewer accumulated AP samples; detector `score_mean` ranks AP better than noisy-or in this setting.
- Current result supports using `10m x 6m` as a reasonable default, but it is still a single-scenario smoke and should not be overclaimed as a final sensitivity study.

## Output Directories

```text
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_grid_sensitivity_5x3_11f/
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_grid_sensitivity_10x6_11f/
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_grid_sensitivity_20x12_11f/
```

## Paper Boundary

可用于 rebuttal / revision 的保守说法：

```text
We also perform a single-scenario grid-size sensitivity smoke test. The default
10m x 6m grid provides the strongest area-frame confidence-to-recall ranking
among the tested 5m x 3m, 10m x 6m, and 20m x 12m settings. We will extend this
analysis to more seeds in the final revision.
```

不能直接声称：

```text
LGCP is fully insensitive to area size.
```

当前只能说 default grid 在已有 11 帧 smoke 中更稳定。
