# SGCP `f(rho)` Calibration

本文档记录 SGCP 点云密度效用函数 `f(rho)` 的离线标定协议、当前结果和论文写作口径。

## 标定目标

SGCP 使用 grid-level LiDAR point density `rho` 估计某个感知网格是否值得上传。当前实现中的基础效用函数为：

```text
f(rho; rho_th) = sigmoid(rho - rho_th)
```

其中 `rho` 为 points / m^2，`rho_th` 来自 lidar 配置的 `density_threshold`。默认 `rho_th=2.0`。

标定需要回答三个问题：

1. 当前场景中 `rho` 的经验分布是什么。
2. `rho_th=2.0` 选中多少 high-density grid。
3. `rho_th` 改变时，通信量和 AP 如何变化。

## 可复现入口

密度分布统计不需要启动 CARLA 或 OpenCOOD：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_density_calibration --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --thresholds "0.5,1.0,2.0,3.0,4.0" --output-dir docs\doc_workspace\SGCP\artifacts\density_calibration_41f
```

输出文件：

- `global_density_summary.csv`：全局 `rho` 分布和非零网格条件分布。
- `frame_density_summary.csv`：逐帧 point/grid 分布摘要。
- `threshold_summary.csv`：每个 `rho_th` 下的 high-density grid 数量、比例和平均 `f(rho)`。
- `f_rho_curve.csv`：按 `rho` bin 采样的 `sigmoid(rho-rho_th)` 曲线。

## 当前 41 帧结果

数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，默认 grid size = 10 m。

| Metric | Value |
| --- | ---: |
| Total CAV-grid samples | 788,020 |
| Nonzero grid samples | 47,119 |
| Nonzero grid ratio | 0.059794 |
| Mean density, all grids | 0.050816 |
| P95 density, all grids | 0.030000 |
| P99 density, all grids | 0.830000 |
| Max density, all grids | 34.410000 |
| Mean density, nonzero grids | 0.849855 |
| P50 density, nonzero grids | 0.130000 |
| P75 density, nonzero grids | 0.460000 |
| P90 density, nonzero grids | 1.400000 |
| P95 density, nonzero grids | 3.600000 |
| P99 density, nonzero grids | 13.255600 |

阈值统计：

| `rho_th` | High-Density Grids | Ratio / All Grids | Ratio / Nonzero Grids | Mean `f(rho)` |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 11,232 | 0.014253 | 0.238375 | 0.383800 |
| 1.0 | 6,481 | 0.008224 | 0.137545 | 0.275282 |
| 2.0 | 3,383 | 0.004293 | 0.071797 | 0.124640 |
| 3.0 | 2,587 | 0.003283 | 0.054904 | 0.051639 |
| 4.0 | 2,192 | 0.002782 | 0.046521 | 0.021290 |

## 与 AP / Payload 的关系

`rho_th` 敏感性实验见 `results.md`。在当前 dump 上：

- `rho_th=0.5/1.0` 选中更多低密度网格，payload 更低但 AP 降低。
- `rho_th=2.0` 位于非零密度 p90 和 p95 之间，筛出约 7.18% 非零网格，是当前默认通信-精度折中点。
- `rho_th=3.0/4.0` 更接近非零密度 p95，AP@0.7 略高但 payload 增加。

因此论文中可以将 `rho_th=2.0` 写为当前 detector / LiDAR / grid size 下的经验折中点，而不是跨场景通用常数。

## 论文写作口径

建议表述：

> We calibrate the density utility on the dumped CARLA frames by rebuilding the same grid-level LiDAR density used by SGCP. For each CAV and frame, `rho` is measured as the number of LiDAR points per square meter in a 10 m grid. The default `rho_th=2.0` lies between the 90th and 95th percentile of non-empty grid densities in the current scenario and selects 7.18% of non-empty grids as high-density candidates. We then sweep `rho_th` and report the resulting AP/payload trade-off, rather than treating `rho_th` as a universal constant.

## 仍需补充

- 重新导出不同交通密度 / CAV 数 / 天气或遮挡条件下的数据，复核 `rho` 分布是否稳定。
- 若更换 detector、LiDAR 通道数、grid size 或点云预处理，应重新运行本工具并更新 `rho_th`。
- 若论文需要严格拟合曲线，可进一步把 `rho` bin 与 per-grid detection recall/IoU 绑定，而不仅是 AP/payload sweep。
