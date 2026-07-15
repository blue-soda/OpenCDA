# Parameter and Utility Calibration Revision

本文档面向 P4 写作修订，集中回应审稿人对 `f(rho)` 标定、`rho_th`、`T_min^stab` 和 `N_max` 参数依据的质疑。

## 审稿意见映射

Reviewer 3:

> The calibration process for this core function is described too briefly, which constitutes a reproducibility concern.

> The value of this key parameter appears to be set arbitrarily, with no theoretical or experimental justification provided in the text.

Reviewer 4:

> The perception utility model relies heavily on empirical calibration of point cloud density specifically for the PointPillars detector. It is not clear how well this model would generalize to different types of sensors or different detection algorithms without constant re-calibration.

## `f(rho)` 标定过程

### 当前证据

当前已完成离线 density calibration：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_density_calibration --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --thresholds "0.5,1.0,2.0,3.0,4.0" --output-dir docs\doc_workspace\SGCP\artifacts\density_calibration_41f
```

结果：

| Metric | Value |
| --- | ---: |
| CAV-grid density samples | 788,020 |
| Nonzero grid ratio | 0.059794 |
| Nonzero density p90 | 1.400000 |
| Nonzero density p95 | 3.600000 |
| Nonzero density p99 | 13.255600 |
| Default `rho_th=2.0` selected grids | 3,383 |
| Ratio / nonzero grids | 0.071797 |

### main.tex 可写段落

```tex
We calibrate the density utility using the same grid reconstruction path as the SGCP replay. For each CAV and frame, LiDAR points are projected to the global grid map, and $\rho$ is measured as points per square meter in each 10 m grid. In the 41-frame CARLA dump, we collect 788,020 CAV-grid density samples. Non-empty grids account for 5.98% of all grid samples, with the 90th and 95th percentiles equal to 1.40 and 3.60 points/m$^2$, respectively. The default threshold $\rho_{\mathrm{th}}=2.0$ lies between these two percentiles and selects 7.18% of non-empty grids as high-density candidates. We then sweep $\rho_{\mathrm{th}}$ and report the AP/payload trade-off to avoid treating the density threshold as a universal constant.
```

### 泛化边界

论文中必须明确：

- `rho_th` depends on LiDAR resolution, grid size, point-cloud preprocessing, and detector backbone.
- 如果换 detector 或 sensor，应重新运行 density calibration。
- 当前实验支持 “calibrated for this setup”，不支持 “universal density utility”。

## `rho_th` 参数依据

### 当前 sweep

| `rho_th` | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.74 | 0.69 | 0.34 | 86,658.74 | lower payload, lower AP |
| 1.0 | 0.75 | 0.71 | 0.33 | 96,968.13 | lower payload than default |
| 2.0 | 0.77 | 0.73 | 0.35 | 109,415.48 | default trade-off |
| 3.0 | 0.77 | 0.73 | 0.37 | 113,689.69 | higher AP@0.7, higher payload |
| 4.0 | 0.77 | 0.74 | 0.37 | 115,754.73 | best AP here, highest payload |

### 论文口径

`rho_th=2.0` 不是 AP 最优点，而是通信-精度折中点：

- 与 `rho_th=0.5/1.0` 相比，AP 更高。
- 与 `rho_th=3.0/4.0` 相比，payload 更低。
- 位于当前非零 density p90 和 p95 之间，具有分布依据。

推荐写法：

```tex
We use $\rho_{\mathrm{th}}=2.0$ as the default because it lies between the 90th and 95th percentiles of non-empty grid densities and provides a balanced AP/payload trade-off. Larger thresholds slightly improve high-IoU AP in this dump but increase point-cloud payload, whereas smaller thresholds reduce payload at the cost of lower AP.
```

## `N_max` 参数依据

### 当前 sweep

| `N_max` | AP@0.5 | Avg. Cluster Size | Avg. Clusters | Full Candidate Skips | Singleton Ratio | Small-Cluster Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.74 | 1.95 | 10.29 | 12,534 | 0.053 | 1.000 |
| 3 | 0.71 | 2.65 | 7.59 | 7,894 | 0.146 | 0.206 |
| 4 | 0.73 | 3.33 | 6.00 | 4,065 | 0.000 | 0.187 |
| 5 | 0.71 | 3.33 | 6.00 | 1,142 | 0.000 | 0.317 |
| 6 | 0.71 | 3.33 | 6.00 | 0 | 0.000 | 0.317 |

### 论文口径

`N_max=4` 的优势不是单一 AP 最优，而是机制折中：

- 避免 `N_max=2` 带来的大量 small clusters。
- 避免 `N_max=5/6` 下 capacity constraint 基本失效。
- 无 singleton fragmentation，且仍有 3.12 个满簇/frame、99.15 次满簇候选跳过/frame，说明硬容量约束实际生效。
- 保持 cluster size 约 3.33，适合每簇头有限 RB 的 PPS 调度。

推荐写法：

```tex
We set $N_{\max}=4$ as a capacity-control parameter rather than a pure accuracy-tuning knob. In the current dump, $N_{\max}=2$ yields many small clusters, while $N_{\max}=5/6$ makes the capacity constraint mostly inactive. The default $N_{\max}=4$ produces no singleton clusters, keeps the average cluster size at 3.33, and still records 99.15 capacity-skipped candidate joins per frame, indicating that the constraint actively prevents oversized coalitions.
```

## `T_min^stab` 参数依据

### 当前 sweep

| `T_min^stab` (ms) | AP@0.5 | Reconfig. Count | Vehicle-Head Changes | Avg. Lifetime |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.73 | 11 | 76 | 6.65 |
| 300 | 0.73 | 11 | 76 | 6.65 |
| 500 | 0.73 | 11 | 76 | 6.65 |
| 700 | 0.73 | 11 | 76 | 6.65 |
| 1000 | 0.73 | 11 | 76 | 6.65 |

### 重要边界

当前 41 帧 dump 对 `T_min^stab` 不敏感，不能证明 500 ms 是最优值。论文若继续写 “500 ms robust choice”，需要补更动态场景。

更安全口径：

- `T_min^stab` 是 hysteresis / anti-oscillation control 参数。
- 当前短序列显示 100-1000 ms 均不改变 AP/reconfiguration，说明结果不是由该参数脆弱调出来的。
- 不能声称 500 ms 最优；只能称为与 10 Hz perception cycle 匹配的 conservative default。

推荐写法：

```tex
We treat $T_{\min}^{\mathrm{stab}}$ as a hysteresis parameter that prevents rapid coalition oscillation after a reconfiguration. In the current 41-frame dump, sweeping 100--1000 ms produces the same AP and reconfiguration count, indicating that the reported result is not sensitive to this parameter in the tested sequence. We therefore use 500 ms as a conservative default corresponding to five 10 Hz perception cycles, while acknowledging that more aggressive traffic dynamics require a separate sensitivity study.
```

## Rebuttal 答法

### `f(rho)` calibration

> We added a reproducible density-calibration protocol. We reconstruct the same 10 m LiDAR grids used by SGCP, measure points/m$^2$ for each CAV and frame, and report both the empirical density distribution and the AP/payload sweep over $\rho_{\mathrm{th}}$. The default value 2.0 is not treated as universal; it lies between the 90th and 95th percentiles of non-empty grid densities in our setup and should be recalibrated when the sensor, detector, or grid size changes.

### `T_min^stab`

> We agree that the original manuscript did not sufficiently justify the stability-window parameter. We now report a sweep from 100 ms to 1000 ms. In the current dump, AP and reconfiguration metrics are unchanged, so we no longer claim that 500 ms is optimal. Instead, we present it as a conservative hysteresis default spanning five sensing cycles and note that more dynamic scenes are needed to tune it aggressively.

### `N_max`

> We added a cluster-capacity sweep and capacity-pressure statistics. $N_{\max}=4$ avoids singleton fragmentation in the tested dump while still actively preventing oversized coalitions, as shown by 99.15 capacity-skipped candidate joins per frame. This clarifies how vehicles are handled when surrounding clusters are full: they remain in feasible coalitions or small clusters and still participate through inter-cluster late fusion.

## Target 状态

本文件完成 P4 中两项：

- 补充 `f(rho)` 标定过程和曲线。
- 补充 `T_min^stab`、`N_max`、`rho_th` 参数选择依据。
