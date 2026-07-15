# SGCP Paper Revision Plan

本文档把当前实验/机制进展转化为论文修订动作，优先回应审稿意见中关于 topology trigger、100 ms 周期、utility calibration、baseline fairness 和 game-theoretic convergence 的问题。

## 1. Topology Trigger 表述矛盾

### 问题

`C:\Workspace\icdcs-paper\SGCP\main.tex` 中存在两处容易被审稿人抓住的矛盾：

- System model / cycle description：cluster formation “triggered only when significant topology changes are detected, rather than every cycle”。
- Formation algorithm：coalition formation “procedure repeats every cycle `T_c` to track dynamics”。

这会让读者误以为 SGCP 一边事件触发，一边每 100 ms 全量重构。

### 建议统一口径

统一为：

> every cycle observes state and runs PPS/resource scheduling; cluster structure is only updated when a topology/stability trigger fires or a periodic guard expires.

也就是：

- 每个 100 ms cycle 都更新 beacon、点云 density、PPS 调度和感知融合。
- Cluster membership / leader election 不应每周期无条件重构。
- Topology trigger 决定 `NO_CHANGE / LOCAL_REPAIR / RECLUSTER`。
- 当前在线 gate 是默认关闭 first version；论文最终若不做在线回归，应把它写成机制设计 + 离线统计，而不是强称所有在线实验都启用了该 gate。

### main.tex 替换建议

位置：cycle phase 描述附近，当前句子：

```tex
This process is triggered only when significant topology changes are detected (e.g., vehicle entry/exit or large deviation in motion), rather than every cycle.
```

建议替换为：

```tex
At every cooperation cycle, CAVs refresh their beacon-level state and perception-density metadata. The cluster membership, however, is updated only when the current coalition becomes infeasible or sufficiently suboptimal, e.g., due to neighbor-set changes, head/member disconnection, relative-motion risk, link-quality degradation, or a periodic guard. Otherwise, SGCP keeps the previous coalition and only updates PPS scheduling and perception fusion.
```

位置：formation algorithm 段，当前句子：

```tex
Resulting coalitions remain robust over $T_{\min}^{\mathrm{stab}}$, and the procedure repeats every cycle $T_c$ to track dynamics.
```

建议替换为：

```tex
Resulting coalitions are retained across cycles unless an event trigger indicates that the current partition is infeasible or sufficiently suboptimal. The $T_{\min}^{\mathrm{stab}}$ term acts as a hysteresis window that suppresses frequent migrations, while PPS is still recomputed every cycle to adapt to time-varying perception density and channel resources.
```

### Rebuttal 答法

> We clarified that SGCP is not intended to perform full coalition reformation at every 100 ms cycle. Each cycle updates beacons, density metadata, and PPS scheduling, while cluster membership is retained unless a topology/stability trigger fires. We now define the trigger set, including neighbor-set changes, head/member disconnection, relative-motion risk, link-quality degradation, utility drop, and a periodic guard. This resolves the ambiguity between event-triggered reconfiguration and periodic sensing/scheduling.

## 2. Real-Time Feasibility

### 审稿意见

Reviewer concern: multiple game-theoretic iterations may be too expensive under a strict 100 ms cycle.

### 新证据

当前 41 帧离线结果：

| Metric | Value |
| --- | ---: |
| PPS converged frames | 41 / 41 |
| Avg. PPS iterations | 3.00 |
| Max PPS iterations | 3 |
| Avg. scheduled links / frame | 10.00 |
| Avg. selected grids / frame | 523.90 |
| Avg. RA runtime | about 39 ms in latest replay |
| Control overhead | 4.56 KB / frame |

### main.tex 修订建议

当前 real-time 段过于笼统：

```tex
Cluster formation converges within 3 iterations per adjustment, and PPS algorithm converges in 3–4 iterations on average.
```

建议改为更可复现：

```tex
In the 41-frame CARLA dump, PPS converges before the maximum iteration limit in all frames, requiring 3.00 iterations on average and at most 3 iterations. The scheduler produces 10.00 upload links and 523.90 selected grids per frame under the default 10-subchannel setting. The measured resource-allocation runtime is approximately 39 ms per frame in the offline replay, and the estimated control metadata is 4.56 KB per frame, less than 1% of the point-cloud payload.
```

边界：如果后续要写 “full pipeline comfortably fits within 100 ms”，需要同时报告 OpenCOOD inference runtime。当前只足以支撑 PPS/cluster replay runtime，不应把深度模型推理耗时含混带过。

## 3. `f(rho)` 标定与泛化

### 审稿意见

Reviewer concern: density utility calibration is too brief and may overfit PointPillars / sensor setup.

### 新证据

当前 `f_rho_calibration.md` 已给出：

- 788,020 CAV-grid samples。
- Nonzero grid ratio = 5.98%。
- Nonzero density p90 = 1.40, p95 = 3.60。
- Default `rho_th=2.0` selects 7.18% of nonzero grids。
- `rho_th` AP/payload sweep already in `results.md`。

### 论文口径

建议不要声称 `rho_th=2.0` 是通用常数。应写为：

> `rho_th` is detector/sensor/grid-size dependent. We calibrate it using dumped CARLA frames and report sensitivity results. For the current LiDAR and 10 m grid setup, `rho_th=2.0` lies between the 90th and 95th percentile of non-empty grid densities and provides a communication-accuracy trade-off.

## 4. Baseline Fairness

### 审稿意见

Reviewer concern: evaluation compares mostly against centralized or simple scheduling baselines; need decentralized V2V-only baselines.

### 当前可写证据

`baseline_fairness.md` 和 `results.md` 已有 same-budget V2V-only selective baselines：

| Baseline | AP@0.3/0.5/0.7 | Note |
| --- | --- | --- |
| SGCP inter-cluster LF | 0.77 / 0.73 / 0.35 | Main SGCP |
| Selective nearest | 0.76 / 0.73 / 0.37 | Same cluster/head evaluation path |
| Selective density | 0.77 / 0.74 / 0.39 | Strong baseline |
| Selective communication-aware | 0.78 / 0.75 / 0.40 | Strongest current baseline, higher payload |
| NS3 RLC-complete aware, 11 frames | 0.68 / 0.63 / 0.27 | Link-feasible selective baseline |

### 论文口径

必须避免写 “SGCP beats all decentralized baselines on AP”。当前更稳的主张是：

> SGCP trades a small amount of AP against explicit coalition stability, PPS channel feasibility, lower payload than the strongest selective baseline, and NS3-verifiable subchannel behavior.

## 5. Game-Theoretic Convergence

### 风险

当前 `main.tex` conclusion 写：

```tex
Both algorithms are guaranteed to converge to Nash-stable equilibria via monotonic potential functions
```

这对当前代码过强。`potential_game_conditions.md` 已指出 PPS 当前更准确是 potential-guided constrained best-response scheduling。

### 建议替换

```tex
The coalition formation stage is guided by a stability-aware utility and empirically produces stable partitions under the capacity constraint. The PPS stage is implemented as a potential-guided constrained best-response scheduler over a finite action set, with hard feasibility checks for RB capacity and SINR. In our replay, PPS converges in all frames within three iterations. A fully general exact-potential proof with replacement dynamics is left as a formal extension.
```

若论文必须保留 exact potential game，则需要补：

- 显式 `Phi`。
- 单边 action replacement。
- `Delta Phi >= 0` 日志。
- 完整 late utility 多簇聚合。

## P4 完成状态

本轮已完成 “修正 topology change 才触发 与 每个周期重复 的表述矛盾” 的写作计划。其他 P4 项仍需继续：

- Related work 重写。
- Novelty 增强。
- `f(rho)` 标定过程写入论文。
- `T_min^stab`、`N_max`、`rho_th` 参数选择依据。
- FullPerception baseline 公平性说明。
- 端到端实时性实验表。
