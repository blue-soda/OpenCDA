# SGCP Table 4 Parameter Sensitivity Candidate

更新时间：2026-07-19

源数据：`table4_parameter_sensitivity.csv`

## 主文建议

Table 4 不应把所有参数都写成同等强结论。当前 41 帧 dump 对不同参数的解释力不同：

- `rho_th`：可进主文。它直接改变 high-density grid 判定和点云划分，呈现清楚 AP-Mbps tradeoff。`rho_th=3` 相比 `rho_th=2` 有小幅 AP@0.5/AP@0.7 增益，payload 从 `56.08` 增至 `57.38 Mbps`。
- 子信道数：可进主文。5/10/20 ch 显示 PPS 可调度成员和 grid 数随资源变化，AP@0.7 从 `0.27` 到 `0.41`，payload 从 `28.91` 到 `73.98 Mbps`。
- 极低带宽 stress：可作为附录或正文一句话。20/40/80 MHz 在当前 dump 不敏感，但 0.1/0.5/1.0 MHz 会触发吞吐瓶颈，说明带宽约束代码有效。
- `N_max`：建议附录。它证明容量约束真实生效，但 AP 非单调，当前短场景不能支持“越大越好/越小越好”的强结论。
- `T_min^stab`：建议附录或负面结果。100--1000 ms 在当前 41 帧短序列上完全不敏感；如果主文必须回应稳定窗口，应说明该场景拓扑变化不足，并补更动态场景。
- `B_h`：建议附录。`B_h=3` 对 AP@0.7 有轻微提升但牺牲 AP@0.3，说明简单放宽 per-head RB 不是主方法改进方向。

## 主文 Table 4 最小候选

| Parameter | Setting | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `rho_th` | 1.0 | 0.76 | 0.72 | 0.34 | 51.31 | Lower payload, lower AP |
| `rho_th` | 2.0 | 0.79 | 0.75 | 0.37 | 56.08 | Low-budget candidate |
| `rho_th` | 3.0 | 0.79 | 0.76 | 0.38 | 57.38 | Better AP with modest payload increase |
| Channels | 5 | 0.56 | 0.53 | 0.27 | 28.91 | Strong resource bottleneck |
| Channels | 10 | 0.79 | 0.75 | 0.37 | 56.08 | Main low-budget channel setting |
| Channels | 20 | 0.80 | 0.76 | 0.41 | 73.98 | Higher localization AP, higher payload |

该最小表更稳：它只报告当前证据最清晰的两个参数，不把 `N_max` / `T_min^stab` 的弱结论硬塞进主文。

## Caption 草稿

Parameter sensitivity under the 20-CAV `v2xp_cluster_carla` dump. Aggregate AP is computed by the pooled OpenCOOD evaluator. `rho_th` controls the density threshold for grid-level raw LiDAR selection, while the number of subchannels controls how many intra-cluster uploads can be scheduled within each 100 ms cycle. Increasing communication resources improves high-IoU localization but raises raw LiDAR payload. Less sensitive parameters, including `N_max` and `T_min^stab`, are reported in the appendix because the current 41-frame sequence is too short to expose stable topology effects.
