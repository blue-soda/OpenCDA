# Online Topology Gate Regression

更新时间：2026-07-16

本文档记录真实 CARLA 在线仿真中打开 `enable_topology_trigger_gate` 的短回归结果。该回归不启动 NS3，目标是确认在线 `ClusteringV2XManager` 能读取 gate 配置、输出 cluster trigger 日志，并完成一次有限 tick 的感知评估。

## 代码与配置

新增专用配置：

```text
opencda/scenario_testing/config_yaml/networking_clustering_topology_gate.yaml
```

关键配置：

```yaml
clustering:
  cluster_interval: 4
  enable_topology_trigger_gate: true
  topology_periodic_guard: 5

resource_allocation:
  algorithm: "potential_game"
```

`v2xp_cluster_carla.py` 新增两个环境变量入口：

- `OPENCDA_CLUSTERING_CONFIG`：向 `vehicle_base.v2x` 和 `traffic_vehicle_base.v2x` 注入 clustering config path。
- `OPENCDA_ONLINE_TICKS`：在线非 dump 模式下运行固定 tick 后自动 `stop()`，默认 `0` 时保持原来的无限运行行为。

## 回归命令

启动 CARLA：

```powershell
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe"
```

确认端口：

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -InformationLevel Quiet
```

短回归：

```powershell
$env:OPENCDA_CLUSTERING_CONFIG = "opencda/scenario_testing/config_yaml/networking_clustering_topology_gate.yaml"
$env:OPENCDA_ONLINE_TICKS = "35"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug
Remove-Item Env:\OPENCDA_CLUSTERING_CONFIG
Remove-Item Env:\OPENCDA_ONLINE_TICKS
```

## 日志路径

本轮原始日志：

```text
docs\doc_workspace\SGCP\artifacts\online_topology_gate\online_gate_stdout.log
opencda\log\opencda_20260716_090705.log
evaluation_outputs\v2xp_cluster_carla_2026_07_16_09_07_11\log.txt
```

注意：`.log` 文件受仓库 `.gitignore` 影响，不纳入提交；关键结果已整理到本文档。

## 结果

在线仿真完成并正常退出：

- Exit code：0
- CARLA：已启动并在回归后关闭
- NS3：未启动
- Online ticks：35
- CP counter：8
- Fusion method：early
- AP@0.3 / AP@0.5 / AP@0.7：`0.84 / 0.82 / 0.69`

Cluster trigger 日志统计：

| Trigger event | Count |
| --- | ---: |
| `recluster reason=initial` | 1 |
| `recluster reason=neighbor_set_change` | 1 |
| `recluster reason=head_member_unreachable` | 3 |
| `skip reason=no_topology_change` | 0 |

Cluster sync 观察：

- 初始阶段只有 ego CAV：`[(1, [1])]`。
- 交通 CAV manager 创建完成后形成 6 个 cluster。
- 后续多次 `head_member_unreachable` 触发 recluster，但 cluster membership 基本保持不变。

## 结论

本轮真实 CARLA 回归确认：

- `v2xp_cluster_carla` 可以通过 `OPENCDA_CLUSTERING_CONFIG` 打开 `enable_topology_trigger_gate`。
- 在线 `ClusteringV2XManager` 已输出 `CLUSTER_TRIGGER` 和 `CLUSTER_SYNC` 日志。
- 有限 tick 在线仿真可正常结束并产出 CP AP。

同时，本轮没有观察到 `skip reason=no_topology_change`。原因不是 gate 没生效，而是当前在线场景使用默认 35 m V2X communication range，cluster formation 后持续触发 `head_member_unreachable` hard condition。因此，该回归只能证明 gate 接入和 trigger logging 生效，不能证明“无事件周期被跳过”的收益。

## 后续建议

- 若要证明 skip/reduced reconfiguration，应增加第二组在线回归：提高 CAV `communication_range` 或选取更静态片段，使 head-member reachable 后观察 `skip reason=no_topology_change`。
- 当前论文中可写为：online gate 已接入并通过 CARLA smoke regression；稳定性收益仍主要由离线 topology-trigger 统计和机制设计支撑。
- 不建议在论文中声称该 35-tick 在线回归已经证明 gate 降低重构次数。
