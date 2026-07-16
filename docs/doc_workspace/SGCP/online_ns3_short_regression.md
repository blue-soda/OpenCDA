# Online CARLA + NS3 Short Regression

更新时间：2026-07-17

本文档记录 SGCP 真实在线 CARLA + NS3 短回归的执行协议和当前准备状态。目标是验证：CARLA tick、OpenCDA `NetworkManager.current_time_slot/current_sim_time`、NS3 `sync_request/sync_ack` 三者使用同一时间基准；OpenCDA PPS 指定的 `sc_start/sc_num` 能进入 NS3 manual subchannel 行为；短程在线仿真不会因时间流速不一致提前断开。

## 当前状态

本轮尚未启动图形 CARLA 长流程，只完成了在线回归前的轻量检查：

```powershell
conda run -n opencda python -m py_compile opencda\core\networking\network_manager.py opencda\core\networking\ns3_co_simulation\bridge\carla_ns3_bridge.py opencda\tools\offline_ns3_replay.py test\test_network_time_sync.py

conda run -n opencda python -c "from test.test_network_time_sync import test_network_time_slot_matches_carla_fixed_delta,test_multiple_network_slots_track_carla_time; test_network_time_slot_matches_carla_fixed_delta(); test_multiple_network_slots_track_carla_time(); print('network_time_sync tests passed')"
```

结果：

```text
network_time_sync tests passed
```

进程检查显示当前无 CARLA / NS3 / 5556 / 5557 残留进程。

## 重要前提

- `opencda.py --network` 只启用 OpenCDA 侧 bridge 和 `NetworkManager.use_ns3`，不会自动启动 WSL ns-3。
- 因此在线 CARLA+NS3 回归必须按顺序启动：
  1. WSL ns-3；
  2. Windows CARLA；
  3. OpenCDA `v2xp_cluster_carla --network`。
- CARLA 进程至多保留一个。启动前必须检查已有 `CarlaUE4`。
- `OPENCDA_ONLINE_TICKS` 应设置为有限 tick，避免无人值守长跑。

## 推荐短回归命令

### 1. 检查残留进程

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "ps -ef | grep -E 'ns3|vanet/main|scratch/vanet|CarlaUE4' | grep -v grep || true; ss -ltnp | grep -E ':5556|:5557' || true"
```

### 2. 启动 ns-3

建议先用 8--12 秒 `simTime` 覆盖 35--80 个 0.05 s CARLA tick，并留足 NS3 drain 时间。

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=12.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
```

需保存 stdout 到：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short\ns3_stdout.log
```

### 3. 启动 CARLA

```powershell
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe" -WindowStyle Hidden
```

确认 2000 端口可用：

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -InformationLevel Quiet
```

### 4. 运行 OpenCDA SGCP + NS3

```powershell
$env:OPENCDA_CLUSTERING_CONFIG = "opencda/scenario_testing/config_yaml/networking_clustering_topology_gate.yaml"
$env:OPENCDA_ONLINE_TICKS = "35"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
Remove-Item Env:\OPENCDA_CLUSTERING_CONFIG
Remove-Item Env:\OPENCDA_ONLINE_TICKS
```

需保存 stdout 到：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short\opencda_stdout.log
```

## 验证项

### 时间同步

OpenCDA log 中应出现递增的同步时间，例如：

```text
sync_with_ns3: sync successful, carla_time=0.0500, ns3_time=0.0500
sync_with_ns3: sync successful, carla_time=0.1000, ns3_time=0.1000
```

验收条件：

- CARLA time 与 NS3 time 差值不超过 `1e-3`。
- 不出现连续 `Sync timeout`。
- `NetworkManager.current_sim_time = tick_count * fixed_delta_seconds`，无旧版 `/5` 缩放漂移。

### Subchannel 语义

NS3 stdout / parsed trace 中应确认：

- 合法 `sc_start/sc_num` request 进入 manual scheduler。
- 无冲突、带宽范围内 request 能到达 application callback / RLC complete。
- 超出 `targetSubchannels` 的 request 应在 bridge/manual scheduler 侧拒绝，不能绕回默认随机调度。

### 感知链路

OpenCDA stdout/log 中应确认：

- `CLUSTER_SYNC` 或 cluster head/member 日志存在。
- CP counter > 0。
- online run 正常退出并写出 evaluation output。

## 当前结论边界

离线 NS3 replay 已经验证 coverage-aware SGCP 10ch `rho_th=3` 的 110/110 scheduled request application/RLC complete，且 5-subchannel exposed regression 能正确拒绝 out-of-window request。本在线短回归的新增价值不是重新替代离线主表，而是证明真实 CARLA tick 驱动下，OpenCDA bridge 与 NS3 不再发生时间流速不一致。
