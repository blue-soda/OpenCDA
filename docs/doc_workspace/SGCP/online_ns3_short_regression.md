# Online CARLA + NS3 Short Regression

更新时间：2026-07-17

本文档记录 SGCP 真实在线 CARLA + NS3 短回归的执行协议和当前准备状态。目标是验证：CARLA tick、OpenCDA `NetworkManager.current_time_slot/current_sim_time`、NS3 `sync_request/sync_ack` 三者使用同一时间基准；OpenCDA PPS 指定的 `sc_start/sc_num` 能进入 NS3 manual subchannel 行为；短程在线仿真不会因时间流速不一致提前断开。

## 当前状态

2026-07-17 已完成真实 CARLA + NS3 35 tick 短回归，并连续修复两个在线协议 bug：车辆注册未完成时过早初始化 NS3、以及多轮 CP 中 scheduler strategy 残留。最新日志位于：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short_strategyclear_20260717_041313\
```

该轮使用策略清空修复后的代码。与上一轮相比，时间同步继续保持 38/38，NS3 fatal/manual reject 仍为 0，PHY decode failure 大幅下降，在线 AP 提升到 `0.88/0.88/0.79`。

上一轮车辆注册 gate 修复日志位于：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short_fixed_20260717_031703\
```

该 artifact 目录包含：

```text
ns3_stdout.log
ns3_stderr.log
opencda_stdout.log
```

本轮同时保留了一次失败日志用于定位 bug：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short_20260717_031125\
```

失败原因：OpenCDA sender thread 在只有 1 辆车注册时就向 NS3 发送 `vehicles_num=1`，随后真实第一帧位置为 20 车，NS3 在已安装协议栈后尝试重新初始化并触发 `Ipv4AddressGeneratorImpl::Add(): Address Collision` / `SIGABRT`。

修复：

- `NetworkManager` 新增 `vehicle_registration_complete` gate。
- `template.init()` 在 single CAV、traffic CAV、RSU/platoon/UAV 创建完成后调用 `mark_vehicle_registration_complete()`。
- NS3 初始化前额外检查 CARLA id 映射，避免 `carla_id=null` 的半初始化帧进入 NS3。

修复后轻量检查：

```powershell
conda run -n opencda python -m py_compile opencda\core\networking\network_manager.py opencda\core\networking\ns3_co_simulation\bridge\carla_ns3_bridge.py opencda\tools\offline_ns3_replay.py test\test_network_time_sync.py

conda run -n opencda python -c "from test.test_network_time_sync import test_network_time_slot_matches_carla_fixed_delta,test_multiple_network_slots_track_carla_time; test_network_time_slot_matches_carla_fixed_delta(); test_multiple_network_slots_track_carla_time(); print('network_time_sync tests passed')"
```

结果：

```text
network_time_sync tests passed
```

修复后在线短回归结果：

| Metric | Value |
| --- | ---: |
| OpenCDA exit code | 0 |
| CARLA ticks | 35 |
| NS3 initialized vehicles | 20 |
| `sync_request` / `sync_ack` | 38 / 38 |
| Sync timeout / reconnect failure | 0 / 0 |
| `MANUAL_CMD_ADD` | 158 |
| `MANUAL_CMD_REJECT` | 0 |
| `cam_received` lines | 137 |
| NS fatal / SIGABRT / address collision | 0 |
| PSCCH decode failures | 1836 |
| PSSCH decode failures | 480 |
| OpenCDA CP counter | 1 |
| Online AP@0.3 / AP@0.5 / AP@0.7 | 0.86 / 0.84 / 0.74 |

策略清空修复后在线短回归结果：

| Metric | Value |
| --- | ---: |
| Artifact | `online_ns3_short_strategyclear_20260717_041313` |
| OpenCDA exit code | 0 |
| CARLA ticks | 35 |
| `sync_request` / `sync_ack` | 38 / 38 |
| Sync timeout / reconnect failure | 0 / 0 |
| `MANUAL_CMD_ADD` | 156 |
| `MANUAL_CMD_REJECT` | 0 |
| `cam_received` lines | 150 |
| NS fatal / SIGABRT / address collision | 0 |
| PSCCH decode failures | 95 |
| PSSCH decode failures | 10 |
| Decoded-overlap failures | 88 |
| OpenCDA successful upload lines | 21 |
| OpenCDA incomplete upload lines | 184 |
| OpenCDA CP counter | 1 |
| Online AP@0.3 / AP@0.5 / AP@0.7 | 0.88 / 0.88 / 0.79 |

使用 `opencda.tools.online_ns3_log_eval` 进一步按 source-target upload episode 去重后：

| Run | Complete Episodes | Partial Episodes | Duplicate Incomplete Lines | Main Failure Shape |
| --- | ---: | ---: | ---: | --- |
| `online_ns3_short_fixed_20260717_031703` | 14 | 8 | 237 | stale scheduler strategy caused many overlap failures |
| `online_ns3_short_strategyclear_20260717_041313` | 21 | 6 | 178 | each remaining partial episode misses exactly one 10000-byte fragment |

该轮新增修复：

- `PotentialGame.clear_resource_allocation_strategy()` 同步清理各 CAV `ClusteringScheduler.channel_allocation`。
- 避免新一轮 PPS 结果与上一轮残留 sender 在同一 receiver/subchannel 上叠加。
- 在线 PHY failure 从上一轮 `PSCCH/PSSCH=1836/480` 降到 `95/10`，说明主要冲突源已经定位并消除。

剩余边界：

- 35 tick 短回归中 OpenCDA 日志仍有重复 incomplete upload line；episode-level 去重后，真实 partial episode 为 6 个，均是缺少一个 10000-byte fragment。
- 这更像单 fragment PHY/PSCCH/PSSCH loss 后缺少应用层重传/重调度，而不是时间同步、车辆初始化或子信道越界问题。
- 该在线短回归用于证明真实 CARLA tick + NS3 bridge + manual subchannel 语义已闭环；论文主表仍以离线 41 帧 mAP 与 request-level NS3 replay 为主。

后续 timeout reupload first trial：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_reupload_20260717_053012\
```

该轮打开 `re_upload_when_timeout=true`、`max_reupload_attempts=1`。Episode-level 结果从 `21/6` complete/partial 改善到 `39/3`，说明应用层补偿能修复一部分单 fragment loss。但该轮不是 clean exit：late CAM completion 到达时对应 sender 的 `uploading_cavs` round state 已被清理，触发 `KeyError: 17`。代码已改为安全处理 late completion，仍需再跑一轮 clean reupload 回归后才能把它作为最终在线协议结果。

时间同步证据：

```text
Sent sync_ack: CARLA t=0.05s, NS3 t=0.05s
Sent sync_ack: CARLA t=0.10s, NS3 t=0.10s
...
Sent sync_ack: CARLA t=1.90s, NS3 t=1.90s
```

结论：

- 在线 CARLA 与 NS3 的时间流速不一致问题已在短回归中闭环：CARLA time 与 NS3 time 按 0.05 s tick 对齐推进。
- OpenCDA 指定的 `sc_start/sc_num` 已真实落到 NS3 manual scheduler，日志中出现合法 `MANUAL_CMD_ADD` 和对应 `PSCCH_DECODE_OK`。
- 本轮在线短回归仍出现大量 PHY decode failures 和部分 OpenCDA 大包上传 incomplete。这不是车辆初始化/时间同步 bug，但说明在线图形仿真的大包分片、同帧并发和 PHY 诊断仍需独立分析；论文主表仍应以离线 request-level replay 的 110/110 RLC-complete 结果作为严格链路可行性证据。
- 回归后已关闭 CARLA，并确认无 CARLA / NS3 / 5556 / 5557 残留进程。

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
$env:OPENCDA_CLEAN_WORLD_ON_INIT = "1"
$env:OPENCDA_CARLA_CLIENT_TIMEOUT = "180"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
Remove-Item Env:\OPENCDA_CLUSTERING_CONFIG
Remove-Item Env:\OPENCDA_ONLINE_TICKS
Remove-Item Env:\OPENCDA_CLEAN_WORLD_ON_INIT
Remove-Item Env:\OPENCDA_CARLA_CLIENT_TIMEOUT
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

在线短回归当前不作为主表结果来源，原因是 35 tick 图形仿真中 CP counter 仅为 1，且大包分片在真实 PHY 下存在 incomplete upload。后续如果要把在线 NS3 结果写入论文正文，需要进一步解析在线 request-level trace，把 application callback、RLC completion 和 PHY failure 逐 request 对齐。
