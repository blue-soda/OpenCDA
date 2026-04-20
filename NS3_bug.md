# CARLA + NS3 联合仿真排查记录

## 0. 当前可用的调试命令

### 0.1 启动顺序

推荐顺序：

1. 清理残留的 CARLA / OpenCDA / NS3 进程。
2. 启动 `CarlaUE4.exe`，等待地图和传感器加载完成。
3. 启动 NS3。
4. 启动 OpenCDA。

如果 OpenCDA 报下面任一错误：

- `Town03 is not found in your CARLA repo`
- `time-out of 10000ms while waiting for the simulator`

优先判断为 CARLA 未加载完成或已经崩溃。此时应先清理全部 `CarlaUE4` 进程，再重启 CARLA。

### 0.2 进程清理

PowerShell：

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
wsl.exe bash -lc "pkill -f ns3.42-main-default || true; fuser -k 5556/tcp 5557/tcp 2>/dev/null || true"
```

只重启 CARLA：

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process 'C:\Workspace\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe'
```

### 0.3 OpenCDA 启动命令

必须使用 `conda run -n opencda`：

```powershell
cd C:\Workspace\OpenCDA
conda run -n opencda python opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug --network
```

### 0.4 NS3 启动命令

Mirrored 模式下，本机最近一次成功联调用的是：

```powershell
wsl.exe bash -lc "cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 --carlaHost=127.0.0.1 > /home/sakakibara/Workspace/carla-ns3-co-simulation/log.txt 2>&1"
```

如果希望输出到带时间戳的新日志：

```powershell
$ns3Log = "/home/sakakibara/Workspace/carla-ns3-co-simulation/log_run_$(Get-Date -Format yyyyMMdd_HHmmss).txt"
Start-Process powershell -ArgumentList "-NoProfile","-Command","wsl.exe bash -lc 'cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 --carlaHost=127.0.0.1 > $ns3Log 2>&1'"
```

### 0.5 日志路径

- OpenCDA：
  - `C:\Workspace\OpenCDA\opencda\log\`
- NS3：
  - `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\`

### 0.6 快速定位日志关键词

OpenCDA：

```powershell
$log = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "Connected to NS-3|sync_with_ns3: sync successful|send_transfer_requests|uploaded its data|CP_EVAL_FRAME|CP_SUBMIT_FRAME|FINAL_DRAIN|Sync timeout|Sync failed|Average Precision" $log
```

NS3：

```powershell
$log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "Connected to Carla on port 5557|Transfer request:|cam_received|SendMsgToCarla: {\"carla_time\"|send_to_carla_fd: -1|connect failed|Carla disconnected or error on port 5556" $log
```

## 1. 关键代码与日志路径

### 1.1 OpenCDA / CARLA 侧

- `C:\Workspace\OpenCDA\opencda\core\clustering\managers\clustering_scheduler.py`
  - CARLA 侧对子信道的分配入口。
- `C:\Workspace\OpenCDA\opencda\core\networking\network_manager.py`
  - OpenCDA 与 NS3 的交互入口，负责拼装和发送 `transfer_requests`。
- `C:\Workspace\OpenCDA\opencda\core\networking\ns3_co_simulation\bridge\carla_ns3_bridge.py`
  - CARLA / OpenCDA 到 NS3 的 socket 桥接。
- `C:\Workspace\OpenCDA\opencda\core\sensing\perception\coperception_manager.py`
  - 点云上传、超时、接收完成判断逻辑。
- `C:\Workspace\OpenCDA\opencda\core\clustering\managers\clustering_perception_manager.py`
  - 协同感知主循环，决定何时发起上传、何时做融合。
- `C:\Workspace\OpenCDA\opencda\core\clustering\managers\clustering_v2x_manager.py`
  - 聚类结果同步到 `cluster_state` 的关键位置。
- `C:\Workspace\OpenCDA\opencda\core\common\vehicle_manager.py`
  - `update_info()` 中会触发聚类算法；本轮 `final_drain` 需要冻结这里的 cluster update。
- `C:\Workspace\OpenCDA\opencda\core\common\cav_world.py`
  - 本轮新增 `freeze_cluster_updates`。
- `C:\Workspace\OpenCDA\opencda\scenario_testing\template.py`
  - 仿真总流程控制；本轮 `final_drain` 主逻辑在这里。
- `C:\Workspace\OpenCDA\opencda\scenario_testing\config_yaml\enable_network.yaml`
  - 网络联仿配置；包含 `final_drain_slots`。

### 1.2 NS3 侧

- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc`
  - 联仿主入口，负责 `vehicles_num`、`vehicles_position`、`transfer_requests`、`sync_request`，以及 `5556/5557` socket。
- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.cc`
  - CAM/点云发送节奏控制。
- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\src\nr-spectrum-phy.cc`
  - PSCCH/PSSCH 接收解码；`RxSlPscch()` 是最初无线丢包定位核心。
- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\src\nr-sl-ue-mac-scheduler-fixed-mcs.cc`
  - 候选资源过滤与 grant 分配。
- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\src\nr-sl-ue-mac-scheduler-manual.cc`
  - 手动调度器；当前逻辑子信道到物理资源的映射落在这里。

## 2. 详细排查时间线

### 2.1 2026-04-17 到 2026-04-18：最初问题现象

最初现象有四个：

- CARLA 已经发起点云上传请求。
- NS3 也收到了请求，并执行了发送。
- 但接收端只能完整收到其中一部分，另一部分会卡住或只收到前半段。
- OpenCDA 统计里经常只有前几个时隙有通信量，后续几乎没有新的有效上传，AP 也明显偏低。

当时的两个初始怀疑方向：

- CARLA 端给不同发送方分配的数据子信道发生冲突。
- NS3 接收侧在同 slot 并发场景下丢弃了其中一条发送。

### 2.2 2026-04-18：确认不是简单的 PSSCH 子信道冲突

复现中可以看到：

- 不同发送方在 CARLA 侧看到的 `scStart` 不同。
- 表面上 PSSCH 子信道并不重叠。
- 但即使这样，仍会出现一条流卡住。

结论：

- 问题不只是“数据子信道编号冲突”。
- 继续单纯调 CARLA 侧 `scStart` 不是根治方案。

### 2.3 2026-04-18：锁定 NS3 接收侧 `decoded_overlap`

关键文件：

- `ns3/src/nr-spectrum-phy.cc`

关键函数：

- `RxSlPscch()`

定位结论：

- 接收侧会用 `rbDecodedBitmap` 记录已经成功解码的 PSCCH 控制资源。
- 第二个 PSCCH 如果命中已占用 RB，会被判定为 `decoded_overlap`。
- 一旦 SCI1 被淘汰，对应 PSSCH 后续也无法正常走完解码链。

因此最初的核心无线问题并不是简单误码，而是：

- 接收侧把一条 PSCCH 主动淘汰了。

### 2.4 2026-04-18：判断 `decoded_overlap` 不是唯一根因

进一步判断后确认：

- `decoded_overlap` 本身更像 PHY 接收模型中的保护性特性。
- 真正的系统性问题是：调度器和上层接口允许了一组“接收侧不能并发处理”的资源组合进入发送阶段。

更具体地说：

- CARLA 传给 NS3 的 `sc_start/sc_num` 被过于直接地当成物理子信道写入。
- 这样破坏了 NS3 候选资源模板里原本合法的 `slot + PSCCH + PSSCH` 绑定关系。

### 2.5 2026-04-18：通过临时放宽 `decoded_overlap` 做验证

当时做过一轮临时验证：

- 在 `RxSlPscch()` 中放宽 `decoded_overlap` 的淘汰逻辑。

结果：

- 原本卡住的那条流恢复完整送达。
- 证明业务失败确实和这条接收淘汰链直接相关。

但这个修改只用于验证，不是最终修复。

### 2.6 2026-04-18：无线侧正式修法改为“逻辑子信道 -> 合法候选资源”

最终正式修法不是继续放宽接收，而是改语义：

- CARLA 继续看到多个“逻辑子信道”。
- 但这些逻辑子信道不再表示“直接指定物理 PSSCH 起点”。
- NS3 先根据自己的候选资源过滤规则，得到当前 slot 上的合法资源集合。
- 再把 CARLA 传来的逻辑编号解释为“合法候选资源中的第 k 个”。

这一步的结果是：

- 保留下来的是一整套合法无线资源。
- 而不是仅仅覆写一个 PSSCH 起点。

### 2.7 2026-04-18：解释“为什么之前也是 10 个子信道却还会冲突”

关键不是数字 `10`，而是语义：

- 修改前：
  - `10` 更接近“10 个可以任意覆写的物理起点编号”。
- 修改后：
  - `10` 表示“NS3 在当前约束下筛出来的 10 个合法逻辑候选编号”。

因此：

- 数字没变。
- 冲突却消失了。

### 2.8 2026-04-18：无线链路修复后的联调结果

当时已经看到：

- OpenCDA 可以发起多轮上传。
- NS3 可以稳定收到并执行对应发送。
- 至少已有带远端点云的 `CP_EVAL_FRAME ... use_remote=True` 和 `CP_SUBMIT_FRAME ... with_stats=True`。

结论：

- 无线链路主丢包问题已经不再是主矛盾。

### 2.9 2026-04-18 到 2026-04-19：OpenCDA 侧残留问题逐渐浮现

无线主问题修完后，剩下的主要问题转移到 OpenCDA 上层逻辑。

#### 2.9.1 `timeout_slots = 4` 的语义容易误读

当时确认：

- `timeout_slots = 4`
- `re_upload_when_timeout = False`

它更应该理解为：

- “实时 deadline / 告警阈值”

而不是：

- “4 个 slot 内必须传完，否则一定失败”

这点很重要，因为三路点云完整上传往往需要 20 到 30 个 slot。

#### 2.9.2 `history_try_volume` 和 `avg_throughput` 容易误导

确认过的误区：

- `history_try_volume` 只有少数位置非零，不一定代表后续一直失败。
- 很多时候是后续根本没有新的 `transfer_requests` 发出来。
- `avg_throughput` 被摊平到整个 episode，也会让一次 burst 上传看起来吞吐很低。

#### 2.9.3 需要区分三种完成状态

必须区分：

- 实时 deadline 内完成
- deadline 外最终完成
- 最终未完成

否则 AP 低时，很难判断究竟是：

- 无线丢包
- 上层没继续发
- 还是统计口径误导

### 2.10 2026-04-18 到 2026-04-19：NS3 / 桥接层工程性修复

#### 2.10.1 NS3 `recv()` 阻塞导致接收线程无法退出

关键文件：

- `ns3/vanet/main.cc`

问题：

- 接收线程里阻塞式 `recv()` 会导致 NS3 在 CARLA 断开或主线程退出后卡死。

修复：

- 改成 `select()` + 超时轮询。
- 让接收线程能周期性检查 `running`。

#### 2.10.2 OpenCDA 同步失败后不应长期假死

关键文件：

- `network_manager.py`
- `carla_ns3_bridge.py`

修复方向：

- `sync_request / sync_ack` 失败时做重试和重连。
- 避免因为向死 socket 发包，把整个桥接提前拖死。

### 2.11 2026-04-20：确认聚类结果没有真正同步到运行态

问题现象：

- 日志里已经能看到车辆互相发现。
- 但 CP 相关日志长期是 `head_id=None`、`members=[]`、`remote_ids=[]`、`with_stats=False`。

这说明问题不在“邻居发现”，而在：

- 聚类结果没有真正进入 CP 使用状态。

修复后观察到：

- `CLUSTER_SYNC [(1, [1, 2, 3, 4])]`
- 后续也出现了 `CP_EVAL_FRAME ... use_remote=True` 和 `CP_SUBMIT_FRAME ... with_stats=True`

### 2.12 2026-04-20：第一次 `final_drain` 验证失败

当时已经把 `final_drain_slots = 20` 做进去了，但第一次验证失败。

失败特征：

- `FINAL_DRAIN` 确实被触发。
- 但刚进入收尾阶段，就出现：
  - `Sync timeout`
  - `Sync failed`
  - 桥接重连和残留 `cam_received`

进一步看代码后发现两个问题：

1. `final_drain` 复用了完整 `_tick_once()`
   - 收尾阶段仍在继续跑聚类、重选 cluster head、继续正常仿真。
2. 桥接层 socket 在重连和发送之间存在竞争
   - 会出现 `WinError 10038` 这类“对非套接字操作”的错误。

### 2.13 2026-04-20：第一次 `final_drain` 修复

为了解决上面的两个问题，做了两类修改。

#### 2.13.1 收尾阶段冻结 cluster update

关键文件：

- `cav_world.py`
- `vehicle_manager.py`
- `template.py`

修复思路：

- 新增 `freeze_cluster_updates`。
- 在 `final_drain` 阶段禁止 `VehicleManager.update_info()` 触发新的 `run_algorithm()`。
- 增加专门的 `_tick_final_drain()`，只推进：
  - 世界 tick
  - 数据更新
  - CP 接收 / 提交
  - network time slot

而不再复用完整 `_tick_once()`。

#### 2.13.2 桥接层发包/重连串行化

关键文件：

- `carla_ns3_bridge.py`

修复思路：

- 对 `_connect()`、`send_something_to_ns3()`、`stop()` 加锁。
- 避免一边重连、一边发包、一边关 socket 时出现竞争。

### 2.14 2026-04-20：发现 WSL 网络模式是关键外部条件

这一步非常关键。

复现中发现：

- NS3 到 OpenCDA 的 `5557` 回连经常失败。
- NS3 日志反复出现：
  - `Connecting to Carla on port 5557...`
  - `connect failed: Connection refused`
  - `send_to_carla_fd is not connected`

最后确认：

- 这不是代码逻辑本身的问题。
- 而是 WSL 网络模式从 Mirrored 退回了 NAT。

影响是：

- NAT 模式下，NS3 侧 `127.0.0.1:5557` 连到的是 WSL 自己，而不是 Windows 上 OpenCDA 的监听器。
- 因此即使 `5556` 方向还能工作，`5557` 回连也会失败。

### 2.15 2026-04-20：恢复 Mirrored 后再次联调

在 WSL 恢复为 Mirrored 后，再次联调，关键现象发生了变化。

NS3 成功回连：

- `Connected to Carla on port 5557.`

OpenCDA 成功进入真实传输：

- 多次 `send_transfer_requests: sending 24 requests`

第一阶段：

- cluster head 为 `1`
- 成功完成一轮带远端点云的 CP
- 日志中明确出现：
  - `CP_EVAL_FRAME ego=1 head_id=1 slot=36 remote_ids=[3, 4, 2] use_remote=True`
  - `CP_SUBMIT_FRAME ego=1 slot=36 remote_ids=[3, 4, 2] with_stats=True`

第二阶段：

- cluster head 后续切换为 `2`
- 尾轮上传进入 `final_drain`
- 日志中出现：
  - `FINAL_DRAIN slot=1/20 pending_heads=[2] time_slot=70`
  - ...
  - `FINAL_DRAIN done before slot 11, no pending uploads.`

这说明：

- `final_drain` 不是简单“触发了一下”
- 而是真的把尾轮在途上传继续推进到了 drain 完成

### 2.16 2026-04-20：成功联调中的尾轮细节

在成功联调日志里可以看到：

- `cav 3 data upload to 2 succeeded ... cost time: 16`
- `cav 1 data upload to 2 succeeded ... cost time: 16`
- 当时 `2 Coperception data uploaded: 2/3 (66.67%), return True`

也就是说：

- `final_drain` 期间尾轮上传确实被继续推进。
- 并不是一进入 drain 就停止通信。

虽然 `4 -> 2` 那条流在这轮 drain 中没有完全完成，但：

- drain 确实把尾轮在途流推进到了“部分完成甚至多数完成”的状态
- 并让最终评估吃到了可用尾轮结果

### 2.17 2026-04-20：最终成功结果

成功联调日志：

- OpenCDA：
  - `C:\Workspace\OpenCDA\opencda\log\opencda_20260420_093048.log`
- NS3：
  - `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log_run_20260420_093043_finaldrain_mirrored_clean.txt`

最终结果：

- `The Average Precision at IOU 0.3 is 0.89`
- `The Average Precision at IOU 0.5 is 0.89`
- `The Average Precision at IOU 0.7 is 0.76`

结论：

- `final_drain_slots = 20` 已经在真实联调中完成验证。
- 它确实能把 episode 尾部仍在途的上传继续推进，再进入最终评估。

## 3. 当前结论

目前可以明确确认的事情有五个：

1. 最初的无线链路主问题确实在 NS3 接收侧 `RxSlPscch()` 的 `decoded_overlap`。
2. 真正的系统根因不是单点接收逻辑，而是上层和调度器没有正确表达 PSCCH 可并发约束。
3. 逻辑子信道 -> 合法候选资源 的映射方案已经落地，之前那类并发上传丢失不再是主矛盾。
4. `timeout_slots = 4` 仍应保留为实时 deadline / 告警阈值，不应直接扩大替代 `final_drain`。
5. `final_drain_slots = 20` 已经在真实联调里证明有效。

## 4. 总结

这次排查大致经历了三个阶段。

第一阶段是无线根因定位：

- 先排除了“CARLA 简单子信道冲突”这种表层解释。
- 最后确认真正的问题在 NS3 接收侧对 PSCCH 的 `decoded_overlap` 淘汰，以及更上层的资源语义缺失。

第二阶段是系统性修复：

- 通过“逻辑子信道 -> 合法候选资源”的映射修掉了最初的并发上传丢失问题。
- 之后又修了 OpenCDA / NS3 桥接、CP 聚类状态同步、收尾阶段的 `final_drain` 执行路径。

第三阶段是环境与收尾验证：

- `final_drain` 第一次失败并不是逻辑无效，而是同时叠加了 WSL 网络模式从 Mirrored 退回 NAT 的环境问题。
- 在恢复 Mirrored 后，`5557` 回连恢复正常，`final_drain` 才在真实联调中完整证明有效。

因此，当前系统的结论不是“还在靠临时补丁勉强运行”，而是：

- 无线主链路问题已经修通。
- 尾轮排空机制已经验证成功。
- 当前联合仿真已经能够稳定完成真实上传、真实协同检测和最终评估。

后续如果再出现“明明都用 localhost，为什么一会儿能跑一会儿不能跑”的现象，优先检查的不是代码，而是：

- WSL 当前到底处于 Mirrored 还是 NAT。
