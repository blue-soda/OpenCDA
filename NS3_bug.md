
## NS3 Co-Simulation Debugging

### Startup Order
Use this order when debugging CARLA + NS3 co-simulation:

1. Kill stale CARLA / OpenCDA / NS3 processes.
2. Start `CarlaUE4.exe` and wait until the simulator is fully loaded.
3. Start the NS3 executable in WSL and redirect to a fresh log file.
4. Start OpenCDA with `--network`.

If OpenCDA reports either of the following:
- `Town03 is not found in your CARLA repo`
- `time-out of 10000ms while waiting for the simulator`

then assume CARLA has crashed or has not finished loading yet. In that case, kill all stale `CarlaUE4` processes and relaunch CARLA before retrying OpenCDA.

### Process Cleanup Commands

Run these from PowerShell before each clean repro:

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
wsl.exe bash -lc "pkill -f ns3.42-main-default || true"
```

If you only want to restart CARLA:

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process 'C:\Workspace\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe'
```

### NS3 Run Command
Run NS3 from PowerShell, but execute the binary inside WSL. This is the command pattern actually used during debugging:

```powershell
$ns3Log = "/home/sakakibara/Workspace/carla-ns3-co-simulation/log_run_$(Get-Date -Format yyyyMMdd_HHmmss).txt"
Start-Process powershell -ArgumentList "-NoExit","-Command","wsl.exe bash -lc 'cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 > $ns3Log 2>&1'"
```

If you prefer to run it in the current terminal instead of a new window:

```powershell
wsl.exe bash -lc "cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 > /home/sakakibara/Workspace/carla-ns3-co-simulation/log.txt 2>&1"
```

Reference paths:
- `/home/sakakibara/Workspace/carla-ns3-co-simulation/ns3/vanet/main.cc`
- `/home/sakakibara/Workspace/carla-ns3-co-simulation/log.txt`

### OpenCDA Run Command
Use the actual scenario command that was used in this debug session. Run it from `C:\Workspace\OpenCDA`:

```powershell
cd C:\Workspace\OpenCDA
python opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug --network
```

### Logs to Inspect
- OpenCDA logs:
  - `C:\Workspace\OpenCDA\opencda\log\`
- NS3 logs:
  - `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log.txt`
  - `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log_run_*.txt`

Do not read the full logs at once. First locate the newest log, then search by keywords.

### Find the Latest Logs

OpenCDA side:

```powershell
Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

NS3 side:

```powershell
Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

### Targeted Log Searches

OpenCDA side:

```powershell
$log = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "Connected to NS-3|sync_with_ns3: sync successful|FIRST time|AGAIN|schedule link=|send_transfer_requests|no communication_requests|uploaded its data|Received size:|waiting exceeded threshold|timeout, current_time_slot|history_try_volume|communication_requests now has" $log
```

NS3 side:

```powershell
$log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "Transfer request:|MANUAL_CMD_ADD|MANUAL_CMD_CHECK|MANUAL_LOGICAL_MAP|PSCCH_DECODE_OK|PSCCH_DECODE_FAIL|PSSCH_DECODE_OK|PSSCH_DECODE_FAIL|SCI2_DECODE_FAIL|cam_received|sync_ack|vehicles_num|vehicles_position" $log
```

If you need counts instead of raw matches:

```powershell
$log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
@("MANUAL_CMD_ADD","MANUAL_CMD_CHECK","MANUAL_LOGICAL_MAP","PSCCH_DECODE_OK","PSCCH_DECODE_FAIL","PSSCH_DECODE_OK","PSSCH_DECODE_FAIL","SCI2_DECODE_FAIL","cam_received") |
  ForEach-Object {
    $count = (rg -c $_ $log)
    "{0} = {1}" -f $_, $count
  }
```

If you need to compare OpenCDA send-side and receive-side events in one file quickly:

```powershell
$log = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "FIRST time|send_transfer_requests|Received size:|uploaded its data|history_try_volume|cp counter" $log
```

### What to Verify First
- Time sync:
  - `carla_time` and `ns3_time` should match in `sync_ack`
- Whether OpenCDA is generating `FIRST time` uploads and actual `send_transfer_requests`
- Whether NS3 receives the same batch of transfer requests and produces matching `cam_received`
- Whether the failure is in NS3 wireless delivery or in OpenCDA aggregation / state transition

### Typical Debugging Workflow

Use this template when you need to reproduce and count what happened in one co-simulation run.

#### Step 1: Clean restart

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
wsl.exe bash -lc "pkill -f ns3.42-main-default || true"
Start-Process 'C:\Workspace\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe'
```

Wait for CARLA to finish loading, then start NS3 and OpenCDA.

#### Step 2: Start NS3 with a fresh log

```powershell
$ns3Log = "/home/sakakibara/Workspace/carla-ns3-co-simulation/log_run_$(Get-Date -Format yyyyMMdd_HHmmss).txt"
Start-Process powershell -ArgumentList "-NoExit","-Command","wsl.exe bash -lc 'cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 > $ns3Log 2>&1'"
```

#### Step 3: Start OpenCDA

```powershell
cd C:\Workspace\OpenCDA
python opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug --network
```

If OpenCDA reports CARLA timeout or missing town, kill all `CarlaUE4` processes and restart CARLA before retrying.

#### Step 4: Find the newest logs after the run

```powershell
$opencdaLog = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$ns3Log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$opencdaLog
$ns3Log
```

#### Step 5: Count the four key numbers

The most useful first-pass comparison is:
- how many uploads OpenCDA tried to send
- how many transfer requests NS3 accepted
- how many packets NS3 reported as received
- how many uploads OpenCDA finally marked as completed

OpenCDA send-side and completion-side:

```powershell
$opencdaLog = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
"FIRST time = $((rg -c 'FIRST time' $opencdaLog))"
"send_transfer_requests = $((rg -c 'send_transfer_requests' $opencdaLog))"
"uploaded its data = $((rg -c 'uploaded its data' $opencdaLog))"
rg -n "FIRST time|send_transfer_requests|uploaded its data|history_try_volume|cp counter" $opencdaLog
```

NS3 receive-side:

```powershell
$ns3Log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
"Transfer request = $((rg -c 'Transfer request:' $ns3Log))"
"cam_received = $((rg -c 'cam_received' $ns3Log))"
"PSCCH_DECODE_OK = $((rg -c 'PSCCH_DECODE_OK' $ns3Log))"
"PSCCH_DECODE_FAIL = $((rg -c 'PSCCH_DECODE_FAIL' $ns3Log))"
rg -n "Transfer request:|cam_received|PSCCH_DECODE_OK|PSCCH_DECODE_FAIL|PSSCH_DECODE_OK|PSSCH_DECODE_FAIL|MANUAL_LOGICAL_MAP" $ns3Log
```

#### Step 6: Interpret the mismatch

Use the counts this way:

- If `FIRST time` exists but `send_transfer_requests` is missing, the break is inside OpenCDA before requests are flushed to NS3.
- If `send_transfer_requests` exists but NS3 has no `Transfer request:`, the break is in the bridge or socket path.
- If NS3 has `Transfer request:` but `cam_received` is much lower than expected, the break is in NS3 wireless delivery or receive-side decode.
- If NS3 has enough `cam_received` but OpenCDA still does not print `uploaded its data`, the break is in OpenCDA aggregation or upload state cleanup.
- If `history_try_volume` only has the first slot non-zero, check whether there were actually later `send_transfer_requests` before blaming the wireless layer.

#### Step 7: For second-round CP failures, search only the state-machine keywords

```powershell
$opencdaLog = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "preparing data from|communicate: set uploading_data|FIRST time|AGAIN|schedule link=|communication_requests now has|send_transfer_requests|no communication_requests|waiting exceeded threshold|uploaded its data" $opencdaLog
```

This is the fastest way to confirm whether the second CP round prepared data but never re-entered `schedule -> transfer_requests`.

### Known Debugging Conclusions
- A real NS3 time-sync bug existed before and was fixed in the NS3 repo. The old failure mode was `sync_ack` returning while NS3 time had already run far ahead of CARLA time.
- The OpenCDA bridge listener also had a real startup bug: it used a one-shot `accept()` timeout and could exit before NS3 connected back.
- `subchannel_start=-1` in the current NS3 path should not be trusted as a safe default during debugging. In observed runs it behaved like overlapping default allocations and caused asymmetric delivery.
- Even after forcing explicit `subchannel_start` and `subchannel_num=1`, one uplink can still stall partway through. That means the remaining bug is inside the NS3 NR sidelink scheduling / delivery path, not only in the OpenCDA fallback logic.

# CARLA + NS3 联合仿真丢包问题排查记录
## 0. 关键代码文件路径
### OpenCDA / CARLA 侧
- `C:\Workspace\OpenCDA\opencda\core\clustering\managers\clustering_scheduler.py`
- 当前实际生效的 fallback 子信道分配逻辑在这里，核心函数为 `_get_naive_subchannel(source_id)`。
- 之前已确认，问题不能靠单纯修改这里的 `scStart` 解决；临时修改 fallback 映射为更分散方式，问题依旧存在。
- 同时也是 CARLA 侧给 NS3 下发子信道选择的主要入口，当前生效的朴素映射在此文件中。
`C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\naive_ra.py`
- 表面上容易怀疑的资源分配文件，但本轮有效联仿中，真正生效的路径并非由它单独决定。
`C:\Workspace\OpenCDA\CLAUDE.md`
- 之前已按用户要求补充过联仿调试说明。
`C:\Workspace\OpenCDA\TODO.md`
- 之前已按用户要求记录问题、处理过程和未解决项。
`C:\Workspace\OpenCDA\opencda\core\networking\network_manager.py`
- NS3 交互入口，`communicate_through_ns3()` 会把大点云拆成多个 `transfer_requests`。
- `send_cams_via_ns3()` 会累计 `communication_requests`，并更新 `try_volume`。
- `send_msg_to_ns3()` 在某个 tick 把当前 `communication_requests` 一次性发给 NS3，然后清空列表。
`C:\Workspace\OpenCDA\opencda\core\networking\ns3_co_simulation\bridge\carla_ns3_bridge.py`
- CARLA 与 NS3 的 socket 桥，已修复启动竞态、重连等待、首帧同步等待问题。
`C:\Workspace\OpenCDA\opencda\core\sensing\perception\coperception_manager.py`
- 点云上传、超时、接收完成判断逻辑，当前 `timeout_slots`、`re_upload_when_timeout`、`all_data_uploaded()` 都在这里。
- 存在 timeout 日志误报、`all_data_uploaded()` 统计逻辑不严谨等问题。
`C:\Workspace\OpenCDA\opencda\core\clustering\managers\clustering_perception_manager.py`
- 协同感知主循环，`collect_cluster_members_data()`、`receive_cluster_members_data()`、`do_cp_every_tick` 在这里。
- 当前存在第二轮 CP 数据已准备但未触发上传调度的问题。
### NS3 侧当前工作目录中的覆写文件
- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\src\nr-spectrum-phy.cc`
- 本轮最关键的接收侧定位文件，`RxSlPscch()` 的 `decoded_overlap` 问题就在这里。
- 接收侧 PSCCH / PSSCH 解码逻辑所在地，已新增日志（`PSCCH_DECODE_OK`、`PSSCH_DECODE_FAIL` 等）用于定位问题。
- 已在该文件中加过 SINR 比较器修正，将非有限 SINR 当作最低优先级，并用索引做稳定 tie-breaker。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\src\nr-sl-ue-mac-scheduler-fixed-mcs.cc`
- 当前调度器对候选资源的 overlap 判断主要在这里，核心函数为 `OverlappedResources(...)`、`FilterTxOpportunities(...)`。
- 当前只对 PSSCH 子信道冲突敏感，不足以表达 PSCCH 可并发约束。
- 负责候选资源过滤与 grant 分配。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\src\nr-sl-ue-mac-scheduler-manual.cc`
- 当前手动调度命令进入调度器后的处理逻辑在这里。
- 当前 Carla 命令进入 NS3 后的逻辑子信道映射在这里完成。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc`
- 联仿入口，与 CARLA 的 socket 通信、时间同步、transfer request 接收都在这里。
- 联仿主入口，`vehicles_num` / `vehicles_position` / `transfer_requests` / `sync_request` 都在这里处理。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.cc`
- 包含 `CamSenderNR::ScheduleCam()` / `SendCam()` 函数。
- 已加入首帧延迟，避免 bearer 尚未 ready 就发包。
### NS3 原始实现路径（实际构建时对应的 nr 模块源码）
- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\contrib\nr\model\nr-spectrum-phy.cc`
- 原始 `NrSpectrumPhy` 实现，`StartTxSlCtrlFrames()`、`StartTxSlDataFrames()`、`RxSlPscch()` 在这里。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\contrib\nr\model\nr-spectrum-phy.h`
- `NrSpectrumPhy` 声明。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\contrib\nr\model\nr-sl-ue-mac.cc`
- PSCCH/PSSCH 的 `VarTti` 资源构造在这里，`ctrlVarTtiInfo.rbStart/rbLength`、`dataVarTtiInfo.rbStart/rbLength` 都在这里设置。
- MAC 侧构造 VarTti 的关键文件，`ctrlVarTtiInfo.rbStart = currentSlot.slPsschSubChStart * subChSize` 等核心逻辑在此。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\contrib\nr\model\nr-ue-phy.cc`
- `SendNrSlCtrlChannels()`、`SendNrSlDataChannels()` 在这里，控制/数据最终是通过这里构造 RB 掩码后下发到 `NrSpectrumPhy`。
- UE PHY 侧将 RB 范围转换为实际发送掩码的关键文件，核心函数为 `SendNrSlCtrlChannels(...)`、`SetSubChannelsForTransmission(...)`。
- 已新增 `SL_VAR_TTI_ADD`、`PSCCH_TX_PREP` 等日志，用于验证发送侧 RB 范围与实际发射 PSD 的一致性。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\contrib\nr\model\nr-phy.cc`
- `GetTxPowerSpectralDensity()`、`DoSetNrSlVarTtiAllocInfo()` 在这里。
- PHY 侧生成 PSD 的关键文件，已新增日志用于验证 RB 范围与 PSD 的一致性。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\contrib\nr\helper\nr-spectrum-value-helper.cc`
- `CreateTxPowerSpectralDensity()` 在这里，用于确认 PSD 是否只覆盖期望 RB。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\contrib\nr\model\nr-sl-comm-resource-pool-factory.cc`
- PSCCH 资源池默认参数在这里定义。
`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns-3-dev\src\lte\model\nr-sl-comm-resource-pool.cc`
- 资源池展开后每个 slot 的 `numSlPscchRbs/slPscchSymStart/slPscchSymLength` 在这里生成。
## 1. 问题现象与初始排查（排查初期）
### 1.1 核心问题现象
- 当前项目是 CARLA 和 NS3 联合仿真，问题表现为：CARLA 端发起两条发送请求，NS3 端收到请求并执行发送，但最终只有一条流完整送达，另一条流只送达前半段，随后停止推进。
- 初始怀疑方向：一是 CARLA 端给不同发送方分配的数据子信道发生冲突；二是 NS3 接收侧在同 slot 并发场景下丢弃了其中一条发送。
### 1.2 初始排查过程与结论
- 第一步排查：排除简单 PSSCH 数据子信道冲突。有效复现中，CARLA 侧两条发送的有效数据子信道为 `2 -> 1, scStart=2` 和 `3 -> 1, scStart=1`，两者本身不重叠，但仍有一条流卡住，说明问题不在 PSSCH 子信道分配结果本身。
- 第二步排查：排除 CARLA 端子信道映射修改的有效性。单纯修改 CARLA 端 `clustering_scheduler.py` 中的 `scStart` 分配逻辑，或调整 fallback 映射为更分散方式，均无法解决问题，说明问题不只是“数据子信道离得不够远”。
- 第三步排查：锁定关键失败点。通过分析 NS3 接收侧日志，发现问题出在 NS3 接收侧 `nr-spectrum-phy.cc` 中的 `RxSlPscch()` 函数，因 `decoded_overlap` 逻辑淘汰了一条 PSCCH 控制信道，导致对应 PSSCH 无法正常解码。
## 2. 根因深度分析（排查中期）
### 2.1 核心根因：PSCCH 控制信道资源冲突
- 问题本质：并非 PSSCH 数据信道冲突，而是 PSCCH 控制信道在接收端的资源视角发生重叠。接收侧 `rbDecodedBitmap` 逻辑会记录已解码成功的 PSCCH 占用的 RB 区间，若后续另一个 PSCCH 命中该区间，会触发 `decoded_overlap` 并被直接淘汰。
- 连锁反应：一旦 PSCCH 解码失败（`PSCCH_DECODE_FAIL reason=decoded_overlap`），对应的 PSSCH 后续无法被完整按正常流程接收，最终表现为“发送端已发送，接收端只完整收到一条流”。
- 日志佐证：接收端日志显示，两条 PSCCH 均落到同一组 RB（`rbStart=0`、`rbEnd=105`），说明接收端认为两者占用同一段控制 RB。
### 2.2 深层问题：调度/资源语义缺失
- 调度器逻辑缺陷：当前调度器（`nr-sl-ue-mac-scheduler-fixed-mcs.cc`）的 overlap 过滤逻辑，仅检查同一 slot 中 PSSCH 子信道范围是否重叠，未将“同一 slot 中映射到共享 PSCCH 资源的两个发送不可并发”这一约束纳入。
- 语义断层：CARLA 以为“不同数据子信道就能并发”，调度器主要按 PSSCH 视角判断，而接收侧 PHY 存在 PSCCH 可解码性约束，三者之间存在语义不一致。
- 补充问题：排查中发现，两条控制消息的 `sinrAvg` 曾出现 `NaN`，导致排序依赖实现细节，已通过修改 `nr-spectrum-phy.cc` 中的比较器修正。
### 2.3 `decoded_overlap` 的性质判断
- `decoded_overlap` 本身是 PHY 接收建模中的保护性特性：`RxSlPscch()` 函数会按 `sinrAvg` 对控制消息排序，已解码成功的 PSCCH 会将 RB 记入 `rbDecodedBitmap`，避免同一组控制 RB 上重复解码多个 SCI1，其设计意图合理，并非凭空出错。
- 业务失败的真正根源：系统允许了接收侧无法同时正确处理的资源组合进入发送阶段，即调度和资源映射未提前纳入 PSCCH 可并发约束，`decoded_overlap` 只是触发失败的表象，而非最终根因。
## 3. 验证操作与结果（排查中期-后期）
### 3.1 接收侧日志增强（验证第一步）
- 操作：在 `nr-spectrum-phy.cc` 中新增日志，包括 `PSCCH_DECODE_OK`、`PSSCH_DECODE_FAIL`、`SCI2_DECODE_FAIL` 等，输出字段涵盖 `txRnti`、`rbStart/rbEnd`、`sinrAvg`、解码失败原因等。
- 作用：为本次问题定位提供了关键依据，明确了 `decoded_overlap` 是导致 PSCCH 解码失败的直接原因。
### 3.2 临时放宽 `decoded_overlap` 验证（验证第二步）
- 操作：在 `nr-spectrum-phy.cc` 中修改逻辑，命中 `decoded_overlap` 时只打标记，不立即设置 `corrupt=true`。
- 结果：原本卡住的流恢复完整送达，两条流均能在 CARLA 端完整拼接，验证了当前业务失败确实由 `decoded_overlap` 触发。
- 说明：该改动仅为验证手段，并非最终修复方案，需后续通过规范资源映射解决根本问题。
### 3.3 发送侧日志增强与验证（验证第三步）
- 操作：在 `nr-phy.cc`、`nr-ue-phy.cc` 中新增 `SL_VAR_TTI_ADD`、`PSCCH_TX_PREP` 等日志，用于对比“请求的 RB 范围”和“实际发射 PSD 的活跃 RB 范围”。
- 待验证点：按发送侧代码链（MAC 构造 VarTti → UE PHY 转换 RB 掩码 → PHY 生成 PSD），不同 `slPsschSubChStart` 的 PSCCH 应出现在不同 RB 段，但接收侧日志显示两者均落到 `rbStart=0, rbEnd=105`，需进一步排查差异原因。
### 3.4 联调验证结果（最终修复后，排查后期）
- 联调状态：已成功跑通真实联调，无 PSCCH/decoded_overlap 导致的两路并发上传失败，至少验证了双并发是稳定的。
- 关键日志：
        
  - OpenCDA 日志：`C:\Workspace\OpenCDA\opencda\log\opencda_20260418_logicalmap7.out.log`、`opencda_20260418_concurrency1.out.log`，显示 `cav 3` 和 `cav 2` 均成功上传数据到 `cav 1`。
  - NS3 日志：`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log_run_20260418_logicalmap7.txt`，显示 `MANUAL_CMD_ADD = 16`、`PSCCH_DECODE_OK = 10897`、`cam_received = 16`，验证了数据收发正常。
- 逻辑子信道映射结果：当前 Carla 可见的逻辑子信道数为 10，物理总子信道数为 11，`sc_start` 不再表示“直接指定物理子信道起点”，而是“当前最早可发 slot 上的第几个合法候选资源”，NS3 会从合法资源中选择，确保 PSCCH 资源不冲突。
## 4. 现存问题（排查后期-后续优化）
无线链路核心丢包问题已修复，当前主要剩余问题集中在 OpenCDA 上层业务调度、统计逻辑和 CARLA-NS3 TCP 死锁。

### 4.1 OpenCDA 业务逻辑问题（✅ 已修复）

#### 4.1.1 timeout 日志存在误报 ✅ 已修复
- 修复方式：`send_cams_via_network()` 中重新组织了超时判断逻辑，增加了 `[NS3-retransmit]` 前缀日志，明确标注”等待 sidelink 重传，不是失败”。
- 关键代码：`coperception_manager.py` lines 121-134。

#### 4.1.2 `all_data_uploaded()` 统计逻辑不严谨 ✅ 已修复
- 修复方式：移除了 `cavs_timeout` 的计数，直接使用 `len(self.uploaded_cavs)`。
- 关键代码：`coperception_manager.py` lines 210-211。

#### 4.1.3 第二轮 CP 未触发上传调度 ✅ 已修复
- 修复方式：在 `clear_uploaded_and_uploading()` 中增加 `self.all_cavs = {}`，确保下一轮 CP 能重新从 `get_coperception_cavs_dict()` 获取最新的邻居信息。
- 配合修复：同时保留了 `self.uploaded_cavs = {}`（已在 HEAD 中存在）。
- 关键代码：`coperception_manager.py` lines 236-240。

### 4.2 NS3 早期退出导致联合仿真在 slot 7 终止 ✅ 已修复
- 现象：联合仿真在 NS3 t=3.5s（CARLA slot 7）左右时，NS3 日志显示 `Carla disconnected or error on port 5556`，之后 NS3 进程退出。CARLA 继续运行但 `send_msg_to_ns3()` 的同步陷入死锁，仿真时间冻结在 t=3.5s。
- NS3 日志证据：
  ```
  [INFO] Simulator::Run() returned, syncPending=0, NS3 time: 3.5s
  [DEBUG] Raw incoming: {“type”: “vehicles_position”, ...(all at 0,0,0)...}
  [DEBUG] Received message type: vehicles_position
  [INFO] Carla disconnected or error on port 5556.
  [INFO] Sync simulation ended at NS3 time: 3.5s
  ```
- 死锁分析：
  1. NS3 在 t=3.5s 发送 `sync_ack` 后，`Simulator::Run()` 返回，`syncPending=0`，进入 `while(running && syncCv.wait())` 等待下一个 `sync_request`。
  2. CARLA 的 `send_msg_to_ns3()` 在 `send_something_to_ns3()` 处阻塞等待 `sync_ack`，主线程无法推进。
  3. CARLA process 可能主动关闭了 port 5556 连接，或 TCP 流控导致双方均阻塞。
- OpenCDA 缓解措施：在 `send_msg_to_ns3()` 中，当 `sync_with_ns3()` 返回 False 时，不再发送 `vehicles_position/transfer_requests`，设置 `connected=False` 后继续循环，避免 TCP 错误向上传播到 `bridge.stop()` 触发 CARLA socket 关闭。
- NS3 根本修复：在 `SocketReceiverServerThread()` 中，将 `recv()` 改为 `select()` + 超时模式。修复文件：`\\\\wsl.localhost\\Ubuntu\\home\\sakakibara\\Workspace\\carla-ns3-co-simulation\\ns3\\vanet\\main.cc`。
  - 修复方式：使用 `select()` 配合 1 秒超时，替代原本阻塞的 `recv()` 调用
  - 效果：接收线程每 1 秒检查一次 `running` 标志，即使 NS3 主线程退出也能及时退出，避免永久阻塞
  - 代码位置：`main.cc` lines 386-420
  ```cpp
  while (running) {
      fd_set read_fds;
      FD_ZERO(&read_fds);
      FD_SET(client_fd, &read_fds);
      struct timeval tv;
      tv.tv_sec = 1;
      tv.tv_usec = 0;

      int select_result = select(client_fd + 1, &read_fds, nullptr, nullptr, &tv);
      if (select_result < 0) {
          std::cerr << “[ERR] select failed in SocketReceiverServerThread\n”;
          break;
      }
      if (select_result == 0) {
          // Timeout - no data available, loop back to check running flag
          continue;
      }
      // Data is available to read
      memset(buffer, 0, sizeof(buffer));
      const ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
      // ... rest of processing
  }
  ```

### 4.3 统计口径导致 throughput 看起来偏低
- 现象：最新业务统计显示 `avg_throughput = 2183.77`，看似很低，实则是统计口径问题。
- 原因：总成功上传体积（152864 Bytes）被平均到 70 个 slot 上，而真实上传是在 slot 0 到 slot 6 之间一次性完成的，属于”一次 burst 上传摊平到整个 episode”，并非无线链路能力不足。
- 补充：`history_try_volume` 首格非零、后续全零，是因为 NS3 在 t=3.5s 退出后，后续 CP 轮次根本没有 NS3 可以接收新传输，而非持续尝试失败。
## 5. 解决思路与下一步建议（后续推进）
### 5.1 整体解决思路（短期+中期+长期）
- 短期目标：回退 `RxSlPscch()` 中对 `decoded_overlap` 的临时放宽，保留当前日志增强，完成发送侧日志复现，对比发送侧 `VarTti` 请求的 `rbStart/rbLength` 与接收侧实际观测到的 `rbBitmap`。
- 中期目标：建立“逻辑子信道 → 真实无线资源”映射层，CARLA 下发逻辑子信道 ID，NS3 负责映射到真正不会发生 PSCCH 冲突的物理资源。
- 长期目标：在 NS3 侧正确建模 PSCCH 可并发约束，将 CARLA 能看到的逻辑信道数从 1 个扩为多个，实现多并发稳定传输，而非保守收缩为“同一时刻只允许 1 个发送”。
### 5.2 下一步执行顺序（优先级排序）
1. 保留现有日志补丁，回退 `RxSlPscch()` 中临时放宽 `decoded_overlap` 的行为改动。
2. 完成带发送侧日志的有效复现，对比发送侧 `VarTti` 请求的 `rbStart/rbLength` 与接收侧实际观测到的 `rbBitmap`。
3. 根据对比结果排查：若发送侧是局部 RB、接收侧仍为整段带宽，查接收/通道侧 PSCCH 资源映射；若发送侧本身为整段带宽，查 `VarTti -> PSD` 之间的异常层。
4. 修复 OpenCDA 上层业务逻辑问题（按优先级）：
        
  - 优先级 1：修正 timeout 语义（`coperception_manager.py`），`re_upload_when_timeout = False` 时不重写 `uploading_cavs[cav_id]`，优化日志描述。
  - 优先级 2：修正 `all_data_uploaded()` 统计逻辑，区分 `uploaded`、`waiting`、`timed_out_but_not_failed`、`failed` 四种状态，不将 `cavs_timeout` 等同于上传完成。
  - 优先级 3：排查第二轮 CP 未触发上传调度的问题，重点检查相关函数的状态变量触发逻辑。
5. 正式实现“逻辑子信道到真实无线资源”的映射层，最终将 CARLA 可见的逻辑信道数从 1 扩为多个。
## 6. 
CARLA + NS3 联合仿真丢包的核心根因是 NS3 接收侧 `RxSlPscch()` 因 `decoded_overlap` 淘汰了一条 PSCCH 控制信道，深层原因是系统未正确建模 PSCCH 可并发约束；当前无线链路核心问题已修复，剩余问题集中在 OpenCDA 上层业务调度与统计逻辑，需按优先级逐步修正，最终通过建立逻辑子信道映射层实现多并发稳定传输。
逻辑信道问题已经得到修复：Carla 侧仍然看到 10 个逻辑子信道，但这 10 个不再是“10 个可随便写死的物理起点”，它们现在是“NS3 在当前最早可发 slot 上筛出来的 10 个合法候选资源编号”
我们做的是：
   - Carla 传逻辑编号
   - NS3 先跑自己的候选资源过滤
   - 再从合法候选里选第 k 个
所以现在保留下来的是整套合法资源的原始 slot + PSCCH + PSSCH 组合，这就是为什么数字还是 10，但冲突不见了。

## 7. 本次修改记录（2026-04-18）

### 7.1 NS3 侧 TCP 死锁修复

**文件：** `carla-ns3-co-simulation/ns3/vanet/main.cc`

**问题：** `SocketReceiverServerThread()` 中使用阻塞 `recv()` 接收 CARLA 数据。当 CARLA 关闭连接或 NS3 主线程退出后，`recv()` 永久阻塞，导致接收线程无法退出，整个联合仿真死锁在 t=3.5s。

**修复：** 将阻塞 `recv()` 替换为 `select()` + 1 秒超时，使接收线程每秒检查一次 `running` 标志：

```cpp
// 修复前（阻塞）：
const ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);

// 修复后（带超时）：
while (running) {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(client_fd, &read_fds);
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;

    int select_result = select(client_fd + 1, &read_fds, nullptr, nullptr, &tv);
    if (select_result < 0) { break; }
    if (select_result == 0) { continue; }  // 超时，重新检查 running

    const ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    // ... 后续处理不变
}
```

**效果：** 接收线程不再永久阻塞，NS3 可在 CARLA 断开后 ≤1 秒内干净退出。

### 7.2 验证状态

| 修复项 | 文件 | 状态 |
|--------|------|------|
| NS3 recv() 阻塞死锁 | `main.cc` SocketReceiverServerThread | ✅ 代码已修复，NS3 已重新编译 |
| OpenCDA sync 失败不发送 | `network_manager.py` send_msg_to_ns3 | ✅ 已在工作区 |
| CP 多轮 uploaded_cavs 重置 | `coperception_manager.py` clear_uploaded_and_uploading | ✅ 已在 HEAD |
| CP all_data_uploaded 统计 | `coperception_manager.py` all_data_uploaded | ✅ 已在 HEAD |

## 8. 2026-04-20 �����޸���¼

### 8.1 ���������������û��ͬ���� `cluster_state`

��������
- OpenCDA ��־���Ѿ��ܿ��������໥���֡�
- �� CP �����־������ `head_id=None`��`members=[]`��`remote_ids=[]`��`with_stats=False`��
- ��˵�����ⲻ�ڡ��ھӷ��֡������ڡ�������û���������� CP ʹ�õ�״̬����

�ؼ������ļ���
- `C:\Workspace\OpenCDA\opencda\core\clustering\managers\clustering_v2x_manager.py`
- `C:\Workspace\OpenCDA\opencda\core\clustering\base\clustering_algorithm.py`
- `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\clustering\coalition_game.py`

����
- `ClusteringV2XManager.run_algorithm()` ����þ����㷨���õ� cluster �б���
- �����߼�û�пɿ��ذѱ��� cluster �����д��ÿ̨���� `v2x_manager.cluster_state`��
- `all_clusters` Ҳû���������¾�����ͬ��ˢ�¡�

�޸���
- �� `clustering_v2x_manager.py` ������ `_sync_cluster_states()`��
- ÿ���� cluster ���������
  - ������г��ɵ� `cluster_state`
  - �� `head_id/member_ids` ��д��ÿ����Ա��
  - ˢ�� `ClusteringV2XManager.all_clusters`
  - ��ӡ `CLUSTER_SYNC [...]` ��־

�޸���֤�ݣ�
- `C:\Workspace\OpenCDA\opencda\log\opencda_20260420_004345.log`
- ��־�ȳ��� `CLUSTER_SYNC [(1, [1])]`
- ����ȶ���Ϊ `CLUSTER_SYNC [(1, [1, 2, 3, 4])]`

### 8.2 �������������У�NS3 ���� OpenCDA ������ `127.0.0.1`

��������
- OpenCDA ������ `5556`���� `5557` ��һֱֻ�У�
  - `Listening for NS-3 connections on localhost:5557`
  - `Still waiting for NS-3 connection...`
- ͬʱͬ���̲߳��ϣ�
  - ���� `sync_request`
  - �ȴ� `sync_ack`
  - ���ʱ

�ؼ������ļ���
- `C:\Workspace\OpenCDA\opencda\core\networking\ns3_co_simulation\bridge\carla_ns3_bridge.py`
- `C:\Workspace\OpenCDA\opencda\core\networking\ns3_co_simulation\config\settings.py`
- `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc`

����
- ֮ǰ�ù��� WSL/Windows NAT ��ַ�Ǿ�ֵ��
- 2026-04-20 ��̨�����ϣ�WSL ���� Windows �����˿�ʵ��ֻ��ͨ�� `127.0.0.1` �ɹ���
- ������ʹ�þ� `172.26.*` ��ַ��NS3 �޷����� OpenCDA����� `sync_ack` ��������������ʱ��һֱͣ�� 0 �롣

��֤��ʽ��
- �� Windows ��ʱ�������ز��Զ˿ڡ�
- �� WSL �ֱ���� `127.0.0.1`��`172.26.160.1`��`10.255.255.254`��
- ʵ��ֻ�� `127.0.0.1` ����ͨ��

�޸���
- ���� NS3 ʱ��ʽʹ�ã�
  - `--carlaHost=127.0.0.1`
- �޸��� OpenCDA ��־���֣�
  - `Connected to NS-3 at ('127.0.0.1', ...)`
  - `sync_with_ns3: sync successful`

### 8.3 ������ʵ���������Զ�˵����Ѿ����� ego ���

�ؼ���־�ļ���
- OpenCDA��`C:\Workspace\OpenCDA\opencda\log\opencda_20260420_004345.log`
- NS3��`\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log_run_20260420_004323_localhost.txt`

ͳ�ƽ����
- OpenCDA �����ϴ��ִΣ�3 ��
- NS3 �յ� `transfer_requests`��3 ��
- NS3 �յ��ķ�Ƭ����������72 ��
- ��һ�������ϴ��ɹ���Ա��3 ��
  - `2 -> 1`
  - `3 -> 1`
  - `4 -> 1`
- ����ʹ��Эͬ�����������ִΣ�1 ��
  - `CP_EVAL_FRAME ego=1 head_id=1 slot=24 remote_ids=[3, 4, 2] use_remote=True`
- ������Эͬ���ݼ���ͳ�Ƶ��ύ��1 ��
  - `CP_SUBMIT_FRAME ego=1 slot=24 remote_ids=[3, 4, 2] with_stats=True`

���� AP��
- IOU 0.3 = 0.89
- IOU 0.5 = 0.89
- IOU 0.7 = 0.75

���ۣ�
- ������һ�� ego �����ȷʹ���˳�Ա `2/3/4` �ϴ���Զ�˵��ơ�
- ��û���κ�һ�� CP ������������Эͬ���ݡ�����жϣ��ѱ����������Ʒ���

### 8.4 `timeout_slots = 4` �Ա�������ֻ��Ϊ����ʱ deadline / �澯��ֵ

�ؼ������ļ���
- `C:\Workspace\OpenCDA\opencda\core\sensing\perception\coperception_manager.py`

��ǰ������
- `timeout_slots = 4`
- `re_upload_when_timeout = False`

���ֲ�ã�
- ��һ����·���ƴӷ���ȫ����ɣ�Լ���� 20~21 �� slot��

���壺
- �ӡ�100ms deadline���Ƕȿ��������ϴ��ǳ�ʱ�ġ�
- ���ӡ���·�ɿ��ԡ��Ƕȿ��������Ƕ�ʧ�����ǡ������ܵ���������������

��˱��ֲ��õĲ�������ǣ�
- ���� `timeout_slots = 4` ����
- ���������� `final_drain_slots = 20`
- `timeout_slots` ������ʾʵʱ deadline / �澯��ֵ
- `final_drain_slots` ֻ�� episode ��β�׶���Ч�����ڵȴ����һ����;�ϴ��������

### 8.5 ���� `final_drain_slots = 20`

�ؼ������ļ���
- `C:\Workspace\OpenCDA\opencda\scenario_testing\template.py`
- `C:\Workspace\OpenCDA\opencda\core\clustering\managers\clustering_perception_manager.py`
- `C:\Workspace\OpenCDA\opencda\scenario_testing\config_yaml\enable_network.yaml`

ʵ��˼·��
- �� `stop()` ���ʽ����ǰ����һ�������޵���β�ſս׶Ρ�
- ����׶β��ٷ����µ��ϴ��ִΣ�ֻ���պ��ƽ���ǰ�Ѿ���;�ĵ��Ʒ�Ƭ��
- �������ƽ� `20` �� slot��

ʵ��ϸ�ڣ�
- `template.py` ������ `_run_final_drain()`��
- `clustering_perception_manager.py` ������ `final_drain_mode`��`enable_final_drain()`��`has_pending_final_drain()`��
- ������ final drain ʱ��
  - cluster head ֻ���� `receive_cluster_members_data()`
  - ���ٵ��ûᴥ�����ϴ��� `collect_cluster_members_data()`
  - ��ǰ����;�ϴ�������β��������ɣ��Կɲ������һ�� CP ����

��һ����Ŀ�Ĳ��Ǹı����������ڵ� 100ms deadline�����Ǳ��⡰���һ���Ѿ����������������˳�����������û�Ե�β�ֽ������
