# 实验环境

更新时间：2026-07-16

## Conda 环境

项目可用 Conda 环境名：

```powershell
conda activate opencda
```

脚本化执行建议使用：

```powershell
conda run -n opencda python <script-or-module>
```

当前环境快照（2026-07-15）：

- Python：`3.7.10`
- pip：`21.1.2`
- CARLA Python API：`0.9.11`（`conda list` 显示为 `carla 0.9.11 dev_0 <develop>`）
- PyTorch：`1.10.0+cu113`
- torchvision：`0.11.1+cu113`
- NumPy：`1.21.6`
- Open3D：`0.10.0.0`
- OmegaConf：`2.3.0`
- PyYAML：`6.0.1`
- scikit-learn：`0.24.2`
- spconv：`spconv-cu113 2.3.6`
- OpenCV：`opencv-python 4.5.2.52`

可复查命令：

```powershell
conda run -n opencda python --version
conda list -n opencda | Select-String -Pattern "^(python|carla|torch|torchvision|numpy|pyyaml|omegaconf|open3d|opencv|scikit-learn|spconv)\\s"
```

## 路径约定

数据集根目录：

```powershell
D:\Data\Carla
```

导出 CARLA/OPV2V 风格数据集时设置：

```powershell
$env:OPENCDA_DATA_DUMP_ROOT = "D:\Data\Carla"
```

CARLA 启动路径：

```powershell
C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe
```

当前 CARLA 程序目录快照（2026-07-15）：

- `CarlaUE4.exe` 路径：`C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe`
- `CarlaUE4.exe` 修改时间：`2026-07-14 23:37:41`
- `CarlaUE4.exe` 文件大小：`188,928 bytes`
- Windows 文件属性未提供 `FileVersion/ProductVersion`。
- 版本口径以 Python API `carla 0.9.11` 为准；若论文需要 simulator binary 的精确 release，应人工确认 CARLA 包来源或压缩包名称。

## 仓库入口

- OpenCDA 主入口：`opencda.py`
- 典型场景配置：`opencda/scenario_testing/config_yaml/`
- 核心自动驾驶模块：`opencda/core/`
- 协同感知/应用逻辑：`opencda/application/`
- 聚类与资源分配相关模块：`opencda/core/clustering/`
- V2X/NS3 联动相关模块：`opencda/core/networking/`
- OpenCOOD 感知模型与融合框架：`opencood/`
- 通用离线推理入口：`opencda.tools.offline_inference`

当前代码版本快照（2026-07-15）：

- OpenCDA 仓库 HEAD：`fcc29fdc9ee9a9fe694c12e1fb6792b4d41bccac`
- 当前工作区包含未提交改动；复现实验时应记录 `git status --short` 或保存 patch。
- OpenCOOD 位于本仓库 `opencood/` 子目录，随同 OpenCDA 仓库 HEAD 固定。

## 基本实验命令

启动 CARLA：

```powershell
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe"
```

SGCP 在线 CARLA 仿真：

```powershell
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug
```

SGCP + NS3 协同仿真：

```powershell
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
```

## WSL ns-3 / 5G-V2X 仿真

CARLA-NS3-co-simulation 已放在 WSL2 Ubuntu 22.04 的 Linux 文件系统中：

```bash
/home/sakakibara/workspace/carla-ns3-co-simulation
```

Windows 侧入口是指向该 WSL 路径的软链接：

```powershell
C:\Workspace\carla-ns3-co-simulation
```

`main.cc` 源文件路径：

```bash
/home/sakakibara/workspace/carla-ns3-co-simulation/ns3/vanet/main.cc
```

Windows 侧同一路径：

```powershell
C:\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc
```

当前 ns-3 / V2X 代码快照（2026-07-15）：

- co-simulation 仓库 HEAD：`10ab54cee04b04bce7f638249ddae1619fb11bf1`
- `ns-3-dev` HEAD：`c90c13b8310a813cf4eaf67a2c90df497bbd1965`
- ns-3 wrapper version：`ns-3-dev-v2x-v1.1-dirty`
- `ns-3-dev` 当前存在 dirty/generated 状态，包括若干 `src/lte/model/*.cc` type-change 标记和 `NrDlMacStats.txt`、`NrUlMacStats.txt`。

可复查命令：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation && git rev-parse HEAD && git status --short"
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && git rev-parse HEAD && git describe --tags --always --dirty && ./ns3 show version"
```

编译 ns-3 / NR V2X：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 build"
```

执行 `ns3/vanet/main.cc`：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=100.0'"
```

短时 smoke test 可禁用同步模式，但仍需要先由 bridge/CARLA 向 ns-3 的 `5556` 端口发送第一帧车辆数据：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=1.0 --enableTimeSync=false --carlaHost=127.0.0.1'"
```

注意：当前 ns-3 wrapper 的参数格式不使用 README 中的第二个 `--` 分隔符；应写成 `./ns3 run 'scratch/vanet/main.cc --simTime=...'`。

CARLA/OPV2V 风格数据集导出：

```powershell
$env:OPENCDA_DATA_DUMP_ROOT = "D:\Data\Carla"
$env:OPENCDA_DATADUMP_TICKS = "140"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --dump
```

`OPENCDA_DATADUMP_TICKS` 控制导出运行 tick 数，默认 `140`。现有 `DataDumper` 会跳过前 60 tick，并每 2 tick 保存一次，因此默认约导出 40 帧。

离线 OpenCOOD 推理 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla
```

指定 scenario、timestamp 和 ego CAV：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1
```

跑完整个 scenario：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0
```

离线 NS3 同步 smoke test：

1. 先启动 ns-3：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true --carlaHost=auto'"
```

2. 另开一个 PowerShell 运行离线回放到 NS3：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --drain-seconds 0.3
```

该工具不启动 CARLA，只从 dump 数据重建车辆位姿和 SGCP cluster 内上传请求，并按数据集帧间隔向 NS3 发送 `vehicles_position`、`sync_request`、`transfer_requests`。

## NS3 Request-Level Trace 约定

当前 co-simulation 的 ns-3 侧已支持跨研究可复用的 request-level trace，用于把 OpenCDA 发送的单个 transfer request 映射到 application / RLC 层事件。

核心约定：

- OpenCDA replay 发送 `transfer_requests` 时，每个请求带 `pkt_id`。
- ns-3 CAM application 将 `pkt_id` 写入 CAM header 的 `request_id`。
- `cam_received` 日志输出 `request_id`，可按 `(frame_index, request_id)` 回连到 OpenCDA 侧请求表。
- ns-3 侧新增 `LteRlcRequestIdTag`，CAM packet 同时写入 PacketTag 和 ByteTag。
- NR sidelink RLC UM 日志输出 `[NRSL_RLC_TX]`、`[NRSL_RLC_RX]`、`[NRSL_RLC_DROP]`，并携带 `request_id`。

相关 ns-3 文件：

```text
C:\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.h
C:\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.cc
C:\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc
C:\Workspace\carla-ns3-co-simulation\ns3\src\lte-model\lte-rlc-request-id-tag.h
C:\Workspace\carla-ns3-co-simulation\ns3\src\lte-model\lte-rlc-request-id-tag.cc
C:\Workspace\carla-ns3-co-simulation\ns3\src\lte-model\lte-rlc-um.cc
C:\Workspace\carla-ns3-co-simulation\ns-3-dev\src\lte\CMakeLists.txt
```

OpenCDA 侧解析入口：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout <ns3_stdout.log> --upload-plan <upload_plan.csv> --output-dir <output_dir> --rsu-node-id 21 --max-frames <N>
```

该解析器输出：

```text
cam_received_records.csv
delivery_summary.csv
delivery_by_frame.csv
delivery_by_type.csv
phy_decode_events.csv
phy_decode_summary.csv
rlc_events.csv
rlc_summary.csv
rlc_by_request.csv
```

已验证的 LGCP 11 帧 replay 口径：

- planned requests：`676`
- application `cam_received`：`31`
- RLC TX events：`1131`
- RLC RX events：`252`
- unique RLC RX requests：`164`
- RLC request RX ratio：`0.242604`

注意：

- application callback 统计低于 RLC RX 统计，不能直接替代链路层 delivery ratio。
- 当前 PSCCH / PSSCH decode diagnostics 仍是 aggregate PHY 事件，尚未逐条绑定 `request_id`。
- 若后续研究要解释 HARQ、subchannel collision 或 PHY decode failure，需要继续把 `request_id` 透传到 PHY TB / HARQ trace。
