# 实验环境

更新时间：2026-07-15

## Conda 环境

项目可用 Conda 环境名：

```powershell
conda activate opencda
```

脚本化执行建议使用：

```powershell
conda run -n opencda python <script-or-module>
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

## 仓库入口

- OpenCDA 主入口：`opencda.py`
- 典型场景配置：`opencda/scenario_testing/config_yaml/`
- 核心自动驾驶模块：`opencda/core/`
- 协同感知/应用逻辑：`opencda/application/`
- 聚类与资源分配相关模块：`opencda/core/clustering/`
- V2X/NS3 联动相关模块：`opencda/core/networking/`
- OpenCOOD 感知模型与融合框架：`opencood/`
- 通用离线推理入口：`opencda.tools.offline_inference`

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
