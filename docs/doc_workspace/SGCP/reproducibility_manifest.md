# SGCP Reproducibility Manifest

更新时间：2026-07-16

本文档用于固定 SGCP 当前所有“可复现结果”的代码版本、数据、配置、命令和日志位置，并单独标出论文 `main.tex` 中旧结果尚缺原始日志的问题。

## 版本与环境

OpenCDA 当前复现实验版本：

```text
2cd026ec96691d15e4d764f4bd78af51a2404859
```

当前已知无关 dirty 文件：

```text
D CLAUDE.md
M opencood/opencood/utils/box_overlaps.c
?? MYREADME.md
?? visualization_output/visualize.png
?? visualization_output/visualize_spectator_view.png
?? visualization_output/visualize_transparent.png
```

这些文件未参与本轮 SGCP 文档和离线 replay 复现实验。运行环境以 `docs/doc_workspace/environment.md` 为准：

- Conda：`opencda`
- Python：`3.7.10`
- CARLA Python API：`0.9.11`
- PyTorch：`1.10.0+cu113`
- 数据集根目录：`D:\Data\Carla`
- CARLA 可执行文件：`C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe`

NS3 / co-simulation 说明：环境文档保留了 2026-07-15 的旧版本快照；后续 NS3 manual subchannel 修复已经另行提交到 co-simulation 工作区。Windows 侧直接 `git -C C:\Workspace\carla-ns3-co-simulation` 会触发 safe.directory ownership 检查，复查时应在 WSL 用户环境中执行环境文档给出的命令。

## 数据集

当前主要复现实验使用：

```text
D:\Data\Carla\2026_07_15_01_26_56
```

数据摘要：

- 场景：`v2xp_cluster_carla`
- CAV 数量：20
- 帧数：41
- 帧范围：`000060` 到 `000140`
- 每个 CAV：41 个 `.pcd` + 41 个 `.yaml`
- 根目录包含：`data_protocol.yaml`

导出命令：

```powershell
$env:OPENCDA_DATA_DUMP_ROOT = "D:\Data\Carla"
$env:OPENCDA_DATADUMP_TICKS = "140"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --dump
```

## 论文旧主表状态

`C:\Workspace\icdcs-paper\SGCP\main.tex` 当前旧主表写作：

| Method | mAP@0.3 | mAP@0.5 | mAP@0.7 |
| --- | ---: | ---: | ---: |
| NC | 0.13 | 0.12 | 0.10 |
| RS | 0.31 | 0.28 | 0.28 |
| MUG | 0.37 | 0.35 | 0.33 |
| FullPerception | 0.81 | 0.68 | 0.57 |
| Ours | 0.85 | 0.84 | 0.69 |

旧通信开销写作：

| Method | Comm. Overhead |
| --- | ---: |
| SGCP / Ours | 22.33 Mbps |
| FullPerception | 35.35 Mbps |
| MUG | 30.27 Mbps |
| RS | 25.43 Mbps |

当前状态：未在 `docs/doc_workspace/SGCP` 或仓库中找到这些旧表格对应的原始日志、随机种子、代码提交和完整配置。因此这些数值暂时不能作为“已复现结果”继续强写。论文修订时应采用当前 manifest 中已复现结果，或找回旧实验日志后再恢复旧主表。

## 当前已复现主结果

### Full 20-CAV early fusion upper reference

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0
```

结果：

- Frames：41
- CAVs：20
- Fusion：OpenCOOD early fusion
- AP@0.3 / AP@0.5 / AP@0.7：`0.85 / 0.83 / 0.48`

说明：这是无 SGCP 通信约束的 full-sharing upper reference，不是 SGCP 主方法。

### SGCP constrained + inter-cluster late fusion

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --max-frames 0
```

结果：

- Frames：41
- Cluster heads：6/frame
- Resource allocation：`potential_game`
- AP@0.3 / AP@0.5 / AP@0.7：`0.77 / 0.73 / 0.35`
- Avg. upload：`109,415.48 bytes/source`
- Total upload：`26,916,208 bytes`
- Avg. source CAVs / cluster head：`2.67`

### Scheduler ablations

同数据、同 SGCP inter-cluster late-fusion evaluation path：

| RA Algorithm | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Total Upload |
| --- | ---: | ---: | ---: | ---: | ---: |
| `potential_game` | 41 | 0.77 | 0.73 | 0.35 | 26,916,208 |
| `random` | 41 | 0.44 | 0.39 | 0.17 | 9,725,376 |
| `mws` | 41 | 0.31 | 0.26 | 0.11 | 9,910,032 |

### Same-budget CAV-only selective-sharing baselines

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline communication_aware --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --max-frames 0
```

| Baseline | AP@0.3 | AP@0.5 | AP@0.7 | Total Upload | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Nearest | 0.76 | 0.73 | 0.37 | 28,026,832 | CAV-only, same cluster/head late-fusion path |
| Density | 0.77 | 0.74 | 0.39 | 30,574,368 | Strong baseline, higher payload |
| Communication-aware | 0.78 | 0.75 | 0.40 | 30,222,256 | Strongest current baseline, higher payload |

结论：当前 41 帧 dump 上，SGCP 不能声称 AP 全面优于所有 same-budget selective-sharing baseline。论文主张应转向 PPS channel feasibility、NS3 request-level 可验证传输、控制开销、稳定性和可解释调度。

## NS3 可复现链路结果

### 10-subchannel scheduled replay

相关 artifact：

```text
docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target10_globalpkt\
```

结果：

- Frames：11
- Scheduled requests：110
- Skipped unscheduled demand：44
- CAM received：110/110
- RLC complete：110/110
- PHY decode failures：0
- Avg. delay：23.909 ms
- P95 delay：24.000 ms

### 5-subchannel exposed-window replay

相关 artifact：

```text
docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\
```

结果：

- Planned requests：110
- `sc_start=0..4`：55/55 complete
- `sc_start=5..9`：55/55 rejected before CAM/RLC creation
- CAM delivery：55/110
- RLC complete：55/110
- `MANUAL_CMD_REJECT=55`
- PHY decode failures：0

解释：该结果验证 OpenCDA 指定的子信道窗口能真实落到 NS3；带宽范围内且无冲突的 request 成功，超出暴露子信道范围的 request 被拒绝，不污染后续合法 request。

## 运行时与控制开销

运行时 profiling 命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

日志：

```text
docs\doc_workspace\SGCP\artifacts\runtime_breakdown_41f\offline_replay_runtime.log
```

结果：

- SGCP algorithm total：avg `105.24 ms`，max `127.58 ms`
- Coalition formation：avg `64.39 ms`，max `82.32 ms`
- PPS scheduling：avg `40.58 ms`，max `53.05 ms`
- Offline frame loading：avg `448.40 ms`
- Offline world build：avg `151.33 ms`

控制开销：

- Total control bytes：`187,112`
- Avg control bytes/frame：`4,563.71`
- Control / point-cloud payload：约 `0.70%`

## 待补证事项

- 找回论文旧主表 `NC/RS/MUG/FullPerception/Ours` 的原始日志、随机种子、代码提交和完整配置；找不到则应替换为当前复现结果。
- 把 `Comm. Overhead (Mbps)` 从当前 payload bytes/frame 换算到统一周期/带宽口径，并区分 point-cloud payload、detection-box exchange 和 control metadata。
- 在真实 CARLA 在线仿真中打开 `enable_topology_trigger_gate` 回归一次；这需要启动 CARLA，执行前应确认没有已有 CARLA 进程。
- 重新复查 NS3 / co-simulation 当前 HEAD，并把 safe.directory 问题解决后更新 `environment.md` 的版本快照。
