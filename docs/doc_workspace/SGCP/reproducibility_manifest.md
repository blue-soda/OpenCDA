# SGCP Reproducibility Manifest

更新时间：2026-07-17

本文档用于固定 SGCP 当前所有“可复现结果”的代码版本、数据、配置、命令和日志位置，并单独标出论文 `main.tex` 中旧结果尚缺原始日志的问题。

## 版本与环境

OpenCDA 当前复现实验版本：

```text
23cbc0530c18a92c0545bf776b513e3def7c2baa
```

当前已知无关 dirty 文件：

```text
D CLAUDE.md
M opencda/core/clustering/managers/clustering_scheduler.py
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

## 论文旧主表状态（不可作为当前复现结果）

`C:\Workspace\icdcs-paper\SGCP\main.tex` 曾经使用如下旧主表写作：

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

当前状态：未在 `docs/doc_workspace/SGCP` 或仓库中找到这些旧表格对应的原始日志、随机种子、代码提交和完整配置。因此这些数值不能作为“已复现结果”继续强写。当前 `main.tex` 已替换为 PAPG 主表口径：`SGCP (PAPG, 10 ch.) = 0.81/0.78/0.39`、62.54 Mbps；FullPerception/full 20-CAV early fusion 作为 centralized upper reference。

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

### SGCP PAPG main setting

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation perception_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2
```

结果：

- Frames：41
- CAVs：20
- Resource allocation：`perception_aware_potential_game`
- AP@0.3 / AP@0.5 / AP@0.7：`0.81 / 0.78 / 0.39`
- Total upload：`32,049,872 bytes`
- Payload rate：`62.54 Mbps`
- Scheduled links：`410`（10 links/frame）
- Object-level diagnostics：full-reference-detected but PAPG-missed rows 从 target-aware PG 的 106 降到 59。

说明：这是当前 SGCP 主表候选。它低于 full 20-CAV upper reference，但在近似相同通信量下高于 forced-budget random，并在 AP@0.3/AP@0.5 上高于 high-budget density selective baseline，同时 payload 更低。

### Scheduler ablations

同数据、同 SGCP inter-cluster late-fusion evaluation path：

| RA Algorithm | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Total Upload |
| --- | ---: | ---: | ---: | ---: | ---: |
| `potential_game` | 41 | 0.77 | 0.73 | 0.35 | 26,916,208 |
| `random` | 41 | 0.44 | 0.39 | 0.17 | 9,725,376 |
| `mws` | 41 | 0.31 | 0.26 | 0.11 | 9,910,032 |

说明：`random` 和 `mws` 未充分利用通信预算，只能作为 w/o-PPS 诊断，不作为主公平 baseline。

### Same-budget CAV-only selective-sharing baselines

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline communication_aware --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --max-frames 0
```

| Baseline | AP@0.3 | AP@0.5 | AP@0.7 | Total Upload | Payload Mbps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Nearest | 0.76 | 0.73 | 0.37 | 28,026,832 | 54.69 | CAV-only, same cluster/head late-fusion path |
| Density | 0.77 | 0.74 | 0.39 | 30,574,368 | 59.66 | Strong baseline |
| Communication-aware | 0.78 | 0.75 | 0.40 | 30,222,256 | 58.97 | Strong 2-member/87-grid baseline |
| Forced-budget random | 0.77 | 0.73 | 0.38 | 31,613,424 | 61.68 | 3 members/head, 117 grid budget, payload matched to PAPG |
| Density high-budget | 0.80 | 0.76 | 0.40 | 37,710,864 | 73.58 | 3 members/head, 117 grid budget |

结论：PAPG 是当前主表方法：它以 62.54 Mbps 达到 `0.81/0.78/0.39`，相比 forced-budget random 在近似相同 payload 下提升 `+0.04/+0.05/+0.01` AP；相比 high-budget density baseline，AP@0.3/AP@0.5 更高且 payload 低约 15.0%。论文主张边界仍应保持保守：PAPG 不超过 centralized full-sharing upper reference。

## NS3 可复现链路结果

### PAPG 10-subchannel scheduled replay

相关 artifact：

```text
docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\
```

结果：

- Frames：11
- Scheduled requests：110
- Skipped unscheduled demand：44
- CAM/application callback received：110/110
- RLC complete requests：110/110
- RLC TX/RX events：2970/2970
- RLC drops：0
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

- 若 reviewer 坚持 FullPerception-RSU 真实 baseline，需要重新导出带 RSU sensor 的场景；当前 full 20-CAV early fusion 只能作为 centralized upper reference。
- forced-budget random 尚未补真实 NS3 replay；如版面需要，可按 PAPG replay 协议追加 11 帧链路验证。
- 仍缺 LaTeX 编译验证；当前机器未检测到 `latexmk/pdflatex`，需要在具备 TeX 环境后检查 `main.tex` 表格宽度和引用。
- 若继续优化主表，可围绕 PAPG 剩余 missed grids（`0_-2`、`3_-1`、`0_1`、`2_-2`）做对象级可视化，而不是恢复旧不可复现主表。
