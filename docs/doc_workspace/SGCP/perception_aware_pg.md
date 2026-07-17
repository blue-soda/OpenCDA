# Perception-Aware Potential Game

## 背景

Target-grid case study 显示，旧 `potential_game + spatial_diverse` 和 `target_aware_potential_game` 的主要剩余问题不是子信道语义错误，而是调度目标不够平衡：某些帧中最佳目标视角 CAV 被覆盖大量普通 grid 的 sender 挤出；反过来，`object_aware_potential_game` 过分追逐 object peak 时又会牺牲上下文和 source diversity，导致 11 帧 AP 下降。

因此新算法不采用逐案 fallback，而是把“覆盖保持”和“目标增益”合并到一个两层 perception-aware potential-guided scheduler 中。

## 机制

入口：

```text
perception_aware_potential_game
perception_aware_pg
papg
```

核心代码：

```text
opencda/core/clustering/algorithms/resource_allocation/perception_aware_potential_game.py
```

两层调度：

1. Coverage layer：在子信道预算允许时，为每个 cluster head 分配一个高质量外部视角，保护低阈值 AP、空间上下文和 source diversity。
2. Target layer：剩余 RB 分配给 object-prototype marginal gain 最高的 sender/head 链路，优先补充漏检目标对应 grid。

两层使用同一套 object prototype / grid utility。论文中可写为 perception-aware potential scheduling：coverage term 保证每个 head 的基础外部观测，target term 对 object-like connected components、head weak grids 和 multi-view confirmation grids 给出正边际效用。

## 主结果

实验口径：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV，20 MHz / 10 channel，`rho_th=3`，`B_h=2`，SGCP constrained + inter-cluster late fusion。

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation perception_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\papg_bh2_rho3_41f_trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\object_diag_papg_bh2_rho3_41f.csv
```

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Scheduled links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG, 20MHz/10ch/rho3/`B_h=2` | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 410 |

对比：

- `target_aware_potential_game`：`0.80/0.76/0.39`，31,069,968 bytes，60.62 Mbps。
- 强 high-budget selective baseline：`0.80/0.76/0.40`，37,710,864 bytes，73.58 Mbps。
- Full 20-CAV early upper reference：`0.85/0.83/0.48`，60,838,528 bytes，118.71 Mbps。

PAPG 当前满足更稳妥的主张边界：低于 full 20-CAV 上界，但以更低通信量超过强 selective baseline 的 AP@0.3/AP@0.5。

## 诊断

对象级诊断命令：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_failure_diagnostics --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --object-diagnostics-csv docs\doc_workspace\SGCP\artifacts\object_diag_papg_bh2_rho3_41f.csv --output-dir docs\doc_workspace\SGCP\artifacts\failure_diag_papg_bh2_rho3_41f
```

结果摘要：

- missed GT count：59。
- target-aware PG 对应 missed rows：106。
- scheduled links：410，即 41 帧每帧 10 条链路。
- top missed world grids：`0_-2`、`3_-1`、`0_1`、`2_-2`。

这说明 PAPG 的收益来自更合理的 sender/head/grid 共同调度，而不是绕过预算或增加隐含通信链路。

## 待验证

- NS3 真实 socket replay 已完成 11 帧：110/110 scheduled requests application callback complete，RLC complete 110/110，PHY decode failures 0。日志路径：`docs/doc_workspace/SGCP/artifacts/papg_ns3_20260717_210304/`。
- 做短在线 CARLA+NS3 smoke test，确认 deadline-aware CP delivery 与离线 final-delivery 口径差异。
- 将 `main.tex` 中当前 coverage-aware 机制文字升级为 perception-aware two-layer potential scheduling，并把 PAPG 作为新的主表候选。

## NS3 Replay

启动 ns-3：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd /home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
```

离线 replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --upload-plan-output docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\upload_plan.csv --drain-seconds 0.3 --sync-timeout 30
```

解析结果：

| Metric | Value |
| --- | ---: |
| planned requests | 110 |
| application `cam_received` | 110 |
| matched callback ratio | 1.000 |
| RLC complete requests | 110 |
| RLC partial/no-RX requests | 0 / 0 |
| RLC TX/RX events | 2970 / 2970 |
| PHY decode failures | 0 |
| average callback delay | 23.91 ms |
| p95 callback delay | 24.00 ms |

注意：PowerShell `Start-Job` 捕获的 ns-3 stdout 默认为 UTF-16 LE，需转成 UTF-8 后再交给 `opencda.tools.ns3_log_eval`，否则解析器会读到 0 条事件。
