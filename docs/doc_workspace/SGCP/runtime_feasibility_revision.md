# Runtime Feasibility Revision

更新时间：2026-07-16

本文档用于回应审稿意见中关于 100 ms 协作周期实时性的质疑，并把当前可复现实验结果转成论文正文和 rebuttal 可用的保守写法。

## 实验口径

数据与配置：

- 数据集：`D:\Data\Carla\2026_07_15_01_26_56`
- 帧数：41
- CAV 数量：20
- 资源分配：`potential_game`
- cluster：`CoalitionGame`
- 子信道：10
- Conda 环境：`opencda`
- 不启动 CARLA / NS3，仅离线读取 dump 并重建 SGCP clustering + PPS。

命令：

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_replay.py
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

日志：

```text
docs\doc_workspace\SGCP\artifacts\runtime_breakdown_41f\offline_replay_runtime.log
```

## 当前结果

| Stage | Mean (ms) | Max (ms) | 是否计入在线算法周期 | Notes |
| --- | ---: | ---: | --- | --- |
| Dump frame loading | 448.40 | 513.31 | 否 | 离线 PCD/YAML 文件读取，不代表在线 CARLA sensor callback。 |
| Offline world build | 151.33 | 199.34 | 否/部分 | 离线适配层重建 `OfflineCavWorld`，在线系统已有 manager 状态。 |
| Coalition formation | 64.39 | 82.32 | 是 | `CoalitionGame.run()`，当前 Python 原型主要算法开销。 |
| Post-cluster state update | 0.24 | 0.44 | 是 | 写回 cluster head/member 与 topology state。 |
| PPS scheduling | 40.58 | 53.05 | 是 | `PotentialGame` resource allocation。 |
| Control overhead accounting | 0.03 | 0.05 | 否 | 论文统计用估算，在线不必每帧执行。 |
| SGCP algorithm total | 105.24 | 127.58 | 是 | 不含离线文件 I/O、OpenCOOD detector inference、真实传输等待。 |
| Offline total | 704.97 | 789.68 | 否 | 包含离线数据读取和重建，只用于工程回放吞吐。 |

关联统计：

- PPS 41/41 帧收敛。
- 平均 PPS iterations：3.00。
- 平均 scheduled links：10.00/frame。
- 平均 selected grids：523.90/frame。
- 控制面开销：4,563.71 bytes/frame。
- NS3 10 子信道 request-level replay：110/110 scheduled requests application callback + RLC complete，平均链路 delay 23.909 ms，P95 24.000 ms。

## 论文写作口径

当前结果不能支撑“完整端到端系统一定在 100 ms 内完成”的强断言，因为还缺少同一计时框架下的 OpenCOOD detector inference、intra-cluster fusion、inter-cluster late fusion 和真实在线 CARLA sensor callback 耗时。

可以支撑的保守结论是：

- SGCP control-plane 原型的核心 clustering + PPS scheduling 平均为 105.24 ms，接近 100 ms 协作周期。
- PPS scheduling 本身平均 40.58 ms，41/41 帧在 3 轮内收敛。
- 当前超过 100 ms 的主要原因来自 Python 原型中的 coalition formation；topology-trigger gate 可避免每个周期都重跑 cluster membership，从而把 clustering cost 摊销到 topology-change 周期。
- 离线回放总耗时 704.97 ms/frame 主要由 PCD/YAML 文件读取和离线 world 重建造成，不应作为在线系统周期耗时。

建议正文写法：

```tex
We profile the SGCP control-plane on the 20-CAV CARLA dump without launching CARLA. The measured control-plane runtime excludes offline file loading and world reconstruction, which are artifacts of the replay pipeline. The current Python prototype spends 64.39 ms on coalition formation and 40.58 ms on PPS scheduling on average, resulting in 105.24 ms per profiled control update. PPS converges within three iterations for all 41 frames. This result indicates that the unoptimized prototype is close to the 100 ms cooperation period; in online execution, cluster formation is further guarded by topology-change triggers, so the coalition update cost is not paid in every sensing cycle.
```

建议 rebuttal 写法：

```text
We added a millisecond-level runtime breakdown. In the 20-CAV replay, the SGCP control-plane prototype takes 105.24 ms per profiled update on average, with 64.39 ms for coalition formation and 40.58 ms for PPS scheduling. Offline file loading and replay-world construction take an additional 599.73 ms but are replay artifacts and are not counted as online control latency. We now state the limitation explicitly: the current evidence supports near-real-time feasibility of the Python control-plane prototype, while full end-to-end 100 ms closure still requires profiling detector inference and online CARLA callbacks. To reduce repeated cost, the revised mechanism also uses topology-change triggers so coalition formation is not executed in every cycle.
```

## 仍需补充

- OpenCOOD detector inference 的 GPU 端到端耗时。
- Intra-cluster early fusion 和 inter-cluster late fusion/NMS 耗时。
- 在线 CARLA 回调中的传感器准备、manager 更新和 control-loop 调度开销。
- topology trigger gate 打开后，在真实在线 CARLA 中实际跳过多少次 coalition formation。
