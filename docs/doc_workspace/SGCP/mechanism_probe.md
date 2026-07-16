# SGCP Mechanism Probe

更新时间：2026-07-16

本文档记录主表修复阶段的第二轮机制 probe，用同一套 cluster 和 inter-cluster late fusion 评估 head-only、SGCP grid-constrained 和 full-cluster upload，定位当前 AP 损失来自哪里。

## 代码更新

`opencda.tools.offline_inference` 新增：

```text
--sgcp-upload-mode {grid,head_only,full_cluster}
```

含义：

- `grid`：默认 SGCP 口径，只上传 PPS/grid selection 选中的 sender grid。
- `head_only`：每个 cluster head 只使用自身点云，随后做 inter-cluster late fusion。
- `full_cluster`：每个 cluster head 接收本 cluster 所有成员的完整点云，随后做 inter-cluster late fusion。

该开关只用于机制 probe。默认仍是 `grid`，不改变主实验命令。

## 命令

Head-only：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-upload-mode head_only --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_trace.csv
```

Full-cluster：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-upload-mode full_cluster --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\full_cluster_41f_trace.csv
```

默认 SGCP grid-constrained 结果来自：

```text
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_trace.csv
```

## 结果

| Mode | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Bytes / Receiver | Avg. Sources | Avg. Uploaded Sources | Avg. Uploaded Points | Avg. Selected Grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head-only | 0.26 | 0.22 | 0.09 | 0 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| SGCP grid-constrained | 0.77 | 0.73 | 0.35 | 26,916,208 | 109,415.48 | 2.67 | 1.67 | 6,838.47 | 87.32 |
| Full-cluster upload | 0.82 | 0.79 | 0.42 | 44,850,528 | 182,319.22 | 3.33 | 2.33 | 11,394.95 | 0.00 |

## 结论

- Cluster-head local-only perception 很弱，说明协同上传对 AP 有显著贡献。
- Full-cluster upload 在同一 cluster 和 inter-cluster late fusion 结构下达到 `0.82/0.79/0.42`，说明 cluster formation 和 late fusion 主体并没有崩。
- SGCP grid-constrained 使用约 60.0% 的 full-cluster payload，AP@0.5 保留约 92.4% 的 full-cluster AP，但 AP@0.7 损失明显。
- 当前主表偏低的主要嫌疑从协议链路转移到 grid/PPS 选择：选格质量、grid budget、`B_h=1` 以及 AP@0.7 所需的定位精度。

## 下一步

- 增加随机 grid selection probe，确认 SGCP grid utility 是否明显优于随机。
- 增加 fixed-cluster / all-in-one / singleton 的同 upload-mode 对照，拆分 cluster 质量影响。
- 检查 `PotentialGame.best_response()` 中 `B_h=1` 和 `max_grids_per_rb` 的实际约束，评估是否需要更灵活的 per-head member budget。
- 优化 grid utility，使其优先选择对 AP@0.7 更有贡献的近距离、高密度、低遮挡或高定位增益 grid。
