# RSU BEV Attentive Calibration

本文档汇总 reference-aligned point-slice -> leader scatter BEV -> RSU attentive BEV fusion 原型的校准结果。该路线用于判断 LGCP neural / BEV feature hierarchy 是否具备继续训练的价值；当前不作为论文主性能结果。

## 机制入口

- 代码入口：`opencda/tools/lgcp_pointpillar_rsu_bev_fusion.py`
- 输入计划：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_plan_area23_11f/area_assignment_plan.csv`
- 数据集：`D:\Data\Carla/2026_07_15_02_33_21`
- checkpoint：`pointpillar_attentive_fusion`
- 核心流程：
  - 按 LGCP planned area 裁剪 member / leader point cloud。
  - 将裁剪点云投到同一 reference / RSU lidar frame。
  - 用 PointPillar `pillar_vfe + scatter` 生成每个 area leader 的 `64 x 200 x 704` BEV scatter canvas。
  - 将 leader canvases 输入 `AttBEVBackbone + heads`，输出 RSU 端 detection。

## 主要结论

- 直接使用默认 score threshold 时，输出置信度偏低，性能接近不可用。
- Score threshold `0.01` 是当前同场景 11 帧、train5、val6 中最稳定的 AP@0.5 点。
- `mean` query 明显优于 `first_leader` / `zero`，但它只是未训练 RSU query workaround。
- AP@0.7 仍弱，说明高 IoU box quality / localization 仍不稳定。
- Sparse-cell accounting 才有讨论通信量的意义；full scatter canvas 通信量过大，不能作为 LGCP 省通信 claim。

## Threshold Sweep

Top-23 planned-area，11 frames，`query-mode=mean`：

| Score threshold | Pred boxes / frame | AP@0.3 | AP@0.5 | AP@0.7 |
| ---: | ---: | ---: | ---: | ---: |
| `0.005` | `67.454545` | `0.484971` | `0.331495` | `0.101267` |
| `0.010` | `40.909091` | `0.637777` | `0.463679` | `0.136646` |
| `0.020` | `19.636364` | `0.450929` | `0.333014` | `0.132677` |
| `0.050` | `6.818182` | `0.182482` | `0.136468` | `0.099602` |

解释：`0.005` 引入过多低质量框，`0.02/0.05` 召回不足，`0.01` 在 AP@0.3 / AP@0.5 上最好。

## Query-Mode Comparison

Top-23 planned-area，11 frames，score threshold `0.01`：

| Query mode | Pred boxes / frame | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: |
| `mean` | `40.909091` | `0.637777` | `0.463679` | `0.136646` |
| `first_leader` | `26.909091` | `0.435864` | `0.278095` | `0.067018` |
| `zero` | `24.090909` | `0.476781` | `0.261325` | `0.054635` |

解释：当前 OpenCOOD attentive fusion 会返回第一个 agent query 的融合结果。`mean` query 在未训练条件下更接近 RSU global query 的输入分布，但不能被写成完整训练过的 RSU aggregation mechanism。

## Train5-To-Val6 Check

Top-23 planned-area，`query-mode=mean`：

| Split | Score threshold | Pred boxes / frame | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train5 | `0.005` | `67.400000` | `0.546083` | `0.371240` | `0.141289` |
| train5 | `0.010` | `39.400000` | `0.657273` | `0.470975` | `0.174367` |
| train5 | `0.020` | `19.000000` | `0.437342` | `0.312345` | `0.147231` |
| train5 | `0.050` | `6.400000` | `0.168421` | `0.128286` | `0.114463` |
| val6 | `0.010` | `42.166667` | `0.649422` | `0.495974` | `0.119201` |

解释：前 5 帧会选择 `0.01`，后 6 帧使用同一阈值仍保持 AP@0.5 `0.495974`。这支持 score calibration 在同场景时间切分上可复现，但它不是多场景独立 validation。

## Communication Accounting

Top-23 11 frames：

| Route | Total bytes | Bytes / frame | Note |
| --- | ---: | ---: | --- |
| Member-to-leader raw area points | `527840` | `47985.45` | raw point slices |
| Leader full scatter BEV | `4559667200` | `414515200.00` | not communication-practical |
| Leader sparse BEV cells | `5950080` | `540916.36` | requires explicit sparse packet format |

当前 sparse-cell BEV bytes 仍高于 raw member area point bytes。若论文强调通信收益，主证据应继续使用 raw point / area-slice accounting；neural feature route 只能作为机制 feasibility 和 byte-boundary 证据。

## 论文安全口径

- 可以写：LGCP 的 neural / BEV feature hierarchy data path 已跑通，且同场景 train5-to-val6 score calibration 显示 AP@0.5 有明显信号。
- 可以写：未训练 RSU attentive aggregation 的 AP@0.7 不稳定，说明需要训练显式 RSU query / aggregation head。
- 不应写：当前 RSU BEV attentive prototype 已经提供论文主性能结果。
- 不应写：feature / BEV upload 天然比 raw area slice 更省通信。

## 下一步

- 短期论文修订：将该路线收窄为 feasibility / calibration boundary / limitation。
- 模型级路线：训练或微调 RSU aggregation head，引入显式 RSU query，并在多场景 validation split 上重新选择 score threshold。
