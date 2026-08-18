# RSU BEV Aggregation Training Plan

本文档说明如果继续把 LGCP neural / BEV feature hierarchy 推进为模型级结果，需要训练什么、复用什么、改哪些代码，以及工作量和风险。结论先行：当前不能只靠 `pointpillar_attentive_fusion` checkpoint + 后处理阈值作为主性能 claim；需要训练显式 RSU query / aggregation head。

## 当前证据

已有原型见 `rsu_bev_attentive_calibration.md`：

- reference-aligned point-slice -> leader scatter BEV -> RSU attentive BEV fusion 链路已跑通。
- `query-mode=mean`、score threshold `0.01` 时，Top-23 11 frames AP@0.5 达到 `0.463679`。
- train5 选择 `0.01`，val6 AP@0.5 为 `0.495974`。
- AP@0.7 仍弱，val6 为 `0.119201`。
- 当前 `mean` query 是 workaround，不是训练过的 RSU query mechanism。

## 可复用代码

- OpenCOOD 训练入口：
  - `opencood/opencood/tools/train.py`
  - `opencood/opencood/tools/train_utils.py`
- 现有模型模块：
  - `opencood/opencood/models/point_pillar_intermediate.py`
  - `opencood/opencood/models/sub_modules/att_bev_backbone.py`
  - `opencood/opencood/models/fuse_modules/self_attn.py`
- 现有 loss：
  - `opencood/opencood/loss/point_pillar_loss.py`
- 当前 LGCP 原型入口：
  - `opencda/tools/lgcp_pointpillar_rsu_bev_fusion.py`

## 为什么不能只改 YAML

OpenCOOD 默认 `IntermediateFusionDataset` 的训练 batch 语义是：围绕一个 ego / reference agent，聚合多个 CAV 的 `processed_lidar`，并用 `record_len` 表示参与 agent 数量。当前 LGCP RSU-BEV route 的 batch 语义不同：

- 一个样本对应一个 timestamp 的多个 LGCP area leader packet。
- 每个 packet 来自 area-specific point slice，而不是完整 CAV point cloud。
- RSU query 不是真实 CAV ego，而是需要显式定义的 global query。
- Label 应该是 planned areas 内或 RSU reference frame 下的 detection label。

因此训练需要新增 LGCP 专用 dataset / collate / model wrapper，不能直接用现有 `point_pillar_intermediate_fusion.yaml` 训练。

## 推荐实现路线

### Phase 1：离线样本导出

目标：把当前在线构造过程固化为训练样本，避免训练时反复读 PCD / 重算 area crop。

建议新增：

- `opencda/tools/lgcp_rsu_bev_training_sample_export.py`

当前状态：已实现并完成 smoke / 11-frame export。

已验证输出：

```text
docs/doc_workspace/LGCP/experiments/rsu_bev_training_samples/20260720_rsu_bev_sparse_smoke_1f2a
docs/doc_workspace/LGCP/experiments/rsu_bev_training_samples/20260720_rsu_bev_sparse_area23_11f
```

11-frame Top-23 summary：

| Item | Value |
| --- | ---: |
| Frames | `11` |
| GT boxes, planned areas | `411` |
| Compressed sparse sample NPZ bytes | `1439391` |
| Member raw area point bytes | `527840` |
| Leader sparse feature bytes | `5950080` |

每个样本保存：

- `leader_scatter`: `N_area x 64 x 200 x 704`，float16 或 sparse representation。
- `record_len`: `[N_query_plus_area]`。
- `rsu_query`: `mean` / learnable-zero / explicit query seed。
- `label_dict`: PointPillar anchor label，沿用 OpenCOOD postprocessor 生成。
- `metadata`: scenario id、timestamp、area ids、leader ids、reference pose、packet byte accounting。

当前导出版本保存的是 sparse BEV cells、planned-area GT boxes 和 metadata；`label_dict` 建议在 Phase 2 dataset wrapper 中由 postprocessor 动态生成，避免样本格式与具体 anchor 设置强耦合。

产物目录建议：

```text
docs/doc_workspace/LGCP/experiments/rsu_bev_training_samples/<run_id>/
```

### Phase 2：Dataset / collate

建议新增在 OpenCDA 工具侧或 OpenCOOD 扩展侧，避免破坏 SGCP：

- `opencda/core/ml_libs/lgcp_rsu_bev_dataset.py` 或
- `opencood/opencood/data_utils/datasets/lgcp_rsu_bev_dataset.py`

当前状态：已新增 `opencda/core/ml_libs/lgcp_rsu_bev_dataset.py`。

已验证：

- sparse-only 读取 Top-23 11-frame samples。
- 1-frame 2-area smoke dense reconstruction。
- `query-mode=mean` 后 `spatial_features` 从 `2 x 64 x 200 x 704` 变为 `3 x 64 x 200 x 704`。
- `record_len=[3]`。
- label_dict 可由 OpenCOOD `VoxelPostprocessor` 生成：
  - `pos_equal_one`: `1 x 100 x 352 x 2`
  - `targets`: `1 x 100 x 352 x 14`

最低可行版本：

- 读取离线 `leader_scatter`。
- 动态拼接 batch，限制 batch size 为 1 或实现 padding。
- 返回 `ego['spatial_features']`、`ego['record_len']`、`ego['label_dict']`。

### Phase 3：Model wrapper

建议新增：

- `opencood/opencood/models/lgcp_rsu_bev_attentive.py`

当前状态：已新增 `opencood/opencood/models/lgcp_rsu_bev_attentive.py`。

已验证：

- `train_utils.create_model` 可以通过 `core_method: lgcp_rsu_bev_attentive` 创建模型。
- `query_mode=input` 可复用 dataset 已构造的 mean query stack。
- `query_mode=learnable_channel` 可从 `leader_features + leader_record_len` 构造显式 RSU query。
- 1-frame 2-area smoke forward：
  - `psm`: `1 x 2 x 100 x 352`
  - `rm`: `1 x 14 x 100 x 352`
  - PointPillarLoss 可计算。
  - `learnable_channel` RSU query 有非零 gradient。

模型结构：

- 输入已是 scatter BEV，不再调用 `PillarVFE` / `PointPillarScatter`。
- 复用 `AttBEVBackbone`。
- 新增显式 RSU query：
  - 简单路线：learnable `1 x 64 x 200 x 704` query。
  - 省显存路线：learnable low-rank / per-channel query，再 broadcast。
  - 稳定路线：初始化为 training split 的 mean scatter，再允许 finetune。
- 复用或微调 `cls_head` / `reg_head`。

冻结策略：

- Option A：冻结 `AttBEVBackbone`，只训练 RSU query + heads。
- Option B：解冻最后一层 fusion module + heads。
- Option C：全量 fine-tune backbone + heads。

推荐从 Option A 开始，若 AP@0.7 仍弱再进入 Option B。

当前仍缺：

- YAML 配置文件。
- 多场景 train / validation split。

### Phase 3.5：Train-loop smoke

当前状态：已新增 `opencda/tools/lgcp_rsu_bev_train_smoke.py`。

已验证输出：

```text
docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_smoke_1f2a
```

Smoke 设置：

- train root / val root 均为 `20260720_rsu_bev_sparse_smoke_1f2a`
- `query_mode=learnable_channel`
- `freeze_mode=query_heads`
- `device=cpu`
- `max_train_steps=1`
- `max_val_steps=1`

结果：

| Metric | Value |
| --- | ---: |
| Trainable parameters | `6224` |
| Train loss | `11.980690373` |
| Val loss | `0.610613982` |
| psm shape | `1 x 2 x 100 x 352` |
| rm shape | `1 x 14 x 100 x 352` |

解释：该 smoke 只证明 dataset -> model -> loss -> optimizer -> val loss trace 能运行；由于 train/val 使用同一 1-frame 2-area sample，不能作为任何性能提升证据。

### Phase 3.6：Top-5 train5 / val6 smoke

当前状态：已完成。

动机：

- Top-23 每帧 `23 x 64 x 200 x 704` dense reconstruction 在 CPU / 内存上压力较大。
- 先限制为 Top-5 planned areas，验证真实 train split / val split 的训练循环，而不是继续使用同一帧作为 train/val。

样本：

| Split | Output root | Frames | GT boxes | Sample npz bytes | Member upload bytes | Leader sparse feature bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train5 | `20260720_rsu_bev_sparse_top5_train5` | `5` | `60` | `143715` | `85696` | `589056` |
| val6 | `20260720_rsu_bev_sparse_top5_val6` | `6` | `72` | `199553` | `115216` | `829056` |

训练设置：

- output root：`docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_train5_val6_smoke`
- `query_mode=learnable_channel`
- `dataset_query_mode=mean`
- `freeze_mode=query_heads`
- `device=cpu`
- `epochs=1`
- `max_train_steps=2`
- `max_val_steps=2`
- trainable parameters: `6224`

结果：

| Phase | Frame | Loss |
| --- | --- | ---: |
| train | `000060` | `6.825837248` |
| train | `000062` | `4.661619867` |
| val | `000070` | `0.902511302` |
| val | `000072` | `0.847593252` |

输出形状：

- `psm`: `1 x 2 x 100 x 352`
- `rm`: `1 x 14 x 100 x 352`

AP hook：

- `opencda/tools/lgcp_rsu_bev_train_smoke.py` 已新增 `--eval-ap` 和 `--postprocess-score-threshold`。
- 默认不启用 AP hook；启用时只在 validation 阶段调用 OpenCOOD postprocessor 并统计 AP。
- Top-5 val2 smoke：
  - output root：`docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_train5_val6_ap_smoke`
  - score threshold：`0.01`
  - val samples evaluated：`2`
  - GT boxes：`24`
  - predicted samples：`57`
  - AP@0.3 / AP@0.5 / AP@0.7：`0.758776 / 0.682186 / 0.285283`
- Top-5 full val6 smoke：
  - output root：`docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_train5_val6_full_ap_smoke`
  - score threshold：`0.01`
  - train steps / val samples evaluated：`5 / 6`
  - GT boxes：`72`
  - predicted samples：`166`
  - train final loss / val final loss：`4.150855602 / 0.919490865`
  - AP@0.3 / AP@0.5 / AP@0.7：`0.563040 / 0.499721 / 0.164391`
- Top-5 val6 threshold sweep：
  - output root：`docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_val6_threshold_sweep`
  - thresholds：`0.005, 0.01, 0.02, 0.05, 0.1`
  - best AP@0.5 threshold：`0.05`

| Score threshold | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 |
| ---: | ---: | ---: | ---: | ---: |
| `0.005` | `420` | `0.337772` | `0.301497` | `0.107598` |
| `0.01` | `166` | `0.563040` | `0.499721` | `0.164391` |
| `0.02` | `96` | `0.808087` | `0.717947` | `0.226872` |
| `0.05` | `67` | `0.820811` | `0.806983` | `0.245818` |
| `0.1` | `40` | `0.555556` | `0.555556` | `0.201811` |

解释：

- 该结果证明 LGCP sparse BEV samples 可以进入一个独立 train/val 训练闭环。
- AP hook 已闭环；完整 val6 相比 val2 明显回落，说明小样本 val2 会高估模型级效果。
- 多阈值 sweep 已闭环；当前 Top-5 val6 对 score threshold 高度敏感，`0.05` 的 AP@0.5 最好，但 AP@0.7 仍弱。
- 当前仍没有 Top-23 / 多场景验证，因此不能作为论文模型性能 claim。
- 若继续推进，下一步应把 Top-5 AP hook 扩展到 Top-23 或多场景；若短期修订论文，应继续把 RSU BEV route 写成 feasibility / limitation。

### Phase 4：Calibration / evaluation

最低验证：

- train5 / val6 同场景 sanity check。（Top-5 train-loop + full val6 AP hook + threshold sweep 已完成；仍缺 Top-23 / 多场景 AP calibration）
- 多场景 validation split。
- threshold 在 validation 上选择，再固定到 test。

论文级验证需要：

- 不同 scenario / seed。
- 与 box-level hierarchy、flat comm-aware baseline 对齐。
- 同一 communication accounting 口径：raw area slice、sparse BEV packet、control-plane overhead 分开报。

## 工作量评估

| 路线 | 工作量 | 难度 | 风险 | 产出 |
| --- | --- | --- | --- | --- |
| 只做阈值校准 | 0.5-1 天 | 低 | 高：无法支撑主 claim | calibration boundary |
| 离线样本导出 + head/query 训练 | 3-5 天 | 中 | 中：可能 AP@0.7 仍低 | 可判断模型级路线是否值得 |
| Dataset/model wrapper + 多场景训练 | 1-2 周 | 中高 | 中高：数据量和显存压力 | 可能形成论文补充结果 |
| 端到端重训 area-aware PointPillar | 2-4 周 | 高 | 高：工程和调参成本大 | 最完整但最慢 |

## 主要风险

- 训练数据不足：当前 `lgcp_carla` dump 只有 11 帧，不足以训练 head。
- AP@0.7 弱：可能来自 area slice coverage、anchor label mismatch、localization head 分布偏移。
- 通信量不占优：sparse BEV packet 仍大于 raw area point slice，需要更强稀疏编码或压缩。
- Query 语义不稳：未训练 `mean` query 不能写成 RSU attention 机制。
- 多场景数据依赖：本地没有 OPV2V / V2XSet dataset，需要远端 `mindspore-186` 或重新导出更多 CARLA dump。

## 推荐决策

短期论文修订优先：

- 使用 box-level hierarchy late-fusion 作为主感知质量证据。
- 使用 raw / area-slice accounting 作为通信收益证据。
- 将 RSU BEV attentive route 写成 feasibility、calibration boundary 和 future work。

## V2X-ViT Compressed Feature Route

2026-07-20 新增 `opencda/tools/lgcp_v2xvit_feature_probe.py`，用于检查 `pointpillar_v2xvit_fusion` 是否比 `pointpillar_attentive_fusion` 的 scatter BEV 更适合 LGCP leader-to-RSU feature packet。

模型结构：

- `pillar_vfe`
- `scatter`
- `BaseBEVBackbone`
- `shrink_conv`
- `NaiveCompressor.encoder`
- `V2XTransformer`
- detection heads

通信解释：

- 当前 attentive route 传的是 `scatter` 后的 `64 x H x W` feature，层级太早。
- V2X-ViT route 可以传 `NaiveCompressor.encoder` 的 bottleneck latent，当前配置为 `8 x 48 x 176`。
- 如果只传 planned area 对应的 compressed crop，而不是 full latent canvas，第二跳 feature bytes 有机会低于第一次 raw area point slice。

Top-23 11-frame probe：

| Payload | Total bytes | Bytes/frame | Ratio vs member raw area points |
| --- | ---: | ---: | ---: |
| Member raw area points | `527840` | `47985.45` | `1.00x` |
| Scatter sparse feature | `5949952` | `540904.73` | `11.27x` |
| V2X-ViT compressed full latent | `34197504` | `3108864.00` | `64.79x` |
| V2X-ViT compressed area crop | `248912` | `22628.36` | `0.47x` |

结论：

- `pointpillar_v2xvit_fusion` 比当前 attentive scatter route 更适合继续探索通信友好的 feature hierarchy。
- `pointpillar_cobevt_fusion` 当前 checkpoint 使用 camera BEV / segmentation 配置，不适合直接作为 LiDAR point-slice detection route。
- 已新增 `opencda/tools/lgcp_v2xvit_rsu_detection_probe.py`，完成 compressed latent RSU assembly、decoder、V2XTransformer fusion 和 detection AP smoke。

Top-5 1-frame detection smoke：

| Packet mode | Pred / GT boxes | AP@0.3 | AP@0.5 | AP@0.7 | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| crop | `18 / 12` | `0.468204` | `0.065476` | `0.000000` | communication-friendly packet |
| full latent | `4 / 12` | `0.333333` | `0.229167` | `0.000000` | upper-bound payload for same latent layer |

Threshold sweep:

- Crop mode thresholds `0.005/0.01/0.02/0.05/0.1` all produce `18` predictions and AP@0.3/0.5/0.7 `0.468204/0.065476/0.000000`.
- Full latent thresholds `0.005/0.01/0.02/0.05/0.1` all produce `4` predictions and AP@0.3/0.5/0.7 `0.333333/0.229167/0.000000`.
- Therefore score threshold calibration is not the immediate bottleneck for this V2X-ViT RSU route.

Query-mode sweep:

| Packet mode | Query mode | Pred / GT boxes | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | ---: | ---: | ---: | ---: |
| crop | mean | `18 / 12` | `0.468204` | `0.065476` | `0.000000` |
| crop | zero | `13 / 12` | `0.083333` | `0.037037` | `0.000000` |
| crop | first | `7 / 12` | `0.583333` | `0.476190` | `0.113095` |
| full latent | mean | `4 / 12` | `0.333333` | `0.229167` | `0.000000` |
| full latent | zero | `0 / 12` | `0.000000` | `0.000000` | `0.000000` |
| full latent | first | `15 / 12` | `0.980769` | `0.731481` | `0.273810` |

Multi-frame / area-count check:

| Setting | Pred / GT samples | AP@0.3 | AP@0.5 | AP@0.7 | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-5 11-frame crop+first | `66 / 132` | `0.500000` | `0.369657` | `0.081239` | `4.88 KB/frame` |
| Top-10 1-frame crop+first | `2 / 22` | `0.090909` | `0.090909` | `0.022727` | not summarized |

Leader-query selection diagnostic:

| Setting | Selected query | Pred / GT samples | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | ---: | ---: | ---: | ---: |
| Top-10 1-frame crop+first+max_area_points | area `12_9`, leader `5` | `2 / 22` | `0.090909` | `0.090909` | `0.022727` |
| Top-10 1-frame crop+first+max_group_size | area `12_9`, leader `5` | `2 / 22` | `0.090909` | `0.090909` | `0.022727` |

解释：

- V2X-ViT compressed route 已证明通信量和接口可行。
- Query mode 是当前检测质量的关键因素；`first` 明显优于 `mean/zero`。
- `first` 不是最终 RSU global query 机制；Top-10 首帧退化说明它对 area/leader 顺序和覆盖范围敏感。
- 简单 leader-query selection 没有修复 Top-10 退化；`max_area_points` 和 `max_group_size` 仍与 plan-order Top-10 相同。
- 下一步应训练显式 RSU query / head，而不是继续手工选择 leader query 或直接把 `first` 当作 LGCP global aggregation。

### Phase 5：V2X-ViT explicit RSU query/head smoke

当前状态：已新增并完成 1-step train smoke，以及 1 train frame / 1 val frame 的 AP-hook smoke。

新增代码：

- `opencood/opencood/models/lgcp_v2xvit_rsu.py`
  - 输入 `compressed_features` 或 `decoded_features`。
  - 复用 `NaiveCompressor.decoder`、`V2XTransformer`、`cls_head`、`reg_head`。
  - 支持 `input` / `mean` / `zero` / `learnable_channel` query。
- `opencda/tools/lgcp_v2xvit_rsu_train_smoke.py`
  - 复用 LGCP area point-slice 构造、V2X-ViT feature encoder 和 planned-area GT label。
  - 默认 `freeze_mode=query_heads`，只训练 RSU query 和 detection heads。

Smoke 输出：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_train_smoke_1f2a
```

| Metric | Value |
| --- | ---: |
| Train steps | `1` |
| Planned areas / valid leader features | `2 / 2` |
| GT boxes | `6` |
| Compressed input shape | `2 x 8 x 48 x 176` |
| Output psm / rm | `1 x 2 x 48 x 176` / `1 x 14 x 48 x 176` |
| Trainable parameters | `4368` |
| Train final loss | `24.775737337` |
| Query gradient norm | `348.719512939` |

Train-val AP-hook smoke：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_smoke_1f2a
```

| Metric | Value |
| --- | ---: |
| Train / val timestamps | `000060 / 000062` |
| Train / val steps | `1 / 1` |
| Train / val loss | `24.375071751 / 22.165417312` |
| Val predictions / GT boxes | `7 / 6` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.01` | `0.055556 / 0.055556 / 0.000000` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.05` | `0.055556 / 0.055556 / 0.000000` |

Top-5 train5 / val6 query-head smoke：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_top5_train5_val6
```

| Metric | Value |
| --- | ---: |
| Train / val steps | `5 / 6` |
| Train / val loss | `13.608617907 / 13.522501738` |
| Query gradient norm | `177.555145264` |
| Val GT / pred samples | `72 / 86` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.01` | `0.053819 / 0.030382 / 0.000000` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.05` | `0.053819 / 0.030382 / 0.000000` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.1` | `0.053819 / 0.030382 / 0.000000` |

`query_fusion_heads` sanity check：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_1f2a_query_fusion_heads
```

| Metric | Value |
| --- | ---: |
| Trainable parameters | `5490569` |
| Train / val loss | `24.629682239 / 9.806766111` |
| Query gradient norm | `348.058624268` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.01` | `0.041667 / 0.041667 / 0.000000` |

Top-5 train5 / val6 `query_fusion_heads` smoke：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_top5_train5_val6_query_fusion_heads
```

| Metric | Value |
| --- | ---: |
| Trainable parameters | `5490569` |
| Train / val loss | `2.266083973 / 2.659839972` |
| Query gradient norm | `9.641363144` |
| Val GT / pred samples, threshold `0.01` | `72 / 96` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.01` | `0.018649 / 0.003945 / 0.000000` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.05` | `0.018649 / 0.003945 / 0.000000` |
| AP@0.3 / AP@0.5 / AP@0.7, threshold `0.1` | `0.000694 / 0.000000 / 0.000000` |

解释：

- 这一步证明 V2X-ViT compressed route 已经能做显式 RSU query/head 微调。
- 极小 train/val smoke 与 Top-5 train5 / val6 都不能作为论文性能结果，但 Top-5 结果已经足以说明 `query_heads` 和 `query_fusion_heads` 都无法在当前小数据设置下校准 V2X-ViT compressed crop route。
- 相比 PointPillar scatter route，V2X-ViT route 的优势是第二跳可以统计 compressed area crop bytes；劣势是 query/head 语义必须重新训练。
- `query_fusion_heads` loss 明显下降但 AP 更低，说明当前 loss/label 与 planned-area AP 之间存在 mismatch。

继续研究优先：

- 先实现 Phase 1 / Phase 3 的最小训练闭环。
- 短期停止把 V2X-ViT compressed crop route 作为主性能路线；保留 feature byte-boundary、接口可行性和 limitation 证据。
- 若未来继续该路线，需要更大训练集、重新审视 loss / planned-area label alignment，而不是继续只扩大当前 smoke。

### Phase 6：V2X-ViT native intermediate fusion from area point crops

当前状态：已新增并完成 Top-5 1-frame / 11-frame smoke。

动机：

- Phase 5 的问题主要来自手工裁剪 compressed latent，并把 area leader packet 当作 V2X-ViT agent。
- 更合理的路线是只在点云通信层做 LGCP area crop，后续完全复用原 OpenCOOD V2X-ViT intermediate fusion。
- 这样 `record_len` 仍表示真实 agents，第 0 个 slot 仍是 ego/reference，模型语义更接近 checkpoint 训练目标。

新增代码：

- `opencda/tools/lgcp_v2xvit_area_point_crop_eval.py`
  - RSU 可通过 `--ego-cav-id -1` 作为 ego slot。
  - 每个 CAV/RSU 的 `lidar_np` 先裁剪到 planned-area union。
  - 后续调用 OpenCOOD `get_item_test -> collate_batch_test -> native inference`。
  - 输出 area point communication bytes 和 planned-area AP。

重要 caveat：

- 真实 RSU LiDAR pose 为 `z=12m`，而 `pointpillar_v2xvit_fusion` checkpoint 是车载 LiDAR range。
- 直接把高架 RSU 作为 ego 会让地面点投影到 `z≈-10m`，被预处理裁掉。
- 当前 smoke 使用 `--reference-z-override 2.0`，保留 RSU x/y/yaw，但用车载高度作为 OpenCOOD reference z。

Top-5 results：

| Setting | Pred / GT | AP@0.3 | AP@0.5 | AP@0.7 | CAV upload bytes/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-frame, RSU ego, z=2.0 | `2 / 9` | `0.222222` | `0.222222` | `0.111111` | `88704.00` |
| 11-frame, RSU ego, z=2.0 | `21 / 85` | `0.228011` | `0.208964` | `0.011765` | `100766.55` |

解释：

- 该 route 明显比 Phase 5 compressed latent RSU route 健康，AP 没有崩到接近 0。
- AP@0.7 仍弱，且当前只覆盖 Top-5 areas / max 5 agents / 单场景 11 帧。
- 它更符合“点云区域裁剪通信 + 原模型 intermediate fusion”的 LGCP 机制解释。

继续研究优先：

- 对该 route 做 score threshold sweep。
- 扩展到 Top-10 / Top-23 areas，并记录 max-cav 限制下的 agent selection 影响。
- 与 box-level hierarchy、PointPillar scatter BEV RSU attentive 和 flat selective baselines 做同口径 AP/bytes 对照。
