# LGCP Neural Feature Proxy

本文档记录当前 PointPillar neural feature hierarchy 的可用论文口径：feature-path feasibility、coverage 和 byte proxy。当前结果不能作为 model-level AP claim。

## 结论摘要

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_neural_feature_proxy_summary_area23
```

核心表：

```text
neural_feature_proxy_summary.csv
```

| Stage | Scope | Bytes / frame | Ratio vs raw member area23 | Ratio vs comm-aware area23 slice | Coverage | AP@0.5 | AP@0.7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw member area slice | Area23 11-frame mean | 47,985.45 | 1.000000 | 0.189568 |  |  |  |
| Flat comm-aware area slice | Area23 11-frame mean | 253,130.18 | 5.275144 | 1.000000 |  |  |  |
| PointPillar feature crop | Top23 first frame | 810,688.00 | 16.894453 | 3.202652 |  |  |  |
| Leader scatter fusion | Top23 first frame | 936,298.00 | 19.512121 | 3.698879 |  |  |  |
| RSU index canvas | Top23 first frame | 82,974.00 | 1.729149 | 0.327792 | 0.033161 |  |  |
| Coordinate warp nearest | Top23 first frame, ref CAV 1 | 110,883.00 | 2.310763 | 0.438047 | 0.060724 | 0.010000 | 0.000000 |
| Coordinate warp bilinear | Top23 first frame, ref CAV 1 | 149,700.00 | 3.119695 | 0.591395 | 0.060625 | 0.011364 | 0.003472 |

## 论文安全口径

- 当前 PointPillar feature crop / leader fusion 证明 neural feature hierarchy 的数据路径已经能落地。
- 未优化的 feature crop 并不天然节省通信：Top23 首帧压缩 feature crop 是 raw member area-slice 11 帧均值的 `16.89x`，也是 comm-aware flat area-slice 口径的 `3.20x`。
- RSU canvas / coordinate-warp canvas 的压缩体积较小，但它们是聚合后的中间产物，不等同于 leader upload 通信负载。
- Nearest / bilinear coordinate warp 的 AP 都很低，说明当前问题不是简单采样方式，而是 feature crop、local fusion、坐标重投影与预训练 detection head 缺少校准。
- Reference-aligned point-slice RSU BEV attentive fusion 绕开了 leader-local feature warp：先把点云切片投到统一参考系，再生成 leader scatter BEV，最后复用 `pointpillar_attentive_fusion` 的 attentive backbone/head；该链路 11 帧可运行，但 AP@0.5 仍只有 `0.136468`，说明预训练 head 对稀疏 area-leader BEV canvas 仍明显失配。

## RSU BEV attentive point-slice route

运行目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr005
```

| Stage | Scope | Member raw bytes | Leader feature bytes | Score threshold | Pred / GT mean | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RSU BEV attentive, sparse-cell accounting | Top23 11 frames, planned areas | 527,840 total | 5,950,080 sparse-cell total | 0.050 | 6.818 / 37.364 | 0.182482 | 0.136468 | 0.099602 |
| RSU BEV attentive, score-calibrated smoke | Top23 11 frames, planned areas | 527,840 total | 5,950,080 sparse-cell total | 0.010 | 40.909 / 37.364 | 0.637777 | 0.463679 | 0.136646 |

该路线比 nearest / bilinear feature warp 更接近 OpenCOOD attentive checkpoint 的原生输入假设，因为所有点云在 VFE/scatter 前已经处于同一个 reference lidar frame。阈值 sweep 显示 score calibration 可以显著改善 AP@0.5；前 5 帧 train split 会选择 `0.01`，后 6 帧 val split 中同一阈值仍达到 AP@0.5 `0.495974`，支持校准方向不是完全偶然。但 AP@0.7 仍弱，且同场景时间切分不等于多场景独立 validation。因此它只能证明机制链路与校准潜力，不能替代 box-level hierarchy late-fusion 作为主感知质量证据。

Query-mode 对照显示当前 `mean` query 是较合理的未训练原型默认值：在同一 `0.01` threshold 下，`mean` 的 AP@0.5/AP@0.7 为 `0.463679/0.136646`，高于 `first_leader` 的 `0.278095/0.067018` 和 `zero` 的 `0.261325/0.054635`。但这仍是 workaround，不是训练过的 RSU query mechanism。

## 后续选择

可选路线 A：继续做模型级 AP。

- 优先从 reference-aligned point-slice BEV route 出发，而不是继续扩大 nearest / bilinear feature warp。
- 加入 feature normalization / score calibration。
- 训练或 fine-tune leader / RSU aggregation head，使 detection head 见过稀疏 area-leader BEV canvas；同时用显式 RSU query 替代当前 `mean` query workaround。
- 重新跑多帧 AP 后再判断是否能进入论文主结果。

可选路线 B：收窄为 feature-level proxy。

- 将 neural hierarchy 写作限制为 data path、coverage、byte proxy 和 limitation。
- 主感知质量证据使用 box-level hierarchy late-fusion 结果。
- 通信收益使用 raw-byte / area-slice accounting，而不是未优化 feature crop 体积。
