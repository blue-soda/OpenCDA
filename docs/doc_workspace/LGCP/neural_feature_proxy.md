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

## 后续选择

可选路线 A：继续做模型级 AP。

- 实现 affine / `grid_sample` 级 feature rotation 与 translation。
- 加入 feature normalization / calibration。
- 训练或 fine-tune leader / RSU aggregation head。
- 重新跑多帧 AP 后再判断是否能进入论文主结果。

可选路线 B：收窄为 feature-level proxy。

- 将 neural hierarchy 写作限制为 data path、coverage、byte proxy 和 limitation。
- 主感知质量证据使用 box-level hierarchy late-fusion 结果。
- 通信收益使用 raw-byte / area-slice accounting，而不是未优化 feature crop 体积。
