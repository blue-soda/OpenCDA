# LGCP 两跳融合路线当前结论

本文档整理当前 LGCP model-level hierarchy 的两条主要技术路线：

1. 第一跳早期融合，第二跳中期特征融合。
2. 第一跳 Where2comm 中期特征融合，第二跳 Where2comm 中期特征融合。

这里的“第一跳”指 member CAV 到 leader CAV；“第二跳”指 leader CAV 到 RSU。当前结论不是最终论文表格，而是用于决定后续实现与论文叙事边界的工作记录。

## 一、第一跳早期融合 + 第二跳中期特征融合

### 路线定义

这条路线的含义是：

1. member CAV 按 LGCP area 上传局部点云切片。
2. leader CAV 将本 area 内多个 member 的点云统一到 leader 或 reference 坐标系。
3. leader 用点云生成 BEV / backbone feature。
4. leader 上传 BEV 中期特征 packet 给 RSU。
5. RSU 侧用中期特征融合模块进行全局融合和检测。

它本质上是“第一跳传 raw point slice，第二跳传 neural feature”。这和用户当前更认可的直觉一致：第一跳传输按 area 限定的点云，第二跳传输 backbone 之后、fusion module 之前的特征。

### 当前实现入口

主要相关代码和实验入口：

- `C:\Workspace\OpenCDA\opencda\tools\lgcp_pointpillar_rsu_bev_fusion.py`
- `C:\Workspace\OpenCDA\opencda\tools\lgcp_where2comm_leader_feature_fusion.py`
- `C:\Workspace\OpenCDA\opencda\tools\lgcp_reassign_limited_leaders.py`
- SGCP attentive-derived model dir：
  - `C:\Workspace\OpenCDA\docs\doc_workspace\LGCP\experiments\model_dirs\pointpillar_intermediate_from_sgcp_attentive_early`
- 当前主诊断数据集：
  - `D:\Data\Carla\2026_07_22_22_00_04`
  - 普通十字路口，`10 CAV + 1 RSU + 10 background vehicles`，21 帧。

### 当前最好结果

当前质量最好的结果来自 SGCP attentive-derived early 权重移植后的 leader-BEV feature -> RSU attentive fusion 路线。

普通十字路口 10-CAV、Top-10 area、未限制 leader 数的主诊断：

| 口径 | AP@0.3 | AP@0.5 | AP@0.7 | 第一跳 member 点云 | 第二跳 sparse BEV feature |
| --- | ---: | ---: | ---: | ---: | ---: |
| planned-area | 0.868668 | 0.797311 | 0.733363 | 17.57 KB/frame | 211.08 KB/frame |
| full-scope | 0.813771 | 0.746923 | 0.687017 | 17.57 KB/frame | 211.08 KB/frame |

限制最多 5 个 leader 后的 sweep 中，`K=5` 是当前推荐点：

| Leader 上限 | 口径 | AP@0.3 | AP@0.5 | AP@0.7 |
| ---: | --- | ---: | ---: | ---: |
| 3 | full-scope | 0.710024 | 0.700737 | 0.578433 |
| 4 | full-scope | 0.811164 | 0.748394 | 0.697697 |
| 5 | full-scope | 0.811164 | 0.753897 | 0.705270 |
| 3 | planned-area | 0.809229 | 0.798743 | 0.660574 |
| 4 | planned-area | 0.865886 | 0.798881 | 0.744764 |
| 5 | planned-area | 0.865886 | 0.804755 | 0.752848 |

这个结果的意义是：在一个不太难、可控的普通十字路口场景里，LGCP area + leader hierarchy 的感知质量是说得过去的，尤其 AP@0.5 和 AP@0.7 都比较稳定。

### 第二跳 Where2comm sparse selection 对照

同一类“第一跳早期融合 + 第二跳中期特征融合”路线下，也测试了第二跳使用 Where2comm sparse BEV-cell selection 的版本。此时第一跳仍是 member CAV 按 area 上传点云，leader 生成 feature packet；区别在于 RSU 侧使用 `where2comm_10e` 的 objectness selector，并与 LGCP area mask 做交集，以减少第二跳 BEV feature cells。

普通十字路口 10-CAV、fine-grid Top-10、`K=5` leader reassignment、`lgcp_area_objectness + dilation1`：

| 第二跳融合方式 | 口径 | AP@0.3 | AP@0.5 | AP@0.7 | 第一跳 member 点云 | 第二跳 feature |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SGCP attentive-derived sparse BEV | planned-area | 0.865886 | 0.804755 | 0.752848 | 36.37 KB/frame | 214.58 KB/frame |
| SGCP attentive-derived sparse BEV | full-scope | 0.811164 | 0.753897 | 0.705270 | 36.37 KB/frame | 214.58 KB/frame |
| Where2comm area-objectness sparse selection | planned-area | 0.759775 | 0.743407 | 0.381554 | 36.37 KB/frame | 184.12 KB/frame / 15.083032 Mbps |
| Where2comm area-objectness sparse selection | full-scope | 0.643896 | 0.629983 | 0.323233 | 36.37 KB/frame | 184.12 KB/frame / 15.083032 Mbps |

解释：

- Where2comm sparse selection 的第二跳通信量低于 SGCP attentive-derived sparse BEV，约 `184.12 KB/frame` vs `214.58 KB/frame`。
- 相比未选择的 dense multiscale BEV feature，Where2comm sparse selection 也确实大幅降低通信量。当前 dense multiscale BEV 约 `36960 KB/frame`，Where2comm selected feature 约 `184.12 KB/frame`，只保留约 `0.50%`，等价于约 `200.7x` 压缩，通信量降低约 `99.50%`。
- 但 Where2comm 的高 IoU 感知质量明显弱，尤其 AP@0.7 从 SGCP attentive-derived 的 `0.705270` / `0.752848` 降到 `0.323233` / `0.381554`。
- 因此 Where2comm sparse selection 当前更适合作为 communication-aware 对照，而不是第一部分的主质量结果。它证明第二跳 feature-cell selection 可以进一步压通信，但现有 checkpoint / selector / head 在该 LGCP leader packet 语义下还没有达到 SGCP attentive-derived 路线的检测质量。

与原始点云的关系需要分清口径：

| 对比对象 | 点数 / 通信量 | 与 Where2comm 二跳比较 |
| --- | ---: | --- |
| 当前 LGCP fine-grid K=5 area raw point slice | 约 `2328 points/frame`，即 `36.37 KB/frame` | raw slice 更小，Where2comm 约为其 `5.06x` |
| 一个 `56000 points/s, 10Hz` LiDAR 满帧 | `5600 points/frame`，即 `89.60 KB/frame` | 单个满帧仍小于 `184.12 KB/frame` |
| 当前 Where2comm selected feature | `184.12 KB/frame` | break-even payload |
| dense multiscale BEV feature | `36960 KB/frame` | Where2comm 只占约 `0.50%` |

按 OpenCOOD 点云格式 `x,y,z,intensity = 4 float32 = 16 bytes/point`，Where2comm 二跳 `184.12 KB/frame` 对应的 raw-point break-even 为：

```text
184.12 KB / 16 bytes ~= 11784 points/frame
```

也就是说，只有当同一统计口径下的 raw point slice 超过约 `1.18 万点/frame`，raw point 通信量才会高于当前 Where2comm selected feature。若按 `56000 points/s`、`10Hz` 的单 LiDAR 满帧估计，`5600 points/frame` 约为 `89.60 KB/frame`，仍低于 Where2comm 二跳；需要约 `2.1` 个这样的满帧点数才到 break-even。当前普通十字路口实验里记录的 `2328 points/frame` 是“被选中 area/member 上传的点云切片总量”，不是单个 LiDAR 的满帧点数，所以它比 `5600 points/frame` 更小是合理的。另需注意，仓库 `default.yaml` 的 base lidar 是 `56000 points/s, 10Hz`，而当前 `lgcp_carla_intersection_easy.yaml` 配置中写的是 `800000 points/s, 20Hz`；因此论文或报告中引用 LiDAR 采样率时必须绑定具体场景配置，避免把 sensor full-frame 点数和 LGCP selected area-slice 点数混用。

### 当前主要问题

最大问题是通信量叙事不干净。

第一跳点云切片很小，但第二跳 BEV feature 仍明显更大。fine-grid K=5 下，第一跳约 `36.37 KB/frame`，第二跳 SGCP attentive-derived sparse BEV feature 约 `214.58 KB/frame`；换成 Where2comm sparse selection 后可降到约 `184.12 KB/frame`，但仍高于第一跳 raw point slice。这意味着这条路线当前更像“感知质量验证路线”，还不能自然证明“中期特征一定比原始点云更省通信”。

第二个问题是 feature payload 不是严格意义上的已训练压缩码流。当前 sparse BEV feature 是把 BEV cell 做区域选择后保存特征值，仍然带有较高通道维度。它比 dense full BEV 小很多，但不一定比稀疏点云切片小。

第三个问题是坐标参考系必须非常谨慎。leader 生成的 BEV feature 如果处在 leader-local 坐标系，RSU 侧融合前必须显式 warp 到 RSU/global/reference BEV。最近两跳 Where2comm 诊断显示：把 leader-local fused feature 直接给 RSU 处理会崩；强制生成在 RSU/reference 坐标系后，同帧 AP 从全 0 恢复到非 0。

### 当前难点

这条路线的难点集中在三件事：

1. 如何把第二跳 BEV feature 真正压小。
2. 如何保证 leader feature 到 RSU feature 的坐标对齐。
3. 如何让论文叙事不从网络协同感知滑向深度模型训练。

如果继续推进，优先级应是 feature-cell budget、量化、通道压缩、或 sparse feature 编码，而不是盲目重训一个大模型。因为这篇论文的核心仍是 LGCP 的 local-to-global area hierarchy 和网络调度机制，而不是提出新的深度融合网络。

### 当前论文口径

这条路线可以安全写成：

- LGCP area hierarchy 可以形成有效的 leader-level 感知聚合。
- 在普通十字路口 10-CAV 场景中，第一跳点云切片 + 第二跳 BEV feature fusion 可以达到较高 AP。
- 当前 second-hop feature payload 仍偏大，必须作为限制或后续优化方向说明。

暂时不宜写成：

- 中期特征在当前所有场景下都比原始点云更省通信。
- 当前 sparse BEV feature 已经是最终可部署压缩码流。
- 当前结果已经证明大规模 100 车场景下的完整感知性能。

## 二、第一跳 Where2comm 中期融合 + 第二跳 Where2comm 中期融合

### 路线定义

这条路线更接近“LGCP 原始想象中的两跳中期特征”：

1. member CAV 先生成中期 BEV feature。
2. member 按 LGCP area / Where2comm objectness 上传 selected feature 给 leader。
3. leader 使用 Where2comm 做第一跳中期特征融合。
4. leader 将 fused feature packet 上传 RSU。
5. RSU 再使用 Where2comm 做第二跳中期特征融合和检测。

这条路线理论上最优雅：两跳都传中期特征，并且两跳都可以利用 Where2comm 的 objectness-based BEV-cell selection 来降低通信量。

### 当前实现入口

主要相关代码：

- `C:\Workspace\OpenCDA\opencda\tools\lgcp_where2comm_two_hop_feature_fusion.py`
- `C:\Workspace\OpenCDA\opencda\tools\lgcp_where2comm_area_mask_eval.py`
- `C:\Workspace\OpenCDA\opencda\tools\lgcp_where2comm_leader_feature_fusion.py`

使用的 checkpoint：

- `C:\Workspace\OpenCOOD\checkpoints\where2comm_10e`

相关 Where2comm / OpenCOOD 代码：

- `C:\Workspace\OpenCDA\opencood\opencood\models\comm_modules\where2comm.py`
- `C:\Workspace\OpenCDA\opencood\opencood\models\point_pillar_comm_multiscale.py`
- `C:\Workspace\OpenCDA\opencood\opencood\models\fuse_modules\fusion_in_one.py`
- `C:\Workspace\OpenCDA\opencood\opencood\models\sub_modules\naive_compress.py`

### 当前最好结果

先说明：真正“两跳 Where2comm 中期融合”目前没有得到可用 AP。它的当前最好结果是负结果和诊断结果。

Top-5 / 21 帧、planned-area 两跳 Where2comm：

| 路线 | Areas/frame | Leader packets/frame | 第一跳 Mbps | 第二跳 Mbps | Total Mbps | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| member feature -> leader Where2comm -> RSU Where2comm | 5 | 4 | 7.617585 | 0.290621 | 7.908206 | 0.000000 | 0.000000 | 0.000000 | 210 | 0 |

full-BEV first-hop + area-mask feature communication 诊断：

| 路线 | 口径 | Areas/frame | Packets | 阈值 | 第一跳 Mbps | 第二跳 Mbps | Total Mbps | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-10 first frame leader-packet | planned areas | 10 | 5 | 0.001 | 11.069440 | 3.512320 | 14.581760 | 0.000000 | 0.000000 | 0.000000 | 12 | 9 |
| Top-5 21 frames leader-packet | planned areas | 5 | 4 | 0.001 | 7.112899 | 0.411550 | 7.524450 | 0.000000 | 0.000000 | 0.000000 | 210 | 21 |
| Top-10 first frame area-packet | full scope | 10 | 10 | 0.001 | 11.069440 | 3.051520 | 14.120960 | 0.000000 | 0.000000 | 0.000000 | 13 | 68 |

第一跳 leader-side AP 诊断：

| Stage | Pred samples | GT boxes | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一跳 leader Where2comm fusion 后 | 356 | 147 | 0.470496 | 0.470496 | 0.470496 |
| 第二跳 RSU Where2comm fusion 后 | 21 | 210 | 0.000000 | 0.000000 | 0.000000 |

这个诊断非常重要：第一跳融合后的 leader fused feature 本身不是完全坏的，它在 leader 侧可以检测；崩溃主要发生在第二跳 RSU 再次消费 fused feature 之后。

### 坐标对齐诊断

最近的坐标隔离实验进一步确认：坐标参考系是两跳崩溃的主要因素之一。

| Run | First-hop feature reference | Second-hop pairwise | Query | Areas | Leaders | Pred | GT | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-BEV leader packet | leader coordinate | normal | zero | 10 | 5 | 2 | 12 | 0.000000 | 0.000000 | 0.000000 |
| full-BEV leader packet | leader coordinate | inverse | zero | 10 | 5 | 1 | 12 | 0.000000 | 0.000000 | 0.000000 |
| full-BEV reference diagnostic | RSU/reference coordinate | normal | zero | 10 | 5 | 3 | 12 | 0.166667 | 0.166667 | 0.166667 |

解释：

- 单纯反转 pairwise transform 方向没有恢复 AP，说明不是一个简单的 affine direction bug。
- 将 first-hop packet feature 强制生成在 RSU/reference 坐标系后，同一帧 AP 从全 0 恢复到非 0。
- 因此 leader-local fused feature 不能直接作为 RSU second-hop Where2comm 的输入；它必须被显式生成或 warp 到共同 RSU/global BEV reference。
- 但即使坐标强制对齐，AP 仍很低，说明还有 fused-feature distribution shift 和 second-hop selector/head mismatch。

### Where2comm 如何降低通信量

Where2comm 的 `Communication.forward()` 会将 detection confidence map 经过 `sigmoid()` 后，对 anchor 维取最大值，得到 objectness map。推理时主要有三种选择方式：

- `k_ratio`：传输 Top `H * W * k_ratio` 个 BEV cells。
- `threshold`：传输 objectness 高于阈值的 BEV cells。
- 未配置 selector：传输全部 cells。

当前 `where2comm_10e` 使用 `threshold: 0.01` 和 Gaussian smoothing。LGCP 额外测试了 planned-area mask、`LGCP area ∩ objectness`、以及 dilation。

Where2comm 在 `PointPillarCommMultiscale` 中有三层 multiscale feature：

| scale | channels | height | width |
| --- | ---: | ---: | ---: |
| 0 | 64 | 96 | 352 |
| 1 | 128 | 48 | 176 |
| 2 | 256 | 24 | 88 |

单个 non-ego sender 的 dense full multiscale feature 大约是：

```text
(64 * 96 * 352 + 128 * 48 * 176 + 256 * 24 * 88) * 2 bytes = 7392 KB
```

5 个 leader / non-ego packets 在 sparse selection 前约 `36960 KB/frame`。因此 dense full BEV feature 不能作为通信优势 claim。

### 当前通信量难点

原始点云切片按 OpenCOOD 点格式估计：

```text
raw_bytes = N_points * 4 * 4 = N_points * 16
```

Where2comm selected feature 按被选中的 multiscale BEV cells 估计：

```text
feature_bytes = sum_s(selected_cells_s * channels_s * value_bits / 8)
```

当前按 float16 计算，即 `value_bits = 16`。

普通十字路口 10-CAV 场景中，Where2comm selected feature 相比 raw area point slice 的当前结果如下：

| setting | raw KB/frame | 近似 raw points/frame | feature KB/frame | feature/raw | selected/full dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| fine grid Top-10, K=5, area-objectness+dilation1 | 36.37 | 2328 | 184.12 | 5.06x | 0.50% |
| coarse 9-area all-area, K=5, area-objectness+dilation1 | 94.60 | 6055 | 382.02 | 4.04x | 1.03% |
| coarse 9-area all-area, K=5, area-objectness+dilation0 | 94.60 | 6055 | 359.56 | 3.80x | 0.97% |

这个表说明两件事：

- Where2comm 的 BEV-cell selection 是有效的，selected feature 只有 dense full feature 的约 `0.5%-1.03%`。
- 但 selected feature 仍比当前 raw area point slice 大 `3.80x-5.06x`，因为这个 easy intersection 场景的 area 点云较稀疏，而 BEV feature cell 一旦被选中就携带固定通道向量。

### 通信量 break-even 条件

若存在训练过的通道压缩比 `r`，则：

```text
feature_bytes(r) ~= feature_bytes_no_channel_compression / r
```

要让 selected BEV feature 不大于 raw points，需要：

```text
feature_bytes_no_channel_compression / r <= N_points * 16

N_points >= feature_bytes_no_channel_compression / (16 * r)

r >= feature_bytes_no_channel_compression / (16 * N_points)
```

当前三组设置的 break-even 点数：

| setting | feature KB/frame before channel compression | r=1 | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fine grid Top-10, K=5, dilation1 | 184.12 | 11784 | 5892 | 2946 | 1473 | 737 | 368 | 184 |
| coarse 9-area, K=5, dilation1 | 382.02 | 24449 | 12224 | 6112 | 3056 | 1528 | 764 | 382 |
| coarse 9-area, K=5, dilation0 | 359.56 | 23012 | 11506 | 5753 | 2877 | 1438 | 719 | 360 |

结合当前实际 raw points/frame：

| setting | 当前 raw points/frame | 需要的压缩比 r | 第一个实际可用候选 |
| --- | ---: | ---: | ---: |
| fine grid Top-10, K=5, dilation1 | 2328 | 5.06 | 8 |
| coarse 9-area, K=5, dilation1 | 6055 | 4.04 | 8 |
| coarse 9-area, K=5, dilation0 | 6055 | 3.80 | 4 |

结论是：在当前 selected-cell 数量下，Where2comm 大约需要 `4x-8x` 的训练过通道压缩、量化或等效编码，才有可能让 selected BEV feature 小于 raw area point slice。

### 当前主要问题

第一，模型语义不匹配。Where2comm 原 checkpoint 的训练语义是：多个普通 CAV feature 在 ego/reference 坐标系下做 intermediate fusion。两跳 LGCP 中，第二跳输入却是多个 leader fused feature packet。fused feature 已经不是普通 single-CAV feature，再喂给第二个 Where2comm 会产生分布偏移。

第二，坐标参考系不匹配。第一跳 leader fused feature 如果在 leader-local BEV 坐标系中，RSU 侧必须先统一到 RSU/global/reference 坐标。否则 BEV cell 的同一 index 在不同 packet 中代表不同世界位置。

第三，通信量优势还没有闭环。Where2comm sparse selection 相对 dense full feature 很省，但当前未压缩 selected feature 仍比 raw area point slice 大。

第四，通道压缩不能直接通过改 YAML 得到可靠结果。OpenCOOD 中确实存在 `compression: 4/8/16` 等配置，`NaiveCompressor` 也支持通道压缩；但这会改变模型结构和权重语义。对 `where2comm_10e` 简单改 `compression: 8` 不是可信的 perception result，需要匹配训练过的 checkpoint 或明确的 compressor adaptation。

### 当前难点

这条路线的难点高于第一条路线：

- 它既要解决第一跳 feature selection，又要解决第二跳 feature selection。
- 它必须保证 leader packet 在 RSU/global BEV reference 下几何一致。
- 它还需要让 second-hop Where2comm 能理解 first-hop fused feature，而这可能需要训练 hierarchy adapter、leader feature normalizer 或 RSU second-hop fusion/head。

因此它现在更适合作为 negative diagnostic 和 future-work 方向，而不是论文主性能路线。

### 当前论文口径

可以安全写成：

- Where2comm 可以和 LGCP area mask / objectness mask 结合，验证 selective BEV feature fusion 的机制方向。
- 当前两跳 Where2comm 级联已经端到端打通，但未训练情况下 AP 崩溃。
- 崩溃原因包括坐标参考系、fused-feature distribution shift、second-hop selector/head mismatch。
- 当前通信量分析表明，Where2comm selected feature 相比 dense full BEV 很省，但要小于 raw area point slice，还需要 `4x-8x` 训练过压缩或等效编码。

不宜写成：

- 两跳 Where2comm 已经是 LGCP 的主性能结果。
- 当前 checkpoint 可以直接支持 leader fused feature 到 RSU fused feature 的高 AP 级联。
- 当前 uncompressed selected feature 已经证明中期特征一定比 raw point slice 更省。

## 总体建议

短期主线应优先使用“第一跳早期融合 + 第二跳中期特征融合”作为 model-mechanism validation，因为它已经有较好的 AP；但必须诚实说明 second-hop feature payload 偏大。

“第一跳 Where2comm + 第二跳 Where2comm”保留为机制探索和负结果诊断。若要继续推进，需要先解决：

1. RSU/global reference 下的 leader feature generation 或可靠 feature warp。
2. second-hop fused-feature adapter / normalizer / head。
3. 训练过的通道压缩、量化或更严格 BEV-cell budget。

从论文风险看，当前最稳妥的表述是：LGCP 的 local-to-global hierarchy 能带来有效感知聚合；Where2comm 类方法证明了 area-constrained selective feature fusion 的可行方向；但当前未压缩、未校准的两跳中期特征级联还不能作为最终性能 claim。
