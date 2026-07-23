# LGCP 任务清单

## 总目标

围绕 LGCP 论文审稿意见"C:\Workspace\icdcs-paper\LGCP\LGCP-review.txt"，补强三条主线：

1. 写作与叙事：清楚说明 LGCP 相对传统聚类、任务分配和层次聚合的区别。
2. 补充实验：证明 area confidence、group selection、local-to-global structure 和 transmission scheduling 的有效性。
3. 完善机制：让动态场景、控制面开销、定位误差、leader 失效和多 RSU 扩展等假设更可信。

## P0：必须优先完成

- [x] 梳理 LGCP 当前论文问题，形成 rebuttal / revision 对照表。
- [x] 设计 area confidence 验证实验：统计 confidence 与 area-level AP / recall 的相关性。
- [x] 设计 Eq. (2) 组合规则验证实验：比较 product rule、max、mean、sum、top-k 等组合方式。
- [x] 增加 greedy group selection 的 small-scale optimality gap 实验：用 exhaustive search 或 ILP 做小规模最优参考。
- [ ] 增加 local-to-global ablation：区分 partial sharing 与 LGCP 层次结构本身带来的收益。（已完成 Top-30 11 帧 box-level hierarchy late-fusion、flat-baseline alignment、Top-20/Top-23 near-common-budget runs、flat selected-agent raw-byte accounting 和 flat area-slice accounting；当前结论是 LGCP Top-30 用 `16.11%` selected-agent raw bytes 或 `40.38%` area-slice bytes，保留 comm-aware baseline `87.85%` AP@0.5 / `92.78%` AP@0.7，但 AP 仍未超过强 flat baselines；下一步是 neural feature hierarchy）
- [x] 补充更强通信感知 baseline：至少包含 adaptive sharing 或 selective sharing without LGCP hierarchy。
- [x] 明确大规模 30 CAV 实验只验证 latency，或补充 scalable perception-quality proxy。

## P0：当前未闭环目标

- [x] Common-byte-budget / raw-byte local-to-global ablation：Top-20/Top-23 LGCP 对齐 flat 10-agent fixed proxy，且新增 flat selected-agent raw PCD accounting 与 flat area-slice accounting；结果显示当前 box-level LGCP 不能超过强 flat baseline AP，但 Top-30 仅用约 `16.11%` selected-agent raw bytes / `40.38%` area-slice bytes 保留 comm-aware baseline `87.85%` AP@0.5 和 `92.78%` AP@0.7。
- [ ] Neural feature slicing / model-level hierarchy：已完成 PointPillar `intermediate_attentive` feature geometry probe、真实 feature crop / slice manifest smoke、Top-23 leader-local feature fusion、RSU feature assembly、detection-head probe、reference-frame alignment diagnostic、nearest/bilinear coordinate-warp AP probe、neural feature proxy summary，以及 reference-aligned point-slice -> leader scatter BEV -> RSU attentive BEV fusion 原型；当前 nearest AP@0.5 `0.010000`、AP@0.7 `0.000000`，bilinear AP@0.5 `0.011364`、AP@0.7 `0.003472`；RSU BEV attentive Top-23 11 帧 planned-area 在阈值 `0.05` 下 AP@0.3/0.5/0.7 为 `0.182482/0.136468/0.099602`，score threshold sweep 后 `0.01` 达到 `0.637777/0.463679/0.136646`；前 5 帧 train split 选择 `0.01`，后 6 帧 val split AP@0.5 为 `0.495974`；query-mode 对照显示 `mean` 高于 `first_leader` / `zero`，但 `mean` 仍只是无训练 RSU query workaround；已新增 `opencda/tools/lgcp_rsu_bev_training_sample_export.py`、`opencda/core/ml_libs/lgcp_rsu_bev_dataset.py`、`opencood/opencood/models/lgcp_rsu_bev_attentive.py` 和 `opencda/tools/lgcp_rsu_bev_train_smoke.py`，完成 sparse training sample、dataset、model wrapper、loss smoke、validation AP hook 与多阈值 AP sweep；Top-5 val6 threshold `0.05` 达到 AP@0.3/0.5/0.7 `0.820811/0.806983/0.245818`，但 AP@0.7 仍弱；已新增 `opencda/tools/lgcp_v2xvit_feature_probe.py` 试验 V2X-ViT backbone 后、fusion 前 compressed latent，Top-23 11 帧 compressed area crop bytes 为 `248912` total / `22628.36 bytes/frame`，低于 member area point slice `527840` total / `47985.45 bytes/frame`，而 scatter sparse 仍为 `5949952` total；已新增 `opencda/tools/lgcp_v2xvit_rsu_detection_probe.py`，打通 compressed latent RSU assembly -> decoder -> V2XTransformer -> detection heads；V2X-ViT crop/full threshold sweep 显示 `0.005-0.1` 预测数量和 AP 完全不变，score threshold 不是瓶颈；query-mode sweep 显示 `first` 明显优于 `mean/zero`，Top-5 首帧 `crop+first` AP@0.3/0.5/0.7 为 `0.583333/0.476190/0.113095`，Top-5 11 帧 `crop+first` 为 `0.500000/0.369657/0.081239` 且第二跳仅约 `4.88 KB/frame`，但 Top-10 首帧 `crop+first` 退化到 `0.090909/0.090909/0.022727`；新增 `leader-query-selection` 后，`max_area_points` / `max_group_size` 仍为 `0.090909/0.090909/0.022727`，说明简单启发式 query selection 不能修复扩展性问题；`pointpillar_cobevt_fusion` checkpoint 实际是 camera BEV segmentation 配置，不适合当前 LiDAR detection route 直接替换；已新增 `opencood/opencood/models/lgcp_v2xvit_rsu.py` 和 `opencda/tools/lgcp_v2xvit_rsu_train_smoke.py`，完成显式 RSU query/head 1-frame / 2-area / 1-step train smoke，loss `24.775737337`、query gradient norm `348.719512939`；已进一步打通 train-val AP hook，1 train frame / 1 val frame smoke 的 train/val loss 为 `24.375071751/22.165417312`，AP@0.3/0.5/0.7 为 `0.055556/0.055556/0.000000`，只作为评估链路证据；下一步应扩大到 Top-5 train5 / val6，并视 AP@0.7 决定是否解冻 V2XTransformer 后层。
  - 最新补充：V2X-ViT Top-5 train5 / val6 `query_heads` smoke 的 train/val loss 为 `13.608617907/13.522501738`，threshold `0.01/0.05/0.1` 下 AP@0.3/0.5/0.7 均为 `0.053819/0.030382/0.000000`，说明只训练 RSU query + heads 不足；下一步若继续 V2X-ViT compressed route，应尝试解冻 V2XTransformer 后层，否则收窄为 byte-boundary/limitation。
  - 最新补充：V2X-ViT Top-5 train5 / val6 `query_fusion_heads` smoke 的 train/val loss 为 `2.266083973/2.659839972`，但 threshold `0.01` 下 AP@0.3/0.5/0.7 仅为 `0.018649/0.003945/0.000000`，低于 `query_heads`；说明当前 V2X-ViT compressed crop route 存在 loss 下降但 AP 不改善的 mismatch，短期不应作为主性能 claim，应收窄为 byte-boundary/limitation。
  - 最新补充：按“点云区域裁剪通信 + 原 V2X-ViT intermediate fusion + RSU ego”新增 `opencda/tools/lgcp_v2xvit_area_point_crop_eval.py`；使用 `--reference-z-override 2.0` 解决高架 RSU `z=12m` 与车载 checkpoint z-range 不匹配问题。Top-5 11 帧 planned-area AP@0.3/0.5/0.7 为 `0.228011/0.208964/0.011765`，CAV area upload bytes `100766.55 bytes/frame`。这说明保持原模型 agent/ego 语义后 AP 不再崩，下一步应做 threshold sweep、Top-10/Top-23 扩展和同口径对照。
  - 最新补充：同样 point-crop native intermediate fusion 口径下，`intermediate_attentive` Top-5 11 帧 threshold `0.05` AP@0.3/0.5/0.7 为 `0.615115/0.420702/0.244483`，默认 threshold `0.20` 为 `0.282353/0.282353/0.203315`，通信量同为 CAV area upload `100766.55 bytes/frame`。说明 AP@0.3 `0.228011` 的 V2X-ViT 结果并非不合理，但 attentive checkpoint 明显更适合作为点云区域裁剪通信 + 原模型 intermediate fusion 的下一主线。
  - 最新补充：已新增 Where2comm checkpoint + LGCP external area-mask route：`opencda/tools/lgcp_where2comm_area_mask_eval.py` 复用 `C:\Workspace\OpenCOOD\checkpoints\where2comm_10e` 的 PointPillar backbone、affine alignment、attentive fusion 和 detection heads，用 LGCP planned-area BEV mask 替代或约束 Where2comm 内部 objectness BEV-cell selector；同时补齐 OpenCDA vendored `fusion_in_one.py` 的 `fusion: att`、`external_comm_mask`、`external_comm_recon` 支持。Top-5 11 帧 planned-area AP@0.3/0.5/0.7：LGCP area mask `0.614944/0.411297/0.116882`，第二跳 `25.559041 Mbps`；LGCP area + dilation1 `0.730799/0.608572/0.098913`，第二跳 `35.069208 Mbps`；Where2comm internal mask `0.801557/0.649291/0.163261`，第二跳 `39.112146 Mbps`；`LGCP area ∩ objectness` `0.715218/0.565188/0.202143`，第二跳 `12.964771 Mbps`；`LGCP area ∩ objectness + dilation1` `0.801557/0.658694/0.162600`，第二跳 `16.180131 Mbps`，leader-once lower-bound 二跳为 `4.045033 Mbps`。Top-10 / Top-23 扩展已完成：交集 selector 通信低于 internal mask，但 AP@0.5 分别降至 `0.270222` / `0.227966`，internal 同口径也只有 `0.264850` / `0.240054`，说明主因是 `max_cav=5` 与更大 eval scope 覆盖限制。进一步 per-leader box hierarchy diagnostic 已完成：Top-5 / Top-23 AP@0.5 仅 `0.074938` / `0.056061`，证明多 leader 检测框 late fusion 不能替代 Where2comm intermediate feature fusion。已新增真正 leader feature packet -> RSU feature-level fusion 工具 `opencda/tools/lgcp_where2comm_leader_feature_fusion.py`：Top-5 AP@0.5 `0.599561`，第一跳 member 点云 `18.26 KB/frame`、第二跳 feature `7.274124 Mbps`；Top-23 valid leader packets/frame `17.36`，但 AP@0.5 仅 `0.157367`。下一步应微调/训练 RSU feature aggregation，并补 metadata overhead 和分阶段通信核算。
  - 最新补充：按“4 个 Leader 接管更多 members 与 areas”的设想，新增 `opencda/tools/lgcp_reassign_limited_leaders.py`，并给 `opencda/tools/lgcp_where2comm_leader_feature_fusion.py` 增加 `--packet-granularity leader`。Top-23 selected areas 每帧被重分配给最多 4 个 Leader，RSU 侧接收 `ego + <=4 leader feature packets`；11 帧 planned-area AP@0.3/0.5/0.7 为 `0.650372/0.412919/0.021884`，相比原 Top-23 per-area packet AP@0.5 `0.157367` 明显恢复，但仍低于 Top-5 leader-packet AP@0.5 `0.599561`。通信方面，member-to-leader 第一跳升至 `227.50 KB/frame`，leader-to-RSU feature 为 `24.865513 Mbps`；threshold `0.01` AP@0.5 降至 `0.357752`。下一步应做 3/4/5 Leader 与 `load_weight/member_bonus` sweep，寻找 AP 与第一跳 upload 的平衡点。
  - 最新补充：5-Leader 同口径已补跑。Top-23 每帧最多 5 个 Leader 时，AP@0.3/0.5/0.7 为 `0.642838/0.409177/0.030923`，valid leader packets/frame `4.27`，member-to-leader 第一跳 `191.92 KB/frame`，leader-to-RSU feature `24.886924 Mbps`。与 4-Leader 相比，5-Leader 第一跳更低、AP@0.7 略高，但 AP@0.5 基本持平略低；当前更像通信折中点，而不是感知质量突破。
  - 最新补充：6-Leader 同口径已补跑。Top-23 每帧最多 6 个 Leader 时，AP@0.3/0.5/0.7 为 `0.658934/0.452508/0.026470`，valid leader packets/frame `5.09`，member-to-leader 第一跳 `154.50 KB/frame`，leader-to-RSU feature `24.707258 Mbps`。当前 6-Leader 是 4/5/6 中 AP@0.5 最好且第一跳更低的点，但部分帧为 `ego + 6 leader packets`，超过 checkpoint 常见训练语义；下一步 sweep 应纳入 3/4/5/6/7 Leader 与 assignment 参数。
  - 最新补充：Where2comm CAV-count limit probe 已完成。严格 checkpoint/YAML 口径为 total CAV `5`，即保守 Leader cap `4`；但 `fusion_in_one.py` 的 Where2comm fusion 通过 `record_len` 动态 regroup，synthetic full-size probe 在当前 CUDA 环境下验证到 total CAV `232` 可运行、`234` OOM，说明没有 5-CAV shape 硬上限。LGCP Top-23 4-13 Leader sweep 全部可运行，AP@0.5 最优是 7-Leader `0.484321`，第一跳 `132.88 KB/frame`；13-Leader 第一跳降到 `56.06 KB/frame` 但 AP@0.5 降到 `0.381057`。下一步不应使用 runtime OOM 上限作为实验上限，应围绕 7-Leader 做 assignment 参数、area budget 和 validation sweep，同时保留 4-Leader 作为 checkpoint-conservative 对照。
  - 最新补充：20 CAV full point cloud 上传到 RSU 的上界诊断已完成，确认当前场景为 `20 CAV + 1 RSU + 80 background vehicles`。per-CAV Where2comm 口径 AP@0.5 仅 `0.358292`；centralized raw 口径 AP@0.3/0.5/0.7 为 `0.651840/0.487223/0.080560`，raw upload `1603.89 KB/frame`。LGCP 7-Leader AP@0.5 `0.484321` 几乎追平 all-raw centralized upper bound，但第一跳只需 `132.88 KB/frame`。下一步应将 7-Leader 作为主候选，与 centralized raw upper bound 和 4-Leader conservative 对照共同组成模型级表。
  - 最新补充：已复测 SGCP attentive checkpoint 移植的 early-fusion detector。新增 `opencda/tools/lgcp_attentive_early_all_cav_to_rsu_eval.py`，同样执行 `20 CAV -> RSU centralized raw`，但 detector 从 Where2comm route 换为 `pointpillar_early_from_attentive_weights`。当前 LGCP 11 帧 full-scene threshold `0.20` AP@0.3/0.5/0.7 为 `0.816923/0.779641/0.470207`，明显高于 Where2comm centralized raw 的 `0.651840/0.487223/0.080560`。因此上一条“LGCP 7-Leader 几乎追平 all-raw centralized upper bound”需要修正为“LGCP 7-Leader 接近 Where2comm 同 route centralized raw 诊断”；真正 attentive early all-raw 上界仍明显更高。下一步模型级主表应拆分 detector-route matched diagnostic 与 detector-agnostic all-raw upper bound，避免过度 claim。
  - 最新补充：已按“leader BEV feature -> RSU attentive fusion”复用 SGCP attentive-derived checkpoint。新增 LGCP model dir `docs/doc_workspace/LGCP/experiments/model_dirs/pointpillar_intermediate_from_sgcp_attentive_early`，以 `point_pillar_intermediate` / `AttBEVBackbone` 加载同一份 SGCP `latest.pth`；新增 `--packet-granularity leader` 后，同一 leader 的多个 area 被合成一个 BEV feature packet。Top-5 11 帧 planned-area AP@0.3/0.5/0.7 为 `0.940917/0.828329/0.534530`；Top-23 原始 per-area packet 为 `0.134185/0.134185/0.102179`；Top-23 7-Leader leader-packet 为 `0.663529/0.556226/0.252941`，第一跳 member upload `88.42 KB/frame`，sparse BEV feature `339.25 KB/frame`。该路线目前比 Where2comm 7-Leader AP@0.7 更强，但 feature 通信量仍需压缩/metadata 细化与多 seed 验证。
  - 最新补充：已构造小规模更容易的 `lgcp_carla_small` 场景并导出离线数据集。新增 `opencda/scenario_testing/lgcp_carla_small.py` 与 `opencda/scenario_testing/config_yaml/lgcp_carla_small.yaml`，场景为 Town03 环岛、`8 CAV + 1 RSU + 28 background vehicles`、RoI `120m x 60m`、grid `10m x 6m`。离线数据集为 `D:\Data\Carla\2026_07_22_20_04_41`，共 21 帧。已生成 area confidence、Top-10 hierarchy plan，并完成 SGCP attentive leader-BEV route：AP@0.3/0.5/0.7 为 `0.800311/0.755478/0.393690`，member upload `12.67 KB/frame`，sparse BEV feature `234.69 KB/frame`。下一步可用该数据集做模型机制调参，再回到 20-CAV/100-car dense 场景验证泛化。
  - 最新补充：新增普通十字路口 easy 场景 `lgcp_carla_intersection_easy`，已移除矩形 `range` 采样并改用固定 spawn points，解决车辆偏离路口进入居民区的问题。已按用户要求更新为 `10 CAV + 1 RSU + 10 background vehicles`；新数据集 `D:\Data\Carla\2026_07_22_22_00_04` 共 21 帧。default early + RSU AP@0.3/0.5/0.7 = `0.86/0.86/0.77`，SGCP attentive-derived early + RSU = `0.86/0.86/0.78`。AP@0.3 尚未满足 `>= 0.90`，但相比 2-CAV 版本显著提升；下一步目标是继续微调固定点位或换更干净的普通路口，直到 early-fusion AP gate 达标。
  - 最新补充：已接受 ordinary-intersection 10-CAV 场景的 full early upper bound `0.86/0.86/0.78`，并用该场景推进 LGCP hierarchy。Top-10 plan 下 SGCP attentive leader-BEV -> RSU attentive fusion 的 planned-area AP@0.3/0.5/0.7 为 `0.868668/0.797311/0.733363`，full-scope 为 `0.813771/0.746923/0.687017`；第一跳 member 点云 `17.57 KB/frame`，第二跳 sparse BEV feature `211.08 KB/frame`。下一步从“能跑通”转为“压通信量”：做 feature compression、leader count、area budget sweep。
  - 最新补充：已按用户要求将 Leader 数限制为至多 5，并在 ordinary-intersection 10-CAV Top-10 area 上完成 K=3/4/5 sweep。K=3/4/5 planned-area AP@0.3/0.5/0.7 分别为 `0.809229/0.798743/0.660574`、`0.865886/0.798881/0.744764`、`0.865886/0.804755/0.752848`；full-scope 分别为 `0.710024/0.700737/0.578433`、`0.811164/0.748394/0.697697`、`0.811164/0.753897/0.705270`。K=5 在 `<=5` 约束下质量最好且第一跳 member upload 最低 `36.37 KB/frame`；K=4 可作为保守 checkpoint-friendly 对照；K=3 不建议作为主线。二跳 sparse BEV 仍约 `214-219 KB/frame`，下一步应转向 feature compression / feature-cell selection / metadata accounting，而不是继续放宽 Leader 数。
  - 最新补充：保持 Top-10 / K=5 assignment，已补测 Where2comm leader-feature route。`lgcp_area_objectness + dilation1` 下 planned-area AP@0.3/0.5/0.7 为 `0.759775/0.743407/0.381554`，full-scope 为 `0.643896/0.629983/0.323233`；第一跳 member upload `36.37 KB/frame`，二跳 feature `184.12 KB/frame` / `15.083032 Mbps`。Where2comm 二跳低于 SGCP attentive-derived K=5 的 `214.58 KB/frame`，但 AP@0.7 明显弱，当前应作为 communication-aware checkpoint 对照，而不是主性能路线。
  - 最新补充：已按用户要求大幅增大 ordinary-intersection 的 LGCP area 面积。`lgcp_carla_intersection_easy.yaml` 中 `lgcp.roi.grid_size` 已从 `[10.0, 6.0]` 改为 `[30.0, 24.0]`，使 `90m x 70m` RoI 变为 `9` 个总 area。新 area confidence 为 `2079 = 21 frames x 9 areas x 11 agents`；all-area plan 为 `9` areas/frame，K=5 leaders 为 `8;1;2;10;6`，area load `8:2;1:2;2:2;10:2;6:1`。粗 area 下第一跳 member upload 升至 `94.60 KB/frame`，但二跳仍更大：SGCP attentive-derived sparse BEV `916.46 KB/frame`，Where2comm `area_objectness+dilation1` 为 `382.02 KB/frame`，Where2comm `dilation0` planned 为 `359.56 KB/frame`。说明 area 总数问题已解决，但 feature packet 仍需 compression / stricter cell selection / quantization 才能满足“中间特征小于原始点云”的通信叙事。
- [ ] Area confidence 多 seed / 多场景验证：在 `mindspore-186` 上跑 400-frame gate 多 seed，形成论文级相关性统计。
- [ ] Greedy / O3 optimality gap 多场景扩展：覆盖更大 instance 和 latency-aware objective。
- [ ] 论文源文件落地：将 contribution、workflow、baseline fairness、limitations、stage numbering、Fig. 7 x-axis 等修改写入 `conference_101719.tex` 和图源。

## P1：重要补强

- [x] 增加定位误差敏感性实验。（已完成单场景 11 帧 smoke，论文级结果仍需多 seed）
- [x] 增加车辆速度 / update frequency / stale assignment 敏感性实验。（已完成 update frequency / stale assignment 单场景 11 帧 smoke，显式车速仍需多场景）
- [x] 增加 area size / grid size 敏感性实验。（已完成单场景 11 帧 smoke，论文级结果仍需多 seed）
- [x] 增加 subchannel count `Z` 敏感性实验。（已完成 11 帧 scheduling-capacity proxy，NS3 多 Z 仍需复核）
- [x] 增加 CAV / edge computation capacity 敏感性实验。（已完成 11 帧 compute-latency proxy，模型级 runtime 仍需后续实现）
- [x] 显式统计 control-plane overhead：location、direction、confidence、assignment、global view。
- [x] 解释或修复低车数场景下 LGCP 与 baseline latency 接近的问题。（已形成 `latency_figure_audit.md`，解释 fixed coordination overhead / sparse contention / edge compute 小规模优势）
- [x] 检查并修正 Fig. 7 y-axis 标注问题。（已渲染核查：y-axis `End-to-end latency (ms)` 正确；实际应将 x-axis 从 `Number of vehicles` 改为 `Number of CAVs`，需用原始绘图源重导出）

## P2：机制与写作完善

- [x] 增加 LGCP workflow figure，覆盖 partition、group assignment、leader selection、local fusion、RSU aggregation、broadcast。
- [x] 重新阐述 group 概念，避免被理解为传统 clustering。
- [x] 明确一个 CAV 参与多个 area group 时的数据包粒度、去重和复用机制。
- [x] 完善 leader 到 RSU 上传策略，说明重要结果的优先级、可靠性和失败处理。
- [x] 增加 deployment assumptions / limitations 小节：RSU centralization、mobility、multi-RSU、stale information、failure modes。
- [x] 修正文中 stage 编号不一致问题。
- [x] 将 “heuristic / approximate” 的表述调整为与实际理论保证一致的写法。

## 待落地到仓库的实验方向

- [x] 确认 OpenCOOD 中 OPV2V / V2XSet 的模型评估入口和日志格式。（已形成 `opencood_eval_entry.md`，记录 `inference.py`、`eval.yaml`、`--save_npy` 和 LGCP area-level 复用边界）
- [x] 确认 OpenCDA 中多 CAV + NS3 的基础可复现实验命令。
- [x] 新增 LGCP 专用 Town03 仿真配置，避免修改 `v2xp_cluster_carla`。
- [x] 补齐 RSU 首次启用的基础初始化、注册、读取和销毁路径。
- [x] 完成 `lgcp_carla` 数据导出 smoke test。
- [x] 完成导出数据的离线 OpenCOOD 推理 smoke test。
- [x] 确认 LGCP 所需 area-level confidence / AP / recall 的数据导出位置。
- [x] 建立统一结果目录和命名规则。
- [x] 实现 area confidence 离线导出 smoke test，验证 ROI/grid/agent-area records 链路。
- [x] 扩展 area confidence 离线评估脚本，接入 prediction slicing 与 area-level recall smoke test。
- [x] 扩展 area confidence 离线评估脚本，累计 area-level AP 并统计 confidence-vs-AP/recall 相关性 smoke test。
- [x] 扩大 area confidence validation 到完整 `lgcp_carla` dump 帧，并补充 detector-score confidence 对照。
- [ ] 扩大 area confidence validation 到多 seed / 多场景，形成可进入论文的稳定相关性结果。（已完成 OpenCOOD 远端资源 / 协议 / checkpoint 审计；本地无 `dataset/`，下一步在 `mindspore-186` 跑 400-frame gate 多 seed）
- [x] 实现 greedy group-member selection exhaustive-search gap smoke test。
- [x] 扩展 greedy optimality gap 到 leader assignment / load balancing。
- [ ] 扩大 greedy optimality gap 到多 seed / 更大 instance，并接入 latency-aware O3 objective。（O3 objective 已接入并完成 5-agent / 6-agent 11 帧 smoke；已完成 5-agent sampled seeds 7/11/23/37，仍需多场景）
- [x] 完成 local-to-global ablation 实验设计。
- [x] 实现 offline selective-sharing vs LGCP area-aware subset ablation。
- [x] 扩大 offline subset ablation 到完整 11 帧。
- [x] 为 offline subset ablation 加入 packet-budget / byte-volume proxy 统计。
- [x] 实现 communication-aware top-k selective-sharing baseline，并完成 11 帧 offline 对照。
- [x] 实现 scalable perception-quality proxy，并用 11 帧 offline AP 对照做初步校准。
- [x] 实现 LGCP 离线 RSU area assignment / leader upload plan 导出。
- [x] 将 LGCP upload plan 接入 offline NS3 replay dry-run。
- [x] 运行 LGCP upload plan offline NS3 联机 smoke test。
- [x] 扩大 offline subset ablation 到多 seed。（已完成 random-only seeds 7/11/23/37 汇总；deterministic strong baselines 复用 11 帧结果）
- [x] 扩大 LGCP upload plan offline NS3 replay 到 11 帧，并解析 request-level bridge-observed delivery ratio / delay summary。
- [x] 接入 ns-3 PHY decode trace，补充 decode-failure breakdown。
- [x] 接入 ns-3 application request-id trace，将 `cam_received` 精确映射回 LGCP upload request。
- [x] 接入 ns-3 RLC trace，将 RLC events 进一步映射回 LGCP upload request。
- [x] 完成 NS3 PHY / HARQ request-level trace 设计，明确字段、落点、解析器输出和论文边界。
- [x] 扩展 OpenCDA NS3 log parser，支持未来 request-level PHY / HARQ events 并输出 request lifecycle funnel。
- [x] 接入 ns-3 PSSCH request-level trace，将 PSSCH decode OK/FAIL 映射回 LGCP upload request。
- [x] 将 HARQ feedback 进一步绑定到 LGCP upload request，并确认 replay 配置下 HARQ event 可观测。
- [x] 扩大 request-level PSSCH / HARQ trace 到 11 帧 LGCP replay。
- [x] 接入 LGCP multi-slot replay lifecycle diagnostics，定位 scheduled replay 中 member-to-leader callback 偏低的阶段原因。
- [x] 增加 LGCP source-unique / half-duplex scheduling sensitivity，验证同 source 同 slot 多发不是 member-to-leader 瓶颈的充分解释。
- [ ] 实现 LGCP 专用 RSU area assignment、leader local fusion 和 RSU global aggregation 管线。（已完成 offline assignment / upload plan、hierarchy area-budget sweep、raw LiDAR feature-slice budget sweep、raw-slice-aware upload plan dry-run / 3 帧与 11 帧 request-level NS3 trace、single-slot scheduled NS3 smoke、multi-slot scheduling proxy、3 帧 live multi-slot replay、leader/RSU aggregation proxy、model-level hierarchy 入口审计、box-level late-fusion adapter smoke、Top-30 1 帧完整 area run、Top-30 3 帧 run、Top-30 11 帧 run、baseline 对齐表、common-byte/raw-byte 对照、PointPillar feature geometry probe、真实 feature crop export smoke、leader-local feature fusion smoke、RSU feature assembly smoke、detection-head probe、reference-frame alignment diagnostic、nearest/bilinear coordinate-warp smoke、AP probe、neural feature proxy summary、reference-aligned point-slice RSU BEV attentive fusion 11 帧 planned-area run、sparse training sample export、dataset helper、model wrapper、1-frame train smoke、Top-5 train5/val6 train smoke、Where2comm checkpoint + LGCP external area-mask route、`area ∩ objectness` selector、Top-10/Top-23 scale-boundary smoke、leader-once second-hop accounting、Where2comm per-leader box hierarchy diagnostic、Where2comm leader feature packet RSU fusion；有效 Top-5 intermediate/leader-packet AP 已出现，但 Top-23 仍需 RSU feature aggregation 微调、metadata overhead、多场景验证）
