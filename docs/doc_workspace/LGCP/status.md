# LGCP 当前状态

## 已确认背景

- OpenCDA 是 CARLA / SUMO 联合仿真的 CDA 研究框架，核心模块位于 `opencda/`，包含感知、定位、规划、控制、V2X 通信、RSU / CAV 管理和场景测试。
- 仓库内包含 OpenCOOD，用于协同感知模型训练和推理，支持 OPV2V、V2XSet 以及 CoBEVT、Where2comm、CoAlign 等模型。
- `AGENT_README.md` 记录了当前仓库中 CARLA + NS3 + OpenCDA 联合调试的关键命令、日志路径和已知问题。
- LGCP 论文材料位于 `C:\Workspace\icdcs-paper\LGCP`，审稿意见位于 `C:\Workspace\icdcs-paper\LGCP-review.txt`。

## 论文当前主要风险

1. Area confidence 是优化核心变量，但目前缺少与真实 area-level detection performance 的相关性验证。
2. Group selection 和 transmission scheduling 被审稿人认为是启发式算法，缺少理论保证或 optimality gap 评估。
3. Baseline fairness 不足：LGCP 做 selective area-level sharing，而主要 baseline 多为 complete sharing。
4. 大规模 co-simulation 只报告 latency，尚未证明 perception quality 在 dense CAV 场景下也能保持。
5. 动态场景假设不足：localization error、vehicle mobility、update frequency、stale assignment、leader failure 和 multi-RSU scaling 尚未充分讨论或验证。

## 当前工作区状态

- `docs/doc_workspace/LGCP` 已建立为 LGCP 文档工作区。
- 该目录当前用于记录论文修订、实验设计、实验日志和核心结果。
- 已新增 `revision_matrix.md`，将审稿意见映射为 P0/P1/P2 revision 任务和 rebuttal 要点。
- 已新增 `area_confidence_validation.md`，明确 area confidence validation 的数据单元、area-level AP / recall 计算方式、相关性指标和输出目录。
- 已新增 `eq2_composition_validation.md`，明确 Eq. (2) noisy-or 组合规则与 max/mean/sum/top-k/calibrated noisy-or 的验证方式。
- 已新增 `greedy_optimality_gap.md`，明确 greedy group selection 与 exhaustive / ILP oracle 的 small-scale optimality gap 评价方式。
- 已新增 `experiments/readme.md`，建立 LGCP 实验输出目录、run id、配置快照和结果归档规则。
- 已新增 `opencda/tools/lgcp_area_confidence_eval.py`，并完成 3 帧 `lgcp_carla` area confidence smoke export。
- `opencda/tools/lgcp_area_confidence_eval.py` 已支持 `--with-inference`，可导出 per-area prediction quality records 和 recall / precision smoke results。
- `opencda/tools/lgcp_area_confidence_eval.py` 已支持累计 area-level AP 与 confidence-quality correlation smoke summary。
- 已完成完整 `lgcp_carla` dump 的 11 帧 area confidence validation，并加入 detector-score confidence 对照。
- 已新增 `opencda/tools/lgcp_greedy_gap_eval.py`，完成 greedy group-member selection 与 exhaustive subset oracle 的 small-scale gap smoke。
- `opencda/tools/lgcp_greedy_gap_eval.py` 已扩展 leader assignment / load balancing gap，覆盖论文 selection algorithm 的第二阶段。
- 已新增 `local_to_global_ablation.md`，明确 local-to-global hierarchy 与 selective sharing 的消融变体、指标和实现阶段。
- 已新增 `opencda/tools/lgcp_subset_ablation_eval.py`，完成 offline perception-only subset ablation smoke，并已扩展到完整 11 帧和 packet-budget / byte-volume proxy 统计，比较 full sharing、random selective sharing、confidence top-k 和 area-aware union。
- `opencda/tools/lgcp_subset_ablation_eval.py` 已加入 `comm_aware_topk`，作为不使用 LGCP hierarchy 的 communication-aware selective-sharing baseline。
- 已新增 `communication_aware_baseline.md`，记录强 baseline 定义、当前结果和论文解释边界。
- 已新增 `opencda/tools/lgcp_quality_proxy_eval.py` 和 `large_scale_quality_proxy.md`，定义 scalable perception-quality proxy，并用 11 帧 offline AP 对照做初步校准。
- 已新增 `opencda/tools/lgcp_hierarchy_plan_eval.py` 和 `hierarchy_pipeline.md`，可离线导出 RSU area assignment、area-task group、leader selection、member-to-leader upload 和 leader-to-RSU upload plan。
- `opencda/tools/offline_ns3_replay.py` 已支持 `--lgcp-upload-plan` 和 `--dry-run`，可将 LGCP hierarchy upload plan 转换为 NS3 `transfer_requests`，并在无 NS3 时先验证请求数和字节数。
- 已完成 LGCP upload plan 的 3 帧 offline NS3 联机 smoke test，验证 20 CAV + RSU 正整数节点映射、同步和部分 CAM 接收日志。
- 已新增 `opencda/tools/lgcp_ns3_log_eval.py`，并完成 LGCP upload plan 的 11 帧 offline NS3 replay；当前可输出 request-id 精确匹配的 bridge-observed delivery ratio / delay summary，以及 NS3 PHY decode-failure breakdown。
- ns-3 co-simulation 侧已扩展 CAM header，将 OpenCDA `pkt_id` 作为 `request_id` 透传到 `cam_received`，从而把 application callback 精确映射回 `upload_plan.csv`。
- ns-3 co-simulation 侧已新增 RLC request-id tag，并完成 11 帧 RLC TX/RX/DROP trace 到 LGCP `upload_plan.csv` 的 request-level 映射。
- 已新增 `ns3_phy_harq_request_trace.md`，明确下一步 PHY / HARQ request-level trace 的字段、ns-3 落点、OpenCDA 解析器输出和论文使用边界。
- `opencda/tools/lgcp_ns3_log_eval.py` 已支持解析未来 `[NRSL_PHY_EVENT]` / `[NRSL_HARQ_EVENT]` request-level 日志，并生成 `request_lifecycle.csv` / `request_lifecycle_summary.csv`。
- ns-3 侧已新增 PSSCH request-level `[NRSL_PHY_EVENT]`，3 帧 smoke 中解析到 124 条 request-level PHY event，并全部映射回 LGCP `upload_plan.csv`。
- ns-3 侧已新增 `--enableSlHarq` 和 `--psfchPeriod` 参数；当使用 `--enableSlHarq=true --psfchPeriod=4` 时，3 帧 smoke 中已观测到 request-level HARQ ACK/NACK，并全部映射回 LGCP `upload_plan.csv`。
- 已完成 11 帧 LGCP replay 的 request-level PSSCH / HARQ trace：1177 条 PHY/HARQ request events 全部映射回 `upload_plan.csv`。
- 已新增 `opencda/tools/lgcp_control_overhead_eval.py` 和 `control_plane_overhead.md`，完成 11 帧 LGCP hierarchy control-plane overhead 统计；当前平均 control-plane bytes 为 30.96 KB/frame，占 planned data + control 的 9.52%。
- 已新增 `workflow_and_group_semantics.md`，形成 LGCP workflow figure 草稿、area-task group 语义、packet granularity、dedup / reuse 和 leader-to-RSU reliability 的论文写作素材。
- 已新增 `deployment_assumptions.md`，形成 RSU-assisted assumption、mobility / stale assignment、localization error、failure modes、multi-RSU scaling 和 large-scale claim boundary 的论文讨论素材。
- 已新增 `manuscript_language_audit.md`，定位 TeX 中 stage 编号不一致和 `approximate solution` 高风险表述，并给出替换段落、rebuttal wording 和论文编辑 checklist。
- 已新增 `latency_figure_audit.md`，完成 Fig. 7 轴标核查和低 CAV 数 latency 接近原因解释；当前 Fig. 7 y-axis 语义正确，建议将 x-axis 从 `Number of vehicles` 改为 `Number of CAVs`。
- 已新增 `opencood_eval_entry.md`，确认 OpenCOOD OPV2V / V2XSet 评估主入口为 `opencood/tools/inference.py`，整体 AP 输出到 checkpoint `eval.yaml`，`--save_npy` 输出 prediction / GT 文件但 GT 后缀为 `.npy_test`。
- 已新增 `opencood_multiscene_area_confidence.md`，完成 OpenCOOD 多场景 area confidence validation 的资源审计：本地 `C:\Workspace\OpenCOOD` 无 `dataset/`，论文级多场景应在 `mindspore-186:/data1/wql/gzc/workspace/OpenCOOD` 跑；模板 OPV2V split 与 `OPV2V(Culver) regular-history` 入口需要分开记录。
- 已新增 `grid_size_sensitivity.md`，并在不修改 `lgcp_carla.yaml` 的前提下完成 `5m x 3m`、`10m x 6m`、`20m x 12m` 三组 11 帧 grid-size sensitivity smoke；当前 default `10m x 6m` 在 area-frame noisy-or vs recall@0.5 上 Spearman 最高。
- 已新增 `localization_error_sensitivity.md`，完成 `0.0m / 0.2m / 0.5m / 1.0m` CAV xy pose noise 的 11 帧 localization sensitivity smoke；area-frame noisy-or vs recall@0.5 Spearman 在 1.0m 噪声下仍约 `0.55`。
- 已新增 `opencda/tools/lgcp_stale_assignment_eval.py` 和 `stale_assignment_sensitivity.md`，完成 `0/1/2/3` 帧 stale assignment smoke；lag 1/2 帧 ranking 仍较稳定，lag 3 帧 Spearman 降到 `0.447925`。
- 已新增 `opencda/tools/lgcp_subchannel_sensitivity_eval.py` 和 `subchannel_sensitivity.md`，完成 `Z=5/10/15/20` 的 11 帧 scheduling-capacity proxy；mean slots/frame 从 `12.73` 降到 `3.73`。
- 已新增 `opencda/tools/lgcp_compute_capacity_eval.py` 和 `compute_capacity_sensitivity.md`，完成 CAV leader local-fusion / RSU aggregation compute capacity proxy；代表性均衡容量下 compute mean 从 `8.41ms` 降到 `1.05ms`。
- 已完成 offline subset ablation random-only multiseed 扩展，汇总 seeds `7/11/23/37`；random AP@0.7 均值低于 confidence / area-aware / communication-aware selective baselines，但 `comm_aware_topk` 仍略高于当前 `area_aware_union`。
- `opencda/tools/lgcp_greedy_gap_eval.py` 已支持 `--sample-seeds` 和 sampled candidate pool；已完成 5-agent O3 sampled multiseed smoke，seeds `7/11/23/37` 共 44 个 instance，O3 mean relative gap 为 `0.043650` 到 `0.060727`。
- 已新增 `opencda/tools/lgcp_hierarchy_aggregation_eval.py`，将 hierarchy assignment plan 转换为 leader local result proxy 和 RSU global aggregation proxy；Top-40 11 帧 selected GT ratio 为 `1.0`，mean selected area recall@0.5 为 `0.670455`。
- 已完成 hierarchy area-budget sweep，`max_areas=10/20/30/40`；Top-30 在约 `222.1 KB/frame` 下达到 `0.953193` selected GT ratio，Top-40 达到 `1.0` 但 mean selected area recall 降到 `0.670455`。
- 已完成 feature-slice budget sweep，将 raw LiDAR area-slice manifest 扩展到 `max_areas=10/20/30/40`；Top-30 的 raw member upload bytes 为 `59.42 KB/frame`，低于固定 local upload proxy `158.18 KB/frame`。
- 已新增 `opencda/tools/lgcp_slice_upload_plan_eval.py`，可将 hierarchy fixed-byte upload plan 替换为 raw-slice-aware upload plan；Top-30 11 帧 member-to-leader bytes 从 `1.74 MB` 降到 `653.57 KB`，并通过 `offline_ns3_replay --dry-run`。
- Top-30 raw-slice-aware upload plan 已完成 3 帧 live ns-3 replay smoke，输出 `upload_plan_replayed.csv`；本次只验证 bridge/replay 接受，不报告 request-level delivery ratio。
- Top-30 raw-slice-aware upload plan 已进一步完成 3 帧 request-level trace rerun：137 planned requests、6 application callbacks、106 RLC TX、20 RLC RX、14 requests with PSSCH OK、51 requests with PSSCH FAIL；当前 unscheduled replay 仍显示严重链路瓶颈。
- Top-30 raw-slice-aware upload plan 已扩展到完整 11 帧 request-level trace：504 planned requests、55 application callbacks、546 RLC TX、118 RLC RX、94 requests with PSSCH OK、250 requests with PSSCH FAIL；当前结果验证 trace 路径和调度必要性，不作为最终网络性能行。
- 已新增 `opencda/tools/lgcp_schedule_upload_plan_eval.py`，可将 raw-slice-aware upload plan 转换为单 slot、capacity-gated scheduled smoke plan；Top-30 11 帧在 `Z=10` 下保留 110/504 条 request 和 543.41KB/1.31MB bytes。
- `offline_ns3_replay.py` 现已保留 LGCP upload plan 中的 `sc_start/sc_num` 字段；3 帧 scheduled live ns-3 smoke 达到 30 planned requests、24 application callbacks、0 PHY decode failures，bridge-observed delivery ratio 为 `0.8`。
- `opencda/tools/lgcp_schedule_upload_plan_eval.py` 已扩展 `--schedule-mode multi_slot`，可对完整 raw-slice-aware plan 输出 `slot_index/sc_start/sc_num/stage/scheduled_delay_ms`；Top-30 11 帧在 `Z=10`、`10ms/slot` 下 504/504 requests 全部排入 5 slots/frame，调度延迟 proxy 为 `50ms/frame`。
- `offline_ns3_replay.py` 已新增 `--respect-slot-index`，可按 LGCP `slot_index` 分 slot 发送并同步 ns-3；3 帧 multi-slot live replay 覆盖 137/137 requests，54 application callbacks，110 requests with RLC RX / PSSCH OK，PSSCH FAIL 为 0。
- 已新增 `opencda/tools/lgcp_lifecycle_diagnostics.py`，对 multi-slot replay lifecycle 按 stage / slot / target / terminal state 归因；长 drain 复现实验确认 member-to-leader 低 callback 不是 drain 不足，而是 `47 planned / 40 RLC TX / 28 RLC RX / 2 application received` 的链路与应用层混合瓶颈。
- lifecycle diagnostics 已增加 planned-byte size-bin 输出；member-to-leader 大包并非主要瓶颈，`8000-16000` bytes bin 为 `6/6` RLC RX / PSSCH OK，较弱的是 `1000-4000` bytes bins 与 member slot 1。
- `lgcp_schedule_upload_plan_eval.py` 已增加 `--enforce-source-unique` 半双工敏感性；Top-30 11 帧 mean slots/frame 从 `5.0` 增至 `7.36`，3 帧 live replay 中 member-to-leader application callbacks 从 `2/47` 增到 `5/47`，但总 delivery 为 `52/137`，说明 source-unique 有帮助但不是充分修复。
- 已新增 `opencda/tools/lgcp_feature_slice_manifest.py`，生成 raw LiDAR area-specific slice manifest；Top-40 11 帧 member upload slice 约 `6199` points/frame、`99.19 KB/frame`，为后续 neural feature slicing 提供接口和 byte proxy。
- 已新增 `model_level_hierarchy_entry.md`，审计 OpenCOOD late/intermediate fusion、OpenCDA offline dataset 和 SGCP late-fusion 参考路径；结论是下一步应先实现 box-level hierarchy late-fusion adapter，再推进 PointPillar intermediate neural feature slicing。
- 已新增 `opencda/tools/lgcp_hierarchy_late_fusion_eval.py`，实现 box-level hierarchy late-fusion adapter；1 帧 2 area smoke 已完成，能够真实调用 OpenCOOD late model，输出 leader local prediction、RSU global late-fusion summary 和 AP。
- `lgcp_hierarchy_late_fusion_eval.py` 已扩大到 Top-30 首帧完整 area：30 assignment rows、23 次唯一 group inference、RSU fused pred / GT boxes 为 `35 / 35`，AP@0.5 为 `0.606851`；仍需扩大到 3 帧 / 11 帧后才能作为论文级对照。
- `lgcp_hierarchy_late_fusion_eval.py` 已完成 Top-30 3 帧连续运行：90 assignment rows、68 次唯一 group inference、mean RSU fused pred / GT boxes 均为 `35.666667`，AP@0.5 为 `0.584564`；下一步扩大到 11 帧并与 flat selective-sharing baseline 对齐。
- `lgcp_hierarchy_late_fusion_eval.py` 已完成 Top-30 11 帧运行：330 assignment rows、245 次唯一 group inference、mean RSU fused pred / GT boxes 为 `34.909091 / 37.090909`，AP@0.5 为 `0.602748`；下一步应汇总为 local-to-global ablation 表并对齐既有 flat selective-sharing baseline。
- 已新增 LGCP 专用仿真入口 `opencda/scenario_testing/lgcp_carla.py` 和配置 `opencda/scenario_testing/config_yaml/lgcp_carla.yaml`。
- 未修改 `v2xp_cluster_carla` 原始配置。
- 已补齐 RSU 首次启用所需的基础运行路径：固定基础设施感知初始化、RSU 注册/访问、最近感知结果保存、销毁路径。

## 近期建议焦点

1. 扩大 area confidence validation 到多 seed / 多场景，形成可进入论文的稳定相关性结果；OpenCOOD 评估入口、日志格式、远端数据路径和候选 checkpoint 已确认，下一步应在 `mindspore-186` 跑 400-frame gate。
2. 将 Top-30 11 帧 box-level hierarchy late-fusion 与既有 11 帧 full / confidence_topk / comm_aware_topk / area_aware_union baseline 对齐成 local-to-global ablation 表；随后再推进 PointPillar intermediate neural feature tensor slicing、leader local fusion 和 RSU global perception aggregation。
3. 若继续扩展 ablation，应将 source-unique 作为更真实 scheduler 约束保留，同时继续诊断 member slots 的 target receiver setup 和非 RSU receiver 的 CAM application completion；另一条主线是推进 model-level leader/RSU fusion。
4. 大规模 30 CAV 结果若短期只跑 latency，应在论文中收窄为 communication/computation scalability；若报告 perception scalability，只能报告已校准的 proxy，不能写成真实 AP。
5. 下一步将 request-level PHY/RLC/HARQ trace 和 control-plane overhead 扩展到多 seed，或开始推进完整 LGCP local fusion / RSU aggregation。
6. 以 `revision_matrix.md` 作为论文修改和实验补强的主索引；下一轮若进入论文源文件修改，应优先重导出 Fig. 7 x-axis 并加入低密度 latency 解释段。

## 当前阻塞点

- 当前仓库已有 cluster-oriented cooperative perception / network co-simulation 管线，RSU 现在可以作为固定感知/注册实体启用；offline subset ablation、scalable quality proxy、hierarchy control-plane plan、11 帧 offline NS3 request-id bridge-observed replay、RLC request-id trace、PHY decode-failure breakdown、single-slot scheduled smoke、multi-slot scheduling proxy 和 3 帧 live multi-slot replay 已能支持部分 rebuttal 证据，但尚未实现真实 feature slicing、leader local fusion 和 RSU global perception aggregation。
- 本地 `C:\Workspace\OpenCOOD` 尚无 `dataset/`，不能直接做 OPV2V / V2XSet 多场景 inference；远端 `mindspore-186` 的 OPV2V 数据路径、Python 环境和 checkpoint/log store 已确认。
- NS3 当前版本已可接收 LGCP upload plan transfer requests，`cam_received`、RLC TX/RX/DROP、PSSCH decode OK/FAIL 和 HARQ ACK/NACK 已可通过 `request_id` 精确映射回 upload request；HARQ trace 需要显式使用 `--enableSlHarq=true --psfchPeriod=4`。

## 场景配置判断

- `v2xp_cluster_carla` 满足 Town03 城市环岛、100 辆车、20 个 managed / intelligent vehicles、可选 NS3 协同仿真的基础条件。
- 但它不满足 LGCP 完整机制需求：原脚本只启用 `cluster`，没有启用 RSU；现有算法是车辆簇内成员上传给簇头，簇头融合并广播，不是 RSU 按 RoI area 分配可重叠 CAV group 并聚合全局视图。
- `lgcp_carla` 用作 LGCP 后续机制实现和实验的专用场景：保留 Town03 规模，新增 RSU，并显式开启车辆网格感知配置。
- `lgcp_carla` 已完成纯 CARLA 运行、数据导出和离线 OpenCOOD 推理 smoke test，适合推进 `target.md` 中的数据导出、离线验证、RSU/20 CAV 场景基础任务。
- 受 `BehaviorAgent.is_close_to_destination()` 的 10m 结束阈值限制，ego destination 不能设置到离 spawn 过近，否则会启动即结束。当前目的地被收近到刚超过阈值的位置以缩短在线仿真。
