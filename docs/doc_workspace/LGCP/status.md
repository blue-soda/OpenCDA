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
- 已新增 LGCP 专用仿真入口 `opencda/scenario_testing/lgcp_carla.py` 和配置 `opencda/scenario_testing/config_yaml/lgcp_carla.yaml`。
- 未修改 `v2xp_cluster_carla` 原始配置。
- 已补齐 RSU 首次启用所需的基础运行路径：固定基础设施感知初始化、RSU 注册/访问、最近感知结果保存、销毁路径。

## 近期建议焦点

1. 扩大 area confidence validation 到多 seed / 多场景，形成可进入论文的稳定相关性结果。
2. 扩大 offline subset ablation 到多 seed，确认 `comm_aware_topk` 与 area-aware union 的相对关系是否稳定。
3. 推进完整 LGCP hierarchy 机制；当前 offline proxy 显示强 communication-aware baseline 已非常有竞争力，LGCP 后续主张需要靠 local fusion / RSU aggregation / scheduling 共同支撑。
4. 大规模 30 CAV 结果若短期只跑 latency，应在论文中收窄为 communication/computation scalability；若报告 perception scalability，只能报告已校准的 proxy，不能写成真实 AP。
5. 按 `ns3_phy_harq_request_trace.md` 推进 PHY / HARQ request-level trace；优先实现 PSSCH / HARQ 到 scheduled request 的绑定，再补 PSCCH overlap 的 request-level 回填。
6. 以 `revision_matrix.md` 作为论文修改和实验补强的主索引。

## 当前阻塞点

- 当前仓库已有 cluster-oriented cooperative perception / network co-simulation 管线，RSU 现在可以作为固定感知/注册实体启用；offline subset ablation、scalable quality proxy、hierarchy control-plane plan、11 帧 offline NS3 request-id bridge-observed replay、RLC request-id trace 和 PHY decode-failure breakdown 已能支持部分 rebuttal 证据，但尚未实现真实 feature slicing、leader local fusion、RSU global perception aggregation 和 LGCP 专用 NS3 scheduling。
- 尚未确认 OPV2V / V2XSet 数据集、本地模型 checkpoint 和可用 conda 环境的位置。
- NS3 当前版本已可接收 LGCP upload plan transfer requests，`cam_received` 和 RLC TX/RX/DROP 已可通过 `request_id` 精确映射回 upload request，OpenCDA 解析器也已准备好接收 request-level PHY / HARQ event；但 ns-3 侧尚未实际输出 PHY decode / HARQ 到 request id 的严格 trace。

## 场景配置判断

- `v2xp_cluster_carla` 满足 Town03 城市环岛、100 辆车、20 个 managed / intelligent vehicles、可选 NS3 协同仿真的基础条件。
- 但它不满足 LGCP 完整机制需求：原脚本只启用 `cluster`，没有启用 RSU；现有算法是车辆簇内成员上传给簇头，簇头融合并广播，不是 RSU 按 RoI area 分配可重叠 CAV group 并聚合全局视图。
- `lgcp_carla` 用作 LGCP 后续机制实现和实验的专用场景：保留 Town03 规模，新增 RSU，并显式开启车辆网格感知配置。
- `lgcp_carla` 已完成纯 CARLA 运行、数据导出和离线 OpenCOOD 推理 smoke test，适合推进 `target.md` 中的数据导出、离线验证、RSU/20 CAV 场景基础任务。
- 受 `BehaviorAgent.is_close_to_destination()` 的 10m 结束阈值限制，ego destination 不能设置到离 spawn 过近，否则会启动即结束。当前目的地被收近到刚超过阈值的位置以缩短在线仿真。
