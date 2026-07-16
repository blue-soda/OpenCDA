# SGCP 当前状态

更新时间：2026-07-16

## 运行环境

- 项目可用环境：`conda activate opencda`
- 脚本化执行建议：`conda run -n opencda python ...`

## 仓库理解

OpenCDA 是基于 CARLA/SUMO 的协同驾驶仿真框架，提供感知、定位、规划、控制、V2X 通信等完整模块。当前仓库还包含 OpenCOOD，用于多智能体协同感知模型训练和推理。

SGCP 工作与仓库中以下部分关系最紧密：

- `opencda/core/clustering/`：车辆聚类、资源分配、协同结构形成。
- `opencda/core/networking/`：V2X 网络模拟和 NS3 联动。
- `opencda/application/`：协同感知应用逻辑。
- `opencda/scenario_testing/config_yaml/`：场景、CAV 数量、通信和感知参数配置。
- `opencood/`：PointPillars、early/late/intermediate fusion 等感知后端与评估工具。

## 论文当前主要问题

根据 SGCP 论文和审稿意见，当前最需要处理的问题集中在三类。

### 写作问题

- 相关工作和 novelty 对比不足，尤其需要区分本文与通用 coalition formation 方法、Smartform 类工作、已有去中心化协同感知方法的差异。
- FullPerception baseline 设置不清楚：无 RSU 场景下是否使用虚拟 RSU、全局信息或集中调度，需要明确。
- `f(rho)` 感知效用函数标定过程过短，缺少拟合方式、数据采样方式、复现流程和误差说明。
- `T_min^stab = 500 ms`、`N_max = 4`、`rho_th = 2.0` 等关键参数缺少依据。
- topology change trigger 已形成机制规格，但尚未接入在线/离线代码统计。
- 100 ms 协作周期内的实时性论证过于简略。

### 需要补充实验

- 核心模块消融：无稳定窗口、无 coalition formation、无 PPS、仅 early fusion、仅 late fusion 等。
- 参数敏感性：`T_min^stab`、`N_max`、`rho_th`、CAV 数量、带宽/子信道数量。
- 更公平的 baseline：FullPerception-RSU、FullPerception-Decentralized、其他 V2V-only/decentralized 方法。
- `f(rho)` 标定曲线和泛化实验。
- 运行时开销：聚类耗时、调度耗时、通信耗时、融合与检测耗时。
- 稳定性指标：cluster lifetime、reconfiguration 次数、fragmentation rate。

### 需要完善机制

- 周围 cluster 已满时车辆如何处理，避免大量小 cluster 或孤立车辆。
- 是否允许 cluster merge/split，以及是否允许超过 `N_max` 的临时重组。
- 成员加入后是否重新计算已有成员边际贡献，如何避免振荡。
- coalition value 中用 late-fusion `max` 作为 baseline 的合理性和适用边界。
- potential game 的 exact potential 条件需要更严谨。
- fully decentralized 场景下全局 density/utility 信息如何传播、同步和计入开销。

## 当前文档状态

- 已建立 SGCP 文档工作区。
- 已整理论文问题和任务方向。
- 已记录 SGCP 在线 CARLA 仿真命令：`python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug`，启用 NS3 时追加 `--network`。
- 已新增 SGCP 数据导出场景入口：`v2xp_cluster_carla_datadump`。
- 已新增 OPV2V 风格离线帧加载器基础能力。
- 已完成无需 CARLA 的 OpenCOOD 单帧推理 smoke test：`E:\data\opv2v\test` 第一帧 early fusion 输出 18 个预测框、19 个 GT 框。
- 已完成一次实际 SGCP 数据导出：`D:\Data\Carla\2026_07_15_01_26_56`，20 个 CAV，每个 41 帧 `.pcd/.yaml`。
- 已验证新导出的数据可离线进入 OpenCOOD：`000060` 帧 early fusion 输出 62 个预测框、71 个 GT 框。
- 已完成无 NS3 离线全量测试：41 帧 early fusion，AP@0.3/0.5/0.7 = 0.85/0.83/0.48。
- 已梳理 SGCP 离线回放工程接口，新增 `offline_replay.md`，明确从 dump 帧重建 `cav_world/vehicle_manager/v2x_manager/lidar grid` 的最小适配层。
- 已新增离线状态适配层 `opencda.core.common.offline_replay`，可从单帧 dump 重建 `OfflineCavWorld/OfflineVehicleManager/OfflineV2XManager/OfflineLidarGrid`。
- 已新增离线 SGCP 回放入口：`python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla`。
- 已验证 `000060` 单帧无需 CARLA 可运行 `CoalitionGame`，20 个 CAV 输出 6 个 cluster。
- 已验证单帧默认 `NaiveRA` 资源分配可输出 380 条 channel allocation。
- 已完成 3 帧离线 SGCP smoke test，确认逐帧状态重建可运行。
- 已扩展 `opencda.tools.offline_replay` 多帧汇总能力，支持 `--summary-only` 输出稳定性和运行时指标。
- 已完成 `D:\Data\Carla\2026_07_15_01_26_56` 全量 41 帧离线 SGCP 回放汇总：平均 6.00 个 cluster、平均 cluster size 3.33、0 个孤立 CAV、11 次 reconfiguration、76 次 vehicle-head change、平均 cluster lifetime 6.65 帧、平均总耗时 129.78 ms、平均 RA 耗时 1.13 ms。
- 已新增资源分配算法 factory，在线 `ClusteringScheduler` 与离线 `offline_replay` 均可按配置/参数选择 `potential_game`、`pcs`、`mws`、`random`、`naive`。
- 已将 SGCP 资源分配默认值统一为配置默认 `potential_game`；`naive` 保留为 baseline/fallback。
- 已完成 `potential_game` 全量 41 帧离线 SGCP 回放：平均 6.00 个 cluster、平均 cluster size 3.33、0 个孤立 CAV、11 次 reconfiguration、76 次 vehicle-head change、平均 cluster lifetime 6.65 帧、平均总耗时 285.82 ms、平均 RA 耗时 111.85 ms。
- 已将离线 SGCP 回放接入 OpenCOOD 约束感知评估：`--sgcp-constrained` 会先运行 `CoalitionGame + potential_game`，再按在线 `CoperceptionManager` 的 grid upload 语义裁剪 frame。
- 已完成 41 帧 SGCP constrained early-fusion 评估：AP@0.3/0.5/0.7 = 0.35/0.35/0.21，平均上传 106,790.63 bytes/frame，平均 2.98 个 source CAV/frame。
- 已新增 `all-cluster-heads` 约束评估口径，并完成 41 帧全量实验：246 个簇头样本，AP@0.3/0.5/0.7 = 0.36/0.34/0.17，平均上传 109,415.48 bytes/sample，平均 2.67 个 source CAV/sample。
- 已新增 SGCP inter-cluster late fusion 离线评估，并完成 41 帧全量实验：每帧融合 6 个 cluster head，AP@0.3/0.5/0.7 = 0.77/0.73/0.35，平均上传 109,415.48 bytes/source，总 payload 26,916,208 bytes。
- 已完成 w/o PPS 第一版调度消融：`random` late-fusion 41 帧 AP@0.3/0.5/0.7 = 0.44/0.39/0.17，总 payload 9,725,376 bytes；`mws` late-fusion 41 帧 AP@0.3/0.5/0.7 = 0.31/0.26/0.11，总 payload 9,910,032 bytes。
- 已修复 `PCS/MWS/RandomRA` 离线执行接口：补齐 `PCS.run()`、`self.cav_world` 和缺失导入，使 baseline scheduler 可通过统一 resource allocation builder 运行。
- 已修复 OpenCOOD late fusion 离线推理路径，并完成 41 帧 full 20-CAV late fusion reference：AP@0.3/0.5/0.7 = 0.91/0.85/0.51。
- 已新增 `--t-min-stab` 离线参数，并完成 w/o stability window 第一版消融：`T_min_stab=0` 时 41 帧 AP@0.3/0.5/0.7 = 0.77/0.73/0.35，重配置 11 次、vehicle-head change 76 次、平均 cluster lifetime 6.65 帧，与默认 `T_min_stab=1.0` 一致。
- 已完成 `T_min^stab=100/300/500/700/1000 ms` 参数敏感性第一版：所有设置在当前 41 帧 dump 上均为 AP@0.3/0.5/0.7 = 0.77/0.73/0.35，reconfiguration 11 次，vehicle-head change 76 次。
- 已新增 `--clustering` 离线参数，并完成 w/o coalition formation 的 singleton 第一版参考：20 个单车簇、0 次重配置、AP@0.3/0.5/0.7 = 0.82/0.76/0.37；当前通信统计未计入 prediction-level late-fusion 开销。
- 已新增 `--n-max` 离线参数，并完成 `N_max=2/3/4/5/6` 参数敏感性第一版：AP@0.5 分别为 0.74/0.71/0.73/0.71/0.71；`N_max=2` 当前 AP 最高但 cluster-head source 更多，`N_max=5/6` 在当前 dump 上结果完全一致。
- 已新增 `--rho-th` 离线参数，并完成 `rho_th=0.5/1.0/2.0/3.0/4.0` 参数敏感性第一版：低阈值降低 payload 但 AP 下降，高阈值提升 AP@0.7 到 0.37 且增加 payload；当前默认 `2.0` 是通信-精度折中点。
- 已新增 `--cav-count` / `--cav-ids` 离线参数，并完成 5/10/15/20 CAV 子集规模敏感性第一版：AP@0.5 分别为 0.32/0.59/0.66/0.73；该实验是同一 dump 的协同车辆子集，不等同真实交通密度重采样。
- 已新增 `--num-channels` / `--bandwidth-mhz` 离线参数，并完成网络资源敏感性第一版：5/10/20 子信道 AP@0.5 分别为 0.53/0.73/0.73，平均上传 payload 分别为 60,225/109,415/139,300 bytes/source，平均 selected grids 分别为 45.58/87.32/117.18。
- 已完成 PPS 带宽参数第一轮复核：`bandwidth_per_channel` 已进入 `PotentialGame` 的 max-grid、SINR 和 data-rate 计算；20/40/80 MHz 当前结果一致，是因为本 dump 的调度不由带宽上限主导，主要受子信道数量、每簇头 `B_h=1` RB 和候选网格集合约束。
- 已完成低带宽瓶颈触发实验：0.1/0.5/1.0 MHz 下 AP@0.5 分别为 0.22/0.50/0.61，平均 selected grids 分别为 0.00/4.32/9.66，证明带宽约束可观测生效。
- 已新增 `baseline_fairness.md`，明确 FullPerception-RSU 是 centralized/RSU-assisted upper reference，不作为同通信预算公平主对比；full 20-CAV early/late fusion 也只作为 upper/reference。
- 已新增 same-budget CAV-only selective-sharing baseline：`nearest`、`density`、`communication_aware` 三种成员选择，复用 SGCP clustering + inter-cluster late fusion，默认每簇头 2 个成员、87 个 grid budget；41 帧结果分别为 AP@0.3/0.5/0.7 = 0.76/0.73/0.37、0.77/0.74/0.39、0.78/0.75/0.40。
- 已新增 `topology_trigger.md`，定义 SGCP topology change trigger 的邻居变化、相对运动、链路质量、utility 下降、hard failure 和 periodic guard 条件，并明确 `NO_CHANGE/LOCAL_REPAIR/RECLUSTER` 三类输出。
- 已将 topology trigger 统计接入 `opencda.tools.offline_replay`：支持 summary 输出和 `--print-topology-events` 逐 transition 明细；当前支持 `dump` 与 `pose_delta` 两种速度源，默认使用相邻帧位置差分速度。
- 已完成 topology trigger 速度源复核：`ego_speed` 来自 `get_speed(vehicle)`，默认单位是 km/h；`pose_delta` threshold 3/4 m/s 可覆盖 11 次实际 reconfiguration，5 m/s 会漏掉 2 次。
- 已将 topology trigger gate 接入在线 `ClusteringV2XManager` first version：默认关闭；打开后按 hard failure、邻居集合变化和 periodic guard 决定是否重跑 `CoalitionGame`，否则沿用上一轮 cluster。
- 已新增 `cluster_capacity_policy.md`，明确 cluster 已满时采用硬 `N_max`、保留当前 cluster、加入未满更优 cluster、singleton fallback 和 inter-cluster late fusion 补偿；merge/split 不作为独立无限制原语。
- 已将 cluster capacity 统计接入 `opencda.tools.offline_replay`：默认 `N_max=4` 的 41 帧 dump 中，平均每帧 3.12 个满簇、99.15 次满簇候选跳过、singleton cluster ratio 为 0、small-cluster ratio 为 0.187；`N_max=2/3/5/6` sweep 已写入 `results.md`。
- 已新增 `opencda.tools.sgcp_density_calibration` 和 `f_rho_calibration.md`，完成 41 帧 dump 的 `f(rho)` 标定第一版：788,020 个 CAV-grid 样本中非零网格占 5.98%，非零 density p90=1.40、p95=3.60；默认 `rho_th=2.0` 筛出 3,383 个 high-density grid，约占非零网格 7.18%。
- 已将 SGCP control overhead 估算接入 `opencda.tools.offline_replay`，并新增 `control_overhead.md`。当前 41 帧默认 SGCP 的控制面总开销为 187,112 bytes，平均 4,563.71 bytes/frame，约为 inter-cluster late-fusion 点云 payload 的 0.70%。
- 已新增 `potential_game_conditions.md`，复核 PPS exact potential 成立条件。当前代码更准确的口径是 potential-guided constrained best-response scheduling；若论文继续使用 exact potential game，需要限定固定 cluster、固定候选 grid、硬 feasibility constraints，并补显式势函数/monotonicity 证明。
- 已将 PPS convergence diagnostics 接入 `PotentialGame` 与 `offline_replay`：当前 41 帧默认 SGCP 中 41/41 帧均在 `max_iter=20` 前收敛，平均 3.00 iterations、10.00 scheduled links/frame、523.90 selected grids/frame，最大 RB occupancy 为 1。
- 已新增 `paper_revision_plan.md`，把 topology trigger 表述矛盾、实时性、`f(rho)` 标定、baseline fairness 和 game-theoretic convergence 转成 `main.tex` 级别替换建议；其中 topology trigger 矛盾的 P4 写作项已完成。
- 已新增 `related_work_novelty_revision.md`，完成 P4 related work / novelty 写作口径：SGCP 的新意不写成 coalition game 本身，而是感知效用标定、稳定性约束、容量约束、PPS 子信道可行性和分层 early/late fusion 的组合。
- 已新增 `parameter_calibration_revision.md`，完成 P4 `f(rho)` 标定过程和 `T_min^stab/N_max/rho_th` 参数依据写作口径；特别明确当前短序列不能证明 `T_min^stab=500 ms` 最优，只能写为五个感知周期的保守默认。
- 已新增 `fullperception_baseline_revision.md`，完成 P4 FullPerception baseline 写作口径：FullPerception-RSU、full 20-CAV early/late fusion 作为 centralized/infrastructure-assisted upper reference；同通信预算主对比采用 CAV-only nearest/density/communication-aware selective sharing，并明确当前短序列上 SGCP 不能声称 AP 全面领先强 selective baseline。
- 已扩展 `opencda.tools.offline_replay` 输出 `runtime_breakdown_ms`，并新增 `runtime_feasibility_revision.md`。当前 41 帧控制面 profiling 中，SGCP algorithm total 平均 105.24 ms，coalition formation 64.39 ms，PPS scheduling 40.58 ms；离线读取和 world build 约 599.73 ms/frame，属于 replay artifact，不计入在线周期。论文中应写为 near-real-time feasibility，而不是完整端到端 100 ms 保证。
- 已新增 `reproducibility_manifest.md`，固定当前可复现实验的 OpenCDA commit、数据集、命令、结果和 artifact 路径；同时确认 `main.tex` 旧主表 `NC/RS/MUG/FullPerception/Ours = 0.13/0.31/0.37/0.81/0.85@AP0.3...` 尚未找到原始日志、随机种子和代码版本，论文修订应以当前复现结果替换或找回旧日志后再使用。
- 已完成真实 CARLA 在线 topology-trigger gate 短回归，并新增 `online_topology_gate_regression.md`。通过 `OPENCDA_CLUSTERING_CONFIG=networking_clustering_topology_gate.yaml` 打开 gate，通过 `OPENCDA_ONLINE_TICKS=35` 自动结束在线仿真；回归正常退出，CP counter=8，AP@0.3/0.5/0.7 = 0.84/0.82/0.69，cluster trigger 为 `initial=1`、`neighbor_set_change=1`、`head_member_unreachable=3`、`skip=0`。结论：gate 接入和日志生效，但当前 35 m 通信范围下 hard condition 持续触发，不能用该回归证明 skip 收益。
- 已进入主表修复阶段，新增 `opencda.tools.offline_inference --sgcp-trace-output` 和 `protocol_audit.md`。第一轮离线协议审计显示：41 帧 SGCP inter-cluster late-fusion 共有 246 条 cluster-head receiver trace，`missing_channel_rows=0`，总通信 26,916,208 bytes，AP@0.3/0.5/0.7 = 0.77/0.73/0.35；分簇、grid selection 和 channel allocation 均已真实进入融合输入，暂未发现未调度 sender 绕过 PPS 的协议 bug。
- 已新增 `--sgcp-upload-mode {grid,head_only,full_cluster}` 和 `mechanism_probe.md`。41 帧机制 probe 显示 head-only 为 0.26/0.22/0.09，SGCP grid-constrained 为 0.77/0.73/0.35，full-cluster upload 为 0.82/0.79/0.42；SGCP 使用约 60.0% 的 full-cluster payload，主要 AP 损失集中在 grid/PPS 选择，cluster formation 和 inter-cluster late fusion 主体可用。
- 已新增 `--sgcp-grid-selection-mode {utility,random}`。Random grid probe 保留同一 PPS scheduled links 和每条 link 的 grid 数量，仅将具体 grid 替换为确定性随机候选；41 帧结果为 AP@0.3/0.5/0.7 = 0.78/0.75/0.36，总 payload 27,908,560 bytes，`missing_channel_rows=0`。该结果略高于当前 utility selection，说明 grid utility 对检测 AP 的排序能力不足，是下一轮算法改造优先目标。
- 已扩展 grid scoring / selection 改造探针：`raw_density` 为 0.74/0.70/0.37，`density_distance` 为 0.74/0.71/0.37，`spatial_diverse` 为 0.79/0.75/0.37。三者均保持 SGCP/PPS protocol path；其中 `spatial_diverse` 保留同一 scheduled links 和 grid count，用 density-aware spatial cover 替换原始选格，当前高于 utility 与 random-grid，说明 coverage-aware grid selection 是可继续推进的主表修复方向。
- 已完成 SGCP 11 帧离线 NS3 request-level replay：154 条 SGCP intra-cluster request，CAM callback delivery ratio 0.558442，request with any RLC RX event ratio 0.974026。后者仅表示 request_id 至少出现一个 RLC RX 片段/事件，不代表完整 request delivery。
- 已确认当前可运行环境版本快照，并记录到 `docs/doc_workspace/environment.md`：OpenCDA HEAD `fcc29fdc9ee9a9fe694c12e1fb6792b4d41bccac`，Python 3.7.10，CARLA Python API 0.9.11，PyTorch 1.10.0+cu113，Open3D 0.10.0.0，ns-3 wrapper `ns-3-dev-v2x-v1.1-dirty`。
- 已修复在线 CARLA-NS3 时间流速不一致问题：`NetworkManager.time_slot` 不再将 `world.fixed_delta_seconds` 除以 5，`current_sim_time`、`current_time_slot` 与 CARLA tick 统一推进。
- 已修复在线 NS3 初始化顺序：sender 线程等待车辆注册后先发送真实车辆数和第一帧 `vehicles_position`，再进入 `sync_request/sync_ack` 循环。
- 已新增 `opencda.tools.offline_ns3_replay`，可不启动 CARLA，直接从 dump 数据驱动 NS3 同步和传输请求。
- 已完成 3 帧离线 NS3 smoke test：帧时间 0.000/0.100/0.200 s，20 车、6 个 cluster、每帧 14 条 intra-cluster upload request，NS3 返回多条 `cam_received`。
- 已修复 NS3 manual subchannel 链路：NS3 默认按 OpenCDA 10 个子信道配置可用资源，manual scheduler 严格匹配 `physicalStart == sc_start` 和 `physicalLen == sc_num`，不再对越界子信道取模 wrap；长 JSON socket payload 改为累积解析；RLC command size 修正，避免最后残片落入默认随机调度。
- 已完成 `opencda.tools.ns3_link_probe` 四类链路回归：`success`、`edge_success(sc_start=9)` 均全量 CAM delivery；`conflict` 中两请求均落到 physicalStart=0，并产生 `PSCCH_DECODE_FAIL reason=decoded_overlap`，仅 1/2 CAM delivery；`out_of_band(sc_start=10)` 产生一次 `MANUAL_CMD_REJECT reason=out_of_band totalSubCh=10`，无 RLC TX/RX/CAM。
- 已将 `opencda.tools.offline_ns3_replay` 的 SGCP 资源分配从硬编码 `NaiveRA` 对齐为配置默认 `potential_game`，并默认跳过没有 `sc_start/sc_num` 的未调度需求，防止绕过 OpenCDA PPS 进入 NS3 默认调度。
- 已完成修复后 SGCP potential_game 11 帧 NS3 replay：110 条 PPS scheduled request、44 条 skipped unscheduled demand、CAM delivery 110/110、RLC RX 2970/2970、PHY failures 0、manual reject 0。
- 已补齐 NS3 request-level 三层统计：application callback、RLC request completion/partial/no_tx、PHY decode diagnostics。`offline_ns3_replay` 现使用全局唯一 `pkt_id`，避免 RLC 延迟事件跨帧错配。
- 已完成 NS3 暴露带宽/子信道窗口回归：`targetSubchannels=10` 时 110/110 PPS request application/RLC complete；`targetSubchannels=5` 时 `sc_start=0..4` 的 55 条全部 complete，`sc_start=5..9` 的 55 条全部在 bridge 层 `MANUAL_CMD_REJECT reason=bridge_out_of_band`，无 CAM/RLC/PHY 污染。
- 已将 NS3 request-level delivery 接入 communication-aware selective-sharing baseline：`offline_inference --ns3-link-quality-csv <rlc_by_request.csv>` 使用 `rlc_complete` 作为 link-quality cost。11 帧对照中，distance proxy 为 AP@0.3/0.5/0.7 = 0.71/0.67/0.31、总通信 7,977,680 bytes；NS3 RLC-complete aware 为 0.68/0.63/0.27、总通信 7,796,560 bytes。

## 当前阻塞项

- 论文旧主表已经确认缺少原始日志、随机种子和复现实验配置；当前处理策略是使用 `reproducibility_manifest.md` 中的已复现离线结果修订论文，或后续人工找回旧日志后再恢复旧表。
- 当前 OpenCDA 与 ns-3 工作区均存在未提交/dirty 状态；NS3 exposed-subchannel 修复与 request-level 统计已通过离线回归，但仍需要分别在 OpenCDA 仓库和 co-simulation 仓库提交。
- 离线 SGCP 回放已完成到“多帧 clustering + `potential_game` 资源分配 + 稳定性/运行时指标 + OpenCOOD 约束感知 mAP”层。
- 论文 SGCP 的 PPS/博弈调度当前按配置默认 `potential_game` 执行；已完成 `RandomRA/MWS` 初版对比，但 MWS 结果低于 random，需要结合论文文本复核 baseline 定义与效用函数。
- 早期 `ego-cluster-head` 与 `all-cluster-heads` constrained 评估只包含 intra-cluster early fusion；论文完整 SGCP 结果应优先采用 inter-cluster late fusion 口径。
- full 20-CAV late fusion reference 使用独立 OpenCOOD late checkpoint，不能直接等同于严格同通信约束的 SGCP late-only 消融。
- 当前 `T_min_stab=0` 消融未显示差异，说明当前 41 帧 dump 不足以证明稳定窗口贡献；需要更长序列或更强相对运动/topology change 场景。
- `T_min^stab=100-1000 ms` 参数实验同样未显示差异；当前只能作为“短序列无敏感性”的工程记录，不能作为论文中参数选择依据。
- singleton-cluster 结果会 late-fuse 全部 20 个 CAV 的检测框，但当前只统计点云 payload，不能直接作为零通信公平 baseline；需要补检测框交换开销或实现距离/随机固定簇对比。
- FullPerception-Decentralized / same-budget CAV-only selective baseline 已有 first version；communication-aware baseline 现在同时支持 distance proxy 与 NS3 RLC-complete cost。当前 dump 上 distance proxy AP@0.5/AP@0.7 高于 SGCP 且 payload 更高；NS3-aware 11 帧结果显示链路可行性约束会降低 AP 和通信量。论文主张需要谨慎转向稳定性、PPS channel feasibility 和动态网络约束，而不是简单宣称 AP 全面领先。
- `N_max` 参数实验显示非单调趋势；进入论文前需要补更长序列/不同密度场景，并计入 inter-cluster 检测框交换开销。
- `rho_th` 参数实验已显示通信-精度折中，`f(rho)` 密度分布标定已有第一版；论文级结论仍需要补跨场景/探测器泛化，必要时把 density bin 与 per-grid detection recall/IoU 绑定。
- CAV 数量规模实验目前只是固定场景子集实验；论文级“密度扩展”仍需重新导出不同 CAV/背景车密度的 CARLA 场景。
- 网络资源实验已证明子信道数量影响 PPS 结果；低带宽 stress test 已触发带宽瓶颈。论文中应谨慎区分“常规带宽下该 dump 已饱和”和“极低带宽下吞吐约束有效”。
- Dump 中速度字段目前使用 `ego_speed`；后续如需更严格动态稳定性，应确认单位并评估是否改为相邻帧差分速度。
- 在线 CARLA-NS3 修复尚未在真实 CARLA 图形仿真中长时间回归；当前已通过离线 NS3 socket/sync smoke test 和本地时间基准单元断言。
- Topology trigger 已接入离线 replay 统计，但尚未接入在线 `ClusteringV2XManager` gate；单独 relative-speed trigger 仍偏敏感，在线 gate 应结合 neighbor-set change、utility drop 和 `T_min_stab` 滞回。
- 在线 topology trigger gate 已完成真实 CARLA 35 tick smoke regression；未观察到 skip，原因是当前默认 35 m 通信范围下持续触发 `head_member_unreachable`。若论文需要展示 reduced reconfiguration，应补一组更静态或更大通信范围的在线回归。
- Cluster capacity 策略已有离线统计支撑；optional replacement repair 尚未实现，进入论文前可明确为 future/optional enhancement。
- PPS `PotentialGame` 当前没有显式势函数、action replacement 和 `Delta Phi >= 0` 日志；论文中不宜无条件声称当前实现是完整 exact potential game。若要保留强理论表述，需要补代码诊断和证明。
- SGCP potential_game NS3 replay 已确认“PPS 已调度且无冲突的 request 全部成功”，且低暴露子信道场景能正确拒绝超出带宽窗口的 request；NS3 delivery/PDR 已先接入 selective-sharing baseline。下一步是把该 link-quality 反馈进一步接入 SGCP PPS 本身或 OpenCOOD mAP 的端到端丢包裁剪。
- Random-grid probe 已显示当前 utility selection 不优于随机候选；`spatial_diverse` 已超过 random-grid，但距离 full-cluster upper reference 仍有差距。主表修复下一步应将 coverage-aware selection 固化为 SGCP 机制，并补 fixed-cluster/fixed-link、payload sweep 和 NS3-aware 丢包裁剪。
