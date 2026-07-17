# SGCP 当前状态

更新时间：2026-07-17

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
- 已完成 `spatial_diverse` 子信道 sweep：5/10/20 子信道分别为 AP@0.3/0.5/0.7 = 0.56/0.53/0.27、0.79/0.75/0.37、0.80/0.76/0.41；payload 分别为 14,815,408、28,743,280、37,912,544 bytes。10 子信道是低通信主点，20 子信道 AP@0.7 接近 full-cluster 0.42 且 payload 低约 15.5%。
- 已将 `offline_ns3_replay` 对齐 coverage-aware 主表候选，支持 `--num-channels`、`--bandwidth-mhz`、`--sgcp-grid-score-mode` 和 `--sgcp-grid-selection-mode`。10 子信道 `spatial_diverse` 11 帧 NS3 replay 完成：110/110 scheduled request application/RLC complete，44 条未调度需求被 OpenCDA replay 跳过；20 子信道 `spatial_diverse` 完成：154/154 scheduled request application/RLC complete。两者 `MANUAL_CMD_REJECT=0`，PSCCH/PSSCH decode failure 均为 0。
- 已补齐 `offline_ns3_replay --rho-th` 参数，并完成 `spatial_diverse` 10 子信道 `rho_th=3.0` 的 11 帧 NS3 replay：110/110 scheduled request application/RLC complete，44 条未调度需求跳过，PHY failures 为 0。至此主表推荐的 10ch tuned low-budget row 不再是 NS3 pending。
- 已补全 FullPerception 口径：当前 RSU-free dump 无真实 FullPerception-RSU；可复现 centralized full 20-CAV early reference 为 AP@0.3/0.5/0.7 = 0.85/0.83/0.48，non-ego CAV 上传 payload 60,838,528 bytes。FullPerception-Decentralized 应使用 CAV-only selective sharing；高预算 density/communication-aware selective baseline 为 0.80/0.76/0.40，payload 37,710,864 bytes。
- 已确认旧 Random/MWS scheduler payload 过低（约 9.7/9.9 MB），未充分利用 10 子信道资源，不适合作“通信量减少”的主表证据。它们保留为 w/o PPS 消融；公平主对比改用 payload-matched selective baselines、random-grid same-link probe 和 SGCP spatial-diverse 10/20ch。
- 已完成 `spatial_diverse` 的 `rho_th` 点云阈值 sweep：`rho_th=1/2/3/4` 中，`rho_th=3.0` 达到 0.79/0.76/0.38，payload 29,405,296 bytes，是默认 10ch 低通信候选之外的更高 AP 阈值配置。
- 已新增 `main_table_candidate.md`，把可复现结果收束为论文主表候选：FullPerception centralized upper reference 118.71 Mbps、payload-matched selective high-budget 73.58 Mbps、SGCP coverage-aware 10ch `rho_th=3` 为 57.38 Mbps、SGCP coverage-aware 20ch 为 73.98 Mbps，并明确 Random/MWS 只作消融。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第一轮论文落地：替换旧主表、旧通信开销图引用和过强 FullPerception 对比，改为 centralized full-sharing upper reference、capacity-matched V2V baseline、SGCP coverage-aware 10/20ch、NS3 request-level delivery 和 near-real-time feasibility 叙事。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第二轮机制修订：PPS 机制改为 potential-guided constrained scheduling，加入 coverage-aware / density-aware spatial diversification grid selection；同时修正 topology trigger 与 every-cycle PPS 的关系，并弱化无条件 exact-potential/Nash guarantee。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第三轮参数标定修订：把 `f(rho)` / `rho_th` 标定写成可复现协议，并加入 coverage-aware 10ch `rho_th` sensitivity 表，明确 `rho_th` 依赖 LiDAR、grid size、预处理和 detector backbone，不作为通用常数。
- 已完成 `C:\Workspace\icdcs-paper\SGCP\main.tex` 第四轮参数依据修订：补入 `N_max=4` 的容量控制解释、99.15 次/frame 满簇候选跳过统计，以及 `T_min^stab=500 ms` 作为五个 10 Hz sensing cycle 的保守滞回默认；同时说明当前短序列中 `T_min^stab=100--1000 ms` 不改变 AP/reconfiguration。
- 已新增 `rebuttal_draft.md`，按 R2/R3/R4 concern 整理可直接迁移的 rebuttal 素材，覆盖 coalition baseline、满簇处理、FullPerception 公平性、`f(rho)` 标定、`T_min^stab`、公平 baseline、runtime、topology trigger 和 NS3 request-level delivery。
- 已新增 `rebuttal_short.md`，将长 rebuttal 素材压缩为最终可粘贴版本，保留 FullPerception 公平性、payload-matched V2V baseline、`f(rho)`/`rho_th`/`N_max`/`T_min^stab` 参数依据、runtime、topology trigger、NS3 三层可靠性和更保守的 claim boundary。
- 已新增 `online_ns3_short_regression.md`，明确真实 CARLA+NS3 短回归的三段启动顺序、日志保存位置、时间同步验收条件和 subchannel 语义检查项。本轮已通过 `test_network_time_sync` 轻量回归，且确认没有 CARLA/NS3 残留进程。
- 已执行真实 CARLA+NS3 35 tick 短回归并修复在线初始化时序 bug：原先 sender thread 在 1 车注册时就发送 `vehicles_num=1`，导致 NS3 之后收到 20 车位置帧时重新初始化并 address collision / SIGABRT。现在 `NetworkManager` 等待 `mark_vehicle_registration_complete()` 后再初始化 NS3，并过滤 `carla_id=None` 帧；修复后 NS3 20 车初始化、38/38 sync_ack、无 sync timeout、无 fatal、无 manual reject，在线 AP@0.3/0.5/0.7 = 0.86/0.84/0.74。
- 已诊断并修复在线 PHY failure 的第一类原因：`PotentialGame` 每轮只清空算法内部 strategies，没有清空各车辆 `ClusteringScheduler.channel_allocation`，导致在线多轮 CP 旧链路残留；修复后真实 CARLA+NS3 35 tick 短回归中 PSCCH/PSSCH decode failures 从 `1836/480` 降至 `95/10`，`MANUAL_CMD_REJECT=0`，在线 AP@0.3/0.5/0.7 从 `0.86/0.84/0.74` 升至 `0.88/0.88/0.79`。
- 已新增 `opencda.tools.online_ns3_log_eval`，将在线 OpenCDA 重复 incomplete 轮询行压缩为 upload episode。最新策略清空回归中 `184` 行 incomplete 只有 `6` 个真实 partial episode，且每个都缺少一个 `10000` bytes fragment；这将剩余在线问题收窄到 fragment-level loss 后的重传/重调度或失败裁剪，而不是时间同步或子信道语义。
- 已新增可配置的在线 NS3 timeout reupload：`upload_timeout_slots`、`re_upload_when_timeout`、`max_reupload_attempts`。first trial 中 complete/partial episode 从 `21/6` 改善到 `39/3`，但暴露 late CAM completion 引发的 `KeyError`；已修复为安全处理缺少 `uploading_cavs` start slot 的 late completion。修复后的 clean rerun 尚未完成：一次启动等待超时，一次 CARLA spawn collision，均未进入有效 NS3/reupload 验证。
- 已新增 `OPENCDA_CLEAN_WORLD_ON_INIT=1` 在线回归保护开关：ScenarioManager 在创建 CAV 前清理已有 `vehicle/sensor/walker/controller` 动态 actor 并 tick 3 次，以避免固定 spawn 点被上一轮残留 actor 占用。默认关闭，SGCP 在线自动回归命令显式打开。
- 已新增 `OPENCDA_CARLA_CLIENT_TIMEOUT`，允许在线回归把 CARLA client/load_world timeout 从默认 60 秒提高到 180 秒；最近两次 clean rerun 的 blocker 已从 spawn collision 转移为 `load_world('Town03')` 超时。
- 已新增 `OPENCDA_USE_CURRENT_CARLA_WORLD=1`，用于 CARLA 已经以目标地图启动时跳过 `client.load_world(town)`，避免本轮反复出现的 `load_world('Town03')` RPC 超时。
- 已确认当前在线 clean rerun 的最新 blocker 是 CARLA RPC readiness：直接用 Town03 参数启动 CARLA 并等待 120 秒后，`carla.Client(...).get_world()` 在 180 秒 timeout 下仍超时。2000 端口 ready 不代表 CARLA Python API 可用；继续 NS3/reupload 前必须先恢复 CARLA smoke test。
- 已新增 `opencda.tools.carla_rpc_probe`，在线 OpenCDA/NS3 前必须先通过 `conda run -n opencda python -m opencda.tools.carla_rpc_probe --expect-map Town03 --timeout 30 --wait 180`。
- 已将 CARLA RPC probe 和在线保护开关补充到全局 `docs/doc_workspace/environment.md`，作为 SGCP/LGCP 共用前置检查。
- 已新增 `opencda.tools.sgcp_protocol_trace_summary`，将 `offline_inference --sgcp-trace-output` 的 246 行 receiver trace 汇总为 41 行 frame summary。当前 SGCP 41 帧审计中每帧固定 6 个 cluster-head receiver、10 条 PPS channel links、平均 16 个 fused CAV、平均 10 个 uploaded sources、平均 payload 656,492.88 bytes/frame、平均 selected grids 523.90/frame，`missing_channel_rows=0`。AP 仍按 OpenCOOD 全局累计指标报告，避免把 pred/GT count 误当逐帧 AP。
- 已新增 `offline_inference --clustering fixed_first_frame` 机制 probe：首帧运行 coalition game 固定 head/member 模板，后续帧只复用 cluster membership，每帧仍重新计算 density、PPS 和融合。41 帧结果为 AP@0.3/0.5/0.7 = 0.73/0.70/0.33，总 payload 26,325,216 bytes，`missing_channel_rows=0`；动态 coalition 的 0.77/0.73/0.35 仍更好，说明 topology-aware cluster 更新有贡献，但主表主要损失仍来自 grid/PPS 选择质量。
- 已新增 `offline_inference --head-rb-budget` 和 `offline_ns3_replay --head-rb-budget`，用于覆盖 `PotentialGame` 每簇头最多使用 RB 数 `B_h`。默认仍为 1。41 帧 `spatial_diverse,B_h=2` 为 0.75/0.72/0.41、27,086,400 bytes；`spatial_diverse,rho_th=3,B_h=2` 为 0.76/0.72/0.42、27,962,864 bytes、54.56 Mbps。该设置达到 full-cluster AP@0.7，但 AP@0.3/0.5 下降。11 帧 NS3 replay 已验证 `B_h=2,rho_th=3` 为 110/110 application/RLC complete、PHY failures 0。
- 已新增 `offline_inference --sgcp-late-nms-thresh` 做 inter-cluster late fusion NMS 消融。`spatial_diverse,rho_th=3,B_h=2` 下，late NMS 0.05/0.15/0.30 分别为 0.73/0.70/0.40、0.76/0.72/0.42、0.75/0.71/0.41；默认 0.15 最好，说明 AP@0.3/0.5 下降不是简单 NMS 阈值问题。
- 已新增 `opencda.tools.sgcp_late_fusion_log_summary`，从 offline inference stdout 汇总 source/fused prediction boxes、suppressed boxes、fused GT 和 payload。诊断显示 `B_h=2,rho_th=3` 的 fused GT 平均为 64.83，低于 `B_h=1,rho_th=2` 10ch 的 69.00 和 20ch 的 69.29；fused prediction 平均也从 55.90 降到 53.71。该结果支持“`B_h=2` 改善高 IoU 定位但缩窄覆盖/召回面”的解释，后续应检查 cluster-head/member selection 与 target coverage，而不是继续调 late NMS。
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
- 在线 CARLA-NS3 时间同步修复已通过真实 35 tick 图形短回归；online scheduler 残留策略导致的主要 PHY collision 已通过策略清空修复显著缓解。当前应用层 timeout reupload 已将 partial episode 从 6 降到 3，但 clean rerun 仍 pending；下一步应先让 `opencda.tools.carla_rpc_probe --expect-map Town03` 通过，再直接用 Town03 + 三个环境开关重跑 clean online reupload。
- Topology trigger 已接入离线 replay 统计，但尚未接入在线 `ClusteringV2XManager` gate；单独 relative-speed trigger 仍偏敏感，在线 gate 应结合 neighbor-set change、utility drop 和 `T_min_stab` 滞回。
- 在线 topology trigger gate 已完成真实 CARLA 35 tick smoke regression；未观察到 skip，原因是当前默认 35 m 通信范围下持续触发 `head_member_unreachable`。若论文需要展示 reduced reconfiguration，应补一组更静态或更大通信范围的在线回归。
- Cluster capacity 策略已有离线统计支撑；optional replacement repair 尚未实现，进入论文前可明确为 future/optional enhancement。
- PPS `PotentialGame` 当前没有显式势函数、action replacement 和 `Delta Phi >= 0` 日志；论文中不宜无条件声称当前实现是完整 exact potential game。若要保留强理论表述，需要补代码诊断和证明。
- SGCP potential_game NS3 replay 已确认“PPS 已调度且无冲突的 request 全部成功”，且低暴露子信道场景能正确拒绝超出带宽窗口的 request；NS3 delivery/PDR 已先接入 selective-sharing baseline。下一步是把该 link-quality 反馈进一步接入 SGCP PPS 本身或 OpenCOOD mAP 的端到端丢包裁剪。
- Random-grid probe 已显示当前 utility selection 不优于随机候选；`spatial_diverse` 已超过 random-grid，且 20 子信道下 AP@0.7 接近 full-cluster upper reference。10ch `rho_th=2/3` 与 20ch `spatial_diverse` 候选的 NS3 request-level delivery 均已验证。最新 box-count 诊断显示 `B_h=2` 的低阈值 AP 下降主要与 fused GT 覆盖减少有关。主表候选、coverage-aware PPS 机制、`f(rho)` 参数标定、`N_max/T_min` 参数依据、rebuttal 长草稿和短版均已完成；下一步优先检查 cluster-head/member coverage 与 payload-matched fallback，再补真实在线 CARLA/NS3 短回归或压缩 rebuttal。
