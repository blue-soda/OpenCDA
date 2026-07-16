# SGCP 任务清单

更新时间：2026-07-16
最终目标：解决所有C:\Workspace\icdcs-paper\SGCP中的审稿意见，并使 SGCP 主表在严格实验协议下以较少通信量获得较高 AP。

## P-1：最高优先级 - 主表结果修复与论文落地

- [x] 审计离线 SGCP 测试链路，确认分簇结果是否真实决定每个 cluster head 的融合对象、成员集合和 inter-cluster late fusion 输入。已新增 `protocol_audit.md` 和 `--sgcp-trace-output`；41 帧输出 246 条 receiver trace，cluster head/member/source 列表与融合输入一致。
- [x] 审计点云选择链路，确认 `PotentialGame` 输出的 grid selection 是否真实裁剪 sender 点云，并进入 OpenCOOD early fusion 输入；输出逐帧/逐 CAV 的 selected grids、点数、payload 和 AP 关联日志。41 帧 trace 已记录 selected grids、point counts、payload、pred/gt boxes。
- [x] 审计子信道分配链路，确认 `sc_start/sc_num` 不只用于 NS3 replay，也真实约束离线/在线传输请求；对未调度 member 或超出子信道窗口的请求进行 drop/delay，而不是绕过 PPS 进入融合。41 帧 trace 中 `missing_channel_rows=0`，未发现未调度 sender 绕过 PPS 进入融合。
- [ ] 构造离线单帧可解释 probe：同一帧分别运行 full early、SGCP grid-constrained、关闭 grid selection、随机 grid selection、固定 cluster membership，对比输入点云数量、预测框和 AP，确认每个机制开关能改变融合结果。已完成 head-only / grid-constrained / full-cluster / random-grid / raw-density / density-distance / spatial-diverse 41 帧机制 probe；`spatial_diverse` 在相同 scheduled links 和 grid count 下达到 `0.79/0.75/0.37`，当前优于原始 utility 与 random-grid，还需补固定 cluster membership。
- [ ] 构造离线多帧协议一致性 probe：输出 cluster、grid selection、channel allocation、fused CAV ids、payload、prediction count、GT count 和 AP 的逐帧 CSV。
- [ ] 用真实 CARLA 在线无 NS3 短回归验证 cluster membership、grid selection 和融合结果一致性；确保 CARLA 进程至多一个，并记录完整命令、日志和 AP。
- [ ] 用真实 CARLA + NS3 或离线 NS3 replay 验证时间同步和 transfer request 语义：CARLA tick / OpenCDA network time slot / NS3 sync time 必须一致；带宽内无冲突请求成功，带宽外请求延迟或丢包。已完成 default 10/5 子信道回归，以及 `spatial_diverse` 10ch `rho_th=2/3` 和 20ch replay：10 子信道均为 110/110 request application/RLC complete、44 条未调度需求跳过；20 子信道 154/154 request application/RLC complete；PHY failures 均为 0。后续可补真实 CARLA+NS3 短回归。
- [ ] 若协议链路存在 bug，优先修复并重跑主表；修复项必须配套离线 probe、NS3 request-level trace 和必要的在线 CARLA smoke test。
- [ ] 若协议无误但主表 AP 仍低，展开细致消融定位原因：cluster 质量、grid utility、`rho_th`、`N_max`、member budget、channel budget、inter-cluster late fusion/NMS、检测框坐标变换、payload/AP tradeoff。已新增 `mechanism_probe.md`；当前定位为原始饱和 density utility selection 不足，coverage-aware `spatial_diverse` 是正向改造。
- [ ] 在保持论文叙事的前提下改造算法以提升主表：优先将 `spatial_diverse` 整理成 coverage-aware grid utility / candidate scoring，使其在相同 scheduled links 和相同 grid count 下稳定优于 random-grid。已完成 5/10/20 子信道 sweep，10 子信道为低通信候选 `0.79/0.75/0.37`，20 子信道为高预算候选 `0.80/0.76/0.41`；10/20 子信道候选均已通过 NS3 request-level replay。随后再评估 member selection、cluster-head selection、late fusion weighting、fallback sharing 或 topology-trigger 策略，但必须保留“较少通信量 + 分层融合 + 子信道可行调度”的核心主张。
- [x] 重新生成主表候选结果：至少包含 NC、RS/random、MUG/MWS 或强 selective baseline、Full/reference、SGCP；统一报告 AP@0.3/0.5/0.7、payload/Mbps、control overhead、runtime 和 NS3 delivery。已新增 `main_table_candidate.md`，包含 FullPerception centralized、full-cluster reference、payload-matched selective baselines、SGCP coverage-aware 10ch/20ch、Mbps 换算和不建议进入公平主表的 Random/MWS 说明。
- [x] 修正主表 baseline 口径：FullPerception-RSU 在当前 RSU-free dump 上不可直接填实测，应以 full 20-CAV early `0.85/0.83/0.48`、60,838,528 bytes 作为 centralized FullPerception upper reference；Random/MWS scheduler 因 payload 过低只作 w/o PPS 消融；公平主对比使用 payload-matched selective sharing（高预算 0.80/0.76/0.40，37,710,864 bytes）和 SGCP spatial-diverse 10/20ch。
- [x] 补充通信量可调参数实验：已完成 `spatial_diverse` `rho_th=1/2/3/4` sweep，其中 `rho_th=3.0` 为 0.79/0.76/0.38，payload 29,405,296 bytes；`rho_th=3.0` 的 10ch NS3 replay 也已验证 110/110 application/RLC complete。
- [x] 当主表结果达到论文可写水平后，修改 `C:\Workspace\icdcs-paper\SGCP\main.tex`：已完成第一轮替换，删除旧 `0.85/0.84/0.69` 和 `22.33 Mbps` 主张，改为 FullPerception centralized upper reference、payload-matched selective baseline、SGCP coverage-aware 10/20ch、NS3 request-level delivery 与 near-real-time feasibility 口径。
- [x] 根据最新结果更新 rebuttal 答复，覆盖审稿意见中的 FullPerception 公平性、decentralized baseline、`f(rho)` 标定、500 ms 参数、100 ms 实时性、topology trigger、NS3/通信可靠性。已新增 `rebuttal_draft.md`，作为 reviewer-by-reviewer response 草稿。
- [ ] 将过程中发现的新问题回写到 `status.md` 和 `target.md`，并在 `log.md` 记录每次实验的命令、commit、日志路径和结论。

## P0：先建立可复现实验基线

- [x] 最高优先级：排查并修复 NS3 manual subchannel 链路，使 OpenCDA 指定的 `sc_start/sc_num` 真实落到 NS3 NR sidelink 发送行为。已完成 4 类 probe：非冲突成功、最高合法子信道 9 成功、同子信道冲突触发 PHY decode failure、越界子信道 10 被拒绝且无 RLC/CAM 发送。
- [x] 将 SGCP `offline_ns3_replay` 的资源分配口径对齐为配置默认 `potential_game`，并默认只发送 PPS 已分配 `sc_start/sc_num` 的 request。已完成 11 帧 fixed replay：110 scheduled request 全部 CAM/RLC 成功，44 条未调度需求被跳过。
- [x] 修复 NS3 暴露子信道窗口语义：OpenCDA/bridge 按 `targetSubchannels` 校验 `sc_start/sc_num`，超出范围的 request 在 CAM/RLC 创建前拒绝；manual scheduler 对确定越界命令 drop+pop，避免无效队头阻塞后续合法请求。
- [x] 定位 SGCP 当前实现入口、配置文件和运行命令。
- [x] 建立 `v2xp_cluster_carla` 数据集导出、导入能力，用离线数据替代 CARLA 在线运行。
- [x] 当前优先：确认论文现有结果对应的代码版本、配置、随机种子和日志路径。已新增 `reproducibility_manifest.md`；结论是 `main.tex` 旧主表缺原始日志/随机种子/代码版本，当前只能把已复现离线结果作为修订依据。
- [x] 记录项目运行环境：`conda activate opencda`。
- [x] 记录在线命令：`python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug`，NS3 模式追加 `--network`。
- [x] 新增数据导出入口：`python opencda.py -t v2xp_cluster_carla --dump`。
- [x] 运行一次数据导出，确认每个评估 CAV 每帧生成 `.pcd` 与 `.yaml`。
- [x] 确认导出目录包含 `data_protocol.yaml`，格式兼容 OPV2V 风格目录。
- [x] 新增离线帧加载器，可从 OPV2V 风格目录读取单帧为 OpenCOOD 输入字典。
- [x] 新增离线 OpenCOOD 单帧推理脚本入口：`python -m opencda.tools.offline_inference --dataset-root <root>`。
- [x] 运行离线 OpenCOOD 单帧推理，验证无需启动 CARLA 可完成模型推理。
- [x] 将离线帧加载结果进一步接入 SGCP cluster/resource scheduling 回放。
- [x] 明确离线 SGCP 回放接口设计：见 `offline_replay.md`。
- [x] 新增 `OfflineCavWorld/OfflineVehicleManager/OfflineV2XManager/OfflineLidarGrid` 轻量状态适配层。
- [x] 单帧调用 `CoalitionGame`，确认无需 CARLA 可输出 cluster head/member。
- [x] 单帧调用 cluster resource allocation，确认可输出 channel allocation。
- [x] 多帧回放并输出 cluster lifetime、reconfiguration 次数、平均 cluster size、孤立车辆数量。
- [x] 确认离线资源分配默认 `NaiveRA` 是否应替换为论文 SGCP 使用的 `PotentialGame/PCS/MWS`：配置默认值为 `potential_game`，`naive` 保留为 baseline/fallback。
- [x] 解决资源分配配置与代码默认值不一致：`networking_clustering.yaml`/`ConfigManager` 默认为 `potential_game`，但 `ClusteringScheduler` 原先实例化 `NaiveRA`。
- [x] 将离线回放结果接入 SGCP 约束感知评估。
- [x] 确认 CARLA、OpenCDA、NS3、OpenCOOD 的版本和环境依赖。
- [x] 修复在线 CARLA-NS3 时间流速不一致：`NetworkManager.time_slot` 与 `world.fixed_delta_seconds` 保持一致。
- [x] 新增离线 NS3 replay smoke test，验证不启动 CARLA 时 `sync_request/sync_ack` 与 transfer request 链路可用。
- [x] 确认论文现有结果对应的代码版本、配置、随机种子和日志路径。旧论文表格尚未找到原始日志；当前复现 manifest 固定 OpenCDA commit、数据集、命令、结果和 artifact 路径。
- [x] 建立一次最小可复现实验流程，输出 mAP、通信开销、运行时耗时。
- [x] 在 `log.md` 中记录每次运行的完整命令和日志路径。已将核心复现实验集中整理到 `reproducibility_manifest.md`，`log.md` 继续保留探索过程和每轮新增命令。

## P1：补充关键实验

- [x] 消融：完整 SGCP vs 无稳定窗口。已完成 `T_min_stab=0` 第一版离线 replay + late-fusion AP；当前 41 帧 dump 与默认结果一致，后续需补更动态场景。
- [x] 消融：完整 SGCP vs 无 coalition formation，仅距离/随机聚类。已完成 singleton-cluster 第一版参考；后续需补距离/随机固定簇并计入 prediction-level late-fusion 开销。
- [x] 消融：完整 SGCP vs 无 PPS，仅随机/greedy 调度。已完成 `random` 与 `mws` 第一版离线 late-fusion 结果，后续需复核 MWS 论文口径。
- [x] 消融：完整 SGCP vs 仅 early fusion。已记录 full 20-CAV early baseline 与 constrained early-only/all-head 口径。
- [x] 消融：完整 SGCP vs 仅 late fusion。已完成 full 20-CAV OpenCOOD late checkpoint 第一版参考，后续需设计严格同通信约束 late-only SGCP 口径。
- [x] 参数实验：`T_min^stab` = 100/300/500/700/1000 ms。已完成第一版离线 replay + inter-cluster late-fusion AP；当前 41 帧 dump 对该参数无敏感性，后续需补更动态场景支撑论文结论。
- [x] 参数实验：`N_max` = 2/3/4/5/6。已完成第一版离线 replay + inter-cluster late-fusion AP；后续需补更长序列/密度场景并计入检测框交换开销。
- [x] 参数实验：`rho_th` 多组阈值。已完成 `0.5/1.0/2.0/3.0/4.0` 第一版离线 replay + inter-cluster late-fusion AP；后续需补完整 `f(rho)` 标定曲线和跨场景泛化。
- [x] 密度扩展：不同 CAV 数量或不同背景车密度。已完成同一 dump 下 5/10/15/20 CAV 子集规模敏感性第一版；后续需重新导出真实不同交通密度/背景车密度场景。
- [x] 网络资源扩展：不同带宽或子信道数量。已完成 `num_channels=5/10/20` 与 `bandwidth_mhz=20/40/80` 第一版离线 replay + inter-cluster late-fusion AP；后续需复核带宽参数在 PPS 吞吐模型中的作用。

## P2：补充公平 baseline

- [x] 明确 FullPerception-RSU 设置，作为集中式参考或 upper reference。已新增 `baseline_fairness.md`，不将其作为同通信预算公平主对比。
- [x] 实现或整理 FullPerception-Decentralized 设置，只使用与 SGCP 相同的 V2V 信息。已完成 same-budget CAV-only selective-sharing first version。
- [x] 搜索并选择至少一个 V2V-only/decentralized collaborative perception baseline。已实现 `nearest` / `density` selective-sharing baseline。
- [x] 统一 backbone、场景、通信资源、评价指标和 late fusion 设置。当前 selective baseline 复用同一 dump、OpenCOOD early checkpoint、SGCP clustering 和 inter-cluster late fusion 评价口径，并匹配 grid budget。
- [x] 在 `results.md` 中单独记录 baseline 公平性说明。
- [x] 实现 same-budget CAV-only selective-sharing baseline，例如 nearest/top-k density/communication-aware top-k，匹配 SGCP 的 payload 或 source CAV 数。已完成 nearest/density/communication-aware grid-budget baseline；communication-aware baseline 为当前最强竞争 baseline。
- [x] 补充 communication-aware selective-sharing baseline，加入距离/链路质量/payload cost，而不只按 density 排序。当前 first version 使用 `density_sum / (1 + distance / 100)`；后续可替换为 NS3/link-quality cost。
- [x] 将 communication-aware selective-sharing baseline 的距离 proxy 替换或扩展为 NS3/link-quality cost。已新增 `--ns3-link-quality-csv`，可用 `rlc_by_request.csv` 的 `rlc_complete` 调整成员选择分数。
- [x] 使用 NS3 request-level trace 对 SGCP 离线上传请求做链路层统计。旧 all-member replay 为 CAM callback delivery ratio = 0.558442；修复后 potential_game scheduled replay 为 110/110 CAM delivery、RLC RX 2970/2970、PHY failures 0。
- [x] 继续补充 RLC request completion 口径：按 request_id 对比 TX/RX segment 数、DROP 事件和 application callback，给出 partial reception、complete application delivery、PHY diagnostics 三层指标。已新增 `rlc_complete_requests`、`rlc_partial_requests`、`rlc_no_rx_requests`，并通过 10/5 子信道回归验证。
- [x] 构造 NS3 受限暴露带宽回归：`targetSubchannels=5` 下 110 个 scheduled request 中 `sc_start=0..4` 的 55 个 complete，`sc_start=5..9` 的 55 个 no_tx/no_rx，`MANUAL_CMD_REJECT=55`。
- [x] 将 NS3 request-level delivery/PDR 接入 SGCP PPS 或 selective-sharing baseline 的 link-quality cost。已接入 selective-sharing baseline，并完成 11 帧 distance proxy vs NS3 RLC-complete aware 对照。

## P3：完善机制设计

- [x] 定义 topology change trigger，包括邻居变化、相对速度、链路质量或 utility 下降阈值。已新增 `topology_trigger.md` 机制规格；后续需接入代码统计。
- [x] 将 topology trigger 接入离线 replay，输出每帧 trigger type、是否触发 reconfiguration、vehicle-head change 的对应关系。已在 `opencda.tools.offline_replay` 中新增 summary 和 `--print-topology-events`。
- [x] 将 topology trigger gate 接入在线 `ClusteringV2XManager`，避免无事件时每周期重构 cluster。已完成默认关闭 first version；后续需真实 CARLA 回归。
- [x] 在真实 CARLA 在线仿真中打开 `enable_topology_trigger_gate`，回归 cluster trigger 日志、reconfiguration 次数和感知结果。已新增 `online_topology_gate_regression.md`；35 tick 在线回归正常结束，CP AP@0.3/0.5/0.7 = 0.84/0.82/0.69，触发 `initial=1`、`neighbor_set_change=1`、`head_member_unreachable=3`，未观察到 skip。
- [x] 设计 cluster 已满时的处理策略：保留、替换、等待、split/merge 或 leader-level late fusion 补偿。已新增 `cluster_capacity_policy.md`，当前主策略为硬 `N_max` + 保留/未满簇迁移/singleton fallback/inter-cluster late fusion 补偿。
- [x] 明确是否支持 cluster merge/split，并说明与 `N_max` 的关系。当前口径：不允许超过 `N_max` 的 merge；split/merge 通过 topology trigger + coalition reformation 间接发生。
- [x] 补充成员加入后的边际贡献重算流程。当前迭代会在后续车辆/后续轮次基于更新后的 coalition state 重算贡献，并用 `ita` 抑制振荡。
- [x] 在离线 replay 中统计满簇数量、因 `N_max` 跳过的候选 move 数和 singleton/small-cluster 补偿比例。默认 `N_max=4` 下 41 帧平均每帧 3.12 个满簇、99.15 次满簇候选跳过、singleton ratio 0、small-cluster ratio 0.187。
- [x] 明确 `f(rho)` 的标定协议和 detector/sensor-specific metadata 机制。已新增 `opencda.tools.sgcp_density_calibration` 与 `f_rho_calibration.md`，当前 41 帧 dump 中默认 `rho_th=2.0` 位于非零 density p90/p95 之间，并筛出约 7.18% 非零网格；后续需补跨场景/探测器泛化。
- [x] 重新检查 potential game exact potential 的成立条件。已新增 `potential_game_conditions.md`；当前代码应表述为 potential-guided constrained best-response scheduling，若论文继续使用 exact potential game，需要限定固定 cluster/candidate grids/hard feasibility constraints，并补显式势函数与 monotonicity 证明。
- [x] 估算 density/utility/control message 的控制开销，并决定是否纳入通信开销指标。已接入 `opencda.tools.offline_replay` 并新增 `control_overhead.md`；当前 41 帧默认 SGCP 控制面约 187,112 bytes，平均 4,563.71 bytes/frame，约为点云 payload 的 0.70%，论文中应单独报告而不是混入 perception payload。
- [x] 复核 PPS 中 `bandwidth_all/bandwidth_per_channel` 的吞吐约束实现，解释为何 `bandwidth_mhz=20/40/80` 在当前离线实验中结果一致。结论：参数已进入 `PotentialGame` max-grid/SINR/data-rate 计算，但当前 dump 不由带宽上限主导。
- [x] 构造能触发带宽瓶颈的 SGCP 场景或参数组：已完成 `bandwidth_mhz=0.1/0.5/1.0` stress test，selected grids 和 AP 随带宽恢复而上升。

## P4：论文写作修订

- [x] 重写 related work 的 decentralized CP 和 coalition game 对比。已新增 `related_work_novelty_revision.md`，给出 V2V-only CP、RSU-centric CP、learned communication selection、其他领域 coalition formation 的对比段落和 `main.tex` 插入建议。
- [x] 增强 novelty：突出感知效用驱动、稳定性约束、分层 fusion、分布式资源调度的组合贡献。已在 `related_work_novelty_revision.md` 中形成 introduction contribution 和 rebuttal 可用文本。
- [x] 补充 `f(rho)` 标定过程和曲线。已新增 `parameter_calibration_revision.md`，整合 density calibration 命令、788,020 个 CAV-grid 样本统计、`rho_th` sweep 和论文/rebuttal 写法。
- [x] 补充 `T_min^stab`、`N_max`、`rho_th` 参数选择依据。已新增 `parameter_calibration_revision.md`；明确 `rho_th=2.0` 是 AP/payload 折中，`N_max=4` 是容量/fragmentation 折中，`T_min^stab=500 ms` 只能写为保守默认而非当前证据下的最优值。
- [x] 补充 FullPerception baseline 的实现细节和公平性讨论。已新增 `fullperception_baseline_revision.md`，明确 FullPerception-RSU/full 20-CAV fusion 只能作为 centralized upper reference，公平主对比应使用 same-budget CAV-only selective-sharing baseline。
- [x] 补充实时性实验，包括毫秒级耗时分解。已扩展 `opencda.tools.offline_replay` 输出 `runtime_breakdown_ms`，完成 41 帧控制面 profiling，并新增 `runtime_feasibility_revision.md`；当前 Python 原型 SGCP algorithm avg 105.24 ms，需谨慎写为 near-real-time 而非完整端到端 100 ms 保证。
- [x] 修正 “topology change 才触发” 与 “每个周期重复” 的表述矛盾。已新增 `paper_revision_plan.md`，明确每周期更新 beacon/density/PPS，cluster membership 仅在 topology/stability trigger 或 periodic guard 触发时更新，并给出 `main.tex` 替换建议。
- [x] 将 coverage-aware / spatial-diverse grid selection 和 potential-guided constrained PPS 口径写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`，弱化无条件 exact-potential/Nash guarantee 表述，使机制章节与当前实现和主表结果一致。
- [x] 将 `f(rho)` / `rho_th` 标定统计和 `rho_th` sensitivity 表写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`：包括 788,020 个 CAV-grid 样本、非空网格 5.98%、非空密度 p90/p95 = 1.40/3.60、`rho_th=2.0` 选择 7.18% 非空网格，以及 coverage-aware 10ch `rho_th=1/2/3/4` AP/Mbps 表。
- [x] 将 `N_max` 和 `T_min^stab` 参数依据写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`：说明 `N_max=4` 是容量控制折中而非纯 AP 调参，`T_min^stab=500 ms` 是五个 10 Hz 感知周期的保守滞回默认，并记录当前 sweep 中 `T_min^stab=100--1000 ms` 对 AP/reconfiguration 不敏感。
