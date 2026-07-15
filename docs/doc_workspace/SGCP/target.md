# SGCP 任务清单

更新时间：2026-07-16
最终目标：解决所有C:\Workspace\icdcs-paper\SGCP中的审稿意见。

## P0：先建立可复现实验基线

- [x] 最高优先级：排查并修复 NS3 manual subchannel 链路，使 OpenCDA 指定的 `sc_start/sc_num` 真实落到 NS3 NR sidelink 发送行为。已完成 4 类 probe：非冲突成功、最高合法子信道 9 成功、同子信道冲突触发 PHY decode failure、越界子信道 10 被拒绝且无 RLC/CAM 发送。
- [x] 将 SGCP `offline_ns3_replay` 的资源分配口径对齐为配置默认 `potential_game`，并默认只发送 PPS 已分配 `sc_start/sc_num` 的 request。已完成 11 帧 fixed replay：110 scheduled request 全部 CAM/RLC 成功，44 条未调度需求被跳过。
- [x] 修复 NS3 暴露子信道窗口语义：OpenCDA/bridge 按 `targetSubchannels` 校验 `sc_start/sc_num`，超出范围的 request 在 CAM/RLC 创建前拒绝；manual scheduler 对确定越界命令 drop+pop，避免无效队头阻塞后续合法请求。
- [x] 定位 SGCP 当前实现入口、配置文件和运行命令。
- [x] 建立 `v2xp_cluster_carla` 数据集导出、导入能力，用离线数据替代 CARLA 在线运行。
- [ ] 当前优先：确认论文现有结果对应的代码版本、配置、随机种子和日志路径。
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
- [ ] 确认论文现有结果对应的代码版本、配置、随机种子和日志路径。
- [x] 建立一次最小可复现实验流程，输出 mAP、通信开销、运行时耗时。
- [ ] 在 `log.md` 中记录每次运行的完整命令和日志路径。

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
- [ ] 在真实 CARLA 在线仿真中打开 `enable_topology_trigger_gate`，回归 cluster trigger 日志、reconfiguration 次数和感知结果。
- [x] 设计 cluster 已满时的处理策略：保留、替换、等待、split/merge 或 leader-level late fusion 补偿。已新增 `cluster_capacity_policy.md`，当前主策略为硬 `N_max` + 保留/未满簇迁移/singleton fallback/inter-cluster late fusion 补偿。
- [x] 明确是否支持 cluster merge/split，并说明与 `N_max` 的关系。当前口径：不允许超过 `N_max` 的 merge；split/merge 通过 topology trigger + coalition reformation 间接发生。
- [x] 补充成员加入后的边际贡献重算流程。当前迭代会在后续车辆/后续轮次基于更新后的 coalition state 重算贡献，并用 `ita` 抑制振荡。
- [x] 在离线 replay 中统计满簇数量、因 `N_max` 跳过的候选 move 数和 singleton/small-cluster 补偿比例。默认 `N_max=4` 下 41 帧平均每帧 3.12 个满簇、99.15 次满簇候选跳过、singleton ratio 0、small-cluster ratio 0.187。
- [x] 明确 `f(rho)` 的标定协议和 detector/sensor-specific metadata 机制。已新增 `opencda.tools.sgcp_density_calibration` 与 `f_rho_calibration.md`，当前 41 帧 dump 中默认 `rho_th=2.0` 位于非零 density p90/p95 之间，并筛出约 7.18% 非零网格；后续需补跨场景/探测器泛化。
- [ ] 重新检查 potential game exact potential 的成立条件。
- [ ] 估算 density/utility/control message 的控制开销，并决定是否纳入通信开销指标。
- [x] 复核 PPS 中 `bandwidth_all/bandwidth_per_channel` 的吞吐约束实现，解释为何 `bandwidth_mhz=20/40/80` 在当前离线实验中结果一致。结论：参数已进入 `PotentialGame` max-grid/SINR/data-rate 计算，但当前 dump 不由带宽上限主导。
- [x] 构造能触发带宽瓶颈的 SGCP 场景或参数组：已完成 `bandwidth_mhz=0.1/0.5/1.0` stress test，selected grids 和 AP 随带宽恢复而上升。

## P4：论文写作修订

- [ ] 重写 related work 的 decentralized CP 和 coalition game 对比。
- [ ] 增强 novelty：突出感知效用驱动、稳定性约束、分层 fusion、分布式资源调度的组合贡献。
- [ ] 补充 `f(rho)` 标定过程和曲线。
- [ ] 补充 `T_min^stab`、`N_max`、`rho_th` 参数选择依据。
- [ ] 补充 FullPerception baseline 的实现细节和公平性讨论。
- [ ] 补充实时性实验，包括毫秒级耗时分解。
- [ ] 修正 “topology change 才触发” 与 “每个周期重复” 的表述矛盾。
