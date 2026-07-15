# SGCP 任务清单

更新时间：2026-07-15
最终目标：解决所有C:\Workspace\icdcs-paper\SGCP中的审稿意见。

## P0：先建立可复现实验基线

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

- [ ] 明确 FullPerception-RSU 设置，作为集中式参考或 upper reference。
- [ ] 实现或整理 FullPerception-Decentralized 设置，只使用与 SGCP 相同的 V2V 信息。
- [ ] 搜索并选择至少一个 V2V-only/decentralized collaborative perception baseline。
- [ ] 统一 backbone、场景、通信资源、评价指标和 late fusion 设置。
- [ ] 在 `results.md` 中单独记录 baseline 公平性说明。

## P3：完善机制设计

- [ ] 定义 topology change trigger，包括邻居变化、相对速度、链路质量或 utility 下降阈值。
- [ ] 设计 cluster 已满时的处理策略：保留、替换、等待、split/merge 或 leader-level late fusion 补偿。
- [ ] 明确是否支持 cluster merge/split，并说明与 `N_max` 的关系。
- [ ] 补充成员加入后的边际贡献重算流程。
- [ ] 明确 `f(rho)` 的标定协议和 detector/sensor-specific metadata 机制。
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
