# SGCP 当前状态

更新时间：2026-07-15

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
- topology change trigger 的定义模糊，可能影响动态场景稳定性。
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
- 已确认当前可运行环境版本快照，并记录到 `docs/doc_workspace/environment.md`：OpenCDA HEAD `fcc29fdc9ee9a9fe694c12e1fb6792b4d41bccac`，Python 3.7.10，CARLA Python API 0.9.11，PyTorch 1.10.0+cu113，Open3D 0.10.0.0，ns-3 wrapper `ns-3-dev-v2x-v1.1-dirty`。
- 已修复在线 CARLA-NS3 时间流速不一致问题：`NetworkManager.time_slot` 不再将 `world.fixed_delta_seconds` 除以 5，`current_sim_time`、`current_time_slot` 与 CARLA tick 统一推进。
- 已修复在线 NS3 初始化顺序：sender 线程等待车辆注册后先发送真实车辆数和第一帧 `vehicles_position`，再进入 `sync_request/sync_ack` 循环。
- 已新增 `opencda.tools.offline_ns3_replay`，可不启动 CARLA，直接从 dump 数据驱动 NS3 同步和传输请求。
- 已完成 3 帧离线 NS3 smoke test：帧时间 0.000/0.100/0.200 s，20 车、6 个 cluster、每帧 14 条 intra-cluster upload request，NS3 返回多条 `cam_received`。

## 当前阻塞项

- 尚未确认论文中表格结果对应的原始日志、随机种子和复现实验配置。
- 当前 OpenCDA 与 ns-3 工作区均存在未提交/dirty 状态；论文级复现需要保存 patch 或形成干净 commit/tag。
- 离线 SGCP 回放已完成到“多帧 clustering + `potential_game` 资源分配 + 稳定性/运行时指标 + OpenCOOD 约束感知 mAP”层。
- 论文 SGCP 的 PPS/博弈调度当前按配置默认 `potential_game` 执行；已完成 `RandomRA/MWS` 初版对比，但 MWS 结果低于 random，需要结合论文文本复核 baseline 定义与效用函数。
- 早期 `ego-cluster-head` 与 `all-cluster-heads` constrained 评估只包含 intra-cluster early fusion；论文完整 SGCP 结果应优先采用 inter-cluster late fusion 口径。
- full 20-CAV late fusion reference 使用独立 OpenCOOD late checkpoint，不能直接等同于严格同通信约束的 SGCP late-only 消融。
- 当前 `T_min_stab=0` 消融未显示差异，说明当前 41 帧 dump 不足以证明稳定窗口贡献；需要更长序列或更强相对运动/topology change 场景。
- Dump 中速度字段目前使用 `ego_speed`；后续如需更严格动态稳定性，应确认单位并评估是否改为相邻帧差分速度。
- 在线 CARLA-NS3 修复尚未在真实 CARLA 图形仿真中长时间回归；当前已通过离线 NS3 socket/sync smoke test 和本地时间基准单元断言。
