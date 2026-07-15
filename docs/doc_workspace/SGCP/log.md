# SGCP 实验日志

本文件按时间顺序追加实验记录。每条记录应尽量包含：目的、代码版本、配置、命令、日志路径、关键结果、异常现象和下一步。

## 记录模板

````markdown
## YYYY-MM-DD HH:mm - 实验标题

### 目的

- 

### 代码与环境

- OpenCDA commit：
- NS3 commit / binary：
- CARLA：
- Conda 环境：
- GPU/CPU：

### 配置

- 场景配置：
- CAV 数量：
- 通信参数：
- 感知参数：
- 随机种子：

### 命令

```powershell

```

### 日志路径

- OpenCDA：
- NS3：
- 输出目录：

### 结果摘要

- mAP@0.3：
- mAP@0.5：
- mAP@0.7：
- 通信开销：
- 聚类耗时：
- 调度耗时：
- 端到端周期耗时：

### 观察与异常

- 

### 下一步

- 
````

## 2026-07-15 - 文档工作区初始化

### 目的

- 为 SGCP 论文修订、实验复现和机制完善建立独立文档工作区。

### 已完成

- 阅读 `README.md`，确认 OpenCDA 是 CARLA/SUMO 协同驾驶仿真框架。
- 阅读 `AGENT_README.md`，确认与 SGCP 相关的主要模块包括 clustering、networking、application、scenario config 和 OpenCOOD。
- 新增 `readme.md`、`status.md`、`target.md`、`log.md`、`results.md`。

### 尚未执行

- 未运行 CARLA/OpenCDA/NS3 实验。
- 未修改 SGCP 相关代码。
- 未确认论文表格结果的原始日志。

### 下一步

- 定位 SGCP 实现和配置入口。
- 建立最小可复现实验命令。
- 将第一轮 baseline 运行结果记录到本文件和 `results.md`。

## 2026-07-15 - 记录 SGCP 命令并启动离线数据集能力

### 目的

- 明确 SGCP 在线仿真命令。
- 建立 OPV2V 风格的数据导出/导入基础能力，后续用离线数据替代 CARLA 在线运行。

### 命令

项目环境：

```powershell
conda activate opencda
```

在线 CARLA 仿真：

```powershell
python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug
```

启用 NS3 协同仿真：

```powershell
python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
```

SGCP 数据集导出：

```powershell
python opencda.py -t v2xp_cluster_carla --dump
```

### 已完成

- `DataDumper` 启用每帧 `.pcd` 点云保存。
- datadump 运行时在输出根目录保存 `data_protocol.yaml`。
- 新增 `v2xp_cluster_carla_datadump` 场景脚本和配置。
- 新增 `OPV2VFrameDataset`，可从 OPV2V 风格目录加载单帧为 OpenCOOD 输入字典。
- 新增 `opencda.tools.offline_inference`，用于从 OPV2V 风格目录直接执行 OpenCOOD 推理。
- 在 `conda run -n opencda` 环境下通过语法检查。
- 在 `conda run -n opencda` 环境下确认离线推理脚本 `--help` 可用。
- 使用 `E:\data\opv2v\test` 完成离线加载 smoke test：识别 16 个 scenario，第一帧 `2021_08_18_19_48_05/000068` 包含 CAV `[1045, 1054]`，ego 点云 shape 为 `(57349, 4)`。
- 使用 `E:\data\opv2v\test` 完成离线 OpenCOOD 推理 smoke test：加载 epoch 10000，`fusion_method=early`，输出 `pred_boxes=18`、`gt_boxes=19`、`pred_scores_shape=(18,)`。

### 下一步

- 实际运行数据导出命令，检查 `opencda/data_dumping/<current_time>/`。
- 将离线数据进一步接入 SGCP cluster/resource scheduling 回放。

## 2026-07-15 - 实际运行 SGCP 数据集导出

### 目的

- 在 `v2xp_cluster_carla` 配置下导出每个智能车辆/CAV manager 的逐帧点云数据，输出到 `D:\Data`。

### 环境与路径

- Conda 环境：`opencda`
- CARLA：`C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe`
- 数据集根目录：`D:\Data`

### 命令

```powershell
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe"
$env:OPENCDA_DATA_DUMP_ROOT = "D:\Data\Carla"
$env:OPENCDA_DATADUMP_TICKS = "140"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --dump
```

### 结果摘要

- 导出目录：`D:\Data\Carla\2026_07_15_01_26_56`
- CAV 目录数：20
- 每个 CAV 帧数：41 个 `.pcd` + 41 个 `.yaml`
- 根目录：包含 `data_protocol.yaml`
- 总文件数：820 个 `.pcd`，821 个 `.yaml`，164 个 `.png`

### 离线验证

读取第一帧：

```powershell
conda run -n opencda python -c "from opencda.core.common.offline_dataset import OPV2VFrameDataset; root=r'D:\Data\Carla'; sid='2026_07_15_01_26_56'; ds=OPV2VFrameDataset(root); ts=ds.scenarios[sid]['timestamps'][0]; frame=ds.load_frame(sid, ts, ego_cav_id='1'); print(ts, len(frame), frame[1]['lidar_np'].shape)"
```

结果：`000060` 帧包含 20 台 CAV，ego 点云 shape 为 `(4918, 4)`。

离线 OpenCOOD 推理：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1
```

结果：`fusion_method=early`，`pred_boxes=62`，`gt_boxes=71`，`pred_scores_shape=(62,)`。

### 观察与异常

- 第一次尝试使用 `--apply_cp` 导出会进入 clustering coperception 推理路径，但未启用 `--apply_ml` 时 `ml_manager=None`，已改为数据导出命令不使用 `--apply_cp`。
- 为覆盖 traffic CAV managers，已让 traffic CAV 在 `run_step()` 提前返回前执行 `DataDumper`。
- 新导出的 YAML 中包含 numpy scalar tag，离线 loader 已改为使用 `yaml.Loader` 兼容本地 OpenCDA dump。

### 下一步

- 将离线帧加载进一步接入 SGCP clustering/resource scheduling 回放，替代 CARLA 在线状态更新。

## 2026-07-15 - 无 NS3 离线读取数据集测试

### 目的

- 不启动 NS3，不依赖 CARLA 在线传感器流，直接读取刚导出的 `v2xp_cluster_carla` 数据集进行 OpenCOOD 推理测试。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0
```

### 数据

- 数据集：`D:\Data\Carla\2026_07_15_01_26_56`
- CAV 数量：20
- 测试帧数：41
- 帧范围：`000060` 到 `000140`
- 融合方式：OpenCOOD early fusion

### 结果

- `cp counter`: 41
- AP@0.3：0.85
- AP@0.5：0.83
- AP@0.7：0.48

### 观察

- 测试过程中没有启用 `--network`，因此未接入 NS3。
- 当前测试验证的是“离线数据集 -> OpenCOOD early fusion 推理/评估”链路；尚未模拟 SGCP 的 cluster/resource scheduling 和通信约束。

## 2026-07-15 - 通用化离线推理入口命名

### 目的

- 离线推理能力是通用 OPV2V/OpenCOOD 数据集能力，不应包含 SGCP 专属名称。
- 全局环境文档服务于所有研究路线，路径为 `docs/doc_workspace/environment.md`。

### 已完成

- 将 `opencda.tools.sgcp_offline_inference` 重命名为 `opencda.tools.offline_inference`。
- 更新 `docs/doc_workspace/environment.md`，标题从 SGCP 实验环境改为通用实验环境。
- 更新 SGCP 文档中的所有离线推理命令引用。

### 验证

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1
```

结果：`pred_boxes=62`，`gt_boxes=71`，`pred_scores_shape=(62,)`。

## 2026-07-15 - SGCP 离线回放接口梳理

### 目的

- 继续推进 `target.md` 中“将离线帧加载结果进一步接入 SGCP cluster/resource scheduling 回放”的 P0 任务。
- 明确离线数据需要补齐哪些在线对象字段，避免直接依赖 CARLA actor。

### 代码入口定位

- 聚类入口：`opencda.core.clustering.managers.clustering_v2x_manager.ClusteringV2XManager.run_algorithm()`
- coalition formation：`opencda.core.clustering.algorithms.clustering.coalition_game.CoalitionGame`
- 全局车辆状态构建：`opencda.core.clustering.utils.common.Vehicle_Grid.initialize()`
- cluster-based scheduler：`opencda.core.clustering.managers.clustering_scheduler.ClusteringScheduler`

### 结论

- 离线回放需要优先实现轻量 `OfflineCavWorld/OfflineVehicleManager/OfflineV2XManager/OfflineLidarGrid`。
- 这些对象只需满足 SGCP 当前读取的姿态、速度、方向、lidar grid 和 scheduler 字段。
- 已新增设计文档：`offline_replay.md`。

### 下一步

- 实现单帧 `OfflineCavWorld` 构建，并验证 `CoalitionGame.run()` 能在 `D:\Data\Carla\2026_07_15_01_26_56` 的 `000060` 帧输出 cluster 列表。

## 2026-07-15 - SGCP 离线单帧回放实现

### 目的

- 不启动 CARLA，不启动 NS3，直接从 `v2xp_cluster_carla` dump 数据重建 SGCP 所需在线状态。
- 验证单帧 `CoalitionGame` 和默认资源分配可运行。

### 已完成

- 新增 `opencda.core.common.offline_replay`：
  - `OfflineCavWorld`
  - `OfflineVehicleManager`
  - `OfflineV2XManager`
  - `OfflineLidarGrid`
  - 最小 `OfflineNetworkManager/OfflineScheduler/OfflineCoManager`
- 新增命令行入口：`opencda.tools.offline_replay`。

### 单帧验证命令

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1
```

### 单帧结果

- timestamp：`000060`
- CAV 数量：20
- cluster 数量：6
- 平均 cluster size：3.33
- `CoalitionGame + NaiveRA` 总耗时：约 52 ms
- 默认 `NaiveRA` channel allocation：380 条
- cluster：
  - head=11 members=[1, 2, 10, 11]
  - head=13 members=[9, 13, 14, 19]
  - head=16 members=[5, 7, 16, 20]
  - head=17 members=[3, 17, 18]
  - head=4 members=[4, 8, 12]
  - head=15 members=[6, 15]

### 3 帧 smoke test

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3
```

结果：3 帧均可输出 clustering 和 channel allocation；cluster 结构会随 timestamp 变化，说明逐帧状态重建生效。

### 下一步

- 汇总多帧指标：cluster lifetime、reconfiguration 次数、平均 cluster size、孤立车辆数量。
- 确认论文 SGCP 对应资源分配算法是否应从当前默认 `NaiveRA` 切换为 `PotentialGame/PCS/MWS`。

## 2026-07-15 - SGCP 离线多帧回放汇总

### 目的

- 在无需 CARLA/NS3 的情况下，对已导出的 `v2xp_cluster_carla` 数据集运行全量 SGCP clustering/resource allocation 回放。
- 输出稳定性与运行时指标，服务 rebuttal 中关于稳定性和实时性的补充实验。

### 代码更新

- `opencda.tools.offline_replay` 新增多帧汇总逻辑。
- 新增 `--summary-only`，用于全量数据集回放时只输出 aggregate metrics。
- 汇总指标包括：
  - 平均 cluster 数量
  - 平均 cluster size
  - 平均/最大孤立 CAV 数量
  - reconfiguration events
  - vehicle-head changes
  - cluster lifetime
  - 平均总耗时与平均资源分配耗时

### 验证命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_replay.py opencda\core\common\offline_replay.py
```

通过。

### 3 帧 smoke test

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --summary-only
```

结果：

- frames：3
- avg_clusters：6.00
- avg_cluster_size：3.33
- avg_isolated_cavs：0.00
- reconfiguration_events：1
- vehicle_head_changes：11
- avg_cluster_lifetime_frames：1.80
- avg_total_runtime：113.56 ms
- avg_resource_allocation_runtime：0.32 ms

### 全量 41 帧回放

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --summary-only
```

结果：

- frames：41
- avg_clusters：6.00
- avg_cluster_size：3.33
- avg_isolated_cavs：0.00
- max_isolated_cavs：0
- reconfiguration_events：11
- vehicle_head_changes：76
- avg_cluster_lifetime_frames：6.65
- min_cluster_lifetime_frames：1
- max_cluster_lifetime_frames：38
- avg_total_runtime：129.78 ms
- avg_resource_allocation_runtime：1.13 ms

### 观察

- 当前 41 帧中没有孤立 CAV，平均 cluster size 稳定为 3.33，符合 `N_max=4` 附近的聚类规模预期。
- 平均总耗时超过 100 ms，其中包含 Python 离线读取、PCD 解析和网格重建，不等同在线控制周期耗时；后续若用于论文实时性，应拆分 I/O 与算法耗时。
- 当前资源分配仍使用代码默认 `NaiveRA`，需要确认论文 SGCP 的 PPS/博弈调度对应实现。

### 资源分配算法线索

- `opencda/scenario_testing/config_yaml/networking_clustering.yaml` 中 `resource_allocation.algorithm` 为 `potential_game`。
- `opencda.core.common.config_manager.ResourceAllocationConfig.algorithm` 默认值也是 `potential_game`。
- 但 `opencda/core/clustering/managers/clustering_scheduler.py` 当前注释掉 `PotentialGame/PCS/MWS/RandomRA`，实际实例化 `NaiveRA`。
- 下一步应把离线 `offline_replay` 的资源分配算法做成参数，并确认在线 `ClusteringScheduler` 是否应按配置选择算法。

## 2026-07-15 - CARLA-NS3 时间同步修复与离线 NS3 smoke test

### 背景

- 此前在线联合仿真中 CARLA 与 NS3 时间流速不一致。
- 当前研究主线采用离线实验，因此优先保证 dump 数据驱动 NS3 的同步和传输链路可验证；在线 CARLA 回归优先级较低。

### 修复

- `opencda/core/networking/network_manager.py`
  - `NetworkManager.time_slot` 不再执行 `/ 5.0`，直接使用 `CavWorld` 注入的 `world.fixed_delta_seconds`。
  - `advance_time_slot()` 先归档当前 slot，再递增 `current_time_slot` 并更新 `current_sim_time`。
  - NS3 sender 线程等待车辆注册后，先发送真实车辆数和第一帧 `vehicles_position`，再进入 `sync_request/sync_ack` 循环。
- `opencda/core/networking/ns3_co_simulation/bridge/carla_ns3_bridge.py`
  - 停止 bridge 时不再把主动关闭 socket 产生的 listener 异常记录为错误。
- 新增 `opencda.tools.offline_ns3_replay`
  - 从 OPV2V dump 读取车辆位姿。
  - 重建 SGCP cluster。
  - 生成 cluster 内 member-to-head transfer requests。
  - 按帧间隔向 NS3 发送 `vehicles_position`、`sync_request`、`transfer_requests`。

### 验证

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\core\networking\network_manager.py opencda\tools\offline_ns3_replay.py test\test_network_time_sync.py
```

时间基准断言：

```powershell
conda run -n opencda python -c "from test.test_network_time_sync import test_network_time_slot_matches_carla_fixed_delta,test_multiple_network_slots_track_carla_time; test_network_time_slot_matches_carla_fixed_delta(); test_multiple_network_slots_track_carla_time(); print('network_time_sync tests passed')"
```

结果：`network_time_sync tests passed`。

离线 NS3 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --drain-seconds 0.3 --sync-timeout 10
```

结果：

- frame 1：timestamp `000060`，sim_time `0.000`，20 vehicles，6 clusters，14 requests
- frame 2：timestamp `000062`，sim_time `0.100`，20 vehicles，6 clusters，14 requests
- frame 3：timestamp `000064`，sim_time `0.200`，20 vehicles，6 clusters，14 requests
- final_sync_time：`0.500`
- NS3 日志出现多条 `cam_received`，说明 transfer requests 已触发 NR sidelink 接收回传。

### 仍需注意

- 离线 smoke test 已验证 socket 协议、时间同步和 NS3 收包；真实在线 CARLA 图形仿真仍需后续长时间回归。
- 当前离线请求仍使用默认 `NaiveRA` channel allocation，下一步要处理 `potential_game` 配置与代码默认不一致问题。

## 2026-07-15 - 资源分配默认值统一到 PotentialGame

### 目的

- 继续推进 `target.md` 中“确认论文 SGCP 对应资源分配算法”和“解决配置与代码默认不一致”任务。
- 让在线 `ClusteringScheduler` 与离线 `offline_replay` 都能按配置选择资源分配算法，而不是固定使用 `NaiveRA`。

### 代码更新

- 新增 `opencda.core.clustering.algorithms.resource_allocation.builder.build_resource_allocator()`。
- `ClusteringScheduler` 改为读取 `resource_allocation_algorithm` 或 `resource_allocation.algorithm`，默认 `potential_game`。
- `opencda.tools.offline_replay` 新增 `--resource-allocation`，支持 `potential_game/pcs/mws/random/naive`。
- `OfflineV2XManager` 补齐 `tx_power`、`noise_power`、`communication_range`、`ego_pos/ego_spd`，满足 `PotentialGame` 的物理层和位置接口需求。

### 验证命令

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\core\common\offline_replay.py opencda\tools\offline_replay.py opencda\core\clustering\algorithms\resource_allocation\builder.py opencda\core\clustering\managers\clustering_scheduler.py
```

单帧 `potential_game`：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --resource-allocation potential_game --summary-only
```

结果：

- frames：1
- avg_clusters：6.00
- avg_cluster_size：3.33
- avg_total_runtime：250.98 ms
- avg_resource_allocation_runtime：104.44 ms

全量 41 帧 `potential_game`：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

结果：

- frames：41
- avg_clusters：6.00
- avg_cluster_size：3.33
- avg_isolated_cavs：0.00
- max_isolated_cavs：0
- reconfiguration_events：11
- vehicle_head_changes：76
- avg_cluster_lifetime_frames：6.65
- min_cluster_lifetime_frames：1
- max_cluster_lifetime_frames：38
- avg_total_runtime：285.82 ms
- avg_resource_allocation_runtime：111.85 ms

全量 41 帧 `naive` baseline：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation naive --max-frames 0 --summary-only
```

结果：

- frames：41
- avg_clusters：6.00
- avg_cluster_size：3.33
- avg_isolated_cavs：0.00
- reconfiguration_events：11
- vehicle_head_changes：76
- avg_cluster_lifetime_frames：6.65
- avg_total_runtime：169.94 ms
- avg_resource_allocation_runtime：0.50 ms

### 观察

- `potential_game` 与 `naive` 在当前数据集上聚类稳定性指标相同，因为聚类由 `CoalitionGame` 决定；主要差异体现在资源分配耗时。
- `potential_game` 平均 RA 耗时约 111.85 ms，已经接近或超过 100 ms 周期预算；后续论文实时性部分需要拆分 I/O、聚类、资源分配、感知推理，并考虑优化或解释执行频率。
- 下一步应把 `PotentialGame` 产生的 grid selection/channel allocation 接入 OpenCOOD 输入裁剪与 SGCP 约束感知评估。

## 2026-07-15 - SGCP constrained OpenCOOD 评估接入

### 目的

- 推进 `target.md` 中“将离线回放结果接入 SGCP 约束感知评估”。
- 在不启动 CARLA/NS3 的情况下，将 `CoalitionGame + potential_game` 的 cluster 和 grid selection 转换为 OpenCOOD 可评估的受约束 frame。

### 代码更新

- `opencda.core.common.offline_replay`
  - 新增 `apply_cluster_state()`，把 `CoalitionGame` 输出的 head/member 写回离线 V2X manager。
  - 新增 `select_sgcp_receiver_id()`，支持 `ego` 与 `ego-cluster-head` receiver policy。
  - 新增 `build_constrained_frame()`，按在线 `CoperceptionManager.get_data_from_lidar()` 语义构造受约束 OpenCOOD frame：receiver 保留全点云，sender 只上传 `grid_selection` 中的网格点云。
- `opencda.tools.offline_inference`
  - 新增 `--sgcp-constrained`。
  - 新增 `--resource-allocation`，默认 `potential_game`。
  - 新增 `--sgcp-receiver-policy`，默认 `ego-cluster-head`。
  - 多帧评估时输出 `sgcp_summary`：平均上传字节数、总上传字节数、平均 source CAV 数。

### 验证

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\core\common\offline_replay.py opencda\tools\offline_inference.py opencda\tools\offline_replay.py
```

单帧 constrained inference：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game
```

结果：

- receiver：11
- sources：`[11, 10, 2]`
- clusters：6
- upload：123,200 bytes
- selected grids：`{10: 53, 2: 44}`
- pred boxes：20
- GT boxes：51

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --max-frames 3
```

结果：

- AP@0.3：0.46
- AP@0.5：0.46
- AP@0.7：0.29
- avg_comm_bytes：111,333.33
- total_comm_bytes：334,000
- avg_source_cavs：3.00

全量 41 帧 constrained inference：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --max-frames 0
```

结果：

- AP@0.3：0.35
- AP@0.5：0.35
- AP@0.7：0.21
- avg_comm_bytes：106,790.63
- total_comm_bytes：4,378,416
- avg_source_cavs：2.98

### 观察与注意

- 当前 constrained 评估默认使用 `ego-cluster-head`：当 `ego_cav_id=1` 不是 cluster head 时，评估对象切换为其所在 cluster 的 head。这与此前离线 early fusion baseline “固定 ego=1 + 全 20 CAV 点云”不是同一评价口径。
- 若按 0.1 s 帧间隔估算，平均通信速率约为 8.54 Mbps；该数值尚未包含协议头、控制包和 NS3 重传。
- 当前只实现 intra-cluster grid-constrained early fusion；inter-cluster late fusion 尚未纳入，因此结果不能直接等同论文完整 SGCP。

## 2026-07-15 - 环境与版本快照确认

### 目的

- 推进 `target.md` 中“确认 CARLA、OpenCDA、NS3、OpenCOOD 的版本和环境依赖”。
- 为后续论文结果复现提供当前可运行环境基线。

### 命令

```powershell
git rev-parse HEAD
git status --short
conda run -n opencda python --version
conda list -n opencda | Select-String -Pattern "^(python|carla|torch|torchvision|numpy|pyyaml|omegaconf|open3d|opencv|scikit-learn|spconv)\\s"
Get-Item "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe" | Select-Object FullName,Length,LastWriteTime,@{Name='FileVersion';Expression={$_.VersionInfo.FileVersion}},@{Name='ProductVersion';Expression={$_.VersionInfo.ProductVersion}}
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation && git rev-parse HEAD && git status --short"
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && git rev-parse HEAD && git describe --tags --always --dirty && ./ns3 show version"
```

### 结果摘要

- OpenCDA HEAD：`fcc29fdc9ee9a9fe694c12e1fb6792b4d41bccac`
- OpenCOOD：本仓库 `opencood/` 子目录，随 OpenCDA HEAD 固定。
- Conda 环境：`opencda`
- Python：`3.7.10`
- pip：`21.1.2`
- CARLA Python API：`0.9.11`
- CARLA exe：`C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe`
- CARLA exe 修改时间：`2026-07-14 23:37:41`
- CARLA exe 文件大小：`188,928 bytes`
- PyTorch：`1.10.0+cu113`
- torchvision：`0.11.1+cu113`
- NumPy：`1.21.6`
- Open3D：`0.10.0.0`
- OmegaConf：`2.3.0`
- PyYAML：`6.0.1`
- scikit-learn：`0.24.2`
- spconv：`spconv-cu113 2.3.6`
- OpenCV：`opencv-python 4.5.2.52`
- co-simulation 仓库 HEAD：`10ab54cee04b04bce7f638249ddae1619fb11bf1`
- `ns-3-dev` HEAD：`c90c13b8310a813cf4eaf67a2c90df497bbd1965`
- ns-3 wrapper version：`ns-3-dev-v2x-v1.1-dirty`

### 观察与异常

- OpenCDA 工作区存在未提交改动和新增文件；当前实验结果应绑定“HEAD + 当前工作区 patch”。
- `ns-3-dev` 处于 dirty 状态，包含若干 `src/lte/model/*.cc` type-change 标记和 `NrDlMacStats.txt`、`NrUlMacStats.txt` 生成文件。
- Windows `CarlaUE4.exe` 文件属性未提供 `FileVersion/ProductVersion`；当前只能以 CARLA Python API `0.9.11` 和程序路径/文件时间作为版本线索。

### 下一步

- 确认论文现有表格结果对应的原始日志、随机种子、配置文件和代码状态。
- 若要进入论文/rebuttal，建议将当前 OpenCDA patch 和 ns-3 patch 固化为 commit/tag 或导出 patch 文件。

## 2026-07-15 - SGCP all-cluster-heads 约束感知评估

### 目的

- 将 SGCP constrained OpenCOOD 评估从单个 `ego-cluster-head` 扩展到每帧所有 cluster head，获得更适合论文统计的全局簇头平均口径。

### 代码更新

- `opencda.tools.offline_inference`
  - `--sgcp-receiver-policy` 新增 `all-cluster-heads`。
  - 每个 timestamp 会为所有 `CoalitionGame` cluster head 构造 constrained frame 并逐个提交 AP 统计。
  - 输出 `receiver_sample=i/n`，区分同一帧内多个簇头样本。

### 验证

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_inference.py opencda\core\common\offline_replay.py
```

单帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-receiver-policy all-cluster-heads
```

结果：`000060` 帧输出 6 个 receiver sample，对应 6 个 cluster head。

3 帧 all-head 小实验：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-receiver-policy all-cluster-heads --max-frames 3
```

结果：

- samples：18
- AP@0.3：0.38
- AP@0.5：0.37
- AP@0.7：0.18
- avg_comm_bytes：93,939.56
- total_comm_bytes：1,690,912
- avg_source_cavs：2.67

全量 41 帧 all-head 实验：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-receiver-policy all-cluster-heads --max-frames 0
```

结果：

- frames：41
- samples：246
- AP@0.3：0.36
- AP@0.5：0.34
- AP@0.7：0.17
- avg_comm_bytes：109,415.48
- total_comm_bytes：26,916,208
- avg_source_cavs：2.67

### 观察

- `all-cluster-heads` 口径比 `ego-cluster-head` 更接近全局 SGCP 评估，但仍只包含 intra-cluster grid-constrained early fusion。
- 当前未加入 inter-cluster late fusion，因此 AP 低于全 20 CAV early fusion baseline 是预期现象。
- 后续可直接用同一入口跑 `--resource-allocation random/mws/pcs`，形成 “w/o PPS / greedy / random” 对比。

## 2026-07-15 - SGCP inter-cluster late fusion 离线评估

### 目的

- 修正此前 constrained 评估漏掉 inter-cluster late fusion 的问题。
- 对齐仓库中 `ClusteringPerceptionManager.submit_cp_results()` 的 simple late fusion/NMS 机制：所有簇头先完成簇内 constrained early fusion，再将预测框统一到 ego pose 后做跨簇晚期融合。

### 代码更新

- `opencda.tools.offline_inference`
  - 新增 `--sgcp-inter-cluster-late-fusion`。
  - 该模式会强制使用所有 cluster head 作为 late-fusion source。
  - 每个 cluster head 的 constrained frame 统一传入 `ego_cav_id` 的 `lidar_pose`，保证预测框坐标系一致。
  - 使用 `OpenCOODManager.naive_late_fusion()` 对预测框和 GT 框做 NMS 合并，并每帧提交一次 AP。

### 验证

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_inference.py
```

单帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion
```

结果：`000060` 帧融合 6 个 cluster head，late-fusion 后 `fused_pred_boxes=51`、`fused_gt_boxes=69`。

3 帧实验：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --max-frames 3
```

结果：

- AP@0.3：0.66
- AP@0.5：0.63
- AP@0.7：0.26
- avg_comm_bytes/source：93,939.56
- total_comm_bytes：1,690,912
- avg_source_cavs/source：2.67

全量 41 帧实验：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --max-frames 0
```

结果：

- frames：41
- cluster-head sources/frame：6
- AP@0.3：0.77
- AP@0.5：0.73
- AP@0.7：0.35
- avg_comm_bytes/source：109,415.48
- total_comm_bytes：26,916,208
- avg_source_cavs/source：2.67

### 观察

- 加入 inter-cluster late fusion 后，AP 从 head-wise/intra-cluster-only 的 0.36/0.34/0.17 提升到 0.77/0.73/0.35，接近 full early fusion baseline 的 0.85/0.83/0.48。
- 这说明此前低结果主要来自评估链路缺少跨簇晚期融合，而不是 SGCP 机制本身失效。
- 当前仍未接入 NS3 真实传输成功率/时延；通信开销为根据 grid-selected point cloud 统计的 payload bytes。

## 2026-07-15 - w/o PPS random/MWS 调度消融

### 目的

- 推进 P1 “完整 SGCP vs 无 PPS，仅随机/greedy 调度” 消融。
- 使用已修正的 SGCP inter-cluster late fusion 口径，对比 `potential_game`、`random`、`mws` 三种资源分配算法。

### 代码修复

- `opencda.core.clustering.algorithms.resource_allocation.pcs`
  - 补齐抽象接口 `run()`，使 `PCS/MWS/RandomRA` 可通过统一 builder 实例化和执行。
  - 显式保存 `self.cav_world`，供策略写回阶段使用。
  - 显式导入 `common` 与 `calculate_distance`，修复离线入口下的 NameError。
- `opencda.core.clustering.algorithms.resource_allocation.mws`
  - 显式导入 `common`，修复离线入口下的 NameError。

### 验证

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\core\clustering\algorithms\resource_allocation\pcs.py opencda\core\clustering\algorithms\resource_allocation\mws.py opencda\core\clustering\algorithms\resource_allocation\random_ra.py
```

RandomRA 单帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --sgcp-constrained --resource-allocation random --sgcp-inter-cluster-late-fusion
```

结果：`000060` 帧融合 6 个 cluster head，late-fusion 后 `fused_pred_boxes=36`、`fused_gt_boxes=57`。

MWS 单帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --sgcp-constrained --resource-allocation mws --sgcp-inter-cluster-late-fusion
```

结果：`000060` 帧融合 6 个 cluster head，late-fusion 后 `fused_pred_boxes=37`、`fused_gt_boxes=54`。

RandomRA 全量 41 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation random --sgcp-inter-cluster-late-fusion --max-frames 0
```

结果：

- frames：41
- cluster-head sources/frame：6
- AP@0.3：0.44
- AP@0.5：0.39
- AP@0.7：0.17
- avg_comm_bytes/source：39,534.05
- total_comm_bytes：9,725,376
- avg_source_cavs/source：1.51

MWS 全量 41 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation mws --sgcp-inter-cluster-late-fusion --max-frames 0
```

结果：

- frames：41
- cluster-head sources/frame：6
- AP@0.3：0.31
- AP@0.5：0.26
- AP@0.7：0.11
- avg_comm_bytes/source：40,284.68
- total_comm_bytes：9,910,032
- avg_source_cavs/source：1.50

### 观察

- `potential_game` 在同一 late-fusion 口径下为 0.77/0.73/0.35，总 payload 26,916,208 bytes。
- `random` 与 `mws` 的总 payload 约 9.7-9.9 MB，仅为 `potential_game` 的约 36%-37%，但 AP 明显下降，初步支持 PPS/博弈调度带来感知收益。
- 当前 `mws` 低于 `random`，提示 MWS baseline 的效用函数、链路生成阈值或论文 baseline 对应关系需要进一步复核，暂不应直接作为最终论文结论。

## 2026-07-15 - late-only OpenCOOD baseline

### 目的

- 推进 P1 “完整 SGCP vs 仅 late fusion” 消融的第一版参考结果。
- 先验证现有 OpenCOOD late fusion checkpoint 能否在导出的 `v2xp_cluster_carla` 数据上离线评估。

### 代码修复

- `opencood.tools.inference_utils.inference_late_fusion`
  - 修复 late fusion 推理函数构造结果后未 `return` 的问题。
  - 对 late dataset 兼容不带 `return_object_ids` 的 `post_process()` 签名。
- `opencda.tools.offline_inference`
  - 当 `fusion_method == 'late'` 时不请求 `return_object_ids`，避免 late dataset 签名不兼容。

### 验证

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_inference.py opencood\opencood\tools\inference_utils.py
```

单帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --fusion-method late
```

结果：`000060` 帧 full 20-CAV late fusion 输出 `pred_boxes=70`、`gt_boxes=71`。

全量 41 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --fusion-method late --max-frames 0
```

结果：

- frames：41
- CAVs/frame：20
- AP@0.3：0.91
- AP@0.5：0.85
- AP@0.7：0.51

### 观察

- late-only full 20-CAV checkpoint 高于 full early fusion baseline 的 0.85/0.83/0.48，也高于当前 SGCP constrained late-fusion 的 0.77/0.73/0.35。
- 该结果使用 OpenCOOD late checkpoint，并非“同一 checkpoint 只切换融合机制”的严格 SGCP 消融；进入论文表格前应标注为 full late fusion reference，或重新设计同等通信约束下的 late-only SGCP 口径。

## 2026-07-15 - w/o stability window 消融

### 目的

- 推进 P1 “完整 SGCP vs 无稳定窗口” 消融。
- 增加离线入口参数，允许覆盖 `CoalitionGame.Params.T_min_stab`；用 `--t-min-stab 0` 表示不使用预测稳定窗口。

### 代码更新

- `opencda.tools.offline_replay`
  - 新增 `--t-min-stab`，用于离线 clustering/replay 稳定性指标实验。
- `opencda.tools.offline_inference`
  - 新增 `--t-min-stab`，用于 SGCP constrained + inter-cluster late fusion AP 评估。

### 验证

语法检查：

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_inference.py opencda\tools\offline_replay.py
```

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --t-min-stab 0 --max-frames 3 --summary-only
```

结果：

- frames：3
- avg_clusters：6.00
- avg_cluster_size：3.33
- reconfiguration_events：1
- vehicle_head_changes：11
- avg_cluster_lifetime_frames：1.80
- avg_total_runtime：89.75 ms
- avg_ra_runtime：35.99 ms

41 帧 replay 汇总：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --t-min-stab 0 --max-frames 0 --summary-only
```

结果：

- frames：41
- avg_clusters：6.00
- avg_cluster_size：3.33
- avg_isolated_cavs：0.00
- reconfiguration_events：11
- vehicle_head_changes：76
- avg_cluster_lifetime_frames：6.65
- min/max_cluster_lifetime_frames：1 / 38
- avg_total_runtime：99.99 ms
- avg_ra_runtime：37.39 ms

41 帧 AP 评估：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --t-min-stab 0 --max-frames 0
```

结果：

- frames：41
- cluster-head sources/frame：6
- AP@0.3：0.77
- AP@0.5：0.73
- AP@0.7：0.35
- avg_comm_bytes/source：109,415.48
- total_comm_bytes：26,916,208
- avg_source_cavs/source：2.67

### 观察

- 当前 41 帧 dump 上，`T_min_stab=0` 与默认 `T_min_stab=1.0` 的 cluster/reconfiguration/mAP/communication 指标完全一致。
- 这说明该短片段和当前速度/轨迹条件不足以体现稳定窗口收益；论文中如需支撑稳定窗口，应补更长序列、更高相对速度或更频繁 topology change 的场景。
- `T_min_stab=0` 的离线运行时更低，但当前耗时数据受 Python/日志/机器负载影响，仅作为工程参考。

## 2026-07-15 - w/o coalition formation singleton 消融

### 目的

- 推进 P1 “完整 SGCP vs 无 coalition formation，仅距离/随机聚类” 消融。
- 先建立最简单无 coalition 参考：每辆 CAV 单独成簇，所有 singleton cluster head 的检测结果执行 inter-cluster late fusion。

### 代码更新

- `opencda.core.clustering.algorithms.clustering.naive_cluster`
  - 补齐显式 `common/Cluster` 导入。
  - 保存 `self.cav_world`，适配离线 replay。
  - 新增 `run()`，使其满足 `ClusteringAlgorithm` 抽象接口。
- `opencda.tools.offline_replay`
  - 新增 `--clustering coalition_game|singleton|all_in_one`。
- `opencda.tools.offline_inference`
  - 新增 `--clustering coalition_game|singleton|all_in_one`。
  - 对 singleton late-fusion source 中的空 pillar 输入做明确跳过，避免单车空点云导致 PointPillar scatter 崩溃。

### 验证

语法检查：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; conda run -n opencda python -m py_compile opencda\tools\offline_inference.py opencda\tools\offline_replay.py opencda\core\clustering\algorithms\clustering\naive_cluster.py
```

3 帧 replay smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --clustering singleton --resource-allocation potential_game --max-frames 3 --summary-only
```

结果：

- frames：3
- avg_clusters：20.00
- avg_cluster_size：1.00
- avg_isolated_cavs：20.00
- reconfiguration_events：0
- vehicle_head_changes：0
- avg_cluster_lifetime_frames：3.00

41 帧 replay 汇总：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --clustering singleton --resource-allocation potential_game --max-frames 0 --summary-only
```

结果：

- frames：41
- avg_clusters：20.00
- avg_cluster_size：1.00
- avg_isolated_cavs：20.00
- reconfiguration_events：0
- vehicle_head_changes：0
- avg_cluster_lifetime_frames：41.00
- avg_total_runtime：4.52 ms
- avg_ra_runtime：3.92 ms

41 帧 AP 评估：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --clustering singleton --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --max-frames 0
```

结果：

- frames：41
- singleton sources/frame：20
- AP@0.3：0.82
- AP@0.5：0.76
- AP@0.7：0.37
- avg_comm_bytes/source：0.00
- total_comm_bytes：0
- avg_source_cavs/source：1.00

### 观察

- singleton baseline AP 高于当前 SGCP full 口径的 0.77/0.73/0.35，原因是它 late-fuse 了全部 20 个 CAV 的单车检测结果。
- 当前通信统计只计算 intra-cluster point-cloud upload payload；singleton 没有点云上传，所以显示为 0，但 prediction-level late-fusion box/score 交换开销尚未计入。
- 因此该结果应暂记为 “singleton-cluster full late-fusion reference”，不能直接声称为零通信的公平 baseline。后续要么计入检测框交换开销，要么实现距离/随机固定簇 baseline 与 SGCP 使用相同的 cluster-head exchange 口径。

## 2026-07-15 - `N_max` 参数敏感性实验

### 目的

- 推进 P1 参数实验：`N_max = 2/3/4/5/6`。
- 检查最大簇大小约束对 cluster fragmentation、reconfiguration、communication payload 和 AP 的影响。

### 代码更新

- `opencda.tools.offline_replay`
  - 新增 `--n-max`，可覆盖 `CoalitionGame.Params.N_max`。
- `opencda.tools.offline_inference`
  - 新增 `--n-max`，SGCP constrained / inter-cluster late-fusion 评估可使用同一参数。

### 验证

语法检查：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; conda run -n opencda python -m py_compile opencda\tools\offline_inference.py opencda\tools\offline_replay.py
```

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --n-max 2 --max-frames 3 --summary-only
```

结果：

- frames：3
- avg_clusters：11.00
- avg_cluster_size：1.82
- avg_isolated_cavs：2.00
- reconfiguration_events：0
- vehicle_head_changes：0
- avg_total_runtime：53.27 ms
- avg_ra_runtime：15.23 ms

### 41 帧 replay 汇总

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --n-max <N> --max-frames 0 --summary-only
```

| `N_max` | Avg. Clusters | Avg. Cluster Size | Avg. Isolated CAVs | Reconfig. Events | Vehicle-Head Changes | Avg. Cluster Lifetime (frames) | Avg. Runtime (ms) | Avg. RA Runtime (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10.29 | 1.95 | 0.59 | 16 | 59 | 7.28 | 54.39 | 22.51 |
| 3 | 7.59 | 2.65 | 1.17 | 9 | 62 | 7.59 | 87.94 | 38.31 |
| 4 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 285.82 | 111.85 |
| 5 | 6.00 | 3.33 | 0.00 | 8 | 15 | 10.70 | 110.09 | 38.20 |
| 6 | 6.00 | 3.33 | 0.00 | 8 | 15 | 10.70 | 112.02 | 38.72 |

### 41 帧 AP 评估

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --n-max <N> --max-frames 0
```

| `N_max` | Frames | Cluster-Head Sources | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 41 | 422 | 0.79 | 0.74 | 0.37 | 62198.64 | 26247824 | 1.94 |
| 3 | 41 | 311 | 0.75 | 0.71 | 0.34 | 82226.47 | 25572432 | 2.32 |
| 4 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| 5 | 41 | 246 | 0.75 | 0.71 | 0.32 | 102582.76 | 25235360 | 2.67 |
| 6 | 41 | 246 | 0.75 | 0.71 | 0.32 | 102582.76 | 25235360 | 2.67 |

### 观察

- `N_max=2` 在当前 dump 中 AP 最高，但它产生更多 cluster head source，属于更强 inter-cluster late fusion 覆盖，不能简单解释为“更小簇一定更好”。
- `N_max=4` 接近论文默认候选，AP 与通信开销处于中间位置；`N_max=5/6` 的聚类结构和 AP 完全一致，说明当前 20-CAV 片段中有效簇大小没有继续增大。
- `N_max=3` 反而低于 2/4，提示 coalition search 路径、head 选择和当前 detector 输出之间存在非单调关系；论文写作中应避免把参数敏感性描述成单调趋势。
- 当前 communication payload 只统计 intra-cluster 点云 upload；inter-cluster late-fusion 的检测框交换开销仍需补计。

## 2026-07-15 - `T_min^stab` 参数敏感性实验

### 目的

- 推进 P1 参数实验：`T_min^stab = 100/300/500/700/1000 ms`。
- 检查稳定时间窗口对 cluster reconfiguration、vehicle-head changes、cluster lifetime 和 AP 的影响。

### 单位说明

- 代码参数 `--t-min-stab` 的单位是秒。
- 本组实验命令分别使用 `0.1/0.3/0.5/0.7/1.0`，对应论文表述中的 `100/300/500/700/1000 ms`。

### 41 帧 replay 汇总

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --t-min-stab <seconds> --max-frames 0 --summary-only
```

| `T_min^stab` (ms) | Avg. Clusters | Avg. Cluster Size | Avg. Isolated CAVs | Reconfig. Events | Vehicle-Head Changes | Avg. Cluster Lifetime (frames) | Avg. Runtime (ms) | Avg. RA Runtime (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 97.36 | 36.81 |
| 300 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.59 | 40.05 |
| 500 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.08 | 37.80 |
| 700 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.12 | 40.02 |
| 1000 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.99 | 37.39 |

### 41 帧 AP 评估

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --t-min-stab <seconds> --max-frames 0
```

| `T_min^stab` (ms) | Frames | Cluster-Head Sources | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| 300 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| 500 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| 700 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| 1000 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |

### 观察

- 当前 41 帧 dump 中，`T_min^stab` 从 100 ms 到 1000 ms 的 replay 和 AP 指标完全一致。
- 这进一步确认当前短序列不足以支撑稳定窗口参数选择。论文若要回应审稿意见，需要补更长序列、更高相对速度或更频繁 topology change 的场景。
- 运行时差异处于 Python 执行和机器负载噪声范围内，不宜作为论文结论。

## 2026-07-15 - `rho_th` 参数敏感性实验

### 目的

- 推进 P1 参数实验：`rho_th` 多组阈值。
- 验证点云密度阈值对 PPS grid selection、通信开销和 inter-cluster late-fusion AP 的影响。

### 代码更新

- `opencda.core.common.offline_replay.OfflineCavWorld`
  - 新增 `density_threshold` 覆盖入口，在构建 `OfflineLidarGrid` 前覆盖 lidar config。
- `opencda.tools.offline_replay`
  - 新增 `--rho-th`，覆盖离线 replay 中的 lidar `density_threshold` / `Vehicle_Grid.rho_th`。
- `opencda.tools.offline_inference`
  - 新增 `--rho-th`，SGCP constrained / inter-cluster late-fusion AP 评估使用同一阈值。

### 验证

语法检查：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; conda run -n opencda python -m py_compile opencda\core\common\offline_replay.py opencda\tools\offline_replay.py opencda\tools\offline_inference.py
```

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --rho-th 1.0 --max-frames 3 --summary-only
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --rho-th 4.0 --max-frames 3 --summary-only
```

两组均可完成 3 帧 replay。

### 41 帧 replay 汇总

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --rho-th <rho> --max-frames 0 --summary-only
```

| `rho_th` | Avg. Clusters | Avg. Cluster Size | Avg. Isolated CAVs | Reconfig. Events | Vehicle-Head Changes | Avg. Cluster Lifetime (frames) | Avg. Runtime (ms) | Avg. RA Runtime (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 6.12 | 3.28 | 0.00 | 10 | 60 | 7.61 | 97.74 | 33.10 |
| 1.0 | 6.00 | 3.33 | 0.00 | 9 | 64 | 7.45 | 96.22 | 35.24 |
| 2.0 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.99 | 37.39 |
| 3.0 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 98.87 | 38.51 |
| 4.0 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 103.24 | 40.26 |

### 41 帧 AP 评估

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --rho-th <rho> --max-frames 0
```

| `rho_th` | Frames | Cluster-Head Sources | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 41 | 251 | 0.74 | 0.69 | 0.34 | 86658.74 | 21751344 | 3.27 |
| 1.0 | 41 | 246 | 0.75 | 0.71 | 0.33 | 96968.13 | 23854160 | 2.67 |
| 2.0 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |
| 3.0 | 41 | 246 | 0.77 | 0.73 | 0.37 | 113689.69 | 27967664 | 2.67 |
| 4.0 | 41 | 246 | 0.77 | 0.74 | 0.37 | 115754.73 | 28475664 | 2.67 |

### 观察

- 低阈值 `rho_th=0.5/1.0` 明显降低点云 payload，但 AP 也下降。
- 默认 `rho_th=2.0` 是当前通信-精度折中点；`rho_th=3.0/4.0` 能提升 AP@0.7，但需要更多上传点云。
- 当前结果可以支撑“阈值影响通信-精度折中”的实验描述，但还不能替代完整 `f(rho)` 标定曲线；论文仍需补密度采样、拟合曲线和 detector/scene 泛化。

## 2026-07-15 - CAV 数量规模敏感性实验

### 目的

- 推进 P1 “密度扩展：不同 CAV 数量或不同背景车密度”。
- 在无需重新启动 CARLA 的前提下，先用同一 20-CAV dump 的 CAV 子集验证 SGCP 离线链路对不同协同车辆数量的敏感性。

### 代码更新

- `opencda.tools.offline_replay`
  - 新增 `--cav-count`：按数值顺序选择前 N 个 CAV，并确保指定 ego 在子集中。
  - 新增 `--cav-ids`：手动指定 CAV id 列表，例如 `1,2,3`。
- `opencda.tools.offline_inference`
  - 新增同样的 `--cav-count` / `--cav-ids`，用于 OpenCOOD AP 评估。

### 边界说明

- 本实验固定使用 `D:\Data\Carla\2026_07_15_01_26_56`，只改变参与协同的 CAV 子集。
- 它不是重新生成的不同背景车密度或交通密度场景，不能直接替代论文中“不同车流密度”的完整实验。

### 验证

语法检查：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; conda run -n opencda python -m py_compile opencda\tools\offline_replay.py opencda\tools\offline_inference.py
```

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --cav-count 5 --max-frames 3 --summary-only
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --cav-count 10 --max-frames 3 --summary-only
```

两组均可完成 replay。

### 41 帧 replay 汇总

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --cav-count <N> --max-frames 0 --summary-only
```

| CAV Count | Avg. Clusters | Avg. Cluster Size | Avg. Isolated CAVs | Reconfig. Events | Vehicle-Head Changes | Avg. Cluster Lifetime (frames) | Avg. Runtime (ms) | Avg. RA Runtime (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 2.00 | 2.50 | 0.37 | 6 | 24 | 5.86 | 9.91 | 4.34 |
| 10 | 3.00 | 3.33 | 0.00 | 3 | 14 | 11.18 | 37.47 | 17.96 |
| 15 | 5.00 | 3.00 | 0.20 | 18 | 71 | 3.47 | 68.66 | 29.76 |
| 20 | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.99 | 37.39 |

### 41 帧 AP 评估

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --cav-count <N> --max-frames 0
```

| CAV Count | Frames | Cluster-Head Sources | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 41 | 82 | 0.33 | 0.32 | 0.18 | 113670.63 | 9320992 | 2.50 |
| 10 | 41 | 123 | 0.63 | 0.59 | 0.31 | 165169.30 | 20315824 | 3.33 |
| 15 | 41 | 205 | 0.69 | 0.66 | 0.34 | 130304.62 | 26712448 | 3.00 |
| 20 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 |

### 观察

- AP 随参与 CAV 数量增加而明显提升，说明 SGCP late-fusion 口径确实受协同覆盖范围影响。
- 15 CAV 的 reconfiguration events 高于 20 CAV，提示子集选择会改变局部拓扑和 coalition search 路径；该现象不能简单解释为 CAV 越多越不稳定。
- 该结果适合作为离线规模敏感性第一版；论文级密度扩展仍需要重新导出不同车流密度/背景车密度场景。

## 2026-07-15 - 网络资源参数敏感性实验

### 目的

- 推进 P1 “网络资源扩展：不同带宽或子信道数量”。
- 在离线 SGCP replay/inference 路径中加入网络资源覆盖参数，验证 PPS 对子信道数量和总带宽的敏感性。

### 代码更新

- `opencda.tools.offline_replay`
  - 新增 `--num-channels`，覆盖 `world.network_manager.subchannel_num` 和 PPS `Params.num_channels`。
  - 新增 `--bandwidth-mhz`，覆盖 PPS `Params.bandwidth_all` 并重算 `bandwidth_per_channel`。
- `opencda.tools.offline_inference`
  - 新增相同参数，使 AP 评估与 replay 共享同一网络资源设置。

### 验证

语法检查：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; conda run -n opencda python -m py_compile opencda\tools\offline_replay.py opencda\tools\offline_inference.py
```

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --num-channels 5 --max-frames 3 --summary-only
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --bandwidth-mhz 20 --max-frames 3 --summary-only
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --num-channels 5 --max-frames 1
```

三组均可完成。

### 41 帧 replay 汇总

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --num-channels <N> --max-frames 0 --summary-only
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --bandwidth-mhz <MHz> --max-frames 0 --summary-only
```

| Setting | Avg. Clusters | Avg. Cluster Size | Avg. Isolated CAVs | Reconfig. Events | Vehicle-Head Changes | Avg. Cluster Lifetime (frames) | Avg. Runtime (ms) | Avg. RA Runtime (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `num_channels=5` | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 90.10 | 27.35 |
| `num_channels=10` | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.99 | 37.39 |
| `num_channels=20` | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 104.57 | 42.35 |
| `bandwidth_mhz=20` | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 103.88 | 40.25 |
| `bandwidth_mhz=40` | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 99.99 | 37.39 |
| `bandwidth_mhz=80` | 6.00 | 3.33 | 0.00 | 11 | 76 | 6.65 | 101.11 | 38.93 |

### 41 帧 AP 评估

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --num-channels <N> --max-frames 0
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --bandwidth-mhz <MHz> --max-frames 0
```

| Setting | Frames | Cluster-Head Sources | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `num_channels=5` | 41 | 246 | 0.56 | 0.53 | 0.27 | 60225.24 | 14815408 | 1.83 | 45.58 |
| `num_channels=10` | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |
| `num_channels=20` | 41 | 246 | 0.77 | 0.73 | 0.38 | 139299.64 | 34267712 | 3.33 | 117.18 |
| `bandwidth_mhz=20` | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |
| `bandwidth_mhz=40` | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |
| `bandwidth_mhz=80` | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |

### 观察

- 子信道数量明显影响 PPS 选择的簇内上传成员数：5 个子信道时平均 source CAV 只有 1.83，AP 下降；20 个子信道时平均 source CAV 达到 3.33，payload 增加且 AP@0.7 提升到 0.38。
- replay 中 cluster/reconfiguration 指标不随网络资源变化，因为 coalition formation 与 PPS 调度解耦；网络资源主要影响每个 cluster head 能接收哪些成员点云。
- 单独改变 `bandwidth_mhz=20/40/80` 当前没有改变 AP、payload 或 selected grids。代码复核显示 `bandwidth_per_channel` 已进入 `PotentialGame.calculate_max_grids_per_rb()` 和 SINR/吞吐计算，但当前 41 帧 dump 下实际调度未受该上限约束，主要受离散子信道数量、每簇头 `B_h=1` RB 和候选成员/网格集合约束。

### 机制复核补充

- `opencda.core.clustering.algorithms.resource_allocation.potential_game.PotentialGame` 中，`bandwidth_all` 会被换算为 `bandwidth_per_channel = bandwidth_all / num_channels`。
- `bandwidth_per_channel` 进入 `calculate_max_grids_per_rb()`、`compute_data_rate()` 和 `bits_to_sinr()`。
- 本轮新增 inference summary 字段 `avg_selected_grids`，确认 5/10/20 子信道分别为 45.58/87.32/117.18，而 20/40/80 MHz 均为 87.32。
- 因此当前现象不是参数没有传入，而是该 dump 的 PPS 选择不由带宽上限主导。后续如需论文中展示带宽敏感性，需要尝试更低带宽、更大 grid payload、更高点云密度或更多候选上传网格的场景。

## 2026-07-15 - 低带宽瓶颈触发实验

### 目的

- 推进 P3 “构造能触发带宽瓶颈的 SGCP 场景或参数组”。
- 在不重新导出 CARLA 数据的前提下，使用极低 `bandwidth_mhz` 压力测试确认 `PotentialGame` 的带宽吞吐约束是否可观测生效。

### 3 帧 smoke test

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --bandwidth-mhz 0.1 --max-frames 3
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --bandwidth-mhz 0.5 --max-frames 3
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --bandwidth-mhz 1.0 --max-frames 3
```

结果：

| Bandwidth (MHz) | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Avg. Source CAVs | Avg. Selected Grids |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.27 | 0.23 | 0.10 | 0.00 | 1.00 | 0.00 |
| 0.5 | 0.54 | 0.49 | 0.21 | 34426.67 | 2.39 | 4.17 |
| 1.0 | 0.63 | 0.55 | 0.23 | 59624.00 | 2.56 | 9.33 |

### 41 帧 AP 评估

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --bandwidth-mhz <MHz> --max-frames 0
```

| Bandwidth (MHz) | Frames | Cluster-Head Sources | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 41 | 246 | 0.26 | 0.22 | 0.09 | 0.00 | 0 | 1.00 | 0.00 |
| 0.5 | 41 | 246 | 0.56 | 0.50 | 0.23 | 39694.05 | 9764736 | 2.44 | 4.32 |
| 1.0 | 41 | 246 | 0.66 | 0.61 | 0.31 | 75639.67 | 18607360 | 2.61 | 9.66 |
| 20.0 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |
| 40.0 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |
| 80.0 | 41 | 246 | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |

### 观察

- `bandwidth_mhz=0.1` 时所有 cluster head 均无成员点云上传，退化为 inter-cluster late fusion of cluster heads。
- `0.5/1.0 MHz` 逐步恢复成员上传，selected grids、payload 和 AP 同步上升。
- `20/40/80 MHz` 在当前 dump 上完全重合，说明常规带宽已超过该场景 PPS 可用候选网格需求。
- 论文写作建议：把 0.1/0.5/1.0 MHz 定位为 stress test，用于证明带宽约束实现有效；把 5/10/20 子信道实验作为常规网络资源敏感性主结果。
