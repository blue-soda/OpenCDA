# SGCP Offline Replay Plan

更新时间：2026-07-15

本文档记录 SGCP 离线回放能力的工程设计。目标是在不启动 CARLA、NS3 的情况下，读取 `v2xp_cluster_carla` 导出的 OPV2V 风格数据集，复现每帧的 SGCP clustering/resource scheduling 过程，并将结果接入 OpenCOOD 离线推理与评估。

## 当前输入

- 数据集根目录：见 `../environment.md`，当前为 `D:\Data\Carla`。
- 已验证数据集：`D:\Data\Carla\2026_07_15_01_26_56`。
- 数据规模：20 个 CAV，每个 CAV 41 帧。
- 单帧内容：每个 CAV 包含 `.pcd` 点云与 `.yaml` 位姿/标签元数据。
- 当前通用读取入口：`opencda.core.common.offline_dataset.OPV2VFrameDataset`。
- 当前通用推理入口：`python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla`。
- 当前 SGCP 回放入口：`python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla`。

## SGCP 在线依赖

SGCP 在线路径当前依赖这些对象和字段：

- `ClusteringV2XManager.run_algorithm()`：触发 coalition formation，并在 cluster-based scheduler 启用时触发 resource allocation。
- `CoalitionGame.initialize_vehicles()`：通过 `cav_world.get_vehicle_managers()` 构建 `common.global_vehicles/global_vms`。
- `common.Vehicle_Grid.initialize()`：读取每车的 `position/speed/direction` 和 lidar grid 状态。
- `vm.perception_manager.lidar`：提供 `sens_grids`、`req_grids`、`high_density_grids`、`grid_size`、`density_threshold`、`grid_density_dict`。
- `ClusteringScheduler`：接收 cluster 列表，再调用资源分配算法生成 channel allocation。

因此离线回放不应直接依赖 CARLA actor，而应从 dump 帧重建一个轻量状态快照，满足上述字段访问。

## 最小可行接口

第一阶段新增一个轻量离线状态适配层，建议路径：

- `opencda.core.common.offline_replay`

候选类：

- `OfflineCavWorld`
  - `ego_id`
  - `get_vehicle_managers()`
  - `get_vehicle_manager(vehicle_id)`
- `OfflineVehicleManager`
  - `is_ok`
  - `v2x_manager`
  - `perception_manager`
- `OfflineV2XManager`
  - `vehicle_id`
  - `get_ego_pos()`
  - `get_ego_speed()`
  - `get_ego_dir()`
  - `scheduler`
  - `cluster_state`
- `OfflineLidarGrid`
  - 从 `.pcd` 和 `lidar_pose` 生成 `sens_grids/grid_density_dict/high_density_grids`
  - 从 `lidar_pose`、`required_perception_range`、`grid_size` 生成 `req_grids`

第二阶段新增一个命令行入口，建议路径：

- `opencda.tools.offline_replay`

初始命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0
```

初始输出：

- 每帧 cluster head 与 member 列表。
- 每帧 channel allocation。
- 每帧 cluster 数量、平均 cluster size、孤立车辆数量。
- 汇总指标：cluster lifetime、reconfiguration 次数、平均 cluster size、孤立车辆数量、运行时耗时。

## P0 实施顺序

1. [x] 从 `data_protocol.yaml` 读取 SGCP 所需参数，优先包括 lidar 配置和 network scheduler 类型。
2. [x] 从单帧 dump 构建 `OfflineCavWorld`，验证 `common.Vehicle_Grid.initialize(offline_world)` 能成功填充 20 个 CAV。
3. [x] 在单帧上调用 `CoalitionGame(offline_world).run()`，输出 cluster 列表。
4. [x] 接入默认 resource allocation，输出 channel allocation。
5. [x] 扩展到多帧回放，记录 cluster lifetime、reconfiguration 次数、平均 cluster size。
6. [ ] 将回放结果与 `offline_inference` 对齐到相同 frame/timestamp，形成后续 SGCP 约束感知评估入口。

## 已验证命令

单帧 clustering + 默认资源分配：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1
```

结果摘要：

- CAV 数量：20
- cluster 数量：6
- 平均 cluster size：3.33
- 默认 `NaiveRA` channel allocation：380 条

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3
```

全量 41 帧稳定性汇总：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --summary-only
```

结果摘要：

- frames：41
- avg_clusters：6.00
- avg_cluster_size：3.33
- avg_isolated_cavs：0.00
- max_isolated_cavs：0
- reconfiguration_events：11
- vehicle_head_changes：76
- avg_cluster_lifetime_frames：6.65
- min/max cluster lifetime：1 / 38 frames
- avg_total_runtime：129.78 ms
- avg_resource_allocation_runtime：1.13 ms

## 风险与注意

- Dump 的 `.yaml` 当前主要保存位置和标签，不一定包含速度；若缺失速度，应先用相邻帧位置差分估计，并在结果中标记估计方式。
- 在线 `common.global_vehicles` 是模块级全局状态，离线多帧回放前需要明确清理策略，避免上一帧车辆状态污染下一帧。
- 当前 `ClusteringScheduler` 默认资源分配算法是 `NaiveRA`，论文 SGCP 对应算法需要确认是 `PotentialGame`、`PCS`、`MWS` 还是现有默认实现。
- 需要确认离线 grid 计算是否与在线 `perception_manager.lidar` 完全一致，否则 `f(rho)` 和资源调度结果不可直接与在线实验混用。
