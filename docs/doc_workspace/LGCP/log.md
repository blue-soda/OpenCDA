# LGCP 实验日志

本文档按时间追加记录 LGCP 相关实验、代码验证、日志排查和论文修订过程。每次记录应尽量包含：目标、环境、命令、配置、日志路径、关键观察、结论和下一步。

## 记录模板

````markdown
## YYYY-MM-DD HH:mm - 实验标题

### 目标

- 

### 环境

- 仓库：
- 分支：
- Conda 环境：
- CARLA：
- NS3：
- 数据集 / checkpoint：

### 命令

```powershell

```

### 配置

- 

### 日志与产物

- OpenCDA log：
- NS3 log：
- 结果文件：

### 观察

- 

### 结论

- 

### 下一步

- 
````

## 2026-07-15 - 初始化 LGCP 文档工作区

### 目标

- 基于仓库 `README.md` 和 `AGENT_README.md`，建立 LGCP 文档记录结构。
- 将论文审稿意见中暴露的问题转化为后续实验和机制完善的跟踪入口。

### 环境

- 仓库：`C:\Workspace\OpenCDA`
- 文档目录：`C:\Workspace\OpenCDA\docs\doc_workspace\LGCP`
- 论文材料：`C:\Workspace\icdcs-paper\LGCP`
- 审稿意见：`C:\Workspace\icdcs-paper\LGCP-review.txt`

### 观察

- OpenCDA 适合承载 CARLA / NS3 co-simulation 和多 CAV 通信调度实验。
- OpenCOOD 适合承载 OPV2V / V2XSet 上的协同感知模型评估。
- LGCP 需要同时记录论文级实验结果和系统级仿真日志。

### 结论

- 建立 `readme.md`、`target.md`、`status.md`、`log.md`、`results.md` 五个基础文档。
- 后续每次实验先写入 `log.md`，确认可复现后再沉淀到 `results.md`。

### 下一步

- 建立审稿意见 revision matrix。
- 确认本仓库中是否已有 LGCP 代码或相关实验脚本。
- 设计 area confidence validation 的最小可行实验。

## 2026-07-15 - 评估 v2xp_cluster_carla 与新增 LGCP 场景

### 目标

- 判断 `python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug (--network)` 是否满足 LGCP 仿真需求。
- 在不修改 `v2xp_cluster_carla` 的前提下，必要时新增 LGCP 专用配置。

### 环境

- 仓库：`C:\Workspace\OpenCDA`
- 读取文件：`opencda.py`、`opencda/scenario_testing/template.py`、`v2xp_cluster_carla.py`、`v2xp_cluster_carla.yaml`、`enable_network.yaml`、cluster / coperception 相关 manager。

### 观察

- `v2xp_cluster_carla` 使用 Town03，traffic range 为 `[-50, 40, -100, 100, 3.5, 15, 99, 19]`，含 99 辆背景车，其中 19 辆被包装成 traffic vehicle manager，加上 1 个 single CAV，总计 20 个 managed / intelligent vehicles。
- `--network` 会合并 `enable_network.yaml`，启用 `scheduler: cluster` 和 `use_ns3: true`。
- 现有 cluster 管线是车辆簇机制：成员上传给簇头，簇头做协同感知并向簇成员广播结果。
- LGCP 需要 RSU 中心调度、RoI area 划分、area-specific group、leader local fusion、leader-to-RSU upload、RSU global aggregation；现有 `v2xp_cluster_carla` 没有启用 RSU，也没有完整 LGCP 机制。

### 结论

- `v2xp_cluster_carla` 可作为规模和地形参考，但不能直接作为 LGCP 完整仿真配置。
- 新增 `opencda/scenario_testing/lgcp_carla.py` 和 `opencda/scenario_testing/config_yaml/lgcp_carla.yaml`。
- 新场景保持 Town03、100 total vehicles、20 managed vehicles、NS3 可选启用，并新增 RSU 与 grid-aware LiDAR 参数。

### 验证

```powershell
python -m py_compile C:\Workspace\OpenCDA\opencda\scenario_testing\lgcp_carla.py
conda run -n opencda python -c "from omegaconf import OmegaConf; ..."
```

配置合并检查确认：

- single CAV 数：1
- RSU 数：1
- traffic range：`[-50, 40, -100, 100, 3.5, 15, 99, 19]`
- vehicle / traffic LiDAR grid：enabled，`grid_size=10.0`
- network：`scheduler=cluster`，`use_ns3=True`

### 下一步

- 实现或接入 LGCP 专用 area assignment / leader upload / RSU aggregation 逻辑。
- 明确 LGCP area confidence、group selection 和 transmission scheduling 的日志字段。

## 2026-07-15 - 补齐 RSU 首次启用基础路径

### 目标

- 检查项目此前未启用 RSU 时可能隐藏的实现缺口。
- 修复 LGCP 专用场景启用 RSU 时的基础崩溃点。

### 观察

- `RSUManager` 会以 `v2x_manager=None` 创建 `PerceptionManager`，但 `PerceptionManager` 原先直接读取 `v2x_manager.vid`，会在 RSU 初始化时崩溃。
- `CavWorld.destroy()` 中调用了不存在的 `rsu_manager.destory()`。
- `template.stop()` 没有显式销毁 RSU。
- `CavWorld` 只有 RSU 注册接口，没有读取接口，不利于后续 LGCP 以 RSU 为中心做调度。

### 修改

- `PerceptionManager` 支持 `infra_id` / infrastructure 模式，不再要求必须存在 `v2x_manager`。
- RSU 的 server-side detection 半径使用固定 LiDAR range，而不是车辆默认的 50m。
- `RSUManager` 保存最近一次感知结果，并提供 `get_objects()`。
- `CavWorld` 新增 `get_rsu_managers()` 和 `get_rsu_manager()`。
- 修复 `CavWorld.destroy()` 的 RSU 销毁拼写错误。
- `template.stop()` 增加 RSU 显式销毁。

### 验证

```powershell
python -m py_compile C:\Workspace\OpenCDA\opencda\core\sensing\perception\perception_manager.py C:\Workspace\OpenCDA\opencda\core\common\rsu_manager.py C:\Workspace\OpenCDA\opencda\core\common\cav_world.py C:\Workspace\OpenCDA\opencda\scenario_testing\template.py C:\Workspace\OpenCDA\opencda\scenario_testing\lgcp_carla.py
conda run -n opencda python -c "from opencda.core.common.rsu_manager import RSUManager; from opencda.core.common.cav_world import CavWorld; from opencda.core.sensing.perception.perception_manager import PerceptionManager; import opencda.scenario_testing.lgcp_carla as s; print('rsu imports ok')"
```

结果：编译与项目环境导入均通过。

### 下一步

- 启动 CARLA 后实际运行 `lgcp_carla`，确认 RSU GNSS / LiDAR actor 能正常生成和销毁。
- 在 RSU 基础实体之上实现 LGCP 的 RSU 控制面逻辑。

## 2026-07-15 - 运行 lgcp_carla 纯 CARLA 场景

### 目标

- 通过 CARLA 实际运行一次 `lgcp_carla`，验证新增 RSU 场景能完成启动、车辆生成、RSU 创建、协同感知评估和场景关闭。

### 环境

- CARLA：`C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe`
- 地图：Town03
- Conda 环境：`opencda`
- 命令：`conda run -n opencda python opencda.py -t lgcp_carla --apply_cp --apply_ml --debug`
- 未启用 NS3：未加 `--network`

### 过程

- 第一次运行完成了车辆和 RSU 创建，但 Open3D 截图写入 `visualization_output/visualize.png` 失败，随后 PIL 读取缺失文件时报错。
- 修复 `o3d_lidar_libs.py`：截图前创建输出目录，透明背景转换前检查截图文件是否存在。
- 第二次运行成功结束。

### 观察

- 创建 single CAV：1。
- 创建 traffic vehicle managers：19。
- CARLA traffic flow：99 vehicles and 19 vms。
- 创建 RSU：成功。
- 车辆 LiDAR grid：`grid_size=10.0`，每车生成 2401 perception grids。
- RSU LiDAR grid：`grid_size=10.0`，生成 9409 perception grids。
- CP 计数：8。

### 结果

- 场景正常输出 `Simulation is Over` 和 `Simulation closed`。
- AP 结果：
  - AP@0.3：0.81
  - AP@0.5：0.81
  - AP@0.7：0.71
- 最新日志：`C:\Workspace\OpenCDA\opencda\log\opencda_20260715_015911.log`
- 可视化产物：
  - `C:\Workspace\OpenCDA\visualization_output\visualize.png`
  - `C:\Workspace\OpenCDA\visualization_output\visualize_transparent.png`
  - `C:\Workspace\OpenCDA\visualization_output\visualize_spectator_view.png`

### 残留提示

- 场景关闭时仍出现两个 actor already destroyed 提示：`unable to destroy actor: not found`。它没有阻止场景结束，原因大概率是 RSU 显式销毁后，`ScenarioManager.close()` 又统一销毁 world actors。
- NumPy 出现 empty slice warning，不影响本次运行完成。

### 下一步

- 可选择优化关闭流程，避免显式销毁与 world actor 全量销毁重复。
- 后续再运行 `--network` 前，需要启动并验证 NS3。

## 2026-07-15 - LGCP 数据导出与离线推理 smoke test

### 目标

- 仿照 `v2xp_cluster_carla` 的 data dump 流程，使 `lgcp_carla` 支持快速导出 OPV2V 风格数据。
- 根据 `docs/doc_workspace/environment.md`，完成数据导出和离线 OpenCOOD 推理 smoke test。
- 判断当前 LGCP 配置是否适合继续推进 `target.md`。

### 修改

- `lgcp_carla.py` 新增 `--dump` 分支，应用列表为 `['data_dump', 'rsu']`。
- `lgcp_carla.yaml` 将 ego destination 从 `[8.00, -41, 1.0]` 收近到 `[8.00, -40.50, 1.0]`。

### 说明

- OpenCDA 的 `BehaviorAgent.is_close_to_destination()` 使用 10m 结束阈值：当 ego 与终点的 x、y 误差均不超过 10m 时直接结束仿真。
- 因此 destination 不能放到离 spawn 太近的位置，否则无法导出数据。当前设置是刚超过该阈值的短路线。

### 数据导出

```powershell
$env:OPENCDA_DATA_DUMP_ROOT = "D:\Data\Carla"
$env:OPENCDA_DATADUMP_TICKS = "80"
conda run -n opencda python opencda.py -t lgcp_carla --dump
```

结果：

- 导出目录：`D:\Data\Carla\2026_07_15_02_33_21`
- 包含 RSU 目录：`-1`
- 包含 CAV 目录：`1` 到 `20`
- 每个目录导出 11 个 `.yaml` 和 11 个 `.pcd`
- ego CAV `1` 还包含 4 路 camera PNG

### 离线推理

单帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --timestamp 000060 --ego-cav-id 1
```

结果：

- cavs：`[-1, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 3, 4, 5, 6, 7, 8, 9]`
- `pred_boxes=63`
- `gt_boxes=72`

三帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3
```

结果：

- frame 000060：`pred_boxes=63`，`gt_boxes=72`
- frame 000062：`pred_boxes=64`，`gt_boxes=72`
- frame 000064：`pred_boxes=63`，`gt_boxes=72`
- AP@0.3：0.87
- AP@0.5：0.76
- AP@0.7：0.49

### 结论

- 当前 `lgcp_carla` 配置适合作为 LGCP 的数据导出、离线推理和后续机制验证基础。
- RSU 数据已经能进入导出目录，并可被离线 loader 作为 `-1` agent 加载。
- 后续仍需实现 LGCP 专用 area confidence、area group selection、leader upload 和 RSU global aggregation。

## 2026-07-15 - 建立 LGCP revision matrix

### 目标

- 根据 `readme.md` 的文档规则，继续推进 `target.md`。
- 将 `C:\Workspace\icdcs-paper\LGCP\LGCP-review.txt` 中的审稿意见转化为可执行的 revision / rebuttal 对照表。

### 输入

- 审稿意见：`C:\Workspace\icdcs-paper\LGCP\LGCP-review.txt`
- 当前任务清单：`docs/doc_workspace/LGCP/target.md`
- 当前状态：`docs/doc_workspace/LGCP/status.md`

### 产物

- 新增：`docs/doc_workspace/LGCP/revision_matrix.md`

### 结论

- 审稿意见被归纳为四个最高风险点：
  - area confidence 未验证；
  - 算法被认为是 heuristic；
  - baseline fairness 不足；
  - 大规模实验只报告 latency。
- `revision_matrix.md` 已将每条意见映射到 Priority、修改动作、论文/实验产物和当前状态。
- `target.md` 中 “梳理 LGCP 当前论文问题，形成 rebuttal / revision 对照表” 已标记完成。

### 下一步

- 按 revision matrix 的 P0 顺序推进：
  - area confidence validation；
  - Eq. (2) composition validation；
  - adaptive/selective sharing baseline；
  - local-to-global ablation。

## 2026-07-15 - 设计 area confidence validation 实验

### 目标

- 继续推进 `target.md` 中下一个 P0：设计 area confidence 验证实验。
- 回应审稿意见中 “area confidence 未验证，缺少 correlation with area-level AP / recall” 的核心问题。

### 代码依据

- 离线数据读取：`opencda/core/common/offline_dataset.py`
- 离线推理入口：`opencda/tools/offline_inference.py`
- OpenCOOD AP 统计：`opencda/core/ml_libs/opencood_manager.py`
- 现有 grid density score：`opencda/core/clustering/utils/common.py`
- LGCP ROI / grid 元数据：`opencda/scenario_testing/config_yaml/lgcp_carla.yaml`

### 产物

- 新增：`docs/doc_workspace/LGCP/area_confidence_validation.md`

### 结论

- 第一版验证单元确定为 `(scenario_id, timestamp, area_id, agent_id)`。
- area 划分优先沿用 `lgcp_carla.yaml` 的 ROI 和 grid size。
- confidence 将同时记录 density、distance、density-distance、detector score mean/top-k 等候选变量。
- area-level quality 将按 box center 归属 area，统计 recall/AP，并用 Pearson、Spearman、Kendall tau 和 calibration bins 验证相关性。
- 实验结果建议统一输出到 `docs/doc_workspace/LGCP/experiments/area_confidence/`，确认可复现后再同步到 `results.md`。

### 状态更新

- `target.md` 中 “设计 area confidence 验证实验” 已标记完成。
- `target.md` 中 “确认 LGCP 所需 area-level confidence / AP / recall 的数据导出位置” 已标记完成。
- 下一步应实现离线脚本 `opencda/tools/lgcp_area_confidence_eval.py` 并用已有 `D:\Data\Carla\2026_07_15_02_33_21` 数据做 smoke test。

## 2026-07-15 - 设计 Eq. (2) composition validation 实验

### 目标

- 继续推进 `target.md` 中 P0：设计 Eq. (2) 组合规则验证实验。
- 回应审稿意见中 “whether Eq. (2) is a reasonable composition rule” 的问题。

### 论文依据

- `C:\Workspace\icdcs-paper\LGCP\conference_101719.tex` 中 Eq. (2) 为：

```text
F_i(V_i) = 1 - product_{v_k in V_i}(1 - F_i({v_k}))
```

- 该公式对应 noisy-or / independent success probability composition，隐含多 CAV 对同一区域的感知成功事件近似独立。

### 产物

- 新增：`docs/doc_workspace/LGCP/eq2_composition_validation.md`

### 结论

- Eq. (2) 需要和 `max`、`mean`、`sum_clipped`、`top2_mean`、`top3_mean`、`softmax_weighted`、`calibrated_noisy_or` 做对照。
- 第一版 subset sampling 不枚举 20 CAV 全部组合，而是按 area-frame 对候选 CAV 做小规模枚举和采样。
- 除相关性外，需要计算 group selection regret：比较某组合规则选出的 subset 与真实 fused AP/recall oracle subset 的差距。
- Eq. (2) 的边际收益递减假设需要通过 marginal gain consistency 单独验证。

### 状态更新

- `target.md` 中 “设计 Eq. (2) 组合规则验证实验” 已标记完成。
- 下一步建议推进 greedy group selection small-scale optimality gap，因为它可以复用 Eq. (2) 文档中定义的 subset oracle 和 regret 计算。

## 2026-07-15 - 设计 greedy group selection optimality gap 实验

### 目标

- 继续推进 `target.md` 中 P0：greedy group selection 的 small-scale optimality gap。
- 回应审稿意见中 “heuristic only / no approximation guarantee” 的问题。

### 论文依据

- `conference_101719.tex` 中 selection algorithm 先基于 `Delta_g` 为每个 area 贪心构造 CAV group，再按 group size 降序和当前 leader load 贪心分配 leader。
- 当前 revision 更稳妥的表述应是 practical heuristic + empirical optimality gap，而不是未证明的 approximation guarantee。

### 产物

- 新增：`docs/doc_workspace/LGCP/greedy_optimality_gap.md`

### 结论

- optimality gap 实验应拆成 group member selection gap 和 leader load balancing gap。
- 第一版小规模 oracle 使用 exhaustive search；后续可用预枚举 subset 的 set-packing ILP / MILP 扩展规模。
- 评价指标包括 relative objective gap、quality gap、cost gap、load gap 和 runtime。
- 该实验依赖 `area_confidence_validation.md` 和 `eq2_composition_validation.md` 中的 per-area confidence / subset records。

### 状态更新

- `greedy_optimality_gap.md` 设计已完成。
- `target.md` 中该项仍保持未完成，因为还需要实现离线工具并产出真实 gap 结果。

## 2026-07-15 - 建立 LGCP 实验结果目录规则

### 目标

- 继续推进 `target.md` 中 “建立统一结果目录和命名规则”。
- 为 area confidence、Eq. (2)、greedy gap、ablation 和 baseline 实验建立统一输出约定。

### 产物

- 新增：`docs/doc_workspace/LGCP/experiments/readme.md`

### 结论

- 大型原始数据继续保存在 `../environment.md` 指定的数据目录，例如 `D:\Data\Carla`。
- `docs/doc_workspace/LGCP/experiments/` 只保存轻量级结果、配置快照、CSV summary 和 notes。
- `run_id` 统一为 `YYYYMMDD_<scenario>_<purpose>`。
- 每个实验 run 至少包含 `config.yaml`、`notes.md` 和对应 `*_summary.csv`。

### 状态更新

- `target.md` 中 “建立统一结果目录和命名规则” 已标记完成。

## 2026-07-15 - area confidence 离线导出 smoke test

### 目标

- 实现 `area_confidence_validation.md` 中的第一阶段离线导出链路。
- 验证 `lgcp_carla` dump 是否能按 ROI/grid 生成 `(scenario_id, timestamp, area_id, agent_id)` records。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_area_confidence_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_smoke --max-frames 3
```

### 产物

- 新增脚本：`opencda/tools/lgcp_area_confidence_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_smoke`
- 输出文件：
  - `area_records.csv`
  - `density_gt_summary.csv`
  - `config.yaml`
  - `notes.md`

### 结果

| Timestamp | Rows | GT areas | GT objects | Density-GT Pearson | Density-GT Spearman |
| --- | --- | --- | --- | --- | --- |
| 000060 | 1945 | 12 | 14 | -0.028869 | -0.051536 |
| 000062 | 1949 | 12 | 14 | -0.029360 | -0.072819 |
| 000064 | 1942 | 12 | 14 | -0.028361 | -0.064157 |

总计导出 `5836` 条 area-agent records。

### 结论

- ROI/grid/agent-area records 导出链路跑通。
- 当前结果只验证数据链路，不是论文可用的 area confidence 有效性证据。
- density 与 GT count 的相关性接近 0，说明不能用点云密度本身替代审稿人要求的 confidence-vs-area AP / recall。
- 下一步需要在该脚本上接入 OpenCOOD prediction slicing 和 area-level AP / recall 计算。

### 状态更新

- `target.md` 中 “实现 area confidence 离线导出 smoke test，验证 ROI/grid/agent-area records 链路” 已标记完成。
- `target.md` 新增待办：“扩展 area confidence 离线评估脚本，接入 prediction slicing 与 area-level AP / recall”。

## 2026-07-15 - prediction slicing 与 area-level recall smoke test

### 目标

- 在 `opencda/tools/lgcp_area_confidence_eval.py` 中接入 OpenCOOD inference。
- 将 `pred_box_tensor / pred_score / gt_box_tensor` 按 LGCP area 切分，导出 per-area quality records。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_area_confidence_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_inference_smoke_3f --max-frames 3 --with-inference
```

### 产物

- 更新脚本：`opencda/tools/lgcp_area_confidence_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_inference_smoke_3f`
- 新增输出：`area_quality.csv`

### 结果

| Timestamp | Rows | GT objects in YAML ROI | Quality GT boxes | Quality pred boxes | Recall@0.5 |
| --- | --- | --- | --- | --- | --- |
| 000060 | 1945 | 14 | 48 | 40 | 0.791667 |
| 000062 | 1949 | 14 | 47 | 42 | 0.851064 |
| 000064 | 1942 | 14 | 47 | 40 | 0.829787 |

总计：

- `area_records.csv`：5836 条 area-agent records。
- `area_quality.csv`：97 条 area quality rows。

### 结论

- prediction slicing 链路跑通，脚本现在能输出 per-area `pred_count`、`gt_count`、`tp/fp`、`recall` 和 `precision`。
- 当前仍未累计 area-level AP，也未计算 confidence-vs-AP/recall 相关性。
- YAML ROI GT object count 与 OpenCOOD quality GT boxes 不一致，是因为前者来自 ego YAML 中的车辆列表，后者来自 OpenCOOD fusion/evaluation 范围内的 GT tensor；后续统计以 `area_quality.csv` 为准。

### 状态更新

- `target.md` 中 “扩展 area confidence 离线评估脚本，接入 prediction slicing 与 area-level recall smoke test” 已标记完成。
- 新增待办：“扩展 area confidence 离线评估脚本，累计 area-level AP 并统计 confidence-vs-AP/recall 相关性”。

## 2026-07-15 - area AP 与 confidence-quality correlation smoke test

### 目标

- 继续推进 area confidence validation：累计 area-level AP，并统计 confidence-vs-AP/recall 相关性。
- 将 `area_records.csv` 与 `area_quality.csv` 连接，输出可直接用于后续论文图表的 summary。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_area_confidence_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_smoke_3f --max-frames 3 --with-inference
```

### 产物

- 更新脚本：`opencda/tools/lgcp_area_confidence_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_smoke_3f`
- 新增输出：
  - `area_ap_summary.csv`
  - `confidence_quality_records.csv`
  - `confidence_quality_correlation.csv`

### 结果

| 文件 | 行数 | 含义 |
| --- | --- | --- |
| `area_records.csv` | 5836 | per-agent area confidence records |
| `area_quality.csv` | 97 | per-frame area recall / precision |
| `area_ap_summary.csv` | 34 | accumulated area AP summary |
| `confidence_quality_records.csv` | 97 | joined confidence-quality records |
| `confidence_quality_correlation.csv` | 30 | correlation summary |

代表性相关性：

| Scope | Confidence | Quality | Samples | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| area_frame | confidence_max | recall_05 | 97 | 0.424462 | 0.647552 |
| area_frame | confidence_noisy_or | recall_05 | 97 | 0.385940 | 0.647552 |
| area_accumulated | confidence_max | ap_05 | 33 | 0.430768 | 0.524064 |
| area_accumulated | confidence_noisy_or | ap_05 | 33 | 0.371881 | 0.524064 |

### 结论

- area-level AP 累计和 confidence-quality correlation 链路跑通。
- 3 帧 smoke 出现正相关信号，但样本不足，不能作为最终论文结论。
- 当前 confidence 仍是 density-based proxy，下一步需要扩大样本并补充 detector-score / feature confidence。

### 状态更新

- `target.md` 中 “扩展 area confidence 离线评估脚本，累计 area-level AP 并统计 confidence-vs-AP/recall 相关性 smoke test” 已标记完成。
- 新增待办：“扩大 area confidence validation 到更多帧 / 多 seed，并替换或补充 detector-score confidence”。

## 2026-07-15 - 11 帧 area confidence validation 与 detector-score 对照

### 目标

- 继续推进 `target.md` 中 “扩大 area confidence validation 到更多帧，并替换或补充 detector-score confidence”。
- 使用完整 `lgcp_carla` dump 的 11 帧，评估 density-based confidence 与 detector-score confidence 的相关性差异。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_area_confidence_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score --max-frames 0 --with-inference
```

### 产物

- 更新脚本：`opencda/tools/lgcp_area_confidence_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/area_confidence/20260715_lgcp_carla_area_ap_11f_detector_score`
- 输出文件：
  - `area_records.csv`
  - `area_quality.csv`
  - `area_ap_summary.csv`
  - `confidence_quality_records.csv`
  - `confidence_quality_correlation.csv`

### 结果

| 文件 | 行数 |
| --- | --- |
| `area_records.csv` | 21418 |
| `area_quality.csv` | 363 |
| `area_ap_summary.csv` | 40 |
| `confidence_quality_records.csv` | 363 |
| `confidence_quality_correlation.csv` | 54 |

代表性相关性：

| Scope | Confidence | Quality | Samples | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| area_frame | confidence_max | recall_05 | 354 | 0.242796 | 0.570690 |
| area_frame | confidence_noisy_or | recall_05 | 354 | 0.229353 | 0.570407 |
| area_accumulated | confidence_max | ap_05 | 36 | 0.401529 | 0.411840 |
| area_accumulated | confidence_noisy_or | ap_05 | 36 | 0.380141 | 0.411840 |
| area_accumulated | score_mean | ap_05 | 36 | 0.299850 | 0.402059 |
| area_accumulated | score_top2_mean | ap_07 | 36 | 0.349922 | 0.401030 |

### 结论

- 完整 11 帧 dump 中，density-based confidence 对 area-level recall / AP 有稳定正相关信号。
- Detector-score confidence 在逐帧 recall 上较弱，但对 accumulated AP 有中等正相关，可作为对照指标保留。
- 该结果仍来自单场景 / 单 seed，不能作为最终论文结论；下一步应扩展到多 seed / 多场景。

### 状态更新

- `target.md` 中 “扩大 area confidence validation 到完整 `lgcp_carla` dump 帧，并补充 detector-score confidence 对照” 已标记完成。
- 新增待办：“扩大 area confidence validation 到多 seed / 多场景，形成可进入论文的稳定相关性结果”。

## 2026-07-15 - greedy group-member selection optimality gap smoke test

### 目标

- 继续推进 `target.md` 中 P0：greedy group selection 的 small-scale optimality gap。
- 先覆盖论文 selection algorithm 第一阶段：基于 `Delta_g` 为每个 area 构造 CAV group。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_greedy_gap_eval --input-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score --output-dir docs\doc_workspace\LGCP\experiments\greedy_optimality_gap\20260715_lgcp_carla_greedy_gap_density_distance --confidence-field density_distance --max-agents 6 --max-areas 5 --max-group-size 4 --lambda-size 0.05
```

### 产物

- 新增脚本：`opencda/tools/lgcp_greedy_gap_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260715_lgcp_carla_greedy_gap_density_distance`
- 输出文件：
  - `instance_records.csv`
  - `gap_summary.csv`
  - `config.yaml`
  - `notes.md`

### 结果

| Objective | Delta_g | Mean relative gap | P90 relative gap | Max relative gap |
| --- | --- | --- | --- | --- |
| O1 confidence only | 0.05 | 0.049030 | 0.052639 | 0.053657 |
| O1 confidence only | 0.075 | 0.063452 | 0.066516 | 0.066665 |
| O1 confidence only | 0.1 | 0.068748 | 0.084891 | 0.093525 |
| O1 confidence only | 0.125 | 0.109864 | 0.117836 | 0.117895 |
| O2 confidence minus size | 0.05 | 0.034831 | 0.038233 | 0.039755 |
| O2 confidence minus size | 0.075 | 0.047666 | 0.051154 | 0.051216 |
| O2 confidence minus size | 0.1 | 0.052629 | 0.068291 | 0.075165 |
| O2 confidence minus size | 0.125 | 0.092049 | 0.100414 | 0.100445 |

### 结论

- group-member selection gap smoke 已跑通。
- `density_linear` confidence 下 gap 为 0，说明该 proxy 在当前场景中容易饱和。
- `density_distance` confidence 下 gap 随 `Delta_g` 增大而扩大，能提供更有诊断性的 greedy-vs-oracle 证据。
- 该结果尚未覆盖 leader assignment / load balancing，因此 P0 greedy 项仍不能完全关闭。

### 状态更新

- `target.md` 中 “实现 greedy group-member selection exhaustive-search gap smoke test” 已标记完成。
- 新增待办：“扩展 greedy optimality gap 到 leader assignment / load balancing”。

## 2026-07-15 - leader assignment / load balancing optimality gap smoke test

### 目标

- 继续推进 `target.md` 中 greedy optimality gap 的剩余部分。
- 比较论文 leader greedy assignment 与 exhaustive min-max load oracle。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_greedy_gap_eval --input-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score --output-dir docs\doc_workspace\LGCP\experiments\greedy_optimality_gap\20260715_lgcp_carla_greedy_gap_with_leader --confidence-field density_distance --max-agents 6 --max-areas 5 --max-group-size 4 --lambda-size 0.05
```

### 产物

- 更新脚本：`opencda/tools/lgcp_greedy_gap_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/greedy_optimality_gap/20260715_lgcp_carla_greedy_gap_with_leader`
- 新增输出：
  - `leader_records.csv`
  - `leader_gap_summary.csv`

### 结果

| Delta_g | Mean relative gap | Median relative gap | P90 relative gap | Max relative gap | Mean absolute gap |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.128788 | 0.000000 | 0.250000 | 0.666667 | 0.454545 |
| 0.075 | 0.022727 | 0.000000 | 0.000000 | 0.250000 | 0.090909 |
| 0.1 | 0.022727 | 0.000000 | 0.000000 | 0.250000 | 0.090909 |
| 0.125 | 0.022727 | 0.000000 | 0.000000 | 0.250000 | 0.090909 |

### 结论

- leader assignment / load balancing gap smoke 已跑通。
- `Delta_g=0.05` 时 leader greedy 的 max-load gap 更明显；更大 `Delta_g` 下，多数 instance 与 oracle 一致。
- 至此 P0 greedy optimality gap 已覆盖 group-member selection 和 leader assignment 两部分。
- 后续仍应扩大到多 seed / 更大 instance，并接入 latency-aware `O3` objective。

### 状态更新

- `target.md` 中 “扩展 greedy optimality gap 到 leader assignment / load balancing” 已标记完成。
- `target.md` 中 P0 “增加 greedy group selection 的 small-scale optimality gap 实验” 已标记完成。
- 新增待办：“扩大 greedy optimality gap 到多 seed / 更大 instance，并接入 latency-aware O3 objective”。

## 2026-07-15 - 设计 local-to-global ablation

### 目标

- 继续推进 `target.md` 中 P0：增加 local-to-global ablation。
- 回应审稿意见中 “收益可能只是 partial sharing / fewer transmissions，而不是 LGCP hierarchy” 的问题。

### 产物

- 新增：`docs/doc_workspace/LGCP/local_to_global_ablation.md`

### 设计要点

- Ablation 必须拆开 selective sharing 与 local-to-global hierarchy。
- 最低 rebuttal 组合：
  - Full sharing baseline；
  - Confidence selective sharing without hierarchy；
  - LGCP without scheduling；
  - Full LGCP。
- Fairness 口径优先使用 same packet budget，并补充 same AP target 或 same latency budget。
- 第一阶段做 offline perception-only subset ablation；第二阶段实现真正 RSU area assignment、leader local fusion、RSU aggregation 和 scheduling。

### 结论

- 当前仓库已有 offline inference、area slicing、confidence records、greedy group selection 等基础，可以先实现 offline subset ablation。
- 完整 local-to-global hierarchy 仍依赖 LGCP 专用机制代码，因此 P0 ablation 大项仍不能关闭。

### 状态更新

- `target.md` 中 “完成 local-to-global ablation 实验设计” 已标记完成。
- 新增待办：“实现 offline selective-sharing vs LGCP area-aware subset ablation”。

## 2026-07-15 - offline selective-sharing vs area-aware subset ablation smoke

### 目标

- 继续推进 `target.md` 中 P0：local-to-global ablation 的第一阶段。
- 先用 offline perception-only 方式比较 full sharing、random selective sharing、confidence top-k 和 LGCP area-aware union。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_subset_ablation_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_offline_subset_smoke --max-frames 3 --budgets "5,10" --confidence-field density_distance
```

### 产物

- 新增脚本：`opencda/tools/lgcp_subset_ablation_eval.py`
- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_offline_subset_smoke`
- 输出文件：
  - `subset_frame_records.csv`
  - `ablation_summary.csv`
  - `config.yaml`
  - `notes.md`

### 结果

| Method | Budget | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | --- |
| full | 5 | 0.851542 | 0.828838 | 0.577169 |
| full | 10 | 0.851542 | 0.828838 | 0.577169 |
| random | 5 | 0.449647 | 0.433443 | 0.222553 |
| random | 10 | 0.557382 | 0.550719 | 0.391526 |
| confidence_topk | 5 | 0.349202 | 0.332366 | 0.234146 |
| confidence_topk | 10 | 0.610004 | 0.610004 | 0.482805 |
| area_aware_union | 5 | 0.484637 | 0.484637 | 0.350075 |
| area_aware_union | 10 | 0.678564 | 0.678564 | 0.591502 |

### 备注

- PowerShell 中 `--budgets` 需要写成 `"5,10"`，否则逗号参数会被拆分。
- 首次运行发现 `confidence_topk` 可能漏选 ego，已修正为固定保留 ego，再从非 ego CAV 中按 confidence 取 top-k。
- 本实验为 perception-only subset ablation，不模拟 leader local fusion、RSU aggregation 或 NS3 scheduling。

### 结论

- 相同 budget 下，area-aware union 在该 3 帧 smoke 中优于 confidence top-k 和 random selective sharing。
- 该结果可以作为 local-to-global ablation 的第一阶段证据，但完整 P0 大项仍需要实现 LGCP 专用 RSU area assignment、leader local fusion 和 RSU global aggregation。

### 状态更新

- `target.md` 中 “实现 offline selective-sharing vs LGCP area-aware subset ablation” 已标记完成。
- 新增待办：“扩大 offline subset ablation 到 11 帧 / 多 seed，并加入更稳定的 packet-budget 统计”。

## 2026-07-15 - offline subset ablation 扩展到完整 11 帧

### 目标

- 将 3 帧 smoke 扩展到完整 `lgcp_carla` dump 的 11 帧。
- 检查 area-aware union 相对 random / confidence top-k 的优势是否在更多帧上保持。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_subset_ablation_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_offline_subset_11f --max-frames 11 --budgets "5,10" --confidence-field density_distance
```

### 产物

- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_offline_subset_11f`
- 输出文件：
  - `subset_frame_records.csv`
  - `ablation_summary.csv`
  - `config.yaml`
  - `notes.md`

### 结果

| Method | Budget | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | --- |
| full | 5 | 0.852487 | 0.841146 | 0.527546 |
| full | 10 | 0.852487 | 0.841146 | 0.526237 |
| random | 5 | 0.331922 | 0.314671 | 0.165521 |
| random | 10 | 0.611342 | 0.598993 | 0.380538 |
| confidence_topk | 5 | 0.358450 | 0.345556 | 0.221396 |
| confidence_topk | 10 | 0.629379 | 0.624088 | 0.435861 |
| area_aware_union | 5 | 0.405018 | 0.396807 | 0.251957 |
| area_aware_union | 10 | 0.678388 | 0.676678 | 0.538273 |

### 结论

- 11 帧上，area-aware union 继续高于 confidence top-k 和 random selective sharing。
- Budget=10 时，area-aware union 的 AP@0.5 / AP@0.7 分别比 confidence top-k 高约 `0.052590` / `0.102412`。
- 这条结果比 3 帧 smoke 更适合进入 rebuttal 初稿，但仍需要多 seed 和完整 hierarchy 实现支撑最终论文结论。

### 状态更新

- `target.md` 中 “扩大 offline subset ablation 到完整 11 帧” 已标记完成。
- 原多 seed / packet-budget 任务保留为下一步。

## 2026-07-15 - offline subset ablation 增加 packet-budget / byte-volume proxy

### 目标

- 为 11 帧 offline subset ablation 增加 selected-agent、non-ego packet 和 byte-volume proxy 统计。
- 支撑 `local_to_global_ablation.md` 中 same packet budget 的公平性口径。

### 代码变更

- 更新 `opencda/tools/lgcp_subset_ablation_eval.py`：
  - 新增 `--feature-packet-bytes`，默认 `10000`；
  - `subset_frame_records.csv` 新增 `non_ego_selected_count`；
  - `ablation_summary.csv` 新增 `selected_mean`、`non_ego_selected_mean`、`non_ego_packet_total`、`byte_proxy_total`。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_subset_ablation_eval.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_subset_ablation_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_offline_subset_11f_budget_stats --max-frames 11 --budgets "5,10" --confidence-field density_distance --feature-packet-bytes 10000
```

### 产物

- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_offline_subset_11f_budget_stats`

### 结果

| Method | Budget | Non-ego packets | Byte proxy | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | --- | --- |
| full | 5 | 209 | 2090000 | 0.839868 | 0.526419 |
| full | 10 | 209 | 2090000 | 0.839868 | 0.526567 |
| random | 5 | 44 | 440000 | 0.314671 | 0.165521 |
| random | 10 | 99 | 990000 | 0.598993 | 0.380556 |
| confidence_topk | 5 | 44 | 440000 | 0.345556 | 0.221396 |
| confidence_topk | 10 | 99 | 990000 | 0.624088 | 0.435861 |
| area_aware_union | 5 | 44 | 440000 | 0.396807 | 0.251957 |
| area_aware_union | 10 | 99 | 990000 | 0.676678 | 0.538273 |

### 结论

- 同一 non-ego packet budget 下，area-aware union 继续优于 random 和 confidence top-k。
- Budget=10 使用 99 个 non-ego packet，约为 full sharing 209 个 non-ego packet 的 `47.4%`。
- 该 byte proxy 是统一口径估计，不等价于真实 feature slice 大小，后续需要结合 LGCP slicing / NS3 日志校准。

### 状态更新

- `target.md` 中 “为 offline subset ablation 加入 packet-budget / byte-volume proxy 统计” 已标记完成。
- 多 seed 扩展仍保留为下一步。

## 2026-07-15 - communication-aware top-k selective-sharing baseline

### 目标

- 推进 `target.md` 中 P0：补充更强通信感知 baseline。
- 在 offline subset ablation 中加入不使用 LGCP hierarchy 的 `comm_aware_topk`。

### 方法

`comm_aware_topk` 对每个 CAV 先聚合 area confidence，再除以到 ego 的距离成本：

```text
utility(v) = confidence(v) / (1 + distance(v, ego) / 100)
```

按 utility 选择 top-k CAV，并固定包含 ego。当前通信成本是 distance proxy，后续应替换为 NS3 / link-quality proxy。

### 代码变更

- 更新 `opencda/tools/lgcp_subset_ablation_eval.py`：
  - 从 `area_records.csv` 读取每帧每个 agent 的位置；
  - 新增 `select_comm_aware_topk()`；
  - ablation methods 增加 `comm_aware_topk`。
- 新增 `docs/doc_workspace/LGCP/communication_aware_baseline.md`。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_subset_ablation_eval.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_subset_ablation_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f --max-frames 11 --budgets "5,10" --confidence-field density_distance --feature-packet-bytes 10000
```

### 产物

- 输出目录：`docs/doc_workspace/LGCP/experiments/ablation/20260715_lgcp_carla_comm_aware_baseline_11f`

### 结果

| Method | Budget | Non-ego packets | AP@0.5 | AP@0.7 |
| --- | --- | --- | --- | --- |
| random | 5 | 44 | 0.314671 | 0.165521 |
| confidence_topk | 5 | 44 | 0.345556 | 0.221396 |
| comm_aware_topk | 5 | 44 | 0.443572 | 0.296352 |
| area_aware_union | 5 | 44 | 0.396807 | 0.251957 |
| random | 10 | 99 | 0.598993 | 0.380556 |
| confidence_topk | 10 | 99 | 0.624088 | 0.436000 |
| comm_aware_topk | 10 | 99 | 0.686146 | 0.545736 |
| area_aware_union | 10 | 99 | 0.676678 | 0.538273 |

### 结论

- `comm_aware_topk` 是比 `confidence_topk` 更强的 selective-sharing baseline。
- 在当前 11 帧 offline perception-only proxy 中，`comm_aware_topk` 还略高于 `area_aware_union`。
- 这说明论文不能只依靠当前 offline area-aware union 证明 LGCP 优于强 baseline；必须继续实现完整 hierarchy、leader local fusion、RSU aggregation 和 scheduling。

### 状态更新

- `target.md` 中 “补充更强通信感知 baseline” 已标记完成。
- 多 seed 扩展中必须包含 `comm_aware_topk`。

## 2026-07-15 - scalable perception-quality proxy 初步校准

### 目标

- 推进 `target.md` 中 P0：明确大规模 30 CAV 实验只验证 latency，或补充 scalable perception-quality proxy。
- 用现有 11 帧 offline AP 对照校准 area-confidence proxy 与真实 AP 的相关性。

### 代码变更

- 新增 `opencda/tools/lgcp_quality_proxy_eval.py`。
- 新增 `docs/doc_workspace/LGCP/large_scale_quality_proxy.md`。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_quality_proxy_eval.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_quality_proxy_eval --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --subset-frame-records docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\subset_frame_records.csv --ablation-summary docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\ablation_summary.csv --output-dir docs\doc_workspace\LGCP\experiments\large_scale_proxy\20260715_lgcp_carla_quality_proxy_11f --confidence-field density_distance
```

### 产物

- 输出目录：`docs/doc_workspace/LGCP/experiments/large_scale_proxy/20260715_lgcp_carla_quality_proxy_11f`
- 输出文件：
  - `quality_proxy_frame_records.csv`
  - `quality_proxy_summary.csv`
  - `quality_proxy_ap_joined.csv`
  - `quality_proxy_correlation.csv`
  - `config.yaml`
  - `notes.md`

### 结果

| Proxy | Quality | Samples | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| area_coverage_proxy_mean | AP@0.5 | 10 | 0.863937 | 0.841463 |
| confidence_max_proxy_mean | AP@0.5 | 10 | 0.951439 | 0.926829 |
| confidence_noisy_or_proxy_mean | AP@0.5 | 10 | 0.966055 | 0.926829 |
| confidence_noisy_or_proxy_mean | AP@0.7 | 10 | 0.954195 | 0.802435 |

### 结论

- `confidence_noisy_or_proxy_mean` 与 AP@0.5 / AP@0.7 有强正相关，可作为大规模 quality trend 的候选 proxy。
- 该结果只来自单场景 / 单 seed / 10 个 method-budget 样本，不能替代真实 AP。
- 论文大规模实验必须明确：latency 是直接指标，quality 是 calibrated proxy。

### 状态更新

- `target.md` 中 “明确大规模 30 CAV 实验只验证 latency，或补充 scalable perception-quality proxy” 已标记完成。
- 下一步应在多 seed 上复核 proxy-AP 相关性，并接入 offline NS3 / large-scale replay。

## 2026-07-15 - LGCP hierarchy control-plane plan 导出

### 目标

- 推进 P0 中完整 local-to-global hierarchy 机制。
- 先离线实现 RSU area assignment、area-task group、leader selection、member-to-leader upload 和 leader-to-RSU upload 的控制面计划导出。

### 代码变更

- 新增 `opencda/tools/lgcp_hierarchy_plan_eval.py`。
- 新增 `docs/doc_workspace/LGCP/hierarchy_pipeline.md`。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_hierarchy_plan_eval.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_plan_eval --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f --confidence-field density_distance --delta-g 0.05 --max-group-size 4 --max-areas 40 --feature-packet-bytes 10000 --leader-result-bytes 2000 --assignment-bytes 64 --broadcast-bytes 2000
```

### 产物

- 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f`
- 输出文件：
  - `area_assignment_plan.csv`
  - `upload_plan.csv`
  - `hierarchy_frame_summary.csv`
  - `hierarchy_summary.csv`
  - `config.yaml`
  - `notes.md`

### 结果

| 指标 | 数值 |
| --- | --- |
| frames | 11 |
| covered areas / frame | 40 |
| average group size | 1.536364 |
| average group confidence | 0.908059 |
| member-to-leader packets / frame | 21.454545 |
| leader-to-RSU packets / frame | 40 |
| total byte proxy / frame | 299105.454545 |
| active leaders / frame | 15.090909 |
| leader max load / frame | 8.818182 |

### 备注

- 初始全 ROI 版本覆盖约 275 个 area / frame，包含大量低价值空 area，只适合作为压力上限。
- 当前记录采用 top-40 high-priority area，更符合 LGCP 只调度重要感知区域的设定。

### 结论

- hierarchy control-plane 已有可复现 CSV 产物，能支撑 group 非传统 clustering、leader-to-RSU 上传阶段、control-plane byte proxy 等机制说明。
- 真实 feature slicing、leader local fusion、RSU global aggregation 和 NS3 scheduling 仍未完成。

### 状态更新

- `target.md` 中 “实现 LGCP 离线 RSU area assignment / leader upload plan 导出” 已标记完成。
- 完整 LGCP hierarchy 大项保持未完成。

## 2026-07-15 - LGCP upload plan 接入 offline NS3 replay dry-run

### 目标

- 将 `lgcp_hierarchy_plan_eval.py` 输出的 `upload_plan.csv` 转换为 NS3 bridge 可发送的 `transfer_requests`。
- 先用 dry-run 验证请求构建，不依赖正在运行的 ns-3。

### 代码变更

- 更新 `opencda/tools/offline_ns3_replay.py`：
  - 新增 `--lgcp-upload-plan`；
  - 新增 `--rsu-node-id`，默认 `-1`；
  - 新增 `--dry-run`；
  - LGCP 模式下使用 `upload_plan.csv` 构建 member-to-leader 和 leader-to-RSU transfer requests。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_ns3_replay.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --dry-run
```

### 结果

| Timestamp | Nodes | Requests | Bytes |
| --- | --- | --- | --- |
| 000060 | 21 | 62 | 300000 |
| 000062 | 21 | 62 | 300000 |
| 000064 | 21 | 62 | 300000 |
| 000066 | 21 | 62 | 300000 |
| 000068 | 21 | 60 | 280000 |
| 000070 | 21 | 60 | 280000 |
| 000072 | 21 | 61 | 290000 |
| 000074 | 21 | 61 | 290000 |
| 000076 | 21 | 60 | 280000 |
| 000078 | 21 | 63 | 310000 |
| 000080 | 21 | 63 | 310000 |

### 备注

- 初次 dry-run 只有 20 个节点，是因为 replay 侧误过滤了 RSU；已修正为包含 RSU `-1`。
- 当前 dry-run 不产生 latency / delivery ratio；真实结果需要启动 ns-3 后去掉 `--dry-run`。

### 状态更新

- `target.md` 中 “将 LGCP upload plan 接入 offline NS3 replay dry-run” 已标记完成。
- 新增待办：“运行 LGCP upload plan offline NS3 联机 smoke test，记录 latency / delivery”。

## 2026-07-15 - LGCP upload plan offline NS3 3 帧联机 smoke

### 目标

- 运行 LGCP upload plan 的 offline NS3 联机 smoke test。
- 验证 RSU 节点、同步、transfer requests 和 `cam_received` 回传链路。

### 过程

第一次尝试失败：

- 后台启动 ns-3 的 `Start-Process wsl.exe -ArgumentList ... bash -lc ...` quoting 错误，导致 `/bin/bash: ./ns3: No such file or directory`。
- 改为在 WSL `/tmp/lgcp_ns3_smoke.sh` 写入启动脚本，再由 Windows `Start-Process wsl.exe ... /tmp/lgcp_ns3_smoke.sh` 启动。

第二次尝试连接成功，但发现 RSU ID 问题：

- replay 使用 dump 中的 RSU `-1` 作为 NS3 target；
- ns-3 stderr 报 `(leader, -1) skipped during ProcessData_TransferRequests` 和 `invalid vehicle payload id=-1`；
- 修正 `offline_ns3_replay.py`，将 RSU 自动映射为 `max(CAV id)+1`，当前为节点 `21`。

### 代码变更

- 更新 `opencda/tools/offline_ns3_replay.py`：
  - `--rsu-node-id` 改为可选；
  - LGCP 模式默认将 RSU 映射为正整数节点 `max(CAV id)+1`；
  - vehicle position payload 中 RSU `carla_id` 同步使用该正整数节点。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_ns3_replay.py
```

### 命令

NS3：

```bash
/tmp/lgcp_ns3_smoke.sh
```

Replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --drain-seconds 0.3 --sync-timeout 10
```

### 产物

- 日志目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_smoke_3f_rsu21`
- 新增解析文件：
  - `cam_received_summary.csv`
  - `ns3_smoke_summary.csv`

### 结果

Replay：

| Timestamp | Nodes | Requests | Bytes |
| --- | --- | --- | --- |
| 000060 | 21 | 62 | 300000 |
| 000062 | 21 | 62 | 300000 |
| 000064 | 21 | 62 | 300000 |

Parsed `cam_received`:

| Metric | Value |
| --- | --- |
| cam_received_count | 5 |
| rsu_received_count | 3 |
| avg_delay_ms | 16 |
| max_delay_ms | 31 |

### 结论

- LGCP upload plan 已能进入 ns-3 并完成 3 帧同步 replay。
- RSU 正整数节点映射有效，stdout 中出现 `receiver_id=21` 的 leader-to-RSU 接收事件。
- 当前只解析成功回传的 `cam_received`，尚不能得出完整 delivery ratio。

### 状态更新

- `target.md` 中 “运行 LGCP upload plan offline NS3 联机 smoke test” 已标记完成。
- 新增待办：“扩大 LGCP upload plan offline NS3 replay 到 11 帧，并解析完整 delivery ratio / delay summary”。

## 2026-07-15 - LGCP upload plan offline NS3 11 帧联机 replay 与日志解析

### 目标

- 将 LGCP hierarchy upload plan 从 3 帧 smoke 扩展到完整 11 帧 replay。
- 解析 NS3 stdout 中的 `cam_received` 回调，得到 request-level bridge-observed delivery ratio 和 delay summary。

### 新增工具

- `opencda/tools/lgcp_ns3_log_eval.py`

功能：

- 读取 `upload_plan.csv` 的 planned requests。
- 将 `RSU` 映射为 NS3 正整数节点 `21`。
- 解析 `SendMsgToCarla: {"type":"cam_received", ...}`。
- 按 `(frame, source, target, bytes)` 对齐计划请求和可见接收事件。
- 输出：
  - `cam_received_records.csv`
  - `delivery_by_frame.csv`
  - `delivery_by_type.csv`
  - `delivery_summary.csv`

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_ns3_log_eval.py opencda\tools\offline_ns3_replay.py
```

### 命令

NS3：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- /tmp/lgcp_ns3_smoke.sh
```

Replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10
```

Log parsing：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_smoke_11f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_smoke_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

### 产物

- 日志与结果目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_smoke_11f_rsu21`

### 结果

Replay request summary：

| Timestamp | Nodes | Requests | Bytes |
| --- | --- | --- | --- |
| 000060 | 21 | 62 | 300000 |
| 000062 | 21 | 62 | 300000 |
| 000064 | 21 | 62 | 300000 |
| 000066 | 21 | 62 | 300000 |
| 000068 | 21 | 60 | 280000 |
| 000070 | 21 | 60 | 280000 |
| 000072 | 21 | 61 | 290000 |
| 000074 | 21 | 61 | 290000 |
| 000076 | 21 | 60 | 280000 |
| 000078 | 21 | 63 | 310000 |
| 000080 | 21 | 63 | 310000 |

Bridge-observed summary：

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| observed_cam_received | 31 |
| matched_cam_received | 31 |
| bridge_observed_delivery_ratio | 0.045858 |
| planned_bytes | 3240000 |
| observed_bytes | 174000 |
| avg_delay_ms | 109.645 |
| p95_delay_ms | 209 |
| max_delay_ms | 211 |

By upload type：

| Upload type | Planned | Observed | Bridge-observed ratio | Avg delay ms | P95 delay ms |
| --- | --- | --- | --- | --- | --- |
| leader_to_rsu | 440 | 17 | 0.038636 | 75.471 | 115 |
| member_to_leader | 236 | 14 | 0.059322 | 151.143 | 209 |

### 观察

- 11 帧 replay 能稳定完成同步，最终 `offline_ns3_replay completed frames=11 final_sync_time=1.300`。
- NS3 stdout 中存在大量 `PSCCH_DECODE_FAIL` 和 `reason=error_model`，当前 V2X 参数下 bridge 可见交付率很低。
- 该结果是 `cam_received` 回调可见的保守统计，不等同于完整 PHY/RLC 层 delivery ratio。

### 状态更新

- `target.md` 中 11 帧 LGCP upload plan offline NS3 replay 已标记完成，口径限定为 bridge-observed request-level summary。
- 新增后续待办：接入 ns-3 PHY/RLC trace，补充严格链路层 delivery ratio / decode-failure breakdown。

## 2026-07-15 - NS3 PHY decode breakdown 解析

### 目标

- 在 11 帧 LGCP upload plan replay 的基础上，补充 PHY 层 decode-failure breakdown。
- 区分 bridge-observed `cam_received` 和 ns-3 PHY decode diagnostics，避免把低层解码统计误写成端到端 request delivery。

### 代码变更

- 扩展 `opencda/tools/lgcp_ns3_log_eval.py`：
  - 解析 `PSCCH_DECODE_OK` / `PSCCH_DECODE_FAIL` 两行式日志；
  - 解析 `PSSCH_DECODE_OK` / `PSSCH_DECODE_FAIL` 单行日志；
  - 输出 `phy_decode_events.csv` 和 `phy_decode_summary.csv`；
  - 保留原有 `cam_received` / delivery summary 输出。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_ns3_log_eval.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_smoke_11f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_smoke_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

### 产物

- `phy_decode_events.csv`
- `phy_decode_summary.csv`

### 结果

总计：

| Metric | Value |
| --- | --- |
| PHY decode events | 12740 |
| PHY decode failures | 7779 |
| PSCCH OK | 4709 |
| PSCCH FAIL | 7491 |
| PSSCH OK | 252 |
| PSSCH FAIL | 288 |

Breakdown：

| Channel | Status | Reason | Count | Channel ratio | Avg SINR | Avg TBLER |
| --- | --- | --- | --- | --- | --- | --- |
| PSCCH | FAIL | decoded_overlap | 5736 | 0.470164 | 0.171793 | 0.981760 |
| PSCCH | FAIL | error_model | 1755 | 0.143852 | 0.308541 | 0.953855 |
| PSCCH | OK | - | 4709 | 0.385984 | 52207.608824 | 0.017849 |
| PSSCH | FAIL | decode_fail | 288 | 0.533333 | 3581.433854 | 0.995913 |
| PSSCH | OK | - | 252 | 0.466667 | 373591.447580 | 0.000912 |

### 结论

- 当前日志已足以支撑 PHY decode-failure breakdown：PSCCH 失败主要来自 `decoded_overlap`，其次是 `error_model`。
- PSSCH decode 成功率约 46.7%，但 PSSCH 日志目前没有稳定的 failure reason 字段，只能归为 `decode_fail`。
- 这还不是严格 request-level delivery ratio；下一步需要 ns-3 侧输出 RLC / application request id trace，才能把 PHY/RLC 事件映射回 LGCP `upload_plan.csv` 中的每个 request。

### 状态更新

- `target.md` 中 “接入 ns-3 PHY decode trace，补充 decode-failure breakdown” 已标记完成。
- 新增后续待办：“接入 ns-3 RLC / request-id trace，将 PHY/RLC 事件映射回 LGCP upload request”。

## 2026-07-15 - NS3 request-id trace 与 LGCP upload request 精确映射

### 目标

- 避免仅靠 `(frame, source, target, bytes)` 推断 `cam_received` 对应的 LGCP request。
- 将 OpenCDA replay 发送的 `pkt_id` 透传到 ns-3 CAM header，并在 `cam_received` 中回传 `request_id`。

### 代码变更

ns-3 co-simulation 仓库：

- `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\geo-networking.h`
- `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\geo-networking.cc`
- `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.h`
- `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.cc`
- `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc`

修改内容：

- `CamHeader` 新增 `request_id` 字段。
- `ProcessData_TransferRequests()` 将 `pkt_id` 传入 `ScheduleCam()`。
- `CamSenderDSRC` / `CamSenderNR` 写入 `request_id`。
- `CamReceiverDSRC` / `CamReceiverNR` 在 `cam_received` JSON 中回传 `request_id`。

OpenCDA 仓库：

- `opencda/tools/lgcp_ns3_log_eval.py`

修改内容：

- `cam_received_records.csv` 新增 `request_id` / `pkt_id` / `match_method`。
- 有 `request_id` 时优先按 `(frame_index, request_id)` 精确匹配 `upload_plan.csv`；旧日志无 `request_id` 时回退到 `(frame, source, target, bytes)`。

### 验证

ns-3 build：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 build"
```

OpenCDA parser：

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_ns3_log_eval.py
```

### 11 帧 replay

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_request_id_11f_rsu21/
```

Replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10
```

Parse：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_request_id_11f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_request_id_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

### 结果

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| observed_cam_received | 31 |
| matched_cam_received | 31 |
| match_method | `frame_request_id` for all 31 |
| bridge_observed_delivery_ratio | 0.045858 |
| avg_delay_ms | 109.645 |
| p95_delay_ms | 209 |
| max_delay_ms | 211 |

按阶段：

| Upload type | Planned | Observed | Bridge-observed ratio | Avg delay ms |
| --- | --- | --- | --- | --- |
| leader_to_rsu | 440 | 17 | 0.038636 | 75.471 |
| member_to_leader | 236 | 14 | 0.059322 | 151.143 |

PHY decode breakdown 与上一轮一致：

| Channel | Status | Reason | Count |
| --- | --- | --- | --- |
| PSCCH | FAIL | decoded_overlap | 5736 |
| PSCCH | FAIL | error_model | 1755 |
| PSCCH | OK | - | 4709 |
| PSSCH | FAIL | decode_fail | 288 |
| PSSCH | OK | - | 252 |

### 结论

- Application-level `cam_received` 已能精确映射到 `upload_plan.csv` 中的每帧 request。
- 当前仍不是 RLC-level trace；RLC / PHY event 到 request id 的严格映射需要 ns-3 侧继续输出 RLC PDU / HARQ / request id 关联。

### 状态更新

- `target.md` 中 “接入 ns-3 application request-id trace，将 `cam_received` 精确映射回 LGCP upload request” 已标记完成。
- 后续待办收窄为：“接入 ns-3 RLC trace，将 RLC / PHY 事件进一步映射回 LGCP upload request”。

## 2026-07-15 - NS3 RLC request-id trace 与 11 帧 replay

### 目标

- 将 request id 从 CAM application 继续透传到 NR SL RLC TX / RX / DROP 日志。
- 生成 RLC event 到 LGCP `upload_plan.csv` request 的映射表。

### 代码变更

ns-3 co-simulation / ns-3：

- 新增 `LteRlcRequestIdTag`：
  - `C:\Workspace\carla-ns3-co-simulation\ns-3-dev\src\lte\model\lte-rlc-request-id-tag.h`
  - `C:\Workspace\carla-ns3-co-simulation\ns-3-dev\src\lte\model\lte-rlc-request-id-tag.cc`
  - `C:\Workspace\carla-ns3-co-simulation\ns3\src\lte-model\lte-rlc-request-id-tag.h`
  - `C:\Workspace\carla-ns3-co-simulation\ns3\src\lte-model\lte-rlc-request-id-tag.cc`
- 更新 `C:\Workspace\carla-ns3-co-simulation\ns-3-dev\src\lte\CMakeLists.txt`，纳入新增 tag 源和头文件。
- 更新 `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.cc`：
  - CAM packet 同时添加 `LteRlcRequestIdTag` PacketTag 和 ByteTag。
- 更新 `C:\Workspace\carla-ns3-co-simulation\ns3\src\lte-model\lte-rlc-um.cc`：
  - `[NRSL_RLC_TX]` / `[NRSL_RLC_RX]` / `[NRSL_RLC_DROP]` 输出 `request_id`。

OpenCDA：

- 更新 `opencda/tools/lgcp_ns3_log_eval.py`：
  - 解析 `[NRSL_RLC_TX/RX/DROP]`；
  - 输出 `rlc_events.csv`、`rlc_summary.csv`、`rlc_by_request.csv`；
  - 按 `(frame_index, request_id)` 映射回 `upload_plan.csv`。

### 验证

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 build"
conda run -n opencda python -m py_compile opencda\tools\lgcp_ns3_log_eval.py
```

### 11 帧 replay

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/ns3_rlc_request_id_11f_rsu21/
```

解析命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_rlc_request_id_11f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_rlc_request_id_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

### 结果

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| RLC TX events | 1131 |
| RLC RX events | 252 |
| RLC DROP events | 0 |
| matched RLC TX events | 1131 |
| matched RLC RX events | 252 |
| unique TX requests | 614 |
| unique RX requests | 164 |
| RLC request RX ratio | 0.242604 |

同步保留 application callback 结果：

| Metric | Value |
| --- | --- |
| observed `cam_received` | 31 |
| bridge-observed delivery ratio | 0.045858 |
| avg delay ms | 109.645 |
| p95 delay ms | 209 |

### 结论

- RLC TX / RX events 已能通过 request id 映射回 LGCP upload request。
- RLC 层观察到 164/676 个 request 至少有 RX event，高于 application callback 层 31/676，说明很多 request 到达 RLC 但未形成最终 `cam_received`。
- PHY decode events 仍未逐条绑定 request id；下一步如需完整 HARQ / PHY request-level breakdown，需要继续把 request id 透传到 PHY TB / HARQ trace。

### 状态更新

- `target.md` 中 “接入 ns-3 RLC trace，将 RLC events 进一步映射回 LGCP upload request” 已标记完成。
- 新增后续待办：“将 PHY decode events / HARQ feedback 进一步绑定到 LGCP upload request”。

## 2026-07-16 - NS3 PHY / HARQ request-level trace 设计

### 目标

- 在 application request-id trace 和 RLC request-id trace 已完成的基础上，明确 PHY / HARQ request-level trace 的下一步落点。
- 避免把当前 aggregate PSCCH / PSSCH decode diagnostics 误写成 request-level failure attribution。

### 文档变更

新增：

```text
docs/doc_workspace/LGCP/ns3_phy_harq_request_trace.md
```

同步更新：

- `readme.md`
- `target.md`
- `status.md`

### 设计结论

PHY / HARQ trace 后续应输出 request-aware event：

```text
[NRSL_PHY_EVENT] event=PSSCH_DECODE_FAIL time_s=... frame_index=... request_id=... sender_l2_id=... receiver_l2_id=... harq_id=... subchannel_start=... subchannel_num=... reason=...
```

推荐实现顺序：

1. 确认 `LteRlcRequestIdTag` 作为 ByteTag 是否能从 RLC 继续传播到 MAC / PHY。
2. 先记录 PHY schedule event，形成 request id 到 TB / slot / subchannel / HARQ process 的桥。
3. 优先绑定 PSSCH decode event；PSCCH event 若不天然带 request id，可通过 schedule event 回填。
4. HARQ ACK / NACK / timeout 通过 `(sender_l2_id, receiver_l2_id, harq_id, slot)` 回连到 scheduled request。

### 当前论文边界

- 可以报告 request-level RLC delivery。
- 可以报告 aggregate PHY decode failure breakdown。
- 尚不能声称每个 LGCP upload request 的 PHY / HARQ 失败原因已经可归因。

## 2026-07-16 - OpenCDA parser 接入 request lifecycle funnel

### 目标

- 先完成 OpenCDA 侧 parser，使其能够消费未来 ns-3 输出的 request-level PHY / HARQ event。
- 在现有 11 帧 RLC request-id replay 上生成 request lifecycle funnel，复查 planned / RLC / application 三层状态。

### 代码变更

更新：

```text
opencda/tools/lgcp_ns3_log_eval.py
```

新增解析能力：

- `[NRSL_PHY_EVENT] ... request_id=...`
- `[NRSL_HARQ_EVENT] ... request_id=...`
- `request_ids=1,2,3` 多 request TB 展开

新增输出：

- `phy_harq_request_events.csv`
- `request_lifecycle.csv`
- `request_lifecycle_summary.csv`

### 回归命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_ns3_log_eval.py
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_rlc_request_id_11f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_rlc_request_id_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

### 回归结果

```text
planned_requests=676 observed_cam_received=31 bridge_observed_delivery_ratio=0.045858 avg_delay_ms=109.645 p95_delay_ms=209.000 max_delay_ms=211.000 phy_decode_events=12740 phy_decode_failures=7779 phy_harq_request_events=0 rlc_tx_events=1131 rlc_rx_events=252
```

`request_lifecycle_summary.csv`：

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| requests_with_rlc_tx | 614 |
| requests_with_rlc_rx | 164 |
| requests_with_cam_received | 31 |
| terminal_application_received | 31 |
| terminal_rlc_rx_only | 133 |
| terminal_rlc_tx_no_rx | 450 |
| terminal_planned_only | 62 |

### 结论

- OpenCDA 侧 lifecycle funnel 已就绪。
- 当前 ns-3 日志尚无 request-level PHY / HARQ event，因此 `phy_harq_request_events=0` 是预期结果。
- 下一步应改 ns-3 侧 PHY schedule / PSSCH / HARQ 输出，而不是继续改 OpenCDA parser。

## 2026-07-16 - ns-3 PSSCH request-level trace smoke

### 目标

- 在 ns-3 PSSCH decode OK/FAIL 处读取 `LteRlcRequestIdTag`。
- 输出 `[NRSL_PHY_EVENT]`，让 OpenCDA parser 能将 PSSCH decode event 映射回 LGCP upload request。

### 代码变更

ns-3 co-simulation：

```text
C:\Workspace\carla-ns3-co-simulation\ns3\src\nr-spectrum-phy.cc
```

新增：

- `CollectRequestIds()`
- `PrintRequestPhyEvent()`
- `PrintRequestHarqEvent()`

当前实际可观测：

- `[NRSL_PHY_EVENT] event=PSSCH_DECODE_OK request_ids=...`
- `[NRSL_PHY_EVENT] event=PSSCH_DECODE_FAIL request_ids=...`

HARQ ACK/NACK 输出代码已接入 feedback 分支，但本轮 3 帧 smoke 未观测到 `[NRSL_HARQ_EVENT]`。

### 验证

ns-3 build：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 build"
```

3 帧 replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10
```

解析：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_phy_harq_request_3f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_phy_harq_request_3f_rsu21 --rsu-node-id 21 --max-frames 3
```

### 结果

```text
planned_requests=186 observed_cam_received=5 bridge_observed_delivery_ratio=0.026882 avg_delay_ms=16.000 p95_delay_ms=31.000 max_delay_ms=31.000 phy_decode_events=2619 phy_decode_failures=1653 phy_harq_request_events=124 rlc_tx_events=228 rlc_rx_events=41
```

`request_lifecycle_summary.csv`：

| Metric | Value |
| --- | --- |
| planned_requests | 186 |
| requests_with_rlc_tx | 124 |
| requests_with_pssch_ok | 31 |
| requests_with_pssch_fail | 68 |
| requests_with_rlc_rx | 31 |
| requests_with_cam_received | 5 |
| terminal_application_received | 5 |
| terminal_rlc_rx_only | 26 |
| terminal_pssch_fail | 50 |
| terminal_rlc_tx_no_rx | 43 |
| terminal_planned_only | 62 |

### 结论

- PSSCH decode OK/FAIL 已完成 request-level attribution。
- 124 条 request-level PHY event 全部成功匹配到 `upload_plan.csv`。
- HARQ feedback 尚未在本轮 smoke 中出现，后续需要确认 HARQ enable / PSFCH / feedback callback 触发条件。

## 2026-07-16 - ns-3 HARQ request-level trace smoke

### 目标

- 让 ns-3 replay 可显式启用 sidelink HARQ / PSFCH。
- 验证 `[NRSL_HARQ_EVENT]` 能携带 request id 并映射回 LGCP upload request。

### 代码变更

ns-3 co-simulation：

```text
C:\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc
```

新增命令行参数：

- `--enableSlHarq`
- `--psfchPeriod`

默认值保持原行为：

```text
--enableSlHarq=false --psfchPeriod=0
```

HARQ smoke 使用：

```text
--enableSlHarq=true --psfchPeriod=4
```

### 关键发现

- `slInfo.m_harqEnabled` 此前固定为 `false`。
- `NrSlCommResourcePoolFactory` 的 PSFCH period 默认也是 `0`，即 PSFCH disabled。
- 因此此前 PSSCH request-level trace 能出现，但 HARQ ACK/NACK 不会出现。

### 结果

```text
planned_requests=186 observed_cam_received=5 bridge_observed_delivery_ratio=0.026882 avg_delay_ms=15.800 p95_delay_ms=31.000 max_delay_ms=31.000 phy_decode_events=2622 phy_decode_failures=1660 phy_harq_request_events=233 rlc_tx_events=228 rlc_rx_events=40
```

`phy_harq_request_events.csv` event counts：

| Event | Count |
| --- | --- |
| PSSCH_DECODE_OK | 40 |
| PSSCH_DECODE_FAIL | 85 |
| HARQ_ACK | 40 |
| HARQ_NACK | 68 |

`request_lifecycle_summary.csv`：

| Metric | Value |
| --- | --- |
| planned_requests | 186 |
| requests_with_rlc_tx | 124 |
| requests_with_pssch_ok | 30 |
| requests_with_pssch_fail | 66 |
| requests_with_harq_ack | 30 |
| requests_with_harq_nack | 49 |
| requests_with_rlc_rx | 30 |
| requests_with_cam_received | 5 |
| terminal_application_received | 5 |
| terminal_rlc_rx_only | 25 |
| terminal_pssch_fail | 50 |
| terminal_rlc_tx_no_rx | 44 |
| terminal_planned_only | 62 |

### 结论

- HARQ ACK/NACK 已完成 request-level attribution。
- 当前 trace 链路已经覆盖 planned request -> RLC TX/RX -> PSSCH OK/FAIL -> HARQ ACK/NACK -> application callback。
- 下一步应扩展到 11 帧 replay，并按 `upload_type` / area / leader 分解失败原因。

## 2026-07-16 - 11 帧 PSSCH / HARQ request-level trace

### 目标

- 将 3 帧 HARQ smoke 扩展到完整 11 帧 LGCP replay。
- 形成 planned -> RLC -> PSSCH -> HARQ -> application callback 的 request-level funnel。

### 运行配置

ns-3：

```text
./build/scratch/vanet/ns3.42-main-default --simTime=1.6 --enableTimeSync=true --carlaHost=auto --enableSlHarq=true --psfchPeriod=4
```

OpenCDA replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10
```

解析：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_harq_request_11f_rsu21\ns3_stdout.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\ns3_harq_request_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

### 结果

```text
planned_requests=676 observed_cam_received=29 bridge_observed_delivery_ratio=0.042899 avg_delay_ms=108.276 p95_delay_ms=209.000 max_delay_ms=211.000 phy_decode_events=12736 phy_decode_failures=7789 phy_harq_request_events=1177 rlc_tx_events=1131 rlc_rx_events=251
```

`phy_harq_request_events.csv` event counts：

| Event | Count |
| --- | --- |
| PSSCH_DECODE_OK | 251 |
| PSSCH_DECODE_FAIL | 390 |
| HARQ_ACK | 251 |
| HARQ_NACK | 285 |

`request_lifecycle_summary.csv`：

| Metric | Value |
| --- | --- |
| planned_requests | 676 |
| requests_with_rlc_tx | 614 |
| requests_with_pssch_ok | 167 |
| requests_with_pssch_fail | 316 |
| requests_with_harq_ack | 167 |
| requests_with_harq_nack | 224 |
| requests_with_rlc_rx | 167 |
| requests_with_cam_received | 29 |
| terminal_application_received | 29 |
| terminal_rlc_rx_only | 138 |
| terminal_pssch_fail | 222 |
| terminal_rlc_tx_no_rx | 225 |
| terminal_planned_only | 62 |

### 结论

- 11 帧 request-level RLC / PSSCH / HARQ funnel 已完成。
- PSSCH OK 与 HARQ ACK 事件数一致，均为 251；request 级均为 167。
- Application callback 明显低于 RLC/PSSCH/HARQ 成功层，论文中应避免把 `cam_received` 直接当作链路层 delivery ratio。

## 2026-07-16 - control-plane overhead breakdown

### 目标

- 回应 P1 中 “显式统计 control-plane overhead：location、direction、confidence、assignment、global view”。
- 在已有 11 帧 hierarchy control-plane plan 上输出可复现的 per-frame / summary CSV。

### 代码

新增：

```text
opencda/tools/lgcp_control_overhead_eval.py
docs/doc_workspace/LGCP/control_plane_overhead.md
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_control_overhead_eval --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\area_assignment_plan.csv --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --hierarchy-frame-summary docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\hierarchy_frame_summary.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\control_overhead_11f
```

### 结果

```text
frames=11 control_plane_bytes_mean=30957.090909 planned_data_bytes_mean=294545.454545 control_plane_ratio_mean=0.095202
```

输出目录：

```text
docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260715_lgcp_carla_hierarchy_plan_top40_11f/control_overhead_11f/
```

### 结论

- 当前 20 CAV / top-40 area / 11 frame 设置下，控制面平均约 30.96 KB/frame。
- 主要控制面来源是 area-confidence report，平均约 25.76 KB/frame。
- 控制面占 planned data + control 总量约 9.52%，可作为 rebuttal 中 control-plane overhead 的初步量化证据。

## 2026-07-16 - workflow figure 与 area-task group 机制草稿

### 目标

- 推进 P2 写作 / 机制完善项。
- 回应 group 被误解为传统 clustering、workflow figure 缺失、packet granularity 不清楚和 leader-to-RSU upload 可靠性不足的问题。

### 产物

新增：

```text
docs/doc_workspace/LGCP/workflow_and_group_semantics.md
```

该文档包含：

- LGCP workflow Mermaid figure draft；
- 三层 figure 布局建议：confidence report、area-task group、local-to-global fusion；
- area-task group vs traditional vehicle clustering 对照表；
- packet 粒度 `(frame_id, source_cav_id, target_id, area_id, upload_stage)`；
- adjacent-area batching、feature cache、shared backbone feature reuse 的去重复用机制；
- leader-to-RSU upload 优先级、失败检测和 fallback。

### 结论

- `target.md` 中 workflow figure、group 概念、packet 粒度/去重、leader-to-RSU 上传策略四个 P2 项已形成可直接转写论文的机制草稿。
- 真实 feature slicing、leader local fusion 和 RSU global aggregation 仍未实现，因此文档中已明确当前实现边界。

## 2026-07-16 - deployment assumptions / limitations 草稿

### 目标

- 推进 P2 中 deployment assumptions / limitations 小节。
- 回应 RSU centralization、mobility、multi-RSU、stale information 和 failure modes 相关审稿风险。

### 产物

新增：

```text
docs/doc_workspace/LGCP/deployment_assumptions.md
```

该文档包含：

- RSU-assisted road segment、periodic CAV state report、shared spatial reference、bounded mobility、RSU compute availability 等核心假设；
- RSU centralization 的保守论文表述；
- mobility / stale assignment 来源与 mitigation；
- localization error 对 area assignment / feature slicing 的影响和后续敏感性实验建议；
- member loss、leader failure、leader-to-RSU loss、RSU overload、RSU outage 等 failure mode 表；
- multi-RSU scaling 的边界区域交换和 handover 口径；
- 大规模实验只报告 latency / proxy 时的 claim boundary。

### 结论

- `target.md` 中 deployment assumptions / limitations P2 项已标记完成。
- 该文档是论文 discussion / limitation 草稿，不替代后续 P1 sensitivity experiments。

## 2026-07-16 - stage 编号与 heuristic / approximate 表述审计

### 目标

- 推进 P2 中 stage 编号一致性和 heuristic / approximate 表述边界。
- 避免正文暗示未证明的 approximation guarantee。

### 输入

论文源文件：

```text
C:\Workspace\icdcs-paper\LGCP\conference_101719.tex
```

关键词检查：

```powershell
rg -n -i "stage|fifth|heuristic|approx|optimal|guarantee|greedy" C:\Workspace\icdcs-paper\LGCP\conference_101719.tex
```

### 观察

- `conference_101719.tex:268-292` 中 latency section 写了 first / second / third stage，但 global-view broadcast 被写成 `fifth stage`。
- 同一段公式使用 `\sum_{k=1}^{4} t_i` / `\sum_{i=1}^{4} t_i` 的四阶段 latency，因此应改成 fourth stage，并建议用 `k` 避免和 area index `i` 混淆。
- `conference_101719.tex:355` 中 `derive an approximate solution` 容易被理解为有 approximation guarantee，应改成 `derive an efficient heuristic solution`。
- `conference_101719.tex:491` 中 `two-stage process` 指算法模块，不是 latency stage，建议改成 `two algorithmic modules`。

### 产物

新增：

```text
docs/doc_workspace/LGCP/manuscript_language_audit.md
```

该文档包含：

- TeX line-level audit；
- 四阶段 latency 命名建议；
- `heuristic / approximate / optimal` 推荐和禁用表述；
- 可直接写进 Algorithm / Evaluation / Rebuttal 的英文段落；
- 论文编辑 checklist。

### 结论

- `target.md` 中 stage 编号一致性和 heuristic / approximate 表述两项 P2 已标记完成。
- 本次只沉淀论文修改建议，没有直接修改 `C:\Workspace\icdcs-paper\LGCP\conference_101719.tex`。

## 2026-07-16 - grid / area size sensitivity smoke

### 目标

- 推进 P1 中 area size / grid size sensitivity。
- 不修改 `lgcp_carla.yaml`，只在离线 area confidence 评估中覆盖 grid size。

### 代码

`opencda/tools/lgcp_area_confidence_eval.py` 新增：

```text
--grid-size-x
--grid-size-y
```

### 运行命令

示例：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_area_confidence_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --max-frames 11 --with-inference --fusion-method early --ego-cav-id 1 --grid-size-x 10 --grid-size-y 6 --output-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260716_lgcp_carla_grid_sensitivity_10x6_11f
```

三组 grid：

- `5m x 3m`
- `10m x 6m`
- `20m x 12m`

### 结果

汇总文件：

```text
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_grid_sensitivity_summary.csv
```

| Grid size | Records | Active areas | Area-frame noisy-or vs recall@0.5 Spearman | Area-acc noisy-or vs AP@0.5 Spearman |
| --- | ---: | ---: | ---: | ---: |
| `5m x 3m` | 46993 | 1101 | 0.475952 | 0.213836 |
| `10m x 6m` | 21418 | 337 | 0.570407 | 0.411840 |
| `20m x 12m` | 8386 | 94 | 0.458975 | 0.233766 |

### 结论

- 当前单场景 11 帧 smoke 支持 `10m x 6m` 作为默认 grid：area-frame noisy-or confidence vs recall@0.5 Spearman 最高。
- 细网格样本更多但 AP 排序变弱，粗网格 active areas 和 AP samples 更少。
- `target.md` 中 area size / grid size sensitivity 已标记为单场景 smoke 完成；最终论文仍应扩展多 seed。

## 2026-07-16 - localization error sensitivity smoke

### 目标

- 推进 P1 中 localization error sensitivity。
- 在不重新运行 CARLA 的前提下，对 CAV confidence report 注入 xy pose noise，观察 area-confidence ranking 是否退化。

### 代码

`opencda/tools/lgcp_area_confidence_eval.py` 新增：

```text
--localization-noise-std
--localization-noise-seed
```

噪声按 `(seed, timestamp, agent_id)` deterministic 生成。

### 运行命令

示例：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_area_confidence_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --max-frames 11 --with-inference --fusion-method early --ego-cav-id 1 --localization-noise-std 0.5 --localization-noise-seed 7 --output-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260716_lgcp_carla_localization_noise_0p5m_11f
```

### 结果

汇总文件：

```text
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_localization_noise_summary.csv
```

| Noise std | Area-frame noisy-or vs recall@0.5 Spearman | Area-acc noisy-or vs AP@0.5 Spearman |
| --- | ---: | ---: |
| `0.0m` | 0.570407 | 0.411840 |
| `0.2m` | 0.564515 | 0.411840 |
| `0.5m` | 0.546341 | 0.411840 |
| `1.0m` | 0.550885 | 0.314543 |

### 结论

- 当前单场景 11 帧 smoke 显示 area-frame confidence-to-recall ranking 对 0.2m-1.0m xy noise 相对稳定。
- accumulated AP ranking 在 1.0m 下开始下降。
- `target.md` 中 localization error sensitivity 已标记为单场景 smoke 完成；完整论文结论仍需多 seed 和真实 feature alignment 误差验证。

## 2026-07-16 - update frequency / stale assignment sensitivity smoke

### 目标

- 推进 P1 中 vehicle mobility / update frequency / stale assignment sensitivity。
- 用前若干 frame 的 area confidence 预测当前 frame quality，近似更低 update frequency 或 stale RSU assignment。

### 代码

新增：

```text
opencda/tools/lgcp_stale_assignment_eval.py
docs/doc_workspace/LGCP/stale_assignment_sensitivity.md
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_stale_assignment_eval --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260716_lgcp_carla_stale_assignment_11f --confidence-field density_distance --quality-field recall_05 --lags "0,1,2,3" --top-k 40
```

### 结果

```text
lag=0 samples=354 noisy_or_spearman=0.584992 top_jaccard_mean=1.000000
lag=1 samples=321 noisy_or_spearman=0.527720 top_jaccard_mean=0.911095
lag=2 samples=289 noisy_or_spearman=0.529556 top_jaccard_mean=0.857818
lag=3 samples=257 noisy_or_spearman=0.447925 top_jaccard_mean=0.805484
```

输出目录：

```text
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_stale_assignment_11f/
```

### 结论

- 1/2 帧 stale assignment 在当前 11 帧 smoke 中仍能保留较稳定 ranking。
- 3 帧 stale assignment 开始明显退化，支持短 TTL 或 event-driven reassignment。
- `target.md` 中 update frequency / stale assignment sensitivity 已标记为单场景 smoke 完成；显式车辆速度变化仍需多场景实验。

## 2026-07-16 - subchannel count Z sensitivity proxy

### 目标

- 推进 P1 中 subchannel count `Z` sensitivity。
- 先用 LGCP `upload_plan.csv` 做 scheduling-capacity proxy，避免立即重跑多组 NS3。

### 代码

新增：

```text
opencda/tools/lgcp_subchannel_sensitivity_eval.py
docs/doc_workspace/LGCP/subchannel_sensitivity.md
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_subchannel_sensitivity_eval --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260716_lgcp_carla_subchannel_sensitivity_top40_11f --z-values "5,10,15,20"
```

语法验证：

```powershell
conda run -n opencda python -c "compile(open(r'opencda\tools\lgcp_subchannel_sensitivity_eval.py', encoding='utf-8').read(), r'opencda\tools\lgcp_subchannel_sensitivity_eval.py', 'exec')"
```

`py_compile` 在当前 Windows 工作区写 `__pycache__` 时遇到权限拒绝，因此使用不写 `.pyc` 的 `compile()` 验证。

### 结果

| Z | Mean slots / frame | Max slots / frame | Mean max stage packets / subchannel |
| ---: | ---: | ---: | ---: |
| 5 | 12.727273 | 13.000000 | 8.000000 |
| 10 | 6.727273 | 7.000000 | 4.000000 |
| 15 | 5.000000 | 5.000000 | 2.666667 |
| 20 | 3.727273 | 4.000000 | 2.000000 |

### 结论

- `Z` 从 5 增至 20 时，slot proxy 下降约 70.7%。
- 该结果支持“subchannel 数不足会放大 PSCCH overlap / scheduling pressure”的解释。
- `target.md` 中 subchannel count sensitivity 已标记为 proxy 完成；NS3 多 Z replay 仍需后续复核。

## 2026-07-17 - CAV / edge computation capacity sensitivity proxy

### 目标

- 推进 P1 中 CAV / edge computation capacity sensitivity。
- 使用已有 hierarchy plan 估算 local fusion 与 RSU aggregation compute latency proxy。

### 代码

新增：

```text
opencda/tools/lgcp_compute_capacity_eval.py
docs/doc_workspace/LGCP/compute_capacity_sensitivity.md
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_compute_capacity_eval --hierarchy-frame-summary docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\hierarchy_frame_summary.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_compute_capacity_top40_11f --cav-capacities "2,4,8,16" --rsu-capacities "10,20,40,80"
```

语法验证：

```powershell
conda run -n opencda python -c "compile(open(r'opencda\tools\lgcp_compute_capacity_eval.py', encoding='utf-8').read(), r'opencda\tools\lgcp_compute_capacity_eval.py', 'exec')"
```

### 结果

| CAV capacity | RSU capacity | Compute mean ms | Compute max ms | CAV bottleneck ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 8.409091 | 9.000000 | 0.909091 |
| 4 | 20 | 4.204545 | 4.500000 | 0.909091 |
| 8 | 40 | 2.102273 | 2.250000 | 0.909091 |
| 16 | 80 | 1.051136 | 1.125000 | 0.909091 |

### 结论

- 均衡提升 CAV / RSU capacity 时，compute latency proxy 从 8.41ms 降到 1.05ms。
- 单独提高 CAV 或 RSU capacity 会暴露另一侧瓶颈。
- `target.md` 中 CAV / edge computation capacity sensitivity 已标记为 proxy 完成；真实 runtime 仍需完整 local fusion / RSU aggregation 实现。

## 2026-07-17 - Fig. 7 axis audit and low-density latency explanation

### 目标

- 推进 `target.md` 中 P1：解释低车数场景下 LGCP 与 baseline latency 接近的问题。
- 检查 Fig. 7 axis label 问题，并形成修稿动作。

### 核查命令

```powershell
& "C:\Users\sakakibara\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe" -png -singlefile -r 220 "C:\Workspace\icdcs-paper\LGCP\picture\num_latency_v2.pdf" "C:\Workspace\OpenCDA\docs\doc_workspace\LGCP\fig7_latency_audit"
```

### 结果

- 新增 `docs/doc_workspace/LGCP/latency_figure_audit.md`。
- Fig. 7 y-axis 当前为 `End-to-end latency (ms)`，语义正确。
- Fig. 7 x-axis 当前为 `Number of vehicles`，与 caption / 正文中的 CAV 数不完全一致，应在重导出图源时改为 `Number of CAVs`。
- 低 CAV 数 latency 接近的解释已整理为论文段落和 rebuttal wording：低密度下冗余与冲突少，LGCP 固定控制面开销占比高，edge-assisted baseline 的边缘算力优势尚未被集中式瓶颈抵消。

### 结论

- `target.md` 中两个 P1 写作/图表项已标记完成。
- 后续若进入论文源文件修改，需要用原始绘图源重导出 `num_latency_v2.pdf`。

## 2026-07-17 - OpenCOOD OPV2V / V2XSet evaluation entry audit

### 目标

- 推进 `target.md` 中“确认 OpenCOOD 中 OPV2V / V2XSet 的模型评估入口和日志格式”。
- 为后续多 seed area confidence / subset ablation 复用 OpenCOOD 输出做准备。

### 核查文件

```text
opencood/opencood/tools/inference.py
opencood/opencood/tools/inference_utils.py
opencood/opencood/utils/eval_utils.py
opencood/opencood/hypes_yaml/yaml_utils.py
opencood/opencood/data_utils/datasets/__init__.py
opencood/README.md
```

### 结论

- 主评估入口是 `opencood/tools/inference.py`。
- 标准命令为 `python opencood/tools/inference.py --model_dir <CHECKPOINT_DIR> --fusion_method <late|early|intermediate>`。
- `--model_dir` 会自动读取 `<CHECKPOINT_DIR>/config.yaml`；OPV2V / V2XSet 数据路径由 `root_dir` / `validate_dir` 控制。
- whole-frame AP 输出到 `<CHECKPOINT_DIR>/eval.yaml`，包含 `ap30`、`ap_50`、`ap_70`、`mpre_50`、`mrec_50`、`mpre_70`、`mrec_70`。
- `--save_npy` 会输出 `<CHECKPOINT_DIR>/npy/%04d_pcd.npy`、`%04d_pred.npy` 和 `%04d_gt.npy_test`；GT 后缀不是标准 `.npy`，后续脚本需要显式兼容或包装。
- 新增 `docs/doc_workspace/LGCP/opencood_eval_entry.md`，并将 `target.md` 对应项标记完成。

## 2026-07-17 - Greedy optimality gap O3 latency-aware smoke

### 目标

- 推进 `target.md` 中 greedy optimality gap 的剩余部分：接入 latency-aware `O3_confidence_latency_ratio`。
- 在 11 帧 `lgcp_carla` area confidence records 上跑一个受控 exhaustive smoke。

### 代码

更新：

```text
opencda/tools/lgcp_greedy_gap_eval.py
```

新增能力：

- `--enable-o3`
- `--o3-t-delta`
- `--o3-packet-weight`
- `--o3-load-weight`
- 输出 `o3_instance_records.csv` 和 `o3_gap_summary.csv`

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_greedy_gap_eval --input-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score --output-dir docs\doc_workspace\LGCP\experiments\greedy_optimality_gap\20260717_lgcp_carla_greedy_gap_o3_11f --confidence-field density_linear --max-agents 5 --max-areas 3 --max-group-size 3 --delta-g "0.05,0.075,0.1,0.125" --lambda-size 0.02 --enable-o3 --o3-t-delta 1.0 --o3-packet-weight 0.05 --o3-load-weight 0.1
```

### 结果

| Objective | Delta_g | Instances | Mean relative gap | P90 relative gap | Max relative gap | Greedy packets | Optimal packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O3 | 0.050 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |
| O3 | 0.075 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |
| O3 | 0.100 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |
| O3 | 0.125 | 11 | 0.021944 | 0.034483 | 0.068966 | 3.000000 | 3.454545 |

### 结论

- O3 latency-aware holistic objective 已接入。
- 当前 11 帧 smoke 中，O3 mean relative gap 为 `2.19%`，max 为 `6.90%`。
- O1 / O2 和 leader load gap 在同一配置下仍为 `0.0`。
- `target.md` 对应项仍保持未完成，因为还需要多 seed / 多场景扩展。

## 2026-07-17 - Offline subset ablation random multiseed

### 目标

- 推进 `target.md` 中“扩大 offline subset ablation 到多 seed”。
- 避免重复运行 deterministic 方法，只补跑受 seed 影响的 `random` selective-sharing baseline。

### 代码

`opencda/tools/lgcp_subset_ablation_eval.py` 新增：

```text
--methods
```

可只运行 `random`，用于快速补多 seed。

### 运行命令模板

```powershell
conda run -n opencda python -m opencda.tools.lgcp_subset_ablation_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260717_lgcp_carla_offline_subset_multiseed_11f\random_seed11 --ego-cav-id 1 --fusion-method early --max-frames 11 --start-index 0 --budgets "5,10" --methods random --confidence-field density_distance --random-seed 11
```

实际补跑 seeds：`11`、`23`、`37`，并与原 seed `7` 汇总。

### 输出

```text
docs/doc_workspace/LGCP/experiments/ablation/20260717_lgcp_carla_offline_subset_multiseed_11f/
  config.yaml
  notes.md
  method_summary.csv
  random_seed_summary.csv
  random_seed11/
  random_seed23/
  random_seed37/
```

### 结果

| Method | Budget | Seeds | AP@0.7 mean | AP@0.7 std |
| --- | ---: | --- | ---: | ---: |
| random | 5 | 7,11,23,37 | 0.163843 | 0.026394 |
| random | 10 | 7,11,23,37 | 0.328993 | 0.038178 |
| comm_aware_topk | 5 | deterministic | 0.296352 | 0.000000 |
| comm_aware_topk | 10 | deterministic | 0.545736 | 0.000000 |

### 结论

- Random baseline 的多 seed 波动已量化，明显低于 strong selective baselines。
- `comm_aware_topk` 仍略强于当前 `area_aware_union`，所以 LGCP 的论文主张应继续依赖完整 hierarchy / scheduling，而不是只依赖 offline subset AP。
- `target.md` 中 offline subset ablation 多 seed 项已标记完成。

## 2026-07-17 - Greedy O3 larger 6-agent smoke

### 目标

- 推进 greedy optimality gap 的 “更大 instance” 部分。
- 在已有 O3 objective 基础上，将 candidate agents 从 5 扩大到 6，保持 exhaustive search 仍可运行。

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_greedy_gap_eval --input-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score --output-dir docs\doc_workspace\LGCP\experiments\greedy_optimality_gap\20260717_lgcp_carla_greedy_gap_o3_6agents_11f --confidence-field density_linear --max-agents 6 --max-areas 3 --max-group-size 3 --delta-g "0.05,0.075,0.1,0.125" --lambda-size 0.02 --enable-o3 --o3-t-delta 1.0 --o3-packet-weight 0.05 --o3-load-weight 0.1
```

### 结果

| Objective | Delta_g | Instances | Mean relative gap | P90 relative gap | Max relative gap | Greedy packets | Optimal packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O3 | 0.050 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |
| O3 | 0.075 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |
| O3 | 0.100 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |
| O3 | 0.125 | 11 | 0.050486 | 0.063671 | 0.128245 | 3.000000 | 3.454545 |

### 结论

- 6-agent O3 mean relative gap 为 `5.05%`，max 为 `12.82%`。
- Gap 比 5-agent setting 增大，但仍可作为 online heuristic 的经验 small-scale gap 证据。
- `target.md` 仍保持未完成，因为还缺多 seed / 多场景。

## 2026-07-17 - Hierarchy leader / RSU aggregation proxy

### 目标

- 推进完整 LGCP hierarchy 管线中缺失的 leader local result 和 RSU global aggregation 数据接口。
- 在不声称真实 feature fusion 的前提下，生成可复核的离线 proxy records。

### 代码

新增：

```text
opencda/tools/lgcp_hierarchy_aggregation_eval.py
```

输出：

```text
leader_local_results.csv
rsu_global_frame_summary.csv
rsu_global_summary.csv
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_aggregation_eval --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\area_assignment_plan.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_aggregation_top40_11f --quality-field recall_05
```

### 结果

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Quality areas / frame | 33.000000 |
| Selected hierarchy areas / frame | 40.000000 |
| Selected GT ratio | 1.000000 |
| Mean selected area recall@0.5 | 0.670455 |
| Mean confidence-weighted quality | 0.609181 |
| Active leaders / frame | 15.090909 |
| Leader max load / frame | 8.818182 |

### 结论

- Top-40 hierarchy plan 覆盖了该 11 帧 dump 中所有 GT-bearing quality area。
- `selected_area_ratio` 大于 1 是因为 plan 固定 Top-40，而 `area_quality.csv` 只记录有 quality 的 area；论文应优先报告 `selected_gt_ratio`。
- 该工具补齐了 hierarchy 数据接口，但仍不是真实 feature slicing + OpenCOOD local fusion。

## 2026-07-17 - Raw LiDAR feature slice manifest

### 目标

- 推进完整 LGCP hierarchy 管线中的 area-specific feature slicing。
- 先实现 raw LiDAR point slice manifest，作为 neural feature tensor slicing 的前置接口和可变 byte proxy。

### 代码

新增：

```text
opencda/tools/lgcp_feature_slice_manifest.py
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_feature_slice_manifest --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260715_lgcp_carla_hierarchy_plan_top40_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_feature_slice_top40_11f --grid-size-x 10 --grid-size-y 6 --bytes-per-point 16
```

### 结果

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Areas / frame | 40.000000 |
| Slice rows / frame | 61.454545 |
| Total slice points / frame | 34993.636364 |
| Member upload points / frame | 6199.181818 |
| Leader self points / frame | 28794.454545 |
| Member upload bytes / frame | 99186.909091 |
| Leader self bytes / frame | 460711.272727 |

### 结论

- 已生成 area-specific slice manifest，包含 `leader_self` 与 `member_to_leader` 两类 local fusion 输入。
- 这是 raw LiDAR point slicing，不是 OpenCOOD neural feature slicing。
- 后续应将该接口替换为 BEV / intermediate feature map slice，并让 leader 调用 model-level fusion。

## 2026-07-17 - OpenCOOD multi-scene area confidence resource audit

### 目标

- 推进 `target.md` 中 “扩大 area confidence validation 到多 seed / 多场景” 的前置闭环。
- 确认 OpenCOOD 侧数据、checkpoint、协议入口和可执行命令边界，避免把本地单场景 smoke 当作论文级多场景结果。

### 检查内容

```powershell
Get-Content C:\Workspace\OpenCOOD\agent-doc\environment.md
Get-Content C:\Workspace\OpenCOOD\checkpoints\manifest.md
Get-ChildItem C:\Workspace\OpenCOOD\dataset -Force
rg -n "OPV2V\(Culver\)|my_opv2v|test_culver_city|max_cav|comm_range|binomial_n|num_sweep_frames" C:\Workspace\OpenCOOD\agent-doc\protocol_notes.md
rg -n "^(root_dir|validate_dir|test_dir):" C:\Workspace\OpenCOOD\opencood\hypes_yaml\opv2v\lidar_only -g "*.yaml"
```

### 结果

- 本地 `C:\Workspace\OpenCOOD` 没有 `dataset/` 目录，不能直接跑 OPV2V / V2XSet 多场景 inference。
- 远端 `mindspore-186` 已记录 OPV2V 数据路径：`/data1/wql/gzc/dataset/opv2v/{train,val,test}`。
- 远端 OpenCOOD 工作区为 `/data1/wql/gzc/workspace/OpenCOOD`，Python 为 `/data1/wql/yyq/anaconda3/envs/opencood-gzc/bin/python`。
- 多数模板 YAML 仍使用 `dataset/OPV2V/{train,validate,test}`；OpenCOOD 当前论文可比协议文档则指向 `my_opv2v/test_culver_city`、`num_sweep_frames=2`、`binomial_n=10`、`max_cav=5`、`comm_range=70`。
- 本地 checkpoint inventory 中 B-D07 为 `max_cav=5`，更适合作为 LGCP area confidence 多场景第一轮候选；A-D20/A-D23-A-D26 多为 `max_cav=3`，C-D05 为 `max_cav=2`。

### 文档

新增：

```text
docs/doc_workspace/LGCP/opencood_multiscene_area_confidence.md
```

### 结论

- 该任务尚未完成，因为还没有实际多 seed / 多场景 area-level AP / recall 结果。
- 下一步应在 `mindspore-186` 选择固定 checkpoint family，跑 400-frame gate 多 seed，并导出 postprocessed prediction / GT 供 LGCP area slicer 统计。

## 2026-07-17 - Greedy O3 multiseed sampled smoke

### 目标

- 推进 `target.md` 中 greedy optimality gap 的多 seed 部分。
- 在保持原 deterministic Top-M / Top-N 结果可复现的前提下，新增 seed-controlled sampled instance construction。

### 代码

更新：

```text
opencda/tools/lgcp_greedy_gap_eval.py
```

新增参数：

```text
--sample-seeds
--candidate-pool-factor
```

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_greedy_gap_eval.py
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_greedy_gap_eval --input-dir docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score --output-dir docs\doc_workspace\LGCP\experiments\greedy_optimality_gap\20260717_lgcp_carla_greedy_gap_o3_multiseed_sampled_5agents_11f --confidence-field density_linear --max-agents 5 --max-areas 3 --max-group-size 3 --delta-g "0.05,0.075,0.1,0.125" --lambda-size 0.02 --enable-o3 --o3-t-delta 1.0 --o3-packet-weight 0.05 --o3-load-weight 0.1 --sample-seeds "7,11,23,37" --candidate-pool-factor 2
```

### 结果

| Objective | Delta_g | Instances | Mean relative gap | P90 relative gap | Max relative gap | Greedy packets | Optimal packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O3 | 0.050 | 44 | 0.060727 | 0.136821 | 0.187994 | 3.795455 | 3.227273 |
| O3 | 0.075 | 44 | 0.052953 | 0.116843 | 0.137931 | 3.568182 | 3.227273 |
| O3 | 0.100 | 44 | 0.047440 | 0.124867 | 0.137931 | 3.295455 | 3.227273 |
| O3 | 0.125 | 44 | 0.043650 | 0.124867 | 0.137931 | 3.159091 | 3.227273 |

### 结论

- 多 seed sampled setting 下，O3 mean relative gap 保持在 `4.37%` 到 `6.07%`。
- 该结果补齐了单场景多 seed smoke，但还不是多场景论文级结论。
- 尝试 6-agent multiseed sampled exhaustive run 时超过本机 100s timeout；当前保留 6-agent deterministic larger-instance 结果和 5-agent multiseed sampled 结果。

## 2026-07-17 - Hierarchy area-budget sweep

### 目标

- 推进 local-to-global ablation 中 “hierarchy structure” 的可解释证据。
- 在不实现 model-level fusion 的前提下，量化 RSU area assignment / leader upload / RSU aggregation proxy 随 `max_areas` 的 tradeoff。

### 运行命令

对 `max_areas=10/20/30/40` 分别运行：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_plan_eval --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area<MAX_AREAS> --confidence-field density_distance --delta-g 0.05 --max-group-size 4 --max-areas <MAX_AREAS> --feature-packet-bytes 10000 --leader-result-bytes 2000 --assignment-bytes 64 --broadcast-bytes 2000

conda run -n opencda python -m opencda.tools.lgcp_hierarchy_aggregation_eval --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area<MAX_AREAS>\area_assignment_plan.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area<MAX_AREAS>\aggregation --quality-field recall_05
```

### 结果

| Max areas | Selected GT ratio | Mean area recall@0.5 | Weighted quality | Bytes / frame | Local packets / frame | Leader max load |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.472790 | 0.836364 | 0.766486 | 70821.818182 | 4.818182 | 3.000000 |
| 20 | 0.738288 | 0.827273 | 0.763475 | 138734.545455 | 9.545455 | 4.090909 |
| 30 | 0.953193 | 0.869697 | 0.795948 | 222101.818182 | 15.818182 | 6.181818 |
| 40 | 1.000000 | 0.670455 | 0.609181 | 299105.454545 | 21.454545 | 8.818182 |

### 结论

- Top-30 已覆盖 `95.32%` GT-bearing area，且 byte proxy 约为 Top-40 的 `74.25%`。
- Top-40 覆盖所有 GT-bearing area，但 mean selected area recall 降低，说明低优先级 area 的边际收益较低。
- 该结果补强 hierarchy control-plane 的 budget tradeoff，但仍不等同于真实 leader local fusion / RSU model-level aggregation。

## 2026-07-17 - Feature-slice budget sweep

### 目标

- 将 hierarchy budget sweep 接入 raw LiDAR area-specific slice manifest。
- 用数据依赖的 raw slice byte proxy 替代固定 `10000 bytes` per member packet proxy 的一部分解释。

### 运行命令

对 `max_areas=10/20/30/40` 分别运行：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_feature_slice_manifest --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area<MAX_AREAS>\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_feature_slice_budget_sweep_density_distance\area<MAX_AREAS> --grid-size-x 10 --grid-size-y 6 --bytes-per-point 16
```

### 结果

| Max areas | Selected GT ratio | Fixed local bytes / frame | Raw member slice bytes / frame | Raw total slice points / frame |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.472790 | 48181.818182 | 20933.818182 | 12578.090909 |
| 20 | 0.738288 | 95454.545455 | 39287.272727 | 20149.909091 |
| 30 | 0.953193 | 158181.818182 | 59415.272727 | 24373.090909 |
| 40 | 1.000000 | 214545.454545 | 99186.909091 | 34993.636364 |

### 结论

- Top-30 在 `95.32%` selected GT ratio 下 raw member upload bytes 为 `59.42 KB/frame`。
- Raw slice byte proxy 明显低于固定 packet proxy，说明后续 feature tensor slicing 不应继续使用固定大小估计作为唯一通信量证据。
- 该结果仍是 raw LiDAR proxy，不是 neural feature slicing 或 model-level leader fusion。

## 2026-07-17 - Raw-slice-aware upload plan dry-run

### 目标

- 将 raw LiDAR area-slice byte proxy 接回 LGCP hierarchy `upload_plan.csv`。
- 生成可被 `offline_ns3_replay.py --lgcp-upload-plan` 直接读取的 data-dependent upload plan。

### 代码

新增：

```text
opencda/tools/lgcp_slice_upload_plan_eval.py
```

验证：

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_slice_upload_plan_eval.py
```

### 运行命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_slice_upload_plan_eval --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\upload_plan.csv --feature-slice-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_feature_slice_budget_sweep_density_distance\area30\feature_slice_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30
```

Dry-run：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --dry-run
```

### 结果

| Upload type | Requests | New bytes total | Original bytes total | Ratio vs original |
| --- | ---: | ---: | ---: | ---: |
| member_to_leader | 174 | 653568 | 1740000 | 0.375614 |
| leader_to_rsu | 330 | 660000 | 660000 | 1.000000 |
| all | 504 | 1313568 | 2400000 | 0.547320 |

Dry-run replay 11 帧全部通过，每帧 requests 为 `45-48`，每帧 bytes 为 `105056-133680`。

### 结论

- Top-30 raw-slice-aware plan 将总 planned bytes 降到固定 proxy 的 `54.73%`。
- 该 plan 已可作为 NS3 replay 输入，为后续 raw-slice-aware online replay / request-level trace 做准备。
- 仍未完成 neural feature slicing 和 model-level leader local fusion。

## 2026-07-17 - Raw-slice-aware upload plan live NS3 smoke

### 目标

- 验证 Top-30 raw-slice-aware upload plan 不只可 dry-run，也能被 live ns-3 bridge 接受。

### 命令

ns-3 使用 WSL 前台并行启动：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
```

OpenCDA replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10 --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_3f_rsu21\upload_plan_replayed.csv
```

### 结果

| Frame | Timestamp | Requests | Bytes |
| ---: | --- | ---: | ---: |
| 1 | `000060` | 46 | 125888 |
| 2 | `000062` | 46 | 121456 |
| 3 | `000064` | 45 | 105056 |

### 结论

- Live bridge 接受 raw-slice-aware plan，并成功写出带 `pkt_id` 的 `upload_plan_replayed.csv`。
- 本次 ns-3 stdout 没有作为完整 parser input 保存，因此不能报告 request-level delivery ratio。
- 下一步若需要网络证据，应重新运行并保存完整 ns-3 stdout，再接 `lgcp_ns3_log_eval.py`。

## 2026-07-17 - Raw-slice-aware request-level NS3 trace

### 目标

- 重新运行 Top-30 raw-slice-aware 3 帧 NS3 smoke，完整保存 ns-3 stdout。
- 使用 `lgcp_ns3_log_eval.py` 输出 request-level lifecycle。

### 运行要点

- 3 帧 replay 的最终同步时间约为 `0.5s`，time-sync ns-3 使用 `simTime=0.6`。
- ns-3 在 replay 完成后仍可能等待重连；解析前已手动结束残留进程。

### 解析命令

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_3f_rsu21\ns3_stdout_request.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_3f_rsu21\upload_plan_replayed_request.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_request_trace_3f_rsu21 --rsu-node-id 21 --max-frames 3
```

### 结果

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Planned bytes | 352400 |
| Observed `cam_received` | 6 |
| Bridge-observed delivery ratio | 0.043796 |
| RLC TX events | 106 |
| RLC RX events | 20 |
| Requests with RLC TX | 88 |
| Requests with RLC RX | 14 |
| Requests with PSSCH OK | 14 |
| Requests with PSSCH FAIL | 51 |

### 结论

- Raw-slice-aware plan 的 request-level trace 路径已闭合：plan -> RLC -> PSSCH request events -> application callback。
- 当前 unscheduled replay 的链路瓶颈仍严重，尤其 PSCCH overlap / PSSCH decode failure。
- 该结果适合作为 trace-path validation，不应作为最终网络性能行。

## 2026-07-18 - Raw-slice-aware 11-frame request-level NS3 trace

### 目标

- 将 Top-30 raw-slice-aware request-level trace 从 3 帧扩展到完整 11 帧本地 dump。
- 验证 raw-slice-aware upload plan 在完整 11 帧上也能形成 request lifecycle。

### 修正

- 初次启动失败是因为 `ns3_smoke_11f_rsu21` 输出目录尚不存在，导致 WSL stdout 重定向失败。
- 创建目录后重跑成功。

### 命令

ns-3：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd /home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=1.3 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10' > /mnt/c/Workspace/OpenCDA/docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260717_lgcp_carla_raw_slice_upload_plan_area30/ns3_smoke_11f_rsu21/ns3_stdout_request.log 2>&1"
```

OpenCDA replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 10 --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_11f_rsu21\upload_plan_replayed_request.csv
```

Parser：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_11f_rsu21\ns3_stdout_request.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_smoke_11f_rsu21\upload_plan_replayed_request.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\ns3_request_trace_11f_rsu21 --rsu-node-id 21 --max-frames 11
```

### 结果

| Metric | Value |
| --- | ---: |
| Planned requests | 504 |
| Planned bytes | 1313568 |
| Observed `cam_received` | 55 |
| Bridge-observed delivery ratio | 0.109127 |
| RLC TX events | 546 |
| RLC RX events | 118 |
| Requests with RLC TX | 446 |
| Requests with RLC RX | 94 |
| Requests with PSSCH OK | 94 |
| Requests with PSSCH FAIL | 250 |

### 结论

- Raw-slice-aware request lifecycle 已在完整 11 帧本地 dump 上闭合。
- 当前 unscheduled replay 的 delivery ratio 仍低，主要作为调度必要性的诊断证据。
- 下一步若继续网络方向，应基于 raw-slice-aware plan 接入 subchannel scheduling，而不是继续重复 unscheduled replay。
## 2026-07-18 - Raw-slice scheduled NS3 smoke plan

目标：

- 将 raw-slice-aware LGCP upload plan 接入显式 `sc_start/sc_num` scheduling。
- 验证 `offline_ns3_replay.py` 不再丢弃 LGCP CSV 中的调度字段。
- 做一个保守的 single-slot capacity-gated live ns-3 smoke，确认 request-level trace 是否显著减少 decoded overlap / PSSCH failure。

代码变更：

```text
opencda/tools/offline_ns3_replay.py
opencda/tools/lgcp_schedule_upload_plan_eval.py
```

关键修复：

- `build_lgcp_requests()` 现在会从 upload plan 读取并保留 `sc_start/sc_num`。
- 新增 `lgcp_schedule_upload_plan_eval.py`，每帧最多保留 `Z` 条 request，并给 scheduled rows 写入唯一子信道；其余 rows 写入 `capacity_gated_upload_rows.csv`。

命令：

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_ns3_replay.py opencda\tools\lgcp_schedule_upload_plan_eval.py
conda run -n opencda python -m opencda.tools.lgcp_schedule_upload_plan_eval --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_scheduled_smoke_z10 --subchannels 10 --leader-reserve 3
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 11 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_scheduled_smoke_z10\scheduled_upload_plan.csv --dry-run --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_scheduled_smoke_z10\dry_run_upload_plan_replayed.csv
```

scheduled plan 结果：

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Input requests | 504 |
| Scheduled requests | 110 |
| Capacity-gated requests | 394 |
| Scheduled request ratio | 0.218254 |
| Input bytes | 1313568 |
| Scheduled bytes | 543408 |
| Scheduled byte ratio | 0.413689 |

dry-run 观察：

- 11 帧均显示 `requests=10 scheduled=10 skipped_unscheduled=0`。
- `dry_run_upload_plan_replayed.csv` 中已保留 `sc_start/sc_num`。

live ns-3：

- 第一次 11 帧尝试在 `sim_time=0.0` 同步前失败，ns-3 日志显示 callback `Connection refused`；该失败属于回连时序，不是调度字段错误。
- 随后用 3 帧、`sync-timeout=20` 重试成功。

解析命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_scheduled_smoke_z10\ns3_scheduled_3f_rsu21_retry\ns3_stdout_request.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_scheduled_smoke_z10\ns3_scheduled_3f_rsu21_retry\upload_plan_replayed_request.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_scheduled_smoke_z10\ns3_request_trace_3f_rsu21 --rsu-node-id 21 --max-frames 3
```

3-frame scheduled trace：

| Metric | Value |
| --- | ---: |
| Planned requests | 30 |
| Observed `cam_received` | 24 |
| Bridge-observed delivery ratio | 0.800000 |
| Planned bytes | 146992 |
| Observed bytes | 129696 |
| Avg delay | 20.833 ms |
| P95 delay | 42.000 ms |
| RLC TX events | 415 |
| RLC RX events | 376 |
| Requests with PSSCH OK | 28 |
| Requests with PSSCH FAIL | 0 |

结论：

- LGCP `sc_start/sc_num` CSV 字段现在可以进入 offline replay，并触发 ns-3 manual resource allocation。
- 与 unscheduled raw-slice 3 帧 trace 相比，scheduled smoke 的 bridge-observed delivery ratio 从 `0.043796` 提升到 `0.800000`，且 PHY decode failures 为 0。
- 该结果是 single-slot capacity-gated smoke，不是最终多 slot scheduling 或完整 LGCP throughput；下一步应实现 latency-aware 多 slot LGCP scheduler。

## 2026-07-18 - Raw-slice multi-slot scheduling proxy

目标：

- 将上一轮 single-slot smoke 扩展为完整 raw-slice-aware upload plan 的 multi-slot scheduling proxy。
- 为每条 request 输出 `slot_index/sc_start/sc_num/stage/scheduled_delay_ms`，避免为了 NS3 smoke 丢弃 394 条 request。
- 给 transmission scheduling 写作提供稳定的 slots/frame 与 latency proxy。

代码变更：

```text
opencda/tools/lgcp_schedule_upload_plan_eval.py
```

命令：

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_schedule_upload_plan_eval.py
conda run -n opencda python -m opencda.tools.lgcp_schedule_upload_plan_eval --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_schedule_z10 --subchannels 10 --schedule-mode multi_slot --slot-duration-ms 10
```

结果：

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Input requests | 504 |
| Scheduled requests | 504 |
| Capacity-gated requests | 0 |
| Scheduled request ratio | 1.000000 |
| Input bytes | 1313568 |
| Scheduled bytes | 1313568 |
| Scheduled byte ratio | 1.000000 |
| Mean slots / frame | 5.000000 |
| Max slots / frame | 5 |
| Mean frame scheduling latency | 50.000 ms |
| Max frame scheduling latency | 50.000 ms |

每帧结构：

- member-to-leader：15-18 requests，对应 2 slots。
- leader-to-RSU：30 requests，对应 3 slots。
- total：45-48 requests，对应 5 slots。

结论：

- Top-30 raw-slice-aware plan 可在 `Z=10` 下完整排程，不需要 capacity gate。
- 以 `10ms/slot` 估算，完整两级上传调度延迟为 `50ms/frame`。
- 该结果补齐 full-plan scheduling proxy；live NS3 多 slot replay 仍需将 `slot_index` 接入 replay 时序。

## 2026-07-18 - Multi-slot live NS3 replay smoke

目标：

- 将 multi-slot schedule 中的 `slot_index` 接入 `offline_ns3_replay.py`。
- 验证 live ns-3 是否能按 slot 分批接收 LGCP transfer requests，并保持 request-level trace 可解析。

代码变更：

```text
opencda/tools/offline_ns3_replay.py
```

关键变化：

- 新增 `--respect-slot-index`：仅在显式开启时按 `slot_index` 分批发送 request。
- 新增 `--slot-duration-seconds`：每个 slot 后推进的 ns-3 时间，默认 `0.01s`。
- `upload_plan_output` 现在保留 `area_id/slot_index/stage/scheduled_delay_ms`，便于 request-level parser 和调度诊断回连。

dry-run：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_schedule_z10\scheduled_upload_plan.csv --respect-slot-index --slot-duration-seconds 0.01 --dry-run --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_dryrun_z10\dry_run_upload_plan_replayed.csv
```

dry-run 输出：

| Frame | Timestamp | Requests | Slots | Slotted requests |
| ---: | --- | ---: | ---: | ---: |
| 1 | `000060` | 46 | 5 | 46 |
| 2 | `000062` | 46 | 5 | 46 |
| 3 | `000064` | 45 | 5 | 45 |

live ns-3 replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_schedule_z10\scheduled_upload_plan.csv --respect-slot-index --slot-duration-seconds 0.01 --rsu-node-id 21 --drain-seconds 0.3 --sync-timeout 20 --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_dryrun_z10\ns3_multislot_3f_rsu21\upload_plan_replayed_request.csv
```

parser：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_ns3_log_eval --ns3-stdout docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_dryrun_z10\ns3_multislot_3f_rsu21\ns3_stdout_request.log --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_dryrun_z10\ns3_multislot_3f_rsu21\upload_plan_replayed_request.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_dryrun_z10\ns3_request_trace_3f_rsu21 --rsu-node-id 21 --max-frames 3
```

结果：

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Observed `cam_received` | 54 |
| Bridge-observed delivery ratio | 0.394161 |
| Planned bytes | 352400 |
| Observed bytes | 120176 |
| Avg delay | 68.333 ms |
| P95 delay | 202.000 ms |
| RLC TX events | 1013 |
| RLC RX events | 737 |
| Requests with RLC TX | 127 |
| Requests with RLC RX | 110 |
| Requests with PSSCH OK | 110 |
| Requests with PSSCH FAIL | 0 |

分类型：

| Upload type | Planned | Observed app | Bridge ratio |
| --- | ---: | ---: | ---: |
| member_to_leader | 47 | 2 | 0.042553 |
| leader_to_rsu | 90 | 52 | 0.577778 |

结论：

- `slot_index/sc_start/sc_num` 已经能够驱动 live ns-3 分 slot replay。
- 相比 unscheduled raw-slice 3 帧 trace，bridge-observed delivery ratio 从 `0.043796` 提升到 `0.394161`，且 PSSCH FAIL 从 51 个 request 降到 0。
- member-to-leader application callback 仍偏低；后续应检查 application completion timing、分片聚合或 drain duration。

## 2026-07-18 - Multi-slot lifecycle diagnostics and drain check

目标：

- 解释 multi-slot live replay 中 member-to-leader application callback 偏低的原因。
- 排除 `drain-seconds=0.3` 太短导致 callback 未吐出的可能。
- 增加可复用 lifecycle diagnostics 工具。

代码变更：

```text
opencda/tools/lgcp_ns3_log_eval.py
opencda/tools/lgcp_lifecycle_diagnostics.py
```

修复：

- `lgcp_ns3_log_eval.py::parse_value()` 现在会清理字段末尾的 `,` / `;`。长 drain stdout 中出现过多线程日志拼接，导致 `request_id=21,`，此前会触发 `ValueError`。
- 新增 `lgcp_lifecycle_diagnostics.py`，将 `request_lifecycle.csv` 与 replayed upload plan 按 `(timestamp, pkt_id)` 对齐，输出 `by_stage.csv`、`by_stage_slot.csv`、`by_type_terminal.csv`、`by_upload_type_target.csv` 和 `lifecycle_enriched.csv`。

长 drain replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --ego-cav-id 1 --max-frames 3 --lgcp-upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_schedule_z10\scheduled_upload_plan.csv --respect-slot-index --slot-duration-seconds 0.01 --rsu-node-id 21 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_drain1_z10\ns3_multislot_3f_rsu21\upload_plan_replayed_request.csv
```

解析结果与 `drain=0.3` 一致：

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Observed `cam_received` | 54 |
| Bridge-observed delivery ratio | 0.394161 |
| RLC TX events | 1013 |
| RLC RX events | 737 |
| Requests with PSSCH OK | 110 |
| Requests with PSSCH FAIL | 0 |

diagnostics：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_lifecycle_diagnostics --request-lifecycle docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_drain1_z10\ns3_request_trace_3f_rsu21\request_lifecycle.csv --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_drain1_z10\ns3_multislot_3f_rsu21\upload_plan_replayed_request.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_replay_drain1_z10\lifecycle_diagnostics_3f
```

stage summary：

| Stage | Planned | RLC TX | RLC RX | PSSCH OK | CAM ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| member-to-leader | 47 | 40 | 28 | 28 | 0.042553 |
| leader-to-RSU | 90 | 87 | 82 | 82 | 0.577778 |

terminal states：

| Upload type | State | Requests |
| --- | --- | ---: |
| member-to-leader | application_received | 2 |
| member-to-leader | rlc_rx_only | 26 |
| member-to-leader | rlc_tx_no_rx | 12 |
| member-to-leader | planned_only | 7 |
| leader-to-RSU | application_received | 52 |
| leader-to-RSU | rlc_rx_only | 32 |
| leader-to-RSU | rlc_tx_no_rx | 6 |

size-bin diagnostics：

| Upload type | Planned bytes | Planned | RLC RX | PSSCH OK | CAM ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| member-to-leader | 0-1000 | 7 | 4 | 4 | 0.000000 |
| member-to-leader | 1000-2000 | 9 | 2 | 2 | 0.000000 |
| member-to-leader | 2000-4000 | 21 | 12 | 12 | 0.047619 |
| member-to-leader | 4000-8000 | 4 | 4 | 4 | 0.000000 |
| member-to-leader | 8000-16000 | 6 | 6 | 6 | 0.166667 |
| leader-to-RSU | 2000-4000 | 90 | 82 | 82 | 0.577778 |

结论：

- 延长 drain 不改变 delivery，说明 callback 低不是简单等待时间不足。
- member-to-leader 瓶颈同时包含 RLC/PSSCH 未到达和 RLC RX 后 application callback 未出现。
- member-to-leader 大包不是主要瓶颈；`8000-16000` bytes bin 反而全部达到 RLC/PSSCH。
- 下一步应检查 member slot timing / target receiver setup，以及非 RSU receiver 的 CAM application completion。

## 2026-07-18 - Source-unique multi-slot scheduling sensitivity

目标：

- 验证同一 source 在同一 slot 多目的地发射是否解释 member-to-leader callback 偏低。
- 在 multi-slot scheduler 中加入 source-unique packing，作为半双工敏感性。

代码变更：

```text
opencda/tools/lgcp_schedule_upload_plan_eval.py
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_schedule_upload_plan_eval --upload-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_raw_slice_upload_plan_area30\raw_slice_upload_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_raw_slice_multislot_source_unique_z10 --subchannels 10 --schedule-mode multi_slot --slot-duration-ms 10 --enforce-source-unique
```

schedule proxy：

| Metric | Value |
| --- | ---: |
| Scheduled requests | 504 / 504 |
| Mean slots / frame | 7.363636 |
| Max slots / frame | 8 |
| Mean scheduling latency | 73.636 ms |
| Max scheduling latency | 80.000 ms |

3-frame live replay summary：

| Metric | Value |
| --- | ---: |
| Planned requests | 137 |
| Observed `cam_received` | 52 |
| Bridge-observed delivery ratio | 0.379562 |
| Avg delay | 59.769 ms |
| P95 delay | 111.000 ms |
| PSSCH FAIL requests | 0 |

by type：

| Upload type | Planned | Observed app | RLC RX |
| --- | ---: | ---: | ---: |
| member-to-leader | 47 | 5 | 25 |
| leader-to-RSU | 90 | 45 | 84 |

结论：

- source-unique packing 让 member-to-leader application callbacks 从 `2/47` 提高到 `5/47`。
- 但总 delivery 从 `54/137` 降到 `52/137`，member-to-leader RLC RX 也从 `28/47` 降到 `25/47`。
- 因此同 source 同 slot 多发是一个真实约束，应保留在调度机制中，但不是 member-to-leader 瓶颈的充分解释。
# 2026-07-18

## Model-level hierarchy entry audit

- 目标：继续推进 `target.md` 中 local-to-global ablation 和 RSU / leader / global aggregation 管线的剩余缺口，避免继续只扩展 NS3 网络诊断。
- 阅读入口：
  - `opencda/core/ml_libs/opencood_manager.py`
  - `opencood/opencood/tools/inference_utils.py`
  - `opencood/opencood/data_utils/datasets/intermediate_fusion_dataset.py`
  - `opencda/tools/offline_inference.py`
  - `opencda/core/common/offline_dataset.py`
- 结论：
  - 现有 hierarchy aggregation 仍是 proxy，不能写成完整 model-level LGCP AP。
  - 仓库已有 `OpenCOODManager.naive_late_fusion(...)` 和 SGCP inter-cluster late-fusion 路径，可复用为 LGCP box-level hierarchy adapter。
  - Neural feature slicing 需要改 OpenCOOD intermediate fusion 的 dataset / collate / model feature tensor 暴露，不能只靠 LGCP CSV 完成。
- 新增文档：
  - `docs/doc_workspace/LGCP/model_level_hierarchy_entry.md`
- 下一步：
  - 实现 `lgcp_hierarchy_late_fusion_eval.py`，按 `(timestamp, area_id, leader_id, group_members)` 真实调用 OpenCOOD，输出 leader local prediction 与 RSU global late-fusion AP。

## Box-level hierarchy late-fusion adapter smoke

- 新增脚本：
  - `opencda/tools/lgcp_hierarchy_late_fusion_eval.py`
- 功能：
  - 读取 `area_assignment_plan.csv`。
  - 对每个 `(timestamp, area_id, leader_id, group_members)` 真实调用 OpenCOOD。
  - 将 leader prediction / GT 转到 world 坐标，再按 LGCP area 裁剪。
  - 对同一帧的 area-local leader predictions 做 RSU global late fusion。
  - 输出 `leader_local_predictions.csv`、`rsu_global_frame_summary.csv`、`rsu_global_eval_summary.csv`。
- 语法检查：
  - `python -m py_compile opencda\tools\lgcp_hierarchy_late_fusion_eval.py`
- 初次 smoke：
  - `return_object_ids=True` 与 late-fusion dataset 的 `post_process()` 不兼容，已改为 `return_object_ids=False`。
- 成功命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_smoke_area2 --max-frames 1 --max-areas-per-frame 2 --fusion-method late`
- 结果：
  - frames: `1`
  - assignment rows: `2`
  - cached group inference calls: `2`
  - RSU fused pred / GT boxes: `6 / 6`
  - AP@0.3 / AP@0.5 / AP@0.7: `1.000000 / 1.000000 / 0.833333`
- 结论：
  - box-level model-calling hierarchy path 已打通。
  - 当前只覆盖 1 帧 2 area，不能作为论文数值；下一步应扩大到 Top-30 1 帧完整 area、3 帧和 11 帧。

## Box-level hierarchy Top-30 one-frame run

- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top30_1f --max-frames 1 --fusion-method late`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_1f`
- 结果：
  - frames: `1`
  - assignment rows: `30`
  - cached group inference calls: `23`
  - leader local pred / GT boxes: `38 / 35`
  - RSU fused pred / GT boxes: `35 / 35`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.606851 / 0.606851 / 0.517668`
- 观察：
  - Top-30 首帧完整 area budget 可运行，且重复 leader/group 被缓存为 23 次模型调用。
  - 单帧结果只用于验证 adapter 的完整 area-budget 路径，下一步应扩大到 3 帧 / 11 帧，并与 flat selective-sharing baselines 对齐。

## Box-level hierarchy Top-30 three-frame run

- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top30_3f --max-frames 3 --fusion-method late`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_3f`
- 结果：
  - frames: `3`
  - assignment rows: `90`
  - cached group inference calls: `68`
  - mean RSU fused pred / GT boxes per frame: `35.666667 / 35.666667`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.584564 / 0.584564 / 0.508387`
  - GT total / pred samples: `107 / 107`
- 分帧：
  - `000060`: groups `23`, RSU pred / GT `35 / 35`
  - `000062`: groups `23`, RSU pred / GT `36 / 35`
  - `000064`: groups `22`, RSU pred / GT `36 / 37`
- 结论：
  - Top-30 box-level hierarchy adapter 已通过连续 3 帧验证。
  - 下一步应扩大到 11 帧，并和既有 flat selective-sharing baseline 的 11 帧 AP / byte proxy 做同帧对照。

## Box-level hierarchy Top-30 eleven-frame run

- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top30_11f --max-frames 11 --fusion-method late`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_carla_hierarchy_late_fusion_top30_11f`
- 结果：
  - frames: `11`
  - assignment rows: `330`
  - cached group inference calls: `245`
  - mean RSU fused pred / GT boxes per frame: `34.909091 / 37.090909`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.602748 / 0.602748 / 0.506345`
  - GT total / pred samples: `408 / 384`
- 逐帧：
  - unique group calls per frame: `21-23`
  - RSU fused pred boxes per frame: `32-36`
  - RSU fused GT boxes per frame: `35-38`
- 结论：
  - 本地 `lgcp_carla` 11 帧 box-level hierarchy late-fusion 已跑通。
  - 该结果是真实 OpenCOOD model-calling hierarchy ablation，但仍是 box-level late fusion，不是 neural feature slicing。
  - 下一步应整理与 full / confidence_topk / comm_aware_topk / area_aware_union 的同帧 11 帧对照表。

## Local-to-global ablation alignment

- 新增目录：
  - `docs/doc_workspace/LGCP/experiments/ablation/20260718_lgcp_local_to_global_ablation_alignment`
- 输入：
  - flat baselines: `20260715_lgcp_carla_comm_aware_baseline_11f/ablation_summary.csv`
  - hierarchy perception: `20260718_lgcp_carla_hierarchy_late_fusion_top30_11f/rsu_global_eval_summary.csv`
  - hierarchy bytes: `20260718_lgcp_carla_raw_slice_multislot_schedule_z10/scheduled_summary.csv`
  - coverage / raw slice context: hierarchy and feature-slice budget sweeps
- 输出：
  - `local_to_global_ablation_summary.csv`
  - `config.yaml`
  - `notes.md`
- 关键对齐结果：
  - full sharing AP@0.5 / AP@0.7: `0.839868 / 0.526521`, bytes/frame `190000`
  - comm-aware top-k AP@0.5 / AP@0.7: `0.686146 / 0.545736`, bytes/frame `90000`
  - area-aware union AP@0.5 / AP@0.7: `0.676678 / 0.538273`, bytes/frame `90000`
  - LGCP Top-30 box late fusion AP@0.5 / AP@0.7: `0.602748 / 0.506345`, scheduled raw-slice bytes/frame `119415.272727`
- 结论：
  - 当前 box-level hierarchy validates model-calling local-to-global path，但 AP@0.5 尚未超过 strong flat selective baselines。
  - AP@0.7 明显好于 random，并接近 area-aware / comm-aware flat baselines。
  - byte proxy 类型不一致；论文必须显式标注 fixed selected-agent proxy vs scheduled raw-slice plan，下一步最好补 common-byte-budget 对照。

## Top-20 near-common-budget hierarchy run

- 目标：
  - 补 common-byte-budget 对照的低预算 LGCP 点。
  - Top-20 raw member slice bytes 为 `39.29KB/frame`，leader result bytes 为 `40KB/frame`，合计约 `79.29KB/frame`，接近但低于 flat 10-agent `90KB/frame` proxy。
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area20\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top20_11f --max-frames 11 --fusion-method late`
- 结果：
  - frames: `11`
  - assignment rows: `220`
  - cached group inference calls: `194`
  - mean RSU fused pred / GT boxes per frame: `26.272727 / 29.000000`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.538594 / 0.538594 / 0.440331`
  - GT total / pred samples: `319 / 289`
- 结论：
  - Top-20 低预算 hierarchy AP 明显低于 flat 10-agent confidence / area-aware / comm-aware baselines。
  - 这说明当前 box-level hierarchy adapter 更适合验证机制路径，不能支撑“公平预算下超过强 baseline”的强结论。
  - 若要强化 rebuttal，需要 neural feature slicing，或按统一 raw-slice byte accounting 重跑 flat baselines。

## Top-23 near-90KB hierarchy run

- 目标：
  - 生成更接近 flat 10-agent `90KB/frame` 的 LGCP common-budget 点。
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_hierarchy_plan_eval --area-records docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_records.csv --area-quality docs\doc_workspace\LGCP\experiments\area_confidence\20260715_lgcp_carla_area_ap_11f_detector_score\area_quality.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f --confidence-field density_distance --delta-g 0.05 --max-group-size 4 --max-areas 23 --feature-packet-bytes 10000 --leader-result-bytes 2000 --assignment-bytes 64 --broadcast-bytes 2000`
  - `conda run -n opencda python -m opencda.tools.lgcp_feature_slice_manifest --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_feature_slice_area23_11f --grid-size-x 10 --grid-size-y 6 --bytes-per-point 16`
  - `conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_late_fusion_top23_11f --max-frames 11 --fusion-method late`
- 结果：
  - hierarchy areas/frame: `23`
  - raw member upload bytes/frame: `47985.454545`
  - leader result bytes/frame: `46000.000000`
  - estimated raw upload bytes/frame: `93985.454545`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.554762 / 0.554762 / 0.460461`
  - mean RSU fused pred / GT boxes per frame: `29.090909 / 31.545455`
- 结论：
  - Top-23 是当前最接近 flat 10-agent `90KB/frame` 的 LGCP 实测点。
  - AP@0.5 仍低于 random / confidence_topk / area_aware_union / comm_aware_topk，说明 box-level hierarchy 不能支持“公平预算下质量优于强 baseline”的强 claim。
  - 可在 rebuttal 中诚实表述：当前补充实验验证 LGCP hierarchy path 和 budget-quality tradeoff；真正质量提升仍应依赖 neural feature slicing 或更公平的 raw-slice accounting baseline。

## Flat selected-agent raw-byte accounting

- 新增脚本：
  - `opencda/tools/lgcp_flat_raw_byte_accounting.py`
- 目标：
  - 保留既有 flat selective-sharing baseline 的 selected-agent 决策，不重跑模型。
  - 按实际 PCD point count 和 `16 bytes/point` 估算 selected non-ego agents 的 raw LiDAR upload bytes。
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_flat_raw_byte_accounting --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --subset-frame-records docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\subset_frame_records.csv --ablation-summary docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\ablation_summary.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260718_lgcp_flat_raw_byte_accounting_11f --ego-cav-id 1 --bytes-per-point 16`
- 输出：
  - `flat_raw_byte_frame_records.csv`
  - `flat_raw_byte_summary.csv`
  - `unified_raw_byte_accounting_summary.csv`
- 关键结果：
  - comm-aware top-k 10 agents: AP@0.5 / AP@0.7 `0.686146 / 0.545736`, raw bytes/frame `741029.818182`
  - area-aware union 10 agents: AP@0.5 / AP@0.7 `0.676678 / 0.538273`, raw bytes/frame `743892.363636`
  - LGCP Top-30: AP@0.5 / AP@0.7 `0.602748 / 0.506345`, raw scheduled bytes/frame `119415.272727`
- 结论：
  - 按 raw selected-agent bytes 计，LGCP Top-30 使用约 `16.11%` 的 comm-aware top-k bytes，保留 `87.85%` AP@0.5 和 `92.78%` AP@0.7。
  - 这是目前回应 baseline fairness 最清楚的本地证据：不声称 AP 更高，而是强调 much lower communication with bounded quality loss。

## Flat area-slice raw-byte accounting

- 新增脚本：
  - `opencda/tools/lgcp_flat_area_slice_accounting.py`
- 目标：
  - 保留既有 flat selective-sharing baseline 的 selected-agent 决策，不重跑模型。
  - 只统计这些 selected agents 在同一组 LGCP planned area cells 内的 raw point slices。
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_flat_area_slice_accounting --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --subset-frame-records docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\subset_frame_records.csv --ablation-summary docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\ablation_summary.csv --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260718_lgcp_flat_area_slice_accounting_area23_11f --ego-cav-id 1 --grid-size-x 10 --grid-size-y 6 --bytes-per-point 16`
  - `conda run -n opencda python -m opencda.tools.lgcp_flat_area_slice_accounting --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --subset-frame-records docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\subset_frame_records.csv --ablation-summary docs\doc_workspace\LGCP\experiments\ablation\20260715_lgcp_carla_comm_aware_baseline_11f\ablation_summary.csv --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260717_lgcp_carla_hierarchy_budget_sweep_density_distance\area30\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\ablation\20260718_lgcp_flat_area_slice_accounting_area30_11f --ego-cav-id 1 --grid-size-x 10 --grid-size-y 6 --bytes-per-point 16`
- 关键结果：
  - Top-23 area plan: comm-aware top-k 10 agents area-slice bytes/frame `253130.181818`; LGCP Top-23 bytes/frame `93985.454545`
  - Top-30 area plan: comm-aware top-k 10 agents area-slice bytes/frame `295755.636364`; LGCP Top-30 bytes/frame `119415.272727`
- 结论：
  - 在更有利于 flat baselines 的 area-slice accounting 下，LGCP Top-30 仍只用 comm-aware top-k `40.38%` bytes，保留 `87.85%` AP@0.5 和 `92.78%` AP@0.7。
  - baseline fairness 口径已经基本闭环；下一步应转向 neural feature slicing / model-level hierarchy。

## PointPillar intermediate feature geometry probe

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_feature_probe.py`
- 目标：
  - 推进 P0 `Neural feature slicing / model-level hierarchy`。
  - 对 `intermediate_attentive` checkpoint 运行真实 OpenCOOD forward hook，确认 PointPillar 中间 tensor shape，并把 LGCP world-coordinate area cell 映射到 leader-local BEV feature index range。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_feature_probe.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_feature_probe --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_probe_area23_1f5a --fusion-method intermediate_attentive --max-frames 1 --max-areas-per-frame 5 --grid-size-x 10 --grid-size-y 6`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_probe_area23_1f5a`
- 结果：
  - rows: `5`
  - `model.scatter` output: `N x 64 x 200 x 704`
  - `model.backbone` output: `1 x 384 x 100 x 352`
  - 5 个 area 均在 leader lidar range 内。
  - fused feature slice cells: `126-225`
  - scatter slice float32 bytes for group: `117504-430592`
  - fused slice float32 bytes: `193536-345600`
- 观察：
  - `OpenCOODManager.inference()` 对 intermediate 路径会把 `return_object_ids` 传给不支持该参数的本地 OpenCOOD helper；probe 已改为直接调用 `model(batch_data['ego'])` 并用 forward hook 抓 tensor shape。
  - 当前 byte 是未压缩 float32 上界估计，不能替代 raw-slice communication 结果。
- 结论：
  - LGCP neural feature slicing 的坐标映射入口已经验证。
  - 下一步应把 probe 扩展为真实 feature crop / slice manifest adapter，并再接 leader local fusion 与 RSU global aggregation。

## PointPillar feature slice export smoke

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_feature_slice_export.py`
- 目标：
  - 在 feature geometry probe 基础上实际裁剪并保存 PointPillar feature tensor slices。
  - 生成后续 leader-local feature fusion 可读取的 `.npz` slice 文件和 manifest。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_feature_slice_export.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_feature_slice_export --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_slice_export_area23_1f5a --fusion-method intermediate_attentive --max-frames 1 --max-areas-per-frame 5 --grid-size-x 10 --grid-size-y 6 --slice-level both --dtype float16`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_slice_export_area23_1f5a`
- 输出文件：
  - `feature_slice_manifest.csv`
  - `feature_slice_summary.csv`
  - `slices/*.npz`
- 结果：
  - rows: `5`
  - slice level: `both`
  - dtype: `float16`
  - uncompressed bytes: `1502848`
  - compressed `.npz` bytes: `178855`
  - mean compressed bytes / area: `35771`
  - example saved arrays: `scatter` shape `(2, 64, 29, 29)`, `fused` shape `(1, 384, 15, 15)`
- 验证：
  - 已读取前两个 `.npz` 文件，确认包含 `scatter`、`fused`、bounds、timestamp、area、leader 和 group metadata，且 dtype 为 `float16`。
- 结论：
  - LGCP neural feature crop / slice manifest smoke 已完成。
  - 下一步应实现 leader-local feature fusion adapter，决定使用 pre-fusion `scatter` slices 还是 post-fusion `fused` slices 作为论文机制主路径。

## PointPillar feature slice export Top-23 first-frame extension

- 目标：
  - 将 5-area feature crop smoke 扩大到 Top-23 完整首帧 area budget。
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_feature_slice_export --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_slice_export_area23_1f --fusion-method intermediate_attentive --max-frames 1 --max-areas-per-frame 0 --grid-size-x 10 --grid-size-y 6 --slice-level both --dtype float16`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_feature_slice_export_area23_1f`
- 结果：
  - rows / slice files: `23 / 23`
  - uncompressed bytes: `6183680`
  - compressed `.npz` bytes: `810688`
  - mean compressed bytes / area: `35247.304348`
  - scatter elements: `1485952`
  - fused elements: `1605888`
- 结论：
  - feature crop / slice manifest adapter 已通过 Top-23 完整首帧验证。
  - 下一步不应继续只扩大 export 规模，而应实现 leader-local feature fusion 或明确选择 `scatter` / `fused` 作为机制主路径。

## PointPillar leader-local feature fusion smoke

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_leader_feature_fusion.py`
- 目标：
  - 读取 feature-slice export 的 `.npz`，对 group 内 per-CAV `scatter` slices 做 leader-local fusion。
  - 输出后续 RSU feature assembly 可读取的 leader-local feature manifest。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_leader_feature_fusion.py`
  - `python -m opencda.tools.lgcp_pointpillar_leader_feature_fusion --slice-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_slice_export_area23_1f --feature-slice-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_feature_slice_export_area23_1f\feature_slice_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --fusion-methods mean,max --dtype float16 --keep-model-fused`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f`
- 输出文件：
  - `leader_feature_manifest.csv`
  - `leader_feature_summary.csv`
  - `leader_slices/*.npz`
- 结果：
  - rows: `23`
  - fusion methods: `mean,max`
  - dtype: `float16`
  - uncompressed bytes: `7189760`
  - compressed `.npz` bytes: `936298`
  - mean compressed bytes / area: `40708.608696`
  - example arrays: `leader_scatter_mean` shape `(1, 64, 30, 24)`, `leader_scatter_max` shape `(1, 64, 30, 24)`, `model_fused_reference` shape `(1, 384, 16, 12)`
- 观察：
  - `mean/max` fusion 将 per-CAV `scatter` dimension 从 `N` 合并为 leader-side `1`。
  - 当前 deterministic fusion 是机制 smoke，不等价于训练好的 attentive fusion；`model_fused_reference` 被保留作后续 teacher/reference。
- 结论：
  - leader-local neural feature fusion smoke 已完成。
  - 下一步应实现 RSU global feature assembly，将多个 leader area slices 放回统一 canvas，并处理 area overlap。

## PointPillar RSU feature assembly smoke

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_rsu_feature_assembly.py`
- 目标：
  - 读取 leader-local feature fusion 输出，将多个 area leader slices 放回统一 PointPillar scatter canvas。
  - 统计 RSU canvas coverage、overlap 和压缩大小。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_rsu_feature_assembly.py`
  - `python -m opencda.tools.lgcp_pointpillar_rsu_feature_assembly --leader-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --leader-feature-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f\leader_feature_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f --feature-key leader_scatter_mean --canvas-height 200 --canvas-width 704 --channels 64 --dtype float16`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f`
- 输出文件：
  - `rsu_feature_frame_manifest.csv`
  - `rsu_feature_summary.csv`
  - `rsu_frames/*.npz`
- 结果：
  - frames: `1`
  - input / used leader slices: `23 / 23`
  - canvas shape: `1 x 64 x 200 x 704`
  - coverage cells: `4669`
  - coverage ratio: `0.033161`
  - overlap cells: `2835`
  - max overlap: `16`
  - compressed `.npz` bytes: `82974`
- 验证：
  - 已读取 `rsu_frames/000060_leader_scatter_mean.npz`，确认包含 `rsu_canvas` 与 `coverage_count`；canvas shape `(1, 64, 200, 704)`，coverage_count shape `(1, 1, 200, 704)`。
- 结论：
  - LGCP model-level hierarchy data path 已覆盖 feature crop、leader-local fusion 和 RSU-side canvas assembly。
  - 仍未接 detection head / postprocess，因此不能报告 neural feature hierarchy AP；下一步应研究 assembled canvas 到 PointPillar detection head 的可行入口，或定义 feature-level coverage/byte proxy。

## PointPillar RSU detection head probe

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_rsu_head_probe.py`
- 目标：
  - 将 assembled RSU scatter canvas 接回 PointPillar backbone、`cls_head`、`reg_head` 和 voxel postprocess。
  - 区分接口可运行性与有效模型级 AP。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_rsu_head_probe.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_head_probe --rsu-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f --rsu-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f\rsu_feature_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_rsu_head_probe_area23_1f --fusion-method intermediate_attentive --top-k 20`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_rsu_head_probe_area23_1f`
- 结果：
  - input canvas: `1 x 64 x 200 x 704`
  - backbone output: `1 x 384 x 100 x 352`
  - `psm`: `1 x 2 x 100 x 352`
  - `rm`: `1 x 14 x 100 x 352`
  - score min / mean / max: `0.000000 / 0.002679 / 0.220411`
  - score p95: `0.003571`
  - postprocess threshold: `0.2`
  - postprocess pred boxes: `2`
  - top scores: `0.220411`, `0.200373`, `0.199972`
- 观察：
  - Assembled canvas 与 PointPillar heads/postprocessor 技术兼容。
  - 但当前 RSU assembly 是 index-space smoke，多个 leader-local feature slices 尚未重投影到统一 world/RSU coordinate frame。
- 结论：
  - detection-head 可行性已验证，但不能报告有效 AP。
  - 下一步必须实现跨 leader 坐标对齐，或将论文口径收窄为 feature-level coverage / byte proxy。

## PointPillar reference-frame alignment diagnostic

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_reference_aligned_assembly.py`
- 目标：
  - 以统一 reference CAV lidar frame 重新计算每个 world-coordinate area cell 的 target bounds。
  - 量化 leader-local feature slice 到 reference frame 的 yaw delta、resize ratio、coverage 和 overlap。
  - 复用 detection head probe，检查 reference-frame approximate assembly 的 head response。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_reference_aligned_assembly.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_reference_aligned_assembly --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --leader-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --leader-feature-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f\leader_feature_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1 --reference-cav-id 1 --feature-key leader_scatter_mean --grid-size-x 10 --grid-size-y 6 --dtype float16`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_head_probe --rsu-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1 --rsu-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1\reference_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_reference_aligned_head_probe_area23_1f_ref1 --fusion-method intermediate_attentive --top-k 20 --frame-file-column reference_frame_file --canvas-key reference_canvas`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_reference_aligned_assembly_area23_1f_ref1`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_reference_aligned_head_probe_area23_1f_ref1`
- 结果：
  - reference CAV: `1`
  - input / used leader slices: `23 / 23`
  - reference canvas: `1 x 64 x 200 x 704`
  - coverage cells / ratio: `9189 / 0.065263`
  - overlap cells / max overlap: `293 / 3`
  - mean / max abs yaw delta: `93.412838 / 175.817131 deg`
  - mean resize area ratio: `0.637908`
  - head `psm` / `rm`: `1 x 2 x 100 x 352` / `1 x 14 x 100 x 352`
  - head score max / mean: `0.867036 / 0.003301`
  - postprocess pred boxes: `18`
- 观察：
  - Reference-frame target bounds 比 index-space assembly 更合理，coverage ratio 从 `0.033161` 提高到 `0.065263`，head response 也更强。
  - 但 mean abs yaw delta 已达 `93.41 deg`，nearest resize 明显不能替代 feature rotation / affine warp。
- 结论：
  - 坐标对齐问题已被量化，不能把 reference-frame smoke 当作 AP。
  - 下一步要么实现 coordinate-aware feature warp，要么将论文中的 neural feature hierarchy 结果收窄为 coverage / byte / feasibility proxy。

## PointPillar coordinate-warp feature assembly smoke

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_coordinate_warp_assembly.py`
- 目标：
  - 对 reference-frame target cell 逐格执行 `reference -> world -> leader` 反查采样。
  - 比 nearest-resize reference diagnostic 更接近真实 coordinate-aware feature warp。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_coordinate_warp_assembly.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_coordinate_warp_assembly --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --leader-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --leader-feature-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f\leader_feature_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1 --reference-cav-id 1 --feature-key leader_scatter_mean --grid-size-x 10 --grid-size-y 6 --dtype float16`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_head_probe --rsu-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1 --rsu-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1\coordinate_warp_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_head_probe_area23_1f_ref1 --fusion-method intermediate_attentive --top-k 20 --frame-file-column warped_frame_file --canvas-key warped_canvas`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_head_probe_area23_1f_ref1`
- 结果：
  - input / used leader slices: `23 / 23`
  - target / sampled cells: `8550 / 8550`
  - sample ratio: `1.000000`
  - coverage cells / ratio: `8550 / 0.060724`
  - overlap cells / max overlap: `0 / 1`
  - mean / max abs yaw delta: `93.412838 / 175.817131 deg`
  - head `psm` / `rm`: `1 x 2 x 100 x 352` / `1 x 14 x 100 x 352`
  - head score max / mean: `0.893363 / 0.003926`
  - postprocess pred boxes: `30`
- 观察：
  - 所有 target cells 都能在 leader-local feature slice 中找到对应采样点，说明 coordinate path 是闭合的。
  - 该方法消除了 nearest-resize diagnostic 中的 bbox overlap，但仍是 nearest-neighbor sampling。
- 结论：
  - Coordinate-aware model-level path 已从 diagnostic 推进到可运行 warp smoke。
  - 下一步应做 GT/AP smoke，并比较 nearest-neighbor vs bilinear/affine warp；若 AP 不稳定，论文口径应收窄为 feature-level coverage / byte / feasibility proxy。

## PointPillar coordinate-warp AP probe

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_warp_ap_probe.py`
- 目标：
  - 闭合 coordinate-warp canvas -> PointPillar head/postprocess -> reference-frame GT/AP 的评价链路。
  - 判断 nearest-neighbor coordinate warp 是否能支撑模型级 AP。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_warp_ap_probe.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_warp_ap_probe --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --warped-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1 --warped-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1\coordinate_warp_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_ap_probe_area23_1f_ref1 --reference-cav-id 1 --fusion-method intermediate_attentive --frame-file-column warped_frame_file --canvas-key warped_canvas`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_ap_probe_area23_1f_ref1`
- 结果：
  - frames: `1`
  - pred boxes: `30`
  - GT boxes: `16`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.010000 / 0.010000 / 0.000000`
- 观察：
  - AP 评价链路已经跑通。
  - 但 nearest-neighbor coordinate warp 的 AP 极低，说明当前 model-level feature path 仍是 feasibility evidence，不是有效性能结果。
- 结论：
  - 不应扩大 nearest-neighbor warp AP 当作论文结果。
  - 下一步应做 bilinear/affine warp 校准或 retrained aggregation；短期 rebuttal 更安全的口径是 feature-level coverage / byte / feasibility proxy。

## PointPillar bilinear coordinate-warp AP probe

- 新增脚本改动：
  - `opencda/tools/lgcp_pointpillar_coordinate_warp_assembly.py`
  - 新增 `--sampling nearest|bilinear`，默认 `nearest` 保持旧结果兼容。
- 目标：
  - 用同一 Top-23 首帧和同一 reference CAV 1，验证 bilinear sampling 是否能显著修复 nearest coordinate warp 的低 AP。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_coordinate_warp_assembly.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_coordinate_warp_assembly --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --leader-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f --leader-feature-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f\leader_feature_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1 --reference-cav-id 1 --feature-key leader_scatter_mean --grid-size-x 10 --grid-size-y 6 --dtype float16 --sampling bilinear`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_head_probe --rsu-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1 --rsu-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1\coordinate_warp_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_bilinear_head_probe_area23_1f_ref1 --fusion-method intermediate_attentive --top-k 20 --frame-file-column warped_frame_file --canvas-key warped_canvas`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_warp_ap_probe --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --warped-root docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1 --warped-frame-manifest docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1\coordinate_warp_frame_manifest.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_pointpillar_coordinate_warp_bilinear_ap_probe_area23_1f_ref1 --reference-cav-id 1 --fusion-method intermediate_attentive --frame-file-column warped_frame_file --canvas-key warped_canvas`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_bilinear_head_probe_area23_1f_ref1`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_pointpillar_coordinate_warp_bilinear_ap_probe_area23_1f_ref1`
- 结果：
  - bilinear sample ratio: `0.998363`
  - coverage ratio: `0.060625`
  - max overlap: `1`
  - head score max / mean: `0.815208 / 0.003937`
  - postprocess pred boxes: `29`
  - GT boxes: `16`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.024457 / 0.011364 / 0.003472`
- 对照：
  - nearest AP@0.3 / AP@0.5 / AP@0.7: `0.010000 / 0.010000 / 0.000000`
  - bilinear 只带来极小改善，仍远低于可进入论文的模型级 AP。
- 结论：
  - 简单采样方式不是主要瓶颈。
  - 当前跨 leader 裁剪、均值融合、重投影后的 feature canvas 与预训练 PointPillar head 缺少校准。
  - 短期不应继续扩大 nearest/bilinear AP；要么实现 affine/grid-sample + feature calibration / retrained aggregation，要么将 neural hierarchy 口径收窄为 feature-level coverage / byte proxy。

## Neural feature proxy summary

- 新增脚本：
  - `opencda/tools/lgcp_neural_feature_proxy_summary.py`
- 新增文档：
  - `docs/doc_workspace/LGCP/neural_feature_proxy.md`
- 目标：
  - 将 raw member area-slice reference、flat comm-aware area-slice reference、PointPillar feature crop、leader feature fusion、RSU canvas、nearest/bilinear coordinate warp 和 AP boundary 放到同一张表。
  - 明确当前 neural feature hierarchy 只能作为 feasibility / coverage / byte boundary，不能作为论文级 model AP。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_neural_feature_proxy_summary.py`
  - `python -m opencda.tools.lgcp_neural_feature_proxy_summary --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_neural_feature_proxy_summary_area23`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260718_lgcp_neural_feature_proxy_summary_area23`
- 结果：
  - raw member area23 bytes/frame: `47985.454545`
  - comm-aware flat area23 slice bytes/frame: `253130.181818`
  - PointPillar feature crop compressed bytes/frame: `810688`
  - leader scatter fusion compressed bytes/frame: `936298`
  - coordinate warp nearest compressed bytes/frame: `110883`
  - coordinate warp bilinear compressed bytes/frame: `149700`
  - PointPillar feature crop vs raw member area23 ratio: `16.894453`
  - PointPillar feature crop vs comm-aware flat area23 slice ratio: `3.202652`
  - nearest / bilinear AP@0.5: `0.010000 / 0.011364`
- 观察：
  - 未优化 feature crop 并不天然节省通信，反而明显大于 raw member area-slice reference。
  - RSU canvas / coordinate-warp canvas 更小，但它们是聚合后的中间产物，不是 leader upload 通信负载。
- 结论：
  - 短期论文安全口径应是 feature-path feasibility、coverage 和 byte boundary。
  - 感知质量主证据继续使用 box-level hierarchy late-fusion；若坚持 neural AP，需要 affine/grid-sample 校准、feature normalization 或 retrained aggregation head。

## RSU BEV attentive fusion with reference-aligned point slices

- 新增脚本：
  - `opencda/tools/lgcp_pointpillar_rsu_bev_fusion.py`
- 目标：
  - 不再对 leader-local feature slice 做 nearest/bilinear warp。
  - 改为先将每个 area-task group 的成员点云按 world-coordinate LGCP area 裁剪，再投到统一 reference/RSU lidar frame。
  - 用 `pointpillar_attentive_fusion` 的 `pillar_vfe + scatter` 为每个 area leader 生成 `1 x 64 x 200 x 704` scatter BEV canvas。
  - 将所有 leader scatter BEV stack 后送入原始 `AttBEVBackbone + cls/reg heads`，实现 RSU 侧 BEV feature-level attentive fusion。
- 关键实现边界：
  - 不修改 SGCP 实验代码。
  - 不修改 OpenCOOD checkpoint 或模型定义。
  - 默认 `--query-mode mean` 会在 leader stack 前加入一个 synthetic mean RSU query canvas，因为当前 `AttFusion` 原生返回第一个 agent query 的融合结果；`--query-mode first_leader` 可保留 OpenCOOD 原始 ego-first 语义。
  - 通信量同时记录 member-to-leader raw area point bytes、leader-to-RSU full scatter bytes、以及 sparse nonzero BEV cell bytes。
- 验证命令：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_rsu_bev_fusion.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_bev_fusion --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_pointpillar_rsu_bev_fusion_smoke_1f2a --max-frames 1 --max-areas-per-frame 2 --fusion-method intermediate_attentive --reference-cav-id 1 --grid-size-x 10 --grid-size-y 6 --query-mode mean`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_smoke_1f2a`
- 结果：
  - frames: `1`
  - valid leader features: `2`
  - leader feature stack: `2 x 64 x 200 x 704`
  - fusion input: `3 x 64 x 200 x 704`，其中第一个为 mean query canvas
  - RSU backbone output: `1 x 384 x 100 x 352`
  - psm / rm: `1 x 2 x 100 x 352` / `1 x 14 x 100 x 352`
  - pred / GT boxes: `4 / 72`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.055556 / 0.055556 / 0.041667`
  - member-to-leader raw area bytes: `15088`
  - leader feature full bytes: `36044800`
  - leader feature sparse-cell bytes: `54784`
- 观察：
  - 链路已闭合：area point slicing、reference-frame coordinate alignment、PointPillar scatter encoding、RSU attentive BEV fusion、postprocess 和 AP 统计均跑通。
  - 该 smoke 只覆盖 1 帧 2 area，AP 数值不能作为论文结果。
  - sparse-cell leader feature byte accounting 已明显小于 full scatter canvas，但后续必须明确 packet 形式和压缩口径。
- 结论：
  - 这条路线比此前 nearest/bilinear feature warp 更符合 OpenCOOD attentive checkpoint 的原生对齐假设。
  - 下一步应扩大到 Top-23 / Top-30 首帧完整 area，并比较 `query-mode mean` 与 `first_leader`；若 AP 仍低，再考虑 feature normalization 或轻量 retrained RSU aggregation head。

## RSU BEV attentive fusion planned-area evaluation

- 新增脚本改动：
  - `opencda/tools/lgcp_pointpillar_rsu_bev_fusion.py`
  - 新增 `--eval-scope full|planned_areas`，默认 `full` 保持旧 smoke 兼容。
  - `planned_areas` 会将 prediction / GT box center 投到 world frame 后，只保留当前 run 中 LGCP planned areas 内的框，避免用局部 area 输入评价全局 GT。
- 目标：
  - 检查前一轮 2-area AP 极低是否主要来自 coverage/GT scope 不匹配。
  - 在完整 Top-23 首帧下比较 `query-mode mean` 与 `first_leader`。
  - 用低阈值 `0.05` 诊断 RSU BEV attentive 输出是否只是置信度校准偏低。
- 命令摘要：
  - `python -m py_compile opencda\tools\lgcp_pointpillar_rsu_bev_fusion.py`
  - Top-23 首帧 `mean + planned_areas` 默认阈值 `0.20`：
    - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_1f_mean_planned`
  - Top-23 首帧 `first_leader + planned_areas` 默认阈值 `0.20`：
    - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_1f_firstleader_planned`
  - Top-23 首帧 `mean + planned_areas + score_threshold=0.05`：
    - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_1f_mean_planned_thr005`
  - Top-23 首帧 `first_leader + planned_areas + score_threshold=0.05`：
    - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_1f_firstleader_planned_thr005`
  - Top-23 3 帧 `mean + planned_areas + score_threshold=0.05`：
    - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_3f_mean_planned_thr005`
- 结果：
  - Top-23 首帧默认阈值 `0.20`：
    - `mean`: score max `0.127877`，raw pred boxes `0`，planned-area GT `38`，AP 全为 `0`。
    - `first_leader`: score max `0.142288`，raw pred boxes `0`，planned-area GT `38`，AP 全为 `0`。
  - Top-23 首帧阈值 `0.05`：
    - `mean`: pred / GT `6 / 38`，AP@0.3 / AP@0.5 / AP@0.7 = `0.157895 / 0.131579 / 0.131579`。
    - `first_leader`: pred / GT `3 / 38`，AP@0.3 / AP@0.5 / AP@0.7 = `0.052632 / 0.052632 / 0.026316`。
  - Top-23 3 帧阈值 `0.05`：
    - frames / area rows: `3 / 69`
    - mean pred / GT boxes: `6.333333 / 38.000000`
    - AP@0.3 / AP@0.5 / AP@0.7 = `0.166667 / 0.130019 / 0.119298`
    - member-to-leader raw area bytes: `138304`
    - leader full scatter bytes: `1243545600`
    - leader sparse-cell bytes: `1587712`
- 观察：
  - 完整 Top-23 + planned-area scope 修复了“2 area 输入打全局 GT”的评价不公平问题，但默认阈值下仍无预测框。
  - 低阈值 `0.05` 后可以产生少量有效框，说明 reference-aligned point slice -> scatter BEV -> attentive backbone 的几何链路不是完全失效。
  - `mean` query 明显优于 `first_leader`，但二者都远低于 box-level hierarchy AP，说明主要问题不是 query 选择，而是预训练 detection head 对稀疏 area-leader scatter canvas 的分布不适配。
  - full scatter leader feature byte 极高，sparse-cell accounting 才接近可讨论通信负载；后续必须定义 sparse feature packet 或压缩格式。
- 结论：
  - 当前 RSU BEV attentive fusion 可作为机制原型和 feasibility evidence。
  - 暂不能作为论文主性能结果；若要继续追求 model-level AP，需要至少做 feature normalization / score calibration，并很可能需要轻量 fine-tune 或训练 RSU aggregation head。

## RSU BEV attentive fusion 11-frame stability run

- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr005`
- 命令摘要：
  - `conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_bev_fusion --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr005 --max-frames 11 --fusion-method intermediate_attentive --reference-cav-id 1 --grid-size-x 10 --grid-size-y 6 --query-mode mean --eval-scope planned_areas --postprocess-score-threshold 0.05`
- 结果：
  - frames / area rows: `11 / 253`
  - mean pred / GT boxes: `6.818182 / 37.363636`
  - AP@0.3 / AP@0.5 / AP@0.7 = `0.182482 / 0.136468 / 0.099602`
  - member-to-leader raw area bytes: `527840`
  - leader full scatter bytes: `4559667200`
  - leader sparse-cell bytes: `5950080`
- 观察：
  - 11 帧结果延续 3 帧结果，说明低 AP 不是单帧偶然，而是稳定的模型分布失配。
  - 每帧 prediction 数量稳定在 `6-8` 个，而 planned-area GT 约 `36-38` 个，主要瓶颈是 score/head 对稀疏 leader-area scatter canvas 不适配。
  - sparse-cell feature bytes 比 full scatter canvas 小很多，但仍需要明确 leader-to-RSU sparse feature packet 格式，否则不能作为通信收益 claim。
- 结论：
  - reference-aligned point-slice -> leader scatter BEV -> RSU attentive fusion 是可运行机制链路。
  - 直接复用 `pointpillar_attentive_fusion` checkpoint 不能提供论文级 AP；下一步应转向 feature/score calibration 与 RSU aggregation head retraining，或将 neural path 收窄为 feasibility / limitation。

## RSU BEV attentive score-threshold sweep

- 目标：
  - 检查 11 帧 RSU BEV attentive AP 低是否主要来自 postprocess score threshold / calibration。
  - 在不改模型、不改 checkpoint 的前提下，对同一 Top-23 planned-area 11 帧运行比较 `0.005/0.01/0.02/0.05`。
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr0005`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr001`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr002`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr005`
- 结果：

| Score threshold | Pred boxes / frame | AP@0.3 | AP@0.5 | AP@0.7 |
| ---: | ---: | ---: | ---: | ---: |
| `0.005` | `67.454545` | `0.484971` | `0.331495` | `0.101267` |
| `0.010` | `40.909091` | `0.637777` | `0.463679` | `0.136646` |
| `0.020` | `19.636364` | `0.450929` | `0.333014` | `0.132677` |
| `0.050` | `6.818182` | `0.182482` | `0.136468` | `0.099602` |

- 观察：
  - `0.01` 明显优于 `0.005/0.02/0.05`，说明低 AP 不只是几何链路问题，score calibration 是关键瓶颈之一。
  - `0.005` 会产生过多低质量框，AP@0.5 反而下降；`0.02/0.05` 则召回不足。
  - 即使在当前同场景阈值调参最优点，AP@0.7 仍只有 `0.136646`，明显弱于 box-level hierarchy / strong flat baselines。
- 结论：
  - RSU BEV attentive route 可以通过后处理校准从“几乎不可用”提升到“有感知信号”，但还不能作为论文主性能结果。
  - 若继续模型级路线，下一步应在独立 validation split 上做 calibration，并训练或微调 RSU aggregation head；短期写作仍应把该路线定位为 feasibility / calibration boundary。

## RSU BEV attentive query-mode comparison

- 目标：
  - 在 score threshold `0.01` 下比较 `mean` / `first_leader` / `zero` 三种 query-mode。
  - 检查上一轮 AP 提升是否依赖 synthetic mean query，还是任意 query 占位都能得到类似结果。
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_mean_planned_thr001`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_firstleader_planned_thr001`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_11f_zero_planned_thr001`
- 结果：

| Query mode | Pred boxes / frame | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: |
| `mean` | `40.909091` | `0.637777` | `0.463679` | `0.136646` |
| `first_leader` | `26.909091` | `0.435864` | `0.278095` | `0.067018` |
| `zero` | `24.090909` | `0.476781` | `0.261325` | `0.054635` |

- 观察：
  - `mean` query 明显优于 `first_leader` 和 `zero`，说明 synthetic mean RSU query 不是随意占位，而是在当前未训练 checkpoint 下提供了更接近全局查询的输入分布。
  - `first_leader` 与 `zero` 均能产生检测信号，但 AP@0.5 / AP@0.7 明显更弱。
- 结论：
  - 当前原型应继续默认 `query-mode mean`。
  - 论文中不能把 `mean` query 写成完整训练过的 RSU aggregation mechanism；它只能作为无训练原型的 query workaround。若继续模型级路线，应训练显式 RSU query / aggregation head。

## RSU BEV attentive temporal holdout threshold check

- 目标：
  - 将同一 11 帧序列按时间切分，使用后 6 帧 `000070-000080` 复核 `0.005/0.01/0.02/0.05` threshold sweep。
  - 判断 `0.01` 是否只是在完整 11 帧上后验挑出的偶然点。
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_val6_mean_planned_thr0005`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_val6_mean_planned_thr001`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_val6_mean_planned_thr002`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_val6_mean_planned_thr005`
- 结果：

| Score threshold | Pred boxes / frame | AP@0.3 | AP@0.5 | AP@0.7 |
| ---: | ---: | ---: | ---: | ---: |
| `0.005` | `67.500000` | `0.522418` | `0.381124` | `0.093956` |
| `0.010` | `42.166667` | `0.649422` | `0.495974` | `0.119201` |
| `0.020` | `20.166667` | `0.470008` | `0.368083` | `0.134940` |
| `0.050` | `7.166667` | `0.194570` | `0.148743` | `0.084005` |

- 观察：
  - 后 6 帧 holdout 中 `0.01` 仍是 AP@0.3 / AP@0.5 最优点，支持上一轮 score calibration 结论。
  - AP@0.7 在后 6 帧中由 `0.02` 略高，但整体仍很低，说明高 IoU localization / box quality 仍是短板。
- 结论：
  - `0.01` 可作为当前同场景 temporal holdout 下的 prototype threshold。
  - 这不是真正独立多场景 validation；论文中若报告该数值，只能作为 calibration diagnostic / feasibility boundary。

## RSU BEV attentive train5-to-val6 calibration check

- 目标：
  - 补跑前 5 帧 `000060-000068` 的 threshold sweep，模拟 train split 上选择 score threshold。
  - 对照上一轮后 6 帧 `000070-000080`，形成最小 train5-to-val6 calibration 闭环。
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_train5_mean_planned_thr0005`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_train5_mean_planned_thr001`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_train5_mean_planned_thr002`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_pointpillar_rsu_bev_fusion_area23_train5_mean_planned_thr005`
- 结果：

| Split | Score threshold | Pred boxes / frame | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train5 | `0.005` | `67.400000` | `0.546083` | `0.371240` | `0.141289` |
| train5 | `0.010` | `39.400000` | `0.657273` | `0.470975` | `0.174367` |
| train5 | `0.020` | `19.000000` | `0.437342` | `0.312345` | `0.147231` |
| train5 | `0.050` | `6.400000` | `0.168421` | `0.128286` | `0.114463` |
| val6 | `0.010` | `42.166667` | `0.649422` | `0.495974` | `0.119201` |

- 观察：
  - 前 5 帧按 AP@0.3 / AP@0.5 / AP@0.7 都会选择 `0.01`。
  - 后 6 帧使用同一 `0.01` 仍保持 AP@0.5 `0.495974`，说明 score calibration 的收益在同场景时间切分上可复现。
  - AP@0.7 从 train5 的 `0.174367` 降到 val6 的 `0.119201`，高 IoU box quality 仍不稳定。
- 结论：
  - 当前可说：RSU BEV attentive prototype 在同场景 train5-to-val6 calibration 下，`0.01` score threshold 有稳定收益。
  - 仍不能说：已经完成独立 validation 或可作为论文主性能 claim。下一步若坚持模型级结果，需要多场景 validation 或训练 RSU aggregation head。

## RSU BEV attentive calibration summary document

- 新增文档：
  - `docs/doc_workspace/LGCP/rsu_bev_attentive_calibration.md`
- 目标：
  - 将分散在多个 run 目录中的 RSU BEV attentive threshold sweep、query-mode comparison、train5-to-val6 calibration 和 communication accounting 汇总到一个入口。
  - 明确该路线的论文安全口径：只能作为 feature-path feasibility / calibration boundary / limitation，不能作为主性能表。
- 内容：
  - Threshold sweep: `0.01` 在 11 帧 AP@0.3 / AP@0.5 上最优。
  - Query-mode comparison: `mean` 优于 `first_leader` 和 `zero`，但仍是未训练 RSU query workaround。
  - Train5-to-val6 check: train5 选择 `0.01`，val6 AP@0.5 `0.495974`。
  - Communication accounting: leader full scatter BEV 不可通信化，sparse-cell packet 仍需显式格式定义。
- 结论：
  - 后续若继续模型级路线，应转向训练显式 RSU query / aggregation head 和多场景 validation。
  - 若短期写论文修订，应把 neural / BEV feature route 写成可运行机制与边界，不应写成已完成主感知质量结果。

## RSU BEV aggregation training plan

- 新增文档：
  - `docs/doc_workspace/LGCP/rsu_bev_training_plan.md`
- 目标：
  - 回答如果继续把 RSU BEV attentive route 推进为模型级结果，需要训练什么、复用什么、改哪些代码、工作量和风险。
- 代码审计：
  - OpenCOOD 训练入口为 `opencood/opencood/tools/train.py`。
  - 模型创建和 checkpoint 加载在 `opencood/opencood/tools/train_utils.py`。
  - 当前 PointPillar attentive 模型为 `opencood/opencood/models/point_pillar_intermediate.py`，其 backbone 为 `AttBEVBackbone`。
  - 现有 `IntermediateFusionDataset` 的 batch 语义是 ego/reference CAV + multi-CAV `record_len`，与 LGCP timestamp + area leader packets + RSU query 语义不同。
- 结论：
  - 不能只靠改 YAML 直接训练 LGCP RSU BEV route。
  - 需要新增 LGCP RSU BEV training sample export、dataset/collate、model wrapper 和显式 RSU query / aggregation head。
  - 短期论文修订仍建议收窄为 feasibility / calibration boundary；长期研究可按 plan 进入训练闭环。

## RSU BEV sparse training sample export

- 新增脚本：
  - `opencda/tools/lgcp_rsu_bev_training_sample_export.py`
- 目标：
  - 将 reference-aligned point-slice -> leader scatter BEV 的中间结果导出为 future dataset/model wrapper 可读取的 sparse training samples。
  - 不保存完整 dense scatter canvas，改为保存 `[leader, y, x]` sparse indices 和 `64` 维 BEV features。
- 验证：
  - `python -m py_compile opencda\tools\lgcp_rsu_bev_training_sample_export.py`
  - 1 帧 2 area smoke：
    - 输出目录：`docs/doc_workspace/LGCP/experiments/rsu_bev_training_samples/20260720_rsu_bev_sparse_smoke_1f2a`
    - sparse cells: `428`
    - planned-area GT boxes: `6`
    - sample npz bytes: `15566`
  - Top-23 11 帧 full export：
    - 输出目录：`docs/doc_workspace/LGCP/experiments/rsu_bev_training_samples/20260720_rsu_bev_sparse_area23_11f`
    - frames: `11`
    - planned-area GT boxes: `411`
    - compressed sample npz bytes: `1439391`
    - member raw area point bytes: `527840`
    - leader sparse feature bytes: `5950080`
- 样本字段：
  - `dense_shape`
  - `sparse_indices`
  - `sparse_features`
  - `area_ids`
  - `leader_ids`
  - `reference_pose`
  - `planned_area_centers`
  - `gt_boxes`
  - per-area byte accounting fields
- 结论：
  - Phase 1 sample export 已闭环，可作为后续 dataset/model wrapper 的输入。
  - Sparse BEV sample 格式适合训练落盘，但通信 accounting 仍高于 raw area point slice；论文通信收益仍应以 raw / area-slice accounting 为主。

## RSU BEV sparse dataset helper

- 新增代码：
  - `opencda/core/ml_libs/lgcp_rsu_bev_dataset.py`
- 目标：
  - 读取 `lgcp_rsu_bev_training_sample_export.py` 导出的 sparse NPZ。
  - 支持 sparse-only inspection，避免默认还原 Top-23 dense BEV 时占用大量内存。
  - 支持训练时 dense reconstruction、`mean/zero/first_leader` query stack、`record_len` 和 PointPillar `label_dict` 生成。
- 验证：
  - `python -m py_compile opencda\core\ml_libs\lgcp_rsu_bev_dataset.py`
  - smoke sample:
    - input leader dense shape: `2 x 64 x 200 x 704`
    - `query-mode=mean` spatial features: `3 x 64 x 200 x 704`
    - `record_len`: `[3]`
    - `pos_equal_one`: `1 x 100 x 352 x 2`
    - `targets`: `1 x 100 x 352 x 14`
    - GT boxes: `6 x 8 x 3`
  - Top-23 11-frame sparse-only sample:
    - dataset length: `11`
    - first dense shape metadata: `23 x 64 x 200 x 704`
    - first sparse indices: `4136 x 3`
    - first sparse features: `4136 x 64`
    - `record_len_value`: `24`
    - first GT boxes: `38 x 8 x 3`
- 结论：
  - Phase 2 dataset helper 的最低可用版本已完成。
  - 下一步如果继续模型级路线，应新增 `lgcp_rsu_bev_attentive` model wrapper，直接接收 `spatial_features + record_len` 并训练显式 RSU query / head。

## RSU BEV attentive model wrapper smoke

- 新增代码：
  - `opencood/opencood/models/lgcp_rsu_bev_attentive.py`
- 目标：
  - 跳过 `PillarVFE` / `PointPillarScatter`，直接接收 LGCP sparse dataset helper 还原的 scatter BEV。
  - 复用 `AttBEVBackbone + cls_head + reg_head`。
  - 支持 `query_mode=input` 复用 dataset 已构造 query stack。
  - 支持 `query_mode=learnable_channel` 从 `leader_features + leader_record_len` 构造显式 RSU query。
- 验证：
  - `python -m py_compile opencood\opencood\models\lgcp_rsu_bev_attentive.py`
  - `train_utils.create_model` 可通过 `core_method: lgcp_rsu_bev_attentive` 找到 `LgcpRsuBevAttentive`。
  - 1-frame 2-area smoke：
    - `query_mode=input`:
      - `psm`: `1 x 2 x 100 x 352`
      - `rm`: `1 x 14 x 100 x 352`
      - PointPillarLoss: `11.465249914900221`
    - `query_mode=learnable_channel`:
      - `psm`: `1 x 2 x 100 x 352`
      - `rm`: `1 x 14 x 100 x 352`
      - PointPillarLoss: `1432.6074481449284`
      - RSU query gradient sum: `22637.666015625`
- 结论：
  - Phase 3 model wrapper 的最低训练闭环已打通：dataset -> model -> loss -> query gradient。
  - 当前 loss 数值只是 smoke，不代表性能；后续若继续，应接入 YAML / train loop，加载 attentive checkpoint 的 backbone/head 权重，并在 train/val split 上训练显式 RSU query / head。

## RSU BEV train-loop smoke

- 新增脚本：
  - `opencda/tools/lgcp_rsu_bev_train_smoke.py`
- 目标：
  - 在不接入 OpenCOOD 原生 dataset registry 的前提下，先验证 LGCP sparse sample dataset -> RSU BEV model wrapper -> PointPillarLoss -> optimizer -> validation trace 的最小训练循环。
- 命令：
  - `python -m py_compile opencda\tools\lgcp_rsu_bev_train_smoke.py`
  - `conda run -n opencda python -m opencda.tools.lgcp_rsu_bev_train_smoke --train-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_smoke_1f2a --val-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_smoke_1f2a --output-dir docs\doc_workspace\LGCP\experiments\rsu_bev_training\20260720_rsu_bev_train_smoke_1f2a --query-mode learnable_channel --dataset-query-mode mean --freeze-mode query_heads --lr 0.0001 --epochs 1 --max-train-steps 1 --max-val-steps 1 --device cpu`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_smoke_1f2a`
- 结果：
  - trainable parameters: `6224`
  - train final loss: `11.980690373`
  - val final loss: `0.610613982`
  - psm shape: `1 x 2 x 100 x 352`
  - rm shape: `1 x 14 x 100 x 352`
- 观察：
  - `query_heads` freeze mode 只训练 learnable RSU query 和 heads，参数量很小。
  - train/val 使用同一个 1-frame 2-area smoke sample，因此 loss 不能解释为泛化或性能提升。
- 结论：
  - Phase 3.5 train-loop smoke 已闭环。
  - 下一步如果继续模型级路线，应先解决 Top-23 dense BEV 显存压力，再做 train5/val6 或多场景训练；短期论文仍不应使用该 smoke 作为性能结果。

## RSU BEV Top-5 train5 / val6 smoke

- 时间：2026-07-20
- 目标：
  - 避开 Top-23 dense BEV 在 CPU / 内存上的压力，先用 Top-5 planned areas 验证真实 train split / val split 的训练链路。
  - 该实验只验证 `sparse sample -> dataset -> lgcp_rsu_bev_attentive -> PointPillarLoss -> optimizer -> validation trace`，不作为论文 AP 结果。
- 样本导出：
  - train root：`docs/doc_workspace/LGCP/experiments/rsu_bev_training_samples/20260720_rsu_bev_sparse_top5_train5`
    - frames: `5`
    - GT boxes: `60`
    - sample npz bytes: `143715`
    - member upload bytes: `85696`
    - leader sparse feature bytes: `589056`
  - val root：`docs/doc_workspace/LGCP/experiments/rsu_bev_training_samples/20260720_rsu_bev_sparse_top5_val6`
    - frames: `6`
    - GT boxes: `72`
    - sample npz bytes: `199553`
    - member upload bytes: `115216`
    - leader sparse feature bytes: `829056`
- 训练命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_rsu_bev_train_smoke --train-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_train5 --val-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_val6 --output-dir docs\doc_workspace\LGCP\experiments\rsu_bev_training\20260720_rsu_bev_train_top5_train5_val6_smoke --query-mode learnable_channel --dataset-query-mode mean --freeze-mode query_heads --lr 0.0001 --epochs 1 --max-train-steps 2 --max-val-steps 2 --device cpu`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_train5_val6_smoke`
- 结果：
  - trainable parameters: `6224`
  - train samples / val samples: `5 / 6`
  - train loss trace:
    - frame `000060`: `6.825837248`
    - frame `000062`: `4.661619867`
  - val loss trace:
    - frame `000070`: `0.902511302`
    - frame `000072`: `0.847593252`
  - output shapes:
    - `psm`: `1 x 2 x 100 x 352`
    - `rm`: `1 x 14 x 100 x 352`
- 观察：
  - Top-5 split 训练链路可运行，且能使用独立 val frames 输出 loss trace。
  - `query_heads` 仅训练 learnable RSU query 和 heads，参数量小，适合作为快速可行性检查。
  - 当前没有 AP evaluation、没有多场景验证、没有 threshold calibration，不应写成模型性能提升。
- 结论：
  - Phase 3.6 Top-5 train5 / val6 smoke 已完成。
  - 下一步若继续神经特征路线，应扩展到 Top-23 或多场景，并加入固定 validation threshold 的 AP evaluation；若以短期论文修订为目标，应把该路线收敛为 feasibility / limitation。

## RSU BEV Top-5 validation AP hook smoke

- 时间：2026-07-20
- 代码变更：
  - `opencda/tools/lgcp_rsu_bev_train_smoke.py` 新增可选 `--eval-ap` 和 `--postprocess-score-threshold`。
  - 开启后，在 validation 阶段调用 OpenCOOD postprocessor，并用 sparse sample 内的 `gt_boxes` 统计 AP@0.3 / AP@0.5 / AP@0.7。
  - 默认不启用 AP hook，不改变原有 loss-only smoke 行为。
- 验证：
  - `python -m py_compile opencda\tools\lgcp_rsu_bev_train_smoke.py`
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_rsu_bev_train_smoke --train-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_train5 --val-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_val6 --output-dir docs\doc_workspace\LGCP\experiments\rsu_bev_training\20260720_rsu_bev_train_top5_train5_val6_ap_smoke --query-mode learnable_channel --dataset-query-mode mean --freeze-mode query_heads --lr 0.0001 --epochs 1 --max-train-steps 2 --max-val-steps 2 --device cpu --eval-ap --postprocess-score-threshold 0.01`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_train5_val6_ap_smoke`
- 结果：
  - train final loss: `4.661619867`
  - val final loss: `0.847593252`
  - val samples evaluated: `2`
  - val GT boxes: `24`
  - val predicted samples: `57`
  - score threshold: `0.01`
  - AP@0.3 / AP@0.5 / AP@0.7: `0.758776 / 0.682186 / 0.285283`
- 观察：
  - AP hook 已闭环，可以用于后续 Top-23 或多场景 validation calibration。
  - 该 run 只评估 Top-5 areas 的前 2 个 val frames，且训练只跑 2 step；AP 数值不能进入 `results.md` 或论文主表。
- 结论：
  - 训练链路现在不仅能输出 loss，也能输出 validation AP。
  - 下一步最有价值的是扩大到全部 val6 / Top-23，或把 AP hook 作为后续多场景训练的标准输出。

## RSU BEV Top-5 full val6 AP smoke

- 时间：2026-07-20
- 目标：
  - 将上一轮 Top-5 val2 AP hook 扩展到完整 Top-5 train5 / val6 split。
  - 确认 AP hook 能覆盖全部 6 个 validation frames，并观察 val2 与 val6 之间的 AP 波动。
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_rsu_bev_train_smoke --train-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_train5 --val-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_val6 --output-dir docs\doc_workspace\LGCP\experiments\rsu_bev_training\20260720_rsu_bev_train_top5_train5_val6_full_ap_smoke --query-mode learnable_channel --dataset-query-mode mean --freeze-mode query_heads --lr 0.0001 --epochs 1 --max-train-steps 5 --max-val-steps 6 --device cpu --eval-ap --postprocess-score-threshold 0.01`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_train5_val6_full_ap_smoke`
- 结果：
  - train samples / val samples：`5 / 6`
  - train final loss：`4.150855602`
  - val final loss：`0.919490865`
  - val samples evaluated：`6`
  - val GT boxes：`72`
  - val predicted samples：`166`
  - score threshold：`0.01`
  - AP@0.3 / AP@0.5 / AP@0.7：`0.563040 / 0.499721 / 0.164391`
- 观察：
  - 完整 val6 AP@0.5 从 val2 smoke 的 `0.682186` 回落到 `0.499721`，AP@0.7 从 `0.285283` 回落到 `0.164391`。
  - 这说明 val2 过小，容易高估；完整 val6 更符合此前“AP@0.7 仍弱”的判断。
- 结论：
  - Top-5 完整 split 的训练和 AP calibration smoke 已完成。
  - 当前结果仍不进入 `results.md`：它只覆盖单场景、Top-5 planned areas、1 epoch / 5 train steps，不能支撑论文主性能 claim。
  - 下一步若继续模型级路线，应尝试 Top-23 full validation 或多场景，并考虑训练更多 head/query step；若短期修订，仍建议将 neural route 写成 feasibility / limitation。

## RSU BEV Top-5 val6 threshold sweep

- 时间：2026-07-20
- 代码变更：
  - `opencda/tools/lgcp_rsu_bev_train_smoke.py` 新增 `--ap-score-thresholds`，可在同一次 validation pass 中输出多行 threshold AP summary。
  - `--postprocess-score-threshold` 保持兼容；当 `--ap-score-thresholds` 非空时由 sweep 列表覆盖。
- 验证：
  - `python -m py_compile opencda\tools\lgcp_rsu_bev_train_smoke.py`
- 命令：
  - `conda run -n opencda python -m opencda.tools.lgcp_rsu_bev_train_smoke --train-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_train5 --val-root docs\doc_workspace\LGCP\experiments\rsu_bev_training_samples\20260720_rsu_bev_sparse_top5_val6 --output-dir docs\doc_workspace\LGCP\experiments\rsu_bev_training\20260720_rsu_bev_train_top5_val6_threshold_sweep --query-mode learnable_channel --dataset-query-mode mean --freeze-mode query_heads --lr 0.0001 --epochs 1 --max-train-steps 5 --max-val-steps 6 --device cpu --eval-ap --ap-score-thresholds '0.005,0.01,0.02,0.05,0.1'`
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/rsu_bev_training/20260720_rsu_bev_train_top5_val6_threshold_sweep`
- 结果：

| Score threshold | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 |
| ---: | ---: | ---: | ---: | ---: |
| `0.005` | `420` | `0.337772` | `0.301497` | `0.107598` |
| `0.01` | `166` | `0.563040` | `0.499721` | `0.164391` |
| `0.02` | `96` | `0.808087` | `0.717947` | `0.226872` |
| `0.05` | `67` | `0.820811` | `0.806983` | `0.245818` |
| `0.1` | `40` | `0.555556` | `0.555556` | `0.201811` |

- 观察：
  - Top-5 val6 的 AP 对 score threshold 高度敏感。
  - `0.005` 产生 420 个 prediction samples，假阳性过多；`0.1` prediction samples 降到 40，开始漏检。
  - 当前 sweep 中 threshold `0.05` 的 AP@0.5 最好，为 `0.806983`；AP@0.7 仍只有 `0.245818`。
- 结论：
  - Validation AP calibration 工具已具备多阈值 sweep 能力。
  - 该结果仍是单场景 Top-5 smoke，只能说明 calibration sensitivity，不能证明 LGCP neural feature route 已达到论文主性能要求。
  - 下一步应优先尝试 Top-23 / 多场景 sweep，或将 neural route 在论文中明确降级为 feasibility / limitation。

## V2X-ViT compressed feature payload probe

- 时间：2026-07-20
- 动机：
  - 当前 `pointpillar_attentive_fusion` route 传输的是 `pillar_vfe + scatter` 后的 `64 x H x W` scatter BEV，第二跳 feature bytes 明显大于第一次 area point slice。
  - 用户设想更接近“backbone 之后、fusion 模块之前”的特征传输；`pointpillar_v2xvit_fusion` 正好包含 `BaseBEVBackbone -> shrink_conv -> NaiveCompressor -> V2XTransformer`。
- 新增脚本：
  - `opencda/tools/lgcp_v2xvit_feature_probe.py`
- 代码路径：
  - 复用 LGCP area point-slice pipeline：member CAV 点云按 area 切片，投到统一 reference frame。
  - 使用 V2X-ViT checkpoint 的 `pillar_vfe + scatter + backbone + shrink_conv + naive_compressor.encoder`。
  - 统计三种 leader-to-RSU payload：
    - scatter sparse bytes：作为当前 attentive scatter route 的同口径对照。
    - compressed full bytes：完整 compressed latent canvas。
    - compressed crop bytes：只传每个 planned area 对应的 compressed latent crop，并加 `1` cell halo。
- CoBEVT checkpoint 检查：
  - `opencood/logs/pointpillar_cobevt_fusion/config.yaml` 实际使用 `CamIntermediateFusionDataset`、`RgbPreprocessor`、`corpbevt`、`CameraBevPostprocessor` 和 segmentation loss。
  - 当前不适合作为 LGCP LiDAR point-slice detection route 的直接替换 checkpoint。
- V2X-ViT Top-5 首帧 smoke：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_feature_probe_top5_1f`
  - compressed latent shape：`5 x 8 x 48 x 176`
  - member area point bytes：`17216`
  - scatter sparse bytes：`125440`
  - compressed full bytes：`675840`
  - compressed crop bytes：`4848`
- V2X-ViT Top-23 11 帧 probe：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_feature_probe_area23_11f`
  - compressed latent shape per frame：`23 x 8 x 48 x 176`
  - member area point bytes：`527840` total，`47985.45 bytes/frame`
  - scatter sparse bytes：`5949952` total，`540904.73 bytes/frame`
  - compressed full bytes：`34197504` total，`3108864.00 bytes/frame`
  - compressed crop bytes：`248912` total，`22628.36 bytes/frame`
- 关键比例：
  - scatter sparse / member raw area points：`11.27x`
  - compressed full / member raw area points：`64.79x`
  - compressed crop / member raw area points：`0.47x`
  - compressed crop / scatter sparse：`0.04x`
- 观察：
  - V2X-ViT compressed full canvas 仍然太大，不能直接通信化。
  - V2X-ViT compressed area crop 非常有希望：Top-23 11 帧第二跳只约 `22.63 KB/frame`，低于第一次 area point slice 的 `47.99 KB/frame`。
  - 这说明问题不在“中期特征一定比点云大”，而在“传输 scatter BEV 或 full canvas 的层级/格式不合适”。
- area size 讨论：
  - 调大 area 会增加第一次点云切片 bytes，也会增加 area crop feature cells；如果传 full canvas，feature 形状不变但通信量过大。
  - 如果传 compressed area crop，feature bytes 会随 crop area 增长，但由于 V2X-ViT compressed grid 分辨率较低、channel 只有 `8`，增长速度可能仍低于 raw point bytes。
  - 该方向可以作为 sensitivity：较大 area 更容易解释 feature bytes / point bytes 的优势，但会牺牲 LGCP 的精细 area selection 和调度粒度。
- 结论：
  - 新模型方向值得继续，优先级高于继续压当前 attentive scatter BEV。
  - 下一步应实现 V2X-ViT compressed crop 的 RSU assembly / decoder / transformer / detection probe，验证通信量优势是否能同时保留 AP。

## V2X-ViT compressed feature RSU detection smoke

- 时间：2026-07-20
- 目标：
  - 验证 V2X-ViT compressed feature route 不仅能统计通信量，还能接回 RSU 侧 decoder / V2XTransformer / detection heads。
  - 对比 `packet-mode=crop` 与 `packet-mode=full`，判断 AP 低是否主要由 area crop 丢信息导致。
- 新增脚本：
  - `opencda/tools/lgcp_v2xvit_rsu_detection_probe.py`
- 流程：
  - area point slices -> `pillar_vfe + scatter + BaseBEVBackbone + shrink_conv + NaiveCompressor.encoder`
  - RSU 侧按 `crop/full` packet mode 装配 compressed latent
  - `NaiveCompressor.decoder`
  - mean RSU query + V2XTransformer
  - detection heads + planned-area AP
- 验证：
  - `python -m py_compile opencda\tools\lgcp_v2xvit_rsu_detection_probe.py`
- Top-5 首帧 crop-mode：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_crop`
  - compressed shape：`5 x 8 x 48 x 176`
  - decoded shape：`5 x 256 x 48 x 176`
  - fusion input：`6 x 256 x 48 x 176`
  - fused feature：`1 x 256 x 48 x 176`
  - psm / rm：`1 x 2 x 48 x 176` / `1 x 14 x 48 x 176`
  - pred / GT boxes：`18 / 12`
  - AP@0.3 / AP@0.5 / AP@0.7：`0.468204 / 0.065476 / 0.000000`
- Top-5 首帧 full-latent upper bound：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_full`
  - pred / GT boxes：`4 / 12`
  - AP@0.3 / AP@0.5 / AP@0.7：`0.333333 / 0.229167 / 0.000000`
- 观察：
  - V2X-ViT compressed route 的接口已经打通：compressed latent 可以装配、decode、进入 transformer，并输出 detection。
  - full-latent AP@0.5 高于 crop-mode，但仍很低，说明问题不只是 crop 丢信息。
  - 当前最大疑点是 V2X-ViT 原 checkpoint 的 ego/query 语义与 LGCP RSU area-packet 语义不匹配；mean RSU query 仍是 workaround。
- 结论：
  - V2X-ViT route 在通信量上明显优于 scatter BEV，但检测质量仍未闭环。
  - 下一步如果继续，应做 threshold/query sweep、小 area 数到 Top-23 扩展，以及显式 RSU query / head 微调；短期论文仍只能把它写成 promising feature-packet route，而不是主性能结果。

## V2X-ViT RSU detection threshold sweep

- 时间：2026-07-20
- 代码变更：
  - `opencda/tools/lgcp_v2xvit_rsu_detection_probe.py` 新增 `--score-thresholds`，支持一次输出多阈值 AP summary。
- 验证：
  - `python -m py_compile opencda\tools\lgcp_v2xvit_rsu_detection_probe.py`
- 目标：
  - 判断上一轮 V2X-ViT crop/full detection AP 低是否只是 postprocess score threshold 没调好。
- Top-5 首帧 crop-mode threshold sweep：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_crop_thr_sweep`
  - thresholds：`0.005, 0.01, 0.02, 0.05, 0.1`
  - pred / GT boxes：`18 / 12`
  - 所有阈值下 AP@0.3 / AP@0.5 / AP@0.7 均为：`0.468204 / 0.065476 / 0.000000`
- Top-5 首帧 full-latent threshold sweep：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_full_thr_sweep`
  - thresholds：`0.005, 0.01, 0.02, 0.05, 0.1`
  - pred / GT boxes：`4 / 12`
  - 所有阈值下 AP@0.3 / AP@0.5 / AP@0.7 均为：`0.333333 / 0.229167 / 0.000000`
- 观察：
  - `0.005` 到 `0.1` 之间预测数量完全不变，说明这些 predictions 的 score 都高于 `0.1`。
  - AP 低不是简单 score threshold calibration 问题。
  - full-latent upper bound 也低，继续指向 query 语义、RSU area-packet 输入分布、V2X-ViT ego-centered training assumption 不匹配。
- 结论：
  - V2X-ViT compressed crop 解决了 byte boundary，但 detection quality 需要训练或更合理的 RSU query / packet semantic adaptation。
  - 下一步不应继续单纯扫 score threshold；应做 query-mode sweep 或实现显式 RSU query/head 微调。

## V2X-ViT RSU detection query-mode sweep

- 时间：2026-07-20
- 目标：
  - 在 threshold sweep 排除简单 score calibration 后，检查 V2X-ViT route 是否主要受 query 语义影响。
  - 对比 `mean` / `zero` / `first` 三种 query-mode，其中 `first` 表示不插入 synthetic RSU query，直接使用第一个 leader feature 作为 V2XTransformer 输出 query。
- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_crop_zero`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_crop_first`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_full_zero`
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_1f_full_first`
- 结果：

| Packet mode | Query mode | Pred / GT | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | ---: | ---: | ---: | ---: |
| crop | mean | `18 / 12` | `0.468204` | `0.065476` | `0.000000` |
| crop | zero | `13 / 12` | `0.083333` | `0.037037` | `0.000000` |
| crop | first | `7 / 12` | `0.583333` | `0.476190` | `0.113095` |
| full latent | mean | `4 / 12` | `0.333333` | `0.229167` | `0.000000` |
| full latent | zero | `0 / 12` | `0.000000` | `0.000000` | `0.000000` |
| full latent | first | `15 / 12` | `0.980769` | `0.731481` | `0.273810` |

- 观察：
  - Query mode 是 V2X-ViT RSU route 的强影响因素。
  - `first` 明显优于 `mean` / `zero`，说明原 V2X-ViT checkpoint 很依赖第 0 个 agent / ego query 的语义。
  - `crop + first` 在通信友好 packet 下 AP@0.5 达到 `0.476190`，比 `crop + mean` 的 `0.065476` 明显更好。
  - `full + first` 作为 upper-bound payload AP@0.5 达到 `0.731481`，说明模型本身并非完全不适合 LGCP area packet；关键在 query 与 packet 语义。
- 结论：
  - V2X-ViT route 的下一步应优先围绕 query 语义推进，而不是继续扫 score threshold。
  - `first` 不能直接作为最终 RSU global query 机制，但可作为 upper-bound / diagnostic，证明训练显式 RSU query 或选择更合理 leader query 有价值。
  - 下一步应扩展 `crop + first` 到更多 areas / frames，并设计 learnable RSU query 或 leader-query selection。

## V2X-ViT crop+first multi-frame / area-count check

- 时间：2026-07-20
- 目标：
  - 将上一轮 Top-5 首帧 `crop+first` 的正向信号扩展到 11 帧，检查是否只是首帧偶然结果。
  - 将 Top-5 扩到 Top-10 首帧，检查 first-query diagnostic 是否随 area 数增加仍可用。
- Top-5 11-frame crop+first：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top5_11f_crop_first`
  - frames：`11`
  - pred / GT samples：`66 / 132`
  - AP@0.3 / AP@0.5 / AP@0.7：`0.500000 / 0.369657 / 0.081239`
  - compressed crop cells：`3354`
  - estimated compressed crop bytes：`53664` total / `4878.55 bytes/frame`
- Top-10 1-frame crop+first：
  - 输出目录：`docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_detection_top10_1f_crop_first`
  - pred / GT boxes：`2 / 22`
  - AP@0.3 / AP@0.5 / AP@0.7：`0.090909 / 0.090909 / 0.022727`
  - compressed crop cells：`609`
- 观察：
  - Top-5 11-frame 仍有检测信号，但 AP@0.5 从首帧 `0.476190` 降到 `0.369657`，AP@0.7 仍弱。
  - Top-10 首帧明显退化，说明 `first` query 对 area/leader 顺序和覆盖范围敏感，不是可扩展 RSU global query。
  - 通信量仍非常小：Top-5 11-frame compressed crop 约 `4.88 KB/frame`。
- 结论：
  - V2X-ViT compressed crop route 的 byte boundary 继续成立。
  - 当前 first-query 只能作为 diagnostic / upper-bound，不能作为 LGCP RSU global aggregation 机制。
  - 下一步应实现 learnable RSU query / head 微调，或设计稳定的 leader-query selection，而不是继续直接扩大 `first`。

## V2X-ViT leader-query selection diagnostic

- 时间：2026-07-20
- 代码变更：
  - `opencda/tools/lgcp_v2xvit_rsu_detection_probe.py` 新增 `--leader-query-selection`：
    - `plan_order`
    - `max_area_points`
    - `max_member_upload`
    - `max_group_size`
  - 默认 `plan_order` 保持旧行为；该参数主要用于 `query-mode=first` 的诊断。
- 验证：
  - `python -m py_compile opencda\tools\lgcp_v2xvit_rsu_detection_probe.py`
- 目标：
  - 检查 Top-10 首帧 `crop+first` 退化是否可通过简单重排第 0 个 leader query 修复。
- 结果：

| Setting | Query leader | Pred / GT | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | --- | ---: | ---: | ---: | ---: |
| Top-10 crop+first plan order | plan first | `2 / 22` | `0.090909` | `0.090909` | `0.022727` |
| Top-10 crop+first max area points | `12_9 / leader 5` | `2 / 22` | `0.090909` | `0.090909` | `0.022727` |
| Top-10 crop+first max group size | `12_9 / leader 5` | `2 / 22` | `0.090909` | `0.090909` | `0.022727` |

- 观察：
  - `max_area_points` 与 `max_group_size` 均选择同一个 query leader，且结果与 plan order 无差异。
  - 简单启发式 leader query selection 不能修复 Top-10 退化。
  - 并行运行 `max_member_upload` 时因 V2X-ViT 多进程资源占用超时，已停止遗留 Python 进程；本轮不继续消耗时间补跑，因为前两种启发式已给出明确负面信号。
- 结论：
  - V2X-ViT route 后续应转向 learnable RSU query / head 微调，或更系统的 query 训练，而不是手工选择某个 leader 作为全局 query。

## V2X-ViT explicit RSU query/head train smoke

- 时间：2026-07-20
- 目标：
  - 将 V2X-ViT compressed packet route 从纯推理 probe 推进到最小训练闭环。
  - 验证 area point slices -> compressed leader packet -> RSU learnable query -> V2XTransformer -> detection heads -> PointPillarLoss 的梯度链路可运行。
- 新增代码：
  - `opencood/opencood/models/lgcp_v2xvit_rsu.py`
    - 输入 `compressed_features` 或 `decoded_features`。
    - 复用 `NaiveCompressor.decoder`、`V2XTransformer`、`cls_head`、`reg_head`。
    - 支持 `input` / `mean` / `zero` / `learnable_channel` RSU query。
  - `opencda/tools/lgcp_v2xvit_rsu_train_smoke.py`
    - 在线复用 LGCP area 点云切片与 V2X-ViT feature encoder，生成 1-step train sample。
    - 默认冻结 checkpoint 主体，只训练 RSU query 和 detection heads。
- 验证：
  - `python -m py_compile opencood\opencood\models\lgcp_v2xvit_rsu.py opencda\tools\lgcp_v2xvit_rsu_train_smoke.py`
- 运行命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_rsu_train_smoke --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_v2xvit_rsu_train_smoke_1f2a --fusion-method intermediate_v2xvit --reference-cav-id 1 --max-frames 1 --max-areas-per-frame 2 --grid-size-x 10 --grid-size-y 6 --packet-mode crop --query-mode learnable_channel --freeze-mode query_heads --max-train-steps 1 --device cpu
```

- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_train_smoke_1f2a`
- 结果：
  - train steps：`1`
  - trainable parameters：`4368`
  - train final loss：`24.775737337`
  - query gradient norm：`348.719512939`
  - compressed input shape：`2 x 8 x 48 x 176`
  - output shapes：`psm 1 x 2 x 48 x 176`，`rm 1 x 14 x 48 x 176`
  - planned areas / valid leader features / GT boxes：`2 / 2 / 6`
- 结论：
  - V2X-ViT compressed feature route 已具备显式 RSU query/head 微调的最小代码闭环。
  - 该结果只说明 loss 和 gradient path 可运行，不是感知性能结果，不进入 `results.md`。
  - 下一步应扩大到 train/val split，并增加 validation AP hook；若 AP@0.7 仍弱，再考虑解冻 V2XTransformer 后层或停止 neural feature 主线。

## V2X-ViT explicit RSU query/head train-val AP smoke

- 时间：2026-07-20
- 目标：
  - 将上一轮 1-step train smoke 扩展为 train / validation / AP hook 最小闭环。
  - 验证 `lgcp_v2xvit_rsu_train_smoke.py` 可以生成 `loss_trace.csv`、`train_summary.csv` 和 `val_ap_summary.csv`。
- 代码变更：
  - `opencda/tools/lgcp_v2xvit_rsu_train_smoke.py` 新增：
    - `--val-start-index`
    - `--val-max-frames`
    - `--max-val-steps`
    - `--eval-ap`
    - `--ap-score-thresholds`
  - validation 阶段复用 planned-area GT，并将 prediction 也过滤到同一 planned-area scope。
- 验证：
  - `conda run -n opencda python -m py_compile opencda\tools\lgcp_v2xvit_rsu_train_smoke.py opencood\opencood\models\lgcp_v2xvit_rsu.py`
- 运行命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_rsu_train_smoke --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_v2xvit_rsu_trainval_smoke_1f2a --fusion-method intermediate_v2xvit --reference-cav-id 1 --start-index 0 --max-frames 1 --val-start-index 1 --val-max-frames 1 --max-areas-per-frame 2 --grid-size-x 10 --grid-size-y 6 --packet-mode crop --query-mode learnable_channel --freeze-mode query_heads --max-train-steps 1 --max-val-steps 1 --eval-ap --ap-score-thresholds '0.01,0.05' --device cpu
```

- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_smoke_1f2a`
- 结果：
  - train timestamp / val timestamp：`000060 / 000062`
  - train final loss / val final loss：`24.375071751 / 22.165417312`
  - trainable parameters：`4368`
  - query gradient norm：`342.033935547`
  - val predictions / GT boxes：`7 / 6`
  - thresholds `0.01` 和 `0.05` 均为 AP@0.3/0.5/0.7 `0.055556 / 0.055556 / 0.000000`
- 结论：
  - V2X-ViT compressed route 已具备 train / val / AP hook 闭环。
  - 当前只训练 1 step、2 areas、1 val frame，AP 数值只用于确认评估链路，不是性能结论。
  - 下一步应跑 Top-5 train5 / val6 级别的 smoke，并视 AP@0.7 决定是否从 `query_heads` 扩展到解冻 V2XTransformer 后层。

## V2X-ViT explicit RSU query/head Top-5 train5 / val6 smoke

- 时间：2026-07-20
- 目标：
  - 将 V2X-ViT explicit RSU query/head 从极小 `1+1` smoke 扩大到与 PointPillar RSU BEV attentive 相同的 Top-5 train5 / val6 诊断规模。
  - 检查仅训练 learnable RSU query + detection heads 是否足以校准 compressed area-crop route。
- 运行命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_rsu_train_smoke --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_v2xvit_rsu_trainval_top5_train5_val6 --fusion-method intermediate_v2xvit --reference-cav-id 1 --start-index 0 --max-frames 5 --val-start-index 5 --val-max-frames 6 --max-areas-per-frame 5 --grid-size-x 10 --grid-size-y 6 --packet-mode crop --query-mode learnable_channel --freeze-mode query_heads --max-train-steps 5 --max-val-steps 6 --eval-ap --ap-score-thresholds '0.01,0.05,0.1' --device cpu
```

- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_top5_train5_val6`
- 结果：
  - train / val steps：`5 / 6`
  - trainable parameters：`4368`
  - train final loss / val final loss：`13.608617907 / 13.522501738`
  - query gradient norm：`177.555145264`
  - val GT / pred samples：`72 / 86`
  - thresholds `0.01 / 0.05 / 0.1` 均为 AP@0.3/0.5/0.7 `0.053819 / 0.030382 / 0.000000`
- 观察：
  - 训练 loss 从 `14.415974994` 降到 `13.608617907`，说明优化链路仍有效。
  - validation AP 很低，且 score threshold 不影响结果，和 earlier threshold sweep 一致。
  - 与 PointPillar scatter RSU BEV attentive 的 Top-5 val6 threshold `0.05` AP@0.5 `0.806983` / AP@0.7 `0.245818` 相比，V2X-ViT compressed route 只训练 query+heads 明显不够。
- 结论：
  - V2X-ViT compressed area-crop route 的 byte boundary 仍然有吸引力，但 `query_heads` 冻结策略不能形成可用检测性能。
  - 下一步若继续 neural feature 主线，应尝试解冻 V2XTransformer 后层或更系统的 RSU query/fusion 微调；否则应把 V2X-ViT route 保留为通信可行但模型语义未闭环的 limitation。

## V2X-ViT query_fusion_heads freeze mode sanity check

- 时间：2026-07-20
- 目标：
  - 为上一轮 Top-5 `query_heads` 负向诊断准备更强训练策略。
  - 增加可训练 RSU query + V2XTransformer fusion + detection heads 的 freeze mode。
- 代码变更：
  - `opencda/tools/lgcp_v2xvit_rsu_train_smoke.py`
    - `--freeze-mode` 新增 `query_fusion_heads`。
    - 该模式冻结 encoder / decoder，但训练 `rsu_query_channel`、`fusion_net`、`cls_head`、`reg_head`。
- 验证：
  - `conda run -n opencda python -m py_compile opencda\tools\lgcp_v2xvit_rsu_train_smoke.py`
- 运行命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_rsu_train_smoke --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_v2xvit_rsu_trainval_1f2a_query_fusion_heads --fusion-method intermediate_v2xvit --reference-cav-id 1 --start-index 0 --max-frames 1 --val-start-index 1 --val-max-frames 1 --max-areas-per-frame 2 --grid-size-x 10 --grid-size-y 6 --packet-mode crop --query-mode learnable_channel --freeze-mode query_fusion_heads --max-train-steps 1 --max-val-steps 1 --eval-ap --ap-score-thresholds '0.01,0.05' --device cpu
```

- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_1f2a_query_fusion_heads`
- 结果：
  - trainable parameters：`5490569`
  - train / val loss：`24.629682239 / 9.806766111`
  - query gradient norm：`348.058624268`
  - val AP@0.3/0.5/0.7：`0.041667 / 0.041667 / 0.000000`
- 结论：
  - `query_fusion_heads` 模式可运行，可作为下一轮 Top-5 train5 / val6 的候选策略。
  - 1-step AP 仍只用于确认链路，不作为性能判断。

## V2X-ViT query_fusion_heads Top-5 train5 / val6 smoke

- 时间：2026-07-20
- 目标：
  - 验证解冻 V2XTransformer fusion + detection heads + RSU query 是否能修复 `query_heads` 冻结策略的低 AP。
  - 使用与上一轮 `query_heads` 相同的 Top-5 train5 / val6 split，便于直接比较。
- 运行命令：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_rsu_train_smoke --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_v2xvit_rsu_trainval_top5_train5_val6_query_fusion_heads --fusion-method intermediate_v2xvit --reference-cav-id 1 --start-index 0 --max-frames 5 --val-start-index 5 --val-max-frames 6 --max-areas-per-frame 5 --grid-size-x 10 --grid-size-y 6 --packet-mode crop --query-mode learnable_channel --freeze-mode query_fusion_heads --max-train-steps 5 --max-val-steps 6 --eval-ap --ap-score-thresholds '0.01,0.05,0.1' --device cpu
```

- 输出目录：
  - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260720_lgcp_v2xvit_rsu_trainval_top5_train5_val6_query_fusion_heads`
- 结果：
  - train / val steps：`5 / 6`
  - trainable parameters：`5490569`
  - train final loss / val final loss：`2.266083973 / 2.659839972`
  - query gradient norm：`9.641363144`
  - val GT / pred samples：
    - threshold `0.01`：`72 / 96`
    - threshold `0.05`：`72 / 96`
    - threshold `0.1`：`72 / 28`
  - AP@0.3/0.5/0.7：
    - threshold `0.01`：`0.018649 / 0.003945 / 0.000000`
    - threshold `0.05`：`0.018649 / 0.003945 / 0.000000`
    - threshold `0.1`：`0.000694 / 0.000000 / 0.000000`
- 对比上一轮：
  - `query_heads` Top-5 val6 AP@0.3/0.5/0.7 为 `0.053819 / 0.030382 / 0.000000`。
  - `query_fusion_heads` loss 大幅下降，但 AP 反而更低。
- 结论：
  - 当前 V2X-ViT compressed crop + 小样本微调路线存在明显 optimization/perception mismatch：loss 能下降，但 planned-area AP 不改善。
  - 继续投入完整训练前，应谨慎；短期论文主证据不应依赖该 route。
  - 更稳妥口径：V2X-ViT compressed area crop 证明 byte boundary 可行，但 RSU/global query 语义需要专门数据规模和训练设计；当前 LGCP 主性能 claim 应回到 box-level hierarchy late-fusion 与 raw/area-slice accounting。

## V2X-ViT native intermediate fusion with RSU ego and area point crops

- 时间：2026-07-20
- 背景：
  - 新思路不是手工裁剪 compressed latent，也不是把 area leader packet 当作 agent feature。
  - 改为在点云通信层裁剪：每个真实 agent 的 `lidar_np` 只保留 planned-area union 内的点，随后完全复用原 V2X-ViT intermediate inference 流程。
  - RSU 作为 ego/reference slot；其他输入仍是真实 CAV agent，因此 `record_len` / `prior_encoding` / `spatial_correction_matrix` 语义更接近原 checkpoint。
- 新增代码：
  - `opencda/tools/lgcp_v2xvit_area_point_crop_eval.py`
    - `--ego-cav-id -1`：RSU 作为 ego。
    - `--reference-z-override`：将高架 RSU ego 坐标的 z 调整到车载 checkpoint 可处理高度。
    - 输出 `summary.csv`、`frame_summary.csv`、`cav_area_points.csv`。
- 关键实现 caveat：
  - 真实 RSU LiDAR pose 为 `z=12m`，而 `pointpillar_v2xvit_fusion` 的 LiDAR range 是车辆传感器高度附近。
  - 若直接用真实 RSU z，地面点投到 RSU 坐标约为 `z=-10m`，会被预处理全部裁掉并导致 empty voxel。
  - 当前 smoke 使用 `--reference-z-override 2.0`，保留 RSU x/y/yaw，但用车载高度作为 OpenCOOD reference z，并将 RSU 自身点云变换到该 reference。
- 验证：
  - `conda run -n opencda python -m py_compile opencda\tools\lgcp_v2xvit_area_point_crop_eval.py`
- 1-frame Top-5 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_area_point_crop_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_v2xvit_area_point_crop_rsu_ego_top5_1f_z2 --fusion-method intermediate_v2xvit --ego-cav-id -1 --reference-z-override 2.0 --start-index 0 --max-frames 1 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 1-frame Top-5 result：
  - AP@0.3/0.5/0.7：`0.222222 / 0.222222 / 0.111111`
  - pred / GT：`2 / 9`
  - CAV area upload bytes：`88704`
  - RSU ego area bytes：`2688`
- 11-frame Top-5 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_area_point_crop_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260720_lgcp_v2xvit_area_point_crop_rsu_ego_top5_11f_z2 --fusion-method intermediate_v2xvit --ego-cav-id -1 --reference-z-override 2.0 --start-index 0 --max-frames 11 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 11-frame Top-5 result：
  - AP@0.3/0.5/0.7：`0.228011 / 0.208964 / 0.011765`
  - pred / GT：`21 / 85`
  - CAV area upload bytes：`1108432` total / `100766.55 bytes/frame`
  - RSU ego area bytes：`72128` total
  - total area point bytes：`107323.64 bytes/frame`
- 结论：
  - 该路线明显比 V2X-ViT compressed latent RSU route 更合理：AP 没有崩到接近 0，说明保持原 model intermediate fusion 语义是对的。
  - AP@0.7 仍弱，且当前只用 Top-5 areas / max 5 agents / 单场景 11 帧，因此仍是 smoke。
  - 这条路线符合“点云区域裁剪通信 + 原模型 intermediate fusion”的论文机制解释，可作为下一步更稳的 neural/model-level hierarchy 候选。
  - 后续应做 threshold sweep、Top-10/Top-23 area 扩展和与 box-level hierarchy / PointPillar scatter route 的同口径比较。

## Attentive native intermediate fusion with RSU ego and area point crops

- 时间：2026-07-21
- 目标：
  - 回答同样的“点云区域裁剪通信 + 原模型 intermediate fusion + RSU ego”处理方式下，`pointpillar_attentive_fusion` 的 AP 如何。
  - 与 2026-07-20 的 V2X-ViT point-crop native route 做同口径对照。
- 运行设置：
  - dataset：`D:\Data\Carla\2026_07_15_02_33_21`
  - assignment plan：`docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv`
  - ego：RSU `-1`
  - `--reference-z-override 2.0`
  - Top-5 planned areas，11 frames，max 5 agents，planned-area eval scope。
- 低阈值 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_area_point_crop_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260721_lgcp_attentive_area_point_crop_rsu_ego_top5_11f_z2_thr005 --fusion-method intermediate_attentive --ego-cav-id -1 --reference-z-override 2.0 --start-index 0 --max-frames 11 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 低阈值结果：
  - AP@0.3/0.5/0.7：`0.615115 / 0.420702 / 0.244483`
  - pred / GT：`76 / 85`
  - CAV area upload bytes：`100766.55 bytes/frame`
- 默认阈值 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_v2xvit_area_point_crop_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260721_lgcp_attentive_area_point_crop_rsu_ego_top5_11f_z2_defaultthr --fusion-method intermediate_attentive --ego-cav-id -1 --reference-z-override 2.0 --start-index 0 --max-frames 11 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas
```

- 默认阈值结果：
  - checkpoint default score threshold：`0.20`
  - AP@0.3/0.5/0.7：`0.282353 / 0.282353 / 0.203315`
  - pred / GT：`24 / 85`
  - CAV area upload bytes：`100766.55 bytes/frame`
- 对比 V2X-ViT point-crop native route：
  - V2X-ViT Top-5 11-frame threshold `0.05` AP@0.3/0.5/0.7：`0.228011 / 0.208964 / 0.011765`
  - attentive Top-5 11-frame threshold `0.05` AP@0.3/0.5/0.7：`0.615115 / 0.420702 / 0.244483`
- 结论：
  - V2X-ViT point-crop AP@0.3 `0.228011` 是合理的：它与 AP@0.5 `0.208964` 接近，说明少量预测中能命中的框大多已经超过 IoU 0.5；真正弱点是 AP@0.7 `0.011765`，即定位质量不足。
  - 同样输入处理下 attentive 明显更稳，说明 area point crop 机制本身没有导致 AP 崩溃。
  - 后续若要走“点云区域裁剪通信 + 原模型 intermediate fusion”作为模型级主线，优先用 attentive checkpoint 做 Top-10/Top-23 和 threshold sweep；V2X-ViT 可作为对照或说明模型结构/checkpoint 适配风险。

## Where2comm checkpoint with LGCP external area mask

- 时间：2026-07-22
- 目标：
  - 验证是否可以复用 `C:\Workspace\OpenCOOD\checkpoints\where2comm_10e` 的 PointPillar backbone、affine alignment、attentive fusion 和 detection heads。
  - 用 LGCP planned-area BEV mask 替代 Where2comm 原始 objectness/confidence BEV-cell selector，形成“LGCP 区域选择 + 可选择中期特征融合”的最小闭环。
- 新增 / 修改：
  - `opencood/opencood/models/fuse_modules/fusion_in_one.py`
    - 补回 Where2comm `fusion: att` 所需的 scaled-dot-product attentive fusion。
    - 支持 `external_comm_mask` 和 `external_comm_recon`，并支持 `external_ego_full` / `external_rate_exclude_ego` 语义。
  - `opencood/opencood/tools/train_utils.py`
    - loader 兼容 `net_epoch_bestval_at6.pth` 文件名。
  - `opencood/opencood/data_utils/datasets/__init__.py`
    - dataset builder 兼容外部 checkpoint 中的小写 `fusion.core_method: intermediate`。
  - `opencda/core/ml_libs/opencood_manager.py`
    - 增加可选 `_dataset_root_override`，并在外部 checkpoint 测试时将 `train_params.max_cav` 对齐到 `model.args.max_cav`。
  - `opencda/tools/lgcp_where2comm_area_mask_eval.py`
    - 新增 LGCP Where2comm area-mask runner。
    - `mask-mode=lgcp_area`：planned areas 生成 RSU/ego reference BEV mask，作为 Where2comm 外部通信 mask。
    - `mask-mode=none`：退回 Where2comm 内部 objectness mask。
    - `mask-mode=full`：全特征通信上界。
    - 输出 `frame_summary.csv`、`feature_scale_summary.csv`、`cav_area_points.csv` 和 `summary.csv`。
  - `opencda/scenario_testing/config_yaml/lgcp_carla.yaml` 与 `enable_coperception.yaml`
    - 新增 `intermediate_where2comm: C:/Workspace/OpenCOOD/checkpoints/where2comm_10e` 模型别名。
- 验证：

```powershell
conda run -n opencda python -m py_compile opencda\tools\lgcp_where2comm_area_mask_eval.py opencood\opencood\models\fuse_modules\fusion_in_one.py opencood\opencood\models\point_pillar_comm_multiscale.py opencood\opencood\tools\train_utils.py opencood\opencood\data_utils\datasets\__init__.py opencda\core\ml_libs\opencood_manager.py
```

- 1-frame Top-5 LGCP area-mask command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_area_mask_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_area_mask_top5_1f_z2 --fusion-method intermediate_where2comm --ego-cav-id -1 --reference-z-override 2.0 --max-frames 1 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 1-frame Top-5 comparison：

| Mask mode | AP@0.3 | AP@0.5 | AP@0.7 | Comm rate | Second-hop Mbps |
| --- | ---: | ---: | ---: | ---: | ---: |
| LGCP area | 0.666667 | 0.666667 | 0.277778 | 0.011344 | 27.607041 |
| Where2comm internal | 0.888889 | 0.750000 | 0.369048 | 0.019065 | 44.083200 |
| Full | 0.666667 | 0.537037 | 0.277778 | 1.000000 | 2422.210560 |

- 11-frame Top-5 LGCP area-mask command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_area_mask_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_area_mask_top5_11f_z2 --fusion-method intermediate_where2comm --ego-cav-id -1 --reference-z-override 2.0 --max-frames 11 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 11-frame Top-5 comparison：

| Mask mode | AP@0.3 | AP@0.5 | AP@0.7 | Comm rate | Second-hop Mbps |
| --- | ---: | ---: | ---: | ---: | ---: |
| LGCP area | 0.614944 | 0.411297 | 0.116882 | 0.010549 | 25.559041 |
| LGCP area + dilation 1 | 0.730799 | 0.608572 | 0.098913 | 0.014206 | 35.069208 |
| Where2comm internal | 0.801557 | 0.649291 | 0.163261 | 0.017476 | 39.112146 |

- 结论：
  - 复用 `where2comm_10e` checkpoint 并用 LGCP planned-area mask 替代 BEV-cell selector 已端到端跑通。
  - 内部 Where2comm mask 在同一 RSU ego / point-crop 语义下可达到 AP@0.5 `0.649291`，说明 checkpoint/fusion route 本身可用。
  - LGCP 几何 area mask 在 11 帧下 AP@0.5 为 `0.411297`，与 attentive point-crop native route 的 `0.420702` 接近；dilation 1 可提升到 `0.608572`，但 AP@0.7 没有同步改善。
  - 当前第二跳特征通信仍是 Mbps 级：area mask 约 `25.56 Mbps`，dilation 1 约 `35.07 Mbps`，内部 mask 约 `39.11 Mbps`。这比 full feature 上界低很多，但仍不是低 KB 级。
  - 下一步应优先测试 `LGCP area ∩ Where2comm objectness`、Top-10/Top-23 area 扩展、只传 leader BEV 而不是所有 CAV BEV 的更严格 LGCP 版本，并复核 mask metadata / index overhead。

## Where2comm checkpoint with LGCP area-objectness intersection

- 时间：2026-07-22
- 目标：
  - 保留 LGCP planned-area 作为区域语义边界，同时复用 Where2comm checkpoint 自身的 objectness/confidence BEV-cell selector。
  - 验证 `LGCP area ∩ Where2comm objectness` 能否降低二跳 feature 通信量，并避免纯几何 area mask 带来的无目标 cells 上传。
- 新增 / 修改：
  - `opencood/opencood/models/fuse_modules/fusion_in_one.py`
    - `Where2comm` 增加 `external_mask_mode='intersection'`。
    - 当传入 `external_comm_mask` 且 mode 为 `intersection` 时，先生成内部 objectness mask，再与外部 LGCP area mask 相乘，并复用 `external_ego_full` / `external_rate_exclude_ego` 统计非 ego 通信率。
  - `opencda/tools/lgcp_where2comm_area_mask_eval.py`
    - `--mask-mode` 新增 `lgcp_area_objectness`。
    - 该模式仍由 planned areas 生成 RSU/ego reference BEV mask，但融合模块内部执行 area mask 与 objectness mask 的交集。
- 验证：

```powershell
conda run -n opencda python -m py_compile opencood/opencood/models/fuse_modules/fusion_in_one.py opencda/tools/lgcp_where2comm_area_mask_eval.py
```

- 11-frame Top-5 area-objectness command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_area_mask_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_area_objectness_top5_11f_z2 --fusion-method intermediate_where2comm --ego-cav-id -1 --reference-z-override 2.0 --max-frames 11 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05 --mask-mode lgcp_area_objectness
```

- 11-frame Top-5 area-objectness + dilation 1 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_area_mask_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_area_objectness_top5_11f_z2_dilate1 --fusion-method intermediate_where2comm --ego-cav-id -1 --reference-z-override 2.0 --max-frames 11 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05 --mask-mode lgcp_area_objectness --mask-dilation-cells 1
```

- 11-frame Top-5 comparison：

| Mask mode | AP@0.3 | AP@0.5 | AP@0.7 | Comm rate | Second-hop Mbps |
| --- | ---: | ---: | ---: | ---: | ---: |
| LGCP area | 0.614944 | 0.411297 | 0.116882 | 0.010549 | 25.559041 |
| LGCP area + dilation 1 | 0.730799 | 0.608572 | 0.098913 | 0.014206 | 35.069208 |
| LGCP area ∩ objectness | 0.715218 | 0.565188 | 0.202143 | 0.005577 | 12.964771 |
| LGCP area ∩ objectness + dilation 1 | 0.801557 | 0.658694 | 0.162600 | 0.006948 | 16.180131 |
| Where2comm internal | 0.801557 | 0.649291 | 0.163261 | 0.017476 | 39.112146 |

- 结论：
  - `LGCP area ∩ objectness` 是当前最有希望的模型级路线：相对纯 area mask，通信从 `25.56 Mbps` 降到 `12.96 Mbps`，AP@0.5 从 `0.411297` 升到 `0.565188`，AP@0.7 从 `0.116882` 升到 `0.202143`。
  - `LGCP area ∩ objectness + dilation 1` 在 AP@0.5 上略高于 Where2comm internal mask（`0.658694` vs `0.649291`），二跳通信量约为 internal mask 的 `41.37%`（`16.18 Mbps` vs `39.11 Mbps`）。
  - 该结果仍是单场景 Top-5 11 帧 smoke，不能直接写成论文最终主表，但足以把后续优先级从 V2X-ViT compressed route 转向 Where2comm area-objectness route。
  - 下一步应扩展 Top-10 / Top-23 area，复核 leader-only feature packet 语义和 mask/index metadata overhead，并将第一次 member-to-leader 点云 area slices 与第二次 leader-to-RSU feature cells 分开统计。

## Where2comm area-objectness Top-10 / Top-23 scale boundary

- 时间：2026-07-22
- 目标：
  - 将 `LGCP area ∩ objectness + dilation 1` 从 Top-5 扩展到 Top-10 / Top-23 planned areas。
  - 用 Where2comm internal mask 做同口径对照，判断 AP 下降来自 LGCP selector 还是当前 `max_cav=5` / checkpoint 覆盖限制。
- 运行设置：
  - dataset：`D:\Data\Carla\2026_07_15_02_33_21`
  - checkpoint：`C:\Workspace\OpenCOOD\checkpoints\where2comm_10e`
  - ego：RSU `-1`
  - `--reference-z-override 2.0`
  - 11 frames，max 5 agents，planned-area eval scope，score threshold `0.05`。
- 11-frame comparison：

| Scope | Mask mode | AP@0.3 | AP@0.5 | AP@0.7 | Comm rate | Second-hop Mbps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Top-5 | LGCP area ∩ objectness + dilation 1 | 0.801557 | 0.658694 | 0.162600 | 0.006948 | 16.180131 |
| Top-5 | Where2comm internal | 0.801557 | 0.649291 | 0.163261 | 0.017476 | 39.112146 |
| Top-10 | LGCP area ∩ objectness + dilation 1 | 0.621132 | 0.270222 | 0.046635 | 0.013907 | 32.213179 |
| Top-10 | Where2comm internal | 0.604083 | 0.264850 | 0.045591 | 0.027124 | 57.886721 |
| Top-23 | LGCP area ∩ objectness + dilation 1 | 0.539764 | 0.227966 | 0.033903 | 0.025454 | 57.985399 |
| Top-23 | Where2comm internal | 0.552924 | 0.240054 | 0.032983 | 0.039782 | 84.149530 |

- 结论：
  - Top-10 / Top-23 下，交集 selector 仍稳定低于 internal mask 的通信量：Top-10 为 `55.65%`，Top-23 为 `68.91%`。
  - AP 下降在 internal mask 中同样出现，说明主因不是 LGCP selector，而是当前 checkpoint / runner 的 `max_cav=5` 与更大 planned-area eval scope 覆盖不足。
  - 当前 Where2comm route 最适合作为 Top-5 area 的模型级机制证据；若要支撑 Top-23，需要解除 `max_cav=5` 语义限制、改为真正 leader packet 汇总，或训练/微调 RSU global aggregation。

## Where2comm leader-once second-hop accounting

- 时间：2026-07-22
- 目标：
  - 明确当前 Where2comm runner 的通信统计语义。
  - 增加 LGCP leader-to-RSU 汇总包的 lower-bound proxy，避免把“模型输入中的多个 CAV feature”直接等同于“二跳所有 CAV 分别向 RSU 上传 feature”。
- 新增 / 修改：
  - `opencda/tools/lgcp_where2comm_area_mask_eval.py`
    - `feature_scale_summary.csv` 新增 `leader_once_cells` / `leader_once_bits_per_frame`。
    - `frame_summary.csv` 新增 `second_hop_leader_once_bits` / `second_hop_leader_once_bytes`。
    - `summary.csv` 新增 `avg_second_hop_leader_once_bits_per_frame` / `avg_second_hop_leader_once_mbps`。
  - 语义说明：
    - `second_hop_feature_bits`：保留原 Where2comm non-ego CAV sender 口径，即每个非 ego agent 都上传被选 cells。
    - `second_hop_leader_once_bits`：LGCP leader-to-RSU lower-bound 口径，同一帧 union selected cells 只按一次 leader 汇总包计。
- 验证：

```powershell
conda run -n opencda python -m py_compile opencda/tools/lgcp_where2comm_area_mask_eval.py opencood/opencood/models/fuse_modules/fusion_in_one.py
```

- Top-5 area-objectness + dilation 1 leader-once rerun：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_area_mask_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_area_objectness_top5_11f_z2_dilate1_leader_once --fusion-method intermediate_where2comm --ego-cav-id -1 --reference-z-override 2.0 --max-frames 11 --max-areas-per-frame 5 --max-cavs 5 --grid-size-x 10 --grid-size-y 6 --eval-scope planned_areas --postprocess-score-threshold 0.05 --mask-mode lgcp_area_objectness --mask-dilation-cells 1
```

- 结果：
  - AP@0.3/0.5/0.7：`0.801557 / 0.658694 / 0.162600`
  - non-ego CAV sender second-hop：`16.180131 Mbps`
  - leader-once lower-bound second-hop：`4.045033 Mbps`
  - member-to-leader area point upload：`100766.545455 bytes/frame`
- 结论：
  - leader-once 口径将 Top-5 最优配置的二跳从 `16.18 Mbps` 降到 `4.05 Mbps`，符合“leader 先局部汇总，再上传 RSU”的 LGCP 通信叙事。
  - 该 lower-bound 还不是完整论文通信模型：后续仍需按 area packet 统计 mask/index metadata，并区分多个 leader / 多 area packet 可能带来的重复 cells。

## Where2comm per-leader box hierarchy diagnostic

- 时间：2026-07-22
- 目标：
  - 回答 Top-23 低 AP 是否仅由当前单次 `RSU + 4 CAV` 输入覆盖不足造成。
  - 复用 LGCP assignment plan 中每个 area 的真实 `leader_id/group_members`，逐 area 调用 Where2comm checkpoint，再由 RSU 做 box-level late fusion。
  - 修正旧 box-level evaluator 的 GT 口径：新增 global planned-area GT，避免逐 leader-local GT 拼接造成重复或与 RSU planned-area 口径不一致。
- 新增 / 修改：
  - `opencda/tools/lgcp_hierarchy_late_fusion_eval.py`
    - 支持 `_dataset_root_override`，可直接加载外部 `intermediate_where2comm` checkpoint。
    - 新增 `--postprocess-score-threshold`。
    - 新增 `--global-gt-cav-id` / `--global-reference-z-override`，用于每帧生成单份 RSU/global planned-area GT。
    - global GT 生成时按 OpenCOOD dataset `max_cav` 限制候选 CAV，避免 Where2comm `max_cav=5` pairwise matrix 越界。
- 验证：

```powershell
conda run -n opencda python -m py_compile opencda/tools/lgcp_hierarchy_late_fusion_eval.py
```

- Top-23 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_box_hierarchy_top23_11f_thr005_globalgt --fusion-method intermediate_where2comm --max-frames 11 --max-areas-per-frame 23 --postprocess-score-threshold 0.05 --global-gt-cav-id -1 --global-reference-z-override 2.0 --grid-size-x 10 --grid-size-y 6
```

- Top-5 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_hierarchy_late_fusion_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_box_hierarchy_top5_11f_thr005_globalgt --fusion-method intermediate_where2comm --max-frames 11 --max-areas-per-frame 5 --postprocess-score-threshold 0.05 --global-gt-cav-id -1 --global-reference-z-override 2.0 --grid-size-x 10 --grid-size-y 6
```

- 结果：

| Scope | Assignment rows | Cached group calls | GT | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5 | 55 | 55 | 85 | 75 | 0.264252 | 0.074938 | 0.007897 |
| Top-23 | 253 | 210 | 290 | 203 | 0.165041 | 0.056061 | 0.002896 |

- 结论：
  - 该 diagnostic 确认了一个重要边界：Where2comm checkpoint 的有效性能来自 intermediate feature fusion，而不是每个 leader 独立检测后做 box-level late fusion。
  - 即使 Top-5 使用真实 leader group，box-level hierarchy 的 AP@0.5 也只有 `0.074938`，远低于 Top-5 single-pass area-objectness intermediate fusion 的 `0.658694`。
  - 因此，Top-23 不能靠“多 leader 检测框拼接”补救；下一步必须做 leader feature packet 级汇总，即 leader local point/feature aggregation 后，将 feature packet 上传 RSU 再做 feature-level fusion / detection。

## Where2comm leader point aggregation -> feature packet -> RSU feature fusion

- 时间：2026-07-22
- 目标：
  - 按当前确认的 LGCP 语义实现真正的两跳模型级 pipeline：
    - 第一跳：member CAV 上传 point-cloud area slice 给 leader。
    - leader：合并 leader 自己点云与 member area slices，并编码成 feature packet。
    - 第二跳：leader feature packet 上传 RSU。
    - RSU：对多个 leader feature packets 做 Where2comm multiscale feature fusion，再跑 detection heads。
- 新增：
  - `opencda/tools/lgcp_where2comm_leader_feature_fusion.py`
    - 每个 `area_assignment_plan.csv` row 生成一个 leader packet。
    - packet 内点云来自该 area 的 `leader_id/group_members`，统一投影到 RSU/reference pose。
    - 使用 `where2comm_10e` 的 `pillar_vfe -> scatter -> backbone -> multiscale features` 编码 leader packets。
    - RSU 侧使用 `fusion_in_one.Where2comm` 的 attentive feature fusion，支持 `lgcp_area_objectness` 外部 mask 交集模式。
    - 输出 `frame_summary.csv`、`leader_packets.csv`、`feature_scale_summary.csv`、`summary.csv`。
- 验证：

```powershell
conda run -n opencda python -m py_compile opencda/tools/lgcp_where2comm_leader_feature_fusion.py
```

- Top-5 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_leader_feature_fusion --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_leader_feature_top5_11f_areaobj_dilate1_refposefix --fusion-method intermediate_where2comm --reference-cav-id -1 --reference-z-override 2.0 --max-frames 11 --max-areas-per-frame 5 --grid-size-x 10 --grid-size-y 6 --query-mode mean --mask-mode lgcp_area_objectness --mask-dilation-cells 1 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- Top-23 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_leader_feature_fusion --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_leader_feature_top23_11f_areaobj_dilate1_refposefix --fusion-method intermediate_where2comm --reference-cav-id -1 --reference-z-override 2.0 --max-frames 11 --max-areas-per-frame 23 --grid-size-x 10 --grid-size-y 6 --query-mode mean --mask-mode lgcp_area_objectness --mask-dilation-cells 1 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 结果：

| Scope / setting | Query | Threshold | Valid leader packets/frame | GT | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 | Member upload KB/frame | Second-hop Mbps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5 | mean | 0.05 | 3.73 | 85 | 102 | 0.708405 | 0.599561 | 0.038212 | 18.26 | 7.274124 |
| Top-5 | mean | 0.01 | 3.73 | 85 | 187 | 0.486070 | 0.398927 | 0.029329 | 18.26 | 7.274124 |
| Top-5 | first_leader | 0.05 | 3.73 | 85 | 102 | 0.583483 | 0.484305 | 0.051957 | 18.26 | 5.123724 |
| Top-5 | zero | 0.05 | 3.73 | 85 | 101 | 0.588811 | 0.434281 | 0.027582 | 18.26 | 7.274124 |
| Top-23 | mean | 0.05 | 17.36 | 290 | 55 | 0.157367 | 0.157367 | 0.000123 | 47.99 | 25.600930 |
| Top-23 | mean | 0.01 | 17.36 | 290 | 361 | 0.220698 | 0.106400 | 0.001051 | 47.99 | 25.600930 |

- 结论：
  - 该工具首次实现了“member 点云 area slice -> leader point aggregation -> leader feature packet -> RSU feature fusion/detection”的完整 offline 闭环。
  - Top-5 下 AP@0.5 `0.599561`，已经明显高于 per-leader box hierarchy 的 `0.074938`，证明 feature-level RSU fusion 是正确方向。
  - Top-5 leader-packet AP@0.5 仍低于 CAV-level single-pass `area ∩ objectness + dilation1` 的 `0.658694`，但通信更接近 LGCP 语义：member-to-leader 只统计成员点云上传约 `18.26 KB/frame`，leader 自有点云不计入第一跳，leader-to-RSU feature 为 `7.27 Mbps`。
  - Top-23 解除 `max_cav=5` 后仍只有 AP@0.5 `0.157367`，说明大范围问题不只是 agent count，而是 untrained RSU query / leader-packet feature distribution / multi-packet fusion 语义不匹配。
  - score threshold `0.01` 在 Top-23 让预测数从 `55` 增到 `361`，但 AP@0.5 降到 `0.106400`，因此不是简单 threshold calibration。
  - 下一步若要让 Top-23 成为主结果，需要微调或训练 RSU feature aggregation，而不是继续调 box-level late fusion 或单纯放宽 threshold。

## Where2comm 4-leader reassignment + leader-granularity feature packet diagnostic

- 时间：2026-07-22
- 目标：
  - 按用户设想把 LGCP 参数改为每帧最多 4 个 Leader。
  - 让这 4 个 Leader 接管更多 members 与 areas。
  - RSU 侧不再接收一堆 per-area feature packet，而是每个 Leader 合并其负责的多个 areas 后上传一个 feature packet。
- 新增/修改：
  - `opencda/tools/lgcp_reassign_limited_leaders.py`
    - 读取既有 `area_assignment_plan.csv`。
    - 每帧按 `priority_sum` 选出 4 个候选 Leader。
    - 将 23 个 selected areas 重新分配给这 4 个 Leader，代价包含 leader 到 area center 距离、当前负载惩罚和原 group member bonus。
    - 输出不覆盖原始 plan 的 `area_assignment_plan.csv` 与 `leader_reassignment_summary.csv`。
  - `opencda/tools/lgcp_where2comm_leader_feature_fusion.py`
    - 新增 `--packet-granularity area|leader`。
    - `leader` 模式下同一 Leader 负责的多个 area rows 会合并成一个 packet，点云合并后编码，area mask 使用多个 area 的 union。
- 4-Leader plan command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_reassign_limited_leaders --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260718_lgcp_carla_hierarchy_plan_area23_11f\area_assignment_plan.csv --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_carla_hierarchy_plan_area23_4leaders --max-leaders 4 --max-areas-per-frame 23 --leader-score priority_sum --load-weight 8.0 --member-bonus 30.0
```

- 评估 command：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_leader_feature_fusion --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_carla_hierarchy_plan_area23_4leaders\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_leader_feature_top23_4leaders_11f_areaobj_dilate1 --fusion-method intermediate_where2comm --reference-cav-id -1 --reference-z-override 2.0 --max-frames 11 --max-areas-per-frame 23 --grid-size-x 10 --grid-size-y 6 --query-mode mean --packet-granularity leader --mask-mode lgcp_area_objectness --mask-dilation-cells 1 --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 结果：

| Setting | Areas/frame | Leaders/frame | Valid leader packets/frame | GT | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 | Member upload KB/frame | Leader own KB/frame | Second-hop Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-23 original per-area packets | 23 | 17.36 packets | 17.36 | 290 | 55 | 0.157367 | 0.157367 | 0.000123 | 47.99 | n/a | 25.600930 |
| Top-23 reassigned 4 leaders, leader packets, thr 0.05 | 23 | 4 | 3.73 | 290 | 305 | 0.650372 | 0.412919 | 0.021884 | 227.50 | 119.04 | 24.865513 |
| Top-23 reassigned 4 leaders, leader packets, thr 0.01 | 23 | 4 | 3.73 | 290 | 432 | 0.578643 | 0.357752 | 0.016061 | 227.50 | 119.04 | 24.865513 |
| Top-23 reassigned 5 leaders, leader packets, thr 0.05 | 23 | 5 | 4.27 | 290 | 305 | 0.642838 | 0.409177 | 0.030923 | 191.92 | 154.20 | 24.886924 |
| Top-23 reassigned 6 leaders, leader packets, thr 0.05 | 23 | 6 | 5.09 | 290 | 302 | 0.658934 | 0.452508 | 0.026470 | 154.50 | 191.46 | 24.707258 |

- 结论：
  - 4-Leader reassignment 明显缓解了原 Top-23 per-area packet 语义崩坏，AP@0.5 从 `0.157367` 提升到 `0.412919`。
  - 5-Leader reassignment 的 AP@0.5 为 `0.409177`，与 4-Leader 的 `0.412919` 基本持平略低；AP@0.7 从 `0.021884` 小幅升至 `0.030923`。
  - 6-Leader reassignment 的 AP@0.5 进一步升至 `0.452508`，第一跳 member upload 降至 `154.50 KB/frame`；这说明在当前启发式分配下，增加 Leader 可以减少 leader-local aggregation 噪声并降低 member relay 负载。
  - 4/5/6-Leader 结果仍低于 Top-5 leader-packet 的 AP@0.5 `0.599561`，说明大范围 Top-23 仍存在 RSU global aggregation / feature distribution mismatch。
  - 0.01 threshold 让预测数从 `305` 增到 `432`，AP@0.5 下降到 `0.357752`，所以当前最优点仍是 threshold `0.05`。
  - 4-Leader route 更接近 checkpoint 的 `max_cav=5` 输入假设：RSU ego + 最多 4 个 leader packets。5/6-Leader 会在部分帧形成 `ego + 5/6 leader packets`，超过训练时常见输入语义，但代码路径可运行。
  - 第一跳 member-to-leader 上传从原 Top-23 的 `47.99 KB/frame` 增至 4-Leader `227.50 KB/frame`、5-Leader `191.92 KB/frame`、6-Leader `154.50 KB/frame`。这条趋势说明 Leader 数越多，成员被强制转发给远端 Leader 的代价越低，但第二跳 feature Mbps 仍基本维持在 `24-25 Mbps`。
  - 下一步可以在 3/4/5/6/7 Leader、不同 `load_weight/member_bonus` 和 area budget 上做 sweep，寻找 AP 与第一跳 upload 的平衡点。

## Where2comm CAV count limit probe

- 时间：2026-07-22
- 目标：
  - 明确 `C:\Workspace\OpenCOOD\checkpoints\where2comm_10e` 的 CAV 数上限，后续作为 LGCP Leader 数上限的依据。
  - 区分三种口径：checkpoint/YAML 声明上限、当前代码路径实际 runtime 上限、当前 LGCP 场景下的 AP/通信实用上限。
- 配置审计：
  - `config.yaml` 中 `model.args.max_cav = 5`。
  - 同一 YAML 中 `train_params.max_cav = 2`，但 OpenCDA `OpenCOODManager` 当前会将 `train_params.max_cav` 对齐到 `model.args.max_cav`，因此 canonical OpenCDA/OpenCOOD 数据入口的保守上限是 total CAV `5`。
  - Where2comm multiscale fusion 本身没有固定 agent-count 权重；`fusion_in_one.py` 通过 `record_len` 动态 `regroup`，attention 复杂度随 CAV 数增长，主要受显存/时间限制。
- 新增 probe：
  - `opencda/tools/lgcp_where2comm_cav_limit_probe.py`
  - 使用真实 Where2comm 三尺度 feature 尺寸 `96x352 / 48x176 / 24x88` 和通道数 `64 / 128 / 256`，直接调用 checkpoint 对应的 fusion modules。
  - 输出保存到：
    - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_where2comm_cav_limit_probe/synthetic_cav_limit.csv`
    - `docs/doc_workspace/LGCP/experiments/hierarchy_plan/20260722_lgcp_where2comm_cav_limit_probe/lgcp_leader_sweep_4_13.csv`
- Synthetic runtime 结果：

| Total CAV count | Result |
| ---: | --- |
| 5 / 8 / 10 / 13 / 16 / 20 / 24 / 32 | OK |
| 48 / 64 / 96 / 128 / 160 / 192 / 224 / 232 | OK |
| 234 / 236 / 240 / 256 | CUDA OOM |

- LGCP Top-23 leader sweep 结果：

| Max leaders | Valid leader packets/frame | AP@0.3 | AP@0.5 | AP@0.7 | Member upload KB/frame | Second-hop Mbps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3.727273 | 0.650372 | 0.412919 | 0.021884 | 227.50 | 24.865513 |
| 5 | 4.272727 | 0.642838 | 0.409177 | 0.030923 | 191.92 | 24.886924 |
| 6 | 5.090909 | 0.658934 | 0.452508 | 0.026470 | 154.50 | 24.707258 |
| 7 | 5.909091 | 0.669542 | 0.484321 | 0.026363 | 132.88 | 24.756596 |
| 8 | 6.454545 | 0.623809 | 0.459635 | 0.035736 | 99.98 | 24.767767 |
| 9 | 7.090909 | 0.606688 | 0.424237 | 0.056009 | 93.32 | 24.861789 |
| 10 | 7.909091 | 0.555164 | 0.393419 | 0.051890 | 70.54 | 25.187608 |
| 11 | 8.181818 | 0.553259 | 0.384355 | 0.045574 | 60.82 | 25.210880 |
| 12 | 8.181818 | 0.553259 | 0.384355 | 0.045574 | 55.68 | 25.210880 |
| 13 | 8.272727 | 0.541581 | 0.381057 | 0.043158 | 56.06 | 25.210880 |

- 结论：
  - 严格 checkpoint / dataset 声明口径：total CAV cap 为 `5`，LGCP Leader cap 应为 `4`。
  - 当前自定义 LGCP feature-packet runtime 口径：Where2comm fusion 算子没有 5-CAV 硬上限；当前 CUDA 环境用真实 feature 尺寸验证到 total CAV `232` 可运行，`234` 开始 OOM，因此 runtime-only Leader cap 可写为 `231`，但这没有训练语义保证。
  - 当前 LGCP Top-23 实验口径：在 4-13 Leader sweep 中，AP@0.5 最优是 `7` Leader，对应 `0.484321`；继续增加 Leader 会降低第一跳上传，但 AP@0.5 开始下降。
  - 后续建议：代码参数上可以设硬安全阈值 `max_total_cav_runtime=232`，但论文/实验默认不要用这个数。LGCP leader sweep 的实用上限建议先设为 `7`，同时保留 checkpoint-conservative 对照 `4`。

## All-20-CAV point cloud upload to RSU upper-bound diagnostic

- 时间：2026-07-22
- 背景确认：
  - `opencda/scenario_testing/config_yaml/lgcp_carla.yaml` 明确记录 `target_total_vehicle_num: 100`、`target_cav_num: 20`，并说明为 20 intelligent / managed vehicles + 80 unmanaged background vehicles。
  - 当前导出数据目录 `D:\Data\Carla\2026_07_15_02_33_21` 包含 `-1` RSU 与 `1..20` 共 20 个 CAV 目录，因此离线数据是 `20 CAV + 1 RSU`。
- 新增脚本：
  - `opencda/tools/lgcp_where2comm_all_cav_to_rsu_eval.py`
  - 支持两种诊断口径：
    - `per_cav_where2comm`：20 个 CAV 各自上传完整 raw LiDAR 到 RSU 坐标系，再作为 20 个 agent packets 做 Where2comm intermediate fusion。
    - `centralized_raw`：20 个 CAV 上传完整 raw LiDAR 后，RSU 先把点云合并为一个 centralized raw point cloud，再单次检测。
- 命令摘要：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_where2comm_all_cav_to_rsu_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_all20cav_to_rsu_11f_objectness_thr005 --fusion-method intermediate_where2comm --reference-cav-id -1 --reference-z-override 2.0 --max-frames 11 --query-mode mean --mask-mode objectness --postprocess-score-threshold 0.05

conda run -n opencda python -m opencda.tools.lgcp_where2comm_all_cav_to_rsu_eval --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_where2comm_all20cav_to_rsu_11f_centralized_raw_thr005 --fusion-method intermediate_where2comm --reference-cav-id -1 --reference-z-override 2.0 --max-frames 11 --query-mode first_leader --aggregation-mode centralized_raw --mask-mode objectness --postprocess-score-threshold 0.05
```

- 结果：

| Setting | Threshold | Valid packets/frame | GT | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 | Raw upload KB/frame | Feature Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 CAV per-CAV Where2comm | 0.05 | 19.00 | 542 | 343 | 0.590709 | 0.358292 | 0.021621 | 1603.89 | 512.565848 |
| 20 CAV per-CAV Where2comm | 0.01 | 19.00 | 542 | 723 | 0.514916 | 0.274701 | 0.014300 | 1603.89 | 512.565848 |
| 20 CAV centralized raw | 0.05 | 1.00 | 542 | 798 | 0.651840 | 0.487223 | 0.080560 | 1603.89 | 0.000000 |
| 20 CAV centralized raw | 0.01 | 1.00 | 542 | 875 | 0.616936 | 0.461427 | 0.075630 | 1603.89 | 0.000000 |
| LGCP Top-23 7-Leader feature packet | 0.05 | 5.91 | 290 planned-area | 289 | 0.669542 | 0.484321 | 0.026363 | 132.88 member upload | 24.756596 |

- 观察：
  - per-CAV Where2comm 全量点云上传并不自动变强，AP@0.5 只有 `0.358292`，低于 7-Leader 的 `0.484321`。原因是 RSU/global query + 20 agent feature fusion 明显偏离 checkpoint 训练语义。
  - centralized raw 是更贴近“点云全部传到 RSU 后集中处理”的 upper-bound 口径，AP@0.5 为 `0.487223`，只比 7-Leader 的 `0.484321` 高 `0.002902`。
  - centralized raw 的 AP@0.7 为 `0.080560`，明显高于 7-Leader 的 `0.026363`，说明全量点云集中处理确实改善了高 IoU box quality，但 AP@0.5 收益很小。
  - raw upload 成本约 `1.64 MB/frame`，约为 7-Leader member upload `132.88 KB/frame` 的 `12.1x`；因此当前 7-Leader LGCP 在 AP@0.5 上接近 all-raw centralized upper bound，但通信量低得多。
- 结论：
  - 当前场景确认为 `20 CAV + 1 RSU + 80 background vehicles`。
  - “20 CAV 全量点云传 RSU”最好的当前 AP@0.5 是 centralized raw `0.487223`。
  - 这为 LGCP 7-Leader 提供了一个很强的参照：LGCP 7-Leader AP@0.5 `0.484321`，几乎追平全量 centralized raw，但第一跳只需约 `132.88 KB/frame`，而全量 raw upload 约 `1603.89 KB/frame`。

## Attentive early checkpoint centralized raw upper-bound diagnostic

- 时间：2026-07-22
- 背景确认：
  - 上一节 `20 CAV centralized raw` 使用的是 `intermediate_where2comm` / `C:\Workspace\OpenCOOD\checkpoints\where2comm_10e` 的 PointPillar backbone/head，只是把 20 CAV raw point cloud 先合并成一个 RSU packet；它不是 SGCP 的 attentive early detector。
  - SGCP 已有 attentive checkpoint 移植的早期融合权重：`docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/pointpillar_early_from_attentive_weights`，配置入口为 `enable_coperception_early_from_attentive.yaml`。
- 新增脚本：
  - `opencda/tools/lgcp_attentive_early_all_cav_to_rsu_eval.py`
  - 口径：20 个 CAV 上传完整 raw LiDAR 到 RSU 坐标系，RSU 先合并为一个 centralized raw point cloud，再用 SGCP attentive-derived early PointPillar detector 单次检测。
  - 为避免 RSU 传感器 `z=12m` 与车载 checkpoint 的 z-range 不匹配，沿用 `--reference-z-override 2.0`，保留 RSU 的 x/y/yaw。
- 命令摘要：

```powershell
conda run -n opencda python opencda\tools\lgcp_attentive_early_all_cav_to_rsu_eval.py --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_attentive_early_all20cav_to_rsu_11f_centralized_raw_thr020 --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml --fusion-method early --reference-cav-id -1 --reference-z-override 2.0 --max-frames 11 --postprocess-score-threshold 0.20
```

- 结果：

| Setting | Threshold | Valid packets/frame | GT | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 | Raw upload KB/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 CAV centralized raw + attentive early detector | 0.05 | 1.00 | 542 | 617 | 0.799015 | 0.746692 | 0.424014 | 1603.89 |
| 20 CAV centralized raw + attentive early detector | 0.20 | 1.00 | 542 | 463 | 0.816923 | 0.779641 | 0.470207 | 1603.89 |
| 20 CAV centralized raw + Where2comm detector | 0.05 | 1.00 | 542 | 798 | 0.651840 | 0.487223 | 0.080560 | 1603.89 |
| LGCP Top-23 7-Leader feature packet | 0.05 | 5.91 | 290 planned-area | 289 | 0.669542 | 0.484321 | 0.026363 | 132.88 member upload |

- 观察：
  - 使用 attentive-derived early detector 后，centralized raw upper bound 明显升高，AP@0.5 从 Where2comm detector 的 `0.487223` 提升到 `0.779641`，AP@0.7 从 `0.080560` 提升到 `0.470207`。
  - 这说明此前 centralized raw AP 偏低主要来自 Where2comm checkpoint / detection route 与 centralized raw 单包语义不匹配，而不是“20 CAV 点云合并到 RSU”本身不可行。
  - 该结果是 all-raw centralized upper bound：通信量仍是 `1603.89 KB/frame`，不能直接作为 LGCP 低通信分层机制的性能结果，但可作为检测器选择和上界参照。

## SGCP attentive-derived checkpoint for leader BEV feature -> RSU AttFusion

- 时间：2026-07-22
- 目标：
  - 使用 SGCP `pointpillar_early_from_attentive_weights/latest.pth`，但以 `point_pillar_intermediate` / `AttBEVBackbone` 实例化模型，测试 `leader BEV feature -> RSU attentive fusion -> detection`。
  - 不修改 SGCP 代码，不覆盖原 checkpoint。
- 非破坏式配置：
  - 新增 LGCP model dir：`docs/doc_workspace/LGCP/experiments/model_dirs/pointpillar_intermediate_from_sgcp_attentive_early`
  - `config.yaml` 复制自 `opencood/logs/pointpillar_attentive_fusion/config.yaml`，保持 intermediate attentive model definition。
  - `latest.pth` 是指向 SGCP `pointpillar_early_from_attentive_weights/latest.pth` 的 hardlink；两者 tensor 已确认完全一致。
  - 新增 coperception YAML：`docs/doc_workspace/LGCP/experiments/model_dirs/enable_coperception_intermediate_from_sgcp_attentive_early.yaml`
- 代码改动：
  - `opencda/tools/lgcp_pointpillar_rsu_bev_fusion.py`
  - 新增 `--reference-z-override`，可用 RSU x/y/yaw 同时把 z 调到车载 checkpoint 的有效 lidar range。
  - 新增 `--packet-granularity area|leader`。默认 `area` 保持旧行为；`leader` 会把同一 leader 接管的多个 area 合并成一个 leader BEV feature packet。
- 命令摘要：

```powershell
conda run -n opencda python -m opencda.tools.lgcp_pointpillar_rsu_bev_fusion --dataset-root D:\Data\Carla --scenario-id 2026_07_15_02_33_21 --assignment-plan docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_carla_hierarchy_plan_area23_7leaders\area_assignment_plan.csv --output-dir docs\doc_workspace\LGCP\experiments\hierarchy_plan\20260722_lgcp_sgcp_attentive_intermediate_rsu_bev_top23_7leaders_11f_z2_thr005_leaderpkt --coperception-yaml docs\doc_workspace\LGCP\experiments\model_dirs\enable_coperception_intermediate_from_sgcp_attentive_early.yaml --fusion-method intermediate_attentive --reference-cav-id -1 --reference-z-override 2.0 --max-frames 11 --grid-size-x 10 --grid-size-y 6 --query-mode mean --packet-granularity leader --eval-scope planned_areas --postprocess-score-threshold 0.05
```

- 结果：

| Setting | Packet granularity | Valid leader packets/frame | GT | Pred samples | AP@0.3 | AP@0.5 | AP@0.7 | Member upload KB/frame | Sparse BEV feature KB/frame | Dense full BEV MB/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5 first 5 areas | area | 3.73 | 93 | 95 | 0.940917 | 0.828329 | 0.534530 | 4.95 | 83.00 | 64.07 |
| Top-23 original assignment | area | 17.36 | 313 | 42 | 0.134185 | 0.134185 | 0.102179 | 26.81 | 340.80 | 298.47 |
| Top-23 7-Leader reassignment | leader | 5.91 | 313 | 223 | 0.663529 | 0.556226 | 0.252941 | 88.42 | 339.25 | 101.56 |

- 观察：
  - Top-5 下该路线表现很强，AP@0.5 `0.828329`、AP@0.7 `0.534530`，说明 SGCP/attentive 权重以 intermediate model 形式用于 RSU BEV fusion 是可行的。
  - Top-23 原始 per-area packet 退化严重，AP@0.5 只有 `0.134185`，原因是每帧 17-18 个 area packets 偏离 checkpoint 的 ego-first / limited-agent 训练语义。
  - Top-23 7-Leader leader-packet 版恢复到 AP@0.5 `0.556226`、AP@0.7 `0.252941`，明显优于原始 per-area packet，也高于 Where2comm 7-Leader 的 AP@0.5 `0.484321`。
  - dense full BEV canvas 传输不可接受；当前可讨论的 feature 通信口径只能是 sparse nonzero BEV cells 或进一步压缩后的 feature packet。

## Small LGCP Town03 roundabout dataset

- 时间：2026-07-22
- 背景：
  - LGCP 论文正文的实验设置写到：OPV2V 场景包含传统车辆和 `2 to 7` 个 CAV；V2XSet 场景包含传统车辆和 `2 to 7` 个智能体；检测范围设置为 `280m x 80m`，RoI grid 为 `10m x 6m`；CARLA/OpenCDA/NS3 co-simulation 用于进一步研究 `5 to 30` 个 CAV 的多车部署。
  - 当前 `lgcp_carla` 是 `20 CAV + 1 RSU + 80 background vehicles`，对模型级 neural feature fusion 偏难，也偏离 OpenCOOD checkpoint 常见训练智能体数量。
- 新增小场景：
  - `opencda/scenario_testing/lgcp_carla_small.py`
  - `opencda/scenario_testing/config_yaml/lgcp_carla_small.yaml`
  - Town03 环岛与 RSU 位置保持不变。
  - 车辆规模：`8 CAV + 28 background vehicles = 36 vehicles`。
  - RoI：`120m x 60m`，grid 仍为 `10m x 6m`。
- 数据导出：

```powershell
$env:OPENCDA_DATA_DUMP_ROOT='D:\Data\Carla'
$env:OPENCDA_DATADUMP_TICKS='100'
$env:OPENCDA_CLEAN_WORLD_ON_INIT='1'
$env:OPENCDA_CARLA_CLIENT_TIMEOUT='180'
$env:OPENCDA_USE_CURRENT_CARLA_WORLD='1'
conda run -n opencda python opencda.py -t lgcp_carla_small --dump --debug
```

- 离线数据集：
  - `D:\Data\Carla\2026_07_22_20_04_41`
  - agent folders：`-1` RSU + `1..8` CAV。
  - saved frames：`21`，timestamps `000060..000100`。
  - offline smoke：early detector 首帧输出 `30 pred / 36 GT`。
- 小场景 LGCP 中间文件：
  - area confidence：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_carla_small_area_confidence_21f`
  - hierarchy plan：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_carla_small_hierarchy_plan_top10_21f`
  - Top-10 plan summary：`area_count_mean=10`，`avg_group_size_mean=1.552381`，`leader_count_mean=5.714286`，`leader_max_load_mean=4.666667`，`total_byte_proxy_mean=77.88KB/frame`。
- SGCP attentive leader-BEV route on small scene:
  - output：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_small_sgcp_attentive_intermediate_rsu_bev_top10_21f_z2_thr005_leaderpkt`
  - `packet_granularity=leader`，`query_mode=mean`，`eval_scope=planned_areas`。
  - AP@0.3/AP@0.5/AP@0.7：`0.800311 / 0.755478 / 0.393690`
  - GT / pred samples：`294 / 247`
  - member-to-leader upload：`272448 bytes total`，约 `12.67KB/frame`
  - sparse BEV feature：`5046784 bytes total`，约 `234.69KB/frame`
- 结论：
  - 小场景成功构造并导出为 OPV2V-style 离线数据集。
  - 该场景明显更适合调试 LGCP model-level feature fusion：智能体数量接近 OPV2V/V2XSet，planned-area AP@0.5 达到 `0.755478`，AP@0.7 达到 `0.393690`。
  - 论文表述中应将其标为 small-scale diagnostic / model-mechanism validation，不能替代 `5 to 30 CAV` co-simulation 的 scalability evidence。

## Easy Town03 ordinary-intersection diagnostic scene

- 时间：2026-07-22
- 背景：
  - 用户指出 `lgcp_carla_small` 仍是环岛，希望改成普通十字路口，并希望最佳 early fusion AP@0.3 达到 `0.90+`。
  - 第一版普通路口配置使用矩形 `range` 采样，CARLA `get_waypoint()` 会把矩形内部分采样点吸附到相邻居民区道路，导致车辆偏离十字路口。
- 当前修正：
  - 新增 `opencda/scenario_testing/lgcp_carla_intersection_easy.py`。
  - 新增 `opencda/scenario_testing/config_yaml/lgcp_carla_intersection_easy.yaml`。
  - 当前配置选择 Town03 普通 junction 近似中心 `(1.10, 133.72)`，不再使用矩形 `range`，背景车全部使用显式 CARLA spawn points。
- 导出与评估：
  - `D:\Data\Carla\2026_07_22_20_26_17`：4 CAV + 6 background，36 帧；default early with RSU AP@0.3/0.5/0.7 = `0.76 / 0.75 / 0.68`。
  - `D:\Data\Carla\2026_07_22_20_31_56`：4 CAV + 0 background，26 帧；default early CAV-only AP@0.3/0.5/0.7 = `0.69 / 0.69 / 0.69`。
  - `D:\Data\Carla\2026_07_22_20_37_27`：2 CAV + 0 background，21 帧；default early CAV-only AP@0.3/0.5/0.7 = `0.52 / 0.52 / 0.52`。
  - `D:\Data\Carla\2026_07_22_20_44_26`：当前 2 CAV + 16 background 显式点位版本，21 帧；default early with RSU AP@0.3/0.5/0.7 = `0.75 / 0.73 / 0.52`；排除 RSU 为 `0.42 / 0.41 / 0.17`；SGCP attentive-derived early checkpoint 为 `0.70 / 0.70 / 0.53`。
- 诊断：
  - 车辆偏离路口的问题已由显式 spawn points 修复。
  - AP@0.3 `>= 0.90` 尚未达成。object diagnostics 显示多数 GT 可以匹配，但稳定高分 false positives / 低分 true positives 压低 AP。
  - 已给 `opencda/tools/offline_inference.py` 增加 `--postprocess-nms-thresh` 诊断开关；强 NMS 未能去除主导 FP，说明这些 FP 不是简单重叠重复框。

## Easy intersection 10-CAV / 10-background update

- 时间：2026-07-22
- 用户判断：`2 CAV + 16 background` 中 CAV 数量过少，融合 AP 偏低可以理解；需要改为 `10 CAV + 10 background`。
- 配置更新：
  - `opencda/scenario_testing/config_yaml/lgcp_carla_intersection_easy.yaml`
  - `target_cav_num=10`
  - `target_total_vehicle_num=20`
  - `single_cav_list` 扩展到 10 个显式 CAV spawn points。
  - `carla_traffic_manager.vehicle_list` 保留 10 个显式 background spawn points。
  - 仍然不使用矩形 `range`，避免车辆被吸附到居民区。
- 数据导出：
  - dataset：`D:\Data\Carla\2026_07_22_22_00_04`
  - frames：`21`
  - CAV dirs：`10`
  - runtime log：`CARLA traffic flow generated, with 10 vehicles and 0 vms`
- Early fusion AP probes:
  - default early + RSU, score threshold `0.20`：AP@0.3/0.5/0.7 = `0.86 / 0.86 / 0.77`
  - threshold sweep：`0.10` -> `0.76 / 0.76 / 0.68`; `0.15` -> `0.85 / 0.85 / 0.75`; `0.25` -> `0.84 / 0.84 / 0.75`
  - SGCP attentive-derived early + RSU, threshold `0.20`：`0.86 / 0.86 / 0.78`
  - default early, 10 CAV-only excluding RSU：`0.72 / 0.72 / 0.63`
- 结论：
  - 增加到 10 CAV 后，ordinary intersection AP@0.3 从 `0.75` 提升到 `0.86`，说明 CAV 数量确实是主要限制之一。
  - 当前仍未达到 AP@0.3 `0.90+` gate，但已经接近；下一步可继续微调固定点位或换更干净的普通路口。

## Easy intersection LGCP hierarchy run

- 时间：2026-07-22
- 口径确认：
  - 当前 ordinary-intersection 10-CAV 场景的 full early fusion 上界采用最佳已测值：`0.86 / 0.86 / 0.78`。
  - 这是 full-scene early-fusion sanity upper bound；LGCP planned-area 结果和 full-scene 结果需要分开解释。
- 输入数据：
  - dataset：`D:\Data\Carla\2026_07_22_22_00_04`
  - config：`opencda/scenario_testing/config_yaml/lgcp_carla_intersection_easy.yaml`
  - scale：`10 CAV + 1 RSU + 10 background vehicles`
- Area confidence：
  - output：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_area_confidence_21f`
  - records：`19880`
  - 每帧 RoI GT objects：`4`
  - recall@0.5：每帧 `1.000000`
- Hierarchy plan：
  - output：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_hierarchy_plan_top10_21f`
  - Top areas/frame：`10`
  - avg group size：`1.700000`
  - local upload packets/frame：`7.000000`
  - leader upload packets/frame：`10.000000`
  - leader count/frame：`8.000000`
  - leader max load：`4.000000`
  - plan byte proxy：`92640 bytes/frame`
- SGCP attentive leader-BEV -> RSU attentive fusion：
  - planned-area output：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_sgcp_attentive_intermediate_rsu_bev_top10_21f_z2_thr005_leaderpkt`
  - full-scope output：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_sgcp_attentive_intermediate_rsu_bev_top10_21f_z2_thr005_leaderpkt_fullscope`
  - packet granularity：`leader`
  - query mode：`mean`
  - reference：RSU `-1` with `--reference-z-override 2.0`
  - planned-area AP@0.3/0.5/0.7：`0.868668 / 0.797311 / 0.733363`
  - full-scope AP@0.3/0.5/0.7：`0.813771 / 0.746923 / 0.687017`
  - first-hop member upload：`377888 bytes total`，约 `17.57KB/frame`
  - second-hop sparse BEV feature：`4539136 bytes total`，约 `211.08KB/frame`
  - dense full BEV feature：`3027763200 bytes total`，约 `137.50MB/frame`，仅作为不可用上界。
- 结论：
  - 该普通十字路口 10-CAV 场景已经适合作为 LGCP model-mechanism validation 场景。
  - 在 full-scope 口径下，LGCP leader-BEV 路线相对 full early upper bound `0.86/0.86/0.78` 保留了大部分 AP。
  - planned-area 口径下 AP@0.3 `0.868668`、AP@0.7 `0.733363`，说明 Top-10 area 的局部到全局链路质量较好。
  - 下一步重点不是继续证明能跑通，而是降低 second-hop sparse feature bytes，或给 feature packet compression / leader count / area budget 做 sweep。

## Easy intersection limited-Leader sweep

- 时间：2026-07-22
- 用户约束：10-CAV 普通十字路口场景中，Leader 数量限制为至多 5 个；尝试 `K=3/4/5`。
- 输入：
  - dataset：`D:\Data\Carla\2026_07_22_22_00_04`
  - base plan：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_hierarchy_plan_top10_21f/area_assignment_plan.csv`
  - reassignment：`opencda.tools.lgcp_reassign_limited_leaders --load-weight 8 --member-bonus 30`
  - fusion：`opencda.tools.lgcp_pointpillar_rsu_bev_fusion`
  - checkpoint route：`docs/doc_workspace/LGCP/experiments/model_dirs/enable_coperception_intermediate_from_sgcp_attentive_early.yaml`
  - fusion method：`intermediate_attentive`
  - packet granularity：`leader`
  - reference：RSU `-1` with `--reference-z-override 2.0`
  - frames：`21`
- 输出：
  - K=3 plan：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_hierarchy_plan_top10_3leaders_21f`
  - K=4 plan：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_hierarchy_plan_top10_4leaders_21f`
  - K=5 plan：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_hierarchy_plan_top10_5leaders_21f`
  - K=3 planned/full：`20260722_lgcp_intersection10_sgcp_attentive_intermediate_rsu_bev_top10_3leaders_21f_z2_thr005_leaderpkt` / `_fullscope`
  - K=4 planned/full：`20260722_lgcp_intersection10_sgcp_attentive_intermediate_rsu_bev_top10_4leaders_21f_z2_thr005_leaderpkt` / `_fullscope`
  - K=5 planned/full：`20260722_lgcp_intersection10_sgcp_attentive_intermediate_rsu_bev_top10_5leaders_21f_z2_thr005_leaderpkt` / `_fullscope`

| Max leaders | Leaders | Area load | Avg group size | Max group size | Planned AP@0.3/0.5/0.7 | Full AP@0.3/0.5/0.7 | Member KB/frame | Sparse BEV KB/frame | Dense full BEV MB/frame |
| ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 3 | `6;9;8` | `6:5;9:4;8:1` | 2.10 | 4 | `0.809229/0.798743/0.660574` | `0.710024/0.700737/0.578433` | 68.28 | 218.57 | 51.56 |
| 4 | `6;9;8;4` | `6:3;9:3;8:1;4:3` | 2.00 | 4 | `0.865886/0.798881/0.744764` | `0.811164/0.748394/0.697697` | 48.40 | 215.55 | 68.75 |
| 5 | `6;9;8;4;3` | `6:3;9:2;8:1;4:2;3:2` | 1.80 | 4 | `0.865886/0.804755/0.752848` | `0.811164/0.753897/0.705270` | 36.37 | 214.58 | 85.94 |

- 结论：
  - K=3 过于激进：full-scope AP@0.5 降到 `0.700737`，AP@0.7 降到 `0.578433`。
  - K=4 已基本恢复到原 8-Leader full-scope baseline：`0.811164/0.748394/0.697697` vs `0.813771/0.746923/0.687017`。
  - K=5 相比 K=4 进一步提高 AP@0.5/AP@0.7 到 `0.753897/0.705270`，且第一跳 member upload 进一步降到 `36.37KB/frame`。
  - 由于当前 sparse BEV upload 是按非零 cell 计，K=3/4/5 的二跳 sparse BEV 约 `214-219KB/frame`，差异很小；dense full BEV 会随 Leader 数线性增长，仅作为不可用上界。
  - 在“最多 5 个 Leader”约束下，K=5 是当前推荐点；K=4 可作为更保守的 checkpoint-friendly 对照。

## Easy intersection K=5 Where2comm leader-feature check

- 时间：2026-07-22
- 目的：保持 ordinary-intersection 10-CAV Top-10 / K=5 Leader assignment 不变，将 RSU feature-level fusion 从 SGCP attentive-derived route 换成 Where2comm checkpoint，检查 AP 与通信量。
- 输入：
  - dataset：`D:\Data\Carla\2026_07_22_22_00_04`
  - assignment：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_hierarchy_plan_top10_5leaders_21f/area_assignment_plan.csv`
  - checkpoint：`C:\Workspace\OpenCOOD\checkpoints\where2comm_10e`
  - command entry：`opencda.tools.lgcp_where2comm_leader_feature_fusion`
  - `--max-areas-per-frame 10`
  - `--packet-granularity leader`
  - `--mask-mode lgcp_area_objectness --mask-dilation-cells 1`
  - `--query-mode mean`
  - reference：RSU `-1` with `--reference-z-override 2.0`
- 输出：
  - planned-area：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_where2comm_leader_feature_top10_5leaders_21f_areaobj_dilate1_planned`
  - full-scope：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_where2comm_leader_feature_top10_5leaders_21f_areaobj_dilate1_fullscope`

| Method | Scope | Leaders | Area load | Avg group | Member KB/frame | 2nd-hop feature | 2nd-hop KB/frame | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred samples |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Where2comm leader-feature | planned areas | `6;9;8;4;3` | `6:3;9:2;8:1;4:2;3:2` | 1.80 | 36.37 | 15.083032 Mbps | 184.12 | 0.759775 | 0.743407 | 0.381554 | 252 | 344 |
| Where2comm leader-feature | full scope | `6;9;8;4;3` | `6:3;9:2;8:1;4:2;3:2` | 1.80 | 36.37 | 15.083032 Mbps | 184.12 | 0.643896 | 0.629983 | 0.323233 | 269 | 386 |

- Planned vs Full：
  - `planned areas`：只评估 LGCP Top-10 selected areas 内的预测框和 GT；它回答“LGCP 已决定关注并上传的区域里，融合质量如何”。
  - `full scope`：对整帧/整 RoI 的预测框和 GT 评估；未被 Top-10 覆盖的目标会计入漏检，选中区域外的预测会计入误检；它更接近论文主表需要的 scene-level perception quality。
- 结论：
  - Where2comm K=5 的 AP@0.5 尚可：planned-area `0.743407`，full-scope `0.629983`。
  - AP@0.7 明显弱于 SGCP attentive-derived K=5：Where2comm full `0.323233` vs attentive full `0.705270`。
  - Where2comm 的二跳 feature 通信约 `184.12KB/frame`，低于 attentive sparse BEV 的 `214.58KB/frame`，但质量损失较大。
  - 当前 K=5 主线仍建议使用 SGCP attentive-derived leader-BEV route；Where2comm 可作为 communication-aware checkpoint 对照。

## Easy intersection coarse-area rerun

- 时间：2026-07-22
- 动机：用户指出当前 `90m x 70m` RoI 被 `10m x 6m` grid 切成约 `108` 个小 area，虽然只选择 Top-10，但每个 area 内点云太少，使“原始点云应比中间特征更大”的通信直觉不成立。
- 配置更新：
  - 文件：`opencda/scenario_testing/config_yaml/lgcp_carla_intersection_easy.yaml`
  - `lgcp.roi.grid_size`：从 `[10.0, 6.0]` 改为 `[30.0, 24.0]`
  - RoI 保持 `[90.0, 70.0]`，因此理论总 area 数为 `ceil(90/30) x ceil(70/24) = 3 x 3 = 9`
  - 只修改 LGCP area 划分；LiDAR debug grid 仍保持 `10.0`，避免影响可视化和此前传感器调试。
- 离线复算：
  - area confidence：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_bigarea_area_confidence_21f`
  - all-area plan：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_bigarea_hierarchy_plan_all9_21f`
  - K=5 reassignment：`docs/doc_workspace/LGCP/experiments/small_scene/20260722_lgcp_intersection10_bigarea_hierarchy_plan_all9_5leaders_21f`
  - attentive planned/full：`20260722_lgcp_intersection10_bigarea_sgcp_attentive_all9_5leaders_21f_z2_thr005_leaderpkt` / `_fullscope`
  - Where2comm planned/full：`20260722_lgcp_intersection10_bigarea_where2comm_all9_5leaders_21f_areaobj_dilate1_planned` / `_fullscope`
- Area / hierarchy summary：
  - area records：`2079 = 21 frames x 9 areas x 11 agents`
  - all-area plan：`9` areas/frame，`4` member-to-leader packets/frame，`9` leader uploads/frame
  - K=5 leaders：`8;1;2;10;6`
  - area load：`8:2;1:2;2:2;10:2;6:1`
  - avg group size：`1.777778`

| Route | Scope | Areas/frame | Leaders | Member KB/frame | 2nd-hop feature KB/frame | 2nd-hop Mbps | AP@0.3 | AP@0.5 | AP@0.7 | GT | Pred |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP attentive-derived | planned areas | 9 | 5 | 94.60 | 916.46 |  | 0.618795 | 0.610993 | 0.583450 | 264 | 446 |
| SGCP attentive-derived | full scope | 9 | 5 | 94.60 | 916.46 |  | 0.607293 | 0.599637 | 0.572605 | 269 | 446 |
| Where2comm `area_objectness+dilation1` | planned areas | 9 | 5 | 94.60 | 382.02 | 31.294902 | 0.480339 | 0.480339 | 0.365534 | 243 | 532 |
| Where2comm `area_objectness+dilation1` | full scope | 9 | 5 | 94.60 | 382.02 | 31.294902 | 0.470654 | 0.470654 | 0.358164 | 248 | 532 |
| Where2comm `area_objectness+dilation0` | planned areas | 9 | 5 | 94.60 | 359.56 | 29.455116 | 0.480319 | 0.480319 | 0.371061 | 243 | 532 |

- 结论：
  - 目标“总 area 数约 10”已达成：当前实际为 `9` 个大 area。
  - 第一跳 member 点云从旧 K=5 的 `36.37KB/frame` 升到 `94.60KB/frame`，说明大 area 确实让 raw point slice 更大。
  - 但二跳中间特征仍高于第一跳：attentive sparse BEV 为 `916.46KB/frame`，Where2comm 为 `382.02KB/frame`；不加 dilation 也仍为 `359.56KB/frame`。
  - 这说明问题不只是 area 过小。当前 feature packet 统计随 BEV cell coverage 增长较快，Where2comm 虽有 objectness 选择，但在大 area 下 mask 仍覆盖过多 cells。
  - 下一步若要符合“中间特征显著小于原始点云”的论文直觉，需要继续做 feature compression / stricter objectness threshold / top-k cell budget / quantization，而不是单纯继续放大 area。

## Where2comm feature-size accounting note

- 时间：2026-07-23
- 目的：回答 Where2comm 如何筛选 BEV feature，以及 raw point cloud、直接生成的 dense BEV feature、Where2comm sparse feature 三者大小何时满足 `raw < feature` 或 `raw > feature`。
- 代码依据：
  - `opencood/opencood/models/comm_modules/where2comm.py`：`Communication.forward()` 对单车 `psm_single` 做 sigmoid、anchor max、可选 Gaussian smoothing，再用 `threshold=0.01` 或 `k_ratio` 生成 BEV cell mask。
  - `opencood/opencood/models/point_pillar_comm_multiscale.py`：Where2comm 在三层 multiscale feature 上逐层执行，当前 checkpoint 三层 payload 分别为 `64x96x352`、`128x48x176`、`256x24x88`，未启用 channel compression。
  - `opencda/tools/lgcp_where2comm_area_mask_eval.py`：通信量估计为 `selected_cells x payload_channels x feature_value_bits`，默认 `feature_value_bits=16`。
  - `opencda/tools/lgcp_pointpillar_rsu_bev_fusion.py`：raw point slice 按 `points x 4 float32 = points x 16 bytes` 统计。
- 理论大小：
  - Where2comm 三尺度 dense full feature：每个 non-ego agent 约 `7392 KB`；5 个 leader packets 约 `36960 KB/frame`。
  - SGCP attentive scatter dense full BEV：5 leaders 约 `85.94 MiB/frame`。
  - Where2comm 当前所谓压缩不是低维编码，而是 sparse cell selection；每个保留 cell 仍上传原始多尺度 feature values。

| Case | Raw point slice KB/frame | Raw points/frame | Where2comm selected feature KB/frame | Dense Where2comm 5-agent feature KB/frame | Feature/raw | Selected/full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fine Top-10 K=5 full | 36.37 | 2328 | 184.12 | 36960 | 5.06x | 0.0050 |
| coarse 9-area K=5 dilation1 full | 94.60 | 6055 | 382.02 | 36960 | 4.04x | 0.0103 |
| coarse 9-area K=5 dilation0 planned | 94.60 | 6055 | 359.56 | 36960 | 3.80x | 0.0097 |

- Break-even rule：
  - raw bytes = `N_points x 16`
  - selected feature bytes = `sum_s(selected_cells_s x channels_s x 16 bits / 8)`
  - raw point cloud is larger only when `N_points > selected_feature_bytes / 16`
  - 当前 break-even：
    - fine K=5：需要约 `11784` raw points/frame，实际只有 `2328`
    - coarse K=5 dilation1：需要约 `24449` raw points/frame，实际只有 `6055`
    - coarse K=5 dilation0：需要约 `23012` raw points/frame，实际只有 `6055`
- 结论：
  - 当前 intersection10 场景中，raw point slices 比 Where2comm selected features 小，不满足“中间特征比原始点云更省”的叙事。
  - Where2comm 相比 dense full feature 确实省得很多，只保留约 `0.5%-1.0%` dense feature payload；问题是 dense feature 本身每 cell 通道数高，所以稀疏后仍可能比稀疏点云大。
  - 对网络论文而言，后续不必把重点转到训练，而应把通信核算写成条件式：当 area 内点云足够密、raw point slice 点数超过 break-even 时，Where2comm sparse feature 才比 raw point slice 更省；否则 raw slice 反而更省。
