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
