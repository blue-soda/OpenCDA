# LGCP 文档工作区

本目录用于记录 LGCP（Local-to-Global Collaborative Perception）在 OpenCDA / OpenCOOD / CARLA / NS3 联合仿真环境中的论文修订、机制设计和实验推进过程。

## 文件说明

- `target.md`：任务清单。记录论文修订目标、实验补充项、机制完善项和优先级。
- `status.md`：当前状态。记录已经确认的背景、当前阻塞点、近期工作焦点。
- `log.md`：实验日志。按日期追加每次实验或代码验证过程，保留命令、配置、日志路径、观察结果和结论。
- `results.md`：核心实验结果记录。沉淀可进入论文的表格、图、关键数值和实验解释。
- `revision_matrix.md`：审稿意见到论文修改、补充实验和机制完善任务的映射。
- `area_confidence_validation.md`：area confidence 与 area-level AP / recall 相关性验证实验设计。
- `eq2_composition_validation.md`：Eq. (2) 多 CAV area confidence 组合规则验证实验设计。
- `greedy_optimality_gap.md`：greedy group selection 小规模 optimality gap 实验设计。
- `local_to_global_ablation.md`：区分 partial sharing 与 LGCP local-to-global hierarchy 贡献的消融实验设计。
- `communication_aware_baseline.md`：强 communication-aware selective-sharing baseline 的定义、当前结果和论文解释边界。
- `large_scale_quality_proxy.md`：大规模 latency-only 实验的论文口径，以及可校准 perception-quality proxy。
- `hierarchy_pipeline.md`：LGCP RSU area assignment、area-task group、leader upload 和后续 hierarchy 管线的当前实现状态。
- `ns3_phy_harq_request_trace.md`：NS3 PHY / HARQ request-level trace 的字段、落点、解析器和论文使用边界。
- `control_plane_overhead.md`：LGCP control-plane overhead 的统计口径、工具入口和当前 11 帧结果。
- `workflow_and_group_semantics.md`：LGCP workflow figure 草稿、area-task group 语义、packet 粒度、去重复用和 leader-to-RSU 可靠性说明。
- `deployment_assumptions.md`：LGCP 部署假设、限制、failure modes、multi-RSU 扩展和大规模 claim 边界。
- `manuscript_language_audit.md`：论文 stage 编号一致性、heuristic / approximate 表述边界和替换措辞清单。
- `grid_size_sensitivity.md`：LGCP ROI grid / area size sensitivity 的单场景 11 帧 smoke 结果。
- `localization_error_sensitivity.md`：LGCP CAV pose localization error sensitivity 的单场景 11 帧 smoke 结果。
- `stale_assignment_sensitivity.md`：LGCP update frequency / stale assignment sensitivity 的单场景 11 帧 smoke 结果。
- `subchannel_sensitivity.md`：LGCP subchannel count `Z` sensitivity 的 11 帧 scheduling-capacity proxy 结果。
- `experiments/readme.md`：LGCP 实验输出目录、run id、配置快照和结果归档规则。
- `../environment.md`：全局环境文档，统一维护 Conda 环境、数据路径、CARLA 路径、启动命令和通用工具入口；运行实验前优先查看这里。

## 使用约定

1. 新实验先在 `target.md` 中挂任务，再在 `log.md` 中记录执行过程。
2. 只有经过复核、可复现、可解释的结果才同步到 `results.md`。
3. 当任务状态、阻塞点或论文叙事发生变化时，及时更新 `status.md`。
4. Conda 环境、数据集路径、CARLA 路径、常用启动命令和通用离线工具入口以 `../environment.md` 为准。
5. OpenCDA / NS3 联合仿真调试细节和日志排查方式以仓库根目录的 `AGENT_README.md` 为准。
6. 论文原始材料目前参考 `C:\Workspace\icdcs-paper\LGCP` 与 `C:\Workspace\icdcs-paper\LGCP-review.txt`。

## 仓库相关背景

OpenCDA 是基于 CARLA / SUMO 的 Cooperative Driving Automation 仿真框架，包含感知、定位、规划、控制、V2X 通信和场景管理等模块。当前仓库还包含 OpenCOOD，用于多智能体协同感知模型训练和推理，支持 OPV2V、V2XSet 等数据集。

LGCP 相关实验预计会同时涉及：

- OpenCDA：多 CAV / RSU / UAV 场景、V2X 通信、CARLA 联合仿真。
- OpenCOOD：CoBEVT、Where2comm、CoAlign 等协同感知模型和 OPV2V / V2XSet 评估。
- NS3：V2X 网络传输、冲突、时延、丢包和调度验证。
