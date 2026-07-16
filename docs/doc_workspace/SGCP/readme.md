# SGCP 文档工作区

本目录用于维护 SGCP（Self-Organized Game-Theoretic Collaborative Perception）相关实验、论文修改和实现状态记录。文档服务于 CARLA + OpenCDA + NS3 协同感知仿真工作，优先记录可复现的信息，而不是临时聊天结论。完整论文内容和审稿意见在C:\Workspace\icdcs-paper\SGCP。

## 文件说明

- `status.md`：当前整体状态，包括仓库理解、论文问题、实现/实验进展和阻塞项。
- `target.md`：任务清单，按写作修订、补充实验、机制完善和工程落地拆分。
- `log.md`：实验日志，按日期追加运行命令、配置、现象、结果和下一步。
- `results.md`：核心实验结果记录，用于沉淀可进入论文或 rebuttal 的表格、图和结论。
- `baseline_fairness.md`：baseline 公平性说明，区分 upper reference、same-pipeline ablation 和同通信约束主对比。
- `topology_trigger.md`：拓扑变化触发 cluster 重构的机制规格，说明触发条件、滞回策略和在线/离线接入位置。
- `cluster_capacity_policy.md`：cluster 已满、`N_max`、merge/split 和成员边际贡献重算的机制说明。
- `f_rho_calibration.md`：`f(rho)` 点云密度效用函数的标定协议、当前统计和论文写作口径。
- `control_overhead.md`：SGCP beacon、density metadata、cluster control 和 PPS schedule 控制开销估算口径。
- `potential_game_conditions.md`：PPS potential game / constrained best-response 的成立条件、代码偏差和论文写作边界。
- `paper_revision_plan.md`：面向论文正文和 rebuttal 的修订计划，记录具体表述替换、审稿意见回应和风险边界。
- `related_work_novelty_revision.md`：related work 与 novelty 的重写建议，重点回应 decentralized CP baseline 和 coalition-game 相似性质疑。
- `parameter_calibration_revision.md`：`f(rho)`、`rho_th`、`N_max`、`T_min^stab` 的参数标定与论文写作依据。
- `fullperception_baseline_revision.md`：FullPerception-RSU / FullPerception-Decentralized 的实现口径、公平性边界和 rebuttal 写法。
- `runtime_feasibility_revision.md`：SGCP 控制面毫秒级耗时分解、100 ms 周期可行性边界和 rebuttal 写法。
- `reproducibility_manifest.md`：当前可复现实验的代码版本、数据集、命令、结果和日志路径；同时标注论文旧主表缺少原始日志的问题。
- `online_topology_gate_regression.md`：真实 CARLA 在线打开 topology-trigger gate 的短回归命令、日志、trigger 统计和结论边界。
- `protocol_audit.md`：主表修复阶段的离线协议审计，记录 cluster、grid selection、channel allocation 是否真实进入融合输入。
- `mechanism_probe.md`：head-only、SGCP grid-constrained 和 full-cluster upload 的机制 probe，用于定位 AP 损失来源。
- `main_table_candidate.md`：论文主表候选，收束 FullPerception upper reference、payload-matched selective baseline、SGCP 10/20ch、Mbps 换算和主表/附表边界。
- `../environment.md`：全局环境文档，统一维护 Conda 环境、数据路径、CARLA 路径、启动命令和通用工具入口；运行实验前优先查看这里。

## 维护约定

1. 每次实验前先在 `log.md` 增加一条记录，写清命令、配置文件、代码版本和预期目的。
2. 实验结束后把原始日志路径、关键指标和异常现象补回 `log.md`。
3. 只有经过确认、可复现或计划进入论文的结果才整理到 `results.md`。
4. 新发现的问题先写入 `status.md`，再拆成 `target.md` 中可执行的任务。
5. 不在本目录保存大体积原始数据；只保存路径、摘要、指标和复现方式。

## 范围边界

本文件只说明 SGCP 文档工作区如何使用；环境、路径、启动命令和通用工具入口统一维护在 `../environment.md`。SGCP 相关实验结论写入 `results.md`，探索过程写入 `log.md`。
