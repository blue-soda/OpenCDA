# SGCP 文档工作区

本目录用于维护 SGCP（Self-Organized Game-Theoretic Collaborative Perception）相关实验、论文修改和实现状态记录。文档服务于 CARLA + OpenCDA + NS3 协同感知仿真工作，优先记录可复现的信息，而不是临时聊天结论。完整论文内容和审稿意见在C:\Workspace\icdcs-paper\SGCP。

## 文件说明

- `status.md`：当前整体状态，包括仓库理解、论文问题、实现/实验进展和阻塞项。
- `target.md`：任务清单，按写作修订、补充实验、机制完善和工程落地拆分。
- `log.md`：实验日志，按日期追加运行命令、配置、现象、结果和下一步。
- `results.md`：核心实验结果记录，用于沉淀可进入论文或 rebuttal 的表格、图和结论。
- `baseline_fairness.md`：baseline 公平性说明，区分 upper reference、same-pipeline ablation 和同通信约束主对比。
- `../environment.md`：全局环境文档，统一维护 Conda 环境、数据路径、CARLA 路径、启动命令和通用工具入口；运行实验前优先查看这里。

## 维护约定

1. 每次实验前先在 `log.md` 增加一条记录，写清命令、配置文件、代码版本和预期目的。
2. 实验结束后把原始日志路径、关键指标和异常现象补回 `log.md`。
3. 只有经过确认、可复现或计划进入论文的结果才整理到 `results.md`。
4. 新发现的问题先写入 `status.md`，再拆成 `target.md` 中可执行的任务。
5. 不在本目录保存大体积原始数据；只保存路径、摘要、指标和复现方式。

## 范围边界

本文件只说明 SGCP 文档工作区如何使用；环境、路径、启动命令和通用工具入口统一维护在 `../environment.md`。SGCP 相关实验结论写入 `results.md`，探索过程写入 `log.md`。
