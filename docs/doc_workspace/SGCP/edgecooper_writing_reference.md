# EdgeCooper Writing Reference for SGCP

更新时间：2026-07-19

本文档记录从本地 EdgeCooper 论文 PDF 中提炼出的写作组织方式，用于完成 `target.md` P9。源文件：

```text
C:\Users\sakakibara\OneDrive\Papers\Cooperative Perception\EdgeCooper_Network-Aware_Cooperative_LiDAR_Perception_for_Enhanced_Vehicular_Awareness.pdf
```

注意：本文档只记录可迁移的论文结构和叙事边界，不新增 satisfaction rate 等主指标；SGCP 仍统一使用 aggregate AP@0.3/AP@0.5/AP@0.7 和 Mbps。

## EdgeCooper 实验章节组织

EdgeCooper 的实验章节按系统论文逻辑展开，而不是只堆单张主表：

1. Dataset and simulation architecture：说明 CARLA / SUMO / NS3 / PyTorch 联合仿真链路、数据来源、LiDAR 参数、点云裁剪区域、pillar/grid 设置和 detector。
2. Object detection model：交代 PointPillars 管线，使通信层结果和感知模型挂钩。
3. Communication settings：明确上传阶段时长、5G V2X / NR-V2X 参数、20 MHz、5.9 GHz、10 个子信道等网络条件。
4. Comparison algorithms：逐项说明 baseline 的信息条件和机制差异。
5. Qualitative evaluation：用复杂路口可视化解释为什么协同点云能改善遮挡/稀疏区域。
6. Quantitative evaluation：先报告通信/感知中间指标，再报告 AP/IoU、感知距离和通信-精度 tradeoff。

## 对 SGCP 的可迁移写法

SGCP 当前 `main.tex` 已经基本采用相同思路，但需要持续保持以下边界：

- 平台先行：实验开头必须明确 CARLA-OpenCDA-NS3，而不是只写 OpenCOOD 离线 AP。
- 信息条件分层：Full 20-CAV early fusion 是 full-sharing upper reference；EdgeCooper-HD 是 edge-assisted/global assignment reference；FullPerception-PCS 是内置 PCS baseline；SGCP-compatible scheduler comparison 只比较同一 scaffold 内的调度器。
- 指标克制：不引入 EdgeCooper 的 satisfaction rate。SGCP 使用 aggregate AP 和 Mbps；PPS convergence、NS3 reliability、runtime/control overhead 只作为辅助证据或附录。
- 定性案例要服务机制：case study 应展示 best-view sender、target grid、cluster head、selected sender/grid 和 final detection，而不是泛泛展示点云更密。
- 叙事顺序要从完整系统到内部模块：先 Table 1 protocol-native comparison，再 fusion scaffold ablation、Pareto、scheduler-compatible comparison、parameter sensitivity。

## 建议映射到 SGCP 图表

| EdgeCooper 写作功能 | SGCP 对应材料 | SGCP 写作边界 |
| --- | --- | --- |
| 联合仿真平台 | `environment.md`、`results.md`、`main.tex` Experimental Setup | 强调 CARLA-OpenCDA-NS3 和 20 CAV dense scenario |
| 通信参数 | `environment.md`、`runtime_control_ns3_appendix.md` | 20 MHz / 10 subchannels 是主设置；更宽带宽只做 sensitivity |
| Comparison algorithms | `baseline_fairness.md`、`baseline_reproduction_plan.md` | 按 protocol-native / edge-assisted / V2V-only / SGCP-compatible 分层 |
| Qualitative evaluation | `qualitative_case_study.md` | 三个失败案例解释 PAPG 动机和 high-IoU 边界 |
| Quantitative evaluation | protocol table、fusion figure、Pareto、scheduler table、parameter table | 主指标保持 aggregate AP + Mbps |
| Runtime evidence | `runtime_feasibility_revision.md`、appendix support | 写 near-real-time feasibility，不写 detector-inclusive 100 ms guarantee |

## 需要避免的误写

- 不把 EdgeCooper-HD 写成 fully decentralized V2V baseline。它在本文复现中依赖 virtual edge/global assignment，只能作为 edge-assisted reference。
- 不把 Pure late 写成 zero-communication baseline。它是 prediction-sharing reference，必须说明 detection-box overhead。
- 不把 scheduler comparison 写成完整系统排名。该表只证明 PAPG/PPS 在 SGCP-compatible scaffold 中的边际贡献。
- 不把 AP@0.3 的收益全部归因于 scheduler。当前证据表明 inter-cluster late fusion 对 coverage / AP@0.3 贡献很大。
- 不承诺完整端到端 100 ms。当前 runtime 证据是 control-plane prototype profiling。

## P9 结论

EdgeCooper 的写法值得借鉴的是系统化 evaluation structure：先把联合仿真、通信设置、baseline 信息条件说清楚，再用 qualitative 和 quantitative 证据闭环。SGCP 的修订稿已经采用这一结构，但必须比 EdgeCooper 更谨慎地区分 raw-LiDAR sharing、prediction sharing、edge-assisted scheduling 和 fully decentralized SGCP。
