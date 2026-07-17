# LGCP 任务清单

## 总目标

围绕 LGCP 论文审稿意见"C:\Workspace\icdcs-paper\LGCP\LGCP-review.txt"，补强三条主线：

1. 写作与叙事：清楚说明 LGCP 相对传统聚类、任务分配和层次聚合的区别。
2. 补充实验：证明 area confidence、group selection、local-to-global structure 和 transmission scheduling 的有效性。
3. 完善机制：让动态场景、控制面开销、定位误差、leader 失效和多 RSU 扩展等假设更可信。

## P0：必须优先完成

- [x] 梳理 LGCP 当前论文问题，形成 rebuttal / revision 对照表。
- [x] 设计 area confidence 验证实验：统计 confidence 与 area-level AP / recall 的相关性。
- [x] 设计 Eq. (2) 组合规则验证实验：比较 product rule、max、mean、sum、top-k 等组合方式。
- [x] 增加 greedy group selection 的 small-scale optimality gap 实验：用 exhaustive search 或 ILP 做小规模最优参考。
- [ ] 增加 local-to-global ablation：区分 partial sharing 与 LGCP 层次结构本身带来的收益。（设计文档、offline perception-only smoke、hierarchy budget sweep、raw feature-slice budget sweep、single-slot scheduled NS3 smoke、multi-slot scheduling proxy、3 帧 live multi-slot replay、model-level hierarchy 入口审计、box-level hierarchy late-fusion 1 帧/2 area smoke、Top-30 1 帧完整 area run 和 Top-30 3 帧 run 已完成；下一步扩大到 Top-30 11 帧并对齐 flat baselines，完整 neural feature hierarchy 待实现）
- [x] 补充更强通信感知 baseline：至少包含 adaptive sharing 或 selective sharing without LGCP hierarchy。
- [x] 明确大规模 30 CAV 实验只验证 latency，或补充 scalable perception-quality proxy。

## P1：重要补强

- [x] 增加定位误差敏感性实验。（已完成单场景 11 帧 smoke，论文级结果仍需多 seed）
- [x] 增加车辆速度 / update frequency / stale assignment 敏感性实验。（已完成 update frequency / stale assignment 单场景 11 帧 smoke，显式车速仍需多场景）
- [x] 增加 area size / grid size 敏感性实验。（已完成单场景 11 帧 smoke，论文级结果仍需多 seed）
- [x] 增加 subchannel count `Z` 敏感性实验。（已完成 11 帧 scheduling-capacity proxy，NS3 多 Z 仍需复核）
- [x] 增加 CAV / edge computation capacity 敏感性实验。（已完成 11 帧 compute-latency proxy，模型级 runtime 仍需后续实现）
- [x] 显式统计 control-plane overhead：location、direction、confidence、assignment、global view。
- [x] 解释或修复低车数场景下 LGCP 与 baseline latency 接近的问题。（已形成 `latency_figure_audit.md`，解释 fixed coordination overhead / sparse contention / edge compute 小规模优势）
- [x] 检查并修正 Fig. 7 y-axis 标注问题。（已渲染核查：y-axis `End-to-end latency (ms)` 正确；实际应将 x-axis 从 `Number of vehicles` 改为 `Number of CAVs`，需用原始绘图源重导出）

## P2：机制与写作完善

- [x] 增加 LGCP workflow figure，覆盖 partition、group assignment、leader selection、local fusion、RSU aggregation、broadcast。
- [x] 重新阐述 group 概念，避免被理解为传统 clustering。
- [x] 明确一个 CAV 参与多个 area group 时的数据包粒度、去重和复用机制。
- [x] 完善 leader 到 RSU 上传策略，说明重要结果的优先级、可靠性和失败处理。
- [x] 增加 deployment assumptions / limitations 小节：RSU centralization、mobility、multi-RSU、stale information、failure modes。
- [x] 修正文中 stage 编号不一致问题。
- [x] 将 “heuristic / approximate” 的表述调整为与实际理论保证一致的写法。

## 待落地到仓库的实验方向

- [x] 确认 OpenCOOD 中 OPV2V / V2XSet 的模型评估入口和日志格式。（已形成 `opencood_eval_entry.md`，记录 `inference.py`、`eval.yaml`、`--save_npy` 和 LGCP area-level 复用边界）
- [x] 确认 OpenCDA 中多 CAV + NS3 的基础可复现实验命令。
- [x] 新增 LGCP 专用 Town03 仿真配置，避免修改 `v2xp_cluster_carla`。
- [x] 补齐 RSU 首次启用的基础初始化、注册、读取和销毁路径。
- [x] 完成 `lgcp_carla` 数据导出 smoke test。
- [x] 完成导出数据的离线 OpenCOOD 推理 smoke test。
- [x] 确认 LGCP 所需 area-level confidence / AP / recall 的数据导出位置。
- [x] 建立统一结果目录和命名规则。
- [x] 实现 area confidence 离线导出 smoke test，验证 ROI/grid/agent-area records 链路。
- [x] 扩展 area confidence 离线评估脚本，接入 prediction slicing 与 area-level recall smoke test。
- [x] 扩展 area confidence 离线评估脚本，累计 area-level AP 并统计 confidence-vs-AP/recall 相关性 smoke test。
- [x] 扩大 area confidence validation 到完整 `lgcp_carla` dump 帧，并补充 detector-score confidence 对照。
- [ ] 扩大 area confidence validation 到多 seed / 多场景，形成可进入论文的稳定相关性结果。（已完成 OpenCOOD 远端资源 / 协议 / checkpoint 审计；本地无 `dataset/`，下一步在 `mindspore-186` 跑 400-frame gate 多 seed）
- [x] 实现 greedy group-member selection exhaustive-search gap smoke test。
- [x] 扩展 greedy optimality gap 到 leader assignment / load balancing。
- [ ] 扩大 greedy optimality gap 到多 seed / 更大 instance，并接入 latency-aware O3 objective。（O3 objective 已接入并完成 5-agent / 6-agent 11 帧 smoke；已完成 5-agent sampled seeds 7/11/23/37，仍需多场景）
- [x] 完成 local-to-global ablation 实验设计。
- [x] 实现 offline selective-sharing vs LGCP area-aware subset ablation。
- [x] 扩大 offline subset ablation 到完整 11 帧。
- [x] 为 offline subset ablation 加入 packet-budget / byte-volume proxy 统计。
- [x] 实现 communication-aware top-k selective-sharing baseline，并完成 11 帧 offline 对照。
- [x] 实现 scalable perception-quality proxy，并用 11 帧 offline AP 对照做初步校准。
- [x] 实现 LGCP 离线 RSU area assignment / leader upload plan 导出。
- [x] 将 LGCP upload plan 接入 offline NS3 replay dry-run。
- [x] 运行 LGCP upload plan offline NS3 联机 smoke test。
- [x] 扩大 offline subset ablation 到多 seed。（已完成 random-only seeds 7/11/23/37 汇总；deterministic strong baselines 复用 11 帧结果）
- [x] 扩大 LGCP upload plan offline NS3 replay 到 11 帧，并解析 request-level bridge-observed delivery ratio / delay summary。
- [x] 接入 ns-3 PHY decode trace，补充 decode-failure breakdown。
- [x] 接入 ns-3 application request-id trace，将 `cam_received` 精确映射回 LGCP upload request。
- [x] 接入 ns-3 RLC trace，将 RLC events 进一步映射回 LGCP upload request。
- [x] 完成 NS3 PHY / HARQ request-level trace 设计，明确字段、落点、解析器输出和论文边界。
- [x] 扩展 OpenCDA NS3 log parser，支持未来 request-level PHY / HARQ events 并输出 request lifecycle funnel。
- [x] 接入 ns-3 PSSCH request-level trace，将 PSSCH decode OK/FAIL 映射回 LGCP upload request。
- [x] 将 HARQ feedback 进一步绑定到 LGCP upload request，并确认 replay 配置下 HARQ event 可观测。
- [x] 扩大 request-level PSSCH / HARQ trace 到 11 帧 LGCP replay。
- [x] 接入 LGCP multi-slot replay lifecycle diagnostics，定位 scheduled replay 中 member-to-leader callback 偏低的阶段原因。
- [x] 增加 LGCP source-unique / half-duplex scheduling sensitivity，验证同 source 同 slot 多发不是 member-to-leader 瓶颈的充分解释。
- [ ] 实现 LGCP 专用 RSU area assignment、leader local fusion 和 RSU global aggregation 管线。（已完成 offline assignment / upload plan、hierarchy area-budget sweep、raw LiDAR feature-slice budget sweep、raw-slice-aware upload plan dry-run / 3 帧与 11 帧 request-level NS3 trace、single-slot scheduled NS3 smoke、multi-slot scheduling proxy、3 帧 live multi-slot replay、leader/RSU aggregation proxy、model-level hierarchy 入口审计、box-level late-fusion adapter smoke、Top-30 1 帧完整 area run 和 Top-30 3 帧 run；Top-30 11 帧 late-fusion 与 neural feature slicing + model-level fusion 仍待实现）
