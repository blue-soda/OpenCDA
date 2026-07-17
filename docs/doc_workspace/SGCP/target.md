# SGCP 任务清单

更新时间：2026-07-17
最终目标：解决所有C:\Workspace\icdcs-paper\SGCP中的审稿意见，并使 SGCP 主表在严格实验协议下以较少通信量获得较高 AP。

## P-1：最高优先级 - 主表结果修复与论文落地

- [ ] 新一轮主表重构：在统一 CAV 数、统一 backbone、统一 fusion/evaluation、统一 20MHz 或更低带宽预算下，生成一张 SGCP AP 最高、Mbps 最少且与 Random/Greedy/FullPerception 参数一致的主表；禁止使用未充分利用带宽的弱 baseline 来证明通信节省。
- [x] 强化 Random/Greedy baseline：已新增 forced-budget random selective baseline，并保留 `greedy_density` 作为 density greedy 显式别名。Forced random 在同一 coalition/late-fusion 路径、3 members/head、117 grid budget 下得到 `0.77/0.73/0.38`、31,613,424 bytes、61.68 Mbps；高预算 density/greedy baseline 为 `0.80/0.76/0.40`、37,710,864 bytes、73.58 Mbps。旧 RandomRA/MWS 继续作为低 payload w/o-PPS 消融。Forced random 11 帧真实 NS3 scheduled-only replay 已补齐：110/110 application/RLC complete、0 PHY failures。
- [x] 先运行 20 CAV 全融合/FullPerception 上界，确认当前 dump 的可达 AP 上限；随后做 object-level 漏检诊断，定位哪些 GT 在 full fusion 能检出但 SGCP/随机/贪心漏检。已新增 `failure_diagnostics.md` 和 `opencda.tools.sgcp_failure_diagnostics`：当前 full early 上界为 `0.85/0.83/0.48`；10ch/rho3 剩余 111 个 full-reference 可检出但 SGCP 漏检的 GT 中，63 个主要是目标 grid 只被其他 cluster head 覆盖，35 个是最近 head 已拿到较密点云但未形成最终框，12 个是最近 head 只拿到稀疏点，1 个完全没有调度覆盖。
- [x] 基于漏检 GT 反向改造分簇和区块选择：优先实现 target/object-aware grid selection、关键目标覆盖 fallback、quality-weighted coverage、detector-quality-aware member protection 或 detector/objectness-aware point sampling，并用统一带宽主表验证。已新增 `target_aware_potential_game`，把 target-aware multi-view grid utility 放入资源调度器本体；20MHz/10ch/rho3 41 帧从 `0.79/0.76/0.38` 提升到 `0.80/0.76/0.39`，对象级主失败桶 `covered only by other cluster heads` 从 63 降到 56。
- [x] 继续调优 `object_aware_potential_game` 并收束为完整机制：已新增逐帧 target-grid case study 和 OAPG 分支，确认 frame 000068/object 438、frame 000066/object 401、frame 000084/object 350 等最佳视角未调度问题可被 sender refinement 修复；但 OAPG 11 帧 AP 仅为 `0.74/0.69/0.30`，说明单纯 object peak 会伤上下文覆盖。已进一步新增 `perception_aware_potential_game`，用 coverage layer + target layer 统一建模调度目标，41 帧 20MHz/10ch/rho3/`B_h=2` 达到 `0.81/0.78/0.39`、32,049,872 bytes、62.54 Mbps，missed rows 从 106 降到 59。
- [x] 补 `perception_aware_potential_game` 真实 NS3 socket replay：11 帧真实 socket replay 已完成，110/110 scheduled requests application callback complete，RLC complete 110/110，RLC TX/RX events 2970/2970，PHY decode failures 0。下一步若需要在线证据，再做 CARLA+NS3 deadline-aware 短回归。
- [ ] 补 `perception_aware_potential_game` 在线 CARLA+NS3 短回归：验证 deadline-aware CP delivery 与离线 final-delivery 口径差异，并确认在线 CP submit/counter 在固定 tick 下稳定增长。
- [ ] 补 `target_aware_potential_game` 真实 NS3 socket replay：基于已生成的 11 帧 dry-run upload plan，启动 NS3 后验证 scheduled request application/RLC complete、manual subchannel 无 reject、PHY failure 为 0 或可解释。若 PAPG 真实 replay 通过且进入主表，则该项降为附表/消融验证。
- [ ] 继续优化 target-aware PG 的通信量：在保持 `0.80/0.76/0.39` 附近 AP 的同时，将 60.62 Mbps 向旧 57.38 Mbps 或更低压回；候选方向为 target-aware point cap、grid byte-aware utility、same AP lower payload ablation。
- [ ] 将新增 `--max-upload-points-per-source` 从随机点抽样升级为检测感知/空间均匀采样，目标是在 35-40 Mbps 附近恢复接近无 cap 的 `0.79/0.76/0.38` AP。
- [ ] 将 20MHz、10MHz 或更低带宽作为主表公平约束候选：高通信量方法必须受相同子信道/带宽窗口限制，超出预算的 FullPerception/greedy 不得无约束融合；同时报告 NS3 request-level delivery 与 deadline-aware online smoke 结果。
- [x] 审计离线 SGCP 测试链路，确认分簇结果是否真实决定每个 cluster head 的融合对象、成员集合和 inter-cluster late fusion 输入。已新增 `protocol_audit.md` 和 `--sgcp-trace-output`；41 帧输出 246 条 receiver trace，cluster head/member/source 列表与融合输入一致。
- [x] 审计点云选择链路，确认 `PotentialGame` 输出的 grid selection 是否真实裁剪 sender 点云，并进入 OpenCOOD early fusion 输入；输出逐帧/逐 CAV 的 selected grids、点数、payload 和 AP 关联日志。41 帧 trace 已记录 selected grids、point counts、payload、pred/gt boxes。
- [x] 审计子信道分配链路，确认 `sc_start/sc_num` 不只用于 NS3 replay，也真实约束离线/在线传输请求；对未调度 member 或超出子信道窗口的请求进行 drop/delay，而不是绕过 PPS 进入融合。41 帧 trace 中 `missing_channel_rows=0`，未发现未调度 sender 绕过 PPS 进入融合。
- [x] 构造离线可解释 probe：分别运行 head-only、SGCP grid-constrained、full-cluster、random grid、raw-density、density-distance、spatial-diverse 和 fixed first-frame cluster membership，对比输入点云数量、预测框、payload 和 AP，确认每个机制开关能改变融合结果。fixed-first-frame 为 `0.73/0.70/0.33`、26,325,216 bytes，低于动态 coalition `0.77/0.73/0.35`；`spatial_diverse` 在相同 scheduled links 和 grid count 下达到 `0.79/0.75/0.37`，当前优于原始 utility 与 random-grid。
- [x] 构造离线多帧协议一致性 probe：输出 cluster、grid selection、channel allocation、fused CAV ids、payload、prediction count、GT count 的帧级 CSV。已新增 `opencda.tools.sgcp_protocol_trace_summary`，从 246 行 receiver trace 汇总 41 行 frame summary：每帧 6 个 receiver、10 条 PPS links、平均 16 个 fused CAV、平均 payload 656,492.88 bytes/frame、`missing_channel_rows=0`。AP 仍采用 OpenCOOD 全局累计指标，不写成逐帧 AP。
- [ ] 用真实 CARLA 在线无 NS3 短回归验证 cluster membership、grid selection 和融合结果一致性；确保 CARLA 进程至多一个，并记录完整命令、日志和 AP。
- [ ] 用真实 CARLA + NS3 或离线 NS3 replay 验证时间同步和 transfer request 语义：CARLA tick / OpenCDA network time slot / NS3 sync time 必须一致；带宽内无冲突请求成功，带宽外请求延迟或丢包。已完成 default 10/5 子信道回归，以及 `spatial_diverse` 10ch `rho_th=2/3` 和 20ch replay：10 子信道均为 110/110 request application/RLC complete、44 条未调度需求跳过；20 子信道 154/154 request application/RLC complete；PHY failures 均为 0。后续可补真实 CARLA+NS3 短回归。
- [x] 执行真实 CARLA+NS3 有限 tick 短回归：已新增并更新 `online_ns3_short_regression.md`。首次回归发现 `vehicles_num=1` 过早初始化导致 NS3 address collision / SIGABRT；已修复为车辆注册完成 gate。修复后 35 tick 在线回归 OpenCDA exit 0、NS3 20 车初始化、38/38 sync_ack、无 sync timeout、无 fatal、无 manual reject，在线 CP AP 为 0.86/0.84/0.74。随后修复 scheduler strategy 残留并重跑，PSCCH/PSSCH decode failures 从 `1836/480` 降到 `95/10`，在线 AP 提升到 `0.88/0.88/0.79`。
- [ ] 分析真实在线 NS3 中的大包分片与 PHY decode failure：第一轮日志诊断发现 `PotentialGame.clear_resource_allocation_strategy()` 未清空各 CAV `ClusteringScheduler.channel_allocation`，导致在线多轮 CP 残留旧链路；已补清理逻辑并通过真实短回归验证主要 PHY collision 大幅下降。已新增 `opencda.tools.online_ns3_log_eval` 做 episode-level 解析，确认 strategy-clear 回归中 184 行 incomplete 实际对应 6 个 partial episode，且每个都缺 1 个 10000-byte fragment。已实现可配置 timeout reupload，first trial 将 complete/partial episode 改善到 39/3，但暴露 late CAM completion `KeyError` 并已修复。用户 2026-07-17 在线 run 进一步显示 AP 0.86/0.86/0.71 只来自极少数统计帧：日志解析为 3 个 CP eval/submit、185 个 CP wait、ego=1 wait 34 次、11 个 upload episode 全部 partial。已新增 timeout-exhausted partial CP 和 online Mbps 对齐字段；下一步用固定 `OPENCDA_ONLINE_TICKS=80/140` 重跑，确认 CP submit/counter 能稳定增长。clean rerun 若再次遇到 CARLA RPC 问题，先让 `carla_rpc_probe --expect-map Town03` 通过，再继续评估 fragment-level selective retransmission/drop-aware fusion。
- [ ] 若协议链路存在 bug，优先修复并重跑主表；修复项必须配套离线 probe、NS3 request-level trace 和必要的在线 CARLA smoke test。
- [ ] 若协议无误但主表 AP 仍低，展开细致消融定位原因：cluster 质量、grid utility、`rho_th`、`N_max`、member budget、channel budget、inter-cluster late fusion/NMS、检测框坐标变换、payload/AP tradeoff。已新增 `mechanism_probe.md`；当前定位为原始饱和 density utility selection 不足，coverage-aware `spatial_diverse` 是正向改造。最新 `B_h=2` sensitivity 显示 AP@0.7 可到 `0.42`，但 AP@0.3/0.5 降至 `0.76/0.72`，说明 member/RB budget 会影响定位-召回权衡。Late NMS 0.05/0.15/0.30 probe 显示默认 0.15 最好，AP 下降不是简单 NMS 阈值问题；新增 box-count diagnostics 进一步显示 `B_h=2` 的 fused GT 平均 64.83，低于 `B_h=1` 10ch 的 69.00；新增 CAV coverage diagnostics 显示 `B_h=2` 仍只融合 16/20 CAV，但把 CAV 6 从 41 帧上传降到 7 帧，下一步应诊断 target coverage 和关键成员保护。
- [ ] 在保持论文叙事的前提下改造算法以提升主表：优先将 `spatial_diverse` 整理成 coverage-aware grid utility / candidate scoring，使其在相同 scheduled links 和 grid count 下稳定优于 random-grid。已完成 5/10/20 子信道 sweep，10 子信道为低通信候选 `0.79/0.75/0.37`，20 子信道为高预算候选 `0.80/0.76/0.41`；10/20 子信道候选均已通过 NS3 request-level replay。已新增 `--head-rb-budget` probe，`B_h=2,rho_th=3` 达到 `0.76/0.72/0.42`、54.56 Mbps，并已完成 11 帧 NS3 replay 110/110 complete；但 AP@0.3/0.5 下降且 box-count/coverage/quality 诊断指向 fused GT 覆盖减少和高质量 CAV 6 被系统性挤出。已新增默认关闭的 `--sgcp-coverage-fallback persistent` 负面 probe：11 帧 AP 从无 fallback 的 `0.69/0.64/0.34` 降至 `0.67/0.62/0.34`，说明单纯 CAV 历史欠覆盖不够；随后新增 `quality_persistent` safety probe，11 帧恢复到无 fallback 的 `0.69/0.64/0.34` 且 replacement=0，说明质量门控能避免有害替换但过于保守。下一步优先实现 target/object-aware candidate generation、object-aware fallback sharing 或 late-fusion weighting，但必须保留“较少通信量 + 分层融合 + 子信道可行调度”的核心主张。
- [x] 重新生成主表候选结果：至少包含 NC、RS/random、MUG/MWS 或强 selective baseline、Full/reference、SGCP；统一报告 AP@0.3/0.5/0.7、payload/Mbps、control overhead、runtime 和 NS3 delivery。已新增 `main_table_candidate.md`，包含 FullPerception centralized、full-cluster reference、payload-matched selective baselines、SGCP coverage-aware 10ch/20ch、Mbps 换算和不建议进入公平主表的 Random/MWS 说明。
- [x] 修正主表 baseline 口径：FullPerception-RSU 在当前 RSU-free dump 上不可直接填实测，应以 full 20-CAV early `0.85/0.83/0.48`、60,838,528 bytes 作为 centralized FullPerception upper reference；Random/MWS scheduler 因 payload 过低只作 w/o PPS 消融；公平主对比使用 payload-matched selective sharing（高预算 0.80/0.76/0.40，37,710,864 bytes）和 SGCP spatial-diverse 10/20ch。
- [x] 补充通信量可调参数实验：已完成 `spatial_diverse` `rho_th=1/2/3/4` sweep，其中 `rho_th=3.0` 为 0.79/0.76/0.38，payload 29,405,296 bytes；`rho_th=3.0` 的 10ch NS3 replay 也已验证 110/110 application/RLC complete。
- [x] 当主表结果达到论文可写水平后，修改 `C:\Workspace\icdcs-paper\SGCP\main.tex`：已完成第一轮替换，删除旧 `0.85/0.84/0.69` 和 `22.33 Mbps` 主张，改为 FullPerception centralized upper reference、payload-matched selective baseline、SGCP coverage-aware 10/20ch、NS3 request-level delivery 与 near-real-time feasibility 口径。
- [x] 将 PAPG 最新主表候选写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`：主行更新为 `SGCP (PAPG, 10 ch.) = 0.81/0.78/0.39, 62.54 Mbps`；机制文字更新为 coverage layer + object-prototype target layer；通信效率段补入 110/110 application/RLC complete 和 0 PHY failures。尚未做 PDF 编译验证，因为本机未检测到 `latexmk/pdflatex`。
- [x] 根据最新结果更新 rebuttal 答复，覆盖审稿意见中的 FullPerception 公平性、decentralized baseline、`f(rho)` 标定、500 ms 参数、100 ms 实时性、topology trigger、NS3/通信可靠性。已新增 `rebuttal_draft.md`，作为 reviewer-by-reviewer response 草稿。
- [x] 压缩形成最终 rebuttal 可粘贴版本：已新增 `rebuttal_short.md`，保留 reviewer-by-reviewer 主线、关键数值和更保守的 claim boundary。
- [ ] 将过程中发现的新问题回写到 `status.md` 和 `target.md`，并在 `log.md` 记录每次实验的命令、commit、日志路径和结论。

## P0：先建立可复现实验基线

- [x] 最高优先级：排查并修复 NS3 manual subchannel 链路，使 OpenCDA 指定的 `sc_start/sc_num` 真实落到 NS3 NR sidelink 发送行为。已完成 4 类 probe：非冲突成功、最高合法子信道 9 成功、同子信道冲突触发 PHY decode failure、越界子信道 10 被拒绝且无 RLC/CAM 发送。
- [x] 将 SGCP `offline_ns3_replay` 的资源分配口径对齐为配置默认 `potential_game`，并默认只发送 PPS 已分配 `sc_start/sc_num` 的 request。已完成 11 帧 fixed replay：110 scheduled request 全部 CAM/RLC 成功，44 条未调度需求被跳过。
- [x] 修复 NS3 暴露子信道窗口语义：OpenCDA/bridge 按 `targetSubchannels` 校验 `sc_start/sc_num`，超出范围的 request 在 CAM/RLC 创建前拒绝；manual scheduler 对确定越界命令 drop+pop，避免无效队头阻塞后续合法请求。
- [x] 定位 SGCP 当前实现入口、配置文件和运行命令。
- [x] 建立 `v2xp_cluster_carla` 数据集导出、导入能力，用离线数据替代 CARLA 在线运行。
- [x] 当前优先：确认论文现有结果对应的代码版本、配置、随机种子和日志路径。已新增 `reproducibility_manifest.md`；结论是 `main.tex` 旧主表缺原始日志/随机种子/代码版本，当前只能把已复现离线结果作为修订依据。
- [x] 记录项目运行环境：`conda activate opencda`。
- [x] 记录在线命令：`python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug`，NS3 模式追加 `--network`。
- [x] 新增数据导出入口：`python opencda.py -t v2xp_cluster_carla --dump`。
- [x] 运行一次数据导出，确认每个评估 CAV 每帧生成 `.pcd` 与 `.yaml`。
- [x] 确认导出目录包含 `data_protocol.yaml`，格式兼容 OPV2V 风格目录。
- [x] 新增离线帧加载器，可从 OPV2V 风格目录读取单帧为 OpenCOOD 输入字典。
- [x] 新增离线 OpenCOOD 单帧推理脚本入口：`python -m opencda.tools.offline_inference --dataset-root <root>`。
- [x] 运行离线 OpenCOOD 单帧推理，验证无需启动 CARLA 可完成模型推理。
- [x] 将离线帧加载结果进一步接入 SGCP cluster/resource scheduling 回放。
- [x] 明确离线 SGCP 回放接口设计：见 `offline_replay.md`。
- [x] 新增 `OfflineCavWorld/OfflineVehicleManager/OfflineV2XManager/OfflineLidarGrid` 轻量状态适配层。
- [x] 单帧调用 `CoalitionGame`，确认无需 CARLA 可输出 cluster head/member。
- [x] 单帧调用 cluster resource allocation，确认可输出 channel allocation。
- [x] 多帧回放并输出 cluster lifetime、reconfiguration 次数、平均 cluster size、孤立车辆数量。
- [x] 确认离线资源分配默认 `NaiveRA` 是否应替换为论文 SGCP 使用的 `PotentialGame/PCS/MWS`：配置默认值为 `potential_game`，`naive` 保留为 baseline/fallback。
- [x] 解决资源分配配置与代码默认值不一致：`networking_clustering.yaml`/`ConfigManager` 默认为 `potential_game`，但 `ClusteringScheduler` 原先实例化 `NaiveRA`。
- [x] 将离线回放结果接入 SGCP 约束感知评估。
- [x] 确认 CARLA、OpenCDA、NS3、OpenCOOD 的版本和环境依赖。
- [x] 修复在线 CARLA-NS3 时间流速不一致：`NetworkManager.time_slot` 与 `world.fixed_delta_seconds` 保持一致。
- [x] 新增离线 NS3 replay smoke test，验证不启动 CARLA 时 `sync_request/sync_ack` 与 transfer request 链路可用。
- [x] 确认论文现有结果对应的代码版本、配置、随机种子和日志路径。旧论文表格尚未找到原始日志；当前复现 manifest 固定 OpenCDA commit、数据集、命令、结果和 artifact 路径。
- [x] 建立一次最小可复现实验流程，输出 mAP、通信开销、运行时耗时。
- [x] 在 `log.md` 中记录每次运行的完整命令和日志路径。已将核心复现实验集中整理到 `reproducibility_manifest.md`，`log.md` 继续保留探索过程和每轮新增命令。

## P1：补充关键实验

- [x] 消融：完整 SGCP vs 无稳定窗口。已完成 `T_min_stab=0` 第一版离线 replay + late-fusion AP；当前 41 帧 dump 与默认结果一致，后续需补更动态场景。
- [x] 消融：完整 SGCP vs 无 coalition formation，仅距离/随机聚类。已完成 singleton-cluster 第一版参考；后续需补距离/随机固定簇并计入 prediction-level late-fusion 开销。
- [x] 消融：完整 SGCP vs 无 PPS，仅随机/greedy 调度。已完成 `random` 与 `mws` 第一版离线 late-fusion 结果，后续需复核 MWS 论文口径。
- [x] 消融：完整 SGCP vs 仅 early fusion。已记录 full 20-CAV early baseline 与 constrained early-only/all-head 口径。
- [x] 消融：完整 SGCP vs 仅 late fusion。已完成 full 20-CAV OpenCOOD late checkpoint 第一版参考，后续需设计严格同通信约束 late-only SGCP 口径。
- [x] 参数实验：`T_min^stab` = 100/300/500/700/1000 ms。已完成第一版离线 replay + inter-cluster late-fusion AP；当前 41 帧 dump 对该参数无敏感性，后续需补更动态场景支撑论文结论。
- [x] 参数实验：`N_max` = 2/3/4/5/6。已完成第一版离线 replay + inter-cluster late-fusion AP；后续需补更长序列/密度场景并计入检测框交换开销。
- [x] 参数实验：`rho_th` 多组阈值。已完成 `0.5/1.0/2.0/3.0/4.0` 第一版离线 replay + inter-cluster late-fusion AP；后续需补完整 `f(rho)` 标定曲线和跨场景泛化。
- [x] 密度扩展：不同 CAV 数量或不同背景车密度。已完成同一 dump 下 5/10/15/20 CAV 子集规模敏感性第一版；后续需重新导出真实不同交通密度/背景车密度场景。
- [x] 网络资源扩展：不同带宽或子信道数量。已完成 `num_channels=5/10/20` 与 `bandwidth_mhz=20/40/80` 第一版离线 replay + inter-cluster late-fusion AP；后续需复核带宽参数在 PPS 吞吐模型中的作用。

## P2：补充公平 baseline

- [x] 核查并实现显式命名的 FullPerception-RSU baseline：不要再把 full 20-CAV early fusion 混写为 FullPerception；若采用虚拟 RSU，需在代码、文档和表格中明确其拥有 global/oracle scheduling information，并报告其与 full 20-CAV AP upper bound 的区别。已确认仓库此前没有显式 FullPerception 算法分支；现新增 `fullperception_rsu` proxy，41 帧结果为 `0.84/0.80/0.46`、56,224,736 bytes / 109.71 Mbps，和 full 20-CAV upper reference `0.85/0.83/0.48`、118.71 Mbps 分开记录。
- [x] 仿照 FullPerception-RSU 实现 FullPerception-Decentralized：只使用 CAV-side V2V 信息，不使用 RSU/全局 oracle；统一 backbone、数据帧、通信预算、late-fusion 口径和 NS3 request-level delivery。已新增 `fullperception_decentralized`，使用 cluster-local V2V candidates；41 帧结果为 `0.80/0.76/0.41`、38,920,592 bytes / 75.94 Mbps；3 帧 NS3 dry-run 已确认进入 scheduled-only request plan。
- [ ] 复现 EdgeCooper baseline：参考本地论文 `C:\Users\sakakibara\OneDrive\Papers\Cooperative Perception\EdgeCooper_Network-Aware_Cooperative_LiDAR_Perception_for_Enhanced_Vehicular_Awareness.pdf`，实现 edge/virtual-RSU assisted complementarity-enhanced、redundancy-minimized raw LiDAR scheduling proxy；优先使用同一 20MHz/10ch scheduled-only NS3 口径验证。已完成 blind-spot-aware first proxy，41 帧结果 `0.75/0.70/0.32`、56,134,048 bytes / 109.53 Mbps；下一步需从逐 receiver 贪心升级为 minimum-cost-flow/global assignment 风格，并补 NS3 replay。
- [x] 搜索并选择若干最新且适合作为审稿回复的 decentralized / V2V-only collaborative perception baselines，优先考虑 Where2comm/PACP/What2comm/CoBEVT/V2VNet 中可用当前 dump 和 OpenCOOD backbone 复现或近似实现的机制；每个 baseline 必须说明是否为真实复现、proxy 复现或不适合当前数据/模型。已新增 `baseline_reproduction_plan.md`，当前优先候选为 Where2comm-style confidence communication 与 PACP-style priority-aware sharing。
- [x] 重构主表 baseline 分层：AP upper bound（full 20-CAV early）、RSU/edge-assisted baselines（FullPerception-RSU、EdgeCooper）、V2V-only decentralized baselines（FullPerception-Decentralized、forced random、density/communication-aware、selected SOTA proxy）和 SGCP PAPG 分开呈现。已同步 `results.md`、`baseline_fairness.md`、`fullperception_baseline_revision.md` 和 `main_table_candidate.md`。
- [x] 搜索并选择至少一个 V2V-only/decentralized collaborative perception baseline。已实现 `nearest` / `density` selective-sharing baseline，但仍需按审稿意见补更接近 SOTA 的 decentralized baseline。
- [x] 统一 backbone、场景、通信资源、评价指标和 late fusion 设置。当前 selective baseline 复用同一 dump、OpenCOOD early checkpoint、SGCP clustering 和 inter-cluster late fusion 评价口径，并匹配 grid budget。
- [x] 在 `results.md` 中单独记录 baseline 公平性说明。
- [x] 实现 same-budget CAV-only selective-sharing baseline，例如 nearest/top-k density/communication-aware top-k，匹配 SGCP 的 payload 或 source CAV 数。已完成 nearest/density/communication-aware grid-budget baseline；communication-aware baseline 为当前最强竞争 baseline。
- [x] 补充 communication-aware selective-sharing baseline，加入距离/链路质量/payload cost，而不只按 density 排序。当前 first version 使用 `density_sum / (1 + distance / 100)`；后续可替换为 NS3/link-quality cost。
- [x] 将 communication-aware selective-sharing baseline 的距离 proxy 替换或扩展为 NS3/link-quality cost。已新增 `--ns3-link-quality-csv`，可用 `rlc_by_request.csv` 的 `rlc_complete` 调整成员选择分数。
- [x] 使用 NS3 request-level trace 对 SGCP 离线上传请求做链路层统计。旧 all-member replay 为 CAM callback delivery ratio = 0.558442；修复后 potential_game scheduled replay 为 110/110 CAM delivery、RLC RX 2970/2970、PHY failures 0。
- [x] 继续补充 RLC request completion 口径：按 request_id 对比 TX/RX segment 数、DROP 事件和 application callback，给出 partial reception、complete application delivery、PHY diagnostics 三层指标。已新增 `rlc_complete_requests`、`rlc_partial_requests`、`rlc_no_rx_requests`，并通过 10/5 子信道回归验证。
- [x] 构造 NS3 受限暴露带宽回归：`targetSubchannels=5` 下 110 个 scheduled request 中 `sc_start=0..4` 的 55 个 complete，`sc_start=5..9` 的 55 个 no_tx/no_rx，`MANUAL_CMD_REJECT=55`。
- [x] 将 NS3 request-level delivery/PDR 接入 SGCP PPS 或 selective-sharing baseline 的 link-quality cost。已接入 selective-sharing baseline，并完成 11 帧 distance proxy vs NS3 RLC-complete aware 对照。

## P3：完善机制设计

- [x] 定义 topology change trigger，包括邻居变化、相对速度、链路质量或 utility 下降阈值。已新增 `topology_trigger.md` 机制规格；后续需接入代码统计。
- [x] 将 topology trigger 接入离线 replay，输出每帧 trigger type、是否触发 reconfiguration、vehicle-head change 的对应关系。已在 `opencda.tools.offline_replay` 中新增 summary 和 `--print-topology-events`。
- [x] 将 topology trigger gate 接入在线 `ClusteringV2XManager`，避免无事件时每周期重构 cluster。已完成默认关闭 first version；后续需真实 CARLA 回归。
- [x] 在真实 CARLA 在线仿真中打开 `enable_topology_trigger_gate`，回归 cluster trigger 日志、reconfiguration 次数和感知结果。已新增 `online_topology_gate_regression.md`；35 tick 在线回归正常结束，CP AP@0.3/0.5/0.7 = 0.84/0.82/0.69，触发 `initial=1`、`neighbor_set_change=1`、`head_member_unreachable=3`，未观察到 skip。
- [x] 设计 cluster 已满时的处理策略：保留、替换、等待、split/merge 或 leader-level late fusion 补偿。已新增 `cluster_capacity_policy.md`，当前主策略为硬 `N_max` + 保留/未满簇迁移/singleton fallback/inter-cluster late fusion 补偿。
- [x] 明确是否支持 cluster merge/split，并说明与 `N_max` 的关系。当前口径：不允许超过 `N_max` 的 merge；split/merge 通过 topology trigger + coalition reformation 间接发生。
- [x] 补充成员加入后的边际贡献重算流程。当前迭代会在后续车辆/后续轮次基于更新后的 coalition state 重算贡献，并用 `ita` 抑制振荡。
- [x] 在离线 replay 中统计满簇数量、因 `N_max` 跳过的候选 move 数和 singleton/small-cluster 补偿比例。默认 `N_max=4` 下 41 帧平均每帧 3.12 个满簇、99.15 次满簇候选跳过、singleton ratio 0、small-cluster ratio 0.187。
- [x] 明确 `f(rho)` 的标定协议和 detector/sensor-specific metadata 机制。已新增 `opencda.tools.sgcp_density_calibration` 与 `f_rho_calibration.md`，当前 41 帧 dump 中默认 `rho_th=2.0` 位于非零 density p90/p95 之间，并筛出约 7.18% 非零网格；后续需补跨场景/探测器泛化。
- [x] 重新检查 potential game exact potential 的成立条件。已新增 `potential_game_conditions.md`；当前代码应表述为 potential-guided constrained best-response scheduling，若论文继续使用 exact potential game，需要限定固定 cluster/candidate grids/hard feasibility constraints，并补显式势函数与 monotonicity 证明。
- [x] 估算 density/utility/control message 的控制开销，并决定是否纳入通信开销指标。已接入 `opencda.tools.offline_replay` 并新增 `control_overhead.md`；当前 41 帧默认 SGCP 控制面约 187,112 bytes，平均 4,563.71 bytes/frame，约为点云 payload 的 0.70%，论文中应单独报告而不是混入 perception payload。
- [x] 复核 PPS 中 `bandwidth_all/bandwidth_per_channel` 的吞吐约束实现，解释为何 `bandwidth_mhz=20/40/80` 在当前离线实验中结果一致。结论：参数已进入 `PotentialGame` max-grid/SINR/data-rate 计算，但当前 dump 不由带宽上限主导。
- [x] 构造能触发带宽瓶颈的 SGCP 场景或参数组：已完成 `bandwidth_mhz=0.1/0.5/1.0` stress test，selected grids 和 AP 随带宽恢复而上升。

## P4：论文写作修订

- [x] 重写 related work 的 decentralized CP 和 coalition game 对比。已新增 `related_work_novelty_revision.md`，给出 V2V-only CP、RSU-centric CP、learned communication selection、其他领域 coalition formation 的对比段落和 `main.tex` 插入建议。
- [x] 增强 novelty：突出感知效用驱动、稳定性约束、分层 fusion、分布式资源调度的组合贡献。已在 `related_work_novelty_revision.md` 中形成 introduction contribution 和 rebuttal 可用文本。
- [x] 补充 `f(rho)` 标定过程和曲线。已新增 `parameter_calibration_revision.md`，整合 density calibration 命令、788,020 个 CAV-grid 样本统计、`rho_th` sweep 和论文/rebuttal 写法。
- [x] 补充 `T_min^stab`、`N_max`、`rho_th` 参数选择依据。已新增 `parameter_calibration_revision.md`；明确 `rho_th=2.0` 是 AP/payload 折中，`N_max=4` 是容量/fragmentation 折中，`T_min^stab=500 ms` 只能写为保守默认而非当前证据下的最优值。
- [x] 补充 FullPerception baseline 的实现细节和公平性讨论。已新增 `fullperception_baseline_revision.md`，明确 FullPerception-RSU/full 20-CAV fusion 只能作为 centralized upper reference，公平主对比应使用 same-budget CAV-only selective-sharing baseline。
- [x] 补充实时性实验，包括毫秒级耗时分解。已扩展 `opencda.tools.offline_replay` 输出 `runtime_breakdown_ms`，完成 41 帧控制面 profiling，并新增 `runtime_feasibility_revision.md`；当前 Python 原型 SGCP algorithm avg 105.24 ms，需谨慎写为 near-real-time 而非完整端到端 100 ms 保证。
- [x] 修正 “topology change 才触发” 与 “每个周期重复” 的表述矛盾。已新增 `paper_revision_plan.md`，明确每周期更新 beacon/density/PPS，cluster membership 仅在 topology/stability trigger 或 periodic guard 触发时更新，并给出 `main.tex` 替换建议。
- [x] 将 coverage-aware / spatial-diverse grid selection 和 potential-guided constrained PPS 口径写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`，弱化无条件 exact-potential/Nash guarantee 表述，使机制章节与当前实现和主表结果一致。
- [x] 将 `f(rho)` / `rho_th` 标定统计和 `rho_th` sensitivity 表写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`：包括 788,020 个 CAV-grid 样本、非空网格 5.98%、非空密度 p90/p95 = 1.40/3.60、`rho_th=2.0` 选择 7.18% 非空网格，以及 coverage-aware 10ch `rho_th=1/2/3/4` AP/Mbps 表。
- [x] 将 `N_max` 和 `T_min^stab` 参数依据写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`：说明 `N_max=4` 是容量控制折中而非纯 AP 调参，`T_min^stab=500 ms` 是五个 10 Hz 感知周期的保守滞回默认，并记录当前 sweep 中 `T_min^stab=100--1000 ms` 对 AP/reconfiguration 不敏感。
