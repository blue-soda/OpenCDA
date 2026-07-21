# SGCP 实验日志

本文件按时间顺序追加实验记录。每条记录应尽量包含：目的、代码版本、配置、命令、日志路径、关键结果、异常现象和下一步。

## 2026-07-19 - Paper artifact index for P0

### 目的

推进 `target.md` P0：为当前论文草稿中的主文图表和附录材料建立统一 artifact 索引，确保每个进入论文的结果都有 source CSV/manifest、脚本或命令、log/trace、git commit 和 claim boundary。

### 输出

- `docs/doc_workspace/SGCP/paper_artifact_index.md`
- `docs/doc_workspace/SGCP/artifacts/paper_artifact_index_20260719/paper_artifact_index.csv`

### 覆盖范围

- Table 1 Protocol-Native System Comparison。
- Figure 1 AP-Mbps Pareto。
- Figure 2 Protocol Breakdown。
- Figure 3 Fusion Contribution。
- Table 3 SGCP-Compatible Scheduler Comparison。
- Table 4 Parameter Sensitivity。
- P8 runtime/control/NS3 appendix。
- P8 qualitative case study。
- P9 EdgeCooper writing reference。

### 结论

当前论文草稿第一版 artifact 追踪已补齐；`target.md` P0 的 artifact 要求可标记为完成。但该完成状态只覆盖当前结果版本。若 early-fusion checkpoint 回收、新导出 CARLA 场景、替换主表数值或升级 BEV qualitative figure，必须新增 artifact 版本并更新索引。

## 2026-07-19 - Protocol-native claim audit for P1

### 目的

推进 `target.md` P1 的验收项，确认当前 protocol-native 主表是否已经在论文正文中清楚解释 SGCP 系统优势、late-fusion coverage 贡献、FullPerception/EdgeCooper baseline 信息条件。

### 输入

```text
C:\Workspace\icdcs-paper\SGCP\main.tex
docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\protocol_native_manifest.csv
docs\doc_workspace\SGCP\paper_artifact_index.md
docs\doc_workspace\SGCP\main_table_candidate.md
```

### 输出

- `docs/doc_workspace/SGCP/protocol_native_claim_audit.md`
- 更新 `target.md`：P1 前三项验收标记完成。
- 更新 `status.md` / `readme.md`。

### 结论

当前 `main.tex` 已经分清 Full20Early upper reference、FullPerception-PCS built-in baseline、EdgeCooper-HD edge-assisted reference、Pure late prediction-sharing reference 和 SGCP RSU-free V2V main method。SGCP 的 AP@0.3 贡献被写成 system protocol / inter-cluster late fusion 的 coverage 收益，scheduler comparison 另设 Table 3。P1 剩余风险不是表格结构，而是 Pure late detector/checkpoint fairness 和 early-fusion checkpoint 强度。

## 2026-07-19 - EdgeCooper writing reference for P9

### 目的

完成 `target.md` P9 中“阅读并参考 EdgeCooper 写作方式”的未完成项，重点核查 network-level evaluation、V2V+/edge scheduling 叙事边界，并确认不引入 satisfaction rate。

### 输入

```text
C:\Users\sakakibara\OneDrive\Papers\Cooperative Perception\EdgeCooper_Network-Aware_Cooperative_LiDAR_Perception_for_Enhanced_Vehicular_Awareness.pdf
```

### 结果

- 新增 `docs/doc_workspace/SGCP/edgecooper_writing_reference.md`。
- 更新 `readme.md`、`target.md`、`status.md`。
- 提炼 EdgeCooper 实验章节结构：dataset/simulation architecture、object detection model、communication settings、comparison algorithms、qualitative evaluation、quantitative evaluation。
- 映射到 SGCP 当前图表：protocol-native comparison、fusion scaffold ablation、Pareto、SGCP-compatible scheduler comparison、parameter sensitivity、appendix support。

### 结论

SGCP 可以借鉴 EdgeCooper 的系统化实验组织，但不能沿用 satisfaction rate 作为主指标。SGCP 修订稿应继续使用 aggregate AP + Mbps，并严格区分 Full 20-CAV upper reference、EdgeCooper-HD edge-assisted reference、FullPerception-PCS baseline、Pure late prediction-sharing reference 和 SGCP-compatible scheduler comparison。

## 2026-07-19 - Qualitative BEV overlay draft

### 目的

在 P8 qualitative case study 文字草稿基础上，生成可复现 BEV overlay draft，使三个失败案例不仅有表格说明，也有图形素材。

### 输入

```text
docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f\gt_objects.csv
docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f\vehicles.csv
docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f\schedules.csv
```

### 命令

```powershell
conda run -n opencda python docs\doc_workspace\SGCP\artifacts\appendix_support_20260719\plot_qualitative_case_study.py
```

### 输出

- `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/plot_qualitative_case_study.py`
- `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/qualitative_case_study_bev.png`
- `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/qualitative_case_study_bev.pdf`
- `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/qualitative_case_study_summary.csv`

### 目视检查

PNG 已检查：三个 panel 均能显示目标 grid、GT、cluster head、selected sender 和 best-view sender；当前适合作为 appendix/rebuttal draft。正式论文版可继续补 legend、预测框 overlay 和更紧凑 caption。

## 2026-07-19 - P8 appendix support consolidation

### 目的

继续推进 `target.md` P8，在不引入新主指标的前提下，把 runtime、control overhead、PPS convergence 和 NS3 request-level reliability 整理成可放附录/rebuttal 的证据包。

### 结果

- 新增 `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/runtime_control_ns3_appendix.md`。
- 新增 `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/runtime_control_ns3_summary.csv`。
- 新增 `docs/doc_workspace/SGCP/artifacts/appendix_support_20260719/qualitative_case_study.md`，基于既有 failure/grid diagnostics 选取 `000068/438`、`000066/401`、`000062/337` 三个帧级案例。
- 更新 `target.md`：P8 中 Runtime/control overhead、NS3 request-level reliability、Qualitative case study 和 Appendix raw results 第一版标记完成。
- 更新 `status.md` / `results.md`：记录附录证据包路径和论文写作边界。

### 关键口径

- SGCP control-plane prototype：平均 105.24 ms，最大 127.58 ms；这是 near-real-time feasibility，不是 detector-inclusive end-to-end 100 ms guarantee。
- PAPG 11 帧 NS3 replay：110/110 scheduled requests application callback 与 RLC complete，0 PHY failures，平均/p95 delay 23.91/24.00 ms。
- 控制 metadata：187,112 bytes，4,563.71 bytes/frame；相对 PAPG main raw payload 约 0.58%，可写为 below 1%。
- Qualitative case study：三个案例分别对应 best-view sender 未调度、同簇内 sender/grid 选择不当、dense grid 仍缺 object-level detector support。当前是文字/表格草稿，进入论文前应生成 BEV overlay。

### 训练 watcher

远程 early-fusion checkpoint fine-tune watcher 仍在等待 GPU：`mindspore-187` 上 8 张 3090 当前各占约 22.2GB，`/data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log` 正常轮询。

## 2026-07-19 - Remote early-fusion checkpoint fine-tune setup

### 目的

按用户要求将 SGCP 最大风险项转为可执行训练任务：提升 `pointpillar_early_fusion` checkpoint，使 SGCP 和 Pure late controlled baseline 都使用同一个 raw point-cloud-to-box checkpoint。

### 远程约束

- SSH：`mindspore-187`
- 远程工作目录：`/data2/gzc/sgcp_early_train/`
- conda 环境：`opencood-gzc`
- 远程代码：`/data2/gzc/code/OpenCOOD`

### 执行结果

- 已上传当前 early checkpoint：`/data2/gzc/sgcp_early_train/checkpoints/latest.pth`。
- 已建立远程训练配置：`/data2/gzc/sgcp_early_train/configs/pointpillar_early_ckpt_compat_onecav.yaml`。
- 远程默认 OPV2V train/val 数据不完整；先用 `/data2/gzc/dataset/opv2v/test/2021_08_24_20_49_54/216` 建立 one-CAV symlink split，`find -L` 可见 train/val 各 124 个 `.pcd`。
- 第一次 smoke 使用远程默认 early yaml 失败：checkpoint head 为 384 channel，远程默认模型 head 为 256 channel。
- 改用 checkpoint 配套 config 后，checkpoint 已在远程 OpenCOOD 中加载成功：`0 missing keys, 0 unexpected keys`。
- 当前训练阻塞不是代码或 checkpoint，而是 GPU 资源：8 张 RTX 3090 均被 `VLLM::Worker` 占用约 22.2GB，PointPillar smoke 在卷积处报 cuDNN 无可用算法。
- 已启动后台 watcher：`/data2/gzc/sgcp_early_train/runs/start_train_when_gpu_free.sh`，PID `1532887`，日志 `/data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log`。任一 GPU 显存低于 6000 MiB 后会自动使用 `opencood-gzc`、batch size 1、`--max_steps 200`、`--save_step_freq 50` 启动 fine-tune。

### 下一步

- 继续推进 target 中的实验图表和论文结构，同时轮询 `/data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log`。
- 训练完成后回收 step checkpoint，先做本地 11 帧 smoke，再重跑 41 帧 SGCP-PAPG 与 Pure late controlled baseline，确保二者使用同一个 early checkpoint。

## 2026-07-19 - Pareto source data first pass

### 目的

训练等待 GPU 空闲期间，继续推进 `target.md` P4：先把已复现的 41 帧结果整理为 AP-Mbps Pareto 曲线源数据。

### 结果

- 新增 `docs/doc_workspace/SGCP/artifacts/pareto_20260719/pareto_source.csv`。
- 新增 `docs/doc_workspace/SGCP/artifacts/pareto_20260719/pareto_notes.md`。
- 新增绘图脚本 `docs/doc_workspace/SGCP/artifacts/pareto_20260719/plot_pareto.py`。
- 已生成 `figure1_pareto_ap03.png/.pdf` 与 `figure1_pareto_ap07.png/.pdf`。
- 数据点覆盖 Head-only、Pure late detection-box broadcast/all-to-all、FullPerception-PCS、SGCP coverage/target-aware/PAPG 参数点、forced random、Density/Link-aware、EdgeCooper-HD、PACP-LiDAR、cluster-local/global selective proxy 和 Full20Early upper reference。

### 当前解释

- PAPG 在 `62.54 Mbps` 达到 `0.81/0.78/0.39`，相比 forced random 近似同 payload 提升 AP，相比 density/link-aware 少约 15% raw-LiDAR payload 并提升 AP@0.3/AP@0.5。
- EdgeCooper-HD / PACP-LiDAR proxy 在 AP@0.7 更强，应作为 edge/global 或 proxy boundary，而不是 V2V-only SGCP 的直接失败。
- Pure late prediction sharing payload 很低，必须作为 prediction-sharing reference 单独解释，不能再写成 0 Mbps 同类 baseline。

## 2026-07-19 - Protocol and fusion breakdown figure drafts

### 目的

继续推进 `target.md` P5/P6，将已复现的 protocol-native manifest 与 fusion scaffold manifest 转为可直接检查的 Figure 2/3 草稿。

### 结果

- 新增 `docs/doc_workspace/SGCP/artifacts/figures_20260719/plot_breakdowns.py`。
- 新增 `docs/doc_workspace/SGCP/artifacts/figures_20260719/figure_notes.md`。
- 生成 Figure 2：`figure2_protocol_breakdown.png/.pdf`。
- 生成 Figure 3：`figure3_fusion_contribution.png/.pdf`。

### 验证

```powershell
conda run -n opencda python docs\doc_workspace\SGCP\artifacts\figures_20260719\plot_breakdowns.py
```

图表已目视检查。图内通信标注使用 raw LiDAR Mbps，避免 Pure late 被误读为完全无通信；Pure late detection-box overhead 继续由 `late_fusion_box_comm.md` 解释。

## 2026-07-19 - Table 4 parameter sensitivity candidate

### 目的

推进 `target.md` P7，将已有 `rho_th`、`N_max`、`T_min^stab`、子信道数/带宽和 `B_h` sweep 整理成论文 Table 4 候选，并区分主文可用结论和附录弱结论。

### 结果

- 新增 `docs/doc_workspace/SGCP/artifacts/parameter_sensitivity_20260719/table4_parameter_sensitivity.csv`。
- 新增 `docs/doc_workspace/SGCP/artifacts/parameter_sensitivity_20260719/table4_parameter_sensitivity.md`。
- 主文建议只放 `rho_th` 与子信道数：二者有清楚 AP-Mbps tradeoff。
- `N_max` / `T_min^stab` 建议放附录或 rebuttal：当前短序列中 `N_max` AP 非单调，`T_min^stab=100--1000 ms` 不敏感。

### 解释边界

该表不是最终补实验的替代品。若论文必须强论证稳定窗口，需要重新导出更长或更动态的 CARLA 场景；当前表只能支持“短序列中稳定窗口不敏感，默认值按 10 Hz sensing cycle 的保守滞回设置”。

## 2026-07-19 - Paper experiment section first sync

### 目的

推进 `target.md` P9，将 P1--P7 的最新图表和边界口径同步到 `C:\Workspace\icdcs-paper\SGCP\main.tex`。

### 修改

- 复制 Figure 1/2/3 的 PDF 到 `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_*.pdf`。
- `tab:mAP` 加入 Pure late prediction-sharing reference，并在 caption/正文中说明 raw-LiDAR payload 与 detection-box overhead 的区别。
- 新增 `fig:protocol_breakdown`、`fig:fusion_contribution`、`fig:pareto`。
- 新增 `tab:scheduler_comparison`，将 SGCP-compatible scheduler comparison 从 protocol-native system comparison 中分离出来。
- 将旧 density-only sensitivity 表替换为合并后的 `tab:param_sensitivity`，主文只保留 `rho_th` 和 channel count，`N_max/T_min^stab` 作为弱结论边界说明。

### 验证

- `where.exe latexmk`、`where.exe pdflatex`、`where.exe bibtex` 均未找到，因此未能编译 PDF。
- 轻量 LaTeX 结构检查通过：3 个 `table`、1 个 `table*`、2 个 `figure`、4 个 `figure*`、4 个 `tabular` 的 begin/end 全部配对。
- label/ref 检查通过：32 个 label 均唯一，正文 refs 无缺失。

## 2026-07-18 - Results index baseline naming sync

### 目的

检查 `results.md` 首页和 `baseline_fairness.md` 是否仍保留旧 NC/TBD 或旧 proxy 名称，避免后续自动任务从核心结果索引读取过期口径。

### 结果

- 将 `results.md` 主结果表第一行从 `NC | TBD` 改为当前可复现 lower reference：`Head-only = 0.26/0.22/0.09, 0.00 Mbps`。
- 清理 `results.md` 与 `baseline_fairness.md` 中 `fullperception_rsu` / `fullperception_decentralized` 的当前主表说明，改为直接使用 `global_selective_proxy` / `cluster_local_selective_proxy` 当前命名。
- 同步清理 `baseline_reproduction_plan.md` 和 `fullperception_baseline_revision.md` 中的 `Renamed from` / `formerly` 表述；历史命令和 artifact 路径仍保留在 `log.md`，但当前索引文档只使用现名。
- 保留 `reproducibility_manifest.md` 中旧 NC / `22.33 Mbps` 记录作为“不可复现旧主表”历史说明，不迁入当前主结果表。

下一步：若继续清理，可扫描 `results.md` 中更深处的历史说明，确保凡是旧命名都明确标为历史记录而非当前主表行。

## 2026-07-18 - Novelty and claim-boundary wording pass

### 目的

继续检查 `main.tex` 的 abstract / introduction / conclusion 是否与当前 reviewer response matrix 的保守口径一致，避免 novelty 和机制保证写得过强。

### 结果

- 将 introduction 中 `To the best of our knowledge, no prior work...` 改为 `SGCP targets the less explored combination...`，避免与 Smartform、Where2Comm/PACP 或其他 decentralized CP 工作产生不必要的绝对 novelty 冲突。
- 将方法概述中的 `preserves one strong external view` 改为 `prioritizes a strong external view`，与 PAPG coverage layer 的调度偏好一致，避免被误读成每帧每 head 的硬保证。
- 复查 potential-guided scheduling 段已经限定为 fixed cluster/candidate/hard feasibility assumptions，并说明 implementation uses the potential as a scheduling guide；该段无需进一步降调。

下一步：若继续 final freeze，优先检查 `results.md` 首页是否还有 `NC = TBD` 或旧主表索引会误导后续写作。

## 2026-07-18 - Main-table lower reference cleanup

### 目的

继续核对 `main.tex` 主表与当前可复现结果文档的一致性，重点检查旧 NC 行是否仍混入当前 PAPG 主表。

### 结果

- 发现 `main.tex` 主表中 `NC = 0.13/0.12/0.10` 仍来自旧论文表。`reproducibility_manifest.md` 已记录旧主表缺少原始日志、种子和代码版本，因此该行不应作为当前可复现主表行。
- 当前可复现 lower reference 是 `Head-only = 0.26/0.22/0.09`：cluster heads 不接收点云 upload，只做本地检测并走同一 inter-cluster late-fusion path。
- 已将 `main.tex` 的 baseline 定义和主表首行从 `NC` 改为 `Head-only`，并同步 `main_table_candidate.md` 的命名，避免把旧 NC 与 head-only ablation 混写。

下一步：继续保持旧 NC 仅在 `reproducibility_manifest.md` 中作为历史不可复现主表记录，不进入当前论文主表。

## 2026-07-18 - Paper citation key sanity check

### 目的

继续推进 final paper freeze 前的论文一致性检查，重点排查 `main.tex` 中旧结果、旧 baseline 命名和 BibTeX citation key 不一致问题。

### 检查结果

- `main.tex` 未发现旧 `22.33 Mbps`、旧 `0.85/0.84/0.69` 主表口径、`FullPerception-centralized`、`fullperception_rsu` 或 `fullperception_decentralized` 残留。
- 发现并修复 4 个 BibTeX key 大小写不一致问题：`arnold2020nmslate -> arnold2020nmsLate`、`liu2019fusioneyelate -> liu2019fusioneyeLate`、`li2023hetsdn -> li2023hetSDN`、`zhao2025multidrl -> zhao2025multiDRL`。
- 清理 optimization problem 中旧注释掉的 `\ref{eq:power_constraint}` / `\ref{eq:instant_power}`，避免后续静态 label 检查误报。
- 复查通过：28 个 unique cite keys 全部匹配 `Reference.bib`；28 个 labels 与 14 个 unique refs 全部匹配；正文中未检出旧主表数值或旧 baseline 名称。

下一步：本机仍缺 `latexmk/pdflatex`，无法完整 PDF 编译；若后续安装 LaTeX 工具链，应优先做版面和参考文献编译检查。

## 2026-07-18 - Reviewer response coverage matrix

### 目的

核查 `C:\Workspace\icdcs-paper\SGCP\SGCP-review.txt` 中关于机制、公平 baseline、参数实验、实时性和 topology trigger 的意见是否都已有 rebuttal / `main.tex` 落点，避免最终回复时遗漏。

### 结果

- 新增 `reviewer_response_matrix.md`，按 Reviewer 2/3/4 映射 concern、current response、evidence/location 和 remaining risk。
- 更新 `readme.md`，把该矩阵加入 SGCP 文档工作区索引。
- 扫描 `C:\Workspace\icdcs-paper\SGCP\main.tex` 中的强 claim，未发现旧 `22.33 Mbps`、旧 AP 或 `FullPerception-centralized` 残留；将摘要中的 `guaranteed external views` 改为 `external-view coverage`，并弱化 coalition non-negativity 段的 `guaranteeing` 表述。
- 主要剩余风险：LaTeX 尚未完整编译；Where2Comm/PACP 等模型级严格复现未完成，只能以 same-backbone V2V selective proxy 作为当前公平 baseline；在线 CARLA+NS3 deadline-aware AP 与离线 final-delivery AP 仍需分口径描述。

下一步：在 final paper freeze 前检查 `main.tex` 是否仍有 “outperforms all baselines”、“guaranteed 100 ms” 或无条件 “exact potential game” 等过强表述。

## 2026-07-18 - Paper main-table LaTeX structure sanity check

### 目的

`C:\Workspace\icdcs-paper\SGCP\main.tex` 已改为分组主表。由于本机未检测到 `latexmk/pdflatex`，本轮做轻量 LaTeX 结构检查，避免表格列数或 begin/end 配对错误。

### 检查结果

- 未检测到 `latexmk` / `pdflatex`，因此仍未完成 PDF 编译验证。
- `table/table*` begin/end：3/3。
- `tabular` begin/end：3/3。
- `tab:mAP` 主表普通数据行均为 5 列，即 4 个 `&`。
- 分组行使用 `\multicolumn{5}{...}`，与 `|l|ccc|c|` 的 5 列结构匹配。

结论：主表分组布局通过轻量结构检查；后续若安装 LaTeX 工具链，应再做完整 PDF 编译和版面检查。

## 2026-07-18 - FullPerception MWS/RS heuristic sanity pass

### 目的

PCS tuned 后，复核同论文 heuristic baseline（MWS/RS）是否应采用相同 blind-spot 粒度，并确认它们是否可以进入主表。

### 代码改动

- `MWS` / `RandomRA` 不再硬编码 `min_division=1,min_overlap=50`，改为复用 `PCS.blind_spot_min_division/min_overlap_grids`。
- 两个 heuristic 都记录 `resource_sc_nums`，使 NS3 request plan 的 `sc_num` 与 scheduled link payload 口径一致。
- 网格选择只上传 sender 实际覆盖的 blind-spot grids。
- 修复 `RandomRA` 新路径缺少 `common` import 的 bug。

### 命令

```powershell
$artifact='docs\doc_workspace\SGCP\artifacts\fullperception_heuristics_20260718'
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-constrained --resource-allocation fullperception_mws --sgcp-receiver-policy all-scheduled-receivers --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output "$artifact\fullperception_mws_11f_tuned_trace.csv"
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-constrained --resource-allocation fullperception_random --sgcp-receiver-policy all-scheduled-receivers --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output "$artifact\fullperception_random_11f_tuned_trace.csv"
```

### 结果

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FullPerception-MWS tuned | 11 | 0.36 | 0.32 | 0.15 | 4,289,344 | 39.00 | Greedy heuristic remains weak despite higher payload |
| FullPerception-RS tuned | 11 | 0.54 | 0.49 | 0.23 | 1,644,160 | 14.95 | Random heuristic is closer to PCS on 11f but far below SGCP/strong selective baselines |

结论：MWS/RS 统一 tuned blind-spot 口径后仍不适合作主公平 baseline。它们应作为 FullPerception PCS 的 heuristic/ablation 诊断，而主表保留 tuned PCS、PAPG、payload-matched selective baselines、EdgeCooper-HD 和 full-sharing upper reference。

## 2026-07-18 - FullPerception proxy rename and PCS tuning pass

### 目的

用户要求避免 `fullperception_rsu` / `fullperception_decentralized` 误导论文命名，并在不修改 20MHz/10ch 网络参数的前提下调试仓库内置 PCS，使 FullPerception baseline 能以合理结果进入主表。

### 代码改动

- `offline_inference --selective-sharing-baseline` 中旧 `fullperception_rsu` 改名为 `global_selective_proxy`。
- 旧 `fullperception_decentralized` 改名为 `cluster_local_selective_proxy`。
- 删除 `builder.py` 中易混淆的 `fullperception_rsu_pcs` alias；保留 `fullperception` / `fullperception_pcs` 指向 `pcs.py`。
- `PCS` 修复 blind-spot cache key 忽略 split 粒度、grid mAP cache 索引错误、utility/payload/grid-selection 使用不同 blind-spot 粒度的问题。
- `PCS` 默认 blind-spot split 调为 `blind_spot_min_division=12`、`min_overlap_grids=0`，将论文 blind spot 单元拆成更可调度的小盲区；带宽和子信道数仍由 `--bandwidth-mhz 20 --num-channels 10` 固定。
- `offline_inference` / `offline_ns3_replay` 新增 `--pcs-blind-spot-min-division` 和 `--pcs-min-overlap-grids`，用于后续不改网络资源的 PCS 参数复核。

### 命令

```powershell
$artifact='docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718'
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output "$artifact\pcs_41f_tuned_div12_ov0_trace.csv"
conda run --no-capture-output -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --resource-allocation fullperception_pcs --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --dry-run --upload-plan-output "$artifact\pcs_11f_tuned_div12_ov0_dryrun_plan.csv"
```

### 结果

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PCS tuned `division=12,min_overlap=0` | 41 | 0.59 | 0.53 | 0.22 | 12,959,840 | 25.29 | Current formal FullPerception PCS baseline |
| PCS tuned NS3 dry-run | 11 | N/A | N/A | N/A | request plan only | N/A | 5 scheduled requests/frame, 0 skipped unscheduled |

11 帧扫描中，`division=12,min_overlap=0` 达到 `0.57/0.54/0.30`，`division=4,min_overlap=1` 达到 `0.58/0.51/0.24`。完整 41 帧选择前者作为默认，因为 AP@0.5/AP@0.7 更稳。结论：PCS baseline 不再是异常 under-schedule，但仍明显低于 PAPG/EdgeCooper/strong selective proxy，应作为正式 FullPerception baseline 而不是主张 SGCP 击败所有上界的依据。

## 2026-07-18 - FullPerception PCS protocol repair pass

### 目的

用户指出 FullPerception 原实现应对应 `opencda/core/clustering/algorithms/resource_allocation/pcs.py`。本轮在确认 `pcs.py` 与 FullPerception 论文 PCS 结构匹配后，修复第一批会导致协议口径不正确的工程简化，并重新运行 built-in PCS。

### 代码改动

- `PCS._get_link_required_subchannels()` 不再直接 `return 1`，改为根据 sender 覆盖 blind grids 的点云 payload、`bandwidth_all`、`lambda_subchannels` 和 `time_slot` 估计 required subchannels。
- `PCS` 新增 `resource_sc_nums`，`update_resource_allocation_strategy()` 会把每条 scheduled link 的 `sc_num` 写入 `ClusteringScheduler.channel_allocation_sc_nums`。
- `offline_inference` / `offline_replay` / `offline_ns3_replay` 会把 `--num-channels`、`--bandwidth-mhz` 和 world `time_slot` 同步给 PCS-like allocator。
- `offline_inference` 新增 `--sgcp-receiver-policy all-scheduled-receivers`，用于评估 PCS 这类不一定只调度 coalition cluster head 的资源分配器。
- `offline_ns3_replay` 在 scheduled-only 模式下直接遍历 `channel_allocation` 中的 scheduled links，不再用 coalition cluster membership 过滤 PCS 全局链路；upload plan 中的 `sc_num` 来自 PCS 的 required subchannels。

### 命令

```powershell
$artifact='docs\doc_workspace\SGCP\artifacts\fullperception_pcs_20260718'
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output "$artifact\pcs_scheduled_receivers_afterfix_trace.csv"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --resource-allocation fullperception_pcs --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --dry-run --upload-plan-output "$artifact\pcs_dryrun_plan_afterfix_41f.csv"
```

### 结果

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Scheduled receiver rows | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy PCS all-cluster-head eval | 0.44 | 0.39 | 0.17 | 12,684,880 | 24.75 | 246 | Pre-repair compatibility row |
| Repaired PCS scheduled-receiver eval | 0.33 | 0.29 | 0.14 | 8,100,112 | 15.80 | 104 | Payload-based `c(q)`, real `sc_num` |

`offline_ns3_replay --dry-run` 的 41 帧 upload plan 共 104 条 scheduled request，`sc_num` 不再恒为 1；3 帧 plan 中已观察到 `sc_num=2/3` 的多子信道请求。

### 结论

第一轮修复让 FullPerception PCS 的协议语义更正确，但结果更弱，说明当前内置 PCS 在该 dump 上仍明显 under-schedule。下一步不应在旧结果上“修表”，而应继续校准 PCS 的 RSU/global receiver fusion、多 blind-spot link treatment、payload/utility scaling，并补 `fullperception_decentralized` 的真实 NS3 replay。

## 2026-07-18 - FullPerception-Decentralized NS3 replay

### 目的

PAPG 和 forced-budget random 已有 11 帧真实 NS3 replay。为让强 V2V baseline 的链路证据对称，本轮补 `fullperception_decentralized` 的 scheduled-only socket replay。

### 命令

ns-3 启动：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd /home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
```

OpenCDA replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --selective-sharing-baseline fullperception_decentralized --selective-member-budget 3 --selective-grid-budget 117 --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\fullperception_decentralized_ns3_20260718_007\upload_plan.csv --drain-seconds 0.3 --sync-timeout 30
```

评估：

```powershell
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\fullperception_decentralized_ns3_20260718_007\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\fullperception_decentralized_ns3_20260718_007\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\fullperception_decentralized_ns3_20260718_007\eval --rsu-node-id 21 --max-frames 11
```

### 结果

| Metric | Value |
| --- | ---: |
| Planned / scheduled requests | 110 |
| Application `cam_received` | 110 |
| Matched callback ratio | 1.000 |
| RLC complete requests | 110 / 110 |
| RLC TX / RX events | 2970 / 2970 |
| RLC drops | 0 |
| PHY decode failures | 0 |
| Avg / p95 callback delay | 23.91 / 24.00 ms |

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\fullperception_decentralized_ns3_20260718_007\
```

### 观察

Windows `Get-NetTCPConnection` 看不到 WSL 内部监听的 5556 端口；需要用 WSL 内 `ss -ltnp | grep :5556` 确认 ns-3 是否已启动。本轮结束后已停止本次启动的 ns-3 进程，避免后台残留。

### 结论

FullPerception-Decentralized 的 11 帧链路交付与 PAPG、forced-budget random 一样完整。因此它作为强 V2V baseline 的 AP 差异不来自 NS3 丢包，而来自调度/点云选择策略本身。

## 2026-07-16 - 主表结果修复目标重开

### 目的

根据当前复现结果，SGCP 主表 AP 与论文旧表存在较大差距，且现有主表结果难以直接写入论文。本轮将下一阶段最高优先级改为：先审计并修复离线/在线实验协议，确认分簇、点云选择和子信道分配真实影响融合结果；再通过 CARLA/NS3 验证；若协议无误但 AP 仍低，则展开消融和算法改造，最终形成“较少通信量 + 较高 AP”的可写主表，并据此修改论文和 rebuttal。

### 已更新

- 在 `target.md` 顶部新增 `P-1：最高优先级 - 主表结果修复与论文落地`。
- 重新打开 SGCP heartbeat 自动化，间隔 31 分钟。

### 下一步

- 审计 `offline_inference` / `offline_replay` 中 cluster、grid selection、channel allocation 到 OpenCOOD 输入的链路。
- 设计单帧 probe 和多帧 CSV trace，确认每个机制开关都会改变 fused input、payload 和 AP。

## 2026-07-16 - Offline SGCP protocol audit

### 目的

检查当前离线测试流程中，分簇结果、点云选择、子信道分配结果是否能够正确影响融合结果。

### 代码修正

- `opencda.tools.offline_inference` 新增 `--sgcp-trace-output`。
- `build_constrained_frame()` metadata 新增 `cluster_member_ids` 和 `channel_allocation`。
- Trace CSV 记录 receiver、cluster members、source CAVs、uploaded sources、selected grids、point counts、payload、channel allocation、missing channel sources、pred/gt boxes。

### 命令

单帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --timestamp 000060 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_000060_trace.csv
```

41 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_trace.csv
```

### 日志路径

```text
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_000060_stdout.log
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_000060_trace.csv
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_stdout.log
docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_trace.csv
```

### 结果

- 单帧 `000060`：6 个 cluster-head receiver 的 uploaded sources 均有 channel allocation，`missing_channel_sources` 全空。
- 41 帧：246 条 receiver trace，`missing_channel_rows=0`。
- 41 帧 AP@0.3 / AP@0.5 / AP@0.7：0.77 / 0.73 / 0.35。
- 41 帧总通信：26,916,208 bytes，平均 109,415.48 bytes / receiver sample。

### 结论

- 分簇结果真实决定 cluster head / member / fusion source 列表。
- `PotentialGame` 输出的 grid selection 真实裁剪 sender 点云，并进入 OpenCOOD early fusion 输入。
- 当前离线融合没有发现未调度 sender 绕过 PPS 进入融合；主表 AP 偏低更可能来自 grid/cluster/late-fusion 质量和通信预算，而不是协议链路未接入。

## 2026-07-16 - Head-only / full-cluster mechanism probe

### 目的

定位当前 SGCP 主表 AP 损失来自 cluster/late fusion 主体，还是来自 grid/PPS 选择。

### 代码修正

- `opencda.tools.offline_inference` 新增 `--sgcp-upload-mode {grid,head_only,full_cluster}`。
- `build_constrained_frame()` 支持三种上传模式：
  - `grid`
  - `head_only`
  - `full_cluster`

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-upload-mode head_only --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_trace.csv

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-upload-mode full_cluster --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\full_cluster_41f_trace.csv
```

### 日志路径

```text
docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_stdout.log
docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_trace.csv
docs\doc_workspace\SGCP\artifacts\mechanism_probe\full_cluster_41f_stdout.log
docs\doc_workspace\SGCP\artifacts\mechanism_probe\full_cluster_41f_trace.csv
```

### 结果

| Mode | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Bytes / Receiver | Avg. Uploaded Sources | Avg. Uploaded Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head-only | 0.26 | 0.22 | 0.09 | 0 | 0.00 | 0.00 | 0.00 |
| SGCP grid-constrained | 0.77 | 0.73 | 0.35 | 26,916,208 | 109,415.48 | 1.67 | 6,838.47 |
| Full-cluster upload | 0.82 | 0.79 | 0.42 | 44,850,528 | 182,319.22 | 2.33 | 11,394.95 |

### 结论

- Cluster-head local-only 很弱，说明协同上传对 AP 确实有贡献。
- Full-cluster upload 在同一 cluster 和 inter-cluster late fusion 结构下效果明显更高，说明 cluster / late fusion 主体可用。
- SGCP grid-constrained 使用约 60.0% 的 full-cluster payload，AP@0.5 保留约 92.4%，但 AP@0.7 损失明显。
- 下一步应聚焦 grid utility、member/grid budget、`B_h=1` 和高精度定位相关 grid selection。

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

## 2026-07-15 - baseline 公平性口径整理

### 目的

- 推进 P2 “补充公平 baseline”。
- 先明确 FullPerception-RSU、FullPerception-Decentralized、full early/late reference 和 SGCP same-pipeline ablation 的层级，避免论文主表中混用不公平 baseline。

### 文档更新

- 新增 `docs/doc_workspace/SGCP/baseline_fairness.md`。
- 在 `results.md` 增加 “Baseline 公平性说明” 小节。
- 在 `target.md` 标记完成：
  - 明确 FullPerception-RSU 设置，作为 centralized/RSU-assisted upper reference。
  - 在 `results.md` 中单独记录 baseline 公平性说明。

### 当前结论

| Method | Layer | 是否作为公平主对比 | 说明 |
| --- | --- | --- | --- |
| Full 20-CAV early fusion | Upper reference | No | 全点云共享，无 SGCP 通信约束 |
| Full 20-CAV late checkpoint | Upper reference | No | 独立 late checkpoint，不是同 checkpoint 消融 |
| FullPerception-RSU | Upper reference | No | 集中式/RSU-assisted reference；当前 `v2xp_cluster_carla` 未启用 RSU |
| SGCP constrained + inter-cluster late fusion | Main method | Yes | 当前完整 SGCP 离线主口径 |
| Random scheduler | Same pipeline ablation | Yes | 同 SGCP clustering + late-fusion path，替换 PPS |
| MWS scheduler | Same pipeline ablation | Pending | 需要复核 MWS 论文 baseline 定义 |
| Singleton full late-fusion reference | Reference only | No | late-fuse 全部 20 CAV，当前未计 detection-box exchange overhead |

### 下一步

- 实现或整理 FullPerception-Decentralized / same-budget CAV-only selective-sharing baseline。
- 候选：nearest/top-k density/communication-aware top-k，匹配 SGCP 的 payload、source CAV 数或 selected-grid 数。
- 对 singleton reference 估算 prediction-level detection box exchange overhead，避免误写为零通信 baseline。

## 2026-07-15 - same-budget CAV-only selective-sharing baseline

### 目的

- 推进 P2 “实现 same-budget CAV-only selective-sharing baseline”。
- 在不使用 PPS 的前提下，复用 SGCP coalition formation 和 inter-cluster late fusion 评价口径，构造可比的 decentralized V2V baseline。

### 代码更新

- `opencda.tools.offline_inference`
  - 新增 `--selective-sharing-baseline nearest|density|communication_aware`。
  - 新增 `--selective-member-budget`，默认每个 cluster head 最多接收 2 个非 head 成员。
  - 新增 `--selective-grid-budget`，默认每个 cluster head 总 grid budget 为 87，贴近 SGCP 默认 `avg_selected_grids=87.32`。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline nearest --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --max-frames 0
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline density --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --max-frames 0
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline communication_aware --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --max-frames 0
```

### 结果

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Avg. Upload (bytes/source) | Total Upload (bytes) | Avg. Source CAVs | Avg. Selected Grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP `potential_game` | 0.77 | 0.73 | 0.35 | 109415.48 | 26916208 | 2.67 | 87.32 |
| Selective nearest | 0.76 | 0.73 | 0.37 | 113930.21 | 28026832 | 2.81 | 81.38 |
| Selective density | 0.77 | 0.74 | 0.39 | 124286.05 | 30574368 | 2.81 | 81.38 |
| Selective communication-aware | 0.78 | 0.75 | 0.40 | 122854.70 | 30222256 | 2.81 | 81.38 |

### 观察

- `communication_aware` 是当前最强的 CAV-only selective-sharing baseline，在 AP@0.5/AP@0.7 上高于 SGCP，但 payload 也更高。
- 这说明 SGCP 论文不能只对比 weak/random baseline；必须把 same-budget selective baseline 纳入公平性讨论。
- 当前 baseline 仍是离线 first version：它复用 SGCP clustering，但没有建模 PPS channel feasibility、干扰、稳定窗口、拓扑变化控制开销。

## 2026-07-15 - topology change trigger 机制定义

### 目的

- 推进 P3 “定义 topology change trigger，包括邻居变化、相对速度、链路质量或 utility 下降阈值”。
- 修正文稿中 “topology change 才触发” 与 “每个周期重复” 的潜在矛盾。

### 文档更新

- 新增 `docs/doc_workspace/SGCP/topology_trigger.md`。
- 在 `target.md` 中将 topology trigger 机制定义标记为完成，并新增代码接入任务。
- 在 `status.md` 中记录该机制当前仍是规格，尚未接入在线/离线统计。

### 当前机制口径

Trigger 条件包括：

- 邻居集合变化：CAV 进入/离开通信范围，或当前 head/member 不再可达。
- 相对运动风险：预测稳定性低于 `beta_min`。
- 链路质量下降：SINR、data rate、PDR 或 NS3 link-quality 低于阈值。
- Utility 下降：当前 coalition utility 相比上次 accepted state 下降超过阈值。
- Hard failure：head 丢失、成员车辆消失或链路断开。
- Periodic guard：超过最大保鲜周期后强制重评估。

Trigger 输出分为 `NO_CHANGE`、`LOCAL_REPAIR`、`RECLUSTER`。建议每个周期都更新 beacon 和 PPS 输入，但只有触发条件满足时才更新 cluster membership。

### 下一步

- 先在 `opencda.tools.offline_replay` 中实现 trigger 统计，验证当前 41 帧 dump 中 trigger 与 reconfiguration 的对应关系。
- 再考虑接入在线 `ClusteringV2XManager`，实现无事件时跳过 coalition formation。

## 2026-07-15 - topology trigger 离线 replay 统计接入

### 目的

- 推进 P3 “将 topology trigger 接入离线 replay，输出每帧 trigger type、是否触发 reconfiguration、vehicle-head change 的对应关系”。
- 先用 dump 数据做统计，不改变在线 CARLA clustering 行为。

### 代码更新

- `opencda.tools.offline_replay`
  - 每帧保存 CAV 位置、速度、通信半径和邻居集合。
  - 相邻帧比较 topology trigger：`neighbor_set_change`、`relative_speed_risk`、`head_member_unreachable`、`cav_set_change/hard_failure`。
  - Summary 输出 trigger/reconfiguration 对齐统计。
  - 新增 `--print-topology-events` 输出逐 transition 明细。
  - 新增 `--trigger-relative-speed-threshold` 调整相对速度风险阈值；阈值单位使用 dump 中 `ego_speed` 的原始单位。

### Smoke test

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --summary-only --print-topology-events
```

结果：

| Frames | Transitions | Triggered | Actual Reconfig. | Matched | Trigger Types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 2 | 2 | 1 | 1 | `relative_speed_risk`: 2 |

### 41 帧完整统计

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --summary-only --print-topology-events
```

结果：

| Frames | Transitions | Triggered | Actual Reconfig. | Matched | Reconfig. Without Trigger | Trigger Without Reconfig. | Trigger Types |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 41 | 40 | 40 | 11 | 11 | 0 | 29 | `relative_speed_risk`: 40; `neighbor_set_change`: 12 |

### 观察

- 当前 trigger 覆盖了全部实际 reconfiguration，没有出现 “reconfig without trigger”。
- `relative_speed_risk` 在默认阈值下每个 transition 都触发，说明该信号过敏或 dump 中 `ego_speed` 单位/尺度需要复核。
- `neighbor_set_change` 只出现在 12/40 个 transition，其中部分与实际 reconfiguration 重合；它更适合作为强触发信号。
- 下一步若要接入在线 gate，应先改用相邻帧 pose 差分速度或确认 `ego_speed` 单位，再设置 `beta_min/epsilon_u` 滞回，否则会退化成每帧触发。

## 2026-07-15 - topology trigger 速度源复核

### 目的

- 复核上一轮 `relative_speed_risk` 每个 transition 都触发的原因。
- 避免在线 gate 直接使用单位不清的速度阈值。

### 代码更新

- `opencda.tools.offline_replay`
  - 新增 `--trigger-speed-source pose_delta|dump`。
  - 默认速度源改为 `pose_delta`，用相邻帧位置差和 `--trigger-frame-interval-sec` 计算速度，默认帧间隔 0.1 s。
  - 保留 `dump` 速度源，用于复现原始 `ego_speed` 行为。
  - `--trigger-relative-speed-threshold` 对 `pose_delta` 使用 m/s，对 `dump` 使用 km/h。

### 单位确认

- `opencda.core.common.misc.get_speed(vehicle, meters=False)` 默认返回 km/h。
- `DataDumper` 写入的 `ego_speed` 使用默认 `get_speed(veh)`，因此 dump 中 `ego_speed` 是 km/h。
- 在线 `V2XManager` 注释也标注 ego speed 为 km/h。

### 41 帧阈值扫

命令模板：

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --summary-only --trigger-speed-source pose_delta --trigger-relative-speed-threshold <m/s>
```

| Speed Source | Threshold | Triggered | Actual Reconfig. | Matched | Reconfig. Without Trigger | Trigger Without Reconfig. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pose_delta` | 3 m/s | 40 | 11 | 11 | 0 | 29 |
| `pose_delta` | 4 m/s | 40 | 11 | 11 | 0 | 29 |
| `pose_delta` | 5 m/s | 37 | 11 | 9 | 2 | 28 |

### 观察

- `pose_delta` 解决了 km/h/m/s 混用问题，但 relative-speed trigger 仍然不是一个足够精确的单独 gate。
- 3/4 m/s 能覆盖全部实际 reconfiguration，但误触发仍多；5 m/s 会漏掉 2 次实际 reconfiguration。
- 在线 `ClusteringV2XManager` 不应只用 relative speed 决定是否重构，应组合：
  - hard failure/head-member unreachable 立即触发；
  - neighbor-set change 作为强触发；
  - relative speed 作为风险提示；
  - utility drop 和 `T_min_stab` 滞回决定是否真正进入 `RECLUSTER`。

## 2026-07-15 - 在线 topology trigger gate first version

### 目的

- 推进 P3 “将 topology trigger gate 接入在线 `ClusteringV2XManager`”。
- 先做默认关闭的安全接入，不改变当前默认 CARLA/离线实验行为。

### 代码更新

- `opencda.core.common.config_manager.ClusteringConfig`
  - 新增 `enable_topology_trigger_gate: bool = False`。
  - 新增 `topology_periodic_guard: int = 0`。
- `opencda/scenario_testing/config_yaml/networking_clustering.yaml`
  - 显式记录 `enable_topology_trigger_gate: false`。
  - 显式记录 `topology_periodic_guard: 0`。
- `opencda.core.clustering.managers.clustering_v2x_manager.ClusteringV2XManager`
  - 新增 class-level accepted topology signature。
  - 新增邻居集合 signature 计算。
  - 新增 head/member reachability failure 检查。
  - 周期到达时先调用 `_should_recluster()`；若无 topology change，则跳过 `CoalitionGame.run()` 并沿用上一轮 cluster。
  - 日志输出 `CLUSTER_TRIGGER recluster/skip reason=...`。

### 当前 gate 口径

| Trigger | 当前在线 gate | 说明 |
| --- | --- | --- |
| Initial state | Yes | 首次必须运行 clustering |
| Neighbor-set change | Yes | 通信范围内邻居集合变化时重构 |
| Head/member unreachable | Yes | 当前 cluster 内 head-member 不可达时重构 |
| Periodic guard | Yes | `topology_periodic_guard > 0` 时启用 |
| Relative-speed risk | No | 离线统计显示单独使用偏敏感 |
| Utility drop | No | 待补上一次 accepted utility 缓存 |
| NS3 link-quality drop | No | 待接入 NS3/offline NS3 link-quality |

### 验证

命令：

```powershell
conda run -n opencda python -m py_compile opencda\core\common\config_manager.py opencda\core\clustering\managers\clustering_v2x_manager.py
```

结果：通过。

### 下一步

- 在真实 CARLA 在线仿真中打开 `enable_topology_trigger_gate: true` 做回归。
- 记录 `CLUSTER_TRIGGER` 日志、reconfiguration 次数、感知 AP/稳定性变化。
- 后续再把 utility drop 和 NS3 link-quality drop 加入 gate。

## 2026-07-15 - cluster capacity / merge-split 机制说明

### 目的

- 推进 P3 “设计 cluster 已满时的处理策略”。
- 推进 P3 “明确是否支持 cluster merge/split，并说明与 `N_max` 的关系”。
- 推进 P3 “补充成员加入后的边际贡献重算流程”。

### 代码观察

- `CoalitionGame.coalition_formation()` 中，当候选 cluster `c.size() >= self.p.N_max` 时直接跳过。
- 当前实现不会临时超过 `N_max`。
- 当前实现不会主动 replacement、等待队列、split 或 merge。
- 若车辆无法加入更优未满 cluster，则保留当前 cluster；如果它本来是 singleton，则继续作为 singleton。

### 文档更新

- 新增 `docs/doc_workspace/SGCP/cluster_capacity_policy.md`。
- 在 `readme.md` 中加入该文档说明。
- 在 `target.md` 中将 cluster 已满策略、merge/split 关系、成员加入后边际贡献重算三项标记完成。

### 当前机制口径

| Issue | Current Position |
| --- | --- |
| Cluster full | `N_max` 是硬上限，默认不允许加入满簇 |
| Vehicle blocked by full cluster | 保留当前 cluster 或 singleton fallback |
| Replacement | 可作为 optional local repair，但当前不声称已实现 |
| Merge | 只允许通过 coalition move 形成 `size <= N_max` 的结果 |
| Split | 通过 topology trigger + re-cluster 间接发生 |
| Marginal contribution recompute | 每轮迭代基于更新后的 coalition state 重新计算，`ita` 抑制振荡 |
| Compensation | singleton/small cluster 仍可通过 inter-cluster late fusion 输出检测结果 |

### 下一步

- 在离线 replay 中统计满簇数量、因 `N_max` 跳过的候选 move 数。
- 若论文需要更强机制，再实现默认关闭的 replacement repair，并补消融。

## 2026-07-16 - SGCP 离线 NS3 request-level replay

### 目的

- 复用 LGCP 已补充的 NS3 request-level trace 能力，推进 SGCP 离线实验 + NS3 仿真闭环。
- 先验证 SGCP intra-cluster transfer request 能输出 `upload_plan.csv`，并能和 NS3 CAM/RLC 日志按 `pkt_id` 对齐。

### 代码更新

- 新增 `opencda.tools.ns3_log_eval`，作为通用 NS3 request-level 日志解析入口，复用现有 LGCP parser。
- 扩展 `opencda.tools.offline_ns3_replay`：
  - 新增 `--upload-plan-output`。
  - SGCP 模式下输出 `timestamp/source_id/target_id/bytes/upload_type/pkt_id`。
  - LGCP 模式也复用同一输出机制。

### Dry-run

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --dry-run --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_upload_plan.csv
```

结果：11 帧，每帧 20 车、6 个 cluster、14 条 SGCP intra-cluster request，共 154 条 request、1,540,000 bytes。

### NS3 replay

NS3 启动命令：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd /home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=20.0 --enableTimeSync=true --carlaHost=auto'"
```

SGCP replay 命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --drain-seconds 0.5 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_upload_plan.csv
```

运行结果：NS3 bridge 成功连接，11 帧均完成 `sync_request/sync_ack` 和 transfer request 发送。

### Log eval

命令：

```powershell
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_11f_eval --max-frames 11
```

结果：

| Metric | Value |
| --- | ---: |
| Planned requests | 154 |
| Planned bytes | 1,540,000 |
| Observed `cam_received` | 86 |
| Bridge-observed delivery ratio | 0.558442 |
| Avg. delay | 26.756 ms |
| P95 delay | 28.000 ms |
| Max delay | 101.000 ms |
| PHY decode events | 2,512 |
| PHY decode failures | 0 |
| RLC TX events | 4,158 |
| RLC RX events | 2,512 |
| Requests with any RLC RX event | 150 |
| Any RLC RX ratio | 0.974026 |

### 观察

- SGCP 离线 replay 已能生成和 NS3 request-level trace 对齐的 upload plan。
- application `cam_received` ratio 明显低于 any-RLC-RX ratio；论文中应明确这两个口径代表不同层级。
- Any-RLC-RX 仅表示某个 request_id 至少出现一个 RLC RX 片段/事件，不表示完整 request 已被应用层接收；完整应用可见交付仍以 `cam_received` 为准。
- 当前结果还没有反馈到 OpenCOOD mAP；下一步需要把 request delivery/PDR 接入 SGCP PPS 或 constrained inference 的传输过滤。

## 2026-07-16 - NS3 RLC 指标口径修正

复核 `cam-application.cc` 后确认：`cam_received` 在 CAM header 被应用层 receiver 解析后触发，代表完整 CAM/UDP packet 到达应用回调；RLC RX trace 则可能是一条 request 的某个片段/事件。因此此前文档中的 `RLC request RX ratio = 0.974026` 容易被误读为完整 request delivery。

已将评估脚本字段改为：

- `requests_with_any_rlc_rx`：至少出现一个 RLC RX 片段/事件的 request 数，当前为 150/154。
- `request_any_rlc_rx_ratio`：上述 partial RLC reception 比例，当前为 0.974026。
- `cam_received` / `bridge_observed_delivery_ratio`：完整应用回调交付，当前为 86/154 = 0.558442。

后续论文写作必须区分 application callback、RLC partial reception、RLC request completion 和 PHY decode diagnostics。下一步需要补 request completion 口径，按 request_id 统计 TX/RX segment 完整性和 DROP 事件，再决定如何将链路结果接入 SGCP PPS 或 mAP 过滤。

## 2026-07-16 - NS3 manual subchannel bug fix and validation

目标：将“OpenCDA 指定的子信道真实落到 NS3 侧发送行为”列为最高优先级，先修通信仿真正确性，再继续推进 SGCP 论文任务。

### 修复内容

- `ns3/vanet/main.cc`
  - 新增 `--targetSubchannels`、`--slSubchannelSize`、`--slBandwidthIn100kHz` 参数。
  - 默认 `--targetSubchannels=10`，与 OpenCDA `networking_clustering.yaml` 的 `subchannel_num: 10` 对齐。
  - 修复 socket receive buffer：长 JSON payload 由覆盖改为累积，避免 `transfer_requests` 被分片时丢前半段。
- `ns3/src/nr-sl-ue-mac-scheduler-manual.cc/.h`
  - manual scheduler 不再把 `sc_start` 当作“第 N 个候选资源”并取模 wrap。
  - 改为严格匹配 `physicalStart == sc_start` 且 `physicalLen == sc_num`。
  - 越界命令保留在队列中并拒绝分配，阻止 RLC buffer 被默认调度器随机发送；同一无效命令只记录一次 reject。
- `ns3/vanet/cam-application.cc`
  - 修正 manual command 的 `maxDataSize`，覆盖 NR SL RLC 实际分片总量，避免最后 4 bytes 残片回落到默认随机调度。
- `opencda.tools.ns3_link_probe`
  - 新增不依赖 CARLA 的确定性链路探针，用于构造 success / edge_success / conflict / out_of_band 四类请求。

### 测试命令

NS3：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 45s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=1.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
```

Probe：

```powershell
conda run -n opencda python -m opencda.tools.ns3_link_probe --case success --packet-size 400 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\ns3_link_probe_success\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_link_probe --case edge_success --packet-size 400 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\ns3_link_probe_edge_success\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_link_probe --case conflict --packet-size 400 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\ns3_link_probe_conflict\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_link_probe --case out_of_band --packet-size 400 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\ns3_link_probe_out_of_band\upload_plan.csv
```

### 验证结果

| Case | Requested SC | Manual Apply | Reject | PHY Fail | RLC TX | RLC RX | CAM callback | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| success | 0 and 1 | 6 | 0 | 0 | 6 | 6 | 2 | 非冲突子信道全部成功收发 |
| edge_success | 9 | 3 | 0 | 0 | 3 | 3 | 1 | 最高合法子信道 9 可用 |
| conflict | 0 and 0 | 6 | 0 | 6 PSCCH | 6 | 3 | 1 | 同时同子信道冲突导致部分失败 |
| out_of_band | 10 | 0 | 1 | 0 | 0 | 0 | 0 | 超出 0-9 范围被拒绝且不偷跑 |

关键 trace：

- success：`requestedStart=0 physicalStart=0`、`requestedStart=1 physicalStart=1`，无 `PSCCH_DECODE_FAIL/PSSCH_DECODE_FAIL`，2 条 `cam_received`。
- edge_success：`requestedStart=9 physicalStart=9`，1 条 `cam_received`。
- conflict：两个请求均为 `requestedStart=0 physicalStart=0`，NS3 输出 `PSCCH_DECODE_FAIL reason=decoded_overlap`，仅 request 1 完整到达应用层。
- out_of_band：`MANUAL_CMD_REJECT reason=out_of_band src=1 dst=2 scStart=10 scSize=1 totalSubCh=10`，无 RLC TX/RX 和 application callback。

结论：当前 NS3 manual subchannel 行为已满足 SGCP 后续实验的基础要求：带宽范围内且无冲突的通信需求可成功收发；合法边界子信道可用；冲突子信道会通过 PHY decode failure 表现为丢包；超出带宽范围的请求不会被错误映射或随机发送。

## 2026-07-16 - SGCP potential_game NS3 replay after subchannel fix

### 代码修正

- `opencda.tools.offline_ns3_replay` 不再硬编码 `NaiveRA`，改为使用 `build_resource_allocator()` 和 `data_protocol.yaml` / `--resource-allocation` 指定的算法，默认 `potential_game`。
- SGCP replay 默认只发送已分配 `sc_start/sc_num` 的 request；未调度 member-to-head demand 计入 `skipped_unscheduled`，不再交给 NS3 默认调度路径。
- `upload_plan.csv` 新增 `sc_start`、`sc_num` 字段，便于追踪 PPS 输出到 NS3 physical resource 的映射。

### Dry-run

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --dry-run --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_3f_upload_plan.csv
```

结果：3 帧均为 `mode=sgcp_potential_game`，每帧 10 条 scheduled request、4 条 skipped unscheduled demand。NaiveRA 对照仍为每帧 14/14 scheduled。

### 11-frame NS3 replay

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 90s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=2.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_fixed\eval --max-frames 11
```

结果：

| Metric | Value |
| --- | ---: |
| Frames | 11 |
| Scheduled requests | 110 |
| Skipped unscheduled demand | 44 |
| Planned bytes | 1,100,000 |
| CAM received | 110 |
| CAM delivery ratio | 1.000000 |
| Avg. delay | 23.909 ms |
| P95 delay | 24.000 ms |
| PHY decode failures | 0 |
| RLC TX events | 2,970 |
| RLC RX events | 2,970 |
| Request any RLC RX ratio | 1.000000 |

Trace 计数：`MANUAL_RESOURCE_APPLY=2970`、`MANUAL_CMD_REJECT=0`、`PSCCH_DECODE_FAIL=0`、`PSSCH_DECODE_FAIL=0`。Upload plan 中 `sc_start=0..9` 各出现 11 次。

结论：修复后的 NS3 + SGCP potential_game replay 已经验证，在当前 11 帧 dump 中，PPS 已调度且无冲突的通信需求均能完整到达 application callback。旧 154-request 结果应标注为 legacy all-member diagnostic，因为它包含未调度需求并可能绕过 PPS。

## 2026-07-16 - NS3 request-level completion and exposed subchannel bounds

### 代码修正

- `opencda.tools.offline_ns3_replay` 改为全局递增 `pkt_id`，避免每帧从 1 重新编号导致 RLC 延迟事件与同名 request 跨帧错配。
- `opencda.tools.ns3_log_eval` / `lgcp_ns3_log_eval` 新增 request-level RLC completion 口径：`rlc_complete_requests`、`rlc_partial_requests`、`rlc_no_rx_requests`，并在 RLC 事件中优先使用全局唯一 `request_id` 对齐 upload plan。
- NS3 bridge 在 `ProcessData_TransferRequests` 前置校验 `sc_start/sc_num`，超出 OpenCDA 暴露的 `targetSubchannels` 范围时直接输出 `MANUAL_CMD_REJECT reason=bridge_out_of_band` 并跳过 CAM/RLC 创建。
- NS3 manual scheduler 对进入 MAC 的确定越界命令执行 drop+pop，防止无效队头阻塞后续合法请求；`no_exact_resource` 仍保留为等待后续资源的语义。

### 10-subchannel normal replay

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 120s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --drain-seconds 2.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target10_globalpkt\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target10_globalpkt\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target10_globalpkt\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target10_globalpkt\eval --max-frames 11
```

结果：110/110 application callback，RLC complete 110/110，PHY decode failures 0，`MANUAL_CMD_REJECT=0`。`sc_start=0..9` 各 11 条均成功。

### 5-subchannel exposed-bandwidth replay

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 120s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=5'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --drain-seconds 2.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\eval --max-frames 11
```

结果：110 planned requests 中，`sc_start=0..4` 共 55 条全部 complete，`sc_start=5..9` 共 55 条全部 no_tx；application callback 55/110，RLC complete 55/110，partial 0，PHY decode failures 0，`MANUAL_CMD_REJECT=55`，`MANUAL_RESOURCE_APPLY=1485`。

结论：当前 NS3 与 OpenCDA 的手动子信道接口已区分三层指标：application callback 表示应用层完整可见交付；RLC complete 表示 request 的 RLC TX/RX segment 数闭合；PHY decode diagnostics 用于解释冲突/信道失败。正常带宽内、无冲突的 SGCP PPS 请求可完整收发；超出 OpenCDA 暴露子信道范围的请求在 bridge 层被拒绝，不进入 CAM/RLC，也不会污染后续合法请求。

## 2026-07-16 - NS3 link-quality aware selective-sharing baseline

### 代码修正

- `opencda.tools.offline_inference` 新增 `--ns3-link-quality-csv`。
- 当 `--selective-sharing-baseline communication_aware` 同时传入 `rlc_by_request.csv` 时，成员选择分数从旧的 `density_sum / (1 + distance / 100)` 扩展为 `density_sum * rlc_complete_ratio / (1 + distance / 100)`。
- `rlc_complete_ratio` 优先使用同 timestamp 的 `(source_node, target_node)`，缺失时退回该 pair 的全局平均；未传入 CSV 时保持旧距离 proxy 行为。

### 11-frame 对照命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline communication_aware --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --max-frames 11

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline communication_aware --sgcp-inter-cluster-late-fusion --selective-member-budget 2 --selective-grid-budget 87 --ns3-link-quality-csv docs\doc_workspace\SGCP\artifacts\sgcp_ns3_pg_11f_target5_exposedfixed\eval\rlc_by_request.csv --max-frames 11
```

日志路径：

- `docs\doc_workspace\SGCP\artifacts\selective_commaware_ns3_quality_11f\distance_proxy_stdout.log`
- `docs\doc_workspace\SGCP\artifacts\selective_commaware_ns3_quality_11f\ns3_quality_stdout.log`

### 结果

| Baseline | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Bytes / Source | Avg. Source CAVs | Avg. Selected Grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Communication-aware distance proxy | 11 | 0.71 | 0.67 | 0.31 | 7,977,680 | 120,873.94 | 2.80 | 80.85 |
| Communication-aware NS3 RLC-complete aware | 11 | 0.68 | 0.63 | 0.27 | 7,796,560 | 118,129.70 | 2.80 | 80.85 |

结论：NS3 link-quality cost 已能进入 same-budget selective baseline。受限 5 子信道链路质量会使 baseline 避开不可达链路，通信量略降，但短 11 帧窗口 AP 也下降。论文叙事应强调这是网络可行性约束下的公平 baseline，而不是只按感知密度贪心的上界。

## 2026-07-16 - Cluster capacity statistics for `N_max`

### 代码修正

- `CoalitionGame` 新增 `capacity_stats.full_candidate_skips`，统计 coalition formation 中因 `c.size() >= N_max` 被跳过的候选 coalition 次数。
- `opencda.tools.offline_replay` summary 新增容量统计：`avg_full_clusters`、`max_full_clusters`、`full_candidate_skips_total`、`avg_full_candidate_skips`、`avg_singleton_cluster_ratio`、`avg_small_cluster_ratio`。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --n-max <2|3|4|5|6> --max-frames 0 --summary-only
```

日志路径：`docs\doc_workspace\SGCP\artifacts\capacity_stats_nmax\nmax_<N>.log`

### 结果

| `N_max` | Avg. Clusters | Avg. Size | Avg. Full Clusters | Max Full Clusters | Full Candidate Skips | Avg. Skips / Frame | Avg. Singleton Ratio | Avg. Small-Cluster Ratio | Reconfig. | Head Changes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10.29 | 1.95 | 9.71 | 10 | 12534 | 305.71 | 0.053 | 1.000 | 16 | 59 |
| 3 | 7.59 | 2.65 | 6.00 | 6 | 7894 | 192.54 | 0.146 | 0.206 | 9 | 62 |
| 4 | 6.00 | 3.33 | 3.12 | 4 | 4065 | 99.15 | 0.000 | 0.187 | 11 | 76 |
| 5 | 6.00 | 3.33 | 1.00 | 1 | 1142 | 27.85 | 0.000 | 0.317 | 8 | 15 |
| 6 | 6.00 | 3.33 | 0.00 | 0 | 0 | 0.00 | 0.000 | 0.317 | 8 | 15 |

结论：`N_max` 是实际生效的硬约束。`N_max=2/3` 时容量压力很强，导致大量候选加入被跳过，并产生更高 singleton/small-cluster 比例；默认 `N_max=4` 没有 singleton，但平均每帧仍有约 3.12 个满簇和 99.15 次满簇候选跳过，说明机制确实在处理“周围 cluster 已满”情况；`N_max=6` 下当前 20-CAV dump 不再受容量约束。论文中可据此说明：车辆不会被丢弃，而是保留在当前 coalition 或以小簇形式通过 inter-cluster late fusion 补偿。

## 2026-07-16 - `f(rho)` density calibration

### 代码与文档

- 新增 `opencda.tools.sgcp_density_calibration`，从 OPV2V/CARLA dump 重建与 SGCP replay 相同的 `OfflineLidarGrid`，输出全局、逐帧、阈值和曲线级 density 统计。
- 新增 `f_rho_calibration.md`，记录 `f(rho)=sigmoid(rho-rho_th)` 的标定协议、当前 41 帧结果、论文可写口径和仍需泛化的边界。

### 命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\sgcp_density_calibration.py
conda run -n opencda python -m opencda.tools.sgcp_density_calibration --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --thresholds "0.5,1.0,2.0,3.0,4.0" --output-dir docs\doc_workspace\SGCP\artifacts\density_calibration_41f
```

日志 / artifact 路径：

- `docs\doc_workspace\SGCP\artifacts\density_calibration_41f\global_density_summary.csv`
- `docs\doc_workspace\SGCP\artifacts\density_calibration_41f\frame_density_summary.csv`
- `docs\doc_workspace\SGCP\artifacts\density_calibration_41f\threshold_summary.csv`
- `docs\doc_workspace\SGCP\artifacts\density_calibration_41f\f_rho_curve.csv`
- `docs\doc_workspace\SGCP\artifacts\density_calibration_41f\run_notes.md`

### 结果

全局 density 分布：41 帧、20 CAV，共 788,020 个 CAV-grid 样本。非零 grid 47,119 个，占 0.059794；全量 density mean=0.050816，p99=0.830000，max=34.410000；非零 density mean=0.849855，p90=1.400000，p95=3.600000，p99=13.255600。

| `rho_th` | High-Density Grids | Ratio / All Grids | Ratio / Nonzero Grids | Mean `f(rho)` |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 11,232 | 0.014253 | 0.238375 | 0.383800 |
| 1.0 | 6,481 | 0.008224 | 0.137545 | 0.275282 |
| 2.0 | 3,383 | 0.004293 | 0.071797 | 0.124640 |
| 3.0 | 2,587 | 0.003283 | 0.054904 | 0.051639 |
| 4.0 | 2,192 | 0.002782 | 0.046521 | 0.021290 |

结论：默认 `rho_th=2.0` 位于当前非零网格 density 的 p90 和 p95 之间，筛出约 7.18% 非零网格作为 high-density candidates。结合此前 `rho_th` AP/payload sweep，可将其解释为当前 detector / LiDAR / 10 m grid 设置下的经验折中点，而不是跨场景通用常数。后续论文级版本仍应补不同场景或 detector metadata 下的泛化。

## 2026-07-16 - SGCP control overhead accounting

### 代码与文档

- `opencda.tools.offline_replay` 新增 `estimate_control_overhead()`，在 summary 中输出 SGCP 控制面开销。
- 新增 `control_overhead.md`，记录 beacon、density metadata、cluster membership 和 PPS schedule 的估算假设。

### 命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_replay.py
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

### 结果

输出：

```text
summary control_overhead total_bytes=187112 avg_bytes_per_frame=4563.71 beacon_bytes=52480 density_metadata_bytes=40184 cluster_control_bytes=3608 pps_schedule_bytes=90840 avg_high_density_grids=82.51 avg_scheduled_links=10.00 avg_selected_grids=523.90
```

| Component | Total Bytes | Avg. Bytes / Frame |
| --- | ---: | ---: |
| Beacon | 52,480 | 1,280.00 |
| Density metadata | 40,184 | 980.10 |
| Cluster control | 3,608 | 88.00 |
| PPS schedule | 90,840 | 2,215.61 |
| Total control | 187,112 | 4,563.71 |

结论：当前 41 帧 SGCP inter-cluster late-fusion 点云 payload 为 26,916,208 bytes，控制面估算约为 0.70%。论文中应区分 perception payload 和 control metadata；控制信令不是主导通信开销，但需要作为轻量 overhead 单独报告。

## 2026-07-16 - PPS potential game condition review

### 目的

推进 P3 “重新检查 potential game exact potential 的成立条件”，避免论文中过强声称与当前代码实现不一致。

### 代码复核

复核文件：

```text
opencda/core/clustering/algorithms/resource_allocation/potential_game.py
opencda/core/clustering/utils/common.py
```

关键观察：

- 当前 `PotentialGame` 使用 sequential best-response scheduling：每个 cluster head 基于 `grid_score()` 选择 member/grid/RB。
- `grid_score()` 是 grid-level utility 的边际提升，但代码没有显式全局势函数 `Phi`。
- RB 占用、channel capacity 和 SINR 目前是 feasibility gate，不是同时进入局部 utility 与全局 potential 的 penalty。
- 被注释掉的 replacement 逻辑说明当前主要是追加 schedule；收敛更多来自有限 action / 有限追加，而不是完整 best-response dynamics 的 exact-potential 证明。
- `get_participating_clusters()` 当前遇到第一个 cluster 后 `break`，不等于完整 inter-cluster late utility 聚合。

### 文档

新增：

```text
docs/doc_workspace/SGCP/potential_game_conditions.md
```

结论：当前代码可以支撑 “potential-guided constrained best-response scheduling” 和 “finite empirical convergence”。若论文继续使用 exact potential game，需要明确限定条件：固定 cluster membership、固定候选 grid、additive grid utility、资源/SINR 作为 hard action constraints，且局部 utility 定义为全局 grid utility 的边际变化。更强的 exact-potential 声称需要补显式 `Phi` 计算、action replacement 和 `Delta Phi >= 0` 日志。

## 2026-07-16 - PPS convergence diagnostics

### 代码修正

- `PotentialGame` 新增 `convergence_stats`，记录 `iterations`、`cluster_updates`、`scheduled_links`、`selected_grids`、`used_rbs`、`reused_rbs`、`max_rb_occupancy` 和 `converged`。
- `opencda.tools.offline_replay` summary 新增 `summary pps_convergence ...`。

### 命令

```powershell
conda run -n opencda python -m py_compile opencda\core\clustering\algorithms\resource_allocation\potential_game.py opencda\tools\offline_replay.py
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

### 结果

```text
summary pps_convergence avg_iterations=3.00 max_iterations=3 converged_frames=41 avg_cluster_updates=10.00 avg_scheduled_links=10.00 avg_selected_grids=523.90 avg_used_rbs=10.00 avg_reused_rbs=0.00 max_rb_occupancy=1
```

结论：当前默认 20-CAV / 10-subchannel dump 中，PPS 41/41 帧都在 `max_iter=20` 前收敛，平均 3 轮停止。每帧平均 10 条 scheduled links、使用 10 个 RB，未触发 RB 复用，最大 RB occupancy=1。这支持 “finite empirical convergence” 与 “10-subchannel NS3 replay 无冲突” 的实验叙事，但仍不是 exact potential 的数学证明。

## 2026-07-16 - Paper revision plan for topology-trigger wording

### 目的

推进 P4 “修正 topology change 才触发 与 每个周期重复 的表述矛盾”，并把已经完成的机制/实验工作转成论文正文和 rebuttal 可直接使用的修订计划。

### 复核位置

论文源文件：`C:\Workspace\icdcs-paper\SGCP\main.tex`

冲突位置：

- 系统周期描述附近：cluster formation 被描述为 topology change 才触发。
- Formation Algorithm 段：又写 procedure repeats every cycle `T_c`。
- Conclusion 段：对两个 game-theoretic algorithms 的收敛保证表述偏强。

### 输出

新增：

```text
docs/doc_workspace/SGCP/paper_revision_plan.md
```

核心统一口径：

> 每个 100 ms cooperation cycle 更新 beacon、density metadata、PPS scheduling 和 perception fusion；cluster membership / leader election 仅在 topology/stability trigger 或 periodic guard 触发时更新。

文档同时给出 `main.tex` 替换建议、rebuttal 答法、实时性补充口径、`f(rho)` 标定口径、baseline fairness 风险边界和 game-theoretic convergence 的保守改写建议。

## 2026-07-16 - Related work and novelty revision plan

### 目的

推进 P4：

- 重写 related work 的 decentralized CP 和 coalition game 对比。
- 增强 novelty：突出感知效用驱动、稳定性约束、分层 fusion、分布式资源调度的组合贡献。

### 复核材料

- `C:\Workspace\icdcs-paper\SGCP\main.tex` 的 Related Work 与 Introduction contribution。
- `C:\Workspace\icdcs-paper\SGCP\SGCP-review.txt` 中 Reviewer 2/3/4 对 Smartform 相似性、baseline fairness 和 decentralized CP baseline 的意见。

### 输出

新增：

```text
docs/doc_workspace/SGCP/related_work_novelty_revision.md
```

关键口径：

- 不把 novelty 写成 “使用 coalition game” 本身。
- 强调 SGCP 面向 RSU-free dense urban CP 的组合贡献：grid-level perception utility、LiDAR density calibration、motion/stability-aware coalition control、硬 `N_max` 容量约束、PPS 子信道可行调度和 inter-cluster late fusion。
- Related work 增补 V2V-only / decentralized CP、learned communication selection、RSU-centric scheduling 和其他领域 coalition formation 的边界。
- Rebuttal 中承认 coalition formation 在其他领域已有应用，但说明 SGCP 的 utility、deadline、payload、subchannel 和 hierarchical fusion 约束不同。

## 2026-07-16 - Parameter and utility calibration revision

### 目的

推进 P4：

- 补充 `f(rho)` 标定过程和曲线。
- 补充 `T_min^stab`、`N_max`、`rho_th` 参数选择依据。

### 复核材料

- `f_rho_calibration.md`
- `results.md` 中 `T_min^stab`、`N_max`、`rho_th` sweep。
- Reviewer 3/4 对 density calibration、500 ms stability window 和 detector/sensor 泛化的意见。

### 输出

新增：

```text
docs/doc_workspace/SGCP/parameter_calibration_revision.md
```

核心口径：

- `rho_th=2.0`：位于当前非零 grid density p90/p95 之间，是 AP/payload 折中点，不是通用常数。
- `N_max=4`：容量/fragmentation 折中；默认下无 singleton，但仍有 99.15 次满簇候选跳过/frame，说明容量约束实际生效。
- `T_min^stab=500 ms`：当前 41 帧 sweep 对 100-1000 ms 不敏感，不能写为最优，只能写为覆盖五个 10 Hz 感知周期的 conservative hysteresis default；需要更动态场景支撑强结论。

## 2026-07-16 - FullPerception baseline revision

### 目的

推进 P4 “补充 FullPerception baseline 的实现细节和公平性讨论”，把审稿意见中对 FullPerception 设置不清的质疑转成论文正文和 rebuttal 可用文本。

### 复核材料

- `baseline_fairness.md`
- `results.md` 中 full early/late reference、SGCP 主结果和 same-budget selective-sharing baseline。
- 当前 `v2xp_cluster_carla` dump：`D:\Data\Carla\2026_07_15_01_26_56`，20 CAV，41 帧，无 RSU 目录。

### 输出

新增：

```text
docs/doc_workspace/SGCP/fullperception_baseline_revision.md
```

核心口径：

- `FullPerception-RSU` 与 full 20-CAV early/late fusion 只能作为 centralized/infrastructure-assisted upper reference，不作为同通信预算公平主对比。
- `FullPerception-Decentralized` 应具体化为 same-budget CAV-only selective sharing，匹配数据、backbone、cluster-head late-fusion path 和 grid budget。
- 当前 strong V2V baseline 包括 nearest、density、communication-aware selective sharing；其中 communication-aware 41 帧 AP@0.3/0.5/0.7 = 0.78/0.75/0.40，payload 高于 SGCP。
- 论文叙事应避免宣称 SGCP 在该短 dump 上 AP 全面领先，转而强调 SGCP 的 cluster stability、PPS channel feasibility、NS3 request-level 可验证传输和可解释控制开销。

## 2026-07-16 - Runtime feasibility breakdown

### 目的

推进 P4 “补充实时性实验，包括毫秒级耗时分解”，区分离线 replay artifact、SGCP 控制面算法耗时和仍未测量的端到端感知耗时。

### 代码与环境

- Conda 环境：`opencda`
- 数据集：`D:\Data\Carla\2026_07_15_01_26_56`
- CAV 数量：20
- 帧数：41
- 资源分配：`potential_game`

### 代码修正

- `opencda.tools.offline_replay` 新增 `runtime_breakdown_ms`：
  - `load_frame`
  - `world_build`
  - `clustering`
  - `post_cluster_state`
  - `resource_allocation`
  - `control_accounting`
  - `algorithm_total`
  - `offline_total`

### 命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_replay.py
conda run -n opencda python -m opencda.tools.offline_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --resource-allocation potential_game --max-frames 0 --summary-only
```

### 日志路径

```text
docs\doc_workspace\SGCP\artifacts\runtime_breakdown_41f\offline_replay_runtime.log
```

### 结果摘要

- SGCP algorithm total：avg 105.24 ms，max 127.58 ms。
- Coalition formation：avg 64.39 ms，max 82.32 ms。
- PPS scheduling：avg 40.58 ms，max 53.05 ms。
- Post-cluster state update：avg 0.24 ms，max 0.44 ms。
- Control overhead accounting：avg 0.03 ms，max 0.05 ms。
- Dump frame loading：avg 448.40 ms，max 513.31 ms。
- Offline world build：avg 151.33 ms，max 199.34 ms。
- Offline total：avg 704.97 ms，max 789.68 ms。

### 观察

- 离线文件读取和 world build 合计约 599.73 ms/frame，是 replay pipeline artifact，不应写成在线 CARLA 周期耗时。
- SGCP 控制面 Python 原型平均 105.24 ms，接近但略高于 100 ms，因此论文只能写 near-real-time feasibility，不能写完整端到端 100 ms 保证。
- PPS 本身平均 40.58 ms 且 41/41 帧 3 轮收敛；主要优化空间在 coalition formation。
- 在线 topology-trigger gate 可用于摊销 coalition formation 成本，因为 cluster membership 不必每帧重构。

## 2026-07-16 - Reproducibility manifest

### 目的

推进 P0 “确认论文现有结果对应的代码版本、配置、随机种子和日志路径”，把当前可复现实验和论文旧主表分开管理。

### 复核材料

- `C:\Workspace\icdcs-paper\SGCP\main.tex`
- `docs/doc_workspace/SGCP/results.md`
- `docs/doc_workspace/SGCP/artifacts/`
- 当前 OpenCDA git 状态

### 输出

新增：

```text
docs/doc_workspace/SGCP/reproducibility_manifest.md
```

### 结论

- 当前 OpenCDA 复现实验 commit：`2cd026ec96691d15e4d764f4bd78af51a2404859`。
- 论文旧主表 `NC/RS/MUG/FullPerception/Ours` 尚未找到原始日志、随机种子、代码提交和完整配置。
- 当前可复现实验固定为 `D:\Data\Carla\2026_07_15_01_26_56`，20 CAV，41 帧。
- Manifest 已记录 full 20-CAV early upper reference、SGCP constrained + inter-cluster late fusion、scheduler ablation、same-budget selective-sharing baseline、NS3 request-level replay、runtime breakdown 和 control overhead。
- 后续论文修订应优先使用 manifest 中的已复现结果；如果要继续使用旧主表，必须先找回旧日志。

## 2026-07-16 - Online topology-trigger gate regression

### 目的

推进 P3 剩余项：在真实 CARLA 在线仿真中打开 `enable_topology_trigger_gate`，回归 cluster trigger 日志、reconfiguration 次数和感知结果。

### 代码与配置

- 新增 `opencda/scenario_testing/config_yaml/networking_clustering_topology_gate.yaml`。
- `v2xp_cluster_carla.py` 新增环境变量：
  - `OPENCDA_CLUSTERING_CONFIG`
  - `OPENCDA_ONLINE_TICKS`

### 命令

```powershell
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe"
$env:OPENCDA_CLUSTERING_CONFIG = "opencda/scenario_testing/config_yaml/networking_clustering_topology_gate.yaml"
$env:OPENCDA_ONLINE_TICKS = "35"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug
Remove-Item Env:\OPENCDA_CLUSTERING_CONFIG
Remove-Item Env:\OPENCDA_ONLINE_TICKS
```

### 日志路径

```text
docs\doc_workspace\SGCP\artifacts\online_topology_gate\online_gate_stdout.log
opencda\log\opencda_20260716_090705.log
evaluation_outputs\v2xp_cluster_carla_2026_07_16_09_07_11\log.txt
```

### 结果

- Exit code：0
- CARLA：回归后已关闭
- NS3：未启动
- Online ticks：35
- CP counter：8
- Fusion method：early
- AP@0.3 / AP@0.5 / AP@0.7：0.84 / 0.82 / 0.69
- Cluster trigger：
  - `recluster reason=initial`：1
  - `recluster reason=neighbor_set_change`：1
  - `recluster reason=head_member_unreachable`：3
  - `skip reason=no_topology_change`：0

### 观察

- `enable_topology_trigger_gate` 已通过 config path 生效，并输出 `CLUSTER_TRIGGER` / `CLUSTER_SYNC`。
- 本轮没有出现 skip，因为默认 35 m 通信范围下持续触发 `head_member_unreachable` hard condition。
- 该在线回归证明 gate 接入和日志可用，但不能证明 reduced reconfiguration；若论文需要展示 skip 收益，应补更静态或更大通信范围的在线回归。

## 2026-07-16 - Random grid selection mechanism probe

### 目的

推进 P-1 主表修复：在保持同一 SGCP cluster、PPS scheduled links 和每条 link 的 grid 数量不变时，将具体 grid selection 替换为确定性随机候选，判断当前 utility selection 是否真的优于随机。

### 代码

`opencda.tools.offline_inference` 新增：

```text
--sgcp-grid-selection-mode {utility,random}
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode random --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\random_grid_41f_trace.csv *> docs\doc_workspace\SGCP\artifacts\mechanism_probe\random_grid_41f_stdout.log
```

### 结果

- Frames：41
- Receiver trace rows：246
- `missing_channel_rows`：0
- AP@0.3 / AP@0.5 / AP@0.7：0.78 / 0.75 / 0.36
- Total payload：27,908,560 bytes
- Avg. payload / receiver：113,449.43 bytes
- Avg. uploaded sources / receiver：1.67
- Avg. uploaded points / receiver：7,090.59
- Avg. selected grids / receiver：87.32

### 观察

- Random grid selection 在相同 scheduled links 和相同 grid count 下略高于当前 SGCP utility selection（`0.77/0.73/0.35`）。
- 该结果排除了“随机低效选择一定明显更差”的假设，说明当前 grid utility 与 OpenCOOD 检测 AP 的目标不够一致。
- 下一步应优先做 fixed-cluster / fixed-link 的 grid scoring 改造与消融，目标是在接近当前 payload 的前提下稳定超过 random-grid，并缩小到 full-cluster upload（`0.82/0.79/0.42`）的差距。

## 2026-07-16 - Grid utility repair probes

### 目的

继续推进 P-1 主表修复：在协议链路已确认有效的基础上，尝试改造 grid utility / selection，判断是否能在相同 PPS scheduled links 和相近 payload 下超过 random-grid。

### 代码

新增离线实验开关：

```text
--sgcp-grid-score-mode {utility,raw_density,density_distance}
--sgcp-grid-selection-mode {utility,random,spatial_diverse}
```

其中：

- `raw_density`：用 sender grid 原始 density 替代饱和 utility。
- `density_distance`：用 sender grid density 除以 receiver-to-grid distance cost。
- `spatial_diverse`：保留 PPS scheduled links 和每条 link 的 grid count，在候选 grid 内用 density-aware spatial cover 替换原始选格。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-score-mode raw_density --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\raw_density_grid_41f_trace.csv *> docs\doc_workspace\SGCP\artifacts\mechanism_probe\raw_density_grid_41f_stdout.log
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-score-mode density_distance --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\density_distance_grid_41f_trace.csv *> docs\doc_workspace\SGCP\artifacts\mechanism_probe\density_distance_grid_41f_stdout.log
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_trace.csv *> docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_stdout.log
```

### 结果

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Total Payload | Avg. Payload / Receiver | Avg. Uploaded Points | Avg. Selected Grids | Missing Channel Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP utility | 0.77 | 0.73 | 0.35 | 26,916,208 | 109,415.48 | 6,838.47 | 87.32 | 0 |
| Random grid | 0.78 | 0.75 | 0.36 | 27,908,560 | 113,449.43 | 7,090.59 | 87.32 | 0 |
| Raw-density score | 0.74 | 0.70 | 0.37 | 29,290,768 | 119,068.16 | 7,441.76 | 88.55 | 0 |
| Density-distance score | 0.74 | 0.71 | 0.37 | 29,219,088 | 118,776.78 | 7,423.55 | 88.00 | 0 |
| Spatial-diverse grid | 0.79 | 0.75 | 0.37 | 28,743,280 | 116,842.60 | 7,302.66 | 87.32 | 0 |

### 观察

- `raw_density` 和 `density_distance` 虽然提升 AP@0.7，但损失 AP@0.3/0.5 且 payload 增加，说明单纯追高密度并不稳健。
- `spatial_diverse` 在相同 scheduled links 和 grid count 下超过 random-grid，同时 payload 约为 full-cluster upload 的 64.1%，是当前最有希望的 grid utility 改造方向。
- 后续应把 `spatial_diverse` 从 probe 整理为 coverage-aware SGCP grid utility，并补 fixed-cluster/fixed-link、payload sweep 和 NS3-aware 交付裁剪。

## 2026-07-16 - Spatial-diverse channel sweep

### 目的

继续推进 P-1 主表修复：评估 coverage-aware `spatial_diverse` grid selection 在不同子信道预算下的 AP/payload tradeoff，选择低通信主点和高预算敏感性设置。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --num-channels 5 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch5_41f_trace.csv *> docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch5_41f_stdout.log
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --num-channels 20 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch20_41f_trace.csv *> docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch20_41f_stdout.log
```

10 子信道结果来自上一轮：

```text
docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_trace.csv
```

### 结果

| Num. Channels | AP@0.3 | AP@0.5 | AP@0.7 | Total Payload | Avg. Payload / Receiver | Avg. Uploaded Sources | Avg. Uploaded Points | Avg. Selected Grids | Payload vs Full-Cluster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.56 | 0.53 | 0.27 | 14,815,408 | 60,225.24 | 0.83 | 3,764.08 | 45.58 | 33.0% |
| 10 | 0.79 | 0.75 | 0.37 | 28,743,280 | 116,842.60 | 1.67 | 7,302.66 | 87.32 | 64.1% |
| 20 | 0.80 | 0.76 | 0.41 | 37,912,544 | 154,116.03 | 2.33 | 9,632.25 | 117.18 | 84.5% |

### 观察

- 5 子信道与原始 utility 的低资源结果基本一致，说明强瓶颈下 AP 主要由 admitted links 决定。
- 10 子信道是低通信主表候选：payload 约为 full-cluster 的 64.1%，AP 为 `0.79/0.75/0.37`。
- 20 子信道是高预算敏感性候选：AP@0.7 = 0.41，接近 full-cluster 0.42，同时 payload 比 full-cluster 低约 15.5%。
- 后续主表可以考虑同时报告 low-budget SGCP 和 high-budget SGCP，或者将 10 子信道作为主设置、20 子信道放入 network-resource sensitivity。

## 2026-07-16 - Spatial-diverse NS3 high-budget replay

### 目的

验证 20 子信道 `spatial_diverse` high-budget 主表候选的 SGCP/PPS transfer requests 是否能在 NS3 暴露子信道窗口内完整交付。

### 代码更新

`opencda.tools.offline_ns3_replay` 新增：

```text
--num-channels
--bandwidth-mhz
--sgcp-grid-score-mode {utility,raw_density,density_distance}
--sgcp-grid-selection-mode {utility,random,spatial_diverse}
```

说明：`spatial_diverse` 影响每条 scheduled link 内的 grid 选择，不改变 NS3 transfer request 的 source/target/subchannel；这里主要用于让 replay artifact 与离线感知主表候选同名、同子信道窗口。

### Dry-run

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --num-channels 10 --sgcp-grid-selection-mode spatial_diverse --dry-run --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_11f_dryrun\upload_plan.csv
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --num-channels 20 --sgcp-grid-selection-mode spatial_diverse --dry-run --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f_dryrun\upload_plan.csv
```

Dry-run 结果：

- 10 子信道：110 rows，`sc_start=0..9`，每帧 10 scheduled requests，4 skipped unscheduled。
- 20 子信道：154 rows，`sc_start=0..13`，每帧 14 scheduled requests，0 skipped unscheduled。

### NS3 实跑

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 90s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=2.5 --enableTimeSync=true --carlaHost=auto --targetSubchannels=20'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --num-channels 20 --sgcp-grid-selection-mode spatial_diverse --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch20_11f\eval --max-frames 11
```

### 结果

- Frames：11
- Scheduled requests：154
- Skipped unscheduled：0
- Planned bytes：1,540,000
- CAM received：154 / 154
- CAM delivery ratio：1.000000
- Avg / P95 delay：23.909 ms / 24.000 ms
- RLC TX / RX events：4,158 / 4,158
- RLC complete requests：154 / 154
- `MANUAL_RESOURCE_APPLY`：4,158
- `MANUAL_CMD_REJECT`：0
- `PSCCH_DECODE_FAIL` / `PSSCH_DECODE_FAIL`：0 / 0

### 观察

- 20 子信道 high-budget `spatial_diverse` 候选增加到每帧 14 条 member-to-head request，NS3 仍全部交付。
- 该结果支撑论文中将 20 子信道作为 high-budget sensitivity：更高 AP 不是通过绕过 PPS 或 NS3 默认调度得到，而是在 OpenCDA 指定子信道上完整交付。
- 后续可补 10 子信道 `spatial_diverse` NS3 replay，使低通信主点也有同口径 delivery 证据。

## 2026-07-16 - Spatial-diverse NS3 low-budget replay

### 目的

补齐 10 子信道 `spatial_diverse` low-budget 主表候选的 NS3 request-level 交付证据。

### 命令

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 90s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=2.5 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --num-channels 10 --sgcp-grid-selection-mode spatial_diverse --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_11f\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_11f\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_11f\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_11f\eval --max-frames 11
```

### 结果

- Frames：11
- Scheduled requests：110
- Skipped unscheduled：44
- Planned bytes：1,100,000
- CAM received：110 / 110
- CAM delivery ratio：1.000000
- Avg / P95 delay：23.909 ms / 24.000 ms
- RLC TX / RX events：2,970 / 2,970
- RLC complete requests：110 / 110
- `MANUAL_RESOURCE_APPLY`：2,970
- `MANUAL_CMD_REJECT`：0
- `PSCCH_DECODE_FAIL` / `PSSCH_DECODE_FAIL`：0 / 0

### 观察

- 10 子信道 low-budget `spatial_diverse` 候选的 110 条 PPS scheduled requests 全部交付。
- 每帧 4 条未调度需求在 OpenCDA replay 侧跳过，没有进入 NS3 默认调度路径。
- 至此，`spatial_diverse` 10 子信道低通信主点和 20 子信道高预算敏感性点均有 NS3 request-level 交付证据。

## 2026-07-16 - Baseline and threshold corrections

### 目的

回应主表设计问题：补全 FullPerception baseline；避免使用 payload 过低的 Random/MWS scheduler 作为通信量减少证据；补充 SGCP 内部点云阈值参数实验。

### FullPerception payload 统计

命令：

```powershell
conda run -n opencda python -c "from opencda.core.common.offline_dataset import OPV2VFrameDataset; ds=OPV2VFrameDataset(r'D:\\Data\\Carla'); scenario='2026_07_15_01_26_56'; ego='1'; frames=[ds.load_frame(scenario,ts,ego_cav_id=ego) for ts in ds.scenarios[scenario]['timestamps']]; total_all=sum(c['lidar_np'].nbytes for f in frames for c in f.values()); total_non=sum(c['lidar_np'].nbytes for f in frames for cid,c in f.items() if str(cid)!=ego); print(total_all,total_non)"
```

结果：

- Full 20-CAV early AP：0.85 / 0.83 / 0.48
- Full 20-CAV all point bytes：64,070,912
- Full 20-CAV non-ego upload bytes：60,838,528
- 结论：当前 dump 无 RSU，因此 FullPerception-RSU 不应填成实测；full 20-CAV early 可作为 centralized/virtual FullPerception upper reference。

### High-budget selective baseline

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline density --sgcp-inter-cluster-late-fusion --selective-member-budget 3 --selective-grid-budget 117 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\selective_high_budget_41f\density_m3_g117_trace.csv
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --selective-sharing-baseline communication_aware --sgcp-inter-cluster-late-fusion --selective-member-budget 3 --selective-grid-budget 117 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\selective_high_budget_41f\communication_aware_m3_g117_trace.csv
```

结果：

- Density high-budget：0.80 / 0.76 / 0.40，payload 37,710,864 bytes。
- Communication-aware high-budget：0.80 / 0.76 / 0.40，payload 37,710,864 bytes。
- 结论：高预算 selective baseline 已充分利用接近 SGCP 20ch 的通信量；SGCP spatial-diverse 20ch 为 0.80 / 0.76 / 0.41，payload 37,912,544 bytes，AP@0.7 略高。

### Spatial-diverse `rho_th` threshold sweep

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 1.0 --max-frames 0
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3.0 --max-frames 0
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 4.0 --max-frames 0
```

结果：

| `rho_th` | AP@0.3 | AP@0.5 | AP@0.7 | Total Payload | Avg. Selected Grids |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.76 | 0.72 | 0.34 | 26,296,464 | 80.75 |
| 2.0 | 0.79 | 0.75 | 0.37 | 28,743,280 | 87.32 |
| 3.0 | 0.79 | 0.76 | 0.38 | 29,405,296 | 89.72 |
| 4.0 | 0.79 | 0.76 | 0.38 | 29,837,744 | 90.62 |

### 观察

- Random/MWS scheduler payload 只有约 9.7/9.9 MB，确实没有充分利用通信资源；它们不适合作通信量减少主证据。
- 公平主表应使用 payload-matched selective baselines 和 SGCP 10/20ch，而不是低通信 Random/MWS。
- `rho_th` 是当前最清晰的点云阈值/通信量调节参数；`rho_th=3.0` 比默认 2.0 有更高 AP@0.5/AP@0.7，payload 只增加约 0.66 MB。

## 2026-07-16 - Main table candidate consolidation

### 目的

将已复现的 FullPerception、payload-matched selective baseline、SGCP coverage-aware 10/20ch、`rho_th` threshold sweep 和 NS3 delivery 证据收束为一份论文主表候选，避免继续在 `results.md` 的散表中手工拼接。

### 输出

新增：

```text
docs\doc_workspace\SGCP\main_table_candidate.md
```

### 换算口径

Mbps 按 41 帧、0.1 s 协作周期换算：

```text
payload_bytes * 8 / 4.1 s / 1e6
```

关键换算：

- FullPerception centralized：60,838,528 bytes，118.71 Mbps。
- Full-cluster reference：44,850,528 bytes，87.51 Mbps。
- Selective high-budget：37,710,864 bytes，73.58 Mbps。
- SGCP coverage-aware 10ch `rho_th=2`：28,743,280 bytes，56.08 Mbps。
- SGCP coverage-aware 10ch `rho_th=3`：29,405,296 bytes，57.38 Mbps。
- SGCP coverage-aware 20ch：37,912,544 bytes，73.98 Mbps。
- Random scheduler / MWS：18.98 / 19.34 Mbps，仅作消融。

### 结论

- 推荐主表包含：Head-only、FullPerception centralized upper reference、Full-cluster upper reference、Selective communication-aware low-budget、Selective density high-budget、SGCP original、SGCP coverage-aware 10ch、SGCP coverage-aware 10ch `rho_th=3`、SGCP coverage-aware 20ch。
- Random/MWS 不进入公平主表，只作为 w/o PPS 消融。
- 论文主叙事应强调：SGCP coverage-aware 20ch 在与 high-budget selective baseline 几乎相同 payload 下 AP@0.7 略高，且有 NS3 request-level 完整交付；SGCP coverage-aware 10ch `rho_th=3` 在 57.38 Mbps 下提供低通信折中，远低于 FullPerception centralized 118.71 Mbps。

## 2026-07-16 - Paper main table first-pass revision

### 目的

把已复现主表候选迁移到 `C:\Workspace\icdcs-paper\SGCP\main.tex`，移除旧论文中无法复现且 baseline 口径不清的主表、通信开销和 FullPerception 对比表述。

### 修改内容

- 将 `rho_th` 写为默认 2.0 与 tuned low-budget 3.0 的敏感性口径。
- 将 baseline 列表改为 NC、FullPerception-centralized upper reference、Full-cluster、capacity-matched Selective V2V、Random/MWS diagnostics 和 SGCP。
- 将旧 `tab:mAP` 替换为 AP + Mbps 统一主表，包含 FullPerception centralized、Full-cluster、Selective V2V、SGCP utility、SGCP coverage-aware 10ch 和 20ch。
- 删除旧 `comm_overhead.eps` 正文引用，避免图中旧 Mbps 与新主表冲突。
- 将实时性段落从“comfortably fits 100ms”改为 near-real-time feasibility，并解释 topology-triggered cluster update 降低 recurring control path 成本。

### 当前结论

论文正文已经不再声称 SGCP 全面超过 FullPerception，也不再用低 payload 的 Random/MWS 证明通信节省。下一步优先补 `rho_th=3` 的 NS3 request-level replay，随后把 coverage-aware grid selection 写入机制章节并准备 rebuttal 文本。

## 2026-07-16 - Spatial-diverse 10ch rho=3 NS3 replay

### 目的

补齐主表推荐行 `SGCP coverage-aware, 10ch, rho_th=3` 的 NS3 request-level delivery 证据，并确保 NS3 replay 与离线感知主表使用同一 `rho_th` 参数。

### 代码修复

- `opencda.tools.offline_ns3_replay` 新增 `--rho-th` 参数。
- `--rho-th` 传入 `OfflineCavWorld(..., density_threshold=rho_th)`，使 replay 的 SGCP grid candidate / resource allocation 与离线 AP 实验同参。

### 命令

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --num-channels 10 --sgcp-grid-selection-mode spatial_diverse --rho-th 3.0 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_rho3_11f\upload_plan.csv
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_rho3_11f\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_rho3_11f\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_ch10_rho3_11f\eval --max-frames 11
```

### 结果

- Frames：11。
- Scheduled requests：110。
- Skipped unscheduled：44。
- Planned bytes：1,100,000。
- Application callback：110/110，delivery ratio 1.000000。
- RLC TX/RX：2,970 / 2,970。
- RLC complete requests：110/110。
- PHY decode failures：0。
- Avg / P95 delay：23.909 / 24.000 ms。

### 结论

`rho_th=3.0` 的 10ch tuned low-budget 主表候选与 `rho_th=2.0` 一样，request-level NS3 交付完整。主表中该行的 NS3 delivery 可以从 Pending 改为 `110/110 complete`。

## 2026-07-16 - Paper mechanism second-pass revision

### 目的

把主表中实际有效的 coverage-aware / spatial-diverse grid selection 写入 `C:\Workspace\icdcs-paper\SGCP\main.tex` 机制章节，并降低 exact-potential/Nash guarantee 的理论风险。

### 修改内容

- 摘要中的调度描述从 non-cooperative potential game 改为 potential-guided constrained scheduling，突出 coverage-aware point cloud regions。
- System model 中明确每周期更新 beacon/density/PPS，cluster membership 仅在 topology/stability trigger 或 periodic guard 触发时更新。
- Coalition formation 段落明确“固定拓扑快照下收敛”，不再写成每个 `T_c` 无条件重构。
- Resource scheduling 小节和算法标题改为 potential-guided PPS。
- Potential 段落从无条件 exact potential/Nash convergence 改成固定候选集合和硬可行约束下的 grid-level potential guide，并强调 empirical convergence/runtime。
- PPS 算法增加 coverage-aware grid subset：每条 scheduled link 不上传全部 candidate grids，而是选择高密度且空间分散的 bounded grid subset。
- Conclusion 中将“Both algorithms are guaranteed...”弱化为 coalition fixed-snapshot convergence + PPS finite feasible action set empirical convergence。

### 结论

论文机制章节现在与代码和主表更一致：`spatial_diverse` 不再只是实验 hack，而是被表述为 density-aware spatial diversification 的 coverage-aware grid selection；同时避免把当前工程实现包装成无条件 exact potential game。

## 2026-07-16 - Paper f(rho) and rho_th calibration revision

### 目的

回应审稿人对 `f(rho)` 标定过短、`rho_th` 像拍脑袋参数、以及 density utility 泛化边界不清的质疑。

### 修改内容

- `C:\Workspace\icdcs-paper\SGCP\main.tex` 中将原本一句 empirical calibration 扩展为可复现协议：使用与 SGCP replay 相同的 10 m global grid 重建，统计每个 CAV/frame 的 points/m^2。
- 写入 41 帧 dump 的密度统计：788,020 个 CAV-grid samples，非空网格 5.98%，非空 density p90/p95 = 1.40 / 3.60 points/m^2。
- 写明默认 `rho_th=2.0` 位于 p90 和 p95 之间，选择 7.18% 非空网格作为 high-density candidates。
- 明确 `rho_th` 依赖 LiDAR resolution、grid size、point-cloud preprocessing 和 detector backbone，不能当作通用常数。
- 新增 coverage-aware SGCP 10ch `rho_th=1/2/3/4` sensitivity table：AP/Mbps 分别为 0.76/0.72/0.34/51.31、0.79/0.75/0.37/56.08、0.79/0.76/0.38/57.38、0.79/0.76/0.38/58.22。
- 统一 introduction/contribution/baseline 中旧的 potential-game wording 为 potential-guided PPS。

### 结论

`rho_th=3.0` 主表行现在有三重支撑：density calibration、AP/Mbps sensitivity、NS3 110/110 request-level delivery。正文也明确了默认 `rho_th=2.0` 是保守通信-精度折中，而非最优或通用阈值。

## 2026-07-16 - Rebuttal draft

### 目的

将当前已完成的实验、机制修订和论文正文修改整理成 reviewer-by-reviewer rebuttal 草稿，避免后续答复继续散落在多个 revision 文档中。

### 输出

新增：

```text
docs\doc_workspace\SGCP\rebuttal_draft.md
```

### 覆盖问题

- R2：coalition max baseline、满簇处理、merge/split、成员贡献重算、Smartform/coalition novelty、FullPerception/RSU fairness。
- R3：`f(rho)` calibration、`T_min^stab=500 ms`、公平 decentralized baseline、ablation 和参数实验。
- R4：100 ms feasibility、density utility 泛化、topology trigger、NS3 三层 delivery metrics。

### 结论

当前 rebuttal 草稿的主线是：承认旧稿中 baseline 和理论表述过强，说明修订后采用 centralized upper reference + fair V2V selective baselines + coverage-aware SGCP 主表；同时用 density calibration、rho sweep、NS3 request-level delivery 和 runtime breakdown 支撑可复现性与工程可行性。

## 2026-07-17 - Paper Nmax and Tmin parameter revision

### 目的

回应审稿人对 `T_min^stab=500 ms` 和 `N_max=4` 参数依据不足的质疑，把已完成的参数 sweep 和 capacity statistics 写入 `C:\Workspace\icdcs-paper\SGCP\main.tex`。

### 修改内容

- 在实验参数表后补充 `N_max=4` 的解释：它是容量控制参数而不是纯 AP tuning knob。
- 写入当前 dump 的 capacity evidence：`N_max=4` 无 singleton clusters，平均 cluster size 3.33，且每帧仍有 99.15 次 capacity-skipped candidate joins，说明硬容量约束实际生效。
- 写入 `T_min^stab=500 ms` 的解释：它是五个 10 Hz sensing cycles 的保守 hysteresis default，不声称全局最优。
- 写入 `T_min^stab=100/300/500/700/1000 ms` 当前短序列中 mAP@0.5 和 reconfiguration count 均保持 0.73 和 11，说明主结果不依赖脆弱稳定窗口调参。
- 在 sensitivity 段落补充 `N_max=2/3/4/5/6` 的 mAP@0.5 = 0.74/0.71/0.73/0.71/0.71，并说明 `N_max=4` 的选择来自避免 singleton fragmentation 与保持 capacity constraint active 的折中。

### 结论

正文现在已经覆盖 `rho_th`、`N_max` 和 `T_min^stab` 三个 reviewer 指出的关键参数。对 `T_min^stab` 的口径保持保守：当前序列显示无敏感性，但更激进动态场景仍需额外分析。

## 2026-07-17 - Short rebuttal consolidation

### 目的

把 `rebuttal_draft.md` 的长证据材料压缩成最终 rebuttal 可粘贴版本，降低后续投稿时的字数整理成本。

### 修改内容

- 新增 `rebuttal_short.md`，按 opening、R2、R3、R4 和 claim boundary 组织。
- 保留关键数值：FullPerception upper reference、payload-matched selective baseline、SGCP coverage-aware 10/20ch、`rho_th` sweep、runtime、NS3 request-level delivery。
- 明确更保守主张：SGCP 不再声称 AP 全面压过所有 selective V2V heuristic，而是强调 decentralized stability、channel feasibility、较低通信量与 NS3 可验证 transmission semantics。

### 结论

Rebuttal 目前已有长版证据稿和短版提交稿两层结构。下一步可根据会议 rebuttal 字数限制继续压缩，或补真实在线 CARLA/NS3 短回归增强系统可信度。

## 2026-07-17 - Online CARLA+NS3 short regression preparation

### 目的

为 P-1 中尚未闭环的“真实 CARLA+NS3 时间同步短回归”建立可执行协议，并先完成无需图形仿真的轻量时间同步检查。

### 命令

```powershell
conda run -n opencda python -m py_compile opencda\core\networking\network_manager.py opencda\core\networking\ns3_co_simulation\bridge\carla_ns3_bridge.py opencda\tools\offline_ns3_replay.py test\test_network_time_sync.py

conda run -n opencda python -c "from test.test_network_time_sync import test_network_time_slot_matches_carla_fixed_delta,test_multiple_network_slots_track_carla_time; test_network_time_slot_matches_carla_fixed_delta(); test_multiple_network_slots_track_carla_time(); print('network_time_sync tests passed')"

wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "ps -ef | grep -E 'ns3|vanet/main|scratch/vanet|CarlaUE4' | grep -v grep || true; ss -ltnp | grep -E ':5556|:5557' || true"
```

### 结果

- `network_time_sync tests passed`
- 未发现 CARLA / NS3 / 5556 / 5557 残留进程。
- 新增 `online_ns3_short_regression.md`，记录真实在线短回归的启动顺序、日志路径、时间同步验收条件和 subchannel 语义检查项。

### 结论

当前轻量测试继续支持此前修复：OpenCDA `NetworkManager.time_slot` 与 CARLA `fixed_delta_seconds` 使用同一时间基准。真实 CARLA+NS3 图形短回归仍待执行；执行时应先启动 WSL ns-3，再启动 CARLA，最后运行有限 tick 的 `opencda.py ... --network`。

## 2026-07-17 - Online CARLA+NS3 short regression and init fix

### 目的

实际执行真实 CARLA + NS3 有限 tick 短回归，验证此前在线时间同步修复是否能在图形仿真中工作。

### 首次回归失败

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short_20260717_031125\
```

现象：

- OpenCDA 正常 exit 0，但 NS3 在 0.05 s 后 SIGABRT。
- NS3 stderr：

```text
Ipv4AddressGeneratorImpl::Add(): Address Collision: 7.0.0.1
NS_FATAL, terminating
```

根因：

- `NetworkManager.send_msg_to_ns3()` 在线程启动后只要看到 `all_vehicles` 非空就初始化 NS3。
- `VehicleManager` 是逐辆创建的，第一辆 single CAV 注册后，traffic CAV 尚未创建；因此 NS3 先收到 `vehicles_num=1`。
- 随后第一帧真实 `vehicles_position` 为 20 车，NS3 尝试重新初始化协议栈并触发 address collision。

### 修复

修改：

- `opencda/core/networking/network_manager.py`
  - 新增 `vehicle_registration_complete` gate；
  - 新增 `mark_vehicle_registration_complete()`；
  - NS3 初始化前等待车辆注册完成，并过滤 `carla_id=None` 的半初始化帧。
- `opencda/scenario_testing/template.py`
  - 在 single CAV、traffic CAV、RSU/platoon/UAV 创建完成后调用 `mark_vehicle_registration_complete()`。

验证：

```powershell
conda run -n opencda python -m py_compile opencda\core\networking\network_manager.py opencda\scenario_testing\template.py test\test_network_time_sync.py
conda run -n opencda python -c "from test.test_network_time_sync import test_network_time_slot_matches_carla_fixed_delta,test_multiple_network_slots_track_carla_time; test_network_time_slot_matches_carla_fixed_delta(); test_multiple_network_slots_track_carla_time(); print('network_time_sync tests passed')"
```

结果：`network_time_sync tests passed`。

### 修复后在线短回归

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short_fixed_20260717_031703\
```

命令摘要：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=12.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe" -WindowStyle Hidden
$env:OPENCDA_CLUSTERING_CONFIG = "opencda/scenario_testing/config_yaml/networking_clustering_topology_gate.yaml"
$env:OPENCDA_ONLINE_TICKS = "35"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
```

关键结果：

- OpenCDA exit code：0
- NS3 initialized vehicles：20
- `sync_request/sync_ack`：38/38
- sync timeout / reconnect failure：0/0
- `MANUAL_CMD_ADD`：158
- `MANUAL_CMD_REJECT`：0
- NS fatal / SIGABRT / address collision：0
- `cam_received`：137
- PSCCH/PSSCH decode failures：1836 / 480
- OpenCDA CP counter：1
- Online AP@0.3/0.5/0.7：0.86 / 0.84 / 0.74

时间同步证据：

```text
Sent sync_ack: CARLA t=0.05s, NS3 t=0.05s
...
Sent sync_ack: CARLA t=1.90s, NS3 t=1.90s
```

### 结论

真实在线短回归确认 CARLA tick / OpenCDA network slot / NS3 sync time 已按 0.05 s 对齐，不再存在此前时间流速不一致或初始化 SIGABRT。新的待处理问题是在线真实 PHY 下大包分片和同帧并发导致部分 upload incomplete；论文主表仍应采用离线 request-level replay 的严格 110/110 RLC-complete 结果，在线短回归用于证明联仿时钟和 bridge 初始化正确。

## 2026-07-17 - Online PHY failure first diagnosis

### 目的

分析 `online_ns3_short_fixed_20260717_031703` 中大量 PSCCH/PSSCH decode failure 和 partial upload 的来源，判断是否属于时间同步/子信道语义 bug。

### 日志诊断

修复后的在线短回归中：

- NS3 初始 `vehicles_num=20`，不再出现 address collision。
- `sync_request/sync_ack=38/38`，无 sync timeout。
- `MANUAL_CMD_REJECT=0`，说明子信道窗口没有越界。
- 第一轮上传在 OpenCDA slot 6/7 可完成，例如 receiver 1/3/4/6/9/16 出现 100% uploaded。
- 后续轮次出现大量 incomplete upload，NS3 PHY failure 多集中于同一 receiver/subchannel，例如 `txRnti=20` 和 `txRnti=5` 同时在 `dstL2Id=16, scStart=5` 上触发 `reason=decoded_overlap`。

代码根因：

- `PotentialGame.run()` 调用 `clear_resource_allocation_strategy()` 只清空算法内部 `self.strategies`。
- 已经写入各 CAV `ClusteringScheduler.channel_allocation` 的旧链路不会被清空。
- 在线多轮 CP 时，新一轮 PPS 策略叠加旧策略，造成同一 receiver/subchannel 上残留链路与新链路并发进入 NS3。

### 修复

修改 `opencda/core/clustering/algorithms/resource_allocation/potential_game.py`：

- 在 `clear_resource_allocation_strategy()` 中遍历 `cav_world.get_vehicle_managers()`；
- 对每个存在 `clear_strategies()` 的 scheduler 调用清理；
- 保证每轮 PPS 更新前，online `ClusteringScheduler.channel_allocation` 与算法内部策略一起清空。

### 结论

在线短回归中的 PHY failure 不是时间同步漂移，也不是 NS3 bridge 越界子信道误收；第一类明确 bug 是 online scheduler strategy 残留。修复后需要重跑真实在线 CARLA+NS3 短回归，重点观察 `PSCCH/PSSCH_DECODE_FAIL`、incomplete upload 和 CP counter 是否改善。

## 2026-07-17 - Online scheduler strategy-clear rerun

### 目的

验证 `PotentialGame.clear_resource_allocation_strategy()` 同步清理各 CAV `ClusteringScheduler.channel_allocation` 后，真实在线 CARLA+NS3 短回归中的同 receiver/subchannel 残留冲突是否下降。

### Artifact

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_short_strategyclear_20260717_041313\
```

### 命令摘要

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=12.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe" -WindowStyle Hidden
$env:OPENCDA_CLUSTERING_CONFIG = "opencda/scenario_testing/config_yaml/networking_clustering_topology_gate.yaml"
$env:OPENCDA_ONLINE_TICKS = "35"
conda run -n opencda python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
```

### 关键结果

- OpenCDA exit code：0
- `sync_request/sync_ack`：38/38
- sync timeout / reconnect failure：0/0
- `MANUAL_CMD_ADD`：156
- `MANUAL_CMD_REJECT`：0
- NS fatal / SIGABRT / address collision：0
- `cam_received`：150
- PSCCH/PSSCH decode failures：95 / 10
- decoded-overlap failures：88
- OpenCDA successful upload lines：21
- OpenCDA incomplete upload lines：184
- OpenCDA CP counter：1
- Online AP@0.3/0.5/0.7：0.88 / 0.88 / 0.79

### 对比与结论

相对于 gate 修复后的上一轮在线短回归，`PSCCH/PSSCH_DECODE_FAIL` 从 `1836/480` 降至 `95/10`，`cam_received` 从 137 升至 150，AP 从 `0.86/0.84/0.74` 升至 `0.88/0.88/0.79`。这说明在线多轮 CP 的主冲突源确实是 scheduler strategy 残留，修复后真实 CARLA tick、OpenCDA manual subchannel request 与 NS3 接收链路基本一致。

剩余 incomplete upload 主要出现在短 35 tick 窗口中的后续轮次，下一步如需论文级在线证据，应补更长 tick 或显式 drain 阶段，并逐 request 对齐 application callback、RLC completion 与 PHY failure。

## 2026-07-17 - Online upload episode-level analysis

### 目的

把在线 OpenCDA 日志中的重复 `incomplete (NS3 mode)` 轮询行压缩成 source-target upload episode，判断剩余问题是整条链路失败、短窗口未 drain，还是单 fragment 丢失。

### 新增工具

```powershell
conda run -n opencda python -m opencda.tools.online_ns3_log_eval `
  --opencda-stdout docs\doc_workspace\SGCP\artifacts\online_ns3_short_strategyclear_20260717_041313\opencda_stdout.log `
  --ns3-stdout docs\doc_workspace\SGCP\artifacts\online_ns3_short_strategyclear_20260717_041313\ns3_stdout.log `
  --output-dir docs\doc_workspace\SGCP\artifacts\online_ns3_short_strategyclear_20260717_041313\online_eval
```

工具输出：

```text
online_upload_lifecycle.csv
online_upload_summary.json
```

### 对比结果

| Artifact | Complete Episodes | Partial Episodes | Incomplete Lines | Duplicate Incomplete Lines | PSCCH/PSSCH Fail |
| --- | ---: | ---: | ---: | ---: | ---: |
| `online_ns3_short_fixed_20260717_031703` | 14 | 8 | 245 | 237 | 1836 / 480 |
| `online_ns3_short_strategyclear_20260717_041313` | 21 | 6 | 184 | 178 | 95 / 10 |

策略清空修复后剩余 6 个 partial episode：

| Source -> Target | Received / Expected | Missing |
| --- | ---: | ---: |
| 11 -> 1 | 25424 / 35424 | 10000 |
| 14 -> 3 | 70320 / 80320 | 10000 |
| 12 -> 4 | 20560 / 30560 | 10000 |
| 7 -> 4 | 60192 / 70192 | 10000 |
| 10 -> 9 | 56896 / 66896 | 10000 |
| 13 -> 9 | 34432 / 44432 | 10000 |

### 结论

`184` 行 incomplete 不是 `184` 个失败请求，其中 `178` 行是同一批 partial episode 的重复轮询。真实未完成 episode 为 6 个，且每个都恰好缺少一个 OpenCDA `max_packet_size=10000` fragment。当前剩余问题更像单 fragment 在 PHY decode failure 后没有应用层重传/重调度；它已经不是车辆初始化、CARLA/NS3 时间同步、子信道越界或整条链路调度绕过问题。

## 2026-07-17 - Online timeout reupload first trial

### 目的

针对上一轮定位出的单 fragment 缺失，增加可配置的应用层 timeout reupload 机制，并用真实 CARLA+NS3 短回归验证是否减少 partial episode。

### 代码变更

- `CoperceptionManager` 从 `network_manager.config` 读取：
  - `upload_timeout_slots`
  - `re_upload_when_timeout`
  - `max_reupload_attempts`
- `enable_network.yaml` 中在线 NS3 默认打开一次 timeout reupload：

```yaml
upload_timeout_slots: 4
re_upload_when_timeout: true
max_reupload_attempts: 1
```

### 轻量验证

```powershell
conda run -n opencda python -m py_compile opencda\core\sensing\perception\coperception_manager.py opencda\tools\online_ns3_log_eval.py
```

另用 fake `NetworkManager/V2X/Scheduler` 验证：首次发送一次，timeout 后只重传一次，超过 `max_reupload_attempts` 后不继续重传。

### 真实回归 artifact

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_reupload_20260717_053012\
```

该轮使用修正后的 NS3 启动路径和 5556 端口等待，真实进入 CARLA+NS3 联仿。`online_ns3_log_eval` 结果：

| Metric | Value |
| --- | ---: |
| `sync_request/sync_ack` | 18 / 18 |
| `MANUAL_CMD_ADD` | 355 |
| `MANUAL_CMD_REJECT` | 0 |
| `cam_received` | 346 |
| PSCCH/PSSCH decode failures | 95 / 10 |
| Complete / partial episodes | 39 / 3 |
| Incomplete lines / duplicate incomplete lines | 44 / 41 |
| CP counter | 4 |
| Online AP@0.3/0.5/0.7 | 0.81 / 0.80 / 0.69 |

相对于 strategy-clear 回归，timeout reupload 将 complete episodes 从 21 提升到 39，partial episodes 从 6 降到 3，说明应用层补偿方向有效。

### 新发现问题与修复

本轮不是 clean exit。OpenCDA 在后续 tick 收到 late CAM completion 时，`receive_cams_via_network()` 直接索引 `self.uploading_cavs[sender_id]`，但该 sender 的 round state 已清理，触发 `KeyError: 17`。已修复为使用 `.get()` 并在缺少 start slot 时记录 `cost_slots=-1`，避免 late completion 让在线联仿崩溃。

### 结论

timeout reupload 已证明能显著减少在线 partial episode，但需要在 `KeyError` 修复后再跑一轮 clean CARLA+NS3 短回归，才能把该机制作为稳定在线协议结论写入论文/结果表。

## 2026-07-17 - Online reupload clean rerun attempts

### 目的

在 late CAM `KeyError` 修复后，尝试重跑 clean CARLA+NS3 reupload 回归，确认 timeout reupload 是否能稳定消除或继续降低 partial episode。

### 尝试记录

| Artifact | Intended Run | Result | Conclusion |
| --- | --- | --- | --- |
| `online_ns3_reupload_clean_20260717_055257` | 35 ticks, `simTime=20.0` | 启动等待阶段超时，未产生 OpenCDA/NS3 日志 | 无效回归，不作为协议证据 |
| `online_ns3_reupload_clean20_20260717_060851` | 20 ticks, `simTime=30.0` | OpenCDA 早期退出：`RuntimeError: Spawn failed because of collision at spawn position` | CARLA spawn 阶段失败，未进入 NS3/reupload 验证 |

### 当前结论

本轮没有得到 clean online reupload 结果；不能据此评价 timeout reupload 的最终效果。上一轮 `online_ns3_reupload_20260717_053012` 仍是目前唯一有效的 reupload first trial：它显示 complete/partial episode 从 `21/6` 改善到 `39/3`，但不是 clean exit。下一轮应重新启动 CARLA 后再跑 35 tick clean reupload；若再遇到 spawn collision，需要先检查 CARLA 端场景清理/初始 spawn 点占用问题。

## 2026-07-17 - CARLA spawn cleanup guard

### 目的

修复在线 clean rerun 中第一个 single CAV 固定 spawn 点被占用的问题：

```text
RuntimeError: Spawn failed because of collision at spawn position
```

### 修复

在 `ScenarioManager` 中新增显式环境开关：

```powershell
$env:OPENCDA_CLEAN_WORLD_ON_INIT = "1"
```

打开后，ScenarioManager 在创建 CAV 前销毁当前 CARLA world 中已有的动态 actor：

- `vehicle.*`
- `sensor.*`
- `walker.*`
- `controller.*`

销毁后 tick 3 次，让 CARLA 释放固定 spawn 点占位。默认不启用，避免影响普通交互实验；SGCP 在线自动回归命令中显式启用。

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\scenario_testing\utils\sim_api.py
```

### 下一步

使用 `OPENCDA_CLEAN_WORLD_ON_INIT=1` 重跑 clean online reupload 回归。若仍失败，再定位 CARLA 初始地图/traffic manager state；若通过，则更新 `online_ns3_short_regression.md` 和 `results.md` 的 clean row。

## 2026-07-17 - CARLA load_world timeout guard

### 目的

继续排查 clean online reupload 回归启动失败。`online_ns3_reupload_clean35_20260717_070046` 和 `online_ns3_reupload_clean35_20260717_070617` 都没有进入 NS3 request 阶段；日志显示 OpenCDA 在 `client.load_world('Town03')` 阶段超时：

```text
Failed to load Town03 from CARLA.
Error: time-out of 60000ms while waiting for the simulator
World loading failed
```

单独启动 CARLA 并探测 2000 端口可以成功，但 RPC 端口 ready 不等同于地图加载完成。

### 修复

`ScenarioManager` 新增环境变量覆盖 CARLA client timeout：

```powershell
$env:OPENCDA_CARLA_CLIENT_TIMEOUT = "180"
```

在线回归命令现在同时建议启用：

```powershell
$env:OPENCDA_CLEAN_WORLD_ON_INIT = "1"
$env:OPENCDA_CARLA_CLIENT_TIMEOUT = "180"
```

### 下一步

使用更长 client timeout 重跑 clean online reupload。若仍失败，应进一步检查 CARLA 是否已直接启动在 Town03，或者在启动 CARLA 时显式指定 Town03 map。

## 2026-07-17 - CARLA current-world reuse guard

### 目的

`OPENCDA_CARLA_CLIENT_TIMEOUT=180` 后，clean reupload rerun 仍卡在 `client.load_world('Town03')`：

```text
Error: time-out of 180000ms while waiting for the simulator
World loading failed
```

这说明当前 blocker 已不是 timeout 数值，而是 `load_world` RPC 在本轮 CARLA 进程中不稳定。

### 修复

`ScenarioManager` 新增显式环境开关：

```powershell
$env:OPENCDA_USE_CURRENT_CARLA_WORLD = "1"
```

启用后，如果 CARLA 已经启动并加载了地图，OpenCDA 将复用当前 world 而不是调用 `client.load_world(town)`。若当前地图名与目标 town 不一致，会打印 warning，要求操作者确认 CARLA 是用目标地图启动的。

### 下一步

下一轮 clean online reupload 应优先用 Town03 直接启动 CARLA，再设置：

```powershell
$env:OPENCDA_CLEAN_WORLD_ON_INIT = "1"
$env:OPENCDA_CARLA_CLIENT_TIMEOUT = "180"
$env:OPENCDA_USE_CURRENT_CARLA_WORLD = "1"
```

这样可以同时规避固定 spawn 点残留、短 timeout 和 `load_world` RPC 卡死。

## 2026-07-17 - CARLA RPC readiness blocker

### 目的

验证直接用 Town03 启动 CARLA 后，是否可以通过 `OPENCDA_USE_CURRENT_CARLA_WORLD=1` 跳过 `load_world()` 并继续 clean online reupload。

### 测试

已测试启动命令：

```powershell
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe" `
  -ArgumentList @('/Game/Carla/Maps/Town03','-quality-level=Low') `
  -WindowStyle Hidden
```

也测试过：

```powershell
Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe" `
  -ArgumentList @('/Game/Carla/Maps/Town03','-carla-server','-quality-level=Low','-benchmark','-fps=20','-windowed','-ResX=800','-ResY=600') `
  -WindowStyle Hidden

Start-Process "C:\Programs\Carla\WindowsNoEditor\CarlaUE4.exe" `
  -ArgumentList @('/Game/Carla/Maps/Town03','-carla-server','-quality-level=Low','-opengl','-windowed','-ResX=800','-ResY=600') `
  -WindowStyle Hidden
```

探测命令：

```powershell
conda run -n opencda python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(180); w=c.get_world(); print(repr(w.get_map().name))"
```

结果：即使等待 120 秒，并将 CARLA Python API timeout 提高到 180 秒，`client.get_world()` 仍超时：

```text
RuntimeError: time-out of 180000ms while waiting for the simulator
```

### 新增工具

新增 `opencda.tools.carla_rpc_probe`，用于在任何在线 OpenCDA/NS3 回归前验证 CARLA RPC：

```powershell
conda run -n opencda python -m opencda.tools.carla_rpc_probe --expect-map Town03 --timeout 30 --wait 180
```

只有输出类似以下内容时，才继续启动 OpenCDA/NS3：

```text
CARLA_RPC_READY map=Carla/Maps/Town03
```

### 结论

当前 blocker 已经前移到 CARLA 进程本身的 RPC readiness：2000 端口可以被探测到，但 CARLA Python API 无法稳定返回 `get_world()`。因此本轮不能继续评价 NS3/reupload。下一步应先恢复 CARLA 服务可用性，例如前台启动观察窗口日志、确认 GPU/渲染模式、确认 Town03 cooked map 可加载，或者重启 CARLA/系统后先通过 `client.get_world().get_map().name` smoke test，再跑 OpenCDA+NS3。

## 2026-07-17 - SGCP frame-level protocol trace summary

### 目的

在 CARLA RPC blocker 尚未解除时，继续补强离线协议证据链：把 receiver-level SGCP trace 汇总为 frame-level CSV，便于逐帧检查 cluster、grid selection、channel allocation、fused CAV ids、payload、prediction count 和 GT count 是否自洽。

### 新增工具

```powershell
conda run -n opencda python -m opencda.tools.sgcp_protocol_trace_summary --trace-csv docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_trace.csv --output-csv docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_frame_summary.csv
```

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\tools\sgcp_protocol_trace_summary.py
```

### 结果

```text
frames=41 trace_rows=246 total_payload_bytes=26916208 missing_channel_rows=0 output=docs\doc_workspace\SGCP\artifacts\protocol_audit\sgcp_41f_frame_summary.csv
```

帧级统计：

- 每帧 receiver count：6。
- 每帧 PPS channel links：10。
- 平均 fused CAV ids / frame：16.00。
- 平均 uploaded source ids / frame：10.00。
- 平均 payload / frame：656,492.88 bytes。
- Payload 范围：550,256 到 698,944 bytes/frame。
- 平均 selected grids / frame：523.90。
- 平均 pred boxes sum / frame：66.78。
- 平均 GT boxes sum / frame：174.90。
- `missing_channel_rows`：0。

### 结论

离线多帧协议链路现在有 receiver-level 和 frame-level 两层可检查输出。现有 OpenCOOD AP 是 41 帧全局累计指标，不能从当前 trace 严谨拆成逐帧 AP；因此文档和论文中保留全局 AP，并用帧级 CSV 解释 protocol variables 与 pred/GT count。

## 2026-07-17 - Fixed first-frame cluster membership probe

### 目的

补齐 P-1 中 fixed cluster membership 机制 probe：首帧使用 coalition game 生成 cluster head/member 模板，后续 40 帧固定该模板；每帧仍重新计算点云密度、PPS resource allocation、grid selection 和 OpenCOOD inter-cluster late fusion。该 probe 用于拆分 cluster 更新贡献与 grid/PPS 选择贡献。

### 代码更新

`opencda.tools.offline_inference` 新增：

```text
--clustering fixed_first_frame
```

### Smoke test

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_inference.py

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --clustering fixed_first_frame --max-frames 3 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\protocol_audit\fixed_first_frame_3f_trace.csv
```

3 帧 smoke test 正常结束，AP@0.3/0.5/0.7 = 0.65/0.62/0.26。

### 41 帧命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --clustering fixed_first_frame --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\protocol_audit\fixed_first_frame_41f_trace.csv
```

日志：

```text
docs\doc_workspace\SGCP\artifacts\protocol_audit\fixed_first_frame_41f_stdout.log
docs\doc_workspace\SGCP\artifacts\protocol_audit\fixed_first_frame_41f_trace.csv
docs\doc_workspace\SGCP\artifacts\protocol_audit\fixed_first_frame_41f_frame_summary.csv
```

### 结果

```text
cp counter: 41
AP@0.3 / AP@0.5 / AP@0.7 = 0.73 / 0.70 / 0.33
sgcp_summary frames=246 avg_comm_bytes=107013.07 total_comm_bytes=26325216 avg_source_cavs=2.67 avg_selected_grids=88.09
frames=41 trace_rows=246 total_payload_bytes=26325216 missing_channel_rows=0
```

帧级统计：

- 每帧 receiver count：6。
- 平均 fused CAV ids / frame：16.00。
- 平均 uploaded source ids / frame：10.00。
- 平均 payload / frame：642,078.44 bytes。
- Payload 范围：549,824 到 717,488 bytes/frame。
- 平均 selected grids / frame：528.54。
- 平均 pred boxes sum / frame：62.10。
- 平均 GT boxes sum / frame：163.66。
- `missing_channel_rows`：0。

### 结论

固定首帧 cluster membership 的 AP `0.73/0.70/0.33` 低于动态 coalition 的 `0.77/0.73/0.35`，说明 topology-aware cluster 更新确实贡献精度；但二者差距小于 SGCP grid-constrained 与 full-cluster upload 的差距，主表结果修复仍应继续集中在 coverage-aware grid utility、member/grid budget 和 AP@0.7 定位质量。

## 2026-07-17 - Per-head RB budget sensitivity

### 目的

检查 `PotentialGame.best_response()` 中写死的 `B_h=1` 是否过于保守。新增显式 probe 参数后，保持默认协议不变，只在离线机制实验中覆盖每个簇头最多使用的 RB 数。

### 代码更新

- `common.Params` 新增 `head_rb_budget=1`。
- `PotentialGame.best_response()` 从 `self.p.head_rb_budget` 读取 `B_h`。
- `opencda.tools.offline_inference` 新增：

```text
--head-rb-budget <int>
```

### 验证

```powershell
conda run -n opencda python -m py_compile opencda\core\clustering\utils\common.py opencda\core\clustering\algorithms\resource_allocation\potential_game.py opencda\tools\offline_inference.py
```

3 帧 smoke test：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --head-rb-budget 2 --max-frames 3 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_3f_trace.csv
```

### 41 帧命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --head-rb-budget 2 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_41f_trace.csv

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_trace.csv
```

### 结果

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Payload | Mbps | Avg selected grids / receiver | Missing channel rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spatial_diverse,B_h=2,rho_th=2` | 0.75 | 0.72 | 0.41 | 27,086,400 | 52.85 | 89.10 | 0 |
| `spatial_diverse,B_h=2,rho_th=3` | 0.76 | 0.72 | 0.42 | 27,962,864 | 54.56 | 90.74 | 0 |

`rho_th=3,B_h=2` 帧级统计：41 帧、平均 16 个 fused CAV/frame、10 个 uploaded sources/frame、平均 payload 682,021.07 bytes/frame、平均 selected grids 544.44/frame、`missing_channel_rows=0`。

### 结论

`B_h=2` 能把 AP@0.7 提到 `0.42`，等于 full-cluster upload 的高 IoU 结果，且通信量仅 54.56 Mbps；但 AP@0.3/0.5 降到 `0.76/0.72`，不适合作为当前主表主行。它说明 member/RB budget 能调节定位质量与召回分布，是后续算法改造方向。由于 scheduled links 可能变化，`B_h=2` 结果进入论文主表前必须补离线 NS3 replay。

## 2026-07-17 - `B_h=2,rho_th=3` offline NS3 replay

### 目的

补齐 `B_h=2,rho_th=3` high-IoU sensitivity 的 NS3 request-level delivery。该设置改变 PPS 参数，不能沿用此前 `B_h=1` 的 110/110 replay 结论。

### 代码更新

`opencda.tools.offline_ns3_replay` 新增：

```text
--head-rb-budget <int>
```

与 `offline_inference --head-rb-budget` 使用同一 `PotentialGame.p.head_rb_budget`。

### Dry-run

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_ns3_replay.py

conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --dry-run --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_bh2_rho3_11f\upload_plan_dry.csv
```

Dry-run 结果：11 帧，每帧 10 条 scheduled request、4 条 skipped unscheduled demand；`sc_start=0..9` 各 11 条。

### NS3 replay

ns-3：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && timeout 90s stdbuf -oL -eL ./ns3 run 'scratch/vanet/main.cc --simTime=2.5 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
```

OpenCDA replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --drain-seconds 1.0 --sync-timeout 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_bh2_rho3_11f\upload_plan.csv
```

Eval：

```powershell
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_bh2_rho3_11f\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_bh2_rho3_11f\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\sgcp_ns3_spatial_bh2_rho3_11f\eval --max-frames 11
```

### 结果

```text
planned_requests=110
observed_cam_received=110
bridge_observed_delivery_ratio=1.000000
avg_delay_ms=23.909
p95_delay_ms=24.000
phy_decode_failures=0
rlc_tx_events=2970
rlc_rx_events=2970
rlc_complete_requests=110
rlc_partial_requests=0
rlc_no_rx_requests=0
MANUAL_RESOURCE_APPLY=2970
MANUAL_CMD_REJECT=0
PSCCH_DECODE_FAIL=0
PSSCH_DECODE_FAIL=0
```

### 结论

`B_h=2,rho_th=3` 的 high-IoU sensitivity 在 10 子信道暴露窗口内可以完整收发：110/110 request application callback complete，110/110 RLC complete，PHY failures 为 0。该结果移除了 NS3 pending 标记；但由于 AP@0.3/0.5 下降，它仍更适合写成定位质量/高 IoU tradeoff，而不是直接替换低预算主行。

## 2026-07-17 - Late NMS threshold probe

### 目的

排查 `B_h=2,rho_th=3` 虽然 AP@0.7 达到 full-cluster 水平，但 AP@0.3/0.5 下降的问题是否由 inter-cluster late fusion NMS 阈值导致。

### 代码更新

`opencda.tools.offline_inference` 新增：

```text
--sgcp-late-nms-thresh <float>
```

默认值仍为 `0.15`，保持历史结果不变。

### 命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_inference.py

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --sgcp-late-nms-thresh 0.05 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_nms005_41f_trace.csv

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --sgcp-late-nms-thresh 0.30 --max-frames 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_nms030_41f_trace.csv
```

### 结果

| Late NMS IoU | AP@0.3 | AP@0.5 | AP@0.7 | Payload | Missing channel rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.73 | 0.70 | 0.40 | 27,962,864 | 0 |
| 0.15 | 0.76 | 0.72 | 0.42 | 27,962,864 | 0 |
| 0.30 | 0.75 | 0.71 | 0.41 | 27,962,864 | 0 |

### 结论

默认 late NMS `0.15` 是三档中最好结果。放宽到 `0.30` 没有恢复 AP@0.3/0.5，收紧到 `0.05` 更差。因此 `B_h=2` 的召回/AP@0.3/0.5 下降不是简单由 inter-cluster late NMS 阈值造成；下一步应检查 member/grid selection、box score distribution 和 per-cluster detection quality。

## 2026-07-17 - Late-fusion box-count diagnostics

### 目的

解释 `B_h=2,rho_th=3` 为什么 AP@0.7 提升到 `0.42`，但 AP@0.3/0.5 降到 `0.76/0.72`。本轮不重跑 detector，只解析已有 offline inference stdout 中的 per-cluster source box count 和 inter-cluster late-fusion box count。

### 代码更新

新增只读解析工具：

```text
opencda.tools.sgcp_late_fusion_log_summary
```

输出逐帧 CSV：source prediction sum、fused prediction boxes、suppressed boxes、source/fused GT boxes、payload 和全局 AP。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.sgcp_late_fusion_log_summary --label spatial10-rho2-bh1 --log docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_stdout.log --output-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_late_summary.csv

conda run -n opencda python -m opencda.tools.sgcp_late_fusion_log_summary --label spatial10-rho2-bh2 --log docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_41f_stdout.log --output-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_41f_late_summary.csv

conda run -n opencda python -m opencda.tools.sgcp_late_fusion_log_summary --label spatial10-rho3-bh2 --log docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_stdout.log --output-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_late_summary.csv

conda run -n opencda python -m opencda.tools.sgcp_late_fusion_log_summary --label spatial20-rho2-bh1 --log docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch20_41f_stdout.log --output-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch20_41f_late_summary.csv
```

### 结果

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Avg. Source Pred. | Avg. Fused Pred. | Avg. Suppressed Pred. | Avg. Fused GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spatial10-rho2-bh1` | 0.79 | 0.75 | 0.37 | 28,743,280 | 68.37 | 55.90 | 12.46 | 69.00 |
| `spatial10-rho2-bh2` | 0.75 | 0.72 | 0.41 | 27,086,400 | 67.37 | 53.51 | 13.85 | 64.83 |
| `spatial10-rho3-bh2` | 0.76 | 0.72 | 0.42 | 27,962,864 | 67.93 | 53.71 | 14.22 | 64.83 |
| `spatial20-rho2-bh1` | 0.80 | 0.76 | 0.41 | 37,912,544 | 73.22 | 56.24 | 16.98 | 69.29 |

补充 NMS probe：`B_h=2,rho_th=3,NMS=0.05` 的 avg fused GT 为 63.10，`NMS=0.30` 为 65.83，但 AP 仍低于默认 0.15。

### 结论

`B_h=2` 高 IoU 提升伴随 fused GT 覆盖减少，而不是融合框数量增加。当前更合理的解释是资源预算变化改变了 head/member/grid 覆盖对象分布，使剩余目标定位更准但低阈值召回面变窄。主表低通信候选仍优先保留 `B_h=1` coverage-aware 10ch/20ch；`B_h=2` 作为 high-IoU sensitivity，下一步检查 cluster-head/member selection 与 target coverage。

## 2026-07-17 - CAV coverage diagnostics for `B_h=2`

### 目的

继续解释 `B_h=2` 的覆盖损失：上一轮已经确认 fused GT 下降，本轮检查每个 CAV 的 head/member/uploaded/fused/unscheduled 帧数，判断是 fused CAV 总数减少，还是具体成员替换导致覆盖分布变化。

### 代码更新

新增只读解析工具：

```text
opencda.tools.sgcp_trace_coverage_summary
```

它从 `--sgcp-trace-output` 生成的 receiver-level trace 中输出 per-CAV coverage CSV 和 per-frame coverage CSV。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.sgcp_trace_coverage_summary --label spatial10-rho2-bh1 --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_trace.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_cav_coverage.csv --output-frame-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_frame_coverage.csv

conda run -n opencda python -m opencda.tools.sgcp_trace_coverage_summary --label spatial10-rho2-bh2 --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_41f_trace.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_41f_cav_coverage.csv --output-frame-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_bh2_41f_frame_coverage.csv

conda run -n opencda python -m opencda.tools.sgcp_trace_coverage_summary --label spatial10-rho3-bh2 --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_trace.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_cav_coverage.csv --output-frame-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_frame_coverage.csv

conda run -n opencda python -m opencda.tools.sgcp_trace_coverage_summary --label spatial20-rho2-bh1 --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch20_41f_trace.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch20_41f_cav_coverage.csv --output-frame-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_ch20_41f_frame_coverage.csv
```

### 结果

| Variant | Avg. Fused CAVs / Frame | Avg. Uploaded CAVs / Frame | Avg. Unscheduled Members / Frame | Avg. Selected Grids / Frame | Avg. Uploaded Points / Frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| `spatial10-rho2-bh1` | 16.00 | 10.00 | 4.00 | 523.90 | 43,815.98 |
| `spatial10-rho2-bh2` | 16.00 | 10.00 | 4.00 | 534.59 | 42,626.32 |
| `spatial10-rho3-bh2` | 16.00 | 10.00 | 4.00 | 544.44 | 42,626.32 |
| `spatial20-rho2-bh1` | 20.00 | 14.00 | 0.00 | 703.10 | 57,793.51 |

核心 per-CAV 差异：

| CAV | Uploaded Frames `B_h=1` | Uploaded Frames `B_h=2,rho3` | Fused Frames `B_h=1` | Fused Frames `B_h=2,rho3` | Unscheduled Frames `B_h=1` | Unscheduled Frames `B_h=2,rho3` | Uploaded Points `B_h=1` | Uploaded Points `B_h=2,rho3` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 29 | 32 | 38 | 41 | 3 | 0 | 109,102 | 131,425 |
| 5 | 6 | 31 | 6 | 31 | 35 | 10 | 7,366 | 50,903 |
| 6 | 41 | 7 | 41 | 7 | 0 | 34 | 187,287 | 32,597 |
| 12 | 5 | 10 | 36 | 41 | 5 | 0 | 21,591 | 34,532 |

### 结论

`B_h=2` 在 10ch 下没有增加 fused CAV 总数，仍为 16/20 CAV；它主要改变了“谁被上传”。CAV 6 从 41 帧全程上传降到 7 帧，CAV 5/4/12 增加覆盖，但 fused GT 和低阈值 AP 下降。下一步算法应加入 coverage fairness / persistent contributor protection / target coverage fallback，而不是简单提高 `B_h`。

## 2026-07-17 - Persistent coverage fallback negative probe

### 目的

把上一轮 coverage diagnostics 转成最小算法 probe：在 10ch `B_h=2,rho_th=3` 下，保持每帧 10 条上传链路不变，用历史 coverage deficit 将长期未调度成员替换进同簇调度，并复用被替换成员的子信道。验证“保护长期欠覆盖成员”是否能恢复 AP@0.3/0.5。

### 代码更新

`opencda.tools.offline_inference` 新增显式关闭默认的 probe 开关：

```text
--sgcp-coverage-fallback {none,persistent}
```

`persistent` 只在候选成员历史欠覆盖达到阈值、且候选 selected-grid density 不明显低于被替换成员时执行替换；替换复用原 subchannel 和 grid count，不增加 link count 或绕过 PPS。默认 `none` 保持既有结果不变。

### 命令

无 fallback 对照：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --max-frames 11 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_baseline_11f_trace.csv
```

Persistent fallback：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --sgcp-coverage-fallback persistent --max-frames 11 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_persistent_conservative_11f_trace.csv
```

### 结果

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Missing Channel Rows | Zero-Pred Rows | Frame Replacements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_h=2,rho3`, no fallback | 11 | 0.69 | 0.64 | 0.34 | 7,416,720 | 0 | 0 | 0 |
| `B_h=2,rho3`, persistent fallback | 11 | 0.67 | 0.62 | 0.34 | 7,453,808 | 0 | 0 | 7 |

### 结论

Persistent fallback 没有修复 AP@0.3/0.5，反而略降。协议层仍正确：`missing_channel_rows=0`，替换复用了已有 subchannel，没有绕过 PPS。负面结果说明“历史欠覆盖公平性”不足以作为上传成员替换准则；下一步应把 fallback 与 detector-quality proxy / per-target coverage / uncertainty 或 frame-level object recall 绑定，而不是只按 CAV 级 coverage deficit 替换。

## 2026-07-17 - Detector-quality proxy summary

### 目的

为下一步 object-aware / quality-aware fallback 建立可复现诊断：从 SGCP receiver-level trace 中提取每个 cluster-head source 的 `pred_boxes/gt_boxes`、zero/low-ratio 风险、payload 和每个 uploaded CAV 参与的 receiver 行质量，解释为什么简单 coverage fallback 会伤 AP。

### 代码更新

新增只读解析工具：

```text
opencda.tools.sgcp_source_quality_summary
```

说明：该工具输出 detector-quality proxy，不计算 AP。AP 仍以 OpenCOOD 全局累计结果为准。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.sgcp_source_quality_summary --label spatial10-rho2-bh1 --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_trace.csv --output-receiver-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_receiver_quality.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_grid_41f_cav_quality.csv

conda run -n opencda python -m opencda.tools.sgcp_source_quality_summary --label spatial10-rho3-bh2 --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_trace.csv --output-receiver-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_receiver_quality.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_41f_cav_quality.csv

conda run -n opencda python -m opencda.tools.sgcp_source_quality_summary --label spatial10-rho3-bh2-baseline-11f --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_baseline_11f_trace.csv --output-receiver-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_baseline_11f_receiver_quality.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_baseline_11f_cav_quality.csv

conda run -n opencda python -m opencda.tools.sgcp_source_quality_summary --label spatial10-rho3-bh2-persistent-11f --trace-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_persistent_conservative_11f_trace.csv --output-receiver-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_persistent_conservative_11f_receiver_quality.csv --output-cav-csv docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_persistent_conservative_11f_cav_quality.csv
```

### 结果

| Variant | Receiver Rows | Avg. Pred/GT Ratio | Low-Ratio Rows | Zero-Pred Rows | Avg. Bytes / Receiver |
| --- | ---: | ---: | ---: | ---: | ---: |
| `spatial10-rho2-bh1`, 41f | 246 | 0.3928 | 10 | 2 | 116,842.60 |
| `spatial10-rho3-bh2`, 41f | 246 | 0.4461 | 9 | 2 | 113,670.18 |
| `spatial10-rho3-bh2`, 11f | 66 | 0.4284 | 7 | 0 | 112,374.55 |
| `spatial10-rho3-bh2 persistent`, 11f | 66 | 0.4242 | 7 | 0 | 112,936.48 |

Key uploaded-CAV proxy deltas between 41f `B_h=1` and 41f `B_h=2,rho3`:

| CAV | Upload Rows `B_h=1` | Upload Rows `B_h=2,rho3` | Avg. Pred/GT Ratio `B_h=1` | Avg. Pred/GT Ratio `B_h=2,rho3` | Low-Ratio Rows `B_h=1` | Low-Ratio Rows `B_h=2,rho3` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 41 | 7 | 0.6341 | 0.5746 | 0 | 0 |
| 5 | 6 | 31 | 0.3129 | 0.3893 | 0 | 0 |
| 12 | 5 | 10 | 0.3977 | 0.3081 | 0 | 0 |
| 10 | 39 | 37 | 0.2827 | 0.2885 | 1 | 1 |
| 18 | 9 | 8 | 0.1837 | 0.1654 | 4 | 4 |

### 结论

`B_h=2` 的 receiver-level quality proxy 高于 `B_h=1`，这解释了 AP@0.7 上升；但它把高质量长期贡献者 CAV 6 的上传从 41 行降到 7 行，并增加了 CAV 5 等较低 ratio 成员。因此下一步不是普通 coverage fairness，而是 quality-weighted coverage：只有当候选成员能提供足够 detector-quality / target-level coverage 时才替换当前成员。

## 2026-07-17 - Quality-persistent fallback safety probe

### 目的

将 plain persistent fallback 扩展为质量约束版本：只有候选成员具备至少 2 条历史 quality sample，且历史 pred/GT ratio 不低于被替换成员的 90% 且不低于 0.25 时，才允许复用原 subchannel 替换当前上传成员。验证质量门槛能否避免上一轮 plain persistent 的负面替换。

### 代码更新

`opencda.tools.offline_inference --sgcp-coverage-fallback` 新增：

```text
quality_persistent
```

默认仍为 `none`，不改变既有主表结果。`quality_persistent` 使用前序帧 receiver-level `pred_boxes/gt_boxes` 作为 detector-quality proxy。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --sgcp-constrained --sgcp-inter-cluster-late-fusion --resource-allocation potential_game --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --head-rb-budget 2 --sgcp-coverage-fallback quality_persistent --max-frames 11 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\mechanism_probe\spatial_diverse_rho3_bh2_quality_persistent_11f_trace.csv
```

### 结果

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Total Bytes | Missing Channel Rows | Zero-Pred Rows | Frame Replacements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_h=2,rho3`, no fallback | 11 | 0.69 | 0.64 | 0.34 | 7,416,720 | 0 | 0 | 0 |
| `B_h=2,rho3`, persistent fallback | 11 | 0.67 | 0.62 | 0.34 | 7,453,808 | 0 | 0 | 7 |
| `B_h=2,rho3`, quality-persistent fallback | 11 | 0.69 | 0.64 | 0.34 | 7,416,720 | 0 | 0 | 0 |

### 结论

Quality-persistent fallback 阻止了 plain persistent 的 7 次有害替换，结果回到 no-fallback baseline。这说明 detector-quality 门槛是必要安全条件；但当前规则过于保守，没有带来 AP 提升。下一步应改为 target-level/object-aware 候选生成，而不是只在 CAV 级别做 history/quality gate。

## 2026-07-17 - Online CP-count and communication-accounting diagnosis

### 目的

排查真实在线 `v2xp_cluster_carla --network` 中 AP 高但 `cp counter` 过少的问题，并把在线通信量统计与离线主表的 payload/Mbps 口径对齐。

### 输入日志

用户在线实验命令：

```powershell
python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
```

OpenCDA 日志：

```text
C:\Workspace\OpenCDA\opencda\log\opencda_20260717_161909.log
```

用户记录的 stdout 结果：

```text
cp counter: 1
AP@0.3/0.5/0.7 = 0.86 / 0.86 / 0.71
total_volume_bytes = 4,495,080
try_volume = 3,367,776
total_slots = 38
```

### 诊断命令

```powershell
conda run -n opencda python -m opencda.tools.online_ns3_log_eval --opencda-stdout C:\Workspace\OpenCDA\opencda\log\opencda_20260717_161909.log --output-dir docs\doc_workspace\SGCP\artifacts\online_ns3_user_20260717_161909_eval
```

### 诊断结果

解析输出：

```text
docs\doc_workspace\SGCP\artifacts\online_ns3_user_20260717_161909_eval\online_upload_summary.json
```

关键结果：

| Metric | Value |
| --- | ---: |
| CP eval frames | 3 |
| CP submit frames | 3 |
| CP wait frames | 185 |
| CP wait frames, ego=1 | 34 |
| Upload episodes observed | 11 |
| Application complete episodes | 0 |
| Application partial episodes | 11 |
| AP@0.3 / AP@0.5 / AP@0.7 | 0.86 / 0.86 / 0.71 |
| Total counted traffic | 4,495,080 bytes |
| Try upload traffic | 3,367,776 bytes |
| Duration for Mbps accounting | 3.8 s |
| Total counted traffic Mbps | 9.46 Mbps |
| Try upload Mbps | 7.09 Mbps |

### 结论

在线 CP 次数过少不是分簇没有运行，而是 late-fusion 评价提交被 ego CP 阻塞：`submit_cp_results()` 需要 ego 本帧完成 CP 来提供 ego GT，但 ego 作为 cluster head 时长期处于 `CP_WAIT_FRAME`。日志中 ego=1 等待 34 次，且 11 个 online upload episode 均为 application partial，没有 complete episode。当前在线 AP 很高，但只有极少数 CP frame 进入统计，不能直接替换 41 帧离线主表。

通信量口径已明确：在线日志中的 `total_volume_bytes` 是 `intra_upload + intra_download + inter_cluster` 的总计，按 `total_slots * time_slot` 换算为 9.46 Mbps；`try_volume` 是尝试发送的 intra-cluster upload，按同一时长换算为 7.09 Mbps。离线主表主要报告点云 upload payload，因此后续在线/离线对齐应同时报告 `total_payload_mbps` 和 `try_payload_mbps`，并标明是否包含 inter-cluster boxes / download。

### 代码更新

- `CoperceptionManager.upload_wait_exhausted()`：当 NS3 上传超过 timeout/re-upload 预算仍未完整到达时，允许本轮 CP 使用实际到达的 partial uploads 继续执行，避免在线评估长时间卡在等待状态。
- `ClusteringPerceptionManager`：`CP_EVAL_FRAME` 日志新增 `uploads_ready` 和 `wait_exhausted`；late-fusion submit 不再因为没有 remote upload 而跳过统计，便于与离线“每帧提交一次”口径对齐。
- `NetworkManager.get_communication_report()`：新增 `duration_s`、`total_payload_mbps`、`try_payload_mbps`。
- `opencda.tools.online_ns3_log_eval`：新增 `CP_EVAL_FRAME`、`CP_SUBMIT_FRAME`、`CP_WAIT_FRAME` 和通信报告解析。

### 下一步验证

在当前 CARLA 实例可用时，建议用固定 tick 重跑：

```powershell
$env:OPENCDA_ONLINE_TICKS = "80"
python opencda.py -t v2xp_cluster_carla --apply_cp --apply_ml --debug --network
```

验收标准：`cp_submit_frames` 应随 tick 数增长，不再只有 1--3 次；`online_upload_summary.json` 应同时给出 AP、CP wait 分布和 Mbps。若 application complete episode 仍为 0，则继续排查 NS3 callback 的 request_id/send_timestamp 分片合并语义。

## 2026-07-17 用户在线重跑：CP 数增加但远端融合仍不足

### 输入日志

OpenCDA 日志：

```text
C:\Workspace\OpenCDA\opencda\log\opencda_20260717_163551.log
```

用户记录的 stdout 结果：

```text
pkts sent = 418
cp counter = 4
AP@0.3 / AP@0.5 / AP@0.7 = 0.51 / 0.48 / 0.41
total_payload_mbps = 18.5054
try_payload_mbps = 15.8301
```

### 诊断命令

```powershell
conda run -n opencda python -m opencda.tools.online_ns3_log_eval --opencda-stdout C:\Workspace\OpenCDA\opencda\log\opencda_20260717_163551.log --output-dir docs\doc_workspace\SGCP\artifacts\online_ns3_user_20260717_163551_eval
```

### 诊断结果

| Metric | Value |
| --- | ---: |
| CP eval frames | 4 |
| CP submit frames | 4 |
| CP wait frames | 193 |
| Upload episodes observed | 35 |
| Application complete episodes | 26 |
| Application partial episodes | 9 |
| Combined message lines | 216 |
| Success log lines | 26 |
| Total counted traffic Mbps | 18.51 Mbps |
| Try upload Mbps | 15.83 Mbps |

### 结论

该结果不能解释为“在线联合仿真中大部分包都发送失败”。日志中有大量 `cam_received` / `Combined message`，且解析出 26 个完整 application upload episode。真正的问题是 CP 消费窗口不足：4 次 CP submit 中，slot 0 和 slot 2 没有远端点云，slot 7 使用 `[2, 11]`，slot 15 仅在 `wait_exhausted=True` 后使用 `[2]`。因此 AP 下降主要来自在线 CP 没有稳定消费已发送/晚到的远端点云，而不是 NS3 完全丢包。

这与离线仿真的差异在于：离线 request-level replay 通常统计“请求最终完成/链路可行”，而在线融合需要“当前 CP 截止前完整 payload 到达并被本帧消费”。后续离线/在线主表对齐应采用 deadline-aware delivery/cropping，或在线端改进 CP scheduling、fragment reassembly 和 late completion 处理。

## 2026-07-17 在线 CARLA/NS3 严格同步修复验证

### 修复内容

- `NetworkManager` 增加严格 NS3 barrier：每个 CARLA tick 先发送本 tick 的车辆位置和 transfer requests，再向 NS3 发送 `sync_request`；CARLA 主循环等待 `sync_ack` 后才进入下一 tick。
- 严格同步模式下 `sync_timeout` 下限提升到 60 s，避免 NS3 事件仿真慢于 wall-clock 时被误判为通信失败。
- `CoperceptionManager` 增加 `min_upload_ratio` 和 `min_upload_count`，在线 deadline 到达时允许“至少 1 个远端上传完成”触发本轮融合，避免 CP 长期卡在完整上传等待。

### 80 tick 在线结果

| Variant | Sync Timeout | CP Submit | Complete / Partial Episodes | AP@0.3 | AP@0.5 | AP@0.7 | Total Mbps | Try Mbps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 用户 run, 修复前 | - | 4 | 26 / 9 | 0.51 | 0.48 | 0.41 | 18.51 | 15.83 | NS3 回调存在，但 CP 消费帧很少 |
| strict sync + `min_upload_count=1` + 1 次受控重传 | 0 | 10 | 55 / 2 | 0.70 | 0.68 | 0.58 | 25.48 | 17.85 | 当前最佳在线口径 |
| strict sync + `min_upload_count=1` + 无重传 | 0 | 7 | 45 / 3 | 0.64 | 0.59 | 0.50 | 23.94 | 19.52 | decode overlap 下降但 deadline 内可用上传减少 |

### 结论

CARLA 与 NS3 时间流速不一致问题已经定位并修复：最新两轮真实在线回归均无 `sync timeout`，NS3 日志中的 `sync_ack` 与 CARLA 目标时间对齐，且 `MANUAL_RESOURCE_APPLY` 显示 `requestedStart == physicalStart`，说明 OpenCDA 指定子信道真实落到 NS3 发送行为。

在线结果已从用户 run 的 `0.51/0.48/0.41` 提升到 `0.70/0.68/0.58`，但仍低于离线 41 帧 SGCP `0.79/0.75/0.37` 或 20ch `0.80/0.76/0.41` 的主表候选。剩余差距主要来自在线 deadline 语义：部分 request 虽会最终完成，但未必在当前融合周期截止前完整到达并被消费。无重传对照说明完全关闭重传会降低 deadline 内可用点云；后续应把“最终 request delivery”和“deadline-aware CP delivery”分开报告。

## 2026-07-17 主表重构：object-level 诊断与 payload cap probe

### 目的

用户要求最终形成一张 SGCP 同时具备最高 AP、最低 Mbps，并与 Random/Greedy/FullPerception 使用统一参数的主表。本轮先补三件事：

1. 用 20 CAV full fusion 确认当前 dump 的 AP 上界。
2. 定位 full fusion 能检出但 SGCP 漏检的 GT 对象和空间区域。
3. 尝试两个低通信算法 probe：目标局部聚集选格和点级 payload cap。

### 关键命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\object_diag_full_41f.csv

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --sgcp-grid-selection-mode spatial_diverse --num-channels 5 --bandwidth-mhz 20 --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\object_diag_sgcp_spatial_5ch20mhz_41f.csv --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\object_diag_sgcp_spatial_5ch20mhz_41f_trace.csv

conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation potential_game --sgcp-inter-cluster-late-fusion --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --max-upload-points-per-source 3000 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\main_table_sgcp_spatial_rho3_10ch_cap3000_41f_trace.csv
```

### 结果

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full 20-CAV early reference | 41 | 0.85 | 0.83 | 0.48 | 60,838,528 | 118.71 | 当前 dump 上界 |
| SGCP `spatial_diverse`, 5ch/20MHz | 41 | 0.56 | 0.53 | 0.27 | 14,815,408 | 28.91 | 强低带宽 stress；AP 太低 |
| Random scheduler, 5ch/20MHz | 41 | 0.43 | 0.38 | 0.18 | 9,531,504 | 18.60 | 未充分利用 payload，不适合证明通信节省 |
| MWS scheduler, 5ch/20MHz | 41 | 0.31 | 0.26 | 0.11 | 9,989,952 | 19.49 | 未充分利用 payload，不适合证明通信节省 |
| SGCP `object_clustered`, 5ch/20MHz | 2 | 0.50 | 0.48 | 0.24 | 643,424 | 25.74 on 0.2s | 负面 probe，局部聚集选格会伤覆盖 |
| SGCP `spatial_diverse,rho_th=3,10ch`, cap=3000 | 41 | 0.74 | 0.70 | 0.33 | 19,510,848 | 38.07 | 有效降低 payload，但 AP 低于无 cap |

Object-level 诊断：

- 5ch/20MHz SGCP 中，full reference matched 但 method missed 的 GT 为 773 个。
- 高频漏检对象包括 `385` 41 帧、`400` 38 帧、`427` 37 帧、`337` 37 帧、`362` 35 帧。
- 漏检中心主要集中在 ego 左侧/左后：x 分位约 `[-95.28, -56.01, -36.67, -17.27, 0.50, 48.93, 90.33]`，y 分位约 `[-36.16, -31.45, -25.13, -10.43, 0.28, 8.33, 37.10]`。
- top missed grid 包括 `-2_-3`、`-4_-4`、`-5_0`、`0_0`、`4_-2`，说明单纯按全局 density/diversity 排序仍会系统性漏掉某些目标轨迹。

### 结论

- 当前 full fusion 上界仍是 `0.85/0.83/0.48`，论文主表不能追求超过该 dump 的 early-fusion AP 上界。
- 5ch/20MHz 可以显著压低 Random/MWS AP，但 Random/MWS payload 也很低，不能支持“SGCP Mbps 最低”的主表叙事。
- `object_clustered` 负面结果说明检测 AP 需要跨区域覆盖，而不是把点云集中在局部高密度目标块。
- `--max-upload-points-per-source 3000` 是有效通信旋钮，把 SGCP `rho_th=3,10ch` 从 57.38 Mbps 降到 38.07 Mbps，但 AP 降为 `0.74/0.70/0.33`；它适合做 payload sensitivity，不宜直接作为最终主表主行。
- 下一步若要满足“SGCP 最高 AP + 最少 Mbps”，必须先把 Random/Greedy baseline 改成强制使用相同带宽/相同 payload cap 的版本，或将主表切换为 payload-matched selective baseline；否则低 payload 的弱 Random/MWS 会破坏表格叙事。

## 2026-07-17 对象级失败诊断：GT 坐标、分簇、调度覆盖

### 目的

用户要求通过打印真值框坐标、车辆分簇结果、车辆坐标等方式诊断 AP 低的原因，并区分四类问题：带宽过低、分簇效果差、资源调度/区块选择不当、最后跨簇晚期融合不当。

### 代码与命令

新增工具：

```text
opencda/tools/sgcp_failure_diagnostics.py
```

诊断命令：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_failure_diagnostics --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-grid-selection-mode spatial_diverse --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --object-diagnostics-csv docs\doc_workspace\SGCP\artifacts\object_diag_sgcp_spatial_rho3_10ch_41f.csv --output-dir docs\doc_workspace\SGCP\artifacts\failure_diag_spatial_rho3_10ch_41f
```

输出：

```text
docs\doc_workspace\SGCP\artifacts\failure_diag_spatial_rho3_10ch_41f\vehicles.csv
docs\doc_workspace\SGCP\artifacts\failure_diag_spatial_rho3_10ch_41f\clusters.csv
docs\doc_workspace\SGCP\artifacts\failure_diag_spatial_rho3_10ch_41f\schedules.csv
docs\doc_workspace\SGCP\artifacts\failure_diag_spatial_rho3_10ch_41f\gt_objects.csv
docs\doc_workspace\SGCP\artifacts\failure_diag_spatial_rho3_10ch_41f\summary.json
```

### 关键发现

10ch/rho3 下，`gt_objects.csv` 共 653 行 GT，其中 111 行为 full-reference 可检出但 SGCP 漏检：

| Bucket | Rows | Meaning |
| --- | ---: | --- |
| Covered only by other cluster heads | 63 | 目标 grid 被某条链路上传，但没有发给最近/最相关 head |
| Nearest head got dense points but no final box | 35 | 正确 head 收到不少点云，但检测/晚融合仍没形成匹配框 |
| Nearest head got sparse object-grid points | 12 | 正确 head 收到目标 grid，但点数太少，sender/view 质量不足 |
| No scheduled covering grid | 1 | 纯粹没有任何调度链路覆盖目标 grid |

代表性打印样例：

- Object `337`，frame `000062`，GT world `(8.000, -30.314)`，grid `0_-3`，nearest CAV `1`，nearest head `1`，nearest CAV 该 grid 点数 `1453`，但 nearest-head covering points `0`，full-reference IoU `0.869183`，SGCP IoU `0.000000`。
- Object `401`，frame `000062`，GT world `(25.433, 4.033)`，grid `2_0`，nearest CAV `12`，nearest head `12`，nearest CAV 该 grid 点数 `2057`，但 nearest-head covering points `0`，full-reference IoU `0.886449`，SGCP IoU `0.000000`。
- Object `374`，frame `000080`，GT world `(8.372, 18.164)`，grid `0_1`，nearest CAV `7`，nearest head `4`，nearest CAV 该 grid 点数 `1487`，但 nearest-head covering points `7`，full-reference IoU `0.679596`，SGCP IoU `0.000000`。
- Object `419`，frame `000060`，GT world `(10.126, -25.023)`，grid `1_-3`，nearest CAV `1`，nearest head `11`，nearest-head covering points `148`，但 SGCP 仍无匹配框。

### 结论

- 带宽过低是 5ch stress AP 低的明确原因，但不是 10ch/rho3 剩余漏检的唯一原因。
- 分簇层次不是完全失效；full-cluster / 20ch 结果说明 hierarchy 可用。但当前 cluster/head assignment 与资源调度没有保证目标 grid 发给最相关 head。
- 当前首要问题是资源调度/区块选择缺少 target-aware receiver/sender 保护：很多关键 grid 有高点数，但没有送到最近/正确 head。
- 晚期融合是 secondary 问题：35 个漏检已经有较密点云送到最近 head，但仍无最终框。下一步需要 dump per-head pre-NMS boxes，确认是 detector 未出框还是 inter-cluster late fusion/NMS 丢框。

详细记录见 `failure_diagnostics.md`。

## 2026-07-17 目标感知势博弈调度器：直接算法改造

### 目的

用户要求不要再在旧结果上做后处理修补，而是直接改资源调度/区块选择算法，同时尽量保留论文中的势博弈叙事。

### 代码改动

新增算法：

```text
opencda/core/clustering/algorithms/resource_allocation/target_aware_potential_game.py
```

注册入口：

```text
target_aware_potential_game
target_aware_pg
tapg
```

机制：第一阶段完全复用原 `PotentialGame` 的 sender / subchannel best-response，保留原势博弈资源竞争叙事；第二阶段在 allocator 内部执行 target-aware grid-action refinement，不再依赖 `offline_inference --sgcp-grid-selection-mode spatial_diverse` 这种外部后处理。新 grid utility 对 head weak grids、member high-density target-like grids、多 CAV 可见区域和 multi-view value 保留正边际效用，避免“单视角 density 超过 `rho_th` 就不再请求”的饱和问题。

### 关键命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation target_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\target_aware_pg_10ch_rho3_41f_trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\object_diag_target_aware_pg_10ch_rho3_41f.csv

conda run -n opencda python -m opencda.tools.sgcp_failure_diagnostics --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --resource-allocation target_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --object-diagnostics-csv docs\doc_workspace\SGCP\artifacts\object_diag_target_aware_pg_10ch_rho3_41f.csv --output-dir docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f

conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --resource-allocation target_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\target_aware_pg_ns3_10ch_rho3_11f_upload_plan.csv --dry-run
```

### 结果

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `potential_game + spatial_diverse`, 10ch/rho3 | 0.79 | 0.76 | 0.38 | 29,405,296 | 57.38 | 89.72 |
| `target_aware_potential_game`, 10ch/rho3 | 0.80 | 0.76 | 0.39 | 31,069,968 | 60.62 | 89.72 |

对象级诊断对比：

| Metric | Old spatial/rho3 | Target-aware PG |
| --- | ---: | ---: |
| Full-reference detected but SGCP missed rows | 111 | 106 |
| Covered only by other cluster heads | 63 | 56 |
| Nearest head got dense points but no final box | 35 | 36 |
| Nearest head got sparse object-grid points | 12 | 12 |
| No scheduled covering grid | 1 | 2 |
| Nearest-head covering point mean | 69.4 | 79.0 |

11 帧 NS3 dry-run 结果：每帧 10 条 scheduled request，4 条 unscheduled demand 被跳过，生成 `target_aware_pg_ns3_10ch_rho3_11f_upload_plan.csv`。尚未做真实 NS3 socket replay；下一步若要写入论文主表，需要启动 NS3 后确认 application/RLC complete。

### 结论

新算法不是在旧结果上修补，而是把 target-aware coverage 纳入资源调度器本身。AP 有小幅稳定收益，且对象级诊断显示此前最主要的“target grid 只被其他 head 覆盖”问题减少。代价是 payload 从 57.38 Mbps 升到 60.62 Mbps；当前仍显著低于 FullPerception centralized 的 118.71 Mbps，但若论文要强调“最低 Mbps”，仍需继续做强制预算 Random/Greedy 或点数 cap 版 target-aware PG。

## 2026-07-17 Target-grid case study 与 object-aware PG 分支

### 目的

用户要求选取若干帧仔细分析，定位当前漏检 GT 对应 grid，解释为什么调度算法没有选择这些 grid，并设计新的算法。重点不是继续修补旧 `spatial_diverse`，而是回到资源调度器本体。

### 新增代码与文档

```text
opencda/tools/sgcp_grid_miss_analysis.py
opencda/core/clustering/algorithms/resource_allocation/object_aware_potential_game.py
docs/doc_workspace/SGCP/target_grid_case_study.md
```

新增资源分配入口：

```text
object_aware_potential_game
object_aware_pg
oapg
```

### 诊断命令

```powershell
conda run -n opencda python -m opencda.tools.sgcp_grid_miss_analysis --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --failure-gt-csv docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f\gt_objects.csv --resource-allocation target_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --max-objects 8 --max-rows-per-object 3 --output-csv docs\doc_workspace\SGCP\artifacts\grid_miss_analysis_target_aware_pg_top8.csv

conda run -n opencda python -m opencda.tools.sgcp_grid_miss_analysis --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --failure-gt-csv docs\doc_workspace\SGCP\artifacts\failure_diag_target_aware_pg_10ch_rho3_41f\gt_objects.csv --resource-allocation object_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --max-objects 8 --max-rows-per-object 3 --output-csv docs\doc_workspace\SGCP\artifacts\grid_miss_analysis_object_aware_pg_fill_top8.csv
```

### 关键发现

- Object 438 / frame 000068：GT grid `3_0`，CAV12 有 424 点、rank=1，但 target-aware PG 调度了 CAV9，CAV9 在该 grid 为 0 点。OAPG 的 sender refinement 可在同一 RB 上把 head4 的 sender 从 CAV9 换成 CAV12。
- Object 401 / frame 000066：GT grid `2_0`，CAV4 有 891 点、rank=4，但原调度选择 CAV7，CAV7 在该 grid 只有 7 点。OAPG 可调度 CAV4 并选中 `2_0`。
- Object 350 / frame 000084：GT grid `1_-2`，CAV8 有 3371 点、rank=1，但原调度只发送 CAV2/CAV11 的 29/27 点。OAPG 可调度 CAV8 并选中该 grid。
- Object 337 / frame 000062：head 自身 CAV1 在 `0_-3` 有 1453 点，但 peer view 未被当成 target-like candidate；这暴露了 head 近身/盲区目标需要 multi-view confirmation 的问题。

### 11 帧快速实验

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-constrained --resource-allocation object_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\object_aware_pg_diverse_10ch_rho3_11f_trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\object_diag_object_aware_pg_diverse_10ch_rho3_11f.csv
```

结果：AP@0.3/0.5/0.7 = `0.74/0.69/0.30`，total payload `8,209,376` bytes，avg source CAVs `2.64`，avg selected grids `73.48`。

### 结论

OAPG 已经在 selected-frame 诊断层面修复“最佳视角 rank 高但未调度”的机制问题，但 11 帧 AP 尚未超过当前主表候选，尤其 AP@0.7 下降。该分支暂不写入主表；下一步需要 41 帧完整评估、限制每个 head 的 sender replacement 数量、加入 detector-quality gate，并对“target grid 已选中但仍漏检”的对象输出 pre-NMS boxes。

## 2026-07-17 Perception-aware potential game 主表候选

### 目的

用户要求继续按 target-grid 诊断方式调优，但最终算法必须完整、可写入论文叙事，不能表现为在旧结果上缝补。上一轮 OAPG 虽能修复若干单帧最佳视角未调度问题，但 11 帧 AP 下降，说明只追逐 object peak 会牺牲上下文和 source diversity。

### 代码改动

新增算法：

```text
opencda/core/clustering/algorithms/resource_allocation/perception_aware_potential_game.py
```

注册入口：

```text
perception_aware_potential_game
perception_aware_pg
papg
```

机制：两层 perception-aware potential-guided scheduling。第一层为每个 cluster head 分配一个高质量外部视角，保护低 IoU recall 和空间覆盖；第二层把剩余 RB 分配给 object-prototype marginal gain 最高的链路。两层使用统一 grid utility/object prototype，不使用外部 `spatial_diverse` 后处理或逐案 fallback。

### 关键命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation perception_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\papg_bh2_rho3_41f_trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\object_diag_papg_bh2_rho3_41f.csv

conda run -n opencda python -m opencda.tools.sgcp_failure_diagnostics --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --object-diagnostics-csv docs\doc_workspace\SGCP\artifacts\object_diag_papg_bh2_rho3_41f.csv --output-dir docs\doc_workspace\SGCP\artifacts\failure_diag_papg_bh2_rho3_41f

conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --upload-plan-output docs\doc_workspace\SGCP\artifacts\papg_bh2_rho3_ns3_11f_upload_plan.csv --dry-run
```

### 结果

| Method | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Scheduled links | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `perception_aware_potential_game`, 20MHz/10ch/rho3/`B_h=2` | 41 | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 410 | 97.22 |

11 帧快速 probe 中，PAPG 为 `0.76/0.73/0.34`，优于 target-aware/OAPG 的 `B_h=2` probe；late NMS 0.05/0.30 均未超过默认 0.15，说明收益主要来自调度层，而非 NMS 调参。`rho_th=4` 的 11 帧结果与 `rho_th=3` 基本一致，说明不是单纯阈值 trick。

对象级诊断：PAPG 将 target-aware PG 下 106 个 full-reference detected but SGCP-missed rows 降到 59。41 帧每帧固定调度 10 条链路，总计 410 条，说明结果没有绕过子信道预算。

### 结论

PAPG 是当前最适合写入主表和机制章节的 SGCP 候选：相比 strong selective baseline，它用更低 Mbps 获得更高 AP@0.3/AP@0.5；相比 full 20-CAV upper reference，它明确低于上界但通信量约为一半。下一步只需补真实 NS3 socket replay / 在线短回归，并将论文中 `coverage-aware` 叙事升级为 perception-aware two-layer potential scheduling。

## 2026-07-17 PAPG 真实 NS3 socket replay

### 目的

补齐 PAPG 主表候选的 NS3 request-level delivery 证据，确认 OpenCDA 指定的 10 个子信道请求在 ns-3 侧真实发送、接收并完成 application callback。

### 关键命令

ns-3 侧：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd /home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"
```

OpenCDA replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --upload-plan-output docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\upload_plan.csv --drain-seconds 0.3 --sync-timeout 30
```

解析：

```powershell
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\ns3_stdout_utf8.log --upload-plan docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\eval_utf8 --rsu-node-id 21 --max-frames 11
```

### 结果

| Metric | Value |
| --- | ---: |
| planned requests | 110 |
| application `cam_received` | 110 |
| matched callback ratio | 1.000 |
| RLC complete requests | 110 |
| RLC partial / no-RX requests | 0 / 0 |
| RLC TX / RX events | 2970 / 2970 |
| RLC drops | 0 |
| PHY decode failures | 0 |
| avg / p95 callback delay | 23.91 / 24.00 ms |

### 结论

PAPG 的 11 帧真实 NS3 replay 已通过：带宽内、无冲突的 scheduled requests 均可完成 application callback 和 RLC request delivery。PowerShell `Start-Job` 捕获的 ns-3 stdout 是 UTF-16 LE，解析前已转为 `ns3_stdout_utf8.log`；该编码问题不影响 ns-3 结果，只影响离线解析器读日志。

## 2026-07-17 PAPG 论文正文同步

### 目的

将 `C:\Workspace\icdcs-paper\SGCP\main.tex` 从旧 coverage-aware 10ch 主表口径同步到 PAPG 主候选，避免论文正文继续引用 `0.79/0.76/0.38`、57.38 Mbps 作为最终主行。

### 修改摘要

- 摘要和 introduction 将 intra-cluster scheduling 改为 perception-aware potential-guided constrained scheduling，突出 coverage layer + object-prototype target layer。
- 仿真参数主带宽从 40 MHz 改为当前主表使用的 20 MHz / 10 subchannels，20ch 作为资源敏感性设置。
- 主表新增并加粗 `SGCP (PAPG, 10 ch.) = 0.81/0.78/0.39, 62.54 Mbps`；旧 coverage-aware 10ch 保留为 ablation。
- 结果分析改为：PAPG 相比 high-budget density selective baseline 提升 AP@0.3/AP@0.5 0.01/0.02，同时减少约 15.0% upload traffic；相比 full centralized 使用 52.7% payload。
- communication efficiency 增加 PAPG 11 帧 NS3 replay 证据：110/110 application/RLC complete、2970/2970 RLC TX/RX、0 PHY failures、平均 delay 23.91 ms。
- conclusion 将 “less than half” 改成 “roughly half”，避免超过 50% payload 时主张过强。

### 验证

本机未检测到 `latexmk` 或 `pdflatex` 命令，因此本轮只做了文本级一致性检查。下一轮如需提交论文 PDF，应在具备 LaTeX 环境后编译并检查表格宽度。

## 2026-07-17 Forced-budget random baseline

### 目的

旧 RandomRA/MWS scheduler payload 只有约 18--19 Mbps，不能用于证明 SGCP 降低通信量。为公平比较，新增 deterministic random selective baseline：复用同一 coalition + inter-cluster late fusion 路径，强制使用 3 uploaded members/head 和 117 grid budget，使通信量接近 PAPG。

### 代码改动

`opencda.tools.offline_inference --selective-sharing-baseline` 新增：

```text
random
greedy_density
```

其中 `random` 使用 timestamp/head/member/grid budget 作为 deterministic seed，随机选择成员和 grid；`greedy_density` 是 `density` 的显式别名，便于论文/脚本用 greedy 名称表达强 baseline。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline random --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\random_forced_3m117g_41f_trace.csv
```

### 结果

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Forced-budget random, 3m/117g | 0.77 | 0.73 | 0.38 | 31,613,424 | 61.68 | 3.33 | 103.20 |

### 结论

PAPG `0.81/0.78/0.39` at 62.54 Mbps 明显优于 forced-budget random `0.77/0.73/0.38` at 61.68 Mbps。论文主表可以把 old RandomRA 放到 w/o-PPS 消融，把 forced-budget random 放入公平 V2V baseline。

## 2026-07-17 Rebuttal 与 PAPG 主表一致性收口

### 目的

此前 `rebuttal_draft.md` / `rebuttal_short.md` 仍使用 coverage-aware 10ch/20ch 作为主要 SGCP 结果，和当前 PAPG 主表 `0.81/0.78/0.39`、62.54 Mbps 以及 forced-budget random baseline 不一致。

### 修改

- FullPerception fairness 回复改为：full 20-CAV early 是 upper reference，PAPG 为主 SGCP 行，强 high-budget selective baseline 为公平 V2V 对比。
- Fair baseline 回复新增 forced-budget random：`0.77/0.73/0.38` at 61.68 Mbps，并说明 PAPG 在近似相同通信量下提升 `+0.04/+0.05/+0.01` AP。
- Ablation 回复保留 coverage-aware 10ch 为 predecessor/ablation，同时加入 PAPG 对象级 missed rows 从 106 降到 59。
- NS3 reliability 回复改为 PAPG 主设置 110/110 application/RLC complete、0 PHY failures。

### 结论

rebuttal、`main.tex`、`main_table_candidate.md` 当前已经使用同一条主线：PAPG 是主算法，forced-budget random 和 selective density 是公平 V2V baseline，FullPerception centralized 是 upper reference。

## 2026-07-17 Baseline fairness 文档口径清理

### 目的

`baseline_fairness.md` 和 `fullperception_baseline_revision.md` 仍有部分旧口径，把 coverage-aware 20ch 写成 payload-matched SGCP 主候选，容易和当前 PAPG 主表、forced-budget random 公平 baseline 产生冲突。

### 修改

- `baseline_fairness.md` 的分层表更新为：FullPerception/full 20-CAV early 为 centralized upper reference；SGCP PAPG 10ch 为主方法；old RandomRA/MWS 因 payload 过低只作为 w/o-PPS 诊断；forced-budget random 作为公平随机 baseline。
- `fullperception_baseline_revision.md` 的 same-budget baseline 说明从 2 members/head + 87 grid budget 调整为当前主公平设置：3 members/head + 117 grid budget，并补入 PAPG `0.81/0.78/0.39`、62.54 Mbps 与 NS3 110/110 complete。
- coverage-aware 10ch/20ch 保留为 PAPG 前身、消融或资源敏感性结果，不再作为主算法行。

### 结论

FullPerception baseline、主表候选和 rebuttal 的公平性口径已进一步对齐：主表应使用 PAPG、forced-budget random、density/communication-aware selective baseline 和 centralized FullPerception upper reference。

## 2026-07-17 Reproducibility/results PAPG 同步

### 目的

`reproducibility_manifest.md` 和 `results.md` 仍停留在 2026-07-16 的旧阶段：manifest 记录旧论文表 `22.33 Mbps`，results 首页仍把 coverage-aware 10ch/20ch 写成主候选。为避免后续写作或自动任务误读，本轮将两个核心索引文档同步到 PAPG 主线。

### 修改

- `reproducibility_manifest.md` 更新当前复现实验版本到 `23cbc0530c18a92c0545bf776b513e3def7c2baa`，将旧论文表标注为不可作为当前复现结果，并新增 PAPG 主设置命令、AP、payload、scheduled links 和对象级诊断。
- `reproducibility_manifest.md` 的 NS3 段更新为 PAPG artifact：`docs\doc_workspace\SGCP\artifacts\papg_ns3_20260717_210304\`，110/110 application callback 与 RLC complete，0 PHY failures。
- `results.md` 首页主表改为 PAPG 当前主表候选：FullPerception centralized upper reference、Full-cluster reference、forced-budget random、communication-aware/density selective baselines、SGCP PAPG 和 coverage-aware 消融/敏感性。
- `status.md` 顶部新增“当前主线快照”，优先暴露 PAPG 主结果、forced-budget random、公平性边界和 NS3 replay 状态。

### 结论

SGCP 文档工作区的入口文档现在优先呈现 PAPG 主线：`0.81/0.78/0.39` at 62.54 Mbps，forced-budget random `0.77/0.73/0.38` at 61.68 Mbps，FullPerception centralized upper reference `0.85/0.83/0.48` at 118.71 Mbps。

## 2026-07-17 Forced-budget random NS3 replay

### 目的

PAPG 主设置已有 11 帧真实 NS3 replay，但公平随机 baseline 只有离线 AP/payload 结果。为避免“PAPG 有链路验证而 random baseline 没有”的证据不对称，本轮将 `offline_ns3_replay` 扩展到 CAV-only selective-sharing baseline，并补 forced-budget random 的 request-level replay。

### 代码改动

`opencda.tools.offline_ns3_replay` 新增参数：

```text
--selective-sharing-baseline
--selective-member-budget
--selective-grid-budget
```

该分支复用 `offline_inference.assign_selective_grid_selection` 的成员/网格选择逻辑，随后把每帧可调度 request 映射到 `--num-channels` 个子信道；超出窗口的候选需求记录为 `skipped_unscheduled`，不会交给 NS3 自主调度。

### Dry-run

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --selective-sharing-baseline random --selective-member-budget 3 --selective-grid-budget 117 --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --dry-run
```

结果：11/11 帧均为 20 CAV、6 clusters、10 scheduled requests、4 skipped unscheduled demands。

### 真实 NS3 replay

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\forced_random_ns3_20260717_2304b\
```

Replay 命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --selective-sharing-baseline random --selective-member-budget 3 --selective-grid-budget 117 --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --upload-plan-output docs\doc_workspace\SGCP\artifacts\forced_random_ns3_20260717_2304b\upload_plan.csv --drain-seconds 0.3 --sync-timeout 30
```

评估命令：

```powershell
conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\forced_random_ns3_20260717_2304b\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\forced_random_ns3_20260717_2304b\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\forced_random_ns3_20260717_2304b\eval --rsu-node-id 21 --max-frames 11
```

| Metric | Value |
| --- | ---: |
| planned / scheduled requests | 110 |
| application `cam_received` | 110 |
| matched callback ratio | 1.000 |
| RLC complete requests | 110 / 110 |
| RLC TX / RX events | 2970 / 2970 |
| RLC drops | 0 |
| PHY decode failures | 0 |
| avg / p95 callback delay | 23.91 / 24.00 ms |

### 结论

forced-budget random baseline 在同一 20 MHz / 10ch scheduled-only replay 口径下也能 110/110 完成交付。因此 PAPG 相对 forced random 的 AP 增益不来自 NS3 链路失败差异，而来自 perception-aware scheduling 本身。

## 2026-07-17 FullPerception baseline code audit and implementation

### 目的

按审稿意见重新核查 FullPerception baseline。用户提醒当前可能是假设虚拟 RSU 的 FullPerception-RSU，因此本轮不再把 full 20-CAV early fusion 和 FullPerception 混写，而是进入实际代码确认是否存在算法分支，并实现显式的 FullPerception-RSU / FullPerception-Decentralized。

### 代码审计

仓库中此前没有显式命名的 `FullPerception` 算法模块或 scheduler 分支。历史结果中的 FullPerception 口径主要来自 full 20-CAV early fusion / full-sharing upper reference，而不是一个可切换 baseline。已在 `opencda.tools.offline_inference` 中新增：

```text
--selective-sharing-baseline fullperception_rsu
--selective-sharing-baseline fullperception_decentralized
--selective-sharing-baseline edgecooper
```

`fullperception_rsu` 使用全局 CAV 候选池，表示 virtual RSU/global scheduler proxy；`fullperception_decentralized` 只使用当前 cluster 内 CAV 候选，表示 V2V-only decentralized proxy；`edgecooper` first proxy 使用 complementarity minus redundancy 的 edge-assisted 选择，但尚未达到可写主表的效果。

### FullPerception-RSU proxy

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline fullperception_rsu --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\fullperception_baselines_20260717\fullperception_rsu_trace.csv
```

结果：

| Metric | Value |
| --- | ---: |
| AP@0.3 / AP@0.5 / AP@0.7 | 0.84 / 0.80 / 0.46 |
| Total payload | 56,224,736 bytes |
| Mbps | 109.71 |
| Avg. source CAVs | 4.00 |
| Avg. selected grids | 117.00 |

### FullPerception-Decentralized proxy

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline fullperception_decentralized --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\fullperception_baselines_20260717\fullperception_decentralized_trace.csv
```

结果：

| Metric | Value |
| --- | ---: |
| AP@0.3 / AP@0.5 / AP@0.7 | 0.80 / 0.76 / 0.41 |
| Total payload | 38,920,592 bytes |
| Mbps | 75.94 |
| Avg. source CAVs | 3.33 |
| Avg. selected grids | 103.20 |

### NS3 dry-run

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --selective-sharing-baseline fullperception_decentralized --selective-member-budget 3 --selective-grid-budget 117 --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --dry-run
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --selective-sharing-baseline fullperception_rsu --selective-member-budget 3 --selective-grid-budget 117 --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --dry-run
```

结果：`fullperception_decentralized` 每帧 10 scheduled requests、4 skipped unscheduled demands；`fullperception_rsu` 每帧 10 scheduled requests、8 skipped unscheduled demands。说明两个显式 baseline 都可以进入 scheduled-only NS3 replay 口径，下一步应优先对 `fullperception_decentralized` 做 11-frame true NS3 replay。

### Baseline search

已查找并整理 EdgeCooper、Where2comm、PACP、What2comm、CoBEVT、V2VNet、RACooper 等候选，新增 `baseline_reproduction_plan.md`。EdgeCooper 本地 PDF 指向 edge server 聚合、complementarity/redundancy-aware raw LiDAR sharing、minimum-cost flow 与二维图着色冲突处理；当前 first proxy 3-frame smoke test 仅 `0.54/0.46/0.15`，说明 naive complementarity 会偏向少数高密度车辆，需改成 blind-spot-aware edge scheduling。

### 结论

FullPerception 口径已拆开：

- full 20-CAV early fusion：AP upper reference，`0.85/0.83/0.48`，118.71 Mbps。
- FullPerception-RSU proxy：RSU/edge-assisted reference，`0.84/0.80/0.46`，109.71 Mbps。
- FullPerception-Decentralized proxy：V2V-only strong baseline，`0.80/0.76/0.41`，75.94 Mbps。

PAPG 当前仍是主方法：`0.81/0.78/0.39`，62.54 Mbps。相对 FullPerception-Decentralized，它以更低 payload 获得更高 AP@0.3/AP@0.5，但 AP@0.7 仍略低；后续应继续优化 PAPG 的高 IoU 定位，同时补 EdgeCooper/Where2comm/PACP proxy。

## 2026-07-17 EdgeCooper-style proxy refinement

### 目的

上一轮 EdgeCooper first proxy 使用全局 candidate pool 和 complementarity-minus-redundancy 评分，但 3-frame smoke test 仅 `0.54/0.46/0.15`。本轮定位问题并改为更贴近 EdgeCooper 叙事的 blind-spot-aware edge scheduling proxy。

### 问题定位

naive proxy 的 complementarity 是相对于“已选 sender 覆盖”计算的，不是相对于 receiver/head 的盲区计算的；同时当 receiver blind candidate 为空时会 fallback 到 sender 全视野。这导致多个 cluster head 反复选择 CAV 14/18/10 等全局高密度 sender，通信量很高但融合视角单一。

### 代码改动

`opencda.tools.offline_inference` 中新增/修改：

```text
receiver_blind_grids()
edgecooper_candidate_grids()
edgecooper_grid_score()
select_edgecooper_members()
select_edgecooper_grids()
```

新 proxy 的语义：

- candidate grid 限定为 sender 可观测且 receiver/head 低密度的 blind grids；
- member utility 使用 blind-grid complementarity、selected-sender redundancy 和 distance/network cost；
- grid selection 同步使用 EdgeCooper 专属 blind-grid utility；
- 不再把 blind candidate 为空的 sender fallback 到全视野上传。

### Smoke test

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --selective-sharing-baseline edgecooper --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20
```

结果从 naive proxy 的 `0.42/0.36/0.16` 修复到 `0.76/0.72/0.33`，sender 不再被少数车辆垄断。

### 41-frame run

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\edgecooper_proxy_20260717\
```

命令：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline edgecooper --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\edgecooper_proxy_20260717\edgecooper_trace.csv
```

| Metric | Value |
| --- | ---: |
| AP@0.3 / AP@0.5 / AP@0.7 | 0.75 / 0.70 / 0.32 |
| Total payload | 56,134,048 bytes |
| Mbps | 109.53 |
| Avg. source CAVs | 4.00 |
| Avg. selected grids | 117.00 |

### 结论

EdgeCooper-style proxy 已可复现，但不是强 baseline：它的通信量接近 FullPerception-RSU proxy，AP 却明显低于 PAPG 和 FullPerception-RSU。当前结果应写为 preliminary proxy，而不是严格复现 EdgeCooper 原论文。下一步若继续推进 EdgeCooper，应实现 minimum-cost-flow/global-assignment 风格调度，并加入 conflict/coloring 或 sender capacity 约束。

## 2026-07-18 FullPerception PCS code correction

### 目的

用户提醒此前 FullPerception 实现应对应 `opencda/core/clustering/algorithms/resource_allocation/pcs.py`，论文参考 `FullPerception_Network-level_Collaborative_Perception_for_Eliminating_Vehicular_Blind_Spots.pdf`。本轮重新核查代码和论文，纠正上一轮“仓库没有显式 FullPerception 算法分支”的不准确说法。

### 论文核查

本地 PDF 抽取结果显示 FullPerception 的核心机制包括：

- blind spot 区域定义和 grid-level mAP utility；
- link weight `w(L_i,j,k)`；
- required subchannels `c(L_i,j,k)`；
- Class A / Class B conflict graph；
- Algorithm 1 Weight Splitting；
- Algorithm 2 Resource Allocation；
- Algorithm 3 PCS；
- MWS 和 RS 作为 heuristic scheduling baselines。

### 代码核查

`pcs.py` 确实对应上述 PCS：

| Paper Component | Code |
| --- | --- |
| blind spots | `_get_vehicle_blind_spots()` |
| potential links | `_generate_potential_links()` |
| grid mAP / link utility | `_precompute_grid_mAP()`, `_calculate_link_utilities()` |
| conflicts | `_build_conflict_graph()` |
| weight splitting | `_weight_splitting()` |
| resource allocation | `_resource_allocation()` |
| recursive PCS | `_pcs_recursion()` |

`mws.py` 和 `random_ra.py` 继承 `PCS`，对应 FullPerception 论文中的 MWS / RS baseline。

### 代码改动

- `builder.py` 新增 alias：`fullperception`、`fullperception_pcs`、`fullperception_rsu_pcs`、`fullperception_mws`、`fullperception_random`。
- `offline_inference.apply_resource_overrides()` 现在会把 `--num-channels` 同步到 PCS/MWS/RS 的 `lambda_subchannels`。

### 当前实验

3-frame smoke：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --sgcp-constrained --resource-allocation pcs --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20
```

结果：`0.36/0.31/0.14`，785,312 bytes。

41-frame run：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation pcs --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\fullperception_pcs_20260718\pcs_trace.csv
```

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\fullperception_pcs_20260718\
```

| Metric | Value |
| --- | ---: |
| AP@0.3 / AP@0.5 / AP@0.7 | 0.44 / 0.39 / 0.17 |
| Total payload | 12,684,880 bytes |
| Mbps | 24.75 |
| Avg. source CAVs | 1.66 |
| Avg. selected grids | 630.66 |

### 当前判断

PCS 是内置 FullPerception 实现，但当前工程结果明显 under-schedule，不能直接代表论文 FullPerception 的强 baseline。主要风险：

- `_get_link_required_subchannels()` 直接 `return 1`，后续基于 feature size/channel capacity 的 `c(q)` 计算是死代码。
- `--bandwidth-mhz` 尚未影响 PCS 的 required subchannels。
- 同一 sender-receiver 的多个 blind spot 最终折叠到 `(sender, receiver) -> start_subchannel`，可能丢失多 blind-spot link 的资源区分。
- 当前 late-fusion evaluation 仍沿用 SGCP cluster-head receiver path；PCS 原论文是 base-station/RSU 全局调度，需要进一步对齐接收/融合口径。

下一步应先修复/校准 `pcs.py`，再把它作为 FullPerception 主 baseline；上一轮新增的 `fullperception_rsu` 和 `fullperception_decentralized` 应改写为 proxy/diagnostic，不再抢占 FullPerception 正名。

## 2026-07-18 EdgeCooper global assignment proxy

### 目的

上一版 `edgecooper` 只做逐 receiver 的 blind-spot complementarity 贪心选择，41 帧结果为 `0.75/0.70/0.32`、56,134,048 bytes / 109.53 Mbps，既不强，也没有体现 EdgeCooper 论文中 edge-side 全局调度的核心优势。本轮新增 `edgecooper_global`，作为更接近 edge/virtual-RSU assisted global assignment 的 proxy。

### 代码改动

- `opencda.tools.offline_inference --selective-sharing-baseline` 新增 `edgecooper_global`。
- `edgecooper_global` 使用全局 CAV 候选池，但加入 35 m V2V feasibility gate，避免调度明显不可达的 sender。
- sender 选择继承 blind-spot complementarity / redundancy utility，并新增 global sender-load penalty 与 sender capacity，避免多个 receiver 重复争抢同一高密度 sender。
- grid selection 仍使用 EdgeCooper-style blind-grid utility：优先发送 sender 可观测、receiver/head 低密度、能补盲区的 grid。
- `offline_ns3_replay` 对 `edgecooper_global` 初始化同一全局 sender-load 状态，使离线 AP 和 NS3 request replay 使用一致选择逻辑。

### 命令

41-frame offline AP：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline edgecooper_global --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\edgecooper_global_35m_probe_20260718\edgecooper_global_35m_trace_41f.csv
```

11-frame true NS3 replay：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --max-frames 11 --selective-sharing-baseline edgecooper_global --selective-member-budget 3 --selective-grid-budget 117 --num-channels 10 --bandwidth-mhz 20 --ns3-host 127.0.0.1 --ns3-port 9999 --output-dir docs\doc_workspace\SGCP\artifacts\edgecooper_global_35m_probe_ns3_20260718 --real-ns3 --analyze
```

### 结果

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids | NS3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EdgeCooper blind-spot proxy | 0.75 | 0.70 | 0.32 | 56,134,048 | 109.53 | 4.00 | 117.00 | Not replayed |
| EdgeCooper-global 35 m proxy | 0.81 | 0.77 | 0.42 | 38,223,408 | 74.58 | 3.26 | 98.75 | 73/110 complete |

NS3 11-frame diagnostics for `edgecooper_global`：

| Metric | Value |
| --- | ---: |
| Application callbacks | 73 / 110 |
| RLC complete requests | 73 / 110 |
| RLC TX/RX events | 2970 / 1971 |
| RLC `tx_no_rx` requests | 37 |
| PHY decode failures | 0 |

Artifact：

```text
docs\doc_workspace\SGCP\artifacts\edgecooper_global_35m_probe_20260718\
docs\doc_workspace\SGCP\artifacts\edgecooper_global_35m_probe_ns3_20260718\
```

### 当前判断

`edgecooper_global` 显著强于第一版 blind-spot proxy，离线 AP@0.7 达到 `0.42`，接近 full-cluster reference，并且 payload 从 109.53 Mbps 降到 74.58 Mbps。这个结果说明 edge/global assignment 方向有价值，也可作为审稿意见中“补更强 baseline”的重要材料。

但该结果不能直接替代 PAPG 主线：`edgecooper_global` 属于 virtual edge/RSU-assisted baseline，使用全局候选池；且真实 NS3 replay 只有 73/110 request complete，而 PAPG 在同一 11 帧口径下为 110/110 complete、0 PHY failures、62.54 Mbps。论文写作应把它放入 RSU/edge-assisted diagnostic baseline 或补充实验表，而不是 V2V-only 公平主表。下一步若要严格复现 EdgeCooper，应继续实现 MCF/conflict-coloring 或 deadline-aware global assignment，使 high-AP proxy 也具备 request-level delivery guarantee。

## 2026-07-18 EdgeCooper half-duplex diagnosis and repair

### 目的

上一轮 `edgecooper_global` 41 帧离线 AP 已经较强，但 11 帧真实 NS3 replay 只有 73/110 request complete。需要定位这是距离/带宽问题、NS3 bug，还是调度协议缺少链路约束。

### 失败诊断

对 `docs\doc_workspace\SGCP\artifacts\edgecooper_global_35m_probe_ns3_20260718\eval\rlc_by_request.csv` 聚合后发现：

- 失败为 37 个 `rlc_tx_no_rx`，没有 partial request，PHY decode failures 为 0。
- 失败集中在 `000068/000070/000072/000076/000078/000080`，这些帧均为 10 个 request 中只有 4 个 complete。
- 失败 target 高度集中在 1 和 4：target 1 有 18 个失败，target 4 有 15 个失败。
- 失败链路距离多在 7.8-32.6 m 内，例如 `12->4` 为 7.78 m，`8->1` 为 15.20 m，`4->1` 为 32.59 m；因此不是 35 m feasibility gate 失效。
- 失败帧存在半双工 role conflict：同一 100 ms slot 内 target 1 和 target 4 同时接收、同时作为 sender，例如 `4->1` 与 `1->4` 同时出现在 upload plan 中。

结论：旧 `edgecooper_global` 的 NS3 failure 主要不是 PHY collision 或距离超限，而是 global edge assignment 没有约束 CAV 的半双工角色。PAPG/SGCP 天然只让 cluster members 上传到 cluster head，因此较少出现 cluster head 同时作为 sender 的问题；EdgeCooper global candidate pool 会引入该冲突。

### 代码改动

- 新增 selective baseline：`edgecooper_global_hd`。
- 继承 `edgecooper_global` 的全局 sender-load balancing、35 m V2V feasibility gate 和 blind-spot complementarity grid selection。
- 在每帧所有 cluster head receiver 集合确定后，设置 `world._edgecooper_global_receiver_ids`。
- `edgecooper_global_hd` 在候选 sender 阶段排除本帧所有 receiver，保证同一 slot 内 cluster-head receiver 不会同时作为 sender。
- `offline_inference` 和 `offline_ns3_replay` 使用同一 receiver exclusion 逻辑。

### 验证命令

11-frame dry-run half-duplex audit：

```powershell
conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --max-frames 11 --selective-sharing-baseline edgecooper_global_hd --selective-member-budget 3 --selective-grid-budget 117 --num-channels 10 --bandwidth-mhz 20 --dry-run --upload-plan-output docs\doc_workspace\SGCP\artifacts\edgecooper_global_hd_dryrun_20260718\upload_plan.csv
```

审计结果：110 requests，half-duplex violations = 0。

41-frame offline AP：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline edgecooper_global_hd --selective-member-budget 3 --selective-grid-budget 117 --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\edgecooper_global_hd_20260718\edgecooper_global_hd_trace_41f.csv
```

11-frame true NS3 replay：

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd /home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'"

conda run -n opencda python -m opencda.tools.offline_ns3_replay --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --max-frames 11 --selective-sharing-baseline edgecooper_global_hd --selective-member-budget 3 --selective-grid-budget 117 --num-channels 10 --bandwidth-mhz 20 --ns3-host 127.0.0.1 --upload-plan-output docs\doc_workspace\SGCP\artifacts\edgecooper_global_hd_ns3_20260718\upload_plan.csv

conda run -n opencda python -m opencda.tools.ns3_log_eval --ns3-stdout docs\doc_workspace\SGCP\artifacts\edgecooper_global_hd_ns3_20260718\ns3_stdout.log --upload-plan docs\doc_workspace\SGCP\artifacts\edgecooper_global_hd_ns3_20260718\upload_plan.csv --output-dir docs\doc_workspace\SGCP\artifacts\edgecooper_global_hd_ns3_20260718\eval --max-frames 11
```

### 结果

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids | NS3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EdgeCooper-global | 0.81 | 0.77 | 0.42 | 38,223,408 | 74.58 | 3.26 | 98.75 | 73/110 complete |
| EdgeCooper-global-HD | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 | 3.00 | 89.02 | 110/110 complete |

NS3 11-frame diagnostics for `edgecooper_global_hd`：

| Metric | Value |
| --- | ---: |
| Application callbacks | 110 / 110 |
| RLC complete requests | 110 / 110 |
| RLC TX/RX events | 2970 / 2970 |
| PHY decode failures | 0 |
| Avg / p95 callback delay | 23.91 / 24.00 ms |

### 当前判断

半双工约束把 EdgeCooper global proxy 从“离线强但 NS3 不完整”修复为“离线强且 NS3 完整”。这条 baseline 很有价值，但也会改变论文叙事压力：它是 virtual edge/RSU-assisted global scheduler，信息条件强于 PAPG；如果和 PAPG 混在同一公平主表，PAPG 不再是 AP@0.7 最高。因此后续论文表格应分层呈现 RSU/edge-assisted 与 V2V-only decentralized baselines，或者继续改造 PAPG 以提升 AP@0.7。

## 2026-07-18 PAPG high-IoU pressure follow-up

### 目的

EdgeCooper-global-HD 达到 `0.81/0.78/0.42`、65.40 Mbps 且 110/110 NS3 complete，对 PAPG 的 AP@0.7 形成压力。本轮先测试不改代码的 PAPG `head_rb_budget=3`，判断是否能通过更宽松的 per-head RB 上限提升高 IoU。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation perception_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 3 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\papg_bh3_rho3_10ch_20260718\papg_bh3_trace_41f.csv
```

### 结果

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG, `B_h=2` | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 3.33 | TBD |
| PAPG, `B_h=3` | 0.80 | 0.78 | 0.40 | 32,051,792 | 62.54 | 2.67 | 97.09 |

结论：简单提高 `B_h` 不是有效修复。`B_h=3` 在同样 10-channel 总预算下没有增加有效多视角覆盖，反而使 avg source CAVs 降至 2.67，只把更多 grid 压到更少的 source 上；AP@0.7 仅从 0.39 到 0.40，仍低于 EdgeCooper-HD 的 0.42。下一步若继续提升 PAPG，应针对高质量 source / target-grid coverage 做机制级改造，而不是继续调 per-head RB 上限。

### 论文同步

已更新 `C:\Workspace\icdcs-paper\SGCP\main.tex`：

- baseline 列表新增 `EdgeCooper-HD`，明确其为 virtual edge-assisted reference。
- 主表新增 `EdgeCooper-HD (edge-assisted) = 0.81/0.78/0.42, 65.40 Mbps`。
- 正文说明 EdgeCooper-HD 使用 global edge-side information，不作为 fully decentralized SGCP 的公平 V2V baseline；PAPG 与 V2V baselines 的比较仍限定在 RSU-free V2V 设置下。

## 2026-07-18 Balanced PAPG source-diversity probe

### 目的

针对 PAPG / `B_h=3` 中源车多样性没有提升的问题，新增独立算法分支 `balanced_perception_aware_potential_game`（alias: `bpapg`）。该分支不修改原 PAPG，而是在同一 coverage layer + target layer 势函数中加入 source-diversity marginal term，并进一步试验跨帧 source-history credit，用于验证“欠服务源车保护”是否能修复高 IoU 漏检。

### 代码

- `opencda/core/clustering/algorithms/resource_allocation/balanced_perception_aware_potential_game.py`
- `opencda/core/clustering/algorithms/resource_allocation/builder.py`

### 命令与结果

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation balanced_perception_aware_potential_game --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\bpapg_rho3_10ch_20260718\bpapg_trace_41f.csv
```

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPAPG source-balanced | 41 | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 2.67 | 97.22 |
| BPAPG + source-history credit | 11 | 0.75 | 0.71 | 0.33 | 8,601,424 | 62.56 | 2.67 | 95.12 |

Trace coverage summary confirms that the first BPAPG variant kept the same per-CAV upload distribution as PAPG: 41-frame avg fused CAVs = 16/20, avg uploaded CAVs = 10/20, avg unscheduled members = 4/20, and per-CAV uploaded frame counts were unchanged. The history-credit variant did change selected senders in early frames, but AP dropped, indicating that naive fairness/rotation can replace stable high-quality views with lower-quality context.

### 结论

BPAPG is a useful negative branch, not a new main method. The high-IoU fix should not be a generic under-served-CAV rotation. It needs detector-quality / target-quality gating: only protect a low-frequency source if it covers a diagnosed target grid with comparable object-prototype quality, otherwise source fairness hurts AP.

## 2026-07-18 Quality-gated and head-urgent PAPG probes

### 目的

上一轮 BPAPG 说明普通 source fairness 无效。本轮继续做两个更收敛的机制分支：

- `quality_gated_perception_aware_potential_game` / `qgpapg`：仅当低频 source 的 object/peak/coverage 接近本簇最佳候选时才给 source-history credit。
- `head_urgent_perception_aware_potential_game` / `hupapg`：完全去掉 source-history，改成 receiver/head 侧 target urgency，把额外 RB 倾向分配给 coverage layer 后仍有强 target-prototype candidate 的 head。

### 代码

- `opencda/core/clustering/algorithms/resource_allocation/quality_gated_perception_aware_potential_game.py`
- `opencda/core/clustering/algorithms/resource_allocation/head_urgent_perception_aware_potential_game.py`
- `opencda/core/clustering/algorithms/resource_allocation/builder.py`

### 结果

| Variant | Frames | `B_h` | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. source CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QG-PAPG | 11 | 2 | 0.75 | 0.72 | 0.33 | 8,602,400 | 62.56 | 2.67 | 95.23 |
| HU-PAPG | 11 | 2 | 0.76 | 0.73 | 0.34 | 8,598,224 | 62.53 | 2.67 | 95.24 |
| HU-PAPG | 41 | 2 | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 2.67 | 97.22 |
| HU-PAPG | 11 | 3 | 0.76 | 0.73 | 0.34 | 8,598,224 | 62.53 | 2.67 | 95.24 |

### 结论

QG-PAPG 仍然会因 source-history credit 替换稳定视角而伤 AP；说明即便有 object-quality gate，跨帧 source fairness 也不适合作为主线。HU-PAPG 去掉 source history 后恢复到 PAPG 主行水平，但没有突破 AP@0.7，`B_h=3` 的短评估也没有改变有效链路集合。下一步不应继续调 source/head fairness 系数，而应进入 detector/pre-NMS 级诊断：确认“nearest head 已收到 dense target grid 但无 final box”的 35 行里，问题发生在 detector 无框、NMS 抑制，还是 grid 内点云没有落到目标实体。

## 2026-07-18 Head-local detector box diagnostics

### 目的

针对 failure diagnostics 中的 secondary bucket：nearest head 已经收到 dense target grid points，但 final late-fused result 仍然漏检。本轮新增 per-head box diagnostic，导出每个 cluster head 在 inter-cluster late fusion 前的 detector 输出 best IoU/score，判断问题发生在 head-local detector、late fusion/NMS，还是坐标/匹配链路。

### 代码

```text
opencda/tools/sgcp_head_box_diagnostics.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.sgcp_head_box_diagnostics --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --failure-gt-csv docs\doc_workspace\SGCP\artifacts\failure_diag_papg_bh2_rho3_41f\gt_objects.csv --output-csv docs\doc_workspace\SGCP\artifacts\head_box_diag_papg_dense_20260718\head_box_diag_dense_top40.csv --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --max-rows 40 --min-nearest-head-points 30
```

### 结果

Dense-miss top40 统计：

| Metric | Value |
| --- | ---: |
| Dense missed objects analyzed | 40 |
| Nearest-head pre-late-fusion matched @0.5 | 0 / 40 |
| Any-head pre-late-fusion matched @0.5 | 0 / 40 |
| Late-fused matched @0.5 | 0 / 40 |
| Full-reference matched @0.5 | 40 / 40 |
| Best any-head IoU mean / max | 0.0000 / 0.0000 |

Representative rows:

| Timestamp | Object | Nearest head | Nearest-head points | Head pred boxes | Head best IoU | Full-reference IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 000060 | 419 | 11 | 148 | 21 | 0.000000 | 0.889081 |
| 000062 | 444 | 1 | 46 | 23 | 0.000000 | 0.833448 |
| 000064 | 337 | 1 | 119 | 21 | 0.000000 | 0.818634 |
| 000066 | 337 | 1 | 119 | 21 | 0.000000 | 0.919990 |
| 000066 | 406 | 12 | 87 | 13 | 0.000000 | 0.747779 |

### 结论

这批 dense target-grid miss 不是 inter-cluster late fusion/NMS 抑制已有正确框，也不是单纯 receiver 没有出任何框；nearest head 有正常数量的 detector boxes，但目标 best IoU 仍为 0。下一步算法不应继续只按 grid id/point count 调度，而要进入 object-level point association：确认 grid 内上传点是否真的落在目标 3D box 内，并设计 box-aware/instance-aware grid scoring 或 point selection。

## 2026-07-18 Object-level point association

### 目的

上一轮 head-local box diagnostics 说明 dense-miss top40 中 nearest head / any head / late-fused 都没有与 GT 重叠的检测框，但 full-reference 全部可检出。本轮进一步检查：这些被调度到 nearest head 的点是否真的落入 GT object box，而不仅仅是落在同一个 coarse grid 内。

### 代码

```text
opencda/tools/sgcp_object_point_association.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.sgcp_object_point_association --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --failure-gt-csv docs\doc_workspace\SGCP\artifacts\failure_diag_papg_bh2_rho3_41f\gt_objects.csv --output-csv docs\doc_workspace\SGCP\artifacts\object_point_assoc_papg_dense_20260718\object_point_assoc_dense_top40.csv --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --max-rows 40 --min-nearest-head-points 30 --margins '0,1,2,4'
```

### 结果

Dense-miss top40 SGCP object-support 统计：

| Metric | Min | Avg | Max | Zero rows |
| --- | ---: | ---: | ---: | ---: |
| Receiver points inside GT BEV box | 0 | 64.30 | 327 | 3 / 40 |
| Uploaded points inside GT BEV box | 0 | 34.17 | 350 | 1 / 40 |
| Total SGCP points inside GT BEV box | 0 | 98.48 | 350 | 1 / 40 |
| Total SGCP points inside GT BEV box + 1 m | 1 | 142.72 | 653 | 0 / 40 |
| Total SGCP points inside GT BEV box + 2 m | 3 | 285.18 | 1612 | 0 / 40 |
| Nearest CAV raw points inside GT BEV box | 0 | 107.82 | 344 | 5 / 40 |
| Nearest CAV uploaded points inside GT BEV box | 0 | 20.68 | 344 | 31 / 40 |

Full-reference 对照：

| Metric | Min | Avg | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| Full-reference points inside exact GT BEV box | 24 | 164.98 | 386 | 0 / 40 rows are empty |
| SGCP / full-reference exact-box point ratio | 0.00 | 0.62 | 0.97 | 18 / 40 rows are below 0.5 |
| Full-reference points inside GT BEV box + 2 m | 46 | 433.70 | 1655 | 0 / 40 rows are empty |
| SGCP / full-reference box+2m point ratio | 0.00 | 0.64 | 0.97 | 16 / 40 rows are below 0.5 |
| Best single raw CAV exact-box points | 9 | 109.00 | 344 | nearest CAV is rank 1 in 34 / 40 rows |

Lowest exact-box support examples include object `419` at frame `000060` with 0 exact-box points but 148 nearest-head coarse-grid points, and object `377` at frames `000076/000078` with only 4/5 exact-box points.

### 结论

这批 dense-miss 不能解释为“目标附近完全没有点”。在 exact BEV box 内，39/40 行已有 SGCP 支撑点；扩展到 2 m 邻域后，40/40 行都有点。但 nearest CAV 的 object-box 点大多没有被直接上传到 nearest head：31/40 行的 `nearest_cav_uploaded_box_points_m0p0=0`。Full-reference 对照进一步说明，SGCP 虽然平均保留了约 62% exact-box 点，但 18/40 行低于 50%，且 nearest CAV 在 34/40 行本来就是最佳 raw object-support source。因此当前失败更像是两类问题叠加：

- coarse grid 覆盖不等价于目标实例级覆盖，部分 grid 点并不形成足够完整的目标形状；
- 即使有目标附近点，SGCP constrained input 的多视角密度/形状上下文仍不足以让 head-local detector 出框。

下一步应把调度 utility 从 grid-density 进一步推进到 instance-support / box-support aware：在不增加 RB 数的前提下，优先保护最佳实例视角到相关 cluster head 的传输，再由 target layer 选择能补全目标形状的 grid。

## 2026-07-18 Instance-support PPS probe

### 目的

基于 object-level point association 的结论，新增 `instance_support_potential_game`。该分支不使用 GT box，而是在 PAPG 的 coverage layer / target layer 内加入实例支撑 proxy：紧凑高密度 grid component、head weak-view gain、unique-best-view gain。目标是在同一 20 MHz / 10 ch / `rho_th=3` / `B_h=2` 预算下，让最佳实例视角更容易成为 sender，并优先选择能补全目标形状的 grid。

### 代码

```text
opencda/core/clustering/algorithms/resource_allocation/instance_support_potential_game.py
opencda/core/clustering/algorithms/resource_allocation/builder.py
```

### 命令

11 帧快速 probe：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --resource-allocation instance_support_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\ispg_11f_20260718\trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\ispg_11f_20260718\objects.csv
```

41 帧完整 probe：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --resource-allocation instance_support_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\ispg_41f_20260718\trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\ispg_41f_20260718\objects.csv
```

### 结果

| Variant | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. fused CAVs | Avg. uploaded CAVs | Avg. selected grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ISPG | 11 | 0.76 | 0.73 | 0.34 | 8,608,016 | 62.60 | 16.00 / 20 | 10.00 / 20 | 95.55 |
| ISPG | 41 | 0.80 | 0.78 | 0.39 | 32,046,336 | 62.53 | 16.00 / 20 | 10.00 / 20 | 97.39 |
| PAPG main reference | 41 | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 16.00 / 20 | 10.00 / 20 | 97.22 |

### 结论

ISPG 是中性/负面 probe：它保持了 PAPG 的 AP@0.5/AP@0.7 和通信量，但 AP@0.3 低 0.01，且覆盖结构没有变化，仍为每帧 16/20 fused、10/20 uploaded、4/20 unscheduled。这说明实例支撑 proxy 只放进簇内 sender/grid utility 不足以突破当前瓶颈。下一步应把实例支撑推进到更高一层：跨簇 receiver assignment / target-to-head routing，确保最佳实例视角能送到真正负责该目标的 head，而不是只在现有 cluster 内重排 grid。

## 2026-07-18 Cross-cluster instance routing probe

### 目的

上一轮 ISPG 说明簇内 sender/grid utility 不足。本轮新增 `cross_cluster_instance_support_potential_game` (`ccispg`)：在同一 20 MHz / 10 ch / `rho_th=3` / `B_h=2` 全局预算下，允许非 cluster-head CAV 作为外部 sender，把强实例支撑视角跨簇送给更相关的 cluster head。机制上仍保留 PAPG 的 coverage layer / target layer。

### 代码

```text
opencda/core/clustering/algorithms/resource_allocation/cross_cluster_instance_support_potential_game.py
opencda/core/clustering/algorithms/resource_allocation/builder.py
```

### 结果

| Variant | Frames | External links | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. fused CAVs | Avg. uploaded CAVs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive CCISPG | 11 | 104 / 110 | 0.68 | 0.64 | 0.37 | 8,663,216 | 62.99 | 16.00 / 20 | 10.00 / 20 |
| Layered CCISPG | 11 | 44 / 110 | 0.75 | 0.71 | 0.33 | 8,605,088 | 62.58 | 16.00 / 20 | 10.00 / 20 |
| Cap1 CCISPG | 11 | 11 / 110 | 0.75 | 0.72 | 0.33 | 8,614,848 | 62.65 | 16.00 / 20 | 10.00 / 20 |
| PAPG short reference | 11 | 0 / 110 external | 0.76 | 0.73 | 0.34 | about 8.60M | about 62.5 | 16.00 / 20 | 10.00 / 20 |

### 结论

跨簇 routing 能改变高 IoU 定位：naive CCISPG 的 AP@0.7 到 0.37，但 AP@0.3/0.5 明显下降，原因是 94.55% 上传链路变成 external，稳定 coverage layer 被破坏。Layered 和 Cap1 版本把 external 限制到 target layer 或每帧 1 条，AP@0.3/0.5 仍未恢复到 PAPG，说明仅靠在线 density/object proxy 自动触发跨簇路由不够稳。

下一步不应继续放宽 cross-cluster routing，而应改成 diagnostic-triggered / persistent-target-triggered routing：只有当目标区域在历史或当前诊断中满足“full-reference 可检、PAPG 漏检、nearest/best raw source 不在相关 head 输入中”时，才用 1 条 target-layer RB 做跨簇实例补偿。这样可以保留论文中的势博弈叙事，同时避免全局外部 sender 替换稳定覆盖源。

## 2026-07-18 Diagnostic routing-hint oracle probe

### 目的

上一轮 CCISPG 表明自动 external sender 过强。本轮新增 `offline_inference --sgcp-routing-hints-csv`，只作为 oracle/debug probe：读取 `sgcp_object_point_association` 的 dense-miss top40 诊断 CSV，在同一 20 MHz / 10 ch / `rho_th=3` / `B_h=2` 预算下，每帧最多替换 1 条既有 scheduled route 或重排 1 个已调度 sender 的 grids，优先把 best raw object-support source 的 object grid/邻域送到 nearest head。

### 代码

```text
opencda/tools/offline_inference.py
```

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-routing-hints-csv docs\doc_workspace\SGCP\artifacts\object_point_assoc_papg_dense_20260718\object_point_assoc_dense_top40.csv --sgcp-routing-hints-max-per-frame 1 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\routing_hints_papg_11f_20260718\trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\routing_hints_papg_11f_20260718\objects.csv
```

### 结果

| Variant | Frames | Hint replacements | External links | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG + diagnostic routing hints | 11 | 9 frame-level replacements | 2 / 110 | 0.75 | 0.71 | 0.35 | 8,521,936 | 61.98 |
| PAPG short reference | 11 | 0 | 0 / 110 | 0.76 | 0.73 | 0.34 | about 8.60M | about 62.5 |

Object diagnostics in this run produced 90 `full_detected_method_missed` rows, higher than the earlier PAPG short-run failure count used in the dense-miss branch. The probe reduced payload slightly and improved AP@0.7 by about 0.01, but AP@0.3/AP@0.5 dropped.

### 结论

即使用 GT/full-reference 诊断生成 routing hints，“把 best raw object-support source 的 object grid 送到 nearest head”也不是充分条件。它可以微弱改善高 IoU，但会破坏 detector 所需的上下文或低阈值召回。下一步机制不应只追 object-box 点数或 object grid 命中，而应引入 detector-benefit proxy：例如预测某次替换是否保持 receiver fused GT/pred context、是否保护稳定 coverage source、是否提升 head-local objectness，而不是只按 full-reference support gap 触发。

### Context-preserving merge follow-up

为排除“整组替换 selected grids 破坏上下文”的可能，本轮将已调度 sender 的 hint 行为从 whole-list replacement 改为 merge replacement：最多插入 3 个 object-neighborhood grids，其余位置保留该 sender 原 selected grids 中密度最高的上下文 grids。

| Variant | Frames | Hint replacements | External links | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG + merged routing hints | 11 | 9 frame-level replacements | 2 / 110 | 0.75 | 0.71 | 0.35 | 8,563,440 | 62.28 |

结果与 whole-list hint 基本一致，甚至 `full_detected_method_missed` 从 90 增至 91。结论进一步收窄：问题不是简单由 hint 替换丢失上下文造成；coarse object-grid / object-box support 本身仍不足以预测 detector 是否会出框。下一步若继续算法改造，应直接引入 head-local detector-benefit proxy，例如快速评估替换前后的 head-local pred/GT proxy、objectness proxy 或多视角形状完整性 proxy；否则应停止在 routing probe 上继续扩张，把当前 PAPG 主线作为论文可写结果。

## 2026-07-18 Detector-benefit post-hoc comparison

### 目的

为避免继续盲目调 routing trigger，本轮重跑 11 帧 PAPG 无 hint 基线，使用同一 object diagnostics 口径与 merged routing hints 逐 GT 对比，统计 routing hint 到底修复了哪些目标、又破坏了哪些目标。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\papg_11f_object_compare_20260718\trace.csv --object-diagnostics-output docs\doc_workspace\SGCP\artifacts\papg_11f_object_compare_20260718\objects.csv
```

### 结果

PAPG 11 帧基线：AP `0.76/0.73/0.34`，payload `8,598,224 bytes`。

Merged routing hints vs PAPG 逐 GT 对比：

| Metric | Value |
| --- | ---: |
| Common GT rows | 782 |
| Same hit | 582 |
| Same miss | 181 |
| Hint gained GTs | 4 |
| Hint lost GTs | 15 |
| PAPG full-detected method-missed rows | 82 |
| Hint full-detected method-missed rows | 91 |

Gained GTs:

| Timestamp | Object | PAPG IoU | Hint IoU | Full IoU |
| --- | ---: | ---: | ---: | ---: |
| 000060 | 417 | 0.000000 | 0.730017 | 0.756145 |
| 000060 | 419 | 0.000000 | 0.835617 | 0.889417 |
| 000060 | 443 | 0.000000 | 0.771785 | 0.875052 |
| 000078 | 377 | 0.000000 | 0.861020 | 0.845149 |

Lost GTs concentrate on persistent/context-sensitive objects: object `337` lost 3 rows, object `432` lost 3 rows, object `400` lost 2 rows, plus several single-row losses. Receiver-level trace showed total pred/GT counts changed little (`pred_delta_sum=-1`, `gt_delta_sum=-1`), so aggregate pred/GT count is not enough to predict benefit.

### 结论

Routing hints do fix some diagnosed objects, proving the underlying target-to-head idea can work. But the cost is larger: 4 gained GTs vs 15 lost GTs. The detector-benefit trigger therefore needs to protect the already-covered object set, not merely preserve total pred/GT count or selected-grid density. A usable non-oracle mechanism would need a local objectness/proposal-level estimate before replacement: only replace if the candidate route is likely to add a new object without suppressing currently covered object prototypes.

## 2026-07-18 Routing-probe paper-boundary consolidation

### 目的

本轮不新增实验，目标是把近期 ISPG、CCISPG、routing-hint 和 detector-benefit post-hoc 结果从“算法继续试探”收束为论文主表和 rebuttal 的边界条件，避免把负面/半 oracle 探针误放进主表。

### 更新文档

```text
docs/doc_workspace/SGCP/main_table_candidate.md
docs/doc_workspace/SGCP/rebuttal_short.md
docs/doc_workspace/SGCP/results.md
docs/doc_workspace/SGCP/status.md
docs/doc_workspace/SGCP/target.md
```

### 结论

PAPG 仍作为当前稳定 V2V-only SGCP 主算法：41 帧 `0.81/0.78/0.39`，62.54 Mbps，首 11 帧 NS3 110/110 request complete。ISPG 与 PAPG 基本持平但 AP@0.3 略低；CCISPG 能移动少数高 IoU 个例但显著伤低阈值 AP；object-grid routing hints 在逐 GT 对比中修复 4 行但损失 15 行。因此这些结果作为 failure analysis / claim boundary，而不是主表算法。

论文口径应明确：PAPG 改善去中心化 V2V 的 AP/payload tradeoff；EdgeCooper-HD 的 AP@0.7 优势来自 edge/global assignment 能力，应单列为 edge-assisted reference。若继续追 AP@0.7，下一步必须设计 proposal/objectness-level trigger 来保护已覆盖 object prototypes，不能继续叠加临时 routing 修补。

## 2026-07-18 main.tex PAPG consistency audit

### 目的

检查 `C:\Workspace\icdcs-paper\SGCP\main.tex` 是否与当前 SGCP 工作区的 PAPG 主线、EdgeCooper-HD 分层和 routing-probe 边界一致。

### 修改

已直接修改 `C:\Workspace\icdcs-paper\SGCP\main.tex`：

- 主表中 `SGCP (PAPG, 10 ch.)` 只保留方法名加粗，移除 AP@0.3/AP@0.5/AP@0.7/Mbps 的数值加粗，避免把非列最优值标成最优。
- 结果段新增 PAPG 与 communication-aware selective V2V 的精确 tradeoff：PAPG 在 AP@0.3/AP@0.5 高 `0.03/0.03`，但上传流量高 6.1%，AP@0.7 低 0.01。
- 结果段补入 routing probe 边界：object-routing 可以修复少数漏检目标，但可能损失更多已覆盖目标，因此作为 failure analysis，不作为主算法。
- 结论从“outperforms capacity-matched V2V scheduling heuristics”改为更准确的“improves low- and medium-IoU accuracy over capacity-matched V2V scheduling heuristics”，并明确 edge-assisted global assignment 是单独的高 IoU reference。

### 结论

这次修改降低了主表和结论被审稿人质疑“选择性加粗/过度声明”的风险。当前正文口径与 `main_table_candidate.md`、`results.md` 和 `rebuttal_short.md` 基本一致：PAPG 是 V2V-only 主线，EdgeCooper-HD 是 edge-assisted reference，routing hints 是失败分析。

## 2026-07-18 FullPerception naming fix and EdgeCooper/PAPG repeat

### 目的

回应当前主表审阅中的三个问题：

1. `FullPerception-centralized` 命名错误；`0.85/0.83/0.48` 实际是 full 20-CAV early-fusion upper reference，不是 FullPerception baseline。
2. EdgeCooper-HD 与 PAPG AP 过近，需要重新跑实验确认。
3. Selective V2V 系列 baseline 名字需要对应到具体算法。

### 代码核查

`opencda/core/clustering/algorithms/resource_allocation/builder.py` 中：

```text
fullperception / fullperception_pcs -> PCS
fullperception_mws -> MWS
fullperception_random -> RandomRA
```

因此 FullPerception 在仓库里对应 `opencda/core/clustering/algorithms/resource_allocation/pcs.py` 的 blind-spot PCS 调度；full 20-CAV early fusion 只能称为 centralized full-sharing upper reference。

`opencda/tools/offline_inference.py` 中 Selective V2V baseline 为同一 coalition/late-fusion pipeline 下的启发式替换：

- forced random：固定预算随机 sender/grid；
- communication-aware：density gain 除以距离代价，可选 NS3 delivery quality；
- density high-budget：density-only greedy sender/grid，使用 3 members/head、117 grid budget。

### 重跑命令

PAPG 41 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1_trace.csv
```

EdgeCooper-HD 41 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline edgecooper_global_hd --selective-member-budget 3 --selective-grid-budget 117 --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1_trace.csv
```

### 结果

| Method | Frames | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPG repeat | 11 | 0.76 | 0.73 | 0.34 | 8,598,224 | 62.53 |
| EdgeCooper-HD repeat | 11 | 0.77 | 0.73 | 0.37 | 9,097,008 | 66.16 |
| PAPG repeat | 41 | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 |
| EdgeCooper-HD repeat | 41 | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 |

### 论文修订

已直接修改 `C:\Workspace\icdcs-paper\SGCP\main.tex`：

- `FullPerception-centralized` 改为 `Full 20-CAV early fusion`；
- 新增 `FullPerception-PCS (built-in)` 行，结果 `0.33/0.29/0.14, 15.80 Mbps`；
- 正文说明 FullPerception-PCS 是仓库 `pcs.py` baseline，full 20-CAV early fusion 是 AP upper reference。

### 结论

EdgeCooper-HD 与 PAPG 接近不是随机波动，而是当前实现和数据集上的稳定结果。论文应继续分层：EdgeCooper-HD 是 edge-assisted/global assignment reference，PAPG 是 V2V-only decentralized main method。PAPG 不应声称全面超过 EdgeCooper-HD；可主张其在无 RSU/无 edge global assignment 下达到相同 AP@0.3/AP@0.5、更低 payload，但 AP@0.7 仍低于 EdgeCooper-HD。

## 2026-07-18 PACP modality audit and LiDAR proxy reproduction

### 目的

核实 PACP 是否为 RGB 方法，并尝试迁移到当前 SGCP raw LiDAR 点云通信场景。

### 核实

PACP 原文使用 camera perception、SinBEVT/CoBEVT BEV feature、BEV-match priority 和 adaptive autoencoder 压缩/重建 raw camera data；不是点云原生方法。当前实现命名为 `pacp_lidar`，只复现其 priority-aware / BEV-match scheduling idea 到 LiDAR BEV grid，占据一致性和 blind-grid complementarity 替代 RGB BEV feature match。

### 代码修改

`opencda/tools/offline_inference.py` 新增 `--selective-sharing-baseline pacp_lidar`：

- sender priority = LiDAR BEV occupancy match + blind-grid complementarity + distance/NS3 link-quality cost；
- grid priority = overlap match + blind-grid complementarity + novelty + raw density；
- 保持 raw point-grid upload、SGCP coalition/receiver、OpenCOOD early fusion 和 inter-cluster late fusion 评价口径不变。

### 命令与结果

```powershell
conda run -n opencda python -m py_compile opencda\tools\offline_inference.py opencda\tools\offline_ns3_replay.py
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 3 --selective-sharing-baseline pacp_lidar --selective-member-budget 3 --selective-grid-budget 117 --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pacp_lidar_3f_trace.csv
```

3-frame smoke：AP `0.77/0.74/0.37`，total payload 3,249,312 bytes。

41-frame full:

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pacp_lidar`, 3 members/head, 117 grids/head | 0.81 | 0.79 | 0.42 | 44,361,424 | 86.56 |
| `pacp_lidar`, 2 members/head, 87 grids/head | 0.76 | 0.73 | 0.37 | 34,498,160 | 67.31 |

11-frame NS3 dry-run:

- high-budget: 110 scheduled requests, 44 skipped unscheduled demands；
- low-budget: 110 scheduled requests, 9 skipped unscheduled demands。

### 结论

PACP 不能写成点云严格复现；当前 `pacp_lidar` 是 PACP-style priority-aware LiDAR proxy。它证明 PACP 的 priority idea 可迁移到点云通信，但 raw LiDAR payload 较高：高预算 AP@0.7 强但 Mbps 高，低预算 AP 低于 PAPG。建议将其作为近年 V2V priority-aware proxy baseline 或附表，不作为 SGCP 主线替代。

## 2026-07-19 Network-level satisfaction metric first pass

### 目的

推进 `target.md` P0/P5：建立一个可复现的 network-level satisfaction / coverage recovery 指标，避免后续主文只依赖 ego AP 或 aggregate AP。

### 代码修改

新增 `opencda.tools.sgcp_satisfaction_summary`：

- 输入 `offline_inference --object-diagnostics-output` 生成的 per-GT CSV；
- 按 receiver-frame 分组；
- 计算 full-reference-detectable GT recovery；
- 按阈值 `tau` 输出 satisfaction rate；
- 输出 per-sample CSV 和 per-method summary CSV。

### 命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\sgcp_satisfaction_summary.py

conda run -n opencda python -m opencda.tools.sgcp_satisfaction_summary --satisfaction-threshold 0.85 --object-csv Full=docs\doc_workspace\SGCP\artifacts\object_diag_full_41f.csv --object-csv Spatial10ch=docs\doc_workspace\SGCP\artifacts\object_diag_sgcp_spatial_rho3_10ch_41f.csv --object-csv TargetAware=docs\doc_workspace\SGCP\artifacts\object_diag_target_aware_pg_10ch_rho3_41f.csv --object-csv PAPG=docs\doc_workspace\SGCP\artifacts\object_diag_papg_bh2_rho3_41f.csv --sample-output docs\doc_workspace\SGCP\artifacts\satisfaction_p0_20260719\samples_existing_methods.csv --summary-output docs\doc_workspace\SGCP\artifacts\satisfaction_p0_20260719\summary_existing_methods_thr085.csv

conda run -n opencda python -m opencda.tools.sgcp_satisfaction_summary --satisfaction-threshold 0.90 --object-csv Full=docs\doc_workspace\SGCP\artifacts\object_diag_full_41f.csv --object-csv Spatial10ch=docs\doc_workspace\SGCP\artifacts\object_diag_sgcp_spatial_rho3_10ch_41f.csv --object-csv TargetAware=docs\doc_workspace\SGCP\artifacts\object_diag_target_aware_pg_10ch_rho3_41f.csv --object-csv PAPG=docs\doc_workspace\SGCP\artifacts\object_diag_papg_bh2_rho3_41f.csv --summary-output docs\doc_workspace\SGCP\artifacts\satisfaction_p0_20260719\summary_existing_methods_thr090.csv
```

### 结果

| Method | Mean Recovery | P10 Recovery | Satisfaction@0.85 | Satisfaction@0.90 | Payload bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full reference | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Spatial-diverse 10ch | 0.877 | 0.807 | 0.707 | 0.366 | 29,405,296 |
| Target-aware PG | 0.885 | 0.825 | 0.756 | 0.415 | 31,069,968 |
| PAPG | 0.924 | 0.855 | 0.927 | 0.756 | 32,049,872 |

### 结论

`tau=0.70` 对当前强方法过宽，无法区分；`tau=0.85` 与 `mean recovery` 可以支撑 Figure 2 / satisfaction distribution。PAPG 的 p10 recovery 达到 0.855，说明其优势体现在 receiver-frame 尾部覆盖稳定性，而不仅是 aggregate AP。下一步需要为 FullPerception-PCS、EdgeCooperV2V+、pure late fusion 和最终 SGCP protocol-native rows 统一生成 object diagnostics。

### 2026-07-19 correction

用户明确要求主文只使用 aggregate AP，不额外引入 satisfaction metric。上述工具与文档随后已删除，`target.md` 已改为 aggregate AP + Mbps 口径。保留本日志仅作为一次已废弃探索记录，不作为后续实验目标或论文指标。

## 2026-07-19 Aggregate AP manifest first pass

### 目的

继续推进 `target.md` P0：删除 satisfaction metric 后，用 aggregate AP + Mbps 作为统一实验口径，并为后续主表、消融和 Pareto 图建立可复现 manifest。

### 代码与文档

- 新增 `opencda.tools.sgcp_aggregate_ap_manifest`。
- 曾短暂新增 `aggregate_ap_protocol.md` 记录 aggregate AP 口径；按用户后续要求该独立文档已删除，口径改为直接维护在 `target.md` / `status.md` / `results.md` 中。
- 更新 `target.md`、`status.md`、`results.md` 和 `readme.md`。

### 验证命令

```powershell
conda run -n opencda python -m py_compile opencda\tools\sgcp_aggregate_ap_manifest.py

conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "PAPG=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1_trace.csv" --run "EdgeCooperHD=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1_trace.csv" --output-csv docs\doc_workspace\SGCP\artifacts\aggregate_ap_manifest_20260719\repeat_check_manifest.csv --notes "repeat check for aggregate AP manifest"
```

### 结果

| Method | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Evaluated Samples | Trace Rows | Late Fusion | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| PAPG | 0.81 | 0.78 | 0.39 | 41 | 246 | yes | 32,049,872 | 62.54 |
| EdgeCooper-HD | 0.81 | 0.78 | 0.42 | 41 | 246 | yes | 33,519,040 | 65.40 |

### 结论

P0 的 aggregate AP 口径已经可以落地到表格源数据。下一步应对 Table 1 protocol-native comparison 的所有候选行补齐 stdout log + trace CSV + manifest row，优先处理 Head-only、Pure late fusion、FullPerception-PCS、EdgeCooperV2V+、SGCP full 和 Full 20-CAV upper reference。

## 2026-07-19 Table 1 protocol-native manifest first pass

### 目的

推进 `target.md` P1：把 protocol-native comparison 的核心候选行统一到 aggregate AP manifest，消除 FullPerception、full-sharing upper reference、pure late 和 SGCP-compatible scheduler comparison 混写的问题。

### 新跑实验

Full 20-CAV early upper reference：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 2>&1 | Tee-Object -FilePath docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\full20_early_41f.log
```

Pure late singleton 20-CAV：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --clustering singleton --sgcp-upload-mode head_only --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv 2>&1 | Tee-Object -FilePath docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f.log
```

Manifest：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "Full20Early=docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\full20_early_41f.log," --run "HeadOnly=docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_stdout.log,docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_trace.csv" --run "PureLate=docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f.log,docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv" --run "FullPerceptionPCS=docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_41f_tuned_div12_ov0.log,docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_41f_tuned_div12_ov0_trace.csv" --run "EdgeCooperHD=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1_trace.csv" --run "SGCP_PAPG=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1_trace.csv" --override "Full20Early.payload_bytes=60838528" --override "Full20Early.receiver_policy=full-20-cav" --override "Full20Early.trace_rows=41" --override "Full20Early.unique_timestamps=41" --override "Full20Early.inter_cluster_late_fusion=no" --override "Full20Early.resource_allocation=full_sharing_upper_reference" --override "Full20Early.clustering=none" --override "Full20Early.upload_mode=full_point_cloud" --override "Full20Early.avg_source_cavs=20.00" --override "Full20Early.avg_selected_grids=N/A" --output-csv docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\protocol_native_manifest.csv --notes "P1 protocol-native manifest first pass"
```

### 结果

| Method | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Evaluated Samples | Trace Rows | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head-only | 0.26 | 0.22 | 0.09 | 41 | 246 | 0 | 0.00 |
| Pure late singleton 20-CAV | 0.82 | 0.76 | 0.37 | 41 | 820 | 0 point-cloud bytes | 0.00 point-cloud Mbps |
| FullPerception-PCS tuned | 0.59 | 0.53 | 0.22 | 41 | 281 | 12,959,840 | 25.29 |
| EdgeCooper-HD proxy | 0.81 | 0.78 | 0.42 | 41 | 246 | 33,519,040 | 65.40 |
| SGCP-PAPG full | 0.81 | 0.78 | 0.39 | 41 | 246 | 32,049,872 | 62.54 |
| Full 20-CAV early upper | 0.85 | 0.83 | 0.48 | 41 | 41 | 60,838,528 | 118.71 |

### 结论

P1 核心行已有统一 manifest artifact。Pure late singleton 20-CAV 的 AP@0.3 很高，说明 SGCP 叙事不能只强调 late fusion 覆盖；应把 pure late 写成 prediction-sharing reference，并补 detection-box exchange overhead，或者明确它不是 raw point-cloud communication baseline。SGCP-PAPG 的优势应放在 V2V-only 点云预算、PAPG 调度、NS3 子信道可行性和接近 edge/global assignment 的低/中 IoU AP 上。

## 2026-07-19 Table 2 fusion scaffold ablation first pass

### 目的

推进 `target.md` P2：验证 early fusion、late fusion、one-cluster/full-sharing、SGCP clustering + PAPG 的贡献边界。

### 修正

`sgcp_aggregate_ap_manifest` 的 Mbps 计算从使用 `evaluated_samples` 改为优先使用 `unique_timestamps`。原因：no-late clustered early-only 会产生 246 个 receiver samples，但通信时长仍是 41 个仿真 timestamp；如果用 246*0.1 s 会低估 Mbps。

### 新跑实验

Clustered early-only：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\clustered_early_only_papg_41f_trace.csv
```

One-cluster full early-only：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --clustering all_in_one --sgcp-upload-mode full_cluster --sgcp-receiver-policy all-cluster-heads --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\one_cluster_full_early_only_41f_trace.csv
```

Manifest：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "HeadOnly=docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_stdout.log,docs\doc_workspace\SGCP\artifacts\mechanism_probe\head_only_41f_trace.csv" --run "PureLate=docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f.log,docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv" --run "OneClusterEarlyOnly=docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\one_cluster_full_early_only_41f.log,docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\one_cluster_full_early_only_41f_trace.csv" --run "ClusteredEarlyOnly=docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\clustered_early_only_papg_41f.log,docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\clustered_early_only_papg_41f_trace.csv" --run "OneClusterEarlyLate=docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\one_cluster_full_early_only_41f.log,docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\one_cluster_full_early_only_41f_trace.csv" --run "FullSGCP=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1_trace.csv" --override "OneClusterEarlyLate.inter_cluster_late_fusion=identity_single_cluster" --override "OneClusterEarlyLate.notes=P2 fusion scaffold: one-cluster late fusion is identity; shares early-only artifact" --output-csv docs\doc_workspace\SGCP\artifacts\fusion_ablation_20260719\fusion_scaffold_manifest.csv --notes "P2 fusion scaffold manifest first pass"
```

### 结果

| Variant | Aggregate AP@0.3 | AP@0.5 | AP@0.7 | Evaluated Samples | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Head-only | 0.26 | 0.22 | 0.09 | 41 | 0 | 0.00 |
| Pure late singleton 20-CAV | 0.82 | 0.76 | 0.37 | 41 | 0 point-cloud bytes | 0.00 point-cloud Mbps |
| One-cluster full early-only | 0.85 | 0.83 | 0.48 | 41 | 60,838,528 | 118.71 |
| Clustered early-only, PAPG | 0.38 | 0.36 | 0.20 | 246 | 32,049,872 | 62.54 |
| One-cluster early+late | 0.85 | 0.83 | 0.48 | 41 | 60,838,528 | 118.71 |
| Full SGCP, PAPG | 0.81 | 0.78 | 0.39 | 41 | 32,049,872 | 62.54 |

### 结论

该表有效支撑 two-layer fusion 叙事：同样 32.05 MB payload 下，clustered early-only 覆盖严重不足，而 Full SGCP 通过簇间 late fusion 恢复到接近 full-sharing upper reference 的 AP@0.3/AP@0.5。AP@0.7 仍低于 full-sharing，说明点云预算和分簇限制会损失局部几何质量；这恰好为 PAPG/Pareto 和参数敏感性留出解释空间。

## 2026-07-19 Table 3 scheduler comparison first pass

### 目的

推进 `target.md` P3：在同一 SGCP-compatible scaffold 下比较 sender/grid scheduler，避免把 scheduler comparison 混写成 protocol-native baseline comparison。

### 新跑实验

PACP-style LiDAR high-budget 缺 stdout，因此重跑：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --selective-sharing-baseline pacp_lidar --selective-member-budget 3 --selective-grid-budget 117 --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\scheduler_comparison_20260719\pacp_lidar_3m117g_41f_trace.csv
```

Manifest：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "RandomBudget=docs\doc_workspace\SGCP\artifacts\random_forced_3m117g_41f_stdout.log,docs\doc_workspace\SGCP\artifacts\random_forced_3m117g_41f_trace.csv" --run "DensityGreedy=docs\doc_workspace\SGCP\artifacts\selective_high_budget_41f\density_m3_g117_stdout.log,docs\doc_workspace\SGCP\artifacts\selective_high_budget_41f\density_m3_g117_trace.csv" --run "LinkAwareDensity=docs\doc_workspace\SGCP\artifacts\selective_high_budget_41f\communication_aware_m3_g117_stdout.log,docs\doc_workspace\SGCP\artifacts\selective_high_budget_41f\communication_aware_m3_g117_trace.csv" --run "PACP_LiDAR=docs\doc_workspace\SGCP\artifacts\scheduler_comparison_20260719\pacp_lidar_3m117g_41f.log,docs\doc_workspace\SGCP\artifacts\scheduler_comparison_20260719\pacp_lidar_3m117g_41f_trace.csv" --run "EdgeCooperHD=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\edgecooper_hd_41f_r1_trace.csv" --run "SGCP_PAPG=docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1.log,docs\doc_workspace\SGCP\artifacts\repeat_check_20260718\papg_41f_r1_trace.csv" --output-csv docs\doc_workspace\SGCP\artifacts\scheduler_comparison_20260719\scheduler_comparison_manifest.csv --notes "P3 SGCP-compatible scheduler comparison manifest"
```

### 结果

| Scheduler | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps | Avg. Source CAVs | Avg. Selected Grids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random budgeted | 0.77 | 0.73 | 0.38 | 31,613,424 | 61.68 | 3.33 | 103.20 |
| Density-greedy | 0.80 | 0.76 | 0.40 | 37,710,864 | 73.58 | 3.33 | 102.18 |
| Link-aware density | 0.80 | 0.76 | 0.40 | 37,710,864 | 73.58 | 3.33 | 102.18 |
| PACP-style LiDAR proxy | 0.81 | 0.79 | 0.42 | 44,361,424 | 86.56 | 3.33 | 104.93 |
| EdgeCooper-HD proxy | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 | 3.00 | 89.02 |
| SGCP-PAPG | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 2.67 | 97.22 |

### 结论

PAPG 的优势是 AP-Mbps tradeoff，不是 AP@0.7 单点最优。它与 EdgeCooper-HD 在 AP@0.3/AP@0.5 持平但 payload 更低，与 PACP-LiDAR AP@0.3 持平但 payload 低约 27.8%；PACP-LiDAR 和 EdgeCooper-HD 的 AP@0.7 更高，必须在论文中解释为高预算/stronger-prior scheduler 的边界。Density 和 link-aware density 在 high-budget 设置下选择相同，说明该预算点下距离/link penalty 没有实际改变 action，后续 P4 Pareto 需要扫描预算才能更公平展示边界。

## 2026-07-19 Pure late prediction-box communication budget

### 目的

回应 Pure late fusion 过强且 raw-LiDAR payload 为 0 的公平性问题：检查 20-CAV 全局 late fusion 的 detection-box exchange 是否会被 20MHz/10ch/100ms deadline、收发延迟或子信道冲突自然限制。

### 新增工具

`opencda.tools.sgcp_late_box_comm_budget`：输入 offline trace，按每帧每 CAV 的 `pred_boxes` 估算 broadcast 与 all-to-all unicast prediction-box payload、贪心调度完成时间，以及 unscheduled random subchannel contention delivery proxy。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.sgcp_late_box_comm_budget --trace-csv docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv --output-dir docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719 --box-bytes 80 --message-overhead-bytes 64 --packet-overhead-bytes 48 --mtu-bytes 1200 --total-bandwidth-mhz 20 --subchannels 10 --spectral-efficiency 6 --deadline-ms 100
conda run -n opencda python -m opencda.tools.sgcp_late_box_comm_budget --trace-csv docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\pure_late_singleton_41f_trace.csv --output-dir docs\doc_workspace\SGCP\artifacts\late_box_comm_20260719_box128 --box-bytes 128 --message-overhead-bytes 64 --packet-overhead-bytes 48 --mtu-bytes 1200 --total-bandwidth-mhz 20 --subchannels 10 --spectral-efficiency 6 --deadline-ms 100
```

### 结果

- `80 B/box`：broadcast mean/max `0.739/0.823 Mbps`、scheduled mean `1.153 ms`；all-to-all unicast mean/max `14.043/15.638 Mbps`、scheduled mean `19.102 ms`。
- `128 B/box`：broadcast mean/max `1.132/1.265 Mbps`、scheduled mean `1.560 ms`；all-to-all unicast mean/max `21.515/24.028 Mbps`、scheduled mean `27.336 ms`。
- Unscheduled random subchannel contention proxy：broadcast full success 为 100%，all-to-all unicast full success 为 0%。该结果说明只有“无调度 all-to-all 同步抢信道”假设会明显限制 Pure late，不能把 prediction-box sharing 的低 payload 直接写成 broadcast storm。

### 结论

Pure late fusion 应作为 strong prediction-sharing reference 写入论文，主表需显式报告 detection-box overhead 或标注 `0 raw-LiDAR Mbps`。SGCP 的优势应强调 raw LiDAR early fusion 能恢复本地 detector 漏检和高 IoU 几何质量，并通过分簇/PPS 控制 raw point-cloud payload；不能声称有调度的 20-CAV prediction-box late fusion 在当前场景下无法 100 ms 内完成。

## 2026-07-19 Pure late checkpoint sanity

### 目的

核查用户提出的方向：Pure late 过强是否因为当前 early fusion 太弱，或当前实际 early fusion 是否加载 `C:\Workspace\OpenCDA\opencood\logs\pointpillar_early_fusion`。

### 配置核查

- `opencda/scenario_testing/config_yaml/v2xp_cluster_carla.yaml`：`fusion_method: early`，`early: opencood/logs/pointpillar_early_fusion`。
- `opencda/scenario_testing/config_yaml/v2xp_cluster_carla_datadump.yaml`：同样使用 `pointpillar_early_fusion`。
- `opencda/scenario_testing/config_yaml/enable_coperception.yaml`：offline inference 默认读取该文件，也使用 `fusion_method: early`。
- `OpenCOODManager` 通过 `models[fusion_method]` 作为 `model_dir`，所以 SGCP early/full-sharing/PAPG 当前确实加载 `opencood/logs/pointpillar_early_fusion`。
- `pointpillar_early_fusion/config.yaml` 的模型名为 `point_pillar_early_fusion_low_res`，`voxel_size=0.4m`，full 20-CAV early upper reference 为 `0.85/0.83/0.48`，说明高 IoU 上界本身不高。

### 实验

Actual late checkpoint 11 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --fusion-method late --sgcp-constrained --clustering singleton --sgcp-upload-mode head_only --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pure_late_actual_late_20260719\pure_late_actual_late_11f_trace.csv
```

Early checkpoint singleton proxy 11 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 11 --fusion-method early --sgcp-constrained --clustering singleton --sgcp-upload-mode head_only --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pure_late_actual_late_20260719\pure_late_early_singleton_11f_trace.csv
```

Actual late checkpoint 41 帧：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --fusion-method late --sgcp-constrained --clustering singleton --sgcp-upload-mode head_only --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pure_late_actual_late_20260719\pure_late_actual_late_41f_trace.csv
```

### 结果

| Variant | Frames | Fusion Method | AP@0.3 | AP@0.5 | AP@0.7 | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Pure late early-singleton proxy | 11 | early | 0.78 | 0.72 | 0.32 | 当前旧 manifest 的实现口径 |
| Pure late actual late checkpoint | 11 | late | 0.90 | 0.84 | 0.46 | `resuming by loading epoch 30` |
| Pure late actual late checkpoint | 41 | late | 0.89 | 0.83 | 0.49 | 比 full 20-CAV early upper `0.85/0.83/0.48` 略高 |

Actual-late prediction-box overhead：

- `80 B/box`：broadcast `1.068/1.148 Mbps` mean/max，all-to-all `20.298/21.815 Mbps` mean/max，scheduled mean `25.072 ms`。
- `128 B/box`：broadcast `1.654/1.782 Mbps` mean/max，all-to-all `31.431/33.853 Mbps` mean/max，scheduled mean `38.906 ms`。

### 结论

当前 SGCP early fusion 确实使用 `pointpillar_early_fusion`，且 full-sharing early 上界 AP@0.7 只有 `0.48`，说明 early backend 本身不强。与此同时，真正 late checkpoint 的 Pure late 更强，说明 Pure late 过强不是由于误用了 early checkpoint，而是当前场景下 prediction-box sharing reference 本身非常强。后续主表应避免把 Pure late 写成普通低通信 baseline；更合理是作为 prediction-sharing reference/upper，或者换一个 local detector 漏检更明显、raw LiDAR early fusion 能恢复目标的场景。

## 2026-07-19 Unified detector sanity

### 目的

用户明确公平原则：SGCP 两层融合中，点云到检测框的 checkpoint 应统一；所有 baseline 包括 Pure late 都应使用同一 detector checkpoint；SGCP 和 Pure late 的最终 late fusion 都应使用 `naive_late_fusion()` box-level NMS。

### 代码确认

- `opencda/core/ml_libs/opencood_manager.py`：`fusion_method=early` 调用 `inference_early_fusion()`；`fusion_method=late` 调用 `inference_late_fusion()`。
- `opencda/tools/offline_inference.py`：`--sgcp-inter-cluster-late-fusion` 汇总 late sources 后调用 `manager.naive_late_fusion()`。
- `OpenCOODManager.naive_late_fusion()` 是预测框拼接 + torchvision NMS；当前 SGCP 的第二层晚期融合已经是这个函数。

### 实验

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --fusion-method late --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --resource-allocation perception_aware_potential_game --rho-th 3 --num-channels 10 --bandwidth-mhz 20 --head-rb-budget 2 --sgcp-inter-cluster-late-fusion --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\late_detector_unified_20260719\sgcp_papg_late_detector_41f_trace.csv
```

Manifest：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "SGCP_PAPG_LateDetector41=docs\doc_workspace\SGCP\artifacts\late_detector_unified_20260719\sgcp_papg_late_detector_41f.log,docs\doc_workspace\SGCP\artifacts\late_detector_unified_20260719\sgcp_papg_late_detector_41f_trace.csv" --output-csv docs\doc_workspace\SGCP\artifacts\late_detector_unified_20260719\manifest_41f.csv --notes "PAPG constrained replay with fusion_method=late; sanity only because intra-cluster stage is no longer early raw-point fusion"
```

### 结果

| Variant | Frames | Detector / first-stage fusion | Box-level fusion | AP@0.3 | AP@0.5 | AP@0.7 | Payload Mbps | Notes |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| Pure late actual late | 41 | `pointpillar_late_fusion` local detector | `naive_late_fusion()` | 0.89 | 0.83 | 0.49 | 0 raw LiDAR | Prediction-sharing reference |
| SGCP PAPG forced late detector | 41 | `pointpillar_late_fusion` over scheduled source set | `naive_late_fusion()` | 0.87 | 0.81 | 0.48 | 62.54 | Not strict SGCP; first stage is late inference |
| SGCP PAPG mainline | 41 | `pointpillar_early_fusion` raw point-cloud early fusion | `naive_late_fusion()` | 0.81 | 0.78 | 0.39 | 62.54 | Actual SGCP protocol |

### 结论

如果所有方法都强制使用 late checkpoint 的 local detector，Pure late 仍最强，SGCP forced late-detector 版本接近但不超过它。这说明当前场景下 late detector + prediction sharing 的 reference 很强。为了公平且保持 SGCP 论文语义，主线实验应统一使用 `pointpillar_early_fusion` 作为 raw point-cloud-to-box checkpoint；Pure late 可以作为 `pointpillar_early_fusion` singleton detector + `naive_late_fusion()` 的 controlled ablation，actual-late 则单独作为 prediction-sharing reference。

## 2026-07-19 Pareto source second pass

### 目的

继续推进 `target.md` P4，把已经复现但未进入 Figure 1 源表的低预算、带宽压力和敏感性点补入 Pareto source，避免 AP-Mbps 图只依赖少数手工点。

### 操作

- 更新 `docs\doc_workspace\SGCP\artifacts\pareto_20260719\pareto_source.csv`，新增 5 个源表点：`SGCPCoverage5ch20MHz`、`SGCPCoverage10chRho3Bh2`、`SelectiveCommunicationAwareLowBudget`、`SGCP_PAPG_Bh3`、`PACP_LiDAR_LowBudget`。
- 更新 `pareto_notes.md`：明确当前 SGCP 已覆盖 `5ch/10ch/20ch`、`rho2/rho3`、`B_h=2/3`、`cap=3000` first pass；Pure late 必须作为 prediction-sharing reference 分层解释。
- 更新 `target.md`：P4 中 SGCP 和 PACP-LiDAR first-pass 扫描标为完成；Random/Density/Link-aware、FullPerception-PCS、EdgeCooper-HD 的系统 sweep 保持未完成。

### 结果

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Mbps |
| --- | ---: | ---: | ---: | ---: |
| SGCPCoverage5ch20MHz | 0.56 | 0.53 | 0.27 | 28.91 |
| SGCPCoverage10chRho3Bh2 | 0.76 | 0.72 | 0.42 | 54.56 |
| SelectiveCommunicationAwareLowBudget | 0.78 | 0.75 | 0.40 | 58.97 |
| SGCP_PAPG_Bh3 | 0.80 | 0.78 | 0.40 | 62.54 |
| PACP_LiDAR_LowBudget | 0.76 | 0.73 | 0.37 | 67.31 |

## 2026-07-19 Scheduler budget sweep for Pareto

### 目的

继续推进 P4 中 `Random / Density / Link-aware` 的 member/grid budget sweep。固定同一 clustered two-layer fusion scaffold、41 帧、all-cluster-heads pooled aggregate AP，仅改变 selective baseline 和 `member_budget/grid_budget`，用于补齐 Figure 1 AP-Mbps Pareto 的公平预算轴。

### 计划命令

低预算点：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --selective-sharing-baseline random --selective-member-budget 2 --selective-grid-budget 87 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\random_m2_g87_trace.csv
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --selective-sharing-baseline density --selective-member-budget 2 --selective-grid-budget 87 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\density_m2_g87_trace.csv
```

高预算/对齐主表点：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --selective-sharing-baseline communication_aware --selective-member-budget 3 --selective-grid-budget 117 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\communication_aware_m3_g117_trace.csv
```

### 结果

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "Random_m2_g87=docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\random_m2_g87.log,docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\random_m2_g87_trace.csv" --run "Density_m2_g87=docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\density_m2_g87.log,docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\density_m2_g87_trace.csv" --run "CommunicationAware_m3_g117=docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\communication_aware_m3_g117.log,docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\communication_aware_m3_g117_trace.csv" --output-csv docs\doc_workspace\SGCP\artifacts\scheduler_budget_sweep_20260719\scheduler_budget_sweep_manifest.csv --notes "P4 scheduler budget sweep: 41-frame all-cluster-heads inter-cluster late fusion, rho_th=3"
```

| Method | Members/head | Grids/head | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random_m2_g87 | 2 | 87 | 0.75 | 0.70 | 0.34 | 24,772,192 | 48.34 |
| Density_m2_g87 | 2 | 87 | 0.78 | 0.74 | 0.40 | 31,421,408 | 61.31 |
| CommunicationAware_m3_g117 | 3 | 117 | 0.80 | 0.76 | 0.42 | 38,920,592 | 75.94 |

### 结论

Random/Density/Link-aware 已形成 first-pass low/high budget coverage。PAPG `0.81/0.78/0.39, 62.54 Mbps` 在 AP@0.3/AP@0.5 上优于 density low-budget 且只比 random high-budget 多 0.86 Mbps；但 AP@0.7 仍低于 density/communication-aware high-quality reference，需要在 Pareto caption 中保留边界。

## 2026-07-19 EdgeCooper-HD budget sweep

### 目的

继续推进 P4 中 EdgeCooperV2V+ / EdgeCooper-inspired 的 sender cap、assignment budget 与 half-duplex constraint 扫描。固定同一 clustered two-layer fusion scaffold、41 帧、all-cluster-heads pooled aggregate AP，使用 `edgecooper_global_hd` 表示 edge/global assignment + sender load cap + half-duplex proxy，只改变 member/grid budget。

### 计划命令

低预算端：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --selective-sharing-baseline edgecooper_global_hd --selective-member-budget 1 --selective-grid-budget 58 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_hd_m1_g58_trace.csv
```

高预算端：

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --rho-th 3 --selective-sharing-baseline edgecooper_global_hd --selective-member-budget 3 --selective-grid-budget 117 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_hd_m3_g117_trace.csv
```

### 结果

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "EdgeCooperHD_m1_g58=docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_hd_m1_g58.log,docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_hd_m1_g58_trace.csv" --run "EdgeCooperHD_m3_g117=docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_hd_m3_g117.log,docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_hd_m3_g117_trace.csv" --output-csv docs\doc_workspace\SGCP\artifacts\edgecooper_budget_sweep_20260719\edgecooper_budget_sweep_manifest.csv --notes "P4 EdgeCooper-HD budget sweep: edgecooper_global_hd, 41-frame all-cluster-heads inter-cluster late fusion, rho_th=3"
```

| Method | Members/head | Grids/head | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EdgeCooperHD_m1_g58 | 1 | 58 | 0.65 | 0.61 | 0.30 | 18,501,232 | 36.10 |
| EdgeCooperHD_m3_g117 | 3 | 117 | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 |

### 结论

EdgeCooper-HD 低预算端 AP 明显下降，高预算端复现既有主点。这支持将 EdgeCooper-HD 写成 edge-assisted / global assignment reference：它可以在较高预算下取得更强 AP@0.7，但不是低通信量下自然优于 SGCP 的同类纯分布式方法。

## 2026-07-19 FullPerception-PCS parameter sweep

### 目的

继续推进 P4 中 FullPerception-PCS 的原生参数扫描。固定 20MHz/10ch、41 帧、`fullperception_pcs`、`all-scheduled-receivers`，不改变主带宽和子信道数；只扫描 PCS 的 blind-spot granularity (`--pcs-blind-spot-min-division`) 与 candidate overlap threshold (`--pcs-min-overlap-grids`)。该实验用于确认 `pcs.py` baseline 的合理工作区间，并避免把 FullPerception-PCS 与 SGCP-compatible scheduler 表混写。

### 计划命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --bandwidth-mhz 20 --num-channels 10 --pcs-blind-spot-min-division 8 --pcs-min-overlap-grids 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pcs_parameter_sweep_20260719\pcs_div8_ov0_trace.csv
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --bandwidth-mhz 20 --num-channels 10 --pcs-blind-spot-min-division 12 --pcs-min-overlap-grids 1 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pcs_parameter_sweep_20260719\pcs_div12_ov1_trace.csv
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --bandwidth-mhz 20 --num-channels 10 --pcs-blind-spot-min-division 16 --pcs-min-overlap-grids 0 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pcs_parameter_sweep_20260719\pcs_div16_ov0_trace.csv
```

### 执行调整

41 帧完整扫描中，`div8/ov0` 和 `div12/ov1` 均超过 10--15 分钟仍未产生日志/trace，需要手动终止残留进程。这说明 PCS 在更细 blind-spot granularity 或 overlap candidate 设置下存在候选规模/运行时不可承受问题。为避免把不完整运行写成正式结果，本轮改为使用此前已经完成的 11 帧 PCS 参数扫描展示趋势，并保留唯一完整 41 帧 tuned anchor `div12/ov0` 进入 Table/Pareto。

Manifest 生成命令：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest --run "PCS_11f_div4_ov1=docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div4_ov1.log,docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div4_ov1_trace.csv" --run "PCS_11f_div8_ov0=docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div8_ov0.log,docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div8_ov0_trace.csv" --run "PCS_11f_div12_ov0=docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div12_ov0.log,docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div12_ov0_trace.csv" --run "PCS_11f_div12_ov1=docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div12_ov1.log,docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div12_ov1_trace.csv" --run "PCS_11f_div16_ov1=docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div16_ov1.log,docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_11f_div16_ov1_trace.csv" --run "PCS_41f_div12_ov0=docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_41f_tuned_div12_ov0.log,docs\doc_workspace\SGCP\artifacts\pcs_tuning_20260718\pcs_41f_tuned_div12_ov0_trace.csv" --output-csv docs\doc_workspace\SGCP\artifacts\pcs_parameter_sweep_20260719\pcs_parameter_sweep_manifest.csv --notes "P4 FullPerception-PCS parameter sweep: 11-frame granularity/overlap trend plus existing 41-frame tuned div12 ov0 point; 20MHz/10ch unchanged"
```

| Variant | Frames | Division | Min overlap | AP@0.3 | AP@0.5 | AP@0.7 | Payload bytes | Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCS_11f_div4_ov1 | 11 | 4 | 1 | 0.58 | 0.51 | 0.24 | 3,668,304 | 26.68 |
| PCS_11f_div8_ov0 | 11 | 8 | 0 | 0.50 | 0.46 | 0.26 | 3,683,232 | 26.79 |
| PCS_11f_div12_ov0 | 11 | 12 | 0 | 0.57 | 0.54 | 0.30 | 4,513,424 | 32.82 |
| PCS_11f_div12_ov1 | 11 | 12 | 1 | 0.56 | 0.49 | 0.16 | 3,634,432 | 26.43 |
| PCS_11f_div16_ov1 | 11 | 16 | 1 | 0.42 | 0.40 | 0.20 | 540,160 | 3.93 |
| PCS_41f_div12_ov0 | 41 | 12 | 0 | 0.59 | 0.53 | 0.22 | 12,959,840 | 25.29 |

### 结论

PCS 的可写主表点仍是 41 帧 `div12/ov0`：`0.59/0.53/0.22`、25.29 Mbps。11 帧趋势显示 `div12/ov0` 在 AP@0.5/AP@0.7 上是最合理的工作点；`min_overlap=1` 会显著伤高 IoU，`div16/ov1` 通信量很低但 AP 明显不可用。更激进的 41 帧 sweep 因候选规模导致运行时间不可接受，可作为 FullPerception-PCS 可扩展性边界而不是主文 Pareto 点。

## 2026-07-19 Detector / checkpoint fairness closure

### 目的

继续推进 P1/P2 中 Pure late baseline 口径和 detector/checkpoint 公平性开放项。用户已经明确：SGCP 两层融合中所有 “点云 -> 检测框” 过程应使用同一 checkpoint，Pure late baseline 也应与 SGCP 一样使用 early checkpoint。

### 处理

新增 `detector_checkpoint_fairness.md`，并同步更新 `target.md`、`status.md`、`results.md`、`readme.md`：

- 主文 raw-LiDAR 系列统一使用 `pointpillar_early_fusion`；
- Pure late 主表行固定为 early-checkpoint singleton local inference + `naive_late_fusion()`；
- actual `pointpillar_late_fusion` checkpoint 结果 `0.89/0.83/0.49` 只作为 detector sensitivity / prediction-sharing reference；
- forced SGCP late-detector row `0.87/0.81/0.48` 不代表 SGCP raw point-cloud early-fusion 协议；
- `Clustered late-only` 不作为核心消融行。

远程 early-fusion fine-tune watcher 仍在等待 GPU：最新日志仍为 `no GPU below 6000 MiB`。

## 2026-07-19 Pareto claim audit

### 目的

继续推进 P4 验收项：确认 SGCP 是否能在合理公平集合内形成 Pareto 优势。此前问题是 Pure late prediction-box sharing 太强，如果把它和 raw-LiDAR point-grid sharing 混入同一 frontier，SGCP 不在 AP@0.3 frontier；因此本轮按通信内容和信息条件分层重新审计。

### 方法

从 `artifacts/pareto_20260719/pareto_source.csv` 中筛选：

- `category in {proposed, sgcp_ablation, sgcp_sensitivity, scheduler_baseline, scheduler_baseline_proxy}`;
- `scaffold == sgcp_compatible`。

排除：

- Pure late prediction-box sharing；
- Edge/global reference；
- full-sharing upper reference；
- negative probe。

### 结果

Raw-LiDAR V2V / SGCP-compatible AP@0.3 frontier 的最高点是 SGCP-PAPG：`0.81` at `62.54 Mbps`。AP@0.5 上，SGCP-PAPG 和 `SGCP_PAPG_Bh3` 在同一 Mbps 下达到 `0.78`，属于同预算 frontier；PACP-LiDAR high-budget 以 `86.56 Mbps` 达到 `0.79`，作为 stronger-priority/high-budget boundary。AP@0.7 上，PAPG 主点不是 frontier，`SGCPCoverage10chRho3Bh2` 以 `54.56 Mbps` 达到 `0.42`，但 AP@0.3/AP@0.5 较低。

### 结论

P4 可按分层口径 first-pass 关闭：论文主张写成 “SGCP-PAPG 在 raw-LiDAR V2V 的 AP@0.3/AP@0.5 中等通信区间处于 Pareto frontier”；AP@0.7 写成 high-IoU localization/checkpoint headroom，不写成全面最优。

## 2026-07-19 Fusion scaffold claim audit

### 目的

继续推进 P2/P6：确认 fusion scaffold ablation 能支撑哪些论文主张，并避免把 AP@0.7 写得过强。

### 结果

从 `fusion_scaffold_manifest.csv` 读取：

- Full20Early / one-cluster early：`0.85/0.83/0.48`，`118.71 Mbps`；
- Clustered early-only：`0.38/0.36/0.20`，`62.54 Mbps`；
- Full SGCP：`0.81/0.78/0.39`，`62.54 Mbps`；
- Pure late controlled：`0.82/0.76/0.37`，0 raw-LiDAR Mbps，prediction-box overhead 另计。

Full SGCP 使用 `52.7%` full raw-sharing payload，保留 full-sharing AP@0.3/AP@0.5/AP@0.7 的 `95.3%/94.0%/81.3%`。与 clustered early-only 相同 payload 对比，inter-cluster late fusion 将 AP 从 `0.38/0.36/0.20` 提升到 `0.81/0.78/0.39`。

### 结论

P2/P6 可按保守口径 first-pass 关闭：two-layer fusion 对 coverage / AP@0.3-0.5 的贡献明确；AP@0.7 仍写成 high-IoU localization/checkpoint headroom，不写成全面最优。

## 2026-07-19 Figure 2/3 readiness check

### 目的

继续推进 P5/P6 剩余验收：检查 protocol breakdown 和 fusion contribution 图是否能区分方法、趋势是否符合当前叙事。

### 发现

视觉检查 `figure2_protocol_breakdown.png` 和 `figure3_fusion_contribution.png` 后发现：Pure late 图内标注为 `raw 0.0`，容易被误读为零通信 baseline，和当前 `late_fusion_box_comm.md` / `main.tex` 中 prediction-box overhead 口径不一致。

### 处理

修改 `artifacts/figures_20260719/plot_breakdowns.py`：

- 增加 per-row communication label；
- Pure late 标为 `box 0.7`；
- 其他方法继续标为 `raw X.X`。

重生成命令：

```powershell
conda run -n opencda python docs\doc_workspace\SGCP\artifacts\figures_20260719\plot_breakdowns.py
```

### 结论

Figure 2 现在可以区分 protocol-native 方法和 reference 边界；Figure 3 可以清楚展示 clustered early-only 到 Full SGCP 的 coverage gain，以及 Full20Early 的 AP@0.7 上界。P5/P6 图表验收项按 first-pass 完成处理。

## 2026-07-19 Scenario sufficiency audit

### 目的

继续推进 P0 剩余项：判断当前 41 帧场景是否无法支撑关键图表，是否需要重新打开 CARLA 导出新场景。

### 结论

新增 `scenario_sufficiency_audit.md`。当前 artifact index 已覆盖 Table 1、Figure 1、Figure 2、Figure 3、Table 3、Table 4、runtime/NS3 appendix 和 qualitative case study draft；所有主文 first-pass 图表均有可追溯 artifact。因此本阶段不重新导出 CARLA 场景。

### 触发条件

后续仅在以下情况下重新采集：

- early checkpoint 回收并重跑后，当前场景仍无法支撑 raw-LiDAR V2V AP@0.3/AP@0.5 frontier；
- 需要更强动态稳定性或真实 CAV 密度 sweep；
- 需要正式 CARLA+NS3 在线端到端图表；
- 论文决定把 AP@0.7 high-IoU 提升作为核心主张。

## 2026-07-19 Early checkpoint recovery protocol

### 目的

继续推进唯一剩余实质项：early-fusion checkpoint fine-tune。当前 GPU 仍未空闲，因此本轮检查远程 watcher 完整性，并把 checkpoint 回收、重跑和验收步骤固化为文档。

### 远程状态

```text
mindspore-187:/data2/gzc/sgcp_early_train/
watcher: runs/start_train_when_gpu_free.sh
log: logs/train_gpu_waiter.log
env: opencood-gzc
```

最新检查显示 8 张 GPU 均约 `22203/24576 MiB` used，watcher 仍输出 `no GPU below 6000 MiB; sleeping 300s`。

### 处理

新增 `early_checkpoint_recovery.md`，记录：

- watcher 行为；
- 每轮轮询命令；
- 训练日志定位命令；
- step checkpoint 查找与下载命令；
- 回收后必须重跑的 SGCP-PAPG / Pure late controlled 实验；
- 是否替换主文结果的验收标准。

同时更新 `protocol_native_claim_audit.md`，将 Pure late detector/checkpoint fairness 从 open 改为 pass；P1 剩余风险只保留 early-fusion checkpoint strength。
## 2026-07-19 - Early-from-late detector checkpoint probe

- 目的：验证是否可以保持 SGCP raw point-cloud early-fusion 通信语义不变，只用更强 late detector checkpoint 初始化/替换 merged point-cloud detector，从而改善当前 early checkpoint 偏弱导致的 AP@0.7 风险。
- 实验目录：`docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/`。
- 模型目录：`pointpillar_early_from_late_weights/`，其中 `config.yaml` 复制自 `opencood/logs/pointpillar_early_fusion/config.yaml`，`latest.pth` 复制自 `opencood/logs/pointpillar_late_fusion/net_epoch30.pth`。
- 专用配置：`enable_coperception_early_from_late.yaml`，仅将 `models.early` 指向上述实验模型目录，`fusion_method` 仍为 `early`。
- 计划命令：先运行 11 帧 SGCP-PAPG smoke test；若结构兼容且结果不差，再运行 41 帧 SGCP-PAPG、Pure late controlled 和 Full20Early upper reference。

## 2026-07-19 - COSDH checkpoint as SGCP early detector probe

- 目的：测试 `D:\Files\Recent\opv2v_cosdh-20260708T114517Z-3-001.zip` 中 COSDH checkpoint 是否可作为 SGCP merged point-cloud detector 的更强初始化。
- 约束：不修改 `C:\Workspace\OpenCOOD`；只读取其代码/配置。权重包解压到本仓库 artifact：`docs/doc_workspace/SGCP/artifacts/cosdh_checkpoint_probe_20260719/opv2v_cosdh/`。
- COSDH 原配置：`core_method=point_pillar_comm_multiscale`、`fusion.core_method=intermediate`、LiDAR range `[-140.8, -38.4, -3, 140.8, 38.4, 1]`。
- 第一阶段实验不引入 COSDH 中期融合模型本体，而是生成 `pointpillar_early_from_cosdh_compatible/latest.pth`：将 COSDH 中 140 个与 SGCP early `point_pillar` 同名同形状权重迁移到 early detector，`cls_head.weight` 与 `reg_head.weight` 因通道不兼容保留原 early 权重。
- 专用配置：`enable_coperception_early_from_cosdh.yaml`，仅将 `models.early` 指向上述兼容模型目录，`fusion_method` 仍为 `early`。
- 计划命令：先跑 11 帧 SGCP-PAPG smoke test；若 AP 有提升，再扩展到 41 帧，并补 Pure late controlled / Full20Early 同 checkpoint 公平对照。

### 结果更新

- late detector checkpoint 直接替换 early detector：11 帧 SGCP-PAPG 为 `0.58/0.48/0.15`，低于原 early checkpoint 的 `0.76/0.73/0.34`，不继续扩展。
- attentive intermediate checkpoint 直接作为 early detector：11 帧为 `0.85/0.77/0.32`，41 帧 SGCP-PAPG 为 `0.87/0.81/0.36`，Full20 early upper reference 为 `0.88/0.85/0.45`。该路线显著提升 AP@0.3/AP@0.5，但 AP@0.7 低于原 PAPG 主线的 `0.39`，适合作 checkpoint sensitivity / potential mainline candidate，不直接替换主表。
- COSDH compatible-weight transplant：仅迁移 140 个同名同形状权重时 11 帧为 `0.00/0.00/0.00`；保留 early heads、仅迁移 COSDH backbone 时为 `0.02/0.00/0.00`。大量误检导致 AP 失效，不继续扩展。
- COSDH 实模型适配：已从 `C:\Workspace\OpenCOOD` 复制所需模型代码到本仓库，新增 `--collapse-to-ego-pointcloud`，可将 SGCP 已调度 raw point cloud 合并为单个 receiver 输入，并加载 COSDH `point_pillar_comm_multiscale` checkpoint。1 帧 smoke test 成功跑通，但 6 个 cluster head 全部 `pred_boxes=0`，最终 `fused_pred_boxes=0`。当前判断为 COSDH 配置/后处理/训练分布与本 CARLA dump 不匹配，需要 logits/threshold calibration 后再扩展。

## 2026-07-19 - COSDH output/postprocess diagnosis

### 目的

继续排查 COSDH 实模型 collapsed smoke test 为何 0 prediction，判断是否只是 postprocess 阈值过高。

### 处理

新增 `offline_inference` 调试参数：

- `--debug-opencood-output`：打印 `psm/rm` shape、sigmoid 置信度分位数、各阈值候选 anchor 数，以及 postprocess 几何过滤 / NMS 计数。
- `--postprocess-score-threshold <float>`：临时覆盖 OpenCOOD `score_threshold`，只用于 checkpoint 校准 probe。

### 结果

Artifact：

- `artifacts/cosdh_checkpoint_probe_20260719/sgcp_papg_cosdh_collapsed_1f_debug.stdout.log`
- `artifacts/cosdh_checkpoint_probe_20260719/sgcp_papg_cosdh_collapsed_1f_thr001.stdout.log`
- `artifacts/cosdh_checkpoint_probe_20260719/sgcp_papg_cosdh_collapsed_1f_thr0005_nms.stdout.log`
- `artifacts/cosdh_checkpoint_probe_20260719/sgcp_papg_cosdh_collapsed_1f_thr0003.stdout.log`

关键发现：

- 默认 `score_threshold=0.2` 下，6 个 cluster-head receiver 的 `psm` sigmoid 最大值仅约 `0.0148--0.0224`，远低于正常检测置信度。
- 降到 `0.01` 仍然没有最终预测框。
- 降到 `0.005/0.003` 后会出现大量低分候选，且部分候选能通过 large-box、z、range 和 NMS 诊断计数；但正式 OpenCOOD postprocess 仍返回 `pred_boxes=0`。这些候选分数极低，不能作为有效检测结果。

### 结论

COSDH 当前不是简单调低阈值即可迁移到 SGCP merged point-cloud detector 的 checkpoint。更可能的问题是 COSDH intermediate model 的训练/输入语义、`proj_first=false`、feature-communication 分支、LiDAR range 或后处理约定与本 CARLA collapsed raw point-cloud 输入不一致。该路线暂不进入主表；后续如继续，只做单独 calibration/debug，不占用主线实验资源。

## 2026-07-19 - Attentive checkpoint Pure late controlled run

### 目的

补齐 attentive intermediate checkpoint 作为 early detector 时的公平 prediction-sharing reference。已有 attentive SGCP-PAPG 41 帧为 `0.87/0.81/0.36`，Full20Early attentive upper reference 为 `0.88/0.85/0.45`；还需要同 checkpoint 下的 Pure late controlled，判断该 detector 是否只是同时增强了 prediction-box sharing reference。

### 配置

- Dataset：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧。
- Detector：`docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/pointpillar_early_from_attentive_weights`。
- Fusion scaffold：`--clustering singleton --sgcp-inter-cluster-late-fusion`，即每个 CAV 单独 local inference，再用 `naive_late_fusion()` 做 box-level NMS。
- 该实验统计 raw LiDAR payload 为 0；prediction-box overhead 仍沿用 `late_fusion_box_comm.md` 口径。

### 命令

```powershell
conda run -n opencda python -m opencda.tools.offline_inference `
  --dataset-root D:\Data\Carla `
  --scenario-id 2026_07_15_01_26_56 `
  --ego-cav-id 1 `
  --max-frames 0 `
  --fusion-method early `
  --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml `
  --sgcp-constrained `
  --clustering singleton `
  --sgcp-receiver-policy all-cluster-heads `
  --sgcp-inter-cluster-late-fusion `
  --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\pure_late_attentive_41f_trace.csv
```

### 结果

- Pure late controlled attentive：41 帧 `AP@0.3/0.5/0.7 = 0.82/0.65/0.28`。
- Trace：`docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\pure_late_attentive_41f_trace.csv`，`820` receiver rows，41 frame groups，raw LiDAR payload `0`。
- 与 SGCP-PAPG attentive `0.87/0.81/0.36` 对比：同 detector/checkpoint 下，SGCP 的 raw point-cloud early fusion + selective scheduling 对 AP@0.5/AP@0.7 有明显收益，避免了“attentive 只是强化 pure late reference”的口径风险。

### Prediction-box overhead

按 `sgcp_late_box_comm_budget` 估算：

- `80 B/box`：broadcast mean/max `1.37/1.51 Mbps`；scheduled all-to-all mean/max `25.97/28.60 Mbps`，平均调度完成 `31.39 ms`。
- `128 B/box`：broadcast mean/max `2.13/2.35 Mbps`；scheduled all-to-all mean/max `40.53/44.65 Mbps`，平均调度完成 `47.60 ms`。

Artifacts：

- `docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\pure_late_attentive_box_comm_80\`
- `docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\pure_late_attentive_box_comm_128\`

## 2026-07-19 17:05 - 论文正文补入 checkpoint sensitivity 口径

### 目的

将 attentive checkpoint 的结论落到论文正文，同时避免把 detector sensitivity 误写成主表替换。该段回应当前最大风险：`pointpillar_early_fusion` checkpoint 可能限制 SGCP raw point-cloud early fusion 的上限。

### 修改

- 文件：`C:\Workspace\icdcs-paper\SGCP\main.tex`。
- 位置：Table 1 protocol-native comparison 解释段之后、SGCP 主行解释之前。
- 内容：补入 attentive intermediate checkpoint sensitivity：SGCP-PAPG `0.87/0.81/0.36`，Pure late controlled `0.82/0.65/0.28`，Full20Early upper reference `0.88/0.85/0.45`。

### 结论

- 该证据支持“同 detector 下，SGCP 的 raw point-cloud sharing 比 prediction-only late fusion 在 AP@0.5/AP@0.7 上更强”。
- 由于该 probe 更改了 detector initialization，且 AP@0.7 不优于当前主线 `pointpillar_early_fusion` PAPG，论文中明确写为 sensitivity evidence，不替换 Table 1。

### 验证

- 已做轻量 LaTeX 结构检查：`table/figure/tabular` begin-end 数量配平。
- 本机仍未检测到 `latexmk` / `pdflatex`，未生成 PDF。

## 2026-07-19 17:35 - 固化 detector checkpoint sensitivity manifest

### 目的

把 mainline、actual-late、attentive、COSDH checkpoint probes 从散落日志整理为统一机器可读证据，服务论文 appendix/rebuttal 和后续 checkpoint 回收重跑。

### 新增 artifact

- `docs\doc_workspace\SGCP\artifacts\checkpoint_sensitivity_20260719\detector_checkpoint_sensitivity_manifest.csv`
- `docs\doc_workspace\SGCP\artifacts\checkpoint_sensitivity_20260719\detector_checkpoint_sensitivity.md`

### 关键边界

- 主表仍使用 `pointpillar_early_fusion` mainline：SGCP-PAPG `0.81/0.78/0.39`、Pure late controlled `0.82/0.76/0.37`。
- Actual late checkpoint 与 attentive checkpoint 均写作 sensitivity，不作为 fair raw-LiDAR 主表替换。
- COSDH 路线当前被记录为 negative/smoke artifact：能跑通部分路径，但 collapsed raw point-cloud 输入下无有效最终框。
- 远程 fine-tune checkpoint 一旦回收，必须另建新 artifact 目录重跑 SGCP/Pure late/Full20Early，而不是覆盖本目录。

## 2026-07-19 18:00 - Paper number audit 与网络口径修正

### 目的

检查 `C:\Workspace\icdcs-paper\SGCP\main.tex` 中 Table 1 / Table 3 / Table 4 的 AP、Mbps 和网络参数标签是否与 OpenCDA artifact manifest 一致。

### 发现

- Table 1 与 Table 3 的 AP/Mbps 与 manifest 对齐。
- Pure late 是有意例外：raw-LiDAR manifest Mbps 为 `0`，论文表格使用 `late_fusion_box_comm.md` 中 80 B/box one-hop broadcast overhead，即 `0.74 Mbps`。
- `table4_parameter_sensitivity.csv` 的 channel sweep 曾标为 `5/10/20 ch / 40 MHz`，但当前可复现实验命令使用 `--bandwidth-mhz 20`；该标签是 legacy/default-code 口径残留。
- 旧 trace CSV 未记录 `bandwidth_mhz`，导致 aggregate manifest 不能独立证明网络口径，只能回看命令日志。

### 修改

- 新增 `docs\doc_workspace\SGCP\artifacts\paper_number_audit_20260719\paper_number_audit.csv`。
- 新增 `docs\doc_workspace\SGCP\artifacts\paper_number_audit_20260719\paper_number_audit.md`。
- 修正 `docs\doc_workspace\SGCP\artifacts\parameter_sensitivity_20260719\table4_parameter_sensitivity.csv` 的 channel labels：`40 MHz` -> `20 MHz`。
- 使用更新后的 `opencda.tools.sgcp_aggregate_ap_manifest` 重生成 `protocol_native_manifest.csv` 和 `scheduler_comparison_manifest.csv`，通过 override 明确补入 `num_channels` / `bandwidth_mhz` 列。
- 增强 `opencda.tools.sgcp_aggregate_ap_manifest`：未来 trace 中若有 `num_channels` / `bandwidth_mhz` 字段，manifest 会保留它们。

### 结论

当前论文主配置仍按复现实验命令写作 `20 MHz / 10 subchannels`。`enable_network.yaml` 和 `Params.bandwidth_all=40` 是 legacy/default code path，不能覆盖当前 artifact 命令口径。后续 checkpoint 回收或新场景实验必须生成带网络元数据的新 trace/manifest。

## 2026-07-19 18:35 - Reviewer response matrix 更新

### 目的

把 2026-07-19 之后新增的 artifact 和论文边界同步到审稿回应矩阵与短版 rebuttal，避免最终 rebuttal 仍使用旧主表/旧网络口径。

### 修改

- 更新 `reviewer_response_matrix.md` 日期和总体策略，加入 artifact index / paper number audit / checkpoint sensitivity 三个新证据。
- 更新 Reviewer 2 的 FullPerception 回应：强调 Full20Early 是 upper reference，FullPerception-PCS 是 formal built-in baseline，但 weak PCS 不能作为唯一优势证据。
- 更新 Reviewer 3 的 baseline/ablation 回应：加入 Table 1、Table 3、Pareto、PACP-style LiDAR proxy、checkpoint sensitivity 和 failure diagnosis。
- 更新 Reviewer 4 的 NS3 回应：强调 online time-sync bug 已修，主表仍采用离线 final-delivery 口径，不能用少量在线 CP 帧替换主表。
- 更新 `rebuttal_short.md`：开头补 `20 MHz / 10 subchannel` manifest 口径；新增 detector checkpoint sensitivity 段落；claim boundary 中显式排除 prediction-sharing Pure late 和 edge-assisted global assignment 的全面统治。

### 剩余风险

- 仍需在最终 PDF 可编译后做视觉检查。
- 远端 early checkpoint fine-tune 仍等待 GPU；若回收新 checkpoint，必须另建 artifact 并重跑 SGCP/Pure late/Full20Early。

## 2026-07-19 18:50 - 论文 claim wording 收紧

### 目的

继续检查 `main.tex` 中与当前实验边界不一致的强表述，防止 rebuttal 已保守、正文仍暗示全局最优或全面支配。

### 修改

- 将 introduction/method 过渡段中的 `optimal cluster formation` 改为 `couples coalition formation with scheduling decisions`。
- 保留 SGCP 的联合优化与势博弈叙事，但不再暗示算法证明或求解了全局最优分簇。

### 剩余风险

- `main.tex` 位于 `C:\Workspace\icdcs-paper\SGCP\main.tex`，不在 OpenCDA git 中；后续需在论文仓库单独提交或同步。

## 2026-07-19 18:55 - 远端 fine-tune watcher 状态

### 检查

- `mindspore-187` 上 watcher 进程仍在：`bash /data2/gzc/sgcp_early_train/runs/start_train_when_gpu_free.sh`。
- `logs/train_gpu_waiter.log` 到 `2026-07-19T18:33:31+08:00` 仍显示 `no GPU below 6000 MiB; sleeping 300s`。
- `checkpoints/` 下仍只有旧的 `2026-07-11 14:30 latest.pth`，没有产生新的 fine-tune checkpoint。

### 结论

远端训练尚未开始，当前论文和主表继续使用已复现的 mainline/checkpoint-sensitivity artifact；不能回收或替换 checkpoint 数值。

## 2026-07-19 19:20 - Attentive checkpoint 下补跑 EdgeCooperHD 对照

### 背景

用户关注：如果把 detector/checkpoint 替换成 attentive，SGCP 相比主表中的 Pure late 和 EdgeCooperHD 是否会更好。此前已有 SGCP-PAPG attentive、Pure late attentive、Full20Early attentive，但缺少 EdgeCooperHD attentive 同权重对照。

### 命令

```powershell
$artifact='docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719'
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference `
  --dataset-root D:\Data\Carla `
  --scenario-id 2026_07_15_01_26_56 `
  --ego-cav-id 1 `
  --max-frames 0 `
  --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml `
  --selective-sharing-baseline edgecooper_global_hd `
  --selective-member-budget 3 `
  --selective-grid-budget 117 `
  --sgcp-receiver-policy all-cluster-heads `
  --sgcp-inter-cluster-late-fusion `
  --rho-th 3 `
  --num-channels 10 `
  --bandwidth-mhz 20 `
  --sgcp-trace-output "$artifact\edgecooper_hd_attentive_3m117g_41f_trace.csv" `
  *> "$artifact\edgecooper_hd_attentive_3m117g_41f.log"
```

Manifest：

```powershell
conda run -n opencda python -m opencda.tools.sgcp_aggregate_ap_manifest `
  --run "SGCP_PAPG_attentive=docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\sgcp_papg_early_from_attentive_41f.stdout.log,docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\sgcp_papg_early_from_attentive_41f_trace.csv" `
  --run "PureLate_attentive=docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\pure_late_attentive_41f.stdout.log,docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\pure_late_attentive_41f_trace.csv" `
  --run "EdgeCooperHD_attentive=docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719\edgecooper_hd_attentive_3m117g_41f.log,docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719\edgecooper_hd_attentive_3m117g_41f_trace.csv" `
  --output-csv docs\doc_workspace\SGCP\artifacts\attentive_scheduler_comparison_20260719\attentive_comparison_manifest.csv `
  --notes "attentive detector sensitivity for SGCP vs PureLate and EdgeCooperHD; 20MHz/10ch; not main table replacement" `
  --override SGCP_PAPG_attentive.num_channels=10 `
  --override SGCP_PAPG_attentive.bandwidth_mhz=20 `
  --override PureLate_attentive.num_channels=10 `
  --override PureLate_attentive.bandwidth_mhz=20 `
  --override EdgeCooperHD_attentive.num_channels=10 `
  --override EdgeCooperHD_attentive.bandwidth_mhz=20
```

### 结果

| Variant | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps |
| --- | ---: | ---: | ---: | ---: |
| Pure late attentive | 0.82 | 0.65 | 0.28 | 0 raw LiDAR |
| EdgeCooperHD attentive | 0.85 | 0.74 | 0.35 | 65.40 |
| SGCP-PAPG attentive | 0.87 | 0.81 | 0.36 | 62.54 |

### 结论

同 attentive detector/checkpoint 口径下，SGCP-PAPG 不再弱于 Pure late 或 EdgeCooperHD：相比 Pure late attentive 高 `+0.05/+0.16/+0.08` AP；相比 EdgeCooperHD attentive 高 `+0.02/+0.07/+0.01` AP，且少约 `2.87 Mbps` raw-LiDAR payload。该结果非常有利于论文叙事，但它改变了 detector 初始化，当前仍应作为 checkpoint sensitivity / candidate；若替换正式主表，需要完整重跑所有主表 baseline 和图表。

## 2026-07-19 20:20 - Attentive 全图表重跑并降级 legacy early 图表

### 目的

用户要求立即重跑 attentive 图表，弱化旧 `pointpillar_early_fusion` checkpoint 图表在文档中的地位，避免后续论文写作被旧结果带偏。本轮将 attentive 从单点 sensitivity 扩展为完整 candidate artifact set。

### 新增/重跑 artifacts

- `artifacts/attentive_protocol_20260719/protocol_native_attentive_manifest.csv`
- `artifacts/attentive_fusion_ablation_20260719/fusion_scaffold_attentive_manifest.csv`
- `artifacts/attentive_scheduler_comparison_20260719/scheduler_comparison_attentive_manifest.csv`
- `artifacts/figures_attentive_20260719/figure2_protocol_breakdown_attentive.png/.pdf`
- `artifacts/figures_attentive_20260719/figure3_fusion_contribution_attentive.png/.pdf`
- `artifacts/figures_attentive_20260719/figure4_scheduler_comparison_attentive.png/.pdf`
- `artifacts/pareto_attentive_20260719/figure1_pareto_ap03_attentive.png/.pdf`
- `artifacts/pareto_attentive_20260719/figure1_pareto_ap07_attentive.png/.pdf`

### Table 1 attentive candidate

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Mbps |
| --- | ---: | ---: | ---: | ---: |
| Head-only attentive | 0.42 | 0.30 | 0.13 | 0.00 |
| Pure late attentive | 0.82 | 0.65 | 0.28 | 1.37 box broadcast |
| FullPerception-PCS attentive | 0.59 | 0.46 | 0.22 | 4.99 |
| EdgeCooperHD attentive | 0.85 | 0.74 | 0.35 | 65.40 |
| SGCP-PAPG attentive | 0.87 | 0.81 | 0.36 | 62.54 |
| Full20Early attentive | 0.88 | 0.85 | 0.45 | 118.71 |

### Table 3 attentive scheduler comparison

| Scheduler | AP@0.3 | AP@0.5 | AP@0.7 | Mbps |
| --- | ---: | ---: | ---: | ---: |
| RandomBudget attentive | 0.85 | 0.75 | 0.36 | 61.25 |
| DensityGreedy attentive | 0.86 | 0.78 | 0.38 | 75.94 |
| LinkAwareDensity attentive | 0.86 | 0.78 | 0.38 | 75.94 |
| PACP-LiDAR attentive | 0.88 | 0.79 | 0.37 | 86.56 |
| EdgeCooperHD attentive | 0.85 | 0.74 | 0.35 | 65.40 |
| SGCP-PAPG attentive | 0.87 | 0.81 | 0.36 | 62.54 |

### 结论

- 后续写作默认引用 attentive candidate 图表；legacy `pointpillar_early_fusion` 图表保留为 checkpoint-reference artifacts。
- SGCP-PAPG attentive 同时高于 Pure late attentive 与 EdgeCooperHD attentive，且通信量低于 EdgeCooperHD。
- PACP-LiDAR attentive AP@0.3/AP@0.7 略高，但使用 `86.56 Mbps`，比 SGCP 高 `24.02 Mbps`；写作时作为 Pareto tradeoff，而不是主方法失败。
- Full20Early attentive `0.88/0.85/0.45` 仍是上界。SGCP attentive 以 `52.7%` raw payload 保留其 AP@0.3/AP@0.5 的 `98.9%/95.3%`。

## 2026-07-19 20:45 - main.tex 同步 attentive candidate

### 修改

- 更新 `C:\Workspace\icdcs-paper\SGCP\main.tex` 的 Table 1：使用 attentive protocol-native candidate 数值。
- 更新 Pure late 通信口径：`80 B/box` one-hop broadcast 从旧 early 的 `0.74 Mbps` 改为 attentive 的 `1.37 Mbps`；scheduled all-to-all 从 `14.04 Mbps` 改为 `25.97 Mbps`。
- 更新 Table 3：使用 attentive scheduler comparison 数值。
- 更新 protocol/fusion/Pareto/scheduler 解释段落：删除“attentive 只作 sensitivity、不替换主表”的自我削弱表述，改为 attentive candidate 是后续写作默认入口，legacy early checkpoint 只作 reference。
- 将 attentive PDF 图复制到论文目录：
  - `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_protocol_breakdown.pdf`
  - `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_fusion_contribution.pdf`
  - `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_pareto_ap03.pdf`
  - `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_pareto_ap07.pdf`

### 验证

- 旧主线关键词扫描通过：`0.81/0.78/0.39`、`0.74 Mbps`、`0.85/0.83/0.48`、`0.59/0.53/0.22` 不再作为正文主表结果出现。
- LaTeX 结构计数正常：`table/table*/figure/figure*/tabular` begin-end 均配平。
- 本机仍未检测到 `latexmk/pdflatex/bibtex`，未完成 PDF 编译。

### Artifact

- 新增 `artifacts/paper_number_audit_attentive_20260719/paper_number_audit_attentive.csv`，核对当前 `main.tex` Table 1/3 与 attentive manifests。

### 边界

`C:\Workspace\icdcs-paper\SGCP` 不在 OpenCDA git 仓库中，本次 paper 目录修改需在论文仓库或外部归档中另行提交。

## 2026-07-19 20:38 - 调整 FullPerception-PCS attentive 主表行

用户指出 `FullPerception-PCS 0.43/0.29/0.14, 16.38 Mbps` 在新的 attentive 主表中读起来异常。本轮只调整 PCS 的 blind-spot granularity / receiver policy，不改变 `20MHz/10ch` 主实验预算。

- 短跑探针：
  - `div8/ov0 + all-scheduled-receivers` 11 帧：`0.48/0.35/0.12`。
  - `div12/ov0 + all-cluster-heads` 11 帧：`0.48/0.33/0.13`。
  - `div16/ov0 + all-scheduled-receivers` 11 帧：`0.64/0.49/0.18`。
- 41 帧 anchor：
  - `div16/ov0 + all-scheduled-receivers`：`0.59/0.46/0.22`，payload `2,556,016 bytes` / `4.99 Mbps`，trace rows `252`，平均每条 trace 上传源车 `1.00`，平均 selected grids `6.78`。
- 结论：
  - 原 `0.43/0.29/0.14, 16.38 Mbps` 不再进入 forward-writing 主表。
  - 新行更符合 PCS 作为 low-payload blind-spot scheduled-receiver reproduction 的定位；它仍不是强 SGCP-compatible scheduler baseline。

## 2026-07-19 21:41 - FullPerception-PCS paper audit and Table 4 attentive rerun

用户进一步要求对照 FullPerception 原论文检查 PCS 实现，尤其是“同一个接收方的不同发送方”是否应归为 A 类硬冲突。

核查结论：

- FullPerception 原文 Class A conflict 是 common-node conflict：一个车辆同一时刻只能参与一条链路，两条链路只要共享任一节点即冲突。
- 因此，同 receiver 多 sender 必须互斥；`pcs.py` 当前 A 类判断与论文一致。
- 临时放宽同 receiver 冲突的实验不符合论文，未进入代码和正式 artifact。

新增/更新：

- `fullperception_pcs_paper_audit.md`
- `artifacts/attentive_pcs_budget_fix_20260719/pcs_grid_paperfaithful_div12_ov0_41f.log`
- `artifacts/attentive_pcs_budget_fix_20260719/pcs_fullsender_paperfaithful_div12_ov0_41f.log`
- `artifacts/parameter_sensitivity_attentive_20260719/table4_parameter_sensitivity_attentive.csv`
- `C:\Workspace\icdcs-paper\SGCP\main.tex`

关键结果：

| Experiment | AP@0.3 | AP@0.5 | AP@0.7 | Mbps |
| --- | ---: | ---: | ---: | ---: |
| PCS strict grid replay | 0.56 | 0.41 | 0.18 | 11.22 |
| PCS raw-LiDAR adaptation | 0.63 | 0.49 | 0.17 | 32.06 |
| SGCP Table 4 rho=1/2/3 | 0.87 | 0.81 | 0.36 | 62.57 / 62.54 / 62.54 |
| SGCP Table 4 5/10/20ch | 0.74/0.87/0.88 | 0.61/0.81/0.81 | 0.24/0.36/0.36 | 31.12 / 62.54 / 67.33 |
- 已更新：
  - `artifacts/attentive_protocol_20260719/protocol_native_attentive_manifest.csv`
  - `artifacts/pareto_attentive_20260719/pareto_attentive_source.csv`
  - `artifacts/figures_attentive_20260719/plot_breakdowns_attentive.py`
  - `artifacts/paper_number_audit_attentive_20260719/paper_number_audit_attentive.csv`
  - `C:\Workspace\icdcs-paper\SGCP\main.tex`
- 已重生并同步论文图：
  - `figure2_protocol_breakdown_attentive`
  - `figure1_pareto_ap03_attentive`
  - `figure1_pareto_ap07_attentive`

## 2026-07-19 20:46 - 远端训练检查与 attentive 口径收口

### 远端训练状态

执行：

```powershell
ssh mindspore-187 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; ps -fp 1532887; tail -n 60 /data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log"
```

结果：

- `mindspore-187` 当前 8 张 GPU 均约 `22203 / 24576 MiB` used，utilization 为 `0%`，仍由外部任务占用显存。
- watcher PID `1532887` 仍存活。
- 日志从 `2026-07-19T15:43` 到 `20:38` 持续输出 `no GPU below 6000 MiB; sleeping 300s`。
- 仍未产生新训练 checkpoint；当前仅有上传的初始 `/data2/gzc/sgcp_early_train/checkpoints/latest.pth`。

### 文档口径修正

检查发现 `main.tex` 已使用 attentive forward-writing Table 1，但 `detector_checkpoint_fairness.md`、`reviewer_response_matrix.md`、`rebuttal_short.md`、`protocol_native_claim_audit.md`、`paper_artifact_index.md`、`results.md` 和 checkpoint sensitivity artifact 中仍有“attentive 只作 sensitivity、不替换主表”的旧句子。本轮已统一为：

- 当前 forward-writing candidate：attentive checkpoint，且 Table 1/2/3/Figure 1/2/3/4 均已完整重跑。
- Legacy `pointpillar_early_fusion`：保留为 checkpoint-reference artifact。
- actual-late / COSDH：仍是 sensitivity 或 negative probe，不进入公平 raw-LiDAR 主表。
- 远端 fine-tune：不是当前 attentive candidate 的 blocker；若回收到更好 checkpoint，则触发新 artifact 版本和全表重跑。

## 2026-07-19 22:40 - 论文 claim 收紧与 reviewer response 同步

本轮按 `target.md` P9/P10 继续推进论文落地，重点检查 `main.tex`、新图表口径和审稿意见覆盖情况。

完成事项：

- `C:\Workspace\icdcs-paper\SGCP\main.tex`：
  - 将 intro 中 “strict real-time constraints / most existing CP systems rely on RSUs” 收紧为 “bounded collaboration cycles / many network-level CP systems rely on RSUs”，避免过度概括 V2V/feature-sharing 工作。
  - 将 related work 中 “Nearly all state-of-the-art frameworks rely on RSUs” 改为 network-level RSU/edge scheduling 分支，并新增 decentralized V2V methods 的边界说明。
  - 新增 SMARTFORM / generic self-managed coalition formation 对比，明确 SGCP 不把 coalition formation 本身作为 novelty，而强调 LiDAR-density utility、motion-stability hysteresis、capacity-constrained cluster maintenance、raw-LiDAR grid selection 和 V2V subchannel scheduling 的组合。
  - 收紧 coalition formation 收敛表述：改为固定 topology snapshot 和 admitted migration rule 下的 finite-action stable partition，不再写成无条件全局最优或泛化 Nash guarantee。
  - 将 baseline detector 公平性句子改为所有 point-cloud-to-box inference 使用同一 PointPillars-family attentive checkpoint，避免 “All methods employ PointPillars” 与 checkpoint sensitivity 口径冲突。
  - 结论中将 “Extensive evaluations” 改为更保守的 “Evaluations”，并把 high-IoU 边界写为 edge-assisted global assignment 和 full raw-data sharing reference。
- `C:\Workspace\icdcs-paper\SGCP\Reference.bib`：
  - 新增 `aslam2024smartform`，用于回应 R2 对 Smartform 相似性的质疑。
- `rebuttal_short.md`：
  - 将旧 `pointpillar_early_fusion` rho sweep 数字替换为 attentive forward-writing Table 4：`rho_th=1/2/3` 均为 `0.87/0.81/0.36`，`62.57/62.54/62.54 Mbps`。
- `reviewer_response_matrix.md`：
  - 更新 Smartform concern 状态：正文已补 citation/difference，剩余风险降为 PDF 编译确认引用编号。

验证：

- `main.tex` 轻量结构检查通过：`table 3/3`、`figure 2/2`、`figure* 4/4`、`tabular 4/4`、`equation 28/28`、`algorithm 3/3`。
- `aslam2024smartform` 引用键已同时出现在 `main.tex` 和 `Reference.bib`。
- 本机仍缺少 `pdflatex` / `latexmk` / `bibtex`，未能生成 PDF。
- 远端 `mindspore-187` watcher 仍等待 GPU：8 张 GPU 均约 `22203/24576 MiB` used，当前仅有上传初始 checkpoint `/data2/gzc/sgcp_early_train/checkpoints/latest.pth`，尚无新训练 checkpoint。

## 2026-07-19 23:02 - PDF 编译前静态完整性检查

继续推进 P9/P10。由于本机没有 LaTeX 工具，先做编译前静态检查并清理摘要强 claim。

完成事项：

- `main.tex` 摘要进一步收紧：
  - `eliminating blind spots` 改为 `reducing blind spots`。
  - `operates entirely through` 改为 `operates through`。
  - `Extensive simulations` 改为 `Simulations`。
- 静态检查结果：
  - citation：43 个 citation、29 个 unique keys，均在 `Reference.bib` 中存在。
  - label/ref：32 个 label 无重复，22 个 ref/eqref 均能解析。
  - figures：7 个 `includegraphics` 文件均存在。
  - LaTeX 环境配平：`table 3/3`、`figure 2/2`、`figure* 4/4`、`tabular 4/4`、`equation 28/28`、`algorithm 3/3`、`itemize 5/5`。
  - 高风险短语检查：未再发现 `eliminating blind spots`、`Extensive simulations`、`operates entirely`、`Nearly all`、`strict real-time`、`outperforms all`、`guaranteed`。
- `paper_artifact_index.md`：
  - 将 attentive artifact 行的 `this update` 替换为实际提交范围 `b9ccf50--5945dea` / `0cfc70c--5945dea` / `5945dea`。
  - 记录 2026-07-19 静态 LaTeX 检查通过。

剩余：

- 仍需在有 `pdflatex` / `latexmk` 的环境中做真实 PDF 编译和视觉检查。

## 2026-07-19 23:43 - 外部论文源码快照归档

继续推进 P9/P10 的可追溯性收尾。由于 `C:\Workspace\icdcs-paper\SGCP` 不在 OpenCDA git 仓库中，本轮将当前论文源码快照复制到 SGCP artifact 目录。

新增 artifact：

```text
docs\doc_workspace\SGCP\artifacts\paper_freeze_snapshot_20260719\
```

内容：

- `main.tex`：当前外部论文正文快照。
- `Reference.bib`：当前外部 bib 快照。
- `MANIFEST.md`：记录 `main.tex`、`Reference.bib` 和论文侧 Figure 1/2/3 PDF 的 SHA256。

同时更新：

- `paper_freeze_check_20260719.md`：加入 snapshot 路径，并说明 active paper 目录仍需单独归档/版本控制。
- `paper_artifact_index.md`：新增 Paper Freeze Source Snapshot 行。

远端训练状态：

- `mindspore-187` 仍无可用 GPU：8 张 GPU 均约 `22207/24576 MiB` used。
- watcher 最新日志仍为 `no GPU below 6000 MiB; sleeping 300s`。
- 仍未产生新训练 checkpoint，只有初始 `/data2/gzc/sgcp_early_train/checkpoints/latest.pth`。

## 2026-07-19 23:50 - 停止远端 checkpoint watcher

用户明确要求暂时使用 attentive checkpoint，不再等待远端 fine-tune。本轮停止 `mindspore-187` watcher。

执行前检查：

```text
1532887       1 S    bash /data2/gzc/sgcp_early_train/runs/start_train_when_gpu_free.sh
```

执行：

```powershell
ssh mindspore-187 "kill 1532887 2>/dev/null || true; sleep 1; ps -p 1532887 -o pid,stat,cmd || true; rm -f /data2/gzc/sgcp_early_train/runs/train_gpu_waiter.pid; echo '[manual stop 2026-07-19T23:50+08:00] SGCP watcher stopped; attentive checkpoint fixed as current paper candidate.' >> /data2/gzc/sgcp_early_train/logs/train_gpu_waiter.log"
```

结果：

- `ps -p 1532887` 只输出 header，无进程残留。
- pid 文件已删除。
- watcher 日志已追加 manual stop 记录。
- `target.md` P1 checkpoint 项已改为完成：当前固定使用 attentive forward-writing candidate；未来若重新训练 checkpoint，必须作为新任务开启并重跑全表/图。

## 2026-07-20 00:21 - SGCP 论文实验结果包整理

用户要求将当前所有实验结果与图表数据打包到 SGCP 论文目录，并为图提供原始数据说明。本轮在论文目录下创建结果包：

```text
C:\Workspace\icdcs-paper\SGCP\experiment_results_20260720
```

包内入口：

- `ALL_RESULTS_AND_FIGURE_DATA.md`：集中说明 Table 1/2/3/4、Figure 1/2/3/4 的所有数据、口径、claim boundary 和 raw CSV 路径。
- `README.md`：目录结构说明。
- `PACKAGE_MANIFEST.csv`：包内 106 个文件的相对路径、字节数和 SHA256。

主要子目录：

- `data/`：清洗后的主表/图 CSV。
- `figures/`：论文侧 PDF 图和 PNG 预览。
- `figure_raw_data/`：每张图对应的 raw data 说明。
- `raw_artifacts/`：从 OpenCDA SGCP artifact 复制的 manifests、traces、logs、plot scripts 和 notes。

已纳入当前 forward-writing attentive 数据：

- Table 1：`protocol_native_attentive_manifest.csv`
- Table 2：`fusion_scaffold_attentive_manifest.csv`
- Table 3：`scheduler_comparison_attentive_manifest.csv`
- Table 4：`table4_parameter_sensitivity_attentive.csv`
- Figure 1：`pareto_attentive_source.csv`
- Figure 2/3/4：attentive breakdown source manifests 与 figure notes

检查：

- 结果包共 106 个文件，约 3.99 MB。
- 不包含 detector checkpoint 权重；只包含结果、图表、日志、trace 和绘图/说明文件。

## 2026-07-20 06:25 - INFOCOM clean experiment directory corrected

用户指出此前放入 `C:\Workspace\2026-7-papers\infocom\SGCP\experiment` 的混合协议候选主表会误导 Table 1 写作。本轮已清理并重建该目录：

- 完全移除误导性 raw candidate table，不再保留 deprecated 目录。
- 新增 `protocol_matrix.md`，强制每张表逐行记录 checkpoint、late fusion、clustering、resource allocation 四个维度；当前 checkpoint 统一为 attentive。
- 新增 `data/table1_original_protocol_baselines_20260720.csv`，将原版/协议适配 baseline 与 SGCP-scaffold scheduler comparison 分离。
- 新增 `data/table5_clustering_ablation_attentive_20260720.csv`，补充分簇算法消融。
- 重生成 `figures/figure0` 到 `figure7` 的 PNG/PDF，并重建 `MANIFEST.csv`。

新增 41 帧实验：

- FullPerception-PCS protocol adaptation 已重跑为 singleton 口径：`attentive / no late / singleton / fullperception_pcs`，AP `0.22/0.16/0.06`，`24.28 Mbps`，258 receiver samples。此前 `all_in_one, 11.07 Mbps` 仅作为 rejected diagnostic，不再进入三表。
- EdgeCooper V2V protocol adaptation：`attentive / no late / singleton / selective_edgecooper_global`，AP `0.54/0.48/0.25`，`282.20 Mbps`，820 receiver samples。
- Fixed first-frame clustering ablation：`attentive / inter_cluster_nms / fixed_first_frame / perception_aware_potential_game`，AP `0.83/0.70/0.28`，`62.63 Mbps`。
- All-in-one full raw-sharing clustering reference：`attentive / identity_single_cluster / all_in_one / full_cluster`，AP `0.89/0.86/0.45`，`118.71 Mbps`。

## 2026-07-20 07:15 - Three-table baseline/ablation structure

根据用户讨论，本轮将 baseline 实验组织为三张互补表：

1. Original / protocol adaptation：不额外添加 SGCP coalition 或 common global box aggregation。
2. `+ global box aggregation` normalized comparison：所有方法每帧输出一个 scene-level fused sample；这是有意给 baseline 加统一 box aggregation，用来公平比较 scene-level AP，不称为原版复现。
3. SGCP-compatible scheduler comparison：固定 coalition formation 和 inter-cluster late fusion，只换 scheduler。

为支持第二张表，`opencda.tools.offline_inference` 新增 `--sgcp-receiver-policy all-cavs`，可让每辆 CAV 作为 potential receiver，未收到上传的 receiver 走 local-only，然后在 `--sgcp-inter-cluster-late-fusion` 下统一 NMS 成每帧一个 aggregate AP sample。

新增 41 帧 `+ global box aggregation` 实验：

- FullPerception-PCS + global box aggregation 已重跑为 singleton 口径：`attentive / global_box_nms / singleton / fullperception_pcs`，AP `0.82/0.64/0.27`，`10.51 Mbps`，每帧 20 receivers。
- EdgeCooper V2V + global box aggregation：`attentive / global_box_nms / singleton / selective_edgecooper_global`，AP `0.88/0.76/0.34`，`282.20 Mbps`，每帧 20 receivers。
- 结论：PCS sparse requests 在 global box aggregation 下几乎不超过 pure late；EdgeCooper 在 AP@0.3 上很强，但 raw-LiDAR demand 明显超出 20MHz/10ch 可合理承载范围。SGCP-PAPG 仍为 `0.87/0.81/0.36`、`62.54 Mbps`。
# 2026-07-20 21:05 PCS singleton deterministic audit

目标：继续推进 `target.md` 中 protocol-native baseline 清理。当前 FullPerception-PCS 已改为 singleton receiver universe，但 protocol run 与 global-box run 的 PCS payload 出现明显差异，因此本轮先检查并修复 `pcs.py` 中可能导致同一协议重复运行漂移的非确定性实现。

计划：

- 不改变 PCS 论文约束、不改变 20MHz/10ch、不改变 blind-spot 参数。
- 固定 blind-spot 起点、邻居遍历、候选链路去重和等权 link 排序。
- 运行 `py_compile` 和 1-frame repeated PCS singleton protocol smoke test，确认 trace 稳定性。

结果：

- `pcs.py` 存在两个问题：其一，离线路径中的 CAV id 可能以字符串/整数混用，原来的 `sender_vid == receiver_vid` 不能完全过滤语义自环；其二，blind-spot grouping 使用 `set.pop()`，同一帧不同进程可能得到不同 PCS receiver/grid selection。
- 已在 `pcs.py` 中补充语义自环过滤，并尝试稳定 blind-spot grouping。1-frame stable-hash smoke test 两次 trace hash 完全一致，且第 1 帧产生 7 个 scheduled receivers，量级正常。
- 41-frame stable-hash protocol run 在 10 分钟超时前推进到第 25/41 帧，说明该确定性切分增加了运行成本；本轮不将其写入 paper-facing 表。
- 当前 `experiment` 目录中 FullPerception-PCS paper-facing 行应视为 pre-determinism candidate，下一轮需要用更长 timeout 完成 41-frame protocol/global rerun，或设计更高效的 deterministic tie-break。

# 2026-07-20 23:59 PCS singleton late/no-late alignment rerun

用户指出：理论上 singleton 下启用或不启用晚期融合不应影响 PCS 调度结果。本轮据此继续修复和重跑。

已确认：

- `offline_ns3_replay --dry-run` 不适合验证 singleton PCS，因为该工具当前固定使用 `CoalitionGame(world).run()`，不走 `clustering=singleton`。
- 新增 artifact profiler `artifacts/pcs_singleton_late_align_20260720/profile_pcs_singleton.py`，直接调用 `offline_inference.apply_sgcp_constraint(..., clustering="singleton", receiver_policy="all-cavs")`，只生成 PCS metadata，不跑 OpenCOOD。
- 41 帧 metadata-only PCS singleton 调度完成：820 receiver rows、每帧 20 receivers、每帧 6--8 个非零 scheduled receivers、总 payload `10,781,296` bytes，即 `21.04 Mbps`。该调度计划应同时用于 no-late protocol evaluation 与 all-cavs global box aggregation evaluation。

下一步：

- 重跑 no-late `all-scheduled-receivers` 41 帧 AP。
- 重跑 late `all-cavs + sgcp_inter_cluster_late_fusion` 41 帧 AP。
- 两者 trace 的非零 scheduled links/payload 必须与 metadata-only plan 对齐；若不一致，修复 `offline_inference` 的 receiver/late path。

完成结果：

- no-late PCS singleton：`conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --clustering singleton --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pcs_singleton_late_align_20260720\pcs_singleton_nolate_41f_trace.csv`
- no-late AP：`0.14/0.13/0.06`，trace rows `295`，payload `10,779,344` bytes / `21.03 Mbps`。
- global-box PCS singleton：`conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 0 --sgcp-constrained --resource-allocation fullperception_pcs --sgcp-receiver-policy all-cavs --sgcp-inter-cluster-late-fusion --clustering singleton --num-channels 10 --bandwidth-mhz 20 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\pcs_singleton_late_align_20260720\pcs_singleton_late_allcavs_41f_trace.csv`
- global-box AP：`0.83/0.77/0.38`，trace rows `820`，每帧 20 receiver samples，payload `10,779,344` bytes / `21.03 Mbps`。
- 关键一致性检查：no-late 与 global-box trace 的 295 条非零 scheduled link rows 完全一致，payload 完全一致。因此 singleton 下晚期融合不影响 PCS 调度；AP 差异来自是否把 20 个 receiver 的检测框做 scene-level/global box aggregation。

# 2026-07-21 09:40 P10 experiment credibility repair started

用户新增最高优先级：修复 PCS baseline 合理性/可复现性、late-fusion 检测框通信量漏算、分簇 baseline 不足。本轮已将任务写入 `target.md` 的 `P10`，并把旧“自动任务规则”顺延为 `P11`。

代码推进：

- `pcs.py` 新增 PCS blind-spot unitization 参数：`blind_spot_adjacency_radius` 与 `blind_spot_min_grids`。默认值保持旧行为等价：radius=2、min grids=1；主实验带宽和子信道不变。
- `offline_inference.py` / `offline_ns3_replay.py` 新增 CLI 参数：`--pcs-blind-spot-radius`、`--pcs-min-spot-grids`。
- `offline_inference.py` 新增 deterministic clustering baselines：`random_balanced`、`distance_greedy`、`density_greedy_cluster`。三者只替换 clustering membership，簇头统一按“距离簇中心最近”选择，以便与 SGCP coalition 口径接近。

验证：

- `py_compile` 通过：`offline_inference.py`、`offline_ns3_replay.py`、`pcs.py`。
- 新分簇 baseline 1-frame smoke 均跑通：`random_balanced`、`distance_greedy`、`density_greedy_cluster` 各生成 5 个 cluster-head late-fusion source sample。
- PCS 1-frame 参数 smoke：
  - default：7 rows，112,640 bytes，avg 10.14 selected grids/row，max 41。
  - radius=3, min spot=48：8 rows，185,824 bytes，avg 10.25 grids/row，max 21。
  - radius=4, min spot=96：6 rows，285,200 bytes，avg 14.83 grids/row，max 24。
  - division=4, radius=4, min spot=128：7 rows，402,976 bytes，avg 35.57 grids/row，max 64。
- `division=4,radius=4,min_spot=128` 重复 1-frame run 的 trace SHA256 完全一致：`9CA5758FC64C698E6391CC2573F8D1AB86A9DE6F24A42E7D4179B706F86101AA`。

下一步：

- 对 PCS `division/radius/min_spot/min_overlap` 做 11-frame metadata/AP sweep，优先验证 `div4/radius4/min128` 是否在 41-frame 上得到更合理 payload/AP。
- 重建 experiment 目录中的通信列：`raw_lidar_mbps`、`box_mbps`、`total_mbps`。
- 跑 41-frame clustering ablation：dynamic coalition、fixed first-frame、random balanced、distance greedy、density/quality greedy，并补 literature-inspired 1--2 个 baseline。

# 2026-07-21 10:25 P10.2 late-box communication accounting

目标：修复启用 late/global box aggregation 的行只统计 raw-LiDAR payload、漏算检测框共享 payload 的问题。

已完成：

- 在 `C:\Workspace\2026-7-papers\infocom\SGCP\experiment\scripts\add_late_box_totals.py` 新增统一通信统计脚本。
- 对所有 paper-facing CSV 增加/更新：
  - `raw_lidar_mbps`
  - `box_mbps`
  - `total_mbps`
  - `box_sharing_mode`
- 旧 `mbps` 列已覆盖为 `total_mbps`，即论文写作默认通信量。
- box payload 估算口径：每个 late-fusion source 每帧广播一次检测框，`80 bytes/box + 64 bytes/message`，周期 `100 ms`。
- 修复脚本幂等性：重复运行不会把 box payload 反复叠加到 raw Mbps。
- 重建 `C:\Workspace\2026-7-papers\infocom\SGCP\experiment` 的 figures 和 `MANIFEST.csv`，并更新 `README.md`、`table_guidance.md`、`experiment_update_summary.md`。

关键更新后的示例：

- SGCP-PAPG：`62.5363 raw + 0.7413 box = 63.2776 Mbps`。
- FullPerception-PCS + global box aggregation：`21.0329 raw + 1.4355 box = 22.4684 Mbps`。
- EdgeCooper V2V + global box aggregation：`282.2037 raw + 2.8661 box = 285.0699 Mbps`。
- Pure late：`0 raw + 1.3667 box = 1.3667 Mbps`。

剩余：P10.1 仍需 11/41 帧 PCS blind-spot sweep；P10.3 仍需 41 帧分簇消融补跑。

# 2026-07-21 11:05 P10.1 PCS blind-spot sweep first pass

目标：测试 `div4/radius4/min128` 是否能通过更大的 PCS blind-spot 单元修复 no-late PCS AP 与通信量异常。

运行：

- `pcs_div4_radius4_min128_nolate_11f`: start-index 0，11 帧，no late。
- `pcs_default_nolate_11f`: start-index 0，11 帧，no late，对照。
- `pcs_div4_radius4_min128_nolate_11f_start20`: start-index 20，11 帧，no late。

结果：

- default 前 11 帧：AP `0.00/0.00/0.00`，78 rows，1,247,952 bytes，约 9.08 Mbps，avg 9.35 selected grids/row。
- div4/radius4/min128 前 11 帧：AP `0.00/0.00/0.00`，70 rows，4,179,440 bytes，约 30.40 Mbps，avg 35.21 selected grids/row。
- div4/radius4/min128 后段 11 帧：AP `0.00/0.00/0.00`，64 rows，3,883,680 bytes，约 28.24 Mbps，avg 34.56 selected grids/row。

结论：

- 放大 blind-spot unit 确实能提高 PCS 通信量和 selected grids，但没有改善 no-late AP。
- PCS 的问题不是单纯 under-schedule；更可能是 paper-faithful blind-spot/link utility 与当前 raw-LiDAR attentive detector 的有效检测区域错位，或者 PCS 选到的 receiver/sender 样本没有覆盖可检出 GT。
- 不应把 `div4/radius4/min128` 直接上 41 帧作为修复；下一步应做 PCS link-level diagnostics：输出 scheduled receiver/sender、selected grids 与 GT object grid 的覆盖关系，并考虑 raw-LiDAR adaptation 下的 object-aware blind-region utility。

# 2026-07-21 16:55 P10.1 fusion-method bug fix for PCS sweep

问题：

- 回看日志发现，正式 PCS singleton 41 帧结果使用 `Fusion method: early`，而 11 帧 blind-spot sweep 误用 `Fusion method: intermediate_attentive`。
- 因此上一轮 `0.00/0.00/0.00` 的 PCS sweep AP 不是 PCS 算法结论，而是实验口径 bug。

重跑：

- default 11 帧 aligned command：`--fusion-method early --clustering singleton --resource-allocation fullperception_pcs --sgcp-receiver-policy all-scheduled-receivers --num-channels 10 --bandwidth-mhz 20`。
- div4/radius4/min128 11 帧 aligned command：在上述命令基础上增加 `--pcs-blind-spot-min-division 4 --pcs-blind-spot-radius 4 --pcs-min-spot-grids 128`。

结果：

- default：AP `0.12/0.11/0.04`，78 rows，1,247,952 bytes，`9.08 Mbps`，avg selected grids/row `9.35`，max `41`。
- div4/radius4/min128：AP `0.16/0.14/0.07`，70 rows，4,179,440 bytes，`30.40 Mbps`，avg selected grids/row `35.21`，max `67`。

结论：

- `intermediate_attentive` sweep 降级为 invalid diagnostic；不能写入论文或作为 PCS 修复失败依据。
- 放大 blind-spot unit 有弱正收益，但仍不足以让 PCS no-late baseline 合理进入主表；下一步进入 PCS object-grid/link utility diagnostics。

# 2026-07-21 17:30 P10.1 PCS object-grid diagnostics

目的：解释 PCS no-late AP 偏低到底来自带宽不足、调度随机性，还是 PCS blind-spot proxy 与检测目标错位。

代码：

- `opencda.tools.sgcp_failure_diagnostics` 新增 GT object grid 的 receiver-side membership 字段：
  - `nearest_head_object_grid_in_req`
  - `nearest_head_object_grid_in_high_density`
  - `nearest_head_object_grid_in_pcs_blind_spot`
- `py_compile` 通过。

实验：

- 先用 `offline_inference` 对 default PCS 前 3 帧生成 `pcs_default_nolate_3f_objects.csv`，同时运行 full-reference 对照。
- 再用 `sgcp_failure_diagnostics` 生成 `failure_default_3f_v3/gt_objects.csv`。

结果：

- 3 帧 PCS AP：`0.14/0.12/0.04`。
- GT rows：47；其中 full-reference detected but PCS missed：30。
- missed GT 中：
  - 30/30 位于 nearest-head `req_grids`。
  - 16/30 位于 nearest-head high-density grids，因此不属于 PCS blind spot。
  - 14/30 属于 PCS blind spot。
  - 只有 5/30 的 object grid 被任何 scheduled link 覆盖。
  - 只有 2/30 是 nearest CAV 直接选中了 object grid。
  - nearest CAV object-grid points 平均/中位数为 `637/455`。

结论：

- PCS 低 AP 主要来自 paper-style blind spot (`req_grids - high_density_grids`) 与 object-level detector utility 不匹配。
- 单纯增加 blind-spot 面积可以增加通信量，但不能系统性选中 detector 需要的目标 grid。
- 正式论文中应把 PCS 作为 paper-faithful protocol baseline 或 raw-LiDAR adaptation baseline；若要继续增强 PCS，只能说明是 adaptation variant，不能混同为原版 FullPerception-PCS。

# 2026-07-21 18:40 P10.3 clustering ablation 41-frame run started

目标：将新增 clustering baselines 从 1-frame smoke 推进到 41-frame paper-facing ablation。

统一设置：

- dataset：`D:\Data\Carla\2026_07_15_01_26_56`
- detector：`early` fusion-method path / attentive forward-writing口径
- resource allocation：`perception_aware_potential_game`
- late fusion：`--sgcp-inter-cluster-late-fusion`
- receiver policy：默认 all-cluster-heads
- network setting：20MHz / 10 subchannels
- max frames：41 全序列

本轮优先运行：

- `random_balanced`
- 若时间允许继续 `distance_greedy`、`density_greedy_cluster`

结果：

- `random_balanced`：AP `0.53/0.49/0.23`，raw `31.4695 Mbps`，box `0.3247 Mbps`，total `31.7942 Mbps`，205 receiver rows / 41 frames。
- `distance_greedy`：AP `0.58/0.54/0.31`，raw `31.5183 Mbps`，box `0.3110 Mbps`，total `31.8293 Mbps`，205 receiver rows / 41 frames。
- `density_greedy_cluster`：AP `0.58/0.53/0.30`，raw `31.6265 Mbps`，box `0.3551 Mbps`，total `31.9816 Mbps`，205 receiver rows / 41 frames。
- `mobility_stability_greedy`：MASS/C-MASS-inspired baseline，AP `0.61/0.55/0.28`，raw `31.4539 Mbps`，box `0.3728 Mbps`，total `31.8267 Mbps`，205 receiver rows / 41 frames。

产物：

- OpenCDA artifact：`docs/doc_workspace/SGCP/artifacts/clustering_ablation_20260721/`
- INFOCOM experiment package：`C:\Workspace\2026-7-papers\infocom\SGCP\experiment\data\table5_clustering_ablation_attentive_20260720.csv`
- Figure：`C:\Workspace\2026-7-papers\infocom\SGCP\experiment\figures\figure7_clustering_ablation_ap_bars.png/.pdf`

结论：

- 三个启发式分簇 baseline 与一个 mobility-aware literature-inspired baseline 在约 `31.8 Mbps` 下只达到 `0.53--0.61` AP@0.3 和 `0.49--0.55` AP@0.5，明显低于 SGCP dynamic coalition `0.87/0.81/0.36`。
- fixed first-frame clusters 虽然通信量接近 SGCP，但 AP@0.5 从 `0.81` 降到 `0.70`，说明动态 coalition 更新本身有贡献。
- Table 5 现在可作为机制消融表使用；已包含 random/proximity/sensing-aware/mobility-aware 四类 baseline。若篇幅允许，可继续补 graph coverage clustering；否则可在文中说明当前 baseline 覆盖 random、proximity、sensing-density 和 mobility-stability 边界。

# 2026-07-21 19:20 P10.1 PCS trace-level closure

目的：关闭 `target.md` 中 P10.1 仍未勾选的 PCS 可信度项，确认 no-late PCS 与 +global-box PCS 的调度是否完全一致。

复核 artifacts：

- no-late trace：`docs/doc_workspace/SGCP/artifacts/pcs_singleton_late_align_20260720/pcs_singleton_nolate_41f_trace.csv`
- global-box trace：`docs/doc_workspace/SGCP/artifacts/pcs_singleton_late_align_20260720/pcs_singleton_late_allcavs_41f_trace.csv`

结果：

- no-late PCS：AP `0.14/0.13/0.06`。
- PCS + global box aggregation：AP `0.83/0.77/0.38`。
- 两个 trace 的非零 scheduled links 完全一致：均为 295 条。
- 两个 trace 的 raw payload 完全一致：均为 `10,779,344` bytes / `21.03 Mbps`。
- 链路序列 hash 前缀完全一致：`a99a282ae02603eb`。

结论：

- `+ global box aggregation` 只改变 evaluation/fusion scaffold，不改变 PCS raw-LiDAR 调度。
- paper-faithful PCS 低 AP 已定位为 blind-spot proxy 与 raw-LiDAR detector object utility 不匹配；不应继续以 GT-aware/object-aware 方式“修高”原版 FullPerception-PCS。
- P10.1 已按“可信度修复/边界说明”关闭。若未来新增 object-aware PCS，应命名为 adaptation variant。

# 2026-07-21 16:05 Table 1 PCS receiver-universe alignment

用户指出 Table 1 中 PCS protocol-native 行不应只统计 `all-scheduled-receivers`，否则 PCS 每帧仅约 6 个 receiver samples，而 EdgeCooper V2V 每帧统计 20 个 singleton receivers。按新的对齐规则补跑：

```powershell
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference `
  --dataset-root D:\Data\Carla `
  --scenario-id 2026_07_15_01_26_56 `
  --ego-cav-id 1 `
  --max-frames 0 `
  --fusion-method early `
  --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml `
  --sgcp-constrained `
  --resource-allocation fullperception_pcs `
  --sgcp-receiver-policy all-cavs `
  --clustering singleton `
  --num-channels 10 `
  --bandwidth-mhz 20 `
  --sgcp-trace-output C:\Workspace\2026-7-papers\infocom\SGCP\experiment\raw_artifacts\original_baselines_20260720\fullperception_pcs_allcavs_protocol_41f_trace.csv `
  --eval-stats-output C:\Workspace\2026-7-papers\infocom\SGCP\experiment\raw_artifacts\original_baselines_20260720\fullperception_pcs_allcavs_protocol_41f_eval_stats.csv
```

结果：

- PCS protocol-native aligned：AP `0.21/0.16/0.06`，payload `6,255,728` bytes / `12.206299 Mbps`。
- receiver samples：`820`，即 41 帧 x 20 CAV；未调度 receiver 保留 local-only detection。
- nonzero PCS upload rows：`268`，说明 PCS 调度仍然稀疏；通信量只统计实际 PCS raw-LiDAR upload。
- current-code 对照 `all-scheduled-receivers/no-late` 也得到同一 payload `6,255,728` bytes，AP `0.18/0.13/0.04`，说明 `all-cavs` 只改变 receiver universe / AP aggregation，不改变当前 PCS 调度。

已更新：

- `C:\Workspace\2026-7-papers\infocom\SGCP\experiment\data\table1_original_protocol_baselines_20260720.csv`
- `figure0_original_protocol_baselines_ap_bars.png/.pdf`
- `experiment_update_summary.md`
- `table_guidance.md`
- `scripts/add_late_box_totals.py`
- `MANIFEST.csv`

表 2 `+ global box aggregation` 和表 3 `SGCP-compatible scheduler comparison` 本轮不改机制；它们的 caption 继续明确其不是 protocol-native no-late baseline。

# 2026-07-21 22:10 PCS forward default and protocol objective audit

用户要求后续 PCS 均采用更合理的 `div4/radius4/min128` 参数，并核查 PCS、EdgeCooper 与 SGCP 的调度目标是否一致。

本轮完成：

- 将 `opencda/core/clustering/algorithms/resource_allocation/pcs.py` 的 PCS raw-LiDAR adaptation 默认值改为 `blind_spot_min_division=4`、`blind_spot_adjacency_radius=4`、`blind_spot_min_grids=128`。后续命令若未显式传参，也会采用该 PCS 工作点。
- 新增 `pcs_edgecooper_protocol_alignment.md`，记录 FullPerception-PCS、EdgeCooper、SGCP 的目标与时间/调度语义差异。
- 在 `target.md` 中新增待办：用 `div4/radius4/min128` 重跑 Table 1 PCS、Table 2/6 PCS + global box aggregation、Table 3/A SGCP-scaffold PCS。旧 PCS 表格若未显式标注该参数，只能作为 archived diagnostic。

协议核查结论：

- FullPerception-PCS 的核心目标是有限通信资源下最大化累计感知收益，通信量是约束/效率结果，不是单纯最小化 Mbps；原文调度对象是 blind-spot semantic/features 与链路资源，不天然等价于一次性 raw-LiDAR grid upload。
- EdgeCooper 的核心目标是 edge-assisted holistic perception；它通过互补性数据选择、relay/channel/packet scheduling 降低冗余，不等价于 20 个 singleton receiver 各自重复请求 raw-LiDAR 的 V2V proxy。
- SGCP 当前实验协议是一帧映射一次 100 ms cooperative perception cycle，要求 intra-cluster raw-LiDAR 和 inter-cluster boxes 在该 cycle 中完成；因此 Table 3/A 可称为 SGCP-compatible scheduler comparison，Table 1 必须称为 protocol-native / protocol adaptation comparison。

# 2026-07-21 23:05 PCS repeated-round deadline probe

用户要求避免使用 `multi-slot` 一词，并尝试在 `div4/radius4/min128` 下做一帧内多轮 PCS 调度，但总通信时间不超过 60ms，为融合计算等后续阶段留出时间。

代码变更：

- `offline_inference.py` 新增 `--pcs-frame-rounds` 与 `--pcs-frame-deadline-ms`。
- `fullperception_pcs` 路径支持 repeated-round scheduling：每轮排除前几轮已被 receiver 接受的 blind grids。
- 若 PCS tentative round 超过剩余 deadline，新增 deadline admission：按每条链路可用子信道数裁剪 grid payload，使被接受的并行链路不超过剩余毫秒。
- trace 新增 `frame_comm_time_ms`、`pcs_rounds_requested`、`pcs_rounds_accepted`、`pcs_round_comm_time_ms_json`、`pcs_round_comm_bytes_json`。

验证：

- `py_compile` 通过。
- 1 帧 no-deadline PCS `div4/radius4/min128`：frame `000060` 单轮估算 `245.696 ms`，说明原始 round 已超过 60ms。
- 1 帧 deadline admission：接受 1 轮，`161,360 bytes`，`60.00 ms`。
- 11 帧 deadline admission：AP `0.22/0.17/0.07`，70 receiver samples，`1,648,368 bytes`，`11.99 Mbps`，avg/max/min frame communication time 全为 `60.00 ms`，每帧接受 1 轮。

通信时间对照：

- SGCP-PAPG attentive 41 帧：AP 表 raw payload 为 `62.54 Mbps`；按离线 exact-payload/resource model 估算 avg/max frame time 为 `320.38/323.84 ms`。已有真实 NS3 scheduled-request replay 为 110/110 application + RLC complete，avg/p95 callback `23.91/24.00 ms`，但该 NS3 replay 使用 10,000-byte request payload 验证调度/子信道/冲突，不是 AP 表 raw payload 的逐字节 replay。
- EdgeCooperHD scaffold 41 帧：raw `65.40 Mbps`；离线 exact-payload demand avg/max `327.02/388.54 ms`；已有 scheduled-request NS3 replay 同样 110/110 complete、avg/p95 `23.91/24.00 ms`。
- EdgeCooper V2V protocol adaptation 41 帧：raw `282.20 Mbps`；离线 exact-payload demand avg/max `1411.02/1508.43 ms`，当前没有 exact-payload NS3 admission 证据，不能写成 60/100ms 内可完成。

结论：PCS 可通过 deadline admission 严格控制到 60ms，但 AP 仍低，支持“PCS blind-spot proxy 与 raw-LiDAR detector utility 不匹配”的旧结论。SGCP/EdgeCooperHD 的 NS3 证据证明调度请求可成功收发，但若论文要声称 AP 表中的全 raw payload 都在 60ms/100ms 内完成，需要后续补 exact-payload NS3 replay 或校准 PHY throughput model。

# 2026-07-21 23:55 NS3 chunked single-round timing

用户要求使用 NS3 给出 PCS `div4/radius4/min128`、EdgeCooper 和 SGCP 的单轮通信时长。

本轮完成真实 NS3 replay，而不是离线 payload/rate 估算：

- 先确认 direct exact-payload replay 不适合大点云包：SGCP/EdgeCooper 的 70-80 KB request 会被记录为 manual command，但超过实际 UDP/CAM 单包承载后不会形成可消费 LC buffer。该现象解释了早先 direct replay 中 SGCP `0/10` callback、EdgeCooper 只完成小包的异常。
- 按在线 OpenCDA 逻辑将 raw payload 切为 `max_packet_size=10000 bytes` 的 CAM chunks，并保持子信道指定后重跑 NS3。
- NS3 设置：`targetSubchannels=10`，日志显示 `totalSubChannel=11`、`slBandwidthIn100kHz=396`；单帧 `000060`，`simTime=5.0`，`drain-seconds=5.0`。

结果：

- PCS `div4/radius4/min128` 原始子信道分配：19 chunks / 161,360 bytes，16/19 callback，avg/p95/max `29.38/51.00/67.00 ms`。失败来自本帧 PCS 两条链路同用 subchannel 2。
- PCS unique-subchannel diagnostic：19/19 callback，avg/p95/max `28.74/51.00/67.00 ms`。说明 PCS payload 本身可在 100 ms 内完成，问题是当前 PCS 子信道冲突。
- SGCP-PAPG：82/82 callback，783,392 bytes，avg/p95/max `59.57/110.00/123.00 ms`。
- EdgeCooper-HD scaffold：68/68 callback，639,408 bytes，avg/p95/max `53.74/107.00/108.00 ms`。
- EdgeCooper V2V protocol-first10 diagnostic：32/73 callback，696,480 bytes，avg/p95/max observed `81.81/177.00/190.00 ms`；该 protocol adaptation 单轮仍过载/冲突。

详细说明与 raw artifacts：`docs/doc_workspace/SGCP/artifacts/ns3_single_round_time_20260721/NS3_SINGLE_ROUND_TIME.md`。

# 2026-07-21 NS3 channel estimator unification

用户要求统一 SGCP/PCS/EdgeCooper 等调度算法的信道估算逻辑，并允许该估算设置为真实 NS3 参数；同时排查为何 NS3 实际带宽高于表格设定却仍可能超过 deadline。

本轮完成：

- OpenCDA 新增 `opencda/core/clustering/utils/channel_model.py`，提供 `ChannelModel(mode='logical'|'ns3')`。
- PCS `_get_link_required_subchannels()` 改为可使用统一 `ChannelModel.required_subchannels()`。
- PAPG/PotentialGame 的 max-grid budget 与 NS3-mode data-rate estimate 改为可读取 `p.channel_model`。
- `offline_inference.py` 新增 `--channel-estimator`、`--ns3-tb-size-bytes`、`--ns3-slot-duration-ms`、`--ns3-subchannel-prbs`、`--ns3-symbols-per-slot`、`--ns3-mcs`。
- EdgeCooper/Selective baselines 的 `frame_comm_time_ms` 改为统一 `ChannelModel`；另新增显式 `--selective-frame-deadline-ms`，默认关闭，避免静默改变旧 fixed-budget baseline。
- NS3 侧通过 Windows 直接访问 WSL 仓库并修改：`main.cc` 暴露 `slMcs/slSymbolsPerSlot/slPscchRbs/slMaxNumPerReserve/slMaxTxTransNumPssch/slRriMs/slProbResourceKeep`；manual scheduler 新增 `SymbolsPerSlot` attribute，移除硬编码 9。

验证：

- OpenCDA `py_compile` 通过。
- 按用户要求未进入 WSL 运行 `./ns3 build`；NS3 C++ 改动尚待后续构建验证。

结论：

- “40MHz 无法跑 60Mbps”不应写成物理带宽不足；准确说法是当前 NR sidelink Mode-2 默认配置下，应用层 raw-LiDAR burst 受 TB size、grant cadence、RLC/CAM chunking、控制资源和 MCS 共同限制。
- 若需要容纳更多通信，应显式跑 high-capacity NS3 diagnostic，并让 OpenCDA 使用 `--channel-estimator ns3` 与相同 TB/slot 参数估算。

# 2026-07-21 Channel model validation experiments

用户要求展开实验验证，并允许构建/执行 NS3。

Artifact：`docs/doc_workspace/SGCP/artifacts/channel_model_validation_20260721/`

OpenCDA smoke：

- SGCP-PAPG `logical` estimator，1 frame / 6 cluster-head rows：`305,520 bytes`，mean/max estimated frame time `203.68/243.65 ms`。
- SGCP-PAPG `ns3` estimator (`tb=400B`, `slot=0.5ms`)，1 frame / 6 rows：`444,608 bytes`，mean/max `92.63/96.58 ms`。PAPG grid budget 明显变化，说明 scheduler 使用了统一 estimator。
- PCS `ns3` estimator，1 frame / 20 CAV receiver rows：`727,360 bytes`，mean/max `99.06/99.06 ms`；metadata bug 已修复，`bandwidth_mhz=20.0` 正常写入 PCS trace。
- EdgeCooper-HD `ns3` estimator，1 frame / 6 rows：`826,832 bytes`，mean/max `17.23/25.43 ms`。该 baseline 仍是 fixed member/grid budget；统一 estimator 目前用于 frame-time metadata，deadline trimming 需显式 `--selective-frame-deadline-ms`。

NS3：

- `./ns3 build` 在 `ns-3-dev` 中通过，重新链接 `scratch/vanet/main.cc`。
- SGCP chunked default replay：82/82 callback，1991 consume events，`allocated_mean=398.86B`，delay mean/p95/max `59.57/110/123 ms`。
- 非法 high-capacity probe：`slPscchRbs=4` 被 pool factory 拒绝；`slRriMs=1` 违反 resource selection window，均触发 NS3 fatal。
- 合法 high-capacity probe：`slMcs=28 --slSymbolsPerSlot=12` 保持默认 PSCCH/RRI，82/82 callback，881 consume events，`allocated_mean=898.91B`，delay mean/p95/max `27.18/54/55 ms`。

结论：

- 默认 NS3 的真实服务率与 `ChannelModel(ns3, tb=400B, slot=0.5ms)` 对齐。
- 提高 MCS/symbols 可以显著增加有效 TB size，使同一 SGCP raw-LiDAR chunk replay 进入 100 ms deadline。
- 不能简单通过降低 PSCCH RB 或 RRI 增容；这需要同步修改 resource pool/window 约束并重新验证。

# 2026-07-21 Paper-facing 40MHz/10ch/60ms NS3 replay

用户确认后续希望采用适合写入论文参数表的整数配置，并强调每帧总周期 `100 ms` 中通信 max 最好控制在 `60 ms` 内，为调度、推理/融合和系统开销保留约 `40 ms`。

本轮结论：

- 正式论文信道口径收束为 `40 MHz` configured sidelink bandwidth、`10` OpenCDA-visible target subchannels、`100 ms` perception cycle、`60 ms` communication deadline。
- 尝试 `--slSubchannelSize=11` 失败：NS3 NR sidelink resource pool 报 `Invalid subchannel size in RBs : 11`。合法 PRB 枚举为 `10/15/20/25/50/75/100`。
- 因此采用可执行正式参数：`--slBandwidthIn100kHz=400 --targetSubchannels=10 --slSubchannelSize=10 --slMcs=28 --slSymbolsPerSlot=12`。NS3 报告 `targetSubchannels=10 totalSubChannel=11 bandwidthIn100kHz=400 slSubchannelSize=10`，OpenCDA 仍只使用 `0..9` 十个目标子信道。
- SGCP-PAPG frame `000060`、10KB chunked upload plan：82 requests / 783,392 bytes，application callback `82/82`，delay mean/P95/max `27.18/54.00/55.00 ms`，PHY failures `0`，满足 60ms 通信窗口。
- 观测到 mean manual grant `898.91 bytes`，对应约 `14.38 Mbps` per target subchannel、`143.83 Mbps` across 10 target subchannels。该值是 NS3 scheduler/service-rate 估计，不是 Shannon capacity。

Artifact：`docs/doc_workspace/SGCP/artifacts/ns3_40mhz_10ch_deadline_20260721/`。

后续执行要求：正式 41 帧 AP/表格重跑必须显式记录 `bandwidth_mhz=40`、`num_channels=10`、`channel_estimator=ns3`、`ns3_tb_size_bytes≈899`、`ns3_slot_duration_ms=0.5`、`ns3_subchannel_prbs=10`、`ns3_symbols_per_slot=12`、`ns3_mcs=28`、`communication_deadline_ms=60`。

# 2026-07-21 Protocol-native PCS / EdgeCooper at 40MHz/10ch/60ms

用户要求在正式 `40 MHz / 10 target subchannels / 60 ms communication deadline` 配置下重测原版 baseline 表中的 PCS 和 EdgeCooper，并给出 AP、通信量、平均延迟、最大延迟；若 PCS 单轮延迟太低，则允许多轮调度直到达到 60ms。

本轮完成：

- 修复 PCS repeated-round deadline trimming 暴露的悬空 link bug：`run_pcs_rounds_with_deadline()` 合并多轮策略时只保留有 grid selection 的 link；`pcs.py` 写回资源策略时跳过没有 selected grids 的 link。该修复不改变 PCS 目标函数，只避免 deadline trimming 后 `KeyError`。
- PCS 单轮 41 帧：AP `0.23/0.17/0.06`，raw payload `27,445,136 bytes / 53.55 Mbps`，offline frame time mean/max `43.93/44.35 ms`。
- PCS 单轮真实 NS3 frame `000060`：77 chunks / 727,360 bytes，application callback `77/77`，avg/max delay `25.71/54.00 ms`，RLC complete `77/77`，满足 60ms。
- PCS repeated-round admission 41 帧：AP `0.22/0.17/0.06`，raw payload `33,709,568 bytes / 65.77 Mbps`，offline admitted time `60.00 ms`。但真实 NS3 不可靠：simultaneous replay `60/97` callbacks、max `214ms`；sequential in-frame replay `67/97` callbacks、frame-start max `242ms`。因此 repeated-round PCS 不进入正式 baseline 表。
- EdgeCooper V2V protocol adaptation 41 帧：AP `0.54/0.48/0.25`，raw payload `141,417,808 bytes / 275.94 Mbps`。真实 NS3 global concurrent replay frame `000060`：348 chunks / 3,274,928 bytes，application callback `15/348`，avg/max delay `127.87/215.00 ms`。因此该行是 offline AP reference，但 deadline infeasible under concurrent V2V replay。

Artifact：`docs/doc_workspace/SGCP/artifacts/protocol_40mhz_10ch_20260721/`。

# 2026-07-21 EdgeCooper deadline-constrained admission

用户指出 EdgeCooper V2V 原版适配超过 60ms 延迟上限，要求限制它。

排查结论：

- 旧 `--selective-frame-deadline-ms` 对 selective baseline 的处理是 per-receiver trimming。对于 protocol-native singleton/all-cavs 表，这相当于 20 个 receiver 每个都拿一份 60ms 预算，因此离线 AP 行 `0.54/0.48/0.25, 275.94 Mbps` 虽然能跑 OpenCOOD，但真实 NS3 frame `000060` 只有 `15/348` callbacks，avg/max delay `127.87/215.00 ms`，不可作为 deadline-feasible baseline。
- 仅改成全局 byte budget 后，41 帧为 `0.32/0.26/0.11, 86.30 Mbps`，frame `000060` plan 为 132 chunks / 1,078,800 bytes；真实 NS3 仍只有 `22/132` callbacks，avg/max `139.14/244.00 ms`。说明总字节数不是唯一瓶颈，端点冲突/半双工 role conflict 同样限制可交付性。
- 最终修正为 frame-level global admission + endpoint-conflict-free matching：EdgeCooper 候选链路先按 blind-spot/grid priority 排序，再选择最多 10 条无共享端点链路，随后在同一个 60ms NS3-calibrated byte budget 内填充 grid。未被调度的 singleton receivers local-only。

最终结果：

- 41 帧 AP：`0.32/0.26/0.10`。
- 通信量：`26,091,536 bytes / 50.91 Mbps`。
- NS3 frame `000060`：68 chunks / 649,904 bytes，application callback `68/68`，avg/max delay `25.90/54.00 ms`，RLC TX/RX `734/734`，PHY failures `0`。

结论：正式 protocol-native baseline 表中，EdgeCooper V2V 若要求 60ms deadline，应使用 constrained matching 行；旧高 AP 行只能作为 deadline-infeasible offline diagnostic。

Artifact：`docs/doc_workspace/SGCP/artifacts/edgecooper_deadline_constrained_20260721/`。

# 2026-07-22 SGCP low-budget operating point

用户希望在 EdgeCooper 受 60ms deadline 约束后，尝试一个通信量接近 EdgeCooper constrained row 的 SGCP low-budget 点，避免主表中 SGCP 成为除 full early upper reference 以外通信最高的方法。

执行与结论：

- 保持正式信道口径：`40 MHz / 10 target subchannels / 60 ms communication deadline`，NS3 estimator 使用 `tb_size=899B, slot=0.5ms, subchannel_prbs=10, symbols=12, mcs=28`。
- 保持 SGCP 机制不变：`coalition_game` clustering、`perception_aware_potential_game` scheduler、`all-cluster-heads` receiver policy、inter-cluster NMS late fusion。
- 扫描发现 `B_h=1` 过低：11 帧约 `36.8 Mbps`，AP `0.75/0.63/0.21`。
- 主线 `B_h=2` 11 帧约 `61.5 Mbps`，AP `0.84/0.75/0.32`。
- 采用 deterministic per-source point cap `--max-upload-points-per-source 4000` 后，41 帧结果为 AP `0.86/0.77/0.33`，raw payload `26,240,000 bytes / 51.20 Mbps`。按当前 box overhead 口径，inter-cluster box broadcast 为 `0.70 Mbps`，total 为 `51.90 Mbps`。
- 真实 NS3 frame `000060`：70 chunks / 640,000 bytes，application callback `70/70`，RLC TX/RX `720/720`，PHY failures `0`，delay mean/P95/max `23.714/46.000/46.000 ms`，满足 60ms 通信窗口。

解释口径：该行应写成 SGCP-PAPG 的 low-budget operating point，而不是替换主线算法。它保留相同分簇、调度和 late fusion，只在已选发送方点云上加 deterministic point cap；适合进入 Pareto/低预算补充表，并可与 EdgeCooper deadline-constrained `0.32/0.26/0.10, 50.91 Mbps` 做同通信量级对照。

Artifact：`docs/doc_workspace/SGCP/artifacts/sgcp_low_budget_20260722/`。

# 2026-07-22 EdgeCooper higher-payload probe

用户希望保留 SGCP-PAPG `0.87/0.81/0.36` 为主要结果，同时再尝试调高 EdgeCooper 通信量。

确认事项：

- low-budget 实验没有改 PAPG 默认参数；`--max-upload-points-per-source` 默认仍为空，主线 no-cap attentive 结果继续是 `0.87/0.81/0.36`。
- 为增强 EdgeCooper constrained baseline，新增 `--edgecooper-global-comm-range-m` CLI，默认 `35m`，并把 `edgecooper_global/_hd` 的 endpoint-disjoint matching 从 greedy 改为 exact matching：先最大化无端点冲突链路数，再最大化候选 payload。该改动只影响 EdgeCooper/selective baseline，不影响 PAPG。

实验结果：

- 默认 35m/60ms exact matching，41 帧：AP `0.32/0.26/0.11`，raw payload `29,257,712 bytes / 57.09 Mbps`，比旧 greedy constrained `50.91 Mbps` 更高且 AP 不下降。
- 默认 35m/60ms exact matching，NS3 frame `000060`：72 chunks / 698,224 bytes，application callback `72/72`，RLC TX/RX `787/787`，PHY failures `0`，delay mean/P95/max `26.556/53.000/55.000 ms`。
- 扩大半径到 100m、deadline 75ms、11 帧：raw payload `62.93 Mbps`，但 AP 降到 `0.23/0.18/0.06`。该点说明硬扩大 EdgeCooper 通信并不会改善感知，不建议进论文主表。

结论：EdgeCooper constrained 可以合理提高到 `57.09 Mbps`，但在严格/可解释协议下难以自然提高到 `65-70 Mbps`；若强行扩大半径或放松 deadline，AP 变差且协议口径更牵强。

Artifact：`docs/doc_workspace/SGCP/artifacts/edgecooper_high_comm_20260722/`。
