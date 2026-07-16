# LGCP Low-Density Latency and Fig. 7 Axis Audit

本文档回应审稿意见中两个相邻问题：

- 低 CAV 数场景下 LGCP 与 baseline latency 接近，原因需要解释。
- Fig. 7 的 axis label 被指出存在问题，需要检查并修正。

## Source Figure

论文 TeX 中 Fig. 7 对应：

```tex
\includegraphics[width=0.9\linewidth]{picture/num_latency_v2.pdf}
\caption{End-to-end latency for varying number of CAVs under the OPV2V dataset.}
\label{fig:Latency1}
```

已将 `C:\Workspace\icdcs-paper\LGCP\picture\num_latency_v2.pdf` 渲染为本目录下的 `fig7_latency_audit.png` 作为核查证据。

当前可见标注：

| Axis / item | Current label | Audit result | Revision action |
| --- | --- | --- | --- |
| y-axis | `End-to-end latency (ms)` | 语义正确，单位明确 | 不需要改成其他指标 |
| x-axis | `Number of vehicles` | 与 caption / 正文中的 CAV 数不完全一致 | 建议改为 `Number of CAVs` |
| legend | `Vehicle-based`, `Edge-assisted`, `LGCP (ours)` | 可接受 | 可保留 |
| caption | `End-to-end latency for varying number of CAVs under the OPV2V dataset.` | 与实验语义一致 | 可保留 |

因此，当前 PDF 中审稿人所谓的 y-axis 问题没有复现；更可能的实际问题是 x-axis 使用了 `vehicles`，而论文讨论的是参与协同感知的 CAV 数。修稿时应重新导出图源，把 x-axis 改为 `Number of CAVs`，并保留 y-axis 为 `End-to-end latency (ms)`。

## Why Low-Density Latency Gains Are Small

低 CAV 数场景下，LGCP latency 和 baseline 接近并不必然表示机制无效，主要原因如下：

1. **Redundancy and contention are weak at low density.** 当只有少量 CAV 参与时，full sharing / edge-assisted baseline 中的冗余传输和链路冲突尚未显著放大，LGCP 的 area-level selective sharing 与 scheduling 优势还没有完全显现。
2. **Fixed coordination overhead is not amortized.** LGCP 需要 confidence report、task assignment、leader upload 和 global-view broadcast 等固定控制面步骤。低 CAV 数时，数据面节省较小，固定开销占比更高。
3. **Edge-assisted compute can dominate at small scale.** Edge-assisted baseline 在低负载下可利用较强边缘算力完成集中融合，线性瓶颈尚未出现，因此与 LGCP 的 decentralized compute 差距较小。
4. **LGCP is designed for scaling behavior.** 随 CAV 数增加，vehicle-based 的 pairwise sharing 和 edge-assisted 的 centralized aggregation / fusion latency 增长更快；LGCP 通过 area-task group、leader local fusion 和 RSU aggregation 将上传任务限制在被选 area 与 leader 结果上，优势在中高密度更明显。

## Suggested Paper Text

可在 Fig. 7 讨论段后加入：

```text
When the number of CAVs is small, the latency gap between LGCP and the
baselines is less pronounced. This is because redundant transmissions and
wireless contention are still limited in sparse collaboration, while LGCP
incurs a fixed coordination cost for confidence reporting, task assignment,
leader upload, and global-view broadcast. In addition, the edge-assisted
baseline can exploit the edge server's computation capability before its
centralized transmission and fusion bottlenecks become dominant. As the number
of participating CAVs increases, redundant vehicle-level sharing and centralized
edge aggregation grow rapidly, whereas LGCP keeps the uploaded area-task data
bounded by selected areas and distributes local fusion to leaders. Therefore,
the latency advantage becomes more visible in denser CAV settings.
```

## Rebuttal Wording

```text
We agree that the previous discussion did not sufficiently explain the
low-density behavior. We have revised the Fig. 7 discussion to clarify that
LGCP has a fixed coordination cost, while the baselines suffer less redundancy
and contention when only a few CAVs participate. The benefit of LGCP becomes
clearer as density increases, where redundant transmissions, centralized edge
aggregation, and fusion latency grow faster. We also checked the figure labels:
the y-axis is correctly labeled as end-to-end latency in milliseconds, and we
revise the x-axis from "Number of vehicles" to "Number of CAVs" for consistency
with the experiment description.
```

## Remaining Action

The paper directory currently contains only the final PDF figures and no plotting script. To fully apply the label correction, regenerate `picture/num_latency_v2.pdf` from the original plotting source with:

- x-axis: `Number of CAVs`
- y-axis: `End-to-end latency (ms)`

