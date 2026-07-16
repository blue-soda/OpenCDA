# LGCP Localization Error Sensitivity

## 目标

回应审稿意见中 localization error 可能影响 area partition、confidence estimation 和 redundancy removal 的问题。

本实验在已有 `lgcp_carla` 11 帧 dump 上离线注入 CAV xy pose Gaussian noise，不重新运行 CARLA。OpenCOOD prediction / GT area slicing 保持使用真实 ego pose；噪声只作用于 CAV confidence report 的点云 world transform 和 agent-to-area distance prior。

## 工具更新

`opencda/tools/lgcp_area_confidence_eval.py` 新增：

```text
--localization-noise-std
--localization-noise-seed
```

噪声按 `(seed, timestamp, agent_id)` deterministic 生成，保证重复运行可复现。

## 实验设置

- scenario：`2026_07_15_02_33_21`
- frames：11
- grid：`10m x 6m`
- fusion method：`early`
- noise std：`0.0m / 0.2m / 0.5m / 1.0m`
- seed：`7`

## 当前结果

| Noise std | Records | Active areas | Area-frame noisy-or vs recall@0.5 Spearman | Area-acc noisy-or vs AP@0.5 Spearman | Area-acc score_mean vs AP@0.5 Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.0m` | 21418 | 337 | 0.570407 | 0.411840 | 0.402059 |
| `0.2m` | 21428 | 343 | 0.564515 | 0.411840 | 0.396911 |
| `0.5m` | 21408 | 347 | 0.546341 | 0.411840 | 0.396911 |
| `1.0m` | 21432 | 356 | 0.550885 | 0.314543 | 0.396911 |

## Interpretation

- Area-frame confidence-to-recall ranking is stable under 0.2m to 1.0m xy pose noise in this smoke: Spearman remains around `0.55`.
- Accumulated AP ranking is stable up to 0.5m but drops at 1.0m for noisy-or confidence.
- Detector-score AP ranking is mostly unchanged because detector scores come from the fixed inference output; this is expected and should not be interpreted as localization robustness of feature alignment.
- Current result supports a limited claim: the confidence-ranking proxy is not immediately destroyed by moderate pose noise in this single scenario.

## Output Directories

```text
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_localization_noise_0p0m_11f/
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_localization_noise_0p2m_11f/
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_localization_noise_0p5m_11f/
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_localization_noise_1p0m_11f/
```

Summary:

```text
docs/doc_workspace/LGCP/experiments/area_confidence/20260716_lgcp_carla_localization_noise_summary.csv
```

## Paper Boundary

可用于 rebuttal / revision 的保守说法：

```text
We inject Gaussian xy localization noise into CAV confidence reports and observe
that the area-frame confidence-to-recall ranking remains stable up to 1.0m in
the current 11-frame smoke. However, accumulated AP ranking degrades at 1.0m,
so we report this as a robustness diagnostic rather than a complete guarantee.
```

不能直接声称：

```text
LGCP is robust to arbitrary localization error.
```
