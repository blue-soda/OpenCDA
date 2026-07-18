# SGCP Protocol-Native Claim Audit

更新时间：2026-07-19

本文档审计 `target.md` P1 的 Table 1 叙事是否已经满足“完整系统比较、baseline 分层、Pure late 边界、FullPerception/EdgeCooper 复现解释”的写作要求。审计对象为：

```text
C:\Workspace\icdcs-paper\SGCP\main.tex
docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\protocol_native_manifest.csv
docs\doc_workspace\SGCP\paper_artifact_index.md
docs\doc_workspace\SGCP\main_table_candidate.md
```

## Audit Summary

| Requirement | Status | Evidence |
| --- | --- | --- |
| Table must explain SGCP's complete-system advantage, not a single scheduler win | Pass with caveats | `main.tex` Table 1 is protocol-native and text attributes gains to clustering, perception-aware point-cloud selection, and inter-cluster late fusion. |
| If AP@0.3 comes mainly from late fusion, the text must say it is a system-protocol advantage | Pass | `main.tex` explicitly states late fusion restores network-level coverage and low-IoU recall; scheduler comparison is separated into its own table. |
| Weak/atypical FullPerception-PCS or EdgeCooperV2V+ results must be explained before entering the table | Pass | FullPerception-PCS is separated from Full20Early upper reference and described as repaired/tuned built-in PCS; EdgeCooper-HD is labeled edge-assisted/global assignment reference. |
| Pure late baseline detector/checkpoint fairness | Still open | Current protocol manifest uses early-singleton proxy; actual late checkpoint is recorded only as sanity. Pending early-checkpoint fine-tune may require rerun. |
| Early-fusion checkpoint strength | Still open | Current full early upper AP@0.7 is only 0.48; remote fine-tune watcher is pending GPU. |

## Evidence in `main.tex`

The current `main.tex` already includes the following boundaries:

- Full 20-CAV early fusion is described as a centralized upper reference, not a fair RSU-free baseline.
- FullPerception-PCS is described as the repository implementation of FullPerception's blind-spot-driven PCS scheduler, separate from full-sharing upper reference.
- EdgeCooper-HD is described as a virtual edge-assisted reference using global blind-spot complementarity and half-duplex role constraints; it is not treated as a fully decentralized V2V baseline.
- Pure late is described as a prediction-sharing reference with detection-box overhead, not a zero-communication baseline.
- PAPG is compared primarily against RSU-free V2V baselines at similar or higher raw-LiDAR payload, while EdgeCooper-HD is kept as a stronger information-condition reference.
- The scheduler comparison table is explicitly scoped to a common SGCP-compatible scaffold and is not used as the protocol-native system ranking.

## Remaining P1 Risks

1. Pure late detector/checkpoint decision:
   - Main controlled protocol uses `pointpillar_early_fusion` singleton local inference + `naive_late_fusion()`.
   - Actual late checkpoint reaches higher AP and must remain a prediction-sharing sanity/reference unless the paper deliberately changes detector fairness policy.

2. Early-fusion checkpoint risk:
   - Current SGCP raw point-cloud early fusion is limited by the existing early checkpoint.
   - Remote training on `mindspore-187:/data2/gzc/sgcp_early_train/` is still waiting for GPU; if a better checkpoint is recovered, Table 1 and dependent figures must be regenerated under a new artifact index version.

3. Main-claim wording:
   - Do not claim SGCP has highest AP and lowest Mbps across all baselines.
   - Safe claim: SGCP-PAPG achieves a favorable RSU-free V2V raw-LiDAR AP/Mbps tradeoff, matches EdgeCooper-HD on AP@0.3/AP@0.5 with less payload, but has lower AP@0.7 because EdgeCooper-HD uses global edge-side assignment.

## Target Update Rationale

The first three P1 acceptance items can be marked complete because the current paper text and artifact index now explicitly address them. The Pure late detector fairness and early checkpoint strength items should remain open until the remote checkpoint task is resolved or the paper makes a final detector-policy decision.
