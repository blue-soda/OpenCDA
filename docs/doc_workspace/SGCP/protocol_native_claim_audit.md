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
| Pure late baseline detector/checkpoint fairness | Pass | `detector_checkpoint_fairness.md` fixes the policy: Pure late main-table row uses the same early checkpoint as SGCP with singleton local inference plus `naive_late_fusion()`; actual late checkpoint remains sensitivity/reference only. |
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

1. Early-fusion checkpoint risk:
   - Current SGCP raw point-cloud early fusion is limited by the existing early checkpoint.
   - Remote training on `mindspore-187:/data2/gzc/sgcp_early_train/` is still waiting for GPU; if a better checkpoint is recovered, Table 1 and dependent figures must be regenerated under a new artifact index version.
   - Recovery protocol is now documented in `early_checkpoint_recovery.md`.

2. Main-claim wording:
   - Do not claim SGCP has highest AP and lowest Mbps across all baselines.
   - Safe claim: SGCP-PAPG achieves a favorable RSU-free V2V raw-LiDAR AP/Mbps tradeoff, matches EdgeCooper-HD on AP@0.3/AP@0.5 with less payload, but has lower AP@0.7 because EdgeCooper-HD uses global edge-side assignment.

## Target Update Rationale

The first four P1 acceptance items can be marked complete because the current paper text, `detector_checkpoint_fairness.md`, and artifact index now explicitly address them. The only remaining P1 blocker is early checkpoint strength; it remains open until the remote checkpoint task is resolved or recorded as blocked by external GPU availability.
