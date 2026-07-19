# SGCP Protocol-Native Claim Audit

更新时间：2026-07-19

本文档审计 `target.md` P1 的 Table 1 叙事是否已经满足“完整系统比较、baseline 分层、Pure late 边界、FullPerception/EdgeCooper 复现解释”的写作要求。审计对象为：

```text
C:\Workspace\icdcs-paper\SGCP\main.tex
docs\doc_workspace\SGCP\artifacts\table1_protocol_20260719\protocol_native_manifest.csv
docs\doc_workspace\SGCP\artifacts\attentive_protocol_20260719\protocol_native_attentive_manifest.csv
docs\doc_workspace\SGCP\paper_artifact_index.md
docs\doc_workspace\SGCP\main_table_candidate.md
```

## Audit Summary

| Requirement | Status | Evidence |
| --- | --- | --- |
| Table must explain SGCP's complete-system advantage, not a single scheduler win | Pass with caveats | `main.tex` Table 1 is protocol-native and text attributes gains to clustering, perception-aware point-cloud selection, and inter-cluster late fusion. |
| If AP@0.3 comes mainly from late fusion, the text must say it is a system-protocol advantage | Pass | `main.tex` explicitly states late fusion restores network-level coverage and low-IoU recall; scheduler comparison is separated into its own table. |
| Weak/atypical FullPerception-PCS or EdgeCooperV2V+ results must be explained before entering the table | Pass | FullPerception-PCS is separated from Full20Early upper reference and described as repaired/tuned built-in PCS; EdgeCooper-HD is labeled edge-assisted/global assignment reference. |
| Pure late baseline detector/checkpoint fairness | Pass | `detector_checkpoint_fairness.md` fixes the policy: Pure late main-table row uses the same checkpoint as SGCP with singleton local inference plus `naive_late_fusion()`; current forward-writing candidate uses attentive checkpoint for all methods; actual late checkpoint remains sensitivity/reference only. |
| Early-fusion checkpoint strength | Open but not blocking current candidate | Attentive candidate improves SGCP-vs-Pure-late/EdgeCooper narrative and has complete artifacts. Remote fine-tune watcher is still pending GPU; any recovered checkpoint must trigger a new artifact version. |

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
   - Current forward-writing candidate uses attentive checkpoint consistently across Table 1/2/3/Figure 1/2/3/4.
   - Remote training on `mindspore-187:/data2/gzc/sgcp_early_train/` is still waiting for GPU; if a better checkpoint is recovered, Table 1 and dependent figures must be regenerated under a new artifact index version.
   - Recovery protocol is now documented in `early_checkpoint_recovery.md`.

2. Main-claim wording:
   - Do not claim SGCP has highest AP and lowest Mbps across all baselines.
   - Safe claim: SGCP-PAPG achieves a favorable RSU-free V2V raw-LiDAR AP/Mbps tradeoff, matches EdgeCooper-HD on AP@0.3/AP@0.5 with less payload, but has lower AP@0.7 because EdgeCooper-HD uses global edge-side assignment.

## Target Update Rationale

The first four P1 acceptance items can be marked complete because the current paper text, `detector_checkpoint_fairness.md`, and artifact index now explicitly address them. The remaining checkpoint task is no longer a blocker for the attentive candidate, but it remains an improvement trigger: if the remote checkpoint task resolves, a new full artifact set is required before replacing the current numbers.
