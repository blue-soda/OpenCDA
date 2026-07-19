# SGCP Attentive Paper Number Audit

Generated: 2026-07-19

This audit checks the current `C:\Workspace\icdcs-paper\SGCP\main.tex` experiment numbers after promoting the attentive checkpoint artifacts to the forward-writing candidate.

Machine-readable audit:

```text
docs\doc_workspace\SGCP\artifacts\paper_number_audit_attentive_20260719\paper_number_audit_attentive.csv
```

## Checked Items

- Table 1 now matches `artifacts/attentive_protocol_20260719/protocol_native_attentive_manifest.csv`.
- Table 3 now matches `artifacts/attentive_scheduler_comparison_20260719/scheduler_comparison_attentive_manifest.csv`.
- Pure late communication now uses the attentive 80 B/box one-hop broadcast estimate: `1.37 Mbps`.
- FullPerception-PCS now uses the paper-faithful PCS scheduling + raw-LiDAR full-sender adaptation: `0.63/0.49/0.17`, `32.06 Mbps`. The stricter PCS grid replay is kept as a boundary result, not the Table 1 anchor.
- Paper figure PDFs were copied from the attentive artifacts into `C:\Workspace\icdcs-paper\SGCP\fig`:
  - `sgcp_protocol_breakdown.pdf`
  - `sgcp_fusion_contribution.pdf`
  - `sgcp_pareto_ap03.pdf`
  - `sgcp_pareto_ap07.pdf`

## Boundary

The legacy `pointpillar_early_fusion` Table 1/3 and Figure 1/2/3 artifacts are retained for checkpoint-reference and reproducibility, but they should not be used as the default forward-writing results.

The paper directory is outside the OpenCDA git repository, so `main.tex` and copied paper figures must be committed or archived separately if the paper project is version controlled.
