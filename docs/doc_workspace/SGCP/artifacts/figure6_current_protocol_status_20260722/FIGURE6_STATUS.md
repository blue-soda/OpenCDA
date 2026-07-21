# Figure6 Current-Protocol Status - 2026-07-22

Figure6 / bootstrap uncertainty is not regenerated as a current-protocol
figure in this pass.

Reason:

- The existing bootstrap source `uncertainty_bootstrap_attentive.csv` is a
  legacy `20 MHz / 10 ch` artifact.
- Current-protocol component tables are still diagnostic rather than final:
  strict SGCP default is weak, Table4 shows `N_max=2` dominates the default,
  and Table6 shows common global box aggregation makes PCS/EdgeCooper strong.
- Bootstrap confidence intervals should be computed only after the final
  paper-facing rows are frozen. Otherwise the figure would look precise while
  summarizing an operating point we already know is unresolved.

Current action:

- Keep the legacy bootstrap file as archived background only.
- Do not cite Figure6 as current-protocol evidence.
- Recompute bootstrap after deciding the final SGCP operating point and
  rerunning the associated Table2/Table3/Table5/TableA rows with per-sample
  eval stats.
