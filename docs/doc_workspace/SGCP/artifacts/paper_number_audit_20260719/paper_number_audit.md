# Paper Number Audit

Date: 2026-07-19

This audit checks the current SGCP `main.tex` numbers against the committed artifact manifests. It also records the network-configuration label issue found during the heartbeat run.

Machine-readable source:

```text
docs\doc_workspace\SGCP\artifacts\paper_number_audit_20260719\paper_number_audit.csv
```

## Result

- Table 1 AP values match `protocol_native_manifest.csv`.
- Table 1 communication values match the manifest after applying the documented Pure late exception: Pure late has `0` raw-LiDAR payload, but the paper reports the `80 B/box` one-hop prediction-box broadcast estimate, `0.74 Mbps`.
- Table 3 scheduler-comparison rows match `scheduler_comparison_manifest.csv`.
- Table 4 AP/Mbps values match `table4_parameter_sensitivity.csv`.
- A metadata-label issue was found and corrected: Table 4 channel-count rows were labeled `5/10/20 ch / 40 MHz`, while the reproduced commands for the current paper runs use `--bandwidth-mhz 20`. The labels now read `5/10/20 ch / 20 MHz`.

## Boundary

The OpenCDA configuration still has legacy defaults that can suggest 40 MHz:

- `enable_network.yaml`: `subchannel_bandwidth: 4` and `subchannel_num: 10`;
- `Params.bandwidth_all = 40` in `opencda/core/clustering/utils/common.py`.

The current paper artifacts, however, are based on reproduced offline commands that explicitly pass `--bandwidth-mhz 20 --num-channels 10` for the main SGCP-compatible runs. Older trace CSVs did not record `bandwidth_mhz`, so the command logs remain the primary source for the current 20 MHz label.

## Tooling Fix

`opencda.tools.sgcp_aggregate_ap_manifest` now includes `num_channels` and `bandwidth_mhz` in its output schema when future traces provide these fields. This prevents the same ambiguity from recurring after new runs or fine-tuned checkpoint recovery.
