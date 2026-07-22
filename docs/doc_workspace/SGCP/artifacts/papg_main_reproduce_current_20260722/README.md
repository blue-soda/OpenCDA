# PAPG Main-Parameter Reproduction - 2026-07-22

Purpose: restore the SGCP-PAPG main operating parameters after the current-protocol diagnostic accidentally used the strict default `head_rb_budget=1` and, in one manual rerun, omitted the attentive detector YAML.

Dataset: `D:\Data\Carla\2026_07_15_01_26_56`, 20 CAVs, 41 frames.

Detector: `docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/enable_coperception_early_from_attentive.yaml`.

Restored SGCP parameters:

- `resource_allocation=perception_aware_potential_game`
- `clustering=coalition_game`
- `receiver_policy=all-cluster-heads`
- `inter_cluster_late_fusion=yes`
- `N_max=4`
- `rho_th=3`
- `head_rb_budget=2`
- no point cap

Current communication protocol:

- `bandwidth_mhz=40`
- `num_channels=10`
- `communication_deadline_ms=60`
- `channel_estimator=ns3`
- `ns3_tb_size_bytes=899`
- `ns3_slot_duration_ms=0.5`
- `ns3_subchannel_prbs=10`
- `ns3_symbols_per_slot=12`
- `ns3_mcs=28`

Primary rerun:

```powershell
conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 41 --fusion-method early --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml --sgcp-constrained --resource-allocation perception_aware_potential_game --clustering coalition_game --sgcp-receiver-policy all-cluster-heads --sgcp-inter-cluster-late-fusion --sgcp-upload-mode grid --sgcp-grid-selection-mode utility --sgcp-grid-score-mode utility --n-max 4 --rho-th 3 --head-rb-budget 2 --num-channels 10 --bandwidth-mhz 40 --communication-deadline-ms 60 --channel-estimator ns3 --ns3-tb-size-bytes 899 --ns3-slot-duration-ms 0.5 --ns3-subchannel-prbs 10 --ns3-symbols-per-slot 12 --ns3-mcs 28 --sgcp-trace-output docs\doc_workspace\SGCP\artifacts\papg_main_reproduce_current_20260722\papg_attentive_nmax4_bh2_ns3_40mhz_trace.csv --eval-stats-output docs\doc_workspace\SGCP\artifacts\papg_main_reproduce_current_20260722\papg_attentive_nmax4_bh2_ns3_40mhz_eval_stats.csv
```

Result:

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | Frame time mean / P95 / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| attentive, `N_max=4`, `B_h=2`, 40MHz/10ch/60ms NS3 estimator | 0.87 | 0.79 | 0.37 | 61.47 | 0.71 | 62.18 | 43.68 / 44.12 / 44.32 ms |

Interpretation:

- The restored main parameters satisfy the 60 ms communication window under the calibrated NS3 estimator.
- The previous current-protocol `SGCP-PAPG strict default` row (`0.64/0.60/0.25`, `37.05 Mbps`) is not the intended SGCP main operating point; it used the default `head_rb_budget=1`.
- A manual rerun without `--coperception-yaml` is invalid for attentive-table comparison. It produced lower AP because it silently fell back to the default detector.
- The remaining difference from the legacy `0.87/0.81/0.36`, `63.28 Mbps` row is consistent with the current NS3 deadline admission: selected grids per receiver row are reduced from about `97.22` to `61.67`, while total raw payload remains close.
