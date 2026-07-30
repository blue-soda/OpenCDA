# VoxelNet Dense Table 1 Backbone Probe

Status: completed 41-frame dense Table 1 rerun with the VoxelNet early-fusion checkpoint. This is a multi-backbone sanity table, not yet the final multi-scene paper table.

Protocol: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), VoxelNet early-fusion checkpoint from `C:\Users\sakakibara\Downloads\voxelnet_early_fusion.zip`, patched only to use the existing OpenCOOD dummy validation root during model initialization. Network and SGCP protocol parameters match the dense Table 1 package: 40 MHz, 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, and 60 ms data-plane deadline. SGCP uses `rho_th=2`, density-capped upload `rho=2`, `head_rb_budget=2`, and 40 Mbps raw admission budget.

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Detector calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 0.98 | 0.98 | 0.89 | 337.08 | 0.00 | 337.08 | 159.42 | 1.00 |
| No collaboration | 0.43 | 0.41 | 0.28 | 0.00 | 0.00 | 0.00 | 3188.50 | 20.00 |
| Pure late | 0.97 | 0.94 | 0.71 | 0.00 | 1.40 | 1.40 | 3188.50 | 20.00 |
| FullPerception-PCS | 0.44 | 0.42 | 0.28 | 70.49 | 0.00 | 70.49 | 3188.50 | 20.00 |
| EdgeCooper-Pmax V2V adaptation | 0.47 | 0.45 | 0.31 | 86.30 | 0.00 | 86.30 | 3188.50 | 20.00 |
| PACP-LiDAR V2V adaptation | 0.40 | 0.38 | 0.26 | 86.30 | 0.00 | 86.30 | 3188.50 | 20.00 |
| SGCP | 0.88 | 0.86 | 0.62 | 27.84 | 0.69 | 28.52 | 1057.65 | 6.63 |

Interpretation:

- VoxelNet is usable as a second backbone: SGCP remains high (`0.88/0.86/0.62`) with much lower raw payload than PCS/EdgeCooper/PACP.
- The protocol-native raw-LiDAR baselines stay close to no-collaboration under the 60 ms deadline, while SGCP benefits from clustered receiver selection and inter-cluster box aggregation.
- Pure late is extremely strong with this VoxelNet checkpoint (`0.97/0.94/0.71`) because all CAVs perform local detection and then share boxes. This row is useful for compute/communication tradeoff discussion, but it weakens an AP-only SGCP dominance story. If this table is used in the paper, it should be interpreted jointly with `GFLOPs/frame`.
- VoxelNet FLOPs are hook-based detector-forward estimates. The current profiler reports `159.42 GFLOPs/forward` and does not expose a point-dependent VFE subtotal for `voxel_net`; therefore the table uses forward-equivalent GFLOPs rather than input-adjusted VFE scaling.

Artifacts:

- Runner: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\run_table1_voxelnet_dense.py`
- YAML: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\enable_coperception_voxelnet_early.yaml`
- Logs/traces/compute profile: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\voxelnet_table1_dense\`
