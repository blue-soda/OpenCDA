# Dense-LiDAR Table 1 Protocol-Native Baselines

Status: updated to the current paper-facing full-frame GT protocol.

Protocol: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060`-`000140`), PointPillar-Attentive early checkpoint, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms raw-LiDAR data-plane deadline, and explicit:

```text
--gt-scope full-frame
```

Under `full-frame` GT, each evaluated receiver sample is compared with the same frame-level GT set projected to that receiver/target pose. This prevents the AP denominator from changing with the number of selected helpers or receiver samples. Results from old `--gt-scope sample` runs should not be mixed with this table.

SGCP uses the rho-Pareto selected density threshold `rho_th=2`, density-capped raw-LiDAR upload, potential-verified coalition formation, and inter-cluster box NMS. The current best SGCP scheduler candidate is `hybrid_round_robin_dynamic_marginal`; the previous `dynamic_cv` row is retained for continuity. PCS and EdgeCooper-Pmax use K=2 receiver-side concurrency to align with SGCP.

| Method | Late fusion | Clustering | Scheduler/protocol | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Detector calls/frame | Avg uploaded CAVs/call | Avg selected grids/call | P95 data-plane time | Max data-plane time | Scope |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Centralized all-in-one raw-LiDAR upper ref | none | all_in_one | full_cluster | 0.90 | 0.90 | 0.80 | 337.08 | 0.00 | 337.08 | 90.55 | 1.00 | 19.00 | 0.00 | 127.33 ms | 127.53 ms | centralized upper reference; not a feasible V2V all-receiver baseline |
| No collaboration | none | singleton | local_detection | 0.30 | 0.27 | 0.18 | 0.00 | 0.00 | 0.00 | 1788.73 | 20.00 | 0.00 | 0.00 | 0.00 ms | 0.00 ms | lower reference; local detector only under full-frame GT |
| Pure late | prediction_nms | singleton | local_detection + box_nms | 0.82 | 0.74 | 0.53 | 0.00 | 1.95 | 1.95 | 1745.11 | 19.51 | 0.00 | 0.00 | 0.00 ms | 0.00 ms | box-level sharing reference; same attentive detector |
| FullPerception-PCS | none | singleton | fullperception_pcs | 0.33 | 0.29 | 0.21 | 70.49 | 0.00 | 70.49 | 1790.47 | 20.00 | 0.50 | 6.82 | 60.00 ms | 60.00 ms | raw-LiDAR adaptation, K=2, div4/radius4/min128/range35, density_threshold=2.0 for blind spots, no local-preserving extra detector |
| EdgeCooper-Pmax V2V adaptation | none | singleton | selective_edgecooper_global_pmax | 0.40 | 0.37 | 0.26 | 86.30 | 0.00 | 86.30 | 1790.59 | 20.00 | 2.50 | 27.17 | 60.00 ms | 60.00 ms | EdgeCooper-style partial raw-point upload; K=2, 35m range, 60ms global admission, Pmax-style density cap rho=2, no local-preserving extra detector |
| PACP-LiDAR V2V adaptation | none | singleton | selective_pacp_lidar | 0.40 | 0.37 | 0.26 | 86.30 | 0.00 | 86.30 | 1790.59 | 20.00 | 2.59 | 33.99 | 60.00 ms | 60.00 ms | paper-aligned LiDAR adaptation of PACP BEV-match priority plus perceptual-region coverage; singleton global candidate fallback, K=2, 35m range, 60ms global admission, density-cap compression proxy rho=2 |
| SGCP | inter_cluster_nms | potential_verified_cov_coalition_game | hybrid_round_robin_dynamic_marginal | 0.88 | 0.82 | 0.61 | 28.42 | 0.89 | 29.31 | 593.34 | 6.63 | 1.51 | 49.12 | 37.39 ms | 37.47 ms | proposed hierarchical method, first round-robin pass then dynamic-marginal greedy scheduling; rho_th=2, N_max=5 |
| SGCP previous scheduler | inter_cluster_nms | potential_verified_cov_coalition_game | dynamic_cv | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 6.63 | 1.51 | 47.89 | 36.73 ms | 37.47 ms | retained previous dense headline for comparison; same rho_th=2, N_max=5 protocol |

Notes:

- `Centralized all-in-one` is an upper reference, not a feasible V2V baseline; it requires collecting all 20 dense point clouds at one receiver and exceeds the 60 ms data-plane deadline.
- `No collaboration` drops sharply under full-frame GT because each local detector is evaluated against the frame-level GT set rather than its local sample-only GT. This is the intended lower reference for the current main-table protocol.
- `Pure late` has very low communication because it shares boxes only, but it requires almost all CAVs to run detector inference (`19.51` calls/frame, `1745.11` GFLOPs/frame).
- `FullPerception-PCS`, `EdgeCooper-Pmax`, and `PACP-LiDAR` are protocol-native singleton-receiver raw-LiDAR adaptations under the same full-frame GT denominator. Their dense raw-LiDAR uploads do not match SGCP's AP despite higher payload and compute. The PCS and EdgeCooper-Pmax rows were rerun without `--local-preserving-output`, so Table 1 does not include hidden local detector + box-NMS refinement under `Late fusion = none`.
- `PACP-LiDAR` was rerun after aligning the LiDAR proxy with PACP's original BEV-match priority plus perceptual-region utility and applying a density-capped upload as the raw-LiDAR counterpart of PACP's compression/rate control.
- The non-Pmax EdgeCooper full-grid row is deprecated for paper-facing comparison and is intentionally omitted here. EdgeCooper's original paper uses per-pillar partial point upload; the `EdgeCooper-Pmax` row is the paper-facing adaptation.
- The new SGCP hybrid scheduler is the best feasible V2V method in this table: compared with Pure late it improves AP by `+0.06/+0.08/+0.08`, and compared with EdgeCooper-Pmax it uses far less payload (`29.31` vs `86.30` Mbps) and detector compute (`593.34` vs `1790.59` GFLOPs/frame).
- `SGCP previous scheduler` is retained only to show continuity with the earlier dense headline. It should not be mixed with the hybrid row as a separate baseline.

Provenance:

| Artifact | Path |
| --- | --- |
| Full-frame GT traces and logs | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_pointpillar_fullgt_20260731\table1_41f` |
| Compute summary | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_pointpillar_fullgt_20260731\table1_41f\compute_all_41f.csv` |
| PCS and EdgeCooper-Pmax no-local-preserving rerun | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\table1_no_local_preserving_20260801` |
| Hybrid scheduler trace and log | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f` |
| PACP paper-aligned rerun | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\pacp_lidar_paper_fix_20260801\table1_protocol_native` |

Compute provenance was generated by `opencda.tools.sgcp_compute_profile`. FLOPs include Conv2d/Conv3d/ConvTranspose2d/Linear/BatchNorm/ReLU hooks, sparse-3D active-output proxy if present, and PillarVFE/MeanVFE elementwise estimates; voxelization/hash/scatter memory/index operations are excluded.
