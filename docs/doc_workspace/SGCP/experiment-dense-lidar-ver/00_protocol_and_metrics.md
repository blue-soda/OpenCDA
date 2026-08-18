# Dense SGCP Experiment Protocol and Metrics

This directory contains the 2026-07-29 dense-LiDAR SGCP experiment data. All result tables are Markdown tables. Raw CSV/log artifacts are kept outside this clean package; the paper-facing data are the Markdown files.

## Common Dense Protocol

| Item | Value |
| --- | --- |
| Dataset | `D:\Data\Carla\2026_07_29_02_32_08` |
| Scenario | `v2xp_cluster_carla_dense` offline replay |
| CAV count | 20 |
| Evaluation frames | 41 frames, `000060` to `000140` |
| Table 1 / scale-sweep GT scope | Explicit `--gt-scope full-frame`; frame-level GT annotations are projected into each evaluated receiver/target pose so the AP denominator is scheduler-independent for a fixed active-CAV set |
| LiDAR | 32 channels, 320000 points/s, 20 Hz rotation, 50 m range |
| Detector/checkpoint | attentive-derived raw point-cloud detector |
| Fusion method for raw LiDAR | early fusion unless the row is box-only/local-only |
| Main bandwidth setting | 40 MHz |
| Target subchannels | 10 |
| Perception cycle | 100 ms/frame |
| Raw-LiDAR data-plane deadline | 60 ms/frame |
| Control-plane setting | Guard `1 ms`, zero-time send delay `0 ms`, activation-synchronized send |
| NS3-calibrated estimator | `tb_size=899 bytes`, `slot=0.5 ms`, `subchannel_prbs=10`, `PSSCH symbols=12`, `MCS=28` |
| PCS raw-LiDAR adaptation | `blind_spot_min_division=4`, `blind_spot_adjacency_radius=4`, `blind_spot_min_grids=128`, `communication_range=35 m` |
| EdgeCooper V2V adaptation | `communication_range=35 m`, `member_budget=3`, deadline-constrained grid budget as specified by table |
| SGCP dense headline candidate | `potential_verified_cov_coalition_game` clustering, `hybrid_round_robin_dynamic_marginal` scheduler with density-capped deterministic random point upload, all cluster heads as receivers, inter-cluster box NMS. The previous `dynamic_cv` row is retained as provenance in Table 1/3. |

## Metrics

| Metric | Meaning |
| --- | --- |
| AP@0.3 / AP@0.5 / AP@0.7 | Aggregate average precision at 3D IoU thresholds 0.3, 0.5, 0.7. The evaluator pools predictions over the evaluated receiver-frame samples unless explicitly stated. |
| Raw LiDAR Mbps | Point-cloud payload rate. |
| Box Mbps | Detection-box sharing overhead for late/global/inter-cluster box aggregation. |
| Total Mbps | `Raw LiDAR Mbps + Box Mbps`. |
| GFLOPs/frame | Input-adjusted detector-side forward compute per perception frame. It includes Conv/Deconv/Linear/BatchNorm/ReLU hooks and approximate PillarVFE point-feature FLOPs; it excludes voxelization/hash/scatter memory-index work, NMS, scheduling, CARLA, and control logic. |
| Mean/P95/Max link time | Trace-estimated communication time under the NS3-calibrated service model after per-link deadline trimming where applicable. |

## Clean Data Boundary

Use only the Markdown tables in this directory for dense-version evidence.
Generated figures, raw traces, older rows, and channel-count diagnostics should
not be mixed into paper-facing dense tables.
