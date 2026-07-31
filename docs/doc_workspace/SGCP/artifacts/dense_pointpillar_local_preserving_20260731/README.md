# Dense PointPillar Local-Preserving Table 1 Rerun

Purpose: fix the Dense Roundabout / PointPillar-Attentive Table 1 issues where protocol-native baselines could lose receiver-local detections and where GFLOPs were nearly identical despite different fused point-cloud sizes.

Protocol:

- Dataset: `D:\Data\Carla\2026_07_29_02_32_08`
- Frames: `000060-000140`, 41 frames, 20 CAVs
- Detector: `docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/enable_coperception_early_from_attentive.yaml`
- Channel estimator: 40 MHz / 10 target subchannels, `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`
- Data-plane deadline: 60 ms
- Baseline fix: `--local-preserving-output` is applied only to `pcs`, `edgecooper_pmax`, and `pacp_lidar`

Main rerun result:

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 0.90 | 0.90 | 0.80 | 337.08 | 0.00 | 337.08 | 90.55 | 1.00 |
| No collaboration | 0.55 | 0.50 | 0.34 | 0.00 | 0.00 | 0.00 | 1788.73 | 20.00 |
| Pure late | 0.81 | 0.73 | 0.52 | 0.00 | 1.95 | 1.95 | 1745.11 | 19.51 |
| FullPerception-PCS | 0.54 | 0.49 | 0.34 | 70.49 | 0.00 | 70.49 | 2412.84 | 26.98 |
| EdgeCooper-Pmax V2V adaptation | 0.52 | 0.47 | 0.32 | 86.30 | 0.00 | 86.30 | 3497.04 | 39.10 |
| PACP-LiDAR V2V adaptation | 0.52 | 0.47 | 0.32 | 86.30 | 0.00 | 86.30 | 2419.44 | 27.05 |
| SGCP | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 6.63 |

Key conclusion:

- GFLOPs are now input-adjusted and no longer collapse to the same value for EdgeCooper-Pmax and PACP-LiDAR.
- Local-preserving receiver output does not materially improve protocol-native PCS/EdgeCooper/PACP AP on this dense PointPillar setting, so their weak AP should be treated as an empirical limitation of these protocol-native raw-LiDAR adaptations rather than an evaluation-entry bug.
- Non-Pmax EdgeCooper is not included in the clean paper-facing Table 1.
