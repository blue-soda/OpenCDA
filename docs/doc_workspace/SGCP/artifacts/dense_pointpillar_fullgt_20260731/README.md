# Dense PointPillar Full-Frame-GT Table 1 Rerun

Purpose: fix the Dense Roundabout / PointPillar-Attentive Table 1 AP-scope issue where No collaboration was evaluated against receiver-local GT while cooperative baselines were evaluated against scheduler-expanded helper GT.

Protocol:

- Dataset: `D:\Data\Carla\2026_07_29_02_32_08`
- Frames: `000060-000140`, 41 frames, 20 CAVs
- Detector: `docs/doc_workspace/SGCP/artifacts/early_from_late_checkpoint_20260719/enable_coperception_early_from_attentive.yaml`
- Channel estimator: 40 MHz / 10 target subchannels, `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`
- Data-plane deadline: 60 ms
- GT scope: `--gt-scope full-frame`
- Baseline fallback: `--local-preserving-output` for `pcs`, `edgecooper_pmax`, and `pacp_lidar`

Main rerun result:

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 0.90 | 0.90 | 0.80 | 337.08 | 0.00 | 337.08 | 90.55 | 1.00 |
| No collaboration | 0.30 | 0.27 | 0.18 | 0.00 | 0.00 | 0.00 | 1788.73 | 20.00 |
| Pure late | 0.82 | 0.74 | 0.53 | 0.00 | 1.95 | 1.95 | 1745.11 | 19.51 |
| FullPerception-PCS | 0.33 | 0.29 | 0.20 | 70.37 | 0.00 | 70.37 | 2415.03 | 27.00 |
| EdgeCooper-Pmax V2V adaptation | 0.40 | 0.37 | 0.25 | 86.30 | 0.00 | 86.30 | 3497.04 | 39.10 |
| PACP-LiDAR V2V adaptation | 0.31 | 0.28 | 0.19 | 86.30 | 0.00 | 86.30 | 2419.44 | 27.05 |
| SGCP | 0.86 | 0.82 | 0.59 | 27.84 | 0.86 | 28.69 | 593.43 | 6.63 |

Key conclusion:

- Full-frame GT makes Table 1 scheduler-independent and removes the false appearance that No collaboration is stronger than raw-LiDAR baselines.
- GFLOPs remain input-adjusted and non-identical across PCS, EdgeCooper-Pmax, and PACP-LiDAR.
- This full-frame-GT rerun is the paper-facing Dense Roundabout / PointPillar-Attentive Table 1 result.
