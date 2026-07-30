# Town06 Multi-Scene Table 1 Results

Status: completed. This artifact covers the requested second scenario: Town06, 10 CAVs, 20 background vehicles, dense LiDAR, 31 frames. Both PointPillar-attentive and VoxelNet early-fusion backbones were evaluated under the same Table 1 protocol.

Dataset: `D:\Data\Carla\2026_07_31_02_24_35`.

Scenario: `v2xp_cluster_town06_dense`, Town06, 1 explicit CAV plus 9 managed traffic CAVs, 20 unmanaged background vehicles. Dense LiDAR matches `v2xp_cluster_carla_dense`: 32 channels, 320000 points/s, 20 Hz rotation, 50 m range. DataDumper output contains 10 CAV folders and 31 frames per CAV.

Common protocol: 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, and 60 ms data-plane deadline. SGCP uses potential-verified clustering, dynamic C/V scheduler, density cap `rho=2`, `head_rb_budget=2`, and `N_max=2` derived for `N=10`.

## Town06 PointPillar-Attentive

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 1.00 | 1.00 | 0.98 | 147.23 | 0.00 | 147.23 | 90.59 | 1.00 |
| No collaboration | 0.74 | 0.71 | 0.54 | 0.00 | 0.00 | 0.00 | 894.87 | 10.00 |
| Pure late | 0.91 | 0.89 | 0.82 | 0.00 | 0.79 | 0.79 | 894.87 | 10.00 |
| FullPerception-PCS | 0.71 | 0.69 | 0.54 | 36.35 | 0.00 | 36.35 | 895.14 | 10.00 |
| EdgeCooper-Pmax V2V adaptation | 0.83 | 0.82 | 0.71 | 69.14 | 0.00 | 69.14 | 895.39 | 10.00 |
| PACP-LiDAR V2V adaptation | 0.74 | 0.72 | 0.56 | 81.15 | 0.00 | 81.15 | 895.48 | 10.00 |
| SGCP | 0.96 | 0.94 | 0.81 | 15.42 | 0.50 | 15.92 | 470.65 | 5.26 |

## Town06 VoxelNet

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Calls/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized all-in-one raw-LiDAR upper ref | 1.00 | 1.00 | 0.99 | 147.23 | 0.00 | 147.23 | 159.42 | 1.00 |
| No collaboration | 0.59 | 0.59 | 0.47 | 0.00 | 0.00 | 0.00 | 1594.25 | 10.00 |
| Pure late | 0.97 | 0.97 | 0.88 | 0.00 | 0.59 | 0.59 | 1594.25 | 10.00 |
| FullPerception-PCS | 0.59 | 0.58 | 0.48 | 36.35 | 0.00 | 36.35 | 1594.25 | 10.00 |
| EdgeCooper-Pmax V2V adaptation | 0.75 | 0.74 | 0.64 | 69.14 | 0.00 | 69.14 | 1594.25 | 10.00 |
| PACP-LiDAR V2V adaptation | 0.63 | 0.63 | 0.51 | 81.15 | 0.00 | 81.15 | 1594.25 | 10.00 |
| SGCP | 0.96 | 0.96 | 0.83 | 15.42 | 0.40 | 15.83 | 838.27 | 5.26 |

Interpretation:

- The Town06 multi-scene result is stronger than the original dense roundabout scene for SGCP: SGCP reaches high AP with about `15.9 Mbps`, much lower than PCS/EdgeCooper/PACP and far lower compute than all-CAV pure late.
- With PointPillar-attentive, SGCP improves over Pure late at AP@0.3/AP@0.5 and is close at AP@0.7, while using about half the detector compute.
- With VoxelNet, Pure late remains strongest at AP@0.3/AP@0.5, but SGCP is very close and uses about half the detector compute and only about `15.8 Mbps`.
- This scenario is suitable as the requested multi-scene robustness evidence. The VoxelNet table also satisfies the requested new-backbone evidence.

Artifacts:

- Scenario code: `C:\Workspace\OpenCDA\opencda\scenario_testing\v2xp_cluster_town06_dense.py`
- Scenario YAML: `C:\Workspace\OpenCDA\opencda\scenario_testing\config_yaml\v2xp_cluster_town06_dense.yaml`
- Runner: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\run_table1_scene_backbone.py`
- PointPillar logs/traces: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\town06_pointpillar_table1\`
- VoxelNet logs/traces: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\multi_backbone_20260731\town06_voxelnet_table1\`
