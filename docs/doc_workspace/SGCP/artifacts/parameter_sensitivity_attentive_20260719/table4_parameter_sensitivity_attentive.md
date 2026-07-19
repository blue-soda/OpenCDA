# SGCP Table 4 Parameter Sensitivity - Attentive Candidate

This artifact replaces the legacy `pointpillar_early_fusion` Table 4 numbers for forward paper writing. All rows use the same 20-CAV CARLA dump, aggregate pooled AP, `early_from_attentive` checkpoint, `perception_aware_potential_game`, `all-cluster-heads` receiver policy, inter-cluster box NMS late fusion, 20 MHz total bandwidth, and `B_h=2`.

## Main Table Candidate

| Parameter | Setting | AP@0.3 | AP@0.5 | AP@0.7 | Mbps | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rho_th` | 1.0 | 0.87 | 0.81 | 0.36 | 62.57 | Lower threshold sends nearly the same selected raw-LiDAR payload in this 41-frame dump. |
| `rho_th` | 2.0 | 0.87 | 0.81 | 0.36 | 62.54 | Same AP/Mbps as the main point; density threshold is not a fragile tuning knob here. |
| `rho_th` | 3.0 | 0.87 | 0.81 | 0.36 | 62.54 | Current SGCP-PAPG attentive main setting. |
| Channels | 5 | 0.74 | 0.61 | 0.24 | 31.12 | Severe subchannel pressure reduces selected sources/grids and hurts all IoU levels. |
| Channels | 10 | 0.87 | 0.81 | 0.36 | 62.54 | Main 20 MHz / 10-subchannel setting. |
| Channels | 20 | 0.88 | 0.81 | 0.36 | 67.33 | Extra channels add payload but only marginal AP@0.3 gain in this short dump. |

## Writing Boundary

The attentive Table 4 should make two conservative claims:

- SGCP-PAPG is not fragile to `rho_th` over `1-3` in this dataset because PAPG and the head budget dominate the actually selected sender/grid set.
- Communication resources matter: moving from 5 to 10 subchannels recovers AP, while 20 subchannels gives little additional return.

Do not overstate `N_max` or `T_min^stab` effects from this 41-frame sequence; those remain appendix/negative-result material until a longer dynamic scene is collected.
