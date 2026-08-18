# Utility Proxy vs AP@0.5

Protocol: dense Table-3 SGCP-compatible scheduler comparison; 11 utility-proxy validation frames sampled every 4 frames from the 41-frame evaluation window, 20 CAVs, PV coalition clustering, inter-cluster box NMS, 40 MHz / 10ch, 60 ms data-plane deadline. Utility is computed by replaying each scheduler without detector inference and evaluating the paper-facing early/late surrogate on the actual selected grid set.

| Method | Utility final | Utility gain | Dynamic marginal gain | AP@0.5 | Avg grids | Links/frame |
|---|---:|---:|---:|---:|---:|---:|
| SGCP | 0.1078 | 0.0408 | 0.0660 | 0.81 | 317.55 | 10.00 |
| EdgeCooper-HD-Pmax | 0.0981 | 0.0312 | 0.0950 | 0.81 | 558.82 | 15.36 |
| Random budget | 0.0791 | 0.0121 | 0.0190 | 0.77 | 55.18 | 13.45 |
| FullPerception-PCS | 0.0692 | 0.0022 | 0.0096 | 0.76 | 43.73 | 3.18 |
| Density greedy | 0.0767 | 0.0097 | 0.0149 | 0.76 | 40.09 | 13.45 |
| Link-aware density | 0.0767 | 0.0097 | 0.0149 | 0.76 | 40.09 | 13.45 |
| EdgeCooper-HD | 0.0710 | 0.0040 | 0.0101 | 0.74 | 23.55 | 9.73 |
| Cluster-head late only | 0.0670 | 0.0000 | 0.0000 | 0.71 | 0.00 | 0.00 |
| PACP-LiDAR | 0.0722 | 0.0052 | 0.0070 | 0.70 | 20.00 | 10.00 |

Correlation with AP@0.5:

- Utility gain: Pearson `0.846`, Spearman `0.829`.
- Final utility: Pearson `0.846`, Spearman `0.829`.
- Selected grid count alone: Pearson `0.782`, Spearman `0.966`.

Interpretation: if utility correlation exceeds selected-grid correlation, the density-derived C/V surrogate explains AP better than a pure communication-volume proxy.
