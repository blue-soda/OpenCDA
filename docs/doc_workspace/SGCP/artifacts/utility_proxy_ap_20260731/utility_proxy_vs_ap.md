# Utility Proxy vs AP@0.5

Protocol: dense Table-3 SGCP-compatible scheduler comparison; 41 frames, 20 CAVs, PV coalition clustering, inter-cluster box NMS, 40 MHz / 10ch, 60 ms data-plane deadline. Utility is computed by replaying each scheduler without detector inference and evaluating the paper-facing early/late surrogate on the actual selected grid set.

| Method | Utility final | Utility gain | C gain | V gain | AP@0.5 | Avg grids | Links/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| SGCP | 0.1164 | 0.0489 | 0.0700 | 0.0510 | 0.81 | 318.07 | 10.00 |
| EdgeCooper-HD-Pmax | 0.1066 | 0.0391 | 0.1018 | 0.0738 | 0.81 | 578.24 | 16.00 |
| Random budget | 0.0805 | 0.0131 | 0.0205 | 0.0173 | 0.77 | 57.73 | 13.37 |
| FullPerception-PCS | 0.0702 | 0.0027 | 0.0114 | 0.0081 | 0.76 | 52.51 | 3.59 |
| Density greedy | 0.0773 | 0.0098 | 0.0150 | 0.0128 | 0.76 | 39.80 | 13.37 |
| Link-aware density | 0.0773 | 0.0098 | 0.0150 | 0.0128 | 0.76 | 39.80 | 13.37 |
| EdgeCooper-HD | 0.0713 | 0.0038 | 0.0103 | 0.0097 | 0.74 | 24.56 | 9.80 |
| Cluster-head late only | 0.0675 | 0.0000 | 0.0000 | 0.0000 | 0.71 | 0.00 | 0.00 |
| PACP-LiDAR | 0.0727 | 0.0052 | 0.0070 | 0.0056 | 0.70 | 20.05 | 10.00 |

Correlation with AP@0.5:

- Utility gain: Pearson `0.846`, Spearman `0.829`.
- Final utility: Pearson `0.846`, Spearman `0.829`.
- Selected grid count alone: Pearson `0.780`, Spearman `0.966`.

Interpretation: if utility correlation exceeds selected-grid correlation, the density-derived C/V surrogate explains AP better than a pure communication-volume proxy.
