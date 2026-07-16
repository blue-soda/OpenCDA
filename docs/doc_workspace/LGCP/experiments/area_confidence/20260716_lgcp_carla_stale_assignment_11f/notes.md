# LGCP Stale Assignment Sensitivity

This run compares current area quality against area-confidence reports from earlier frames.

- confidence_field: `density_distance`
- quality_field: `recall_05`
- lags: `0,1,2,3`
- top_k: `40`

| Lag | Samples | Noisy-or Spearman | Top-k Jaccard mean |
| --- | ---: | ---: | ---: |
| 0 | 354 | 0.584992 | 1.000000 |
| 1 | 321 | 0.527720 | 0.911095 |
| 2 | 289 | 0.529556 | 0.857818 |
| 3 | 257 | 0.447925 | 0.805484 |
