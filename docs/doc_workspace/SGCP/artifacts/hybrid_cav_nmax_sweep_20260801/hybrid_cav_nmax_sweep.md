# Hybrid Scheduler CAV Count / Nmax Sweep

Protocol: dense Town03 `2026_07_29_02_32_08`, 41 frames `000060-000140`, attentive-to-early checkpoint, full-frame GT, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, `rho_th=2`, density cap `rho=2`, raw-LiDAR admission budget 40 Mbps, `head_rb_budget=2`, inter-cluster box NMS. New SGCP rows use paired `(N,N_max)` settings `(5,2)`, `(10,3)`, `(15,4)` and `hybrid_round_robin_dynamic_marginal`; `old_dynamic_cv` and `pure_late` are parsed from `dense_cav_sweep_fullgt_20260801`.

| Method | N CAVs | N_max | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | Samples | Avg source CAVs | Avg selected grids | P95 data time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_late | 5 |  | 0.72 | 0.64 | 0.42 | 0.00 | 0.57 | 0.57 | 205 | 1.00 | 0.00 | 0.00 |
| old_dynamic_cv | 5 |  | 0.55 | 0.49 | 0.31 | 9.09 | 0.28 | 9.36 | 56 | 2.73 | 49.70 | 40.70 |
| hybrid_round_robin_dynamic_marginal | 5 | 2 | 0.70 | 0.62 | 0.41 | 6.49 | 0.39 | 6.88 | 123 | 1.67 | 22.00 | 27.83 |
| pure_late | 10 |  | 0.85 | 0.78 | 0.54 | 0.00 | 1.22 | 1.22 | 410 | 1.00 | 0.00 | 0.00 |
| old_dynamic_cv | 10 |  | 0.79 | 0.73 | 0.52 | 17.65 | 0.56 | 18.21 | 123 | 2.73 | 55.00 | 37.40 |
| hybrid_round_robin_dynamic_marginal | 10 | 3 | 0.83 | 0.76 | 0.54 | 19.20 | 0.76 | 19.95 | 164 | 2.50 | 49.15 | 37.18 |
| pure_late | 15 |  | 0.86 | 0.77 | 0.52 | 0.00 | 1.57 | 1.57 | 615 | 1.00 | 0.00 | 0.00 |
| old_dynamic_cv | 15 |  | 0.81 | 0.73 | 0.51 | 22.97 | 0.64 | 23.60 | 172 | 2.69 | 51.59 | 40.68 |
| hybrid_round_robin_dynamic_marginal | 15 | 4 | 0.84 | 0.76 | 0.57 | 23.44 | 0.73 | 24.17 | 205 | 2.51 | 48.91 | 40.68 |
