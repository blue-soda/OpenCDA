# C/V Coalition Ablation - 2026-07-23

## Code audit

- Formal scheduler: `cov_potential_game`, strict two-stage C/V objective.
- Scheduler stage 1 scores sender-grid actions only by `C=q_i(g)(1-q_h(g))`.
- Scheduler stage 2 scores sender-grid actions only by `V=q_i(g) * 1[q_h(g)>0]`.
- Scheduler candidates require `C+V>0`.
- No connected-component prior, top-U prior, spatial-diversity prior, or O/object switch remains in the formal scheduler path.
- Formal coalition game: `cov_coalition_game`, default `OPENCDA_COV_CLUSTER_TERMS=view`.
- Coalition-side `C/V` uses the same grid-level definitions as scheduling, replacing receiver quality `q_h(g)` with coalition quality `q_S(g)=max_{j in S} q_j(g)`.
- Coalition utility sums grid-level terms over candidate grids. Scheduler utility sums the same terms over the selected grids of a sender-head action.

## Protocol

- Dataset: `D:\Data\Carla`, scenario `2026_07_15_01_26_56`, frames `000060` to `000140`, 41 frames.
- Detector: attentive early-fusion YAML default.
- Network estimator: NS3-calibrated defaults, `tb_size=899`, `symbols=12`, `MCS=28`.
- SGCP: 40 MHz, 10 target subchannels, `N_max=4`, `rho_th=3`, `head_rb_budget=2`, scheduler budget `200 ms`.
- Fusion: intra-cluster raw-LiDAR early fusion plus inter-cluster box aggregation.

## Results

| Variant | Coalition terms | Scheduler stage 1 | Scheduler stage 2 | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Late sources/frame | Source CAVs/sample | Selected grids/sample | Mean/Max comm time (ms) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Formal SGCP C/V | V | C | V | 0.87 | 0.80 | 0.36 | 60.18 | 6.00 | 2.67 | 85.73 | 41.70 / 45.03 |
| Coalition C+V ablation | C+V | C | V | 0.86 | 0.75 | 0.31 | 60.18 | 6.88 | 2.45 | 72.39 | 43.06 / 45.03 |

## Interpretation

Adding C to coalition formation creates more cluster heads and smaller early-fusion groups. It keeps the raw payload almost unchanged under the same channel budget, but reduces source CAVs and selected grids per inference sample, which hurts AP@0.5 and AP@0.7. This supports the formal design choice: coalition formation should stay V-only to build multi-view groups for early-fusion quality, while coverage repair should be handled by the scheduler's first C stage.

## Reproduction

```powershell
$env:OPENCDA_COV_CLUSTER_TERMS='coverage+view'
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 41 --fusion-method early --sgcp-constrained --clustering cov_coalition_game --resource-allocation cov_potential_game --sgcp-receiver-policy all-cluster-heads --sgcp-upload-mode grid --sgcp-inter-cluster-late-fusion --sgcp-grid-selection-mode utility --sgcp-grid-score-mode utility --bandwidth-mhz 40 --num-channels 10 --communication-deadline-ms 200 --n-max 4 --rho-th 3 --head-rb-budget 2
Remove-Item Env:\OPENCDA_COV_CLUSTER_TERMS
```
