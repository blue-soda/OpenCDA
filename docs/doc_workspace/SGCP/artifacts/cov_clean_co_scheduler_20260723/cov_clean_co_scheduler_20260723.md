# Clean C->O Scheduler Ablation - 2026-07-23

## Protocol

- Clustering: `cov_coalition_game`, default V-only coalition utility
- Scheduler: historical `cov_potential_game` C->O ablation path from commit
  `241653a`; the current formal scheduler has since been cleaned back to C->V
  only.
- Coverage stage: select top grids by `C`
- Target/quality stage: select top grids by `O`
- Candidate grids: `C + O > 0`
- No connected-component prior, no top-U prior, no mixed O/V target score
- Detector/network defaults: attentive yaml, NS3 estimator, `tb_size=899`, `symbols=12`, `MCS=28`
- Dataset: `D:\Data\Carla`, scenario `2026_07_15_01_26_56`, 41 frames
- SGCP parameters: 40 MHz, 10 target subchannels, `N_max=4`, `rho_th=3`, `head_rb_budget=2`, scheduler budget `200 ms`

## Result

| Variant | Cluster objective | Coverage stage | Target stage | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Uploaded senders/sample | Source CAVs/sample | Selected grids/sample |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean C->V | V | C | V | 0.87 | 0.80 | 0.36 | 60.18 | 1.67 | 2.67 | 85.73 |
| Clean C->O | V | C | O | 0.87 | 0.80 | 0.36 | 60.70 | 1.67 | 2.67 | 94.38 |

## Interpretation

Under V-only coalition formation, replacing the second-stage target objective
from V to O does not change AP on this scene, but it increases selected grids
and raw payload slightly. This supports the current simplification direction:
after multi-view clusters are fixed, raw sender quality `O=q_i(g)` and
receiver-overlap quality `V=q_i(g) * 1[q_h(g)>0]` rank many candidate grids
similarly. The V objective is marginally more communication-efficient here.

## Reproduction

Historical reproduction at commit `241653a`:

```powershell
$env:OPENCDA_COV_TARGET_TERM='object'
conda run -n opencda python -m opencda.tools.offline_inference --dataset-root D:\Data\Carla --scenario-id 2026_07_15_01_26_56 --ego-cav-id 1 --max-frames 41 --fusion-method early --sgcp-constrained --clustering cov_coalition_game --resource-allocation cov_potential_game --sgcp-receiver-policy all-cluster-heads --sgcp-upload-mode grid --sgcp-inter-cluster-late-fusion --sgcp-grid-selection-mode utility --sgcp-grid-score-mode utility --bandwidth-mhz 40 --num-channels 10 --communication-deadline-ms 200 --n-max 4 --rho-th 3 --head-rb-budget 2
Remove-Item Env:\OPENCDA_COV_TARGET_TERM
```
