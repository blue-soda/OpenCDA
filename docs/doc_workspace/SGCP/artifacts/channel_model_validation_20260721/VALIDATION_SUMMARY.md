# Channel Model Validation 2026-07-21

## OpenCDA estimator smoke

| Variant | Rows | Nonzero rows | Total bytes | Mean frame time ms | Max frame time ms | Estimator | Bandwidth MHz | TB bytes |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| SGCP-PAPG logical estimator | 6 | 6 | 305520 | 203.68 | 243.65 | logical | 20.0 | 400 |
| SGCP-PAPG NS3 estimator | 6 | 6 | 444608 | 92.63 | 96.58 | ns3 | 20.0 | 400 |
| PCS NS3 estimator | 20 | 10 | 727360 | 99.06 | 99.06 | ns3 | 20.0 | 400 |
| EdgeCooper-HD NS3 estimator | 6 | 6 | 826832 | 17.23 | 25.43 | ns3 | 20.0 | 400 |

## NS3 replay

| Variant | Params | Fatal | Manual adds | CAM received | Consume events | Alloc mean B | Delay mean ms | Delay P95 ms | Delay max ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NS3 default | default MCS20 symbols9 PSCCH10 RRI5 | 0 | 82 | 82 | 1991 | 398.86 | 59.57 | 110 | 123 |
| NS3 invalid PSCCH/RRI high-capacity probe | slPscchRbs=4 invalid | 1 | 0 | 0 | 0 |  |  |  |  |
| NS3 invalid RRI high-capacity probe | RRI1 violates resource selection window | 1 | 82 | 0 | 0 |  |  |  |  |
| NS3 high MCS/symbols | MCS28 symbols12 PSCCH10 RRI5 | 0 | 82 | 82 | 881 | 898.91 | 27.18 | 54 | 55 |

## Interpretation

- The shared `ChannelModel` metadata is present in SGCP, PCS, and EdgeCooper/Selective traces.
- `logical` and `ns3` estimator modes materially change SGCP/PAPG grid budget and estimated frame time; this validates that the scheduler path is using the unified estimator, not only writing metadata.
- Default NS3 still serves about 400 B per grant and gives SGCP chunked replay P95/max delay of 110/123 ms.
- Raising MCS and PSSCH symbols while keeping legal PSCCH/RRI settings raises grant size to about 899 B and reduces P95/max delay to 54/55 ms for the same 82 chunks.
- `slPscchRbs=4` is invalid in the current pool factory, and `slRriMs=1` violates the resource-selection-window constraint. These are not safe capacity knobs without deeper pool-window changes.