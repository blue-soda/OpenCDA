# EdgeCooper Singleton Budget Probe - 2026-07-22

Purpose: rerun the EdgeCooper V2V budget/range probe after identifying that the
previous `edgecooper_original_budget_20260722` probe accidentally used
`clustering=coalition_game`. This artifact is the corrected protocol-native
version:

- `clustering=singleton`
- `receiver_policy=all-cavs`
- original greedy endpoint-disjoint EdgeCooper link selection
- attentive detector
- no SGCP late fusion
- `40 MHz / 10` target subchannels
- OpenCDA NS3-calibrated estimator: `tb_size=899B`, `slot=0.5ms`

## Results

| Variant | Frames | Range | Admission budget | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Frame 000060 NS3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Reference greedy constrained | 41 | 35 m | 60 ms | 0.32 | 0.26 | 0.10 | 50.91 | 68/68, max 54 ms |
| Corrected greedy d100 r35 m3 g240 | 11 | 35 m | 100 ms | 0.32 | 0.26 | 0.09 | 49.62 | not replayed |
| Corrected greedy d100 r60 m3 g240 | 11 | 60 m | 100 ms | 0.28 | 0.23 | 0.07 | 53.41 | not replayed |
| Corrected greedy d100 r100 m3 g240 | 11 | 100 m | 100 ms | 0.25 | 0.20 | 0.07 | 56.36 | not replayed |
| Corrected greedy d100 r60 m3 g240 | 41 | 60 m | 100 ms | 0.27 | 0.21 | 0.08 | 54.42 | not replayed |
| Corrected greedy d100 r100 m3 g240 | 41 | 100 m | 100 ms | 0.25 | 0.19 | 0.07 | 55.67 | 73/73, max 55 ms |

## Interpretation

With the corrected singleton protocol, the apparent low-payload anomaly
disappears: increasing the range from `35 m` to `100 m` raises payload from
about `49.62 Mbps` to `56.36 Mbps` on the 11-frame probe. On the full 41-frame
run, `r100/d100` reaches `55.67 Mbps` and remains NS3-deliverable for frame
`000060` with `73/73` callbacks and max delay `55 ms`.

However, the higher communication points reduce AP compared with the 60 ms
reference. Therefore they are useful as diagnostics showing that simply giving
EdgeCooper more candidate range/budget does not improve aggregate AP in this
scene. The paper-facing constrained EdgeCooper row can remain the original
greedy `0.32/0.26/0.10, 50.91 Mbps` row unless the paper needs a stronger
traffic-only diagnostic; in that case use `r100/d100` as `0.25/0.19/0.07,
55.67 Mbps`, not as a better perception baseline.
