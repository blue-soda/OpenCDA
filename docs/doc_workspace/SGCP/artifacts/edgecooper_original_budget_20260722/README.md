# EdgeCooper Original-Greedy Budget Probe - 2026-07-22

Purpose: test whether EdgeCooper V2V can be given a larger admission budget
without changing its link-selection logic. The exact-matching probe from the
previous turn was removed because it changes the baseline algorithm.

## Fixed SGCP Main Result

The SGCP-PAPG paper-facing result remains the no-cap attentive configuration:

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Total Mbps |
| --- | ---: | ---: | ---: | ---: |
| SGCP-PAPG main | 0.87 | 0.81 | 0.36 | 63.28 |

The low-budget `--max-upload-points-per-source 4000` run is an auxiliary
operating point only; it does not change PAPG defaults.

## Original Greedy EdgeCooper Scan

All rows keep the original greedy endpoint-disjoint selection used by the
deadline-constrained EdgeCooper adaptation. Only the admission deadline,
candidate range, and per-receiver grid budget are varied.

| Variant | Frames | Range | Admission budget | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Frame 000060 bytes | NS3 result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Greedy constrained reference | 41 | 35 m | 60 ms | 0.32 | 0.26 | 0.10 | 50.91 | 640 KB-level | 68/68, max 54 ms |
| Greedy d100 r35 m3 g200 | 11 | 35 m | 100 ms | 0.29 | 0.24 | 0.08 | 32.96 | 456,464 | not needed |
| Greedy d100 r60 m3 g240 | 11 | 60 m | 100 ms | 0.27 | 0.21 | 0.07 | 33.90 | 461,072 | 49/49, max 55 ms |
| Greedy d100 r100 m3 g240 | 11 | 100 m | 100 ms | 0.25 | 0.19 | 0.06 | 34.29 | 384,624 | not needed |
| Old unconstrained demand | 41 | protocol default | none | 0.54 | 0.48 | 0.25 | 275.94 | high | 15/348, max 215 ms |

## Conclusion

Increasing the admission budget to `100 ms` does not raise the original greedy
EdgeCooper row toward `60+ Mbps`. With the same greedy selection, the tested
100 ms configurations only reach about `33-34 Mbps`, although the r60 probe is
NS3-deliverable within 60 ms. The old high-payload EdgeCooper row remains
deadline-infeasible, so the paper-facing deadline-feasible EdgeCooper baseline
should stay at the greedy constrained `0.32/0.26/0.10, 50.91 Mbps` row unless a
new paper-faithful EdgeCooper parameterization is identified.
