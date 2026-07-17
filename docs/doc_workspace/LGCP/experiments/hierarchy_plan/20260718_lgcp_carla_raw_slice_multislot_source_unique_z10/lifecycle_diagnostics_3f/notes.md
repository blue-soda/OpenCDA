# LGCP Lifecycle Diagnostics

This diagnostic joins request lifecycle records with the replayed upload plan to expose slot/stage behavior.

- unmatched lifecycle rows: `0`

## Stage Summary

| Stage | Planned | RLC RX ratio | PSSCH OK ratio | CAM ratio |
| --- | ---: | ---: | ---: | ---: |
| leader_to_rsu | 90 | 0.933333 | 0.933333 | 0.433333 |
| member_to_leader | 47 | 0.531915 | 0.531915 | 0.106383 |

## Slot Summary

| Stage | Slot | Planned | RLC RX ratio | CAM ratio |
| --- | ---: | ---: | ---: | ---: |
| leader_to_rsu | 2 | 20 | 0.700000 | 0.250000 |
| leader_to_rsu | 3 | 24 | 1.000000 | 0.583333 |
| leader_to_rsu | 4 | 21 | 1.000000 | 0.666667 |
| leader_to_rsu | 5 | 14 | 1.000000 | 0.285714 |
| leader_to_rsu | 6 | 8 | 1.000000 | 0.000000 |
| leader_to_rsu | 7 | 3 | 1.000000 | 0.666667 |
| member_to_leader | 0 | 28 | 0.500000 | 0.178571 |
| member_to_leader | 1 | 18 | 0.555556 | 0.000000 |
| member_to_leader | 2 | 1 | 1.000000 | 0.000000 |

## Size-Bin Summary

| Upload type | Byte bin | Planned | RLC RX ratio | CAM ratio |
| --- | ---: | ---: | ---: | ---: |
| leader_to_rsu | 2000-4000 | 90 | 0.933333 | 0.433333 |
| member_to_leader | 0-1000 | 7 | 0.571429 | 0.000000 |
| member_to_leader | 1000-2000 | 9 | 0.444444 | 0.111111 |
| member_to_leader | 2000-4000 | 21 | 0.380952 | 0.047619 |
| member_to_leader | 4000-8000 | 4 | 1.000000 | 0.250000 |
| member_to_leader | 8000-16000 | 6 | 0.833333 | 0.333333 |
