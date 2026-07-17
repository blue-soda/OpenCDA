# LGCP Lifecycle Diagnostics

This diagnostic joins request lifecycle records with the replayed upload plan to expose slot/stage behavior.

- unmatched lifecycle rows: `0`

## Stage Summary

| Stage | Planned | RLC RX ratio | PSSCH OK ratio | CAM ratio |
| --- | ---: | ---: | ---: | ---: |
| leader_to_rsu | 90 | 0.911111 | 0.911111 | 0.577778 |
| member_to_leader | 47 | 0.595745 | 0.595745 | 0.042553 |

## Slot Summary

| Stage | Slot | Planned | RLC RX ratio | CAM ratio |
| --- | ---: | ---: | ---: | ---: |
| leader_to_rsu | 2 | 30 | 0.800000 | 0.433333 |
| leader_to_rsu | 3 | 30 | 0.966667 | 0.533333 |
| leader_to_rsu | 4 | 30 | 0.966667 | 0.766667 |
| member_to_leader | 0 | 30 | 0.700000 | 0.066667 |
| member_to_leader | 1 | 17 | 0.411765 | 0.000000 |
