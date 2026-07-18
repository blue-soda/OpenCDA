# Late-Fusion Prediction-Box Communication Budget

This artifact estimates whether all-CAV pure late fusion can be naturally limited by a 100 ms V2V communication deadline.

| Scenario | Mean Mbps | Max Mbps | Mean scheduled ms | Deadline OK mean | Random-access full success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Broadcast one message per sender | 1.068332 | 1.148160 | 1.501528 | 1.000000 | 1.000000 |
| All-to-all unicast | 20.298302 | 21.815040 | 25.071740 | 1.000000 | 0.000000 |

Interpretation: scheduled deadline results model an ideal channel assignment. Random-access results are a coarse collision proxy where each outstanding message chooses a random subchannel each contention round.
