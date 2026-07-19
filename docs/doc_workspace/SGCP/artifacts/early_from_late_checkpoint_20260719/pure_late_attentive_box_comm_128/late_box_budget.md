# Late-Fusion Prediction-Box Communication Budget

This artifact estimates whether all-CAV pure late fusion can be naturally limited by a 100 ms V2V communication deadline.

| Scenario | Mean Mbps | Max Mbps | Mean scheduled ms | Deadline OK mean | Random-access full success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Broadcast one message per sender | 2.133417 | 2.350080 | 2.642081 | 1.000000 | 1.000000 |
| All-to-all unicast | 40.534915 | 44.651520 | 47.597724 | 1.000000 | 0.000000 |

Interpretation: scheduled deadline results model an ideal channel assignment. Random-access results are a coarse collision proxy where each outstanding message chooses a random subchannel each contention round.
