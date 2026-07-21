# EdgeCooper High-Communication Probes - 2026-07-22

Purpose: respond to the request to raise the deadline-constrained EdgeCooper V2V payload while keeping the main SGCP-PAPG result at `0.87/0.81/0.36`.

## PAPG Main Result

No PAPG code or default parameter was changed by the SGCP low-budget experiment. The low-budget point used `--max-upload-points-per-source 4000` only in its command line. The paper-facing main SGCP-PAPG row remains the no-cap attentive result:

| Method | AP@0.3 | AP@0.5 | AP@0.7 | Total Mbps |
| --- | ---: | ---: | ---: | ---: |
| SGCP-PAPG main | 0.87 | 0.81 | 0.36 | 63.28 |

## EdgeCooper Exact Matching

The previous deadline-constrained EdgeCooper V2V row used a greedy endpoint-disjoint matching. This artifact adds an exact one-round matching for `edgecooper_global`: maximize the number of half-duplex endpoint-disjoint links, then maximize candidate payload. The default candidate range remains `35 m`, so existing behavior is preserved unless a different range is passed.

| Variant | Frames | Range | Deadline | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Old constrained greedy matching | 41 | 35 m | 60 ms | 0.32 | 0.26 | 0.10 | 50.91 | Earlier paper-facing constrained EdgeCooper row. |
| Exact matching, default range | 41 | 35 m | 60 ms | 0.32 | 0.26 | 0.11 | 57.09 | Stronger constrained EdgeCooper row; same deadline and range. |
| Exact matching, expanded range | 11 | 100 m | 75 ms | 0.23 | 0.18 | 0.06 | 62.93 | Higher payload, but AP degrades; not recommended for paper table. |

## NS3 Validation

Exact matching, default range, frame `000060`:

| Chunks | Bytes | Callbacks | Avg delay | P95 delay | Max delay | PHY failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 72 | 698,224 | 72/72 | 26.556 ms | 53.000 ms | 55.000 ms | 0 |

This satisfies the 60 ms communication window under the `40 MHz / 10 target-subchannel / MCS28 / 12-symbol` NS3 setting.

## Recommendation

Use the exact-matching `57.09 Mbps` EdgeCooper row if the paper needs a slightly higher communication baseline under the same 60 ms deadline. Do not force EdgeCooper to 65-70 Mbps by expanding its range: in this dump it lowers AP and changes the protocol assumption.
