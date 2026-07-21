# EdgeCooper Singleton d200 r35 NS3 Replay

Purpose: validate the corrected protocol-native EdgeCooper `35m / 200ms`
probe on frame `000060` under the paper-facing NS3 setting.

## Protocol

- Baseline: original greedy endpoint-disjoint EdgeCooper V2V adaptation.
- Clustering: `singleton`.
- Receiver policy: `all-cavs`.
- Detector: attentive checkpoint.
- Late fusion: none.
- Range: `35 m`.
- Admission budget used by the offline scheduler: `200 ms`.
- NS3 setting: `40 MHz / 10` target subchannels with
  `slMcs=28`, `slSymbolsPerSlot=12`, `slSubchannelSize=10`.

## Upload Plan

- Frame: `000060`.
- Links: `9`.
- Chunks: `68`.
- Payload: `654,256 bytes`.
- Chunk size: up to `10,000 bytes`.

## NS3 Result

- Application callbacks: `68/68`.
- Request-level RLC RX: `68/68`.
- PHY decode failures: `0`.
- Delay mean/P95/max: `25.926 / 53.000 / 54.000 ms`.

Conclusion: this configuration is NS3-deliverable within the 60 ms
communication window, but it does not materially increase traffic or AP over
the 35m/60ms paper-facing EdgeCooper reference. The limiting factor is the
candidate link set plus original greedy endpoint-disjoint matching, not the
admission deadline.
