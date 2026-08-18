# SGCP Hybrid Main-Table NS3 Replay

This artifact uses the SGCP hybrid main-table trace as the fixed schedule
source. The upload plan is generated from:

```text
C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f\trace.csv
```

Each scheduled sender-receiver link is replayed as one raw-LiDAR request with
the same payload bytes and requested subchannel as the main-table trace.

NS3 settings:

- `--slBandwidthIn100kHz=400` (`40 MHz`)
- `--targetSubchannels=10`
- `--slSubchannelSize=10`
- `--slMcs=28`
- `--slSymbolsPerSlot=12`
- `--slPscchRbs=10`
- `--slRriMs=5`
- `--slBearerActivationGuardMs=1`
- `--nrSlZeroTimeSendDelayMs=0`
- `--slErrorModelEnabled=false` for deterministic scheduled-capacity replay

The final communication timing is measured on the NS3 RLC TX/RX path with
request-id byte accumulation. Application callbacks are kept only as a
diagnostic because large raw-LiDAR requests can be split into multiple UDP
chunks and therefore may produce more than one callback per logical request.

## Fixed Upload Plan

| Frames | Requests | Total bytes | Average raw Mbps |
| ---: | ---: | ---: | ---: |
| 41 | 410 | 14,564,112 | 28.42 |

Plan file:

```text
C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f\sgcp_hybrid_ns3_upload_plan.csv
```

## Isolated Single-Frame NS3 Replay

The isolated runs send one frame's scheduled requests after bearer activation
and then advance NS3 long enough to observe application callbacks. These are
the cleanest NS3-output numbers for per-frame raw-LiDAR upload latency.

| Timestamp | Payload bytes | Logical load at 100 ms | App callbacks | Mean delay | P95 delay | Max delay | RLC TX/RX | PHY failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `000140` | 383,232 | 30.66 Mbps | 10/10 | 33.20 ms | 43.00 ms | 43.00 ms | 434/434 | 0 |
| `000070` | 340,944 | 27.28 Mbps | 9/10 | 28.44 ms | 35.00 ms | 35.00 ms | 315/315 | 0 |
| `000074` | 340,704 | 27.26 Mbps | 8/10 | 28.62 ms | 38.00 ms | 38.00 ms | 316/315 | 1 |

The highest-payload SGCP main-table frame (`000140`) is fully delivered by NS3
with a maximum application callback delay of `43.00 ms`, below the reserved
`60 ms` raw-LiDAR data-plane budget.

## Continuous 41-Frame NS3 Replay, Final

The final full replay sends all 41 frames at the dataset frame interval
(`100 ms`) and advances NS3 to the `60 ms` data-plane boundary after each
frame. The raw-LiDAR requests use application-level UDP chunking for payloads
larger than a single UDP datagram while preserving the same logical
`request_id`. The replay disables stochastic PHY decode errors so the result
tests whether the scheduled subchannel plan fits the intended NS3 capacity and
timing model.

| Planned requests | RLC complete | RLC delivery | RLC mean delay | RLC P95 delay | RLC max delay | RLC TX/RX | PHY failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 410 | 410 | 100.00% | 20.27 ms | 36.56 ms | 37.56 ms | 16620/16620 | 0 |

Continuous replay artifacts:

```text
C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f\ns3_replay_41f_udpchunk_noerr_rlctime
```

Interpretation: use the 41-frame RLC-side result as the paper-facing NS3
communication-time evidence for the SGCP main-table point. All logical
requests are completely received below the reserved `60 ms` raw-LiDAR
data-plane window. Application callback counts in the same run are intentionally
not used for delivery ratio because they operate above the UDP chunking layer.

## Bug Fixes Found During Replay

- Large raw-LiDAR UDP datagrams around `66 KB` failed before entering RLC with
  socket `ERROR_MSGSIZE`. `CamSenderNR` now chunks large application payloads
  while keeping the same logical `request_id`.
- The manual sidelink scheduler previously behaved like a FIFO queue per
  sender and could miss a pending command when the selected destination did not
  match the queue head. The manual scheduler now scans pending commands for the
  selected source and applies the requested `src,dst,request_id,subchannel`
  tuple.
- Stochastic PHY decode errors are now controlled by the explicit
  `--slErrorModelEnabled` switch. The default remains enabled for baseline NS3
  behavior, while deterministic SGCP schedule replay passes `false`.
