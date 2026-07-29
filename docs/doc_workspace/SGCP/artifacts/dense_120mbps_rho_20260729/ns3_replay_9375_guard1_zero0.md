# NS3 Replay for Dense 93.75 Mbps Diagnostic

Purpose: verify how long the relaxed dense SGCP `rho_th=5` diagnostic point
(`93.75 Mbps` average raw-LiDAR payload) actually takes in NS3 under the
paper-facing dense channel setting.

NS3 setting:

- Bandwidth: `40 MHz` (`--slBandwidthIn100kHz=400`)
- Target subchannels: `10`
- Subchannel size: `10 PRB`
- MCS / symbols: `28` / `12`
- PSCCH RBs: `10`
- RRI: `5 ms`
- Bearer activation guard: `1 ms`
- Zero-time application send delay: `0 ms`
- Transfer injection: after activation sync

Replay source:

- Dataset: `D:\Data\Carla\2026_07_29_02_32_08`
- Trace: `C:\Workspace\2026-7-papers\infocom\SGCP\experiment-0729-dense-ver\sgcp_40mhz_budget120_deadline200p0_nmax5_rho5_41f\trace.csv`
- Scheduler setting: `potential_verified_cov_coalition_game + cov_potential_game`,
  raw frame budget `120 Mbps`, relaxed `communication_deadline_ms=200`,
  `N_max=5`, `rho_th=5`

## Results

| Frame | Logical load | Requests | Links | Payload bytes | App callbacks | RLC TX/RX | PHY failures | Mean delay | P95 delay | Max delay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `000060` | `96.59 Mbps` | 124 | 7 | 1,207,344 | 123/124 | 1346/1345 | 1 | 60.01 ms | 131 ms | 137 ms |
| `000076` | `92.34 Mbps` | 118 | 6 | 1,154,256 | 118/118 | 1286/1286 | 0 | 60.98 ms | 121 ms | 136 ms |

The 41-frame average for this relaxed diagnostic row is `93.75 Mbps`; frame
`000076` is the closest replayed frame to that average.

## Interpretation

- The 93.75 Mbps relaxed diagnostic point is mostly deliverable in NS3 under
  `40 MHz / 10ch / guard=1 ms / zero-delay=0 ms`.
- It does not satisfy the intended `60 ms` data-plane window. The representative
  frame needs about `136 ms` max application delay; the slightly higher-load
  first frame needs about `137 ms`.
- Therefore this point should remain a diagnostic showing AP headroom under
  relaxed transmission time. It should not replace the deadline-feasible dense
  headline row (`~60.37 Mbps`, max link time `60 ms`) in the paper-facing main
  table.

Artifacts:

- `ns3_frame000060_rho5_deadline200/guard1_zero0_40mhz10ch/summary.json`
- `ns3_frame000060_rho5_deadline200/guard1_zero0_40mhz10ch/eval/`
- `ns3_frame000076_rho5_deadline200/guard1_zero0_40mhz10ch/summary.json`
- `ns3_frame000076_rho5_deadline200/guard1_zero0_40mhz10ch/eval/`
