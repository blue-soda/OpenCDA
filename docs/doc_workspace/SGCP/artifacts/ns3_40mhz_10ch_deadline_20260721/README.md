# NS3 40 MHz / 10 Target-Subchannel Deadline Replay

Date: 2026-07-21

Purpose: validate the paper-facing SGCP communication setting with real NS3
chunked replay.

## Setting

- Perception cycle: `100 ms`
- Communication deadline target: `60 ms`
- NS3 bandwidth argument: `--slBandwidthIn100kHz=400`
- OpenCDA-visible target subchannels: `--targetSubchannels=10`
- NS3 subchannel size: `--slSubchannelSize=10`
- NS3 MCS/symbol setting: `--slMcs=28 --slSymbolsPerSlot=12`

`--slSubchannelSize=11` was tested and is invalid in the NR sidelink resource
pool. NS3 accepts `10`, `15`, `20`, `25`, `50`, `75`, and `100` PRBs. The
valid paper-facing configuration therefore reports `totalSubChannel=11`, while
OpenCDA schedules only target subchannels `0..9`.

## Result

Replay plan: SGCP-PAPG frame `000060`, 10KB CAM chunking.

| Planned requests | Bytes | Application callbacks | Mean delay | P95 delay | Max delay | PHY failures | Mean grant |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 82 | 783,392 | 82/82 | 27.18 ms | 54.00 ms | 55.00 ms | 0 | 898.91 B |

This satisfies the intended 60 ms communication window inside a 100 ms
perception cycle.

## Files

- `upload_plan.csv`: replayed SGCP chunk plan.
- `ns3_stdout.log`: full NS3 stdout captured by Python subprocess.
- `replay_stdout.log`: OpenCDA offline replay stdout.
- `eval/delivery_summary.csv`: application callback summary.
- `eval/rlc_summary.csv`: RLC request-level diagnostic.
- `eval/phy_decode_summary.csv`: PHY decode diagnostic.
