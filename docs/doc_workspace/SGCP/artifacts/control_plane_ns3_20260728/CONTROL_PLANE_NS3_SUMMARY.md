# SGCP Control-Plane NS3 Probe 2026-07-28

## Protocol

- NS3 command: `scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5`
- Vehicle count: 20
- Control packet size: 400 bytes
- Target subchannels: 10
- Successful paper-facing probe: compact per-CAV control summaries, 20 packets, endpoint-disjoint batches of 10 packets, 11 ms batch step.

## Result

| Probe | Planned packets | Received callbacks | RLC complete requests | Delivery ratio | Max send timestamp | Max receive timestamp | Mean delay | Max delay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Compact per-CAV summaries | 20 | 20 | 20 | 1.00 | 20 ms | 21 ms | 1 ms | 1 ms |

Output files:

- `control_upload_plan_summary20_matching_step11ms_r1.csv`
- `ns3_stdout_summary20_matching_step11ms_r1.log`
- `eval_summary20_matching_step11ms_r1/delivery_summary.csv`
- `eval_summary20_matching_step11ms_r1/rlc_summary.csv`

## Diagnostic Probes

The following probes are intentionally not paper-facing, but they explain why the control protocol should aggregate metadata into compact summaries instead of sending every logical admission check as a separate application packet.

| Probe | Planned packets | Received callbacks | Delivery ratio | Observation |
| --- | ---: | ---: | ---: | --- |
| Unaggregated logical messages, all at once | 314 | 0 | 0.00 | Severe overlap/collision; not a valid control protocol. |
| Aggregated 70 packets, 10 packets every 5 ms | 70 | 50 | 0.71 | Repeated-source CAM timing loses one or more batches. |
| Aggregated 70 packets, 10 packets every 6 ms | 70 | 60 | 0.86 | Improved but still incomplete. |

Conclusion: the NS3 CAM application is suitable for validating compact control summaries and scheduled payload chunks, but not for representing hundreds of fine-grained logical control events as separate application packets in one control window.

## Cold-Start Probe

The full cold-start control sequence was also tested separately. It should be treated as initialization/topology-change overhead, not as the per-frame steady-state control protocol.

| Probe | Packets | Batch step | Delivery | RLC complete | Max receive timestamp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-round aggregated coalition summaries | 60 | 10 ms | 50/60 | 50/60 | 51 ms |
| Three-round aggregated coalition summaries | 60 | 11 ms | 60/60 | 60/60 | 56 ms |
| Three-round coalition only | 270 | 10 ms | 260/270 | 260/270 | - |
| Three-round coalition only | 270 | 11 ms | 270/270 | 270/270 | 287 ms |
| Three-round coalition + scheduler control | 314 | 11 ms | 314/314 | 314/314 | 342 ms |

Detailed cold-start notes are in `COLD_START_CONTROL_NS3_SUMMARY.md`.
