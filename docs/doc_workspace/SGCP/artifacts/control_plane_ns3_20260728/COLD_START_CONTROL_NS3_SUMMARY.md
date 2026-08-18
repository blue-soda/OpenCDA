# SGCP Cold-Start Control-Plane NS3 Probe

## Protocol

- NS3 command: `scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5`
- Vehicles: `20`
- Control packet size: `400 bytes`
- Target subchannels: `10`
- Batch encoding: endpoint-disjoint batches of 10 packets.
- Aggregated cold-start coalition-control encoding:
  - one compact per-CAV summary packet per round
  - 20 CAVs x 3 rounds
  - total: `60` packets
- Full cold-start coalition-control encoding:
  - `60` coalition proposal packets
  - `60` coalition reply packets
  - `120` potential-verified source/target check packets
  - `30` membership update packets
  - total: `270` packets
- Extended cold-start control encoding additionally includes scheduler control:
  - `14` scheduler summary packets
  - `10` scheduler grant packets
  - `20` scheduler ACK/reservation packets
  - total: `314` packets

## Result

| Probe | Packets | Batch step | Delivery | RLC complete | Max send timestamp | Max receive timestamp | Mean delay | Max delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregated cold-start coalition summaries | 60 | 8 ms | 50/60 | 50/60 | 40 ms | 41 ms | 1 ms | 1 ms |
| Aggregated cold-start coalition summaries | 60 | 10 ms | 50/60 | 50/60 | 50 ms | 51 ms | 1 ms | 1 ms |
| Aggregated cold-start coalition summaries | 60 | 11 ms | 60/60 | 60/60 | 55 ms | 56 ms | 1 ms | 1 ms |
| Cold-start coalition only | 270 | 10 ms | 260/270 | 260/270 | - | - | 1 ms | 1 ms |
| Cold-start coalition only | 270 | 11 ms | 270/270 | 270/270 | 286 ms | 287 ms | 1 ms | 1 ms |
| Cold-start coalition only | 270 | 12 ms | 270/270 | 270/270 | - | - | 1 ms | 1 ms |
| Cold-start coalition + scheduler control | 314 | 11 ms | 314/314 | 314/314 | 341 ms | 342 ms | 1 ms | 1 ms |
| Cold-start coalition + scheduler control | 314 | 12 ms | 314/314 | 314/314 | 372 ms | 373 ms | 1 ms | 1 ms |

## Interpretation

The full unaggregated three-round cold-start control sequence is reliable only when spread over hundreds of milliseconds under the current CAM-style control encoding. Therefore, it is not a valid per-frame control protocol for the reserved 40 ms algorithm/control budget. Aggregating one CAV's round-level metadata into a compact summary cuts the three-round coalition-control sequence to 60 packets and makes it reliable at 56 ms. This is much better than the unaggregated sequence, but it is still above the reserved 40 ms algorithm/control budget and should be treated as initialization/topology-change overhead rather than the normal per-frame path.

This does not invalidate SGCP realtime operation because the online protocol should not cold-start every 100 ms frame. The measured warm-start profile shows that after the first frame, the partition needs one confirmation round and zero accepted migrations over all 40 adjacent-frame transitions in the 41-frame trace. The paper-facing realtime protocol should therefore be described as:

1. cold-start or topology-change reconfiguration can take multiple rounds and is not paid every frame;
2. steady-state frames use warm-start coalition maintenance with compact per-CAV summaries;
3. raw-LiDAR scheduling and data transmission remain within the per-frame 100 ms budget.
