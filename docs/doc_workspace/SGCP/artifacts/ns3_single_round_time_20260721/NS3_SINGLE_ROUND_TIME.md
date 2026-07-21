# NS3 single-round communication time - 2026-07-21

Purpose: measure one-frame transfer latency in real NS3 instead of using the offline payload/rate estimate.

Common setup:

- Dataset frame: `2026_07_15_01_26_56/000060`
- NS3 command: `./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10'`
- NS3 reported radio setup: `targetSubchannels=10`, `totalSubChannel=11`, `slBandwidthIn100kHz=396`.
- Replay command: `opencda.tools.offline_ns3_replay --lgcp-upload-plan <plan.csv> --max-frames 1 --drain-seconds 5.0`
- Payload granularity: OpenCDA-online-compatible chunking, `max_packet_size=10000 bytes` per CAM packet.
- Time metric: application callback delay from NS3 `cam_received`, i.e. `receive_timestamp - send_timestamp`.

## Main NS3 Results

| Method / plan | Planned CAM packets | Planned bytes | Delivered CAM packets | Delivery ratio | Avg delay (ms) | P95 delay (ms) | Max delay (ms) | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PCS div4/radius4/min128, original channel allocation | 19 | 161,360 | 16 | 84.21% | 29.38 | 51.00 | 67.00 | One PCS link collided because two original links used subchannel 2. |
| PCS div4/radius4/min128, unique-subchannel diagnostic | 19 | 161,360 | 19 | 100.00% | 28.74 | 51.00 | 67.00 | With the subchannel collision removed, the whole PCS single round completes within 67 ms. |
| SGCP-PAPG | 82 | 783,392 | 82 | 100.00% | 59.57 | 110.00 | 123.00 | Full single-frame SGCP raw-LiDAR requests complete, but exceed a 100 ms cycle. |
| EdgeCooper-HD scaffold | 68 | 639,408 | 68 | 100.00% | 53.74 | 107.00 | 108.00 | Full single-frame EdgeCooper-HD raw-LiDAR requests complete, also slightly exceed 100 ms. |
| EdgeCooper V2V protocol-first10 diagnostic | 73 | 696,480 | 32 | 43.84% | 81.81 | 177.00 | 190.00 | First-10 protocol adaptation remains overloaded/conflicted in one round. |

## Important Diagnostics

Direct exact-payload replay without 10 KB chunking is not valid for large SGCP/EdgeCooper requests. 70-80 KB point-cloud transfers exceed practical UDP/CAM datagram size; the request can be logged as a manual command but no LC buffer is formed, so the MAC scheduler does not consume it. The chunked replay above matches the online OpenCDA behavior in `NetworkManager.communicate_through_ns3()`.

PCS original channel allocation for this frame assigns two links to the same subchannel:

- `12 -> 11` on subchannel 2
- `18 -> 14` on subchannel 2

The failed payload is exactly one original PCS link worth of chunks. Reassigning the seven PCS links to unique channels delivers all 19 chunks within the same 67 ms maximum callback delay.

## Paper-facing Conclusion

These NS3 measurements support the following communication-timing statements:

- PCS `div4/radius4/min128` is light enough to finish within 100 ms if its subchannel allocation is collision-free; the current original PCS replay has a real channel conflict on frame `000060`.
- SGCP-PAPG and EdgeCooper-HD raw-LiDAR plans are deliverable under NS3 with 10 KB chunking, but their single-frame worst callback delays are about `123 ms` and `108 ms`, respectively, under the current auto-derived `39.6 MHz / 10 target subchannels` NS3 setting.
- EdgeCooper V2V protocol-first10 remains too aggressive for one round; it should not be described as completing a full 20-CAV receiver universe in 100 ms.

Raw artifacts are stored in this directory. Each method subdirectory contains `upload_plan.csv`, `ns3_stdout.log`, `replay_stdout.log`, and `eval/delivery_summary.csv`.
