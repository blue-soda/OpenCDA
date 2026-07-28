# NR sidelink broadcast control-plane probe

Date: 2026-07-29

## Purpose

Verify whether the current CARLA-NS3 bridge can deliver SGCP compact control summaries as true NR sidelink broadcast/groupcast traffic, rather than as one unicast request per receiver.

This matters for SGCP coalition formation because cluster heads are not known before the coalition-maintenance exchange. A compact per-CAV summary is therefore naturally disseminated as broadcast/groupcast control traffic, while raw-LiDAR grid payloads remain scheduled unicast data-plane traffic.

## Code Changes

The default behavior remains unicast. Broadcast is enabled only with an explicit NS3 command-line flag and explicit request metadata.

Changed files:

```text
C:\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc
C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\control_plane_probe.py
```

New NS3 options:

```text
--enableControlBroadcast=true
--controlBroadcastAddress=225.0.0.0
--controlBroadcastL2Id=255
```

New transfer-request field:

```json
{"cast_type": "broadcast"}
```

When this field is set, `ProcessData_TransferRequests()` sends the packet to the multicast address and uses destination L2 ID `255`. The manual scheduler still applies the requested subchannel start/width, so broadcast control packets are subject to the same orthogonal-subchannel accounting as unicast packets.

## NS3 Command

```bash
./ns3 run 'scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5 --slBearerActivationGuardMs=1 --nrSlZeroTimeSendDelayMs=0 --enableControlBroadcast=true'
```

This uses the same calibrated high-capacity sidelink setting as the SGCP data-plane replay, with an optimized startup configuration:

- `40 MHz / 10 target subchannels`;
- `MCS=28`;
- `PSSCH symbols=12`;
- `PSCCH RBs=10`;
- `RRI=5 ms`;
- bearer activation guard `1 ms`;
- zero-time application send delay disabled after a pre-send sync to the activation boundary.

The old defaults remain reproducible because `enableControlBroadcast=false`, `slBearerActivationGuardMs=10`, and `nrSlZeroTimeSendDelayMs=20` unless explicitly overridden.

## Probe Commands

One-round broadcast smoke test:

```powershell
conda run --no-capture-output -n opencda python docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\control_plane_probe.py `
  --vehicles 20 `
  --packet-size 400 `
  --subchannels 10 `
  --profile aggregated `
  --limit-requests 20 `
  --cast-type broadcast `
  --batch-size 10 `
  --batch-step-ms 1 `
  --pre-send-sync-ms 2 `
  --drain-seconds 1.0 `
  --upload-plan-output docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\broadcast_probe_20260729\broadcast_1round_upload_plan.csv
```

Complete aggregated control profile:

```powershell
conda run --no-capture-output -n opencda python docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\control_plane_probe.py `
  --vehicles 20 `
  --packet-size 400 `
  --subchannels 10 `
  --profile aggregated `
  --cast-type broadcast `
  --batch-size 10 `
  --batch-step-ms 2 `
  --pre-send-sync-ms 2 `
  --drain-seconds 1.0 `
  --upload-plan-output docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\broadcast_probe_20260729\broadcast_3round_step2_upload_plan.csv
```

The `aggregated` profile contains:

```text
60 coalition-round summaries
6 scheduler summaries
4 scheduler grants
70 total broadcast control packets
```

## Results

| Probe | Requests | Batch size | Batch step | Expected half-duplex callbacks | Observed callbacks | RLC RX | Manual resources applied | Max send timestamp | Max receive timestamp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One-round broadcast smoke | 20 | 10 | 1 ms | about 200 | 200 | 200 | 20 | 3 ms | 4 ms |
| Aggregated control broadcast | 70 | 10 | 2 ms | about 700 | 699 | 699 | 70 | 14 ms | 15 ms |

The `20`-request smoke test receives `200` callbacks rather than `20`, confirming true one-to-many delivery. It does not receive `20 x 19 = 380` callbacks because a vehicle transmitting in a 10-packet batch is half-duplex and does not receive the other simultaneous broadcasts in that batch. With 10 transmitters and 10 non-transmitting receivers per batch, the half-duplex upper bound is about `20 x 10 = 200`.

For the complete 70-packet profile, 69 requests were received by all 10 non-transmitting receivers. One deterministic receiver-side callback was missing:

```text
request_id=69, sender=9, missing receiver=20
```

Thus every broadcast packet reached at least 9 non-transmitting receivers, and the total application/RLC delivery was `699/700 = 99.86%` relative to the half-duplex fanout bound.

## Interpretation

The current NS3 stack can now run true broadcast/groupcast control traffic for SGCP summaries:

- the bridge accepts `cast_type=broadcast`;
- the NR sidelink bearer is installed with `SidelinkInfo::CastType::Broadcast`;
- destination L2 ID `255` is passed to the manual scheduler;
- requested subchannels are still honored;
- application callbacks show one transmitted control packet being delivered to multiple receiver vehicles.

This supports the paper-facing design:

```text
Control plane: compact per-CAV sensing summaries via NR sidelink broadcast/groupcast.
Data plane: selected raw-LiDAR grid payloads via scheduled unicast links.
```

Broadcast control is better than unicast fanout for coalition formation because the receiver set is not fixed before clusters and heads are finalized. It also avoids multiplying one summary by the number of candidate heads or vehicles.

The broadcast probe should not be interpreted as guaranteed all-receiver delivery in a saturated simultaneous-broadcast batch. Broadcast has no unicast-style per-receiver ACK/HARQ in this configuration, and half-duplex receiver constraints matter. For the SGCP realtime argument, the important result is that compact summaries can be disseminated within `15 ms` at the tested 20-CAV scale, well inside the 40 ms algorithm/control budget.

Raw logs:

```text
C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\broadcast_probe_20260729\
```
