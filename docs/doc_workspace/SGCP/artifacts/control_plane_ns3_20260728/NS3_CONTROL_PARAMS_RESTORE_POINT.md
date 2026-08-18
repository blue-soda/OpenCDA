# NS3 control-plane restore point

Date: 2026-07-29

This note records the NS3-side control-plane parameters used for the SGCP realtime-feasibility probes before any further pacing or timer changes. It is intended as a restoration checklist so the previous experiments remain reproducible.

## Repository state

- Repository: `C:\Workspace\carla-ns3-co-simulation`
- Commit: `03f7326`
- Working tree: clean when this restore point was created.

Key file hashes:

| File | SHA256 |
|---|---|
| `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\main.cc` | `8D39FAD812783D963453BE931D4EE916A0F8C0D876C0819FF0AC68C83C187528` |
| `C:\Workspace\carla-ns3-co-simulation\ns3\vanet\cam-application.cc` | `A63C381DDB68A968720C340F2B9652C729E377EA3AC3A678445608820B9E3920` |
| `C:\Workspace\carla-ns3-co-simulation\ns3\src\lte-model\lte-rlc-um.cc` | `23A919D13E3175BE35A5009190E639B4640F3B24D8A9E7B9E76C83DCBAB36E0E` |

## Baseline command

```powershell
python docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\control_plane_probe.py `
  --ns3-command "scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5"
```

## NS3 parameters that affect batch pacing

The probe-side `batch_step_ms` is not itself a 5G/NS3 PHY parameter. It is the interval used by the CARLA/OpenCDA control-plane probe when it injects endpoint-disjoint request batches into NS3. Its reliable lower bound is affected by the following NS3 and application settings.

| Parameter / code path | Current value | Location | Effect |
|---|---:|---|---|
| `targetSubchannels` | `10` | `main.cc` CLI | Exposes ten requested subchannels to OpenCDA/NS3 manual scheduling. |
| `slMcs` | `28` | `main.cc` CLI | Fixed NR sidelink MCS used by the manual scheduler in these probes. |
| `slSymbolsPerSlot` | `12` | `main.cc` CLI | PSSCH symbols per slot. |
| `slPscchRbs` | `10` | `main.cc` CLI | PSCCH RB overhead. |
| `slRriMs` | `5` | `main.cc` CLI | Sidelink resource reservation interval passed to `slInfo.m_rri`. |
| `configuredSlSubchannelSize` | `10` | `main.cc` default | RBs per subchannel unless overridden. |
| `configuredSlMaxNumPerReserve` | `1` | `main.cc` default | Number of resources per reservation. |
| `configuredSlMaxTxTransNumPssch` | `1` | `main.cc` default | PSSCH transmission count; HARQ/repetition is disabled in the baseline. |
| `configuredSlProbResourceKeep` | `0.8` | `main.cc` default | Mode-2 resource keep probability. |
| `EnableSensing` | `false` | `main.cc` | Manual scheduling path, not sensing-based autonomous reselection. |
| `T1`, `T2` | `1`, `1` | `main.cc` | Resource selection window bounds used with the manual scheduler. |
| `SetSlSelectionWindow` | `5` | `main.cc` | Selection-window configuration. |
| first zero-time send delay | `20 ms` | `cam-application.cc` | `CamSenderNR::ScheduleCam` delays sends scheduled at simulator time zero. This explains the first-batch send timestamp around 20 ms. |
| NR SL RLC buffer-status timer | `10 ms` | `lte-rlc-um.cc` | `ExpireNrSlRbsTimer` reschedules buffer-status reporting every 10 ms. This is the most likely reason that `batch_step_ms=10` is borderline while `11` ms is reliable. |

## Baseline measurements

| Probe | Batch step | Delivery | Max NS3 receive time |
|---|---:|---:|---:|
| One-round compact per-CAV summary, 20 packets | `10 ms` | `10/20` | diagnostic failure point |
| One-round compact per-CAV summary, 20 packets | `11 ms` | `20/20` | `21 ms` |
| Three-round aggregated cold-start summaries, 60 packets | `10 ms` | `50/60` | diagnostic failure point |
| Three-round aggregated cold-start summaries, 60 packets | `11 ms` | `60/60` | `56 ms` |
| Unaggregated cold-start control exchange, 270 packets | `10 ms` | `260/270` | diagnostic failure point |
| Unaggregated cold-start control exchange, 270 packets | `11 ms` | `270/270` | `287 ms` |

## Restoration rule

Before changing any NS3 timer, scheduler, or application delay, either:

1. keep the source unchanged and expose the change through a new command-line argument, or
2. create a commit/branch first and record the changed file hashes next to this document.

To reproduce the existing SGCP realtime-feasibility numbers, use the baseline command and parameter values above.
