# NR sidelink RBS timer probe

Date: 2026-07-29

## Purpose

This probe checks whether the SGCP control-plane `batch_step_ms` lower bound is controlled by the NR sidelink RLC buffer-status-report timer.

The NS3 code was changed in a backward-compatible way:

- default behavior remains `NrSlRbsTimer = 10 ms`;
- `scratch/vanet/main.cc` now accepts `--nrSlRbsTimerMs=<ms>`;
- the timer is exposed as `ns3::LteRlcUm::NrSlRbsTimer`.

The pre-change restoration point is recorded in `NS3_CONTROL_PARAMS_RESTORE_POINT.md`.

## Build

```powershell
wsl -d Ubuntu-22.04 -u sakakibara -- bash -lc "cd ~/workspace/carla-ns3-co-simulation/ns-3-dev && ./ns3 build"
```

Result: build passed.

## Protocol

- Profile: aggregated three-round cold-start control exchange.
- Requests: 60 packets.
- Packet size: 400 B.
- Vehicles: 20 CAVs.
- Batching: endpoint-disjoint batches of 10 packets.
- NS3 command base:

```bash
./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5 --nrSlRbsTimerMs=<value>'
```

Raw logs: `rbs_timer_probe_20260729_v2/`.

## Results

| `nrSlRbsTimerMs` | `batch_step_ms` | Callback delivery | RLC RX | PHY decode OK | Max send time (ms) | Max receive time (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 50/60 | 50/60 | 50/60 | 50 | 51 |
| 10 | 11 | 60/60 | 60/60 | 60/60 | 55 | 56 |
| 5 | 5 | 40/60 | 40/60 | 40/60 | 25 | 26 |
| 5 | 6 | 50/60 | 50/60 | 50/60 | 30 | 31 |
| 5 | 8 | 50/60 | 50/60 | 50/60 | 40 | 41 |
| 5 | 10 | 50/60 | 50/60 | 50/60 | 50 | 51 |
| 5 | 11 | 60/60 | 60/60 | 60/60 | 55 | 56 |
| 5 | 12 | 60/60 | 60/60 | 60/60 | 60 | 61 |
| 2 | 3 | 30/60 | 30/60 | 30/60 | 20 | 21 |
| 2 | 5 | 40/60 | 40/60 | 40/60 | 25 | 26 |
| 2 | 10 | 50/60 | 50/60 | 50/60 | 50 | 51 |
| 2 | 11 | 60/60 | 60/60 | 60/60 | 55 | 56 |
| 2 | 12 | 60/60 | 60/60 | 60/60 | 60 | 61 |
| 1 | 2 | 10/60 | 10/60 | 10/60 | 20 | 21 |
| 1 | 3 | 30/60 | 30/60 | 30/60 | 20 | 21 |
| 1 | 10 | 50/60 | 50/60 | 50/60 | 50 | 51 |
| 1 | 11 | 60/60 | 60/60 | 60/60 | 55 | 56 |

## Interpretation

The RLC BSR timer is adjustable, but reducing it does not reduce the reliable `batch_step_ms` threshold in this control-plane setup.

Across `nrSlRbsTimerMs = 10, 5, 2, 1`, the reliable threshold remains `batch_step_ms = 11 ms` for the 60-packet aggregated three-round profile. `batch_step_ms = 10 ms` remains at `50/60` delivery, and smaller steps drop more packets.

Therefore, the observed `11 ms` batch step is not determined by the RLC BSR timer. A follow-up activation-gap probe shows that `batch_step_ms=10` loses the second batch because it is injected at 10 ms, before the NR sidelink bearer activation boundary at `1 ms + 10 ms = 11 ms`. Adding a probe-side `first_gap_ms=11` restores `60/60` delivery even when the subsequent batch step is below 10 ms.

The remaining timing behavior is shaped by the current manual NR sidelink scheduling path, including:

- 10 endpoint-disjoint packets per batch on 10 target subchannels;
- `slRriMs=5`;
- `T1=T2=1` and `SetSlSelectionWindow(5)`;
- first zero-time application send delay of 20 ms;
- manual scheduler and one resource per reservation.

For paper-facing realtime analysis, keep the conservative and validated `batch_step_ms=11` control-plane pacing or explicitly state that the control exchange starts after bearer activation. The new `--nrSlRbsTimerMs` parameter is useful for diagnostics, but should not be used to claim a shorter control exchange deadline.
