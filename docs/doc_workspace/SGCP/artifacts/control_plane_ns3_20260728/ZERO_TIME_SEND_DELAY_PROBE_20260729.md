# NR zero-time application send delay probe

Date: 2026-07-29

## Question

What is the `20 ms` application-layer send delay in `CamSenderNR::ScheduleCam()`, and can it be set to `0 ms`?

## Code

The original code delayed NR CAM sends that were scheduled at simulator time zero:

```cpp
const Time sendDelay = Simulator::Now().IsZero() ? MilliSeconds(20) : MilliSeconds(0);
```

This was parameterized as:

```bash
--nrSlZeroTimeSendDelayMs=<ms>
```

The default remains `20 ms`, so previous experiments remain reproducible. The value only affects sends scheduled when `Simulator::Now().IsZero()`. It does not affect steady-state sends after the simulator has advanced beyond time zero.

## Interpretation of the delay

The delay is an application-layer startup guard. It prevents the first application packets from being injected before sockets, routes, and sidelink bearers are ready. It is not a 5G sidelink PHY/MAC timing parameter.

It can be set to `0 ms`, but only if the control-plane protocol does not send the first packet before NR sidelink bearer activation. In this probe, that means syncing NS3 to the activation boundary before sending the first transfer batch.

## Probe change

The control-plane probe now supports:

```bash
--pre-send-sync-ms=<ms>
```

This advances NS3 to the requested time before the first transfer batch. The default is `0`, preserving the original probe behavior.

## Results

Profile: complete three-round aggregated-summary control exchange, 60 packets, 400 B/packet, endpoint-disjoint batches of 10.

| Guard | Zero-time send delay | Pre-send sync | Batch step | Callback | RLC RX | PHY decode OK | Max send | Max receive | Mean delay | Max delay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 ms | 20 ms | 0 ms | 2 ms | 60/60 | 60/60 | 60/60 | 20 ms | 21 ms | 1 ms | 1 ms |
| 1 ms | 0 ms | 0 ms | 2 ms | 50/60 | 50/60 | 50/60 | 10 ms | 11 ms | 1 ms | 1 ms |
| 1 ms | 0 ms | 2 ms | 2 ms | 60/60 | 60/60 | 60/60 | 12 ms | 13 ms | 1 ms | 1 ms |
| 1 ms | 0 ms | 2 ms | 1 ms | 60/60 | 60/60 | 60/60 | 7 ms | 8 ms | 1 ms | 1 ms |
| 0 ms | 0 ms | 1 ms | 1 ms | 60/60 | 60/60 | 60/60 | 6 ms | 7 ms | 1 ms | 1 ms |

Raw logs: `zero_time_delay_probe_20260729/`.

## Conclusion

The `20 ms` zero-time send delay can be set to `0 ms`, but it must be paired with a valid startup sequence:

1. activate sidelink bearers, or set a very small explicit activation guard;
2. sync NS3 to at least the activation boundary;
3. then inject the first control-plane transfer batch.

If the delay is set to `0 ms` while the first batch is still injected at simulator time zero, the first batch can be sent before bearer activation and the run loses 10 packets. Therefore, `0 ms` is valid for an optimized startup/warm-start protocol, but the default should remain `20 ms` for backward-compatible cold-start robustness unless paper experiments explicitly report the optimized startup setting.
