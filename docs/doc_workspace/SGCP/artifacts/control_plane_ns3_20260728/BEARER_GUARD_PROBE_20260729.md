# NR sidelink bearer activation guard probe

Date: 2026-07-29

## Question

Can the NR sidelink bearer activation boundary be shorter than the current 10 ms guard?

## Implementation

The NS3 script now exposes the guard as a command-line parameter:

```bash
--slBearerActivationGuardMs=<ms>
```

The default is still `10 ms`, so previous experiments remain reproducible without changing their commands.

The effective activation time is:

```text
finalSlBearersActivationTime
= slBearersActivationTime + slBearerActivationGuardMs
= 1 ms + slBearerActivationGuardMs.
```

The underlying `NrSlHelper::ActivateNrSlBearer()` only schedules activation at the time provided by the caller; it does not enforce a 10 ms minimum. Therefore, the old `10 ms` value is an application/example-level guard, not a hard lower bound in `NrSlHelper`.

## Protocol

- Profile: aggregated three-round cold-start control exchange.
- Requests: 60 packets.
- Packet size: 400 B.
- Vehicles: 20 CAVs.
- Batching: endpoint-disjoint batches of 10 packets.
- No probe-side `first_gap_ms` is used in this experiment.

NS3 command base:

```bash
./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5 --slBearerActivationGuardMs=<value>'
```

Raw logs: `bearer_guard_probe_20260729/`.

## Results

| `slBearerActivationGuardMs` | `batch_step_ms` | Callback delivery | RLC RX | PHY decode OK | Max send time (ms) | Max receive time (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 50/60 | 50/60 | 50/60 | 50 | 51 |
| 5 | 5 | 50/60 | 50/60 | 50/60 | 25 | 26 |
| 5 | 6 | 60/60 | 60/60 | 60/60 | 30 | 31 |
| 5 | 10 | 60/60 | 60/60 | 60/60 | 50 | 51 |
| 2 | 2 | 50/60 | 50/60 | 50/60 | 20 | 21 |
| 2 | 3 | 60/60 | 60/60 | 60/60 | 20 | 21 |
| 2 | 5 | 60/60 | 60/60 | 60/60 | 25 | 26 |
| 1 | 1 | 50/60 | 50/60 | 50/60 | 20 | 21 |
| 1 | 2 | 60/60 | 60/60 | 60/60 | 20 | 21 |
| 1 | 5 | 60/60 | 60/60 | 60/60 | 25 | 26 |
| 0 | 1 | 60/60 | 60/60 | 60/60 | 20 | 21 |
| 0 | 2 | 60/60 | 60/60 | 60/60 | 20 | 21 |
| 0 | 5 | 60/60 | 60/60 | 60/60 | 25 | 26 |

## Interpretation

The activation boundary can be shorter than the current 10 ms guard. Under this control-plane profile:

- `guard=10 ms` needs at least `batch_step_ms=11` or a first-gap activation guard.
- `guard=5 ms` becomes reliable from `batch_step_ms=6`.
- `guard=2 ms` becomes reliable from `batch_step_ms=3`.
- `guard=1 ms` becomes reliable from `batch_step_ms=2`.
- `guard=0 ms` is reliable even at `batch_step_ms=1`.

This confirms that the previous `10 ms` boundary is not a hard NR sidelink bearer activation requirement in the helper. It is a conservative guard inherited from NR sidelink examples and from the local `main.cc` script.

For reproducibility, paper-facing experiments should keep the default `10 ms` guard unless the manuscript explicitly reports an optimized startup configuration. For realtime discussion, it is safe to say that the 10 ms startup guard is conservative and can be shortened in the simulator, while the previously reported data-plane and control-plane headline numbers remain based on the default guard unless stated otherwise.

A dedicated confirmation run with the complete three-round aggregated-summary profile at `--slBearerActivationGuardMs=1` and `batch_step_ms=2` delivered `60/60` callbacks with max receive timestamp `21 ms`. See `FULL_3ROUND_AGGREGATED_GUARD1_CONFIRM_20260729.md`.
