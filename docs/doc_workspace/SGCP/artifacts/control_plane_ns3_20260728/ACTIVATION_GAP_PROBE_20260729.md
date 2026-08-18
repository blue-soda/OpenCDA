# Activation-gap probe for control-plane batching

Date: 2026-07-29

## Purpose

This probe explains why the previous control-plane experiments required `batch_step_ms=11` while `batch_step_ms=10` lost one 10-packet batch.

The key observation is that NR sidelink bearers are activated at:

```cpp
slBearersActivationTime = MilliSeconds(1);
finalSlBearersActivationTime = slBearersActivationTime + MilliSeconds(10);
```

Therefore, the effective activation boundary is `11 ms`.

In the original probe, after the first `transfer_requests` batch, subsequent batches were injected every `batch_step_ms`. With `batch_step_ms=10`, the second batch is scheduled at 10 ms, before the NR sidelink bearer is fully active. This explains the deterministic `50/60` result in the 60-packet aggregated profile: request IDs `11-20` are the missing batch.

No NS3 source parameter was changed for this probe.

## Protocol

- Profile: aggregated three-round cold-start control exchange.
- Requests: 60 packets.
- Packet size: 400 B.
- Vehicles: 20 CAVs.
- Batching: endpoint-disjoint batches of 10 packets.
- NS3 command base:

```bash
./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5'
```

The only probe-side change is `--first-gap-ms`, which delays the first sync after the initial transfer batch. This lets subsequent batches start after the 11 ms bearer-activation boundary while preserving the NS3 default parameters.

Raw logs: `activation_gap_probe_20260729/`.

## Results

| `batch_step_ms` | `first_gap_ms` | Callback delivery | RLC RX | PHY decode OK | Max send time (ms) | Max receive time (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 11 | 60/60 | 60/60 | 60/60 | 51 | 52 |
| 10 | 12 | 60/60 | 60/60 | 60/60 | 52 | 53 |
| 10 | 15 | 60/60 | 60/60 | 60/60 | 55 | 56 |
| 5 | 11 | 60/60 | 60/60 | 60/60 | 31 | 32 |
| 5 | 15 | 60/60 | 60/60 | 60/60 | 35 | 36 |
| 5 | 20 | 60/60 | 60/60 | 60/60 | 40 | 41 |
| 1 | 11 | 60/60 | 60/60 | 60/60 | 20 | 21 |
| 2 | 11 | 60/60 | 60/60 | 60/60 | 20 | 21 |
| 3 | 11 | 60/60 | 60/60 | 60/60 | 23 | 24 |
| 4 | 11 | 60/60 | 60/60 | 60/60 | 27 | 28 |
| 3 | 12 | 60/60 | 60/60 | 60/60 | 24 | 25 |
| 4 | 12 | 60/60 | 60/60 | 60/60 | 28 | 29 |

## Interpretation

The previous `batch_step_ms=10` failure is primarily an activation-boundary artifact:

- with no first-gap guard, the second batch is injected at 10 ms;
- the NR sidelink bearer activation boundary is 11 ms;
- request IDs `11-20` are consequently not transmitted through NR sidelink RLC in the failing run;
- adding `first_gap_ms=11` restores full delivery even when subsequent `batch_step_ms` is below 10 ms.

This also explains why reducing `NrSlRbsTimer` did not help: the failing batch was injected before bearer activation, not delayed by the RLC buffer-status-report cycle.

For paper-facing realtime claims, the conservative statement is:

- steady/warm-start compact control exchange fits within the 40 ms control budget;
- the previously observed 10 ms batch-step boundary was a startup synchronization artifact;
- a valid implementation should not inject control packets before NR sidelink bearer activation, or should perform one initialization sync after activation before the per-frame control exchange.

The `step=1/2/3/4/5 ms` results are diagnostic throughput probes. They should not be interpreted as proof that multi-round coalition decisions with causal dependencies can complete in one such burst.

A follow-up guard sweep (`BEARER_GUARD_PROBE_20260729.md`) confirms that the current 10 ms activation guard is configurable: reducing `--slBearerActivationGuardMs` shifts the reliable no-first-gap batch step accordingly, and `guard=0 ms` delivers `60/60` even at `batch_step_ms=1`.
