# Full three-round aggregated control replay with 1 ms bearer guard

Date: 2026-07-29

## Purpose

Confirm the complete three-round cold-start aggregated-summary control profile under a shortened NR sidelink bearer activation guard.

## NS3 command

```bash
./ns3 run 'scratch/vanet/main.cc --simTime=5.0 --enableTimeSync=true --carlaHost=auto --targetSubchannels=10 --slMcs=28 --slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5 --slBearerActivationGuardMs=1'
```

The default remains `--slBearerActivationGuardMs=10`, so this run is an explicit diagnostic configuration and does not change previous experiment reproducibility.

## Probe command

```powershell
conda run --no-capture-output -n opencda python docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\control_plane_probe.py `
  --profile aggregated `
  --limit-requests 60 `
  --batch-size 10 `
  --endpoint-disjoint `
  --batch-step-ms 2 `
  --upload-plan-output docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\full_3round_aggregated_guard1_20260729\plan_summary60_guard1_step2_confirm.csv `
  --sync-timeout 8 `
  --drain-seconds 2
```

## Result

| Profile | Packets | Guard | Batch step | Callback | RLC RX | PHY decode OK | Max send | Max receive | Mean delay | Max delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregated three-round cold-start summary | 60 | 1 ms | 2 ms | 60/60 | 60/60 | 60/60 | 20 ms | 21 ms | 1 ms | 1 ms |

Raw logs: `full_3round_aggregated_guard1_20260729/`.

## Interpretation

The full three-round aggregated-summary control profile succeeds with a 1 ms bearer guard and 2 ms batch step. The max receive timestamp remains 21 ms because `CamSenderNR::ScheduleCam()` still applies a 20 ms send delay for requests scheduled at simulator time zero. This result therefore shows that shortening the bearer guard removes the earlier 11 ms activation-boundary bottleneck, but the current application-layer startup delay still dominates the absolute first-frame timestamp.

This is a diagnostic startup configuration. Paper-facing default-guard results remain reproducible with `--slBearerActivationGuardMs=10`.

A follow-up zero-time send-delay probe shows that the remaining 20 ms offset can be removed by setting `--nrSlZeroTimeSendDelayMs=0` and syncing to the bearer activation boundary before the first transfer batch. With `guard=1 ms`, `pre_send_sync=2 ms`, and `batch_step=1 ms`, the same 60-packet profile delivers `60/60` callbacks with max receive timestamp `8 ms`. See `ZERO_TIME_SEND_DELAY_PROBE_20260729.md`.

## Aggregated-summary semantics

The `60 packets = 20 CAVs x 3 rounds` profile should be interpreted carefully.

It is rigorous only under a compact broadcast/groupcast-summary abstraction: in each coalition-maintenance round, every CAV emits one compact sensing summary, and that summary is intended to be available to all vehicles or at least all current/candidate coalition heads that may evaluate a migration involving that CAV. Under NR sidelink broadcast/groupcast semantics, one transmitted summary can serve multiple receivers.

This replay encodes each synthetic request as a unicast CAM-style transfer and counts one application callback per packet. Therefore, this replay verifies the timing of 60 scheduled summary transmissions, but it does not by itself verify multi-receiver broadcast fanout.

A follow-up broadcast probe has now verified the fanout directly. With `--enableControlBroadcast=true` and request-side `cast_type=broadcast`, the bridge installs an NR sidelink broadcast bearer and sends compact summaries to multicast IP `225.0.0.0` / destination L2 ID `255`. The 20-packet smoke test produced `200` callbacks, and the complete 70-packet aggregated control profile produced `699/700` expected half-duplex fanout callbacks with max receive timestamp `15 ms`. See `BROADCAST_CONTROL_PROBE_20260729.md`.

If the control plane were instead implemented as pure unicast to every candidate head, the packet count would still have to be multiplied by the number of receivers per summary.

For the current 20-CAV / about 6-head SGCP setting, a conservative unicast-equivalent summary dissemination would be approximately:

```text
per round: 20 CAV summaries x 6 candidate/current heads = about 120 unicast packets
three rounds: about 360 unicast packets
```

If every CAV must receive every summary, the worst-case all-CAV unicast equivalent is:

```text
per round: 20 x 19 = 380 unicast packets
three rounds: 1140 unicast packets
```

These are not the intended SGCP control protocol; they are conservative unicast fanout equivalents.

## Can the real coalition algorithm use aggregated summaries?

Yes, at the algorithm-information level, provided that the summaries are disseminated to the vehicles or heads that perform admission checks. The current replay is still a feasibility probe rather than a full distributed implementation.

The current SGCP-PV coalition algorithm needs:

- each CAV's sensed grid set `G_i`;
- each CAV's normalized grid quality `q_i(g)`;
- current coalition membership and candidate target membership;
- pairwise mutual-confirmation values
  `W_ij = sum_g min(q_i(g), q_j(g))`;
- affected-coalition admission delta
  `Delta Phi_C = sum_{j in S_tgt} W_ij - sum_{j in S_src\{i}} W_ij`.

These quantities can be computed from compact per-CAV grid-quality summaries and coalition membership summaries. The algorithm does not require raw LiDAR control-plane transmission, nor does every logical proposal/reply/PV check have to be a separate CAM packet.

In the current 41-frame trace, each CAV has a sparse non-empty grid map, so a compact summary can plausibly encode `(grid_id, quantized q_i(g))` pairs plus pose and membership metadata in a small control packet or in a small number of control packets. The 400-byte synthetic summary used here is therefore reasonable as a control-plane feasibility approximation, but a production implementation should explicitly define the summary encoding, receiver set, and worst-case fanout.

For the paper, the precise statement should be:

> SGCP's distributed coalition maintenance can be implemented with compact per-CAV sensing summaries from which pairwise `W_ij` and affected-coalition potential increments are computed. The NS3 control-plane probe verifies the delivery latency of scheduled compact-summary transmissions under the broadcast/groupcast-summary abstraction; it does not claim that the current offline replay code already executes a fully distributed message protocol or that unicast fanout is free.
