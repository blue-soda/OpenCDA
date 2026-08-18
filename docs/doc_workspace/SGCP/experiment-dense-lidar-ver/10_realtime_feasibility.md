# SGCP Realtime Feasibility

This document summarizes convergence, iteration counts, computational complexity, distributed solving overhead, and communication latency for the current SGCP algorithm. It is intended as paper-facing evidence that SGCP fits the 100 ms cooperative-perception cycle.

Status note: the payload, AP and GFLOPs values below have been updated with the
new Table-1 hybrid scheduler candidate where available. The distributed
coalition and scheduler-solver timing profiler is still the previous
`dynamic_cv` profiler unless explicitly marked otherwise; rerun that profiler
if `hybrid_round_robin_dynamic_marginal` is promoted as the final algorithm.

## Timing Budget

The realtime protocol uses the following per-frame budget.

| Component | Budget | Notes |
| --- | ---: | --- |
| Full perception cycle | 100 ms | One cooperative perception frame. |
| Raw-LiDAR data-plane transmission | 60 ms | Scheduled point-cloud grid payload. |
| Distributed algorithm and control signaling | 40 ms | Coalition maintenance, scheduler solving, and lightweight control messages. |

The dense SGCP headline operating point uses 20 CAVs, derived coalition
capacity `N_max = ceil(N / floor(K / B_h))`, `rho_th=2`, 40 MHz total
bandwidth, 10 target subchannels, and the NS3-calibrated channel estimator
with `tb_size=899 bytes`, `slot=0.5 ms`, `symbols=12`, and `mcs=28`.

## End-to-End Breakdown

This is the compact table intended for paper writing. It separates measured
wall-clock/control-plane latency from detector compute load, because the clean
package does not contain a full GPU detector-runtime tail-latency trace.

| Pipeline component | Mean | P95 | Max / bound | Evidence source | Paper-facing use |
| --- | ---: | ---: | ---: | --- | --- |
| Pairwise coalition cache | 5.86 ms | 8.71 ms | 9.97 ms | Dense realtime profiler | Conservative full recomputation of `W_ij`; online maintenance can update it incrementally. |
| Coalition admission checks | 8.48 ms cold / 3.25 ms warm | 11.69 ms cold / 5.30 ms warm | 13.84 ms cold / 7.54 ms warm | Dense realtime profiler | Cold start is initialization/topology-change; warm start is the per-frame online mode. |
| Control exchange | - | - | 15 ms optimized compact-summary NS3 max | NS3 control-plane probe | Activation-synchronized `guard=1 ms`, zero-time send delay `0 ms`, compact summaries. |
| Scheduler solving | 7.51 ms | 10.47 ms | 11.77 ms | Dense realtime profiler, previous `dynamic_cv` scheduler | Conservative timing proxy; rerun for hybrid before claiming an exact hybrid solver tail. |
| Raw-LiDAR upload | 20.27 ms NS3 RLC replay; 35.68 ms trace estimate | 36.56 ms NS3 RLC replay; 37.45 ms trace estimate | 37.56 ms NS3 RLC replay; 37.47 ms trace estimate | 41-frame SGCP-main-point NS3 replay and dense hybrid SGCP trace | Fits the 60 ms raw data-plane budget. |
| Detector inference | - | - | 6.63 detector calls/frame, 593.34 GFLOPs/frame | Dense hybrid compute profile | Report as compute load, not wall-clock latency. |
| Box aggregation | - | - | 0.89 Mbps box payload | Dense hybrid communication accounting | Box communication included in Total Mbps; NMS runtime is negligible but not separately tailed. |

The control/algorithm stage is pipelined: local admission computation and
compact-summary exchange overlap. A conservative P95 critical path is

```text
8.71 ms pairwise cache
  + max(11.69 ms cold-start admission, 15 ms optimized NS3 control exchange)
  + 10.47 ms scheduler
  = 34.18 ms < 40 ms.
```

The raw-LiDAR data-plane trace maximum is `37.47 ms`. A direct 41-frame NS3
RLC replay of the SGCP main-table upload plan reports `410/410` complete
logical raw-LiDAR requests with mean/P95/max communication time
`20.27/36.56/37.56 ms`, below the reserved `60 ms` data-plane budget. Thus the
current SGCP evidence supports a `100 ms` cooperation cycle when coalition
maintenance is run in warm-start mode and control messages are aggregated into
compact summaries.

The paper-facing delivery and delay metric is the NS3 RLC-side request
completion time. Application callbacks are retained only as diagnostics because
large raw-LiDAR logical requests are transmitted as multiple UDP chunks, and
callback counts therefore do not define lower-layer request completion.

## Convergence

### Coalition Formation

Source file:

```text
C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\clustering\potential_verified_cov_coalition_game.py
```

SGCP uses potential-verified pairwise multi-view coalition formation. For a partition `Pi`, the coalition potential is

```text
Phi_C(Pi)
= sum_{S in Pi} sum_{i<j, i,j in S} W_ij

W_ij = sum_g min(q_i(g), q_j(g)).
```

A migration of vehicle `i` from source coalition `S_src` to target coalition `S_tgt` is accepted only if

```text
Delta Phi_C
= Phi_C(Pi after i -> S_tgt) - Phi_C(Pi before) > 0.
```

Only the source and target coalitions are affected by a migration, so the exact distributed admission check can be written as

```text
Delta Phi_C(i -> S_tgt)
= sum_{j in S_tgt} W_ij
 - sum_{j in S_src \ {i}} W_ij.
```

Because every accepted migration strictly increases `Phi_C` and the feasible partition space is finite under `N_max`, the coalition process is finite and cannot cycle. The implementation also caps the outer loop at 20 rounds, but the 20-CAV experiment converges much earlier.

Two convergence modes are distinguished. Cold-start replay initializes every frame from singleton coalitions; it is useful for measuring worst-case convergence but is more conservative than online execution. Warm-start maintenance initializes frame `t` from the final partition of frame `t-1`, which is the relevant online control-plane mode because vehicle topology changes smoothly between adjacent 100 ms perception cycles.

Measured over 41 frames:

| Metric | Mean | Median | P95 | Max | Min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold-start outer rounds | 2.80 | 3.00 | 3.00 | 3.00 | 2.00 |
| Cold-start accepted migrations/frame | 14.17 | 14.00 | 15.00 | 16.00 | 12.00 |
| Cold-start potential checks/frame | 373.71 | 428.00 | 437.00 | 437.00 | 257.00 |
| Warm-start outer rounds, steady-state | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Warm-start accepted migrations, steady-state | 0.03 | 0.00 | 0.00 | 1.00 | 0.00 |
| Warm-start potential checks, steady-state | 98.78 | 104.00 | 104.00 | 183.00 | 88.00 |

One outer round means that every CAV receives one migration opportunity. Thus, cold-start SGCP coalition formation converges within at most three vehicle-action rounds in this 20-CAV scenario. In warm-start maintenance, the first cold-start partition remained locally stable for all subsequent 40 frames: the algorithm performed one confirmation round and accepted no migration. The warm-start final partition differs from the per-frame cold-start partition in most frames because both are local optima; the current headline AP table is still reported under the per-frame replay protocol, while the realtime control-plane evidence uses the online warm-start maintenance protocol.

### Resource Scheduling

Source files:

```text
C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\dynamic_cv_potential_game.py
C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\hybrid_round_robin_dynamic_marginal.py
```

The formal `dynamic_cv` scheduler is a deterministic one-pass dynamic C->V
scheduler with density-capped random point upload:

```text
Stage 1: allocate one coverage-oriented link per cluster head when possible,
         using the current receiver evidence Q_h^A(g).
Stage 2: allocate remaining subchannels to the largest view-quality gains,
         again using the updated receiver evidence.
Upload:  for every admitted sender-grid action, cap the receiver grid at
         rho_th and upload a deterministic random subset of points if the
         sender grid would exceed the residual density quota.
```

The hybrid scheduler candidate keeps the same dynamic marginal and density-cap
upload model, but changes the admission order: it gives each cluster head one
round-robin opportunity, then greedily assigns the remaining subchannels to the
largest dynamic marginal gains. The number of subchannels, per-head link cap,
and candidate grid sets are finite. Therefore both schedulers terminate after
one bounded pass and do not require an iterative best-response convergence
process.

Measured over 41 frames:

| Metric | Value |
| --- | ---: |
| Scheduler iterations | 1 |
| Cluster heads/frame | 6.00 |
| Candidate scheduled links/frame before density-cap upload trimming | 10.00 |
| Admitted source CAVs/frame after density-cap upload trimming | 2.51 |
| Admitted selected grids/frame after density-cap upload trimming | 47.94 |

## Complexity

Let:

```text
N = number of CAVs
M = number of coalitions
N_max = maximum coalition size
G = candidate grids per vehicle/link
B = number of target subchannels
I = coalition outer rounds
```

| Algorithm | Complexity | With bounded `N_max` / `B` |
| --- | --- | --- |
| Coalition formation, direct replay implementation | `O(I * N * M * N_max^2 * G)` | approximately `O(I * N * M * G)` |
| Coalition admission with cached `W_ij` | `O(I * N * M * N_max)` after pairwise cache | approximately `O(I * N * M)` |
| Pairwise cache construction | `O(N^2 * G)` | one frame-level preparation step; can be maintained incrementally across frames |
| Dynamic C->V scheduler with density cap | `O(B * N * G log G + P_sel)` | near-linear in candidate grid scans because `B=10` is fixed; `P_sel` is the number of points in admitted grids considered for deterministic random truncation. |

The important distributed property is that a migration only requires the source and target coalition summaries. No global partition recomputation is required.

## Distributed Computation Time

The distributed timing profile was measured by instrumenting the coalition code at the vehicle-action level. For each vehicle, all processing time across rounds is accumulated. Different vehicles are then treated as parallel distributed processors, and the distributed frame-level computation time is estimated by the maximum cumulative vehicle time.

This profile excludes PCD loading and neural inference. The distributed control-plane profiler was run on the same 20-CAV cooperative-perception topology used by the SGCP study; its cost is dominated by CAV/grid-summary exchange and bounded-size admission checks rather than by raw LiDAR point count. Dense data-plane payload statistics are reported separately below using the 2026-07-29 dense headline trace.

| Measurement | Mean | Median | P95 | Max | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Pairwise cache construction | 5.86 ms | 5.07 ms | 8.71 ms | 9.97 ms | Recomputes `W_ij=sum_g min(q_i,q_j)` from compact grid summaries for all CAV pairs. |
| Cached pairwise distributed admission, cold start | 8.48 ms | 8.40 ms | 11.69 ms | 13.84 ms | Per-vehicle distributed estimate: each CAV accumulates its own action time across rounds, and frame time is the max CAV time. |
| Cached pairwise distributed admission, warm start | 3.25 ms | 2.94 ms | 5.30 ms | 7.54 ms | Online maintenance initialized from the previous frame's partition. |
| Scheduler solving | 7.51 ms | 7.17 ms | 10.47 ms | 11.77 ms | Previous `dynamic_cv` one-pass scheduler plus density-capped upload planning; rerun if hybrid is selected as final. |

The 2026-07-30 dense profiler uses the `dynamic_cv` scheduler and the current potential-verified coalition code on `v2xp_cluster_carla_dense`, `rho_th=2`, `N_max=4`, 40 MHz / 10ch. In a deployed distributed implementation, `W_ij` can be maintained incrementally or computed from exchanged grid summaries, so the online admission stage is represented by the cached-admission rows rather than by a centralized replay of every utility check. The 2026-08-01 hybrid run updates AP/payload/GFLOPs but has not yet produced a separate per-frame solver-time profiler.

## Control-Plane Communication Time

The following first gives analytical lower-bound packet accounting under the calibrated 40 MHz / 10-subchannel setting. One subchannel can serve one `899 byte` transport block every `0.5 ms`, so 10 orthogonal control packets can be served per `0.5 ms` tick when each control packet fits in one TB.

| Component | Packets | Lower-bound time | Notes |
| --- | ---: | ---: | --- |
| Warm-start coalition summary exchange, 1 round | 20 | 1.0 ms | One compact summary per CAV; sufficient for steady-state confirmation in the measured 41-frame trace. |
| Cold-start coalition proposals and replies, 3 rounds | 120 | 6.0 ms | `20 proposals + 20 replies` per round; used only for worst-case initialization analysis. |
| Cold-start potential-verified source/target checks, 3 rounds | 120 | 6.0 ms | Two extra packets per vehicle-action if admission checks are not summarized. |
| Cold-start membership updates for accepted migrations | about 30 | 1.5 ms | Based on 14.17 accepted migrations/frame. |
| Cold-start coalition control-plane lower bound | about 270 | about 13.5 ms | Analytical lower bound for a non-aggregated cold-start protocol, not the recommended per-frame online encoding. |
| Scheduler candidate summaries and grants | 24-44 | 1.5-2.5 ms | About 14 member summaries plus 10 grants, with optional ACK/reservation messages. |

The control-plane payload is much smaller than raw-LiDAR grid payload. The lower-bound calculation is therefore mainly useful for showing that control signaling is not the bottleneck.

A direct NS3 control-plane sweep was then rerun with compact per-CAV control
summaries and the optimized startup parameters used by the dense experiment
package: `slBearerActivationGuardMs=1`, `nrSlZeroTimeSendDelayMs=0`, and an
explicit pre-send sync to the bearer activation boundary. This is the
paper-facing steady-state control protocol: a control summary aggregates a
vehicle's maintenance/admission metadata and scheduler candidate metadata
instead of emitting every logical utility check as an independent application
packet.

| NS3 control-plane probe | Planned packets | Received callbacks | RLC complete requests | Delivery ratio | Max receive timestamp | Mean delay | Max delay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Optimized unicast compact summaries, complete aggregated control profile | 70 | 70 | 70 | 1.00 | 15 ms | 1 ms | 1 ms |
| Optimized broadcast/groupcast compact summaries, complete aggregated control profile | 70 | 699/700 expected half-duplex fanout callbacks | 699/700 | 0.9986 | 15 ms | 1 ms | 1 ms |

The complete aggregated control profile contains `60` coalition-round summaries
(`20` CAVs times `3` cold-start rounds), `6` scheduler summaries, and `4`
scheduler grant packets. The probe uses 20 vehicles, 400-byte control packets,
batches of 10 packets, `batch_step=2 ms`, pre-send sync `2 ms`, and the same
legal high-capacity NS3 setting as the data-plane replay (`MCS=28`, `PSSCH
symbols=12`, `PSCCH=10`, `RRI=5`, 10 target subchannels). The broadcast row
verifies one-to-many compact-summary dissemination: each transmitted summary is
received by the non-transmitting vehicles in the batch, subject to half-duplex
reception. The missing `1/700` callback is a single deterministic fanout miss;
all 70 broadcast requests are scheduled and received by at least nine
non-transmitting CAVs.

The same sweep also confirms the startup boundary: if requests are injected at
simulator time zero without pre-send activation sync, only `60/70` request ids
are delivered because the first batch races bearer activation. With the
activation sync, both `guard=1 ms` and the diagnostic `guard=0 ms` settings
deliver all 70 unicast requests; the broadcast/groupcast fanout row reaches
`699/700` callbacks. Therefore the paper-facing control protocol should be
reported with activation-synchronized optimized startup, while the original
default startup delay remains a conservative reproducibility setting.

Diagnostic probes show why SGCP should not send every logical admission check as a separate CAM application packet: an unaggregated 314-packet control stream caused overlap/collision and delivered no application callbacks, while 70-packet repeated-source probes delivered only 50-60 callbacks. Therefore, three cold-start rounds should not be encoded as dozens of repeated-source CAM packets inside one control window. These diagnostics support warm-start maintenance and compact-summary aggregation.

A separate cold-start NS3 probe explicitly tested both aggregated and unaggregated three-round control sequences. With one compact per-CAV summary per round, the three-round coalition-control sequence contains 60 packets. If batching starts immediately at simulator time zero, a 10 ms step loses the second batch because it is injected before the 11 ms NR sidelink bearer-activation boundary; an 11 ms step is reliable and finishes at 56 ms. If the probe waits until the activation boundary before the second batch (`first_gap_ms=11`), a 10 ms step is reliable and finishes at 52 ms, and diagnostic shorter steps also deliver all packets. The unaggregated 270-packet coalition-only cold-start sequence becomes reliable at an 11 ms batch step but finishes at 287 ms. Including scheduler control messages gives 314 packets and finishes at 342 ms with 100% delivery. This confirms that full cold-start reconfiguration is a multi-frame initialization/topology-change procedure, not a per-frame 40 ms control-plane operation.

| Cold-start control probe | Packets | Batch step | Delivery | RLC complete | Max receive timestamp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-round aggregated coalition summaries | 60 | 10 ms | 50/60 | 50/60 | 51 ms |
| Three-round aggregated coalition summaries | 60 | 11 ms | 60/60 | 60/60 | 56 ms |
| Three-round aggregated summaries, after activation guard | 60 | first gap 11 ms, then 10 ms | 60/60 | 60/60 | 52 ms |
| Three-round aggregated summaries, diagnostic burst | 60 | first gap 11 ms, then 5 ms | 60/60 | 60/60 | 32 ms |
| Three-round coalition only | 270 | 10 ms | 260/270 | 260/270 | - |
| Three-round coalition only | 270 | 11 ms | 270/270 | 270/270 | 287 ms |
| Three-round coalition + scheduler control | 314 | 11 ms | 314/314 | 314/314 | 342 ms |

The activation-guard rows diagnose the startup artifact only. They do not imply that causally dependent coalition rounds should be collapsed into one unordered burst; they show that the earlier 10 ms failure came from sending before bearer activation rather than from insufficient steady-state control-plane capacity.

The 10 ms bearer-activation guard is configurable in the simulator and is not a hard `NrSlHelper` lower bound. A diagnostic sweep shows that reducing the guard shifts the reliable no-first-gap batch step accordingly: `guard=5 ms` is reliable from `step=6 ms`, `guard=2 ms` from `step=3 ms`, `guard=1 ms` from `step=2 ms`, and `guard=0 ms` from `step=1 ms`. The dense experiment package uses the optimized `guard=1 ms`, zero-time delay `0 ms`, and activation-synchronized control-plane setting. Previous default-guard results remain reproducible because the NS3 defaults are unchanged unless explicitly overridden.

A dedicated confirmation run for the complete three-round aggregated-summary profile with `guard=1 ms` and `step=2 ms` delivered `60/60` callbacks, `60/60` RLC-complete requests, and `60/60` PHY decodes, with max receive timestamp `21 ms`. This `60-packet` profile represents `20` compact summary transmissions per round under a broadcast/groupcast-summary abstraction; it is not a pure-unicast fanout count. If every summary were unicasted to every candidate head, the packet count would scale with the number of heads. The remaining 20 ms absolute offset comes from the application-layer zero-time send delay in `CamSenderNR`, not from bearer activation.

The zero-time application delay is also configurable and is not a PHY/MAC requirement. With `guard=1 ms`, `nrSlZeroTimeSendDelayMs=0`, and an explicit pre-send sync to the 2 ms activation boundary, the complete 60-packet three-round summary finishes at `13 ms` using `batch_step=2 ms`, or `8 ms` using a diagnostic `batch_step=1 ms`. If the delay is set to zero without the pre-send activation sync, the first batch can be injected before bearer activation and the run loses packets. Thus, zero delay is valid for an optimized startup or warm-start protocol, while the default 20 ms delay is kept for backward-compatible cold-start robustness.

## Paper-Facing Pipeline Breakdown

The table below is the recommended compact breakdown for the paper. It avoids CDF-style claims because the current package has mean/P95 control and data-plane measurements, but no full end-to-end detector-runtime tail-latency trace.

| Pipeline component | Paper-facing value | Evidence source | Notes |
| --- | ---: | --- | --- |
| Coalition maintenance, pairwise cache | 5.86 ms mean / 8.71 ms P95 | Dense realtime profiler | Recomputes `W_ij=sum_g min(q_i(g),q_j(g))` from compact grid summaries. This is conservative; online execution can maintain the cache incrementally. |
| Coalition maintenance, admission checks | 8.48 ms mean / 11.69 ms P95 cold start; 3.25 ms mean / 5.30 ms P95 warm start | Dense realtime profiler | Cold start converges in at most 3 vehicle-action rounds. Warm-start maintenance is the per-frame online mode and needs one confirmation round in the measured sequence. |
| Control exchange | 15 ms optimized compact-summary NS3 max; 21 ms conservative default-delay max | NS3 control-plane probes | Uses Guard `1 ms`, zero-time send delay `0 ms`, and activation-synchronized compact summaries for the optimized row. |
| Scheduler solving | 7.51 ms mean / 10.47 ms P95 | Dense realtime profiler, previous `dynamic_cv` scheduler | Conservative timing proxy; rerun for exact hybrid timing if hybrid is selected as final. |
| Raw-LiDAR upload | 20.27 ms mean / 36.56 ms P95 / 37.56 ms max in 41-frame NS3 RLC replay; 35.68 ms mean / 37.45 ms P95 / 37.47 ms max trace estimate | Dense hybrid SGCP trace and SGCP-main-point NS3 replay | Fits the reserved 60 ms data-plane budget. |
| Detector inference | 6.63 detector forwards/frame, 593.34 GFLOPs/frame | Dense hybrid Table 1/3 compute profile | Reported as compute load rather than wall-clock latency; no detector-runtime CDF should be claimed from this package. |
| Box aggregation | 0.89 Mbps box payload/frame for SGCP | Dense hybrid Table 1/3 communication accounting | NMS/box aggregation wall-clock is not separately profiled; communication overhead is already included in Total Mbps. |

The control/algorithm critical path is pipelined: vehicles compute local checks while compact summaries are exchanged. A conservative control-stage bound is therefore:

```text
8.71 ms P95 pairwise cache
  + max(11.69 ms P95 cold-start admission, 15 ms NS3 optimized control exchange)
  + 10.47 ms P95 scheduler
  = 34.18 ms < 40 ms.
```

## Raw-LiDAR Data-Plane Communication Time

The 41-frame dense SGCP headline trace contains:

| Metric | Mean | Median | P95 | Max | Min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw payload bytes/frame | 355,222.24 | 355,760.00 | 382,720.00 | 383,232.00 | 329,056.00 |
| Raw LiDAR Mbps | 28.42 | - | - | - | - |
| Trace receiver-side max communication time | 35.68 ms | 36.24 ms | 37.45 ms | 37.47 ms | 31.36 ms |
| Service-rate lower-bound data time | 19.63 ms | 20.00 ms | 20.00 ms | 20.00 ms | 18.00 ms |

The trace receiver-side max time is computed from the current NS3-calibrated channel estimator at the rho-Pareto selected dense SGCP operating point. It remains comfortably below the 60 ms data-plane boundary, while the service-rate lower-bound payload time is about 18-21 ms.

## NS3 Calibration Check

The SGCP main-table replay uses the legal high-capacity sidelink settings
(`MCS=28`, `PSSCH symbols=12`, `PSCCH=10`, `RRI=5`,
`slBearerActivationGuardMs=1`, `nrSlZeroTimeSendDelayMs=0`). Stochastic PHY
decode errors are disabled for this deterministic scheduled-capacity replay;
the experiment therefore checks whether SGCP's fixed subchannel plan fits the
NS3 capacity and timing model.

| Replay setting | Frames | Logical requests | RLC complete | Mean delay | P95 delay | Max delay | Allocated payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGCP main-table 41-frame replay | 41 | 410 | 410/410 | 20.27 ms | 36.56 ms | 37.56 ms | about 899 B/grant |

This replay validates the service-rate assumption used by the current trace
estimator: the calibrated NS3 path serves about one `899 byte` payload per
subchannel grant and keeps the replayed SGCP raw-LiDAR stream below the 60 ms
data-plane budget. Large logical requests are chunked at the application socket
layer while preserving one `request_id`; RLC TX/RX byte accumulation is used to
measure logical request completion.

## Measurement Provenance

| Quantity | Source |
| --- | --- |
| Cold-start convergence rounds, accepted migrations, potential checks | Temporary per-vehicle profiler over `PotentialVerifiedCOVCoalitionGame` on the 20-CAV SGCP topology; used as control-plane feasibility evidence. |
| Warm-start coalition maintenance | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\warm_start_cluster_20260729\`; frame `t` is initialized from frame `t-1` final partition. |
| Distributed cached-admission time | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\realtime_profile_dense_20260730\realtime_profile_dense.csv`; exact cached `W_ij=sum_g min(q_i(g),q_j(g))` on the dense 41-frame trace. |
| Scheduler solving time | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\realtime_profile_dense_20260730\realtime_profile_dense.csv`; previous dynamic C->V scheduler profile with density-capped random upload planning over the same 41 frames. |
| Control-plane packet time | Compact-summary, cold-start, RLC-BSR timer, activation-gap, bearer-guard, zero-time-delay, and broadcast/groupcast NS3 replays in `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\control_plane_ns3_20260728\`; the 2026-07-30 rerun writes `guard_zero_sweep_20260729\guard_zero_sweep_results.csv`. Analytical lower-bound rows are kept for intuition. |
| Dense raw payload, link count, grid count, estimator data time | Hybrid row: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f\trace.csv`; previous dynamic row: `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\dense_dynamic_cv_sensitivity_20260729\budget_rho2_mbps40\trace.csv`. |
| Data-plane NS3 replay calibration | `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f\ns3_replay_41f_udpchunk_noerr_rlctime\summary.json`; supporting explanation in `C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts\hybrid_round_robin_dynamic_marginal_20260801\table1_41f\ns3_replay_summary.md`. |

## Realtime Conclusion

SGCP satisfies the 100 ms realtime budget under the current protocol:

```text
Perception cycle budget:       100 ms
Raw-LiDAR data-plane budget:    60 ms
Algorithm/control budget:       40 ms
```

Current evidence:

```text
Raw-LiDAR data-plane trace max:        37.47 ms <= 60 ms
NS3 41-frame RLC replay mean/P95/max:  20.27 / 36.56 / 37.56 ms < 60 ms
Pairwise cache construction mean/P95:   5.86 / 8.71 ms
Cached cold-start admission mean/P95:   8.48 / 11.69 ms
Cached warm-start admission mean/P95:   3.25 / 5.30 ms
Warm-start steady-state rounds:          2.00 max, 0.03 accepted migrations mean
Scheduler solving mean/P95:             7.51 / 10.47 ms (dynamic_cv timing proxy)
Control-plane NS3 compact summary:      15 ms optimized broadcast/groupcast max
                                          15 ms optimized unicast max
                                          21 ms conservative default-delay max
```

The algorithm/control stage is interpreted as a pipelined distributed stage rather than a serial sum of all local computations and all control transmissions. Vehicles compute local migration/admission quantities while compact summaries are prepared and exchanged. Under this protocol, the relevant control-stage critical path is bounded by the larger of distributed admission computation and compact-summary exchange, followed by the one-pass scheduler:

```text
8.71 ms P95 pairwise cache
  + max(11.69 ms P95 cold-start admission, 15 ms NS3 optimized control exchange)
  + 10.47 ms P95 scheduler
  = 34.18 ms
```

The resulting value is below the 40 ms reserved control budget under conservative Python timing. With mean timings and the optimized broadcast/groupcast control probe, the same calculation is `5.86 + max(8.48, 15) + 7.51 = 28.37 ms`; using the conservative default-delay `21 ms` control probe gives `5.86 + max(8.48, 21) + 7.51 = 34.37 ms`. This bound uses cold-start P95 admission and full pair-cache reconstruction every frame, and is therefore conservative for warm-start maintenance, where the measured P95 admission time is only `5.30 ms`. Since the admission profile is measured in Python and the actual distributed implementation only performs bounded-size local checks with cached `W_ij`, the protocol is considered realtime-feasible.

The scheduler is comfortably realtime per frame. Coalition formation is provably finite; cold-start initialization converges within at most three vehicle-action rounds in the 20-CAV trace, while online warm-start maintenance requires only one confirmation round after the first frame in the measured sequence. The centralized Python replay implementation is intentionally conservative and should not be interpreted as the runtime cost of the distributed protocol.

The explicit cold-start NS3 probe further shows that a three-round cold-start exchange takes 56 ms with compact per-CAV summaries and 287-342 ms when logical control messages are unrolled. Therefore, SGCP should not be described as executing full cold-start coalition formation in every 100 ms perception cycle. The realtime claim applies to the steady-state warm-start protocol, while cold-start coalition formation is an initialization or topology-change reconfiguration stage.
