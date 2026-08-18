# SGCP Experiment Parameters and Algorithm Details

This document records the dense-LiDAR experiment parameters and the exact SGCP algorithm used by the dense experiment package.

## Environment and Dataset Parameters

| Item | Value | Paper-facing note |
| --- | --- | --- |
| Scenario | `v2xp_cluster_carla_dense` | Dense-LiDAR variant of the CARLA/OpenCDA cooperative perception scenario. |
| CARLA map/config source | `opencda/scenario_testing/config_yaml/v2xp_cluster_carla_dense.yaml` plus `default.yaml` | Same 20-CAV layout as `v2xp_cluster_carla`; vehicle LiDAR parameters are overridden for dense export. |
| Offline dataset root | `D:\Data\Carla` | Dataset path used for replay experiments. |
| Dataset scenario id | `2026_07_29_02_32_08` | Dense 41-frame experiment dump. |
| Dataset export command | `conda run --no-capture-output -n opencda python opencda.py -t v2xp_cluster_carla_dense --dump` | Run with one CARLA server active; output is placed under `D:\Data\Carla` and then replayed offline. |
| Total simulated vehicles | 100 | One `single_cav` plus 99 traffic-manager vehicles in the scenario; 19 of the traffic-manager vehicles are managed CAVs. |
| CAV count | 20 | One ego/single CAV plus 19 managed traffic CAVs. |
| Background vehicles | 80 | Simulated traffic vehicles without CAV perception participation. |
| Evaluation frames | 41 frames, `000060` to `000140` | Same receiver-frame set for all clean tables unless a parameter is varied. |
| CARLA fixed step | 0.05 s | From `world.fixed_delta_seconds`. |
| Perception interval | 0.1 s/frame | The dump uses every second simulator tick for perception/evaluation. |
| Detector/checkpoint | attentive raw point-cloud detector | All point-cloud-to-box inference uses the same attentive-derived early-fusion checkpoint. |
| Raw-LiDAR fusion | early fusion | Used whenever raw point clouds are exchanged. |
| Box aggregation | NMS over predicted 3D boxes | Used by `prediction_nms`, `global_box_nms`, and `inter_cluster_nms` rows. |
| LiDAR range | 50 m | Inherited from `vehicle_base.sensing.perception.lidar.range`. |
| LiDAR channels | 32 | Inherited from `default.yaml`. |
| LiDAR points per second | 320,000 | Dense setting inspired by HESAI XT32M-class LiDAR throughput. |
| LiDAR rotation frequency | 20 Hz | At least one full rotation within a 100 ms perception cycle. |
| Grid size | 10 m x 10 m | Offline replay default and OpenCDA perception-grid default. |
| Required perception range | 150 m | Offline replay default is `3 x lidar_range`. |
| Raw point format | 16 bytes/point | XYZ plus intensity as `float32`. |

## Network and Channel Parameters

| Item | Value | Paper-facing note |
| --- | --- | --- |
| Total bandwidth | 40 MHz | Main protocol setting. |
| Target subchannels | 10 | Main protocol setting. |
| Perception cycle | 100 ms | End-to-end perception period. |
| Communication feasibility target | reported payload should complete within the communication portion of the 100 ms cycle | SGCP headline payload has exact NS3 replay max delay below 60 ms in selected checks. |
| Channel estimator | NS3-calibrated estimator | Shared by SGCP and baseline schedulers in clean tables. |
| Transport block size | 899 bytes | `--ns3-tb-size-bytes 899`. |
| Slot duration | 0.5 ms | `--ns3-slot-duration-ms 0.5`. |
| Subchannel PRBs | 10 | `--ns3-subchannel-prbs 10`. |
| PSSCH symbols | 12 | `--ns3-symbols-per-slot 12`. |
| MCS | 28 | `--ns3-mcs 28`. |
| K | 2 for reported PCS/EdgeCooper rows | Maximum concurrent orthogonal-subchannel senders accepted by one receiver in one communication window. K=1 values are noted under the affected tables. |
| V2V communication range | 35 m for PCS/EdgeCooper adaptations | Matches OpenCDA default V2X range and avoids infeasible long-range raw-LiDAR scheduling. |

The estimator computes a per-subchannel payload rate as:

```text
R_ch = 8 * TB_size / slot_duration
```

with `TB_size=899 bytes` and `slot_duration=0.5 ms`, giving `14.384 Mbps` per target subchannel before higher-level admission and half-duplex/concurrency constraints.

## SGCP Parameters

| Parameter | Value | Meaning |
| --- | --- | --- |
| Clustering algorithm | `potential_verified_cov_coalition_game` | potential-verified pairwise multi-view coalition formation with next-frame motion stability. |
| Resource scheduler | `dynamic_cv` / `hybrid_round_robin_dynamic_marginal` | `dynamic_cv` is the documented formal scheduler; `hybrid_round_robin_dynamic_marginal` is the current Table 1/3 headline candidate and scheduler probe. Promote the hybrid pseudocode below only after the final method decision. |
| Cluster-size cap `N_max` | `ceil(N / floor(K / B_h))` | Derived from CAV count, target subchannels, and per-head receive budget; not tuned as a hyperparameter. |
| Density threshold `rho_th` | 2 | Normalizes grid quality for the dense headline row; selected by the dense rho Pareto sweep. |
| Head RB budget | 2 | Maximum scheduled raw-LiDAR sender links per cluster head. |
| Target subchannels | 10 | Shared global channel budget. |
| Raw-LiDAR Mbps budget | saturated rho-Pareto point | Hybrid candidate realized payload is `28.42 Raw Mbps` and `29.31 Total Mbps`; the previous `dynamic_cv` row is `27.84 Raw Mbps` and `28.69 Total Mbps`. |
| Receiver policy | all cluster heads | Only cluster heads perform early-fusion detector inference; all cluster heads share boxes for inter-cluster NMS. |
| Upload mode | grid | Selected raw-LiDAR grid blocks are uploaded from members to cluster heads. |
| Late aggregation | inter-cluster NMS | Detection boxes from cluster heads are aggregated at scene level. |
| Stability horizon | actual next perception frame, 0.1 s | Derived from `fixed_delta_seconds=0.05` and timestamp stride 2. |

For the dense main protocol, `N=20`, `K=10`, and `B_h=2`, so:

```text
M = floor(K / B_h) = floor(10 / 2) = 5
N_max = ceil(N / M) = ceil(20 / 5) = 4
```

The intuition is that the subchannel pool can support about `M` active
coalition heads while leaving each head roughly `B_h` orthogonal raw-LiDAR
receive opportunities per perception round.

## Metrics

| Metric | Definition |
| --- | --- |
| AP@0.3/AP@0.5/AP@0.7 | Aggregate 3D detection average precision at IoU thresholds 0.3, 0.5, and 0.7. |
| Raw LiDAR Mbps | Scheduled raw point-cloud payload rate. |
| Box Mbps | Detection-box sharing rate for late/global/inter-cluster NMS. |
| Total Mbps | `Raw LiDAR Mbps + Box Mbps`. |
| GFLOPs/frame | Detector-side forward compute per scene frame. It includes Conv/Deconv/Linear/BatchNorm/ReLU hooks plus approximate PillarVFE point-feature FLOPs, and excludes voxelization/hash/scatter memory-index work, NMS, scheduling, CARLA, and control logic. |
| Avg source CAVs | Average number of CAVs that upload raw-LiDAR grids per evaluated scene frame. |
| Avg selected grids | Average number of uploaded grid blocks per evaluated scene frame. |

## Notation

Let `g` be a spatial grid, `i` a candidate sender CAV, `r` a receiver CAV, `S` a coalition, `x_{i,r,g} in {0,1}` indicate whether sender `i` uploads grid `g` to receiver `r`, and `z_r in {0,1}` indicate whether receiver `r` runs detector inference and contributes boxes to late fusion. The normalized grid quality is:

```text
q_i(g) = min(1, rho_i(g) / rho_th)
```

where `rho_i(g)` is the point density recorded by the LiDAR grid manager for CAV `i` at grid `g`. For a coalition:

```text
q_S(g) = max_{j in S} q_j(g)
```

## Scene-Level Objective

Before introducing SGCP's coalition and scheduling decomposition, the target collaborative perception state is defined only with senders, receivers, grid uploads, and detector participants. It does not assume clusters or cluster heads.

For a sender-receiver-grid action, SGCP uses two dynamic perception-gain terms computed from the receiver's current accumulated evidence in this frame. Let `A_r(g)` be the set of already admitted uploads to receiver `r` on grid `g`, and define:

```text
Q_r^A(g) = min(1, q_r(g) + sum_{j in A_r(g)} uploaded_quality_j(g))
```

The two terms are:

```text
C(i,g|r,A) = min(q_i(g), 1 - Q_r^A(g))
V(i,g|r,A) = min(q_i(g), 1 - Q_r^A(g)) * 1[Q_r^A(g) > 0]
```

`C` is the coverage-recovery value: sender `i` is useful when receiver `r` has weak accumulated evidence on grid `g`. `V` is the multi-view refinement value: sender `i` is useful when receiver `r` already has local or previously admitted context on grid `g`, so the uploaded raw LiDAR can strengthen early-fusion geometry until the grid reaches the density threshold. The residual form makes the score consistent with density-capped upload: a sender-grid action stops gaining value once the receiver grid is saturated.

The same residual evidence state is also used by the payload generator. Let
`P_cap(g)` be the maximum useful point count of one receiver grid:

```text
P_cap(g) = ceil(rho_th * area(g)).
```

For the 10 m x 10 m grids and dense headline `rho_th=2`, this gives
`P_cap=200` points/grid. If receiver `r` already has `P_r^A(g)` points after
its local cloud and previously admitted uploads, the residual quota is:

```text
R_r^A(g) = max(P_cap(g) - P_r^A(g), 0).
```

When a selected sender grid contains more than `R_r^A(g)` points, SGCP uploads
a deterministic random subset of exactly `R_r^A(g)` points and discards the
rest for communication accounting and replay. The sampling seed is derived
from the frame, sender, receiver, and grid id, so repeated runs are
reproducible while avoiding a hand-crafted point-order bias. This is the
dense-version point-cloud random truncation mechanism.

The receiver-side early-fusion proxy utility on grid `g` is:

```text
U_r(g; x) =
    q_r(g)
  + sum_{i != r} x_{i,r,g} uploaded_quality_i(g)
```

Vehicles with `z_r=0` do not run detector inference and do not contribute boxes to the final late-fusion state. Box-level late fusion is approximated by a region-wise max over all detector participants:

```text
Phi(x,z) = sum_{g in G} max_{r in V_all} z_r U_r(g; x)
```

The optimization target is:

```text
maximize_{x,z} Phi(x,z)
```

subject to communication and compute feasibility:

```text
sum_{i,r,g} x_{i,r,g} b_i(g) <= B
T(x) <= T_max
sum_i y_{i,r} <= K, for every receiver r
x_{i,r,g} = 0, if i = r or d(i,r) > R
subchannel, half-duplex, per-receiver link, per-link grid, and detector-call constraints
```

where `b_i(g)` is the raw-LiDAR payload size of sender `i`'s grid `g`, and `y_{i,r}=1` if `sum_g x_{i,r,g}>0`. Communication cost is handled as feasibility constraints rather than as a tunable `-lambda L` penalty. This avoids extra utility weights and matches the reported experiments, where performance is compared under fixed network, concurrency, payload, and deadline constraints.

The stability multiplier used in coalition formation is:

```text
beta_i(S) = |G_i^pred(t + Delta t) intersect G_S^req(t + Delta t)| / |G_i^pred(t + Delta t)|
```

where `Delta t=0.1 s` is the next perception frame. In code, non-empty-coalition stability is lower-bounded by `0.25` to avoid discarding viable members due to one noisy next-frame projection.

## SGCP Coalition Formation

Actual file:

```text
C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\clustering\potential_verified_cov_coalition_game.py
```

Related base file for center-nearest head election and next-frame stability prediction:

```text
C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\clustering\coalition_game.py
```

SGCP uses potential-verified coalition formation. The coalition stage does not
select concrete sender-grid uploads; it builds a stable local multi-view search
space for the later scheduler. The intended writing order is:

```text
receiver-oriented view value V
    -> symmetric pairwise confirmation value V_pair
    -> pairwise coalition weight W_ij
    -> partition potential Phi_C
```

In other words, `W_ij` should not be introduced as an isolated graph weight. It
is the symmetric coalition-level counterpart of the receiver-oriented view
refinement term used by SGCP. The scheduler later decides the exact directed
sender-to-head uploads; the coalition stage only asks whether two vehicles have
stable overlapping evidence that makes them good candidates for the same local
multi-view group.

We first define a one-way receiver-oriented view gain:

```text
V(i,g | j) = q_i(g) * 1[q_j(g) > 0]
```

This term means that sender `i` provides a useful extra view for vehicle `j` at
grid `g` only when `j` already has local sensing context there. It is a
coalition-stage abstraction of the scheduler's view-refinement idea: useful
raw-LiDAR collaboration should reinforce a region that already has receiver
context, rather than merely grouping vehicles that observe unrelated areas.
For coalition search, vehicle `i` ranks candidate coalitions using the
lightweight one-way view-overlap proxy:

```text
u_i(S) = beta_i(S) * sum_{g in G_i^sens intersect G_S^sens} q_i(g) * 1[q_S(g)>0]
```

where `q_S(g)=max_{j in S} q_j(g)` and `beta_i(S)` is the next-frame stability
multiplier. This proxy only ranks target coalitions and gives the moving CAV a
cheap local proposal rule.

The committed migration is then verified by an exact symmetric
mutual-confirmation potential. Symmetry is needed because coalition membership
is undirected: if vehicles `i` and `j` are placed in the same coalition, the
coalition should benefit both sensing directions. We therefore convert the
one-way view gain into a pairwise lower-bound mutual value:

```text
V_pair(i,j,g) = min(V(i,g|j), V(j,g|i)) = min(q_i(g), q_j(g))
```

The equality follows directly from the indicator form: if both vehicles observe
grid `g`, the mutual value is limited by the weaker normalized quality; if
either vehicle has no evidence there, both the mutual view value and the
minimum are zero. Thus `W_ij` is simply the grid-summed pairwise confirmation
derived from `V_pair`, not a separate heuristic:

```text
Phi_C(Pi) = sum_{S in Pi} phi_C(S)
phi_C(S) = sum_{i<j, i,j in S} W_ij
W_ij = sum_{g in G_i^sens intersect G_j^sens} min(q_i(g), q_j(g))
```

`W_ij` measures symmetric multi-view confirmation between two CAVs: a shared
grid contributes only when both vehicles observe it, and the contribution is
limited by the weaker normalized grid quality. This is consistent with the
dynamic scheduler: before exact sender-head uploads are known, coalition
formation uses pairwise overlap as a stable proxy for later view-refinement
opportunities; after clusters are formed, the scheduler uses the directed and
receiver-residual form `V(i,g|r,A)` to choose actual uploads.

A migration of vehicle `i` from source coalition `S_src` to target coalition
`S_tgt` affects only those two coalitions, so the exact local admission
increment is:

```text
Delta Phi_C =
    phi_C(S_src \ {i}) + phi_C(S_tgt union {i})
  - phi_C(S_src)       - phi_C(S_tgt)
```

The migration is committed only if it both improves the local proposal utility and satisfies `Delta Phi_C > 0`. Therefore every accepted migration has `J_i^C = Delta Phi_C > 0`; `Phi_C` increases monotonically; and the coalition process terminates because the partition space is finite.

Pseudo-code:

```text
Algorithm 1: SGCP Potential-Verified Coalition Formation
Input: CAV set V, target subchannels K, per-head receive budget B_h,
       improvement factor eta, next-frame horizon Delta t
Output: coalitions S_1,...,S_M and one head per coalition

1  N_max <- ceil(|V| / floor(K / B_h)).
2  Initialize one singleton coalition for each CAV.
3  For each coalition, elect the head closest to the member-position centroid.
4  repeat for at most 20 iterations:
5      updated <- false
6      for each CAV i in the replay/world order:
7          S_src <- coalition containing i
8          u_cur <- proposal utility of i in S_src after temporarily removing i
9          candidates <- empty
10         for each target coalition S_tgt != S_src:
11             if |S_tgt| >= N_max: continue
12             q_S(g) <- max_{j in S_tgt} q_j(g)
13             beta_i(S_tgt) <- next-frame stability between i and S_tgt
14             u_tgt <- beta_i(S_tgt) * sum_{g in G_i^sens intersect G_S^sens} q_i(g) * 1[q_S(g)>0]
15             Delta Phi_C <- phi_C(S_src\{i}) + phi_C(S_tgt union {i}) - phi_C(S_src) - phi_C(S_tgt)
16             if u_tgt > eta * u_cur and Delta Phi_C > 0:
17                 add (S_tgt, u_tgt, Delta Phi_C) to candidates
18         if candidates is not empty:
19             choose the candidate with the largest u_tgt, using Delta Phi_C as a tie-breaker
20             move i from S_src to the chosen S_tgt
21             remove empty coalitions and refresh affected coalition summaries
22             updated <- true
23 until updated is false
24 return coalitions and heads
```

A 41-frame equivalence check under the current protocol confirmed that the PV implementation preserves the reported coalition members, cluster heads, AP, raw-LiDAR Mbps, and selected-grid statistics. Therefore the existing clean-package experiment values remain valid while the paper-facing algorithm has strict potential-increasing admission and a convergence proof.

## SGCP Resource Scheduling

Actual file:

```text
C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\dynamic_cv_potential_game.py
```

The scheduler is a two-stage potential-game style best-response allocator over
sender-grid actions. Its core idea is to turn each coalition into a small
receiver-centered resource game: each cluster head has a limited number of
orthogonal receiving opportunities, and members compete to upload the raw-LiDAR
grids with the largest marginal perception gain.

The scheduling logic follows the same hierarchy as the SGCP perception goal:

```text
Stage 1: recover missing / weakly observed regions at each cluster head.
Stage 2: use remaining channels to reinforce regions where the head already
         has context, improving multi-view early-fusion quality.
```

This is why the first pass ranks links by dynamic coverage value `C`, while the
second pass ranks remaining feasible links by dynamic view-refinement value
`V`. Both values are residual: after every accepted upload, the receiver-side
evidence `Q_h^A(g)` is updated, so repeated uploads to the same receiver-grid
have diminishing return and eventually stop being useful once the grid reaches
the density threshold. The scheduler therefore does not merely select dense
grids; it selects grids that are dense at the sender and still useful at the
receiver.

The selected action is a link plus a set of grid blocks. SGCP scores the
candidate link by summing the selected grid gains:

```text
U_C(i, G_sel | h,A) = sum_{g in G_sel} C(i,g | h,A)
U_V(i, G_sel | h,A) = sum_{g in G_sel} V(i,g | h,A)
```

Candidate grids are:

```text
G_cand(i,h,A) = {g in refinement_candidates(i,h) : C(i,g | h,A) + V(i,g | h,A) > 0}
```

The coverage stage uses only `C`, and the target-quality stage uses only `V`.
The selected grid set is the top-scoring subset allowed by the per-link grid
capacity derived from the NS3-calibrated channel model.

The same residual state also controls the actual payload. If a selected sender
grid contains more points than the receiver's residual quota, SGCP uploads a
deterministic random subset of the useful residual points. This keeps the
scheduler's utility and the communication accounting consistent: grids that
are already saturated cannot keep increasing the score or the payload.
Randomness is seeded by frame/sender/receiver/grid identifiers, so the replay is
deterministic and reproducible.

Pseudo-code:

```text
Algorithm 2: SGCP Dynamic C->V Scheduler
Input: coalitions S_m with heads h_m, target subchannels K, head RB budget B_h
Output: raw-LiDAR upload links and selected grids

1  Initialize all subchannels as free and each head link count as 0.
2  Coverage stage:
3      for each cluster head h in increasing head id:
4          if no subchannel is free: break
5          for each member i in S_h \ {h}:
6              build G_cand(i,h,A) where C+V is positive
7              G_sel <- top grids in G_cand ranked by C(i,g|h,A)
8              score_i <- sum_{g in G_sel} C(i,g|h,A)
9          schedule the member with largest score_i if score_i > 0
10         assign the next free subchannel and record G_sel
10a        apply residual density cap and deterministic random point sampling
           for the actual uploaded points
11 Target-quality stage:
12     while a subchannel is free:
13         candidates <- empty
14         for each cluster head h:
15             if link_count(h) >= B_h: continue
16             exclude members already scheduled for h
17             for each remaining member i:
18                 build G_cand(i,h,A) where C+V is positive
19                 G_sel <- top grids in G_cand ranked by V(i,g|h,A)
20                 score_i <- sum_{g in G_sel} V(i,g|h,A)
21             add the best positive member for h to candidates
22         if candidates is empty: break
23         schedule the candidate with largest V score
24         assign the next free subchannel and record G_sel
24a        apply residual density cap and deterministic random point sampling
           for the actual uploaded points
25 return all scheduled links and grid selections
```

Cost is represented as feasibility constraints rather than as a tunable scalar in the final implementation: finite subchannels, per-head link budget, grid capacity per link, density cap per receiver grid, and optional raw-Mbps payload cap. This avoids adding extra utility weights that would require another parameter sweep.

## Inter-Cluster Box Aggregation

After cluster-head early fusion, SGCP applies NMS to the detection boxes from all cluster heads:

```text
Algorithm 3: Inter-Cluster Box Aggregation
Input: cluster-head predictions B_1,...,B_M
Output: scene-level prediction set B

1  Collect predicted boxes, scores, and labels from every cluster head.
2  Add their box payload to communication accounting.
3  Transform boxes to the shared world frame if needed.
4  Apply class-aware NMS / duplicate suppression.
5  Evaluate the aggregate prediction set against the scene ground truth.
```

This step recovers scene-level coverage while keeping detector compute low, because only cluster heads run detector inference after raw-LiDAR sharing.

## Source Paths for Reproduction

| Component | Path |
| --- | --- |
| SGCP clustering | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\clustering\potential_verified_cov_coalition_game.py` |
| SGCP scheduler | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\dynamic_cv_potential_game.py` |
| Base coalition game and stability | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\clustering\coalition_game.py` |
| Resource allocator factory | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\builder.py` |
| Offline replay/evaluation CLI | `C:\Workspace\OpenCDA\opencda\tools\offline_inference.py` |
| PCS baseline | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\pcs.py` |
| EdgeCooper V2V scheduler | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\edgecooper.py` |
| EdgeCooper Pmax admission/truncation | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\edgecooper_pmax.py` |
| PACP-LiDAR scheduler | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\pacp_lidar.py` |
| Selective-baseline orchestration/deadline admission | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\selective_baselines.py` |
| Selective-baseline shared helpers | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\selective_baseline_common.py` |
| SGCP diagnostic scheduler probes | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\resource_allocation\sgcp_scheduler_probes.py` |
| Paper-inspired clustering baselines | `C:\Workspace\OpenCDA\opencda\core\clustering\algorithms\clustering\paper_baselines.py` |
| Channel estimator | `C:\Workspace\OpenCDA\opencda\core\clustering\utils\channel_model.py` |

For convergence, complexity, distributed solving overhead, control-plane packet accounting, and NS3-calibrated realtime feasibility, see `10_realtime_feasibility.md`.
