# SGCP Main Table Candidate

更新时间：2026-07-18

本文档把当前可复现结果收束为论文主表候选。目标是避免三类混淆：把 FullPerception upper reference 当作公平主对比、用低 payload 的 Random/MWS 证明通信量降低、只给单一 SGCP 设置而不展示 AP/payload tradeoff。

当前表的主候选已从 coverage-aware grid post-processing 收束为 `perception_aware_potential_game`。PAPG 在同一 20MHz/10ch 约束下，以 62.54 Mbps 达到 `0.81/0.78/0.39`，高于当前强 selective baseline 的 AP@0.3/AP@0.5，并低于 full 20-CAV upper reference。后续仍需要把 Random/Greedy 改成强制使用统一带宽或统一 payload cap 的强 baseline，避免用低 payload 弱 baseline 支撑通信量主张。

## 口径

- 数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。
- 感知：OpenCOOD early-fusion checkpoint，SGCP inter-cluster late fusion evaluation path。
- 通信量：点云 upload payload；Mbps 按 41 帧、0.1 s 协作周期换算，即 `payload_bytes * 8 / 4.1 s / 1e6`。
- NS3：`spatial_diverse` 10ch (`rho_th=2/3`) 和 20ch 已完成 11 帧 request-level replay，均为 application/RLC complete。PAPG 10ch `rho_th=3,B_h=2` 已完成 11 帧真实 socket replay，110/110 application callback 与 RLC request complete，PHY failures 为 0。

## 推荐主表

| Method | Type | AP@0.3 | AP@0.5 | AP@0.7 | Payload | Mbps | NS3 Delivery | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No cooperation / Head-only | Lower reference | 0.26 | 0.22 | 0.09 | 0 | 0.00 | N/A | Cluster heads detect alone, then late-fuse |
| Full 20-CAV early fusion | Upper reference | 0.85 | 0.83 | 0.48 | 60,838,528 | 118.71 | Not constrained | Full point-cloud sharing AP upper bound |
| FullPerception PCS (`pcs.py`), legacy eval | Built-in FullPerception | 0.44 | 0.39 | 0.17 | 12,684,880 | 24.75 | Superseded | Pre-repair compatibility row with simplified `c(q)=1` |
| FullPerception PCS (`pcs.py`), repaired scheduled receivers | Built-in FullPerception | 0.33 | 0.29 | 0.14 | 8,100,112 | 15.80 | Dry-run 104 scheduled requests | Protocol-correct `c(q)`/`sc_num` path; still under-schedules and needs PCS/RSU calibration |
| FullPerception-RSU proxy | RSU/edge-assisted | 0.84 | 0.80 | 0.46 | 56,224,736 | 109.71 | Dry-run only | Virtual RSU/global candidate pool; not V2V-only fair baseline |
| EdgeCooper-style proxy | RSU/edge-assisted | 0.75 | 0.70 | 0.32 | 56,134,048 | 109.53 | Not replayed | Virtual edge/global candidate pool with blind-spot complementarity; preliminary proxy, not strict MCF/coloring reproduction |
| EdgeCooper-global network-aware proxy | RSU/edge-assisted | 0.81 | 0.77 | 0.42 | 38,223,408 | 74.58 | 73/110 complete | Global blind-spot assignment proxy with sender-load balancing and 35 m V2V feasibility gate; high offline AP, but incomplete NS3 deadline delivery |
| EdgeCooper-global-HD proxy | RSU/edge-assisted | 0.81 | 0.78 | 0.42 | 33,519,040 | 65.40 | 110/110 complete | Adds half-duplex exclusion so cluster-head receivers are not selected as senders in the same slot; strongest edge-assisted proxy |
| FullPerception-Decentralized proxy | Fair V2V baseline | 0.80 | 0.76 | 0.41 | 38,920,592 | 75.94 | 110/110 complete | Cluster-local V2V candidate pool, 3 members/head, 117 grid budget |
| Full-cluster intra-cluster upload | Upper reference | 0.82 | 0.79 | 0.42 | 44,850,528 | 87.51 | Not tested | Each cluster head receives all cluster-member point clouds |
| Selective random, 3m/117g | Fair V2V baseline | 0.77 | 0.73 | 0.38 | 31,613,424 | 61.68 | Not replayed | Forced-budget random baseline; same coalition path, 3 members/head, 117 grid budget |
| Selective communication-aware, 2m/87g | Fair V2V baseline | 0.78 | 0.75 | 0.40 | 30,222,256 | 58.97 | 11f diagnostic only | Strong low-budget selective baseline |
| Selective density, 3m/117g | Fair V2V baseline | 0.80 | 0.76 | 0.40 | 37,710,864 | 73.58 | Not constrained | Payload-matched high-budget selective baseline |
| SGCP original utility, 10ch | Previous SGCP | 0.77 | 0.73 | 0.35 | 26,916,208 | 52.52 | 110/110 complete | Original saturated-density utility |
| SGCP coverage-aware, 10ch, `rho_th=2` | Proposed low-budget | 0.79 | 0.75 | 0.37 | 28,743,280 | 56.08 | 110/110 complete | Spatial-diverse grid selection |
| SGCP PAPG, 10ch, `rho_th=3`, `B_h=2` | Proposed main | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 110/110 complete | Perception-aware two-layer potential scheduling: coverage layer + target layer |
| SGCP PAPG, 10ch, `rho_th=3`, `B_h=3` | Sensitivity | 0.80 | 0.78 | 0.40 | 32,051,792 | 62.54 | Not replayed | Negative high-IoU probe: per-head RB relaxation lowers source diversity and does not catch EdgeCooper-HD |
| SGCP target-aware PG, 10ch, `rho_th=3` | Proposed low-budget tuned | 0.80 | 0.76 | 0.39 | 31,069,968 | 60.62 | 11f request plan dry-run | New scheduler: original potential-game sender/RB stage + target-aware grid-action refinement |
| SGCP coverage-aware, 10ch, `rho_th=3` | Previous grid-selection probe | 0.79 | 0.76 | 0.38 | 29,405,296 | 57.38 | 110/110 complete | Kept as ablation for the former post-processing style selection |
| SGCP coverage-aware, 10ch, `rho_th=3`, point cap 3000 | Payload sensitivity | 0.74 | 0.70 | 0.33 | 19,510,848 | 38.07 | Not replayed | Effective payload knob, but AP drops; not recommended as main row yet |
| SGCP coverage-aware, 10ch, `rho_th=3`, `B_h=2` | High-IoU sensitivity | 0.76 | 0.72 | 0.42 | 27,962,864 | 54.56 | 110/110 complete | Matches full-cluster AP@0.7 at lower payload, but AP@0.3/0.5 drops; use as sensitivity unless recall is recovered |
| SGCP coverage-aware, 20ch, `rho_th=2` | Proposed high-budget | 0.80 | 0.76 | 0.41 | 37,912,544 | 73.98 | 154/154 complete | Near full-cluster AP@0.7 with 84.5% full-cluster payload |

## 不建议放入主公平表的行

| Method | Current Result | Reason |
| --- | --- | --- |
| Random scheduler | 0.44 / 0.39 / 0.17, 9,725,376 bytes | Payload only 18.98 Mbps; does not fully use the resource budget. Keep as w/o PPS ablation. |
| MWS scheduler | 0.31 / 0.26 / 0.11, 9,910,032 bytes | Payload only 19.34 Mbps and implementation definition needs review. Keep as diagnostic ablation. |
| Full 20-CAV late checkpoint | 0.91 / 0.85 / 0.51 | Different late-fusion checkpoint; use only as upper reference. |
| True RSU-sensor FullPerception | N/A | Current dump is RSU-free; needs a new RSU-enabled export. Existing `fullperception_rsu` is a virtual-RSU proxy, not a real RSU-sensor replay. |

## Recommended Paper Story

1. Full 20-CAV early fusion and full-cluster upload define upper references, not fair decentralized baselines.
2. The repository's built-in FullPerception implementation is `pcs.py`; the repaired `fullperception_pcs` path now uses payload-based `c(q)` and real `sc_num`, but the result is still weak because the current PCS/receiver fusion protocol under-schedules. FullPerception-RSU and EdgeCooper-style proxies are additional RSU/edge-assisted diagnostics because they use a global/edge candidate pool; FullPerception-Decentralized proxy is the fair V2V-only counterpart.
3. Random/MWS are w/o-PPS ablations; their low payload means they cannot support a communication-reduction claim.
4. The fair baseline should be payload-matched selective sharing. In high-budget mode, selective density reaches `0.80/0.76/0.40` at 73.58 Mbps, and FullPerception-Decentralized reaches `0.80/0.76/0.41` at 75.94 Mbps.
5. SGCP PAPG 10ch `rho_th=3,B_h=2` is the current main point: `0.81/0.78/0.39` at 62.54 Mbps. It beats the forced-budget random baseline by `+0.04/+0.05/+0.01` AP at nearly the same traffic, and beats the strong high-budget selective/FullPerception-Decentralized baselines on AP@0.3/AP@0.5 while using lower payload.
6. EdgeCooper-global-HD reaches `0.81/0.78/0.42` at 65.40 Mbps and 110/110 NS3 delivery after adding half-duplex sender/receiver exclusion. This is now a strong edge-assisted baseline, not a V2V-only fair baseline; if placed in the same table, PAPG no longer has the best AP@0.7.
7. SGCP coverage-aware 20ch reaches `0.80/0.76/0.41` at 73.98 Mbps and can be used as a network-resource sensitivity row: higher channel budget improves AP@0.7 but does not beat PAPG on AP@0.3/AP@0.5.
8. Target-aware PG `0.80/0.76/0.39` at 60.62 Mbps is now best treated as an ablation: it proves that moving target-aware utility into the allocator helps, while PAPG adds the missing coverage layer.
9. `B_h=2` coverage-aware sensitivity remains useful for explaining localization-quality tradeoff (`0.76/0.72/0.42` at 54.56 Mbps), but PAPG recovers AP@0.3/AP@0.5 without relying on that tradeoff.
10. `--max-upload-points-per-source 3000` confirms payload is tunable down to 38.07 Mbps, but current deterministic sampling loses AP. It is evidence for a communication knob, not a final algorithm row.
11. PAPG `B_h=3` is a negative sensitivity row: it keeps roughly the same payload as PAPG `B_h=2` and raises AP@0.7 only to 0.40 while lowering AP@0.3. The next high-IoU improvement should protect high-quality sources and target-grid coverage rather than merely relaxing per-head RB count.

## Next Decisions

- Use PAPG `rho_th=3,B_h=2` as the current 10ch main row unless later forced-budget baselines overturn the table. It improves target-aware PG from `0.80/0.76/0.39` to `0.81/0.78/0.39`, with payload rising from 60.62 Mbps to 62.54 Mbps.
- Decide whether the main table shows both SGCP 10ch and 20ch, or places 20ch in network-resource sensitivity.
- `rho_th=3` NS3 replay is now complete: 110/110 application callbacks and 110/110 RLC-complete requests over the first 11 frames, with no PHY decode failures.
- PAPG 11-frame true NS3 socket replay is complete: 110/110 application callbacks, 110/110 RLC-complete requests, 0 PHY decode failures. The algorithm uses 410 scheduled links over 41 frames in the perception run and does not bypass subchannel budget.
- Forced-budget random has been rerun under the same coalition path with 3 uploaded members/head and 117 grid budget: `0.77/0.73/0.38`, 31,613,424 bytes / 61.68 Mbps. Existing RandomRA/MWS rows still remain weak scheduler ablations because they use only about 18-19 Mbps.
- Built-in FullPerception PCS is now identified, aliased as `fullperception_pcs`, and partially repaired: `c(q)`, `sc_num`, scheduled links and scheduled-receiver evaluation are wired through. The repaired 41-frame result is `0.33/0.29/0.14` at 15.80 Mbps with 104 dry-run NS3 requests, so PCS still needs algorithm/receiver calibration before it can stand as the main FullPerception baseline. FullPerception-Decentralized now has symmetric NS3 evidence with PAPG and forced random: 110/110 application/RLC complete and 0 PHY failures. EdgeCooper-global-HD also has 110/110 NS3 delivery and better AP@0.7, so the next SGCP decision is whether to separate RSU/edge-assisted baselines from V2V-only baselines in the paper table, or improve PAPG high-IoU recall.
