# SGCP Main Table Candidate

更新时间：2026-07-17

本文档把当前可复现结果收束为论文主表候选。目标是避免三类混淆：把 FullPerception upper reference 当作公平主对比、用低 payload 的 Random/MWS 证明通信量降低、只给单一 SGCP 设置而不展示 AP/payload tradeoff。

当前表的主候选已从 coverage-aware grid post-processing 收束为 `perception_aware_potential_game`。PAPG 在同一 20MHz/10ch 约束下，以 62.54 Mbps 达到 `0.81/0.78/0.39`，高于当前强 selective baseline 的 AP@0.3/AP@0.5，并低于 full 20-CAV upper reference。后续仍需要把 Random/Greedy 改成强制使用统一带宽或统一 payload cap 的强 baseline，避免用低 payload 弱 baseline 支撑通信量主张。

## 口径

- 数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。
- 感知：OpenCOOD early-fusion checkpoint，SGCP inter-cluster late fusion evaluation path。
- 通信量：点云 upload payload；Mbps 按 41 帧、0.1 s 协作周期换算，即 `payload_bytes * 8 / 4.1 s / 1e6`。
- NS3：`spatial_diverse` 10ch (`rho_th=2/3`) 和 20ch 已完成 11 帧 request-level replay，均为 application/RLC complete。PAPG 当前完成 11 帧 upload-plan dry-run，真实 NS3 socket replay 待补。

## 推荐主表

| Method | Type | AP@0.3 | AP@0.5 | AP@0.7 | Payload | Mbps | NS3 Delivery | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No cooperation / Head-only | Lower reference | 0.26 | 0.22 | 0.09 | 0 | 0.00 | N/A | Cluster heads detect alone, then late-fuse |
| FullPerception centralized | Upper reference | 0.85 | 0.83 | 0.48 | 60,838,528 | 118.71 | Not constrained | Virtual full 20-CAV point-cloud sharing; current dump has no RSU |
| Full-cluster intra-cluster upload | Upper reference | 0.82 | 0.79 | 0.42 | 44,850,528 | 87.51 | Not tested | Each cluster head receives all cluster-member point clouds |
| Selective communication-aware, 2m/87g | Fair V2V baseline | 0.78 | 0.75 | 0.40 | 30,222,256 | 58.97 | 11f diagnostic only | Strong low-budget selective baseline |
| Selective density, 3m/117g | Fair V2V baseline | 0.80 | 0.76 | 0.40 | 37,710,864 | 73.58 | Not constrained | Payload-matched high-budget selective baseline |
| SGCP original utility, 10ch | Previous SGCP | 0.77 | 0.73 | 0.35 | 26,916,208 | 52.52 | 110/110 complete | Original saturated-density utility |
| SGCP coverage-aware, 10ch, `rho_th=2` | Proposed low-budget | 0.79 | 0.75 | 0.37 | 28,743,280 | 56.08 | 110/110 complete | Spatial-diverse grid selection |
| SGCP PAPG, 10ch, `rho_th=3`, `B_h=2` | Proposed main | 0.81 | 0.78 | 0.39 | 32,049,872 | 62.54 | 11f dry-run | Perception-aware two-layer potential scheduling: coverage layer + target layer |
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
| FullPerception-RSU | N/A | Current dump is RSU-free; needs a new RSU-enabled export. |

## Recommended Paper Story

1. FullPerception centralized and full-cluster upload define upper references, not fair decentralized baselines.
2. Random/MWS are w/o-PPS ablations; their low payload means they cannot support a communication-reduction claim.
3. The fair baseline should be payload-matched selective sharing. In high-budget mode, selective density reaches `0.80/0.76/0.40` at 73.58 Mbps.
4. SGCP PAPG 10ch `rho_th=3,B_h=2` is the current main point: `0.81/0.78/0.39` at 62.54 Mbps. It beats the strong high-budget selective baseline on AP@0.3/AP@0.5 while using about 15.0% less payload, and it remains far below centralized FullPerception's 118.71 Mbps.
5. SGCP coverage-aware 20ch reaches `0.80/0.76/0.41` at 73.98 Mbps and can be used as a network-resource sensitivity row: higher channel budget improves AP@0.7 but does not beat PAPG on AP@0.3/AP@0.5.
6. Target-aware PG `0.80/0.76/0.39` at 60.62 Mbps is now best treated as an ablation: it proves that moving target-aware utility into the allocator helps, while PAPG adds the missing coverage layer.
7. `B_h=2` coverage-aware sensitivity remains useful for explaining localization-quality tradeoff (`0.76/0.72/0.42` at 54.56 Mbps), but PAPG recovers AP@0.3/AP@0.5 without relying on that tradeoff.
8. `--max-upload-points-per-source 3000` confirms payload is tunable down to 38.07 Mbps, but current deterministic sampling loses AP. It is evidence for a communication knob, not a final algorithm row.

## Next Decisions

- Use PAPG `rho_th=3,B_h=2` as the current 10ch main row unless later forced-budget baselines overturn the table. It improves target-aware PG from `0.80/0.76/0.39` to `0.81/0.78/0.39`, with payload rising from 60.62 Mbps to 62.54 Mbps.
- Decide whether the main table shows both SGCP 10ch and 20ch, or places 20ch in network-resource sensitivity.
- `rho_th=3` NS3 replay is now complete: 110/110 application callbacks and 110/110 RLC-complete requests over the first 11 frames, with no PHY decode failures.
- For PAPG, supplement the current dry-run with true NS3 socket replay before final paper submission. The algorithm uses 410 scheduled links over 41 frames and does not bypass subchannel budget.
- Before declaring SGCP "lowest Mbps", rerun Random/Greedy under forced bandwidth utilization or identical point cap. The existing Random/MWS rows use only about 18-19 Mbps and therefore are weak ablations, not fair main baselines.
