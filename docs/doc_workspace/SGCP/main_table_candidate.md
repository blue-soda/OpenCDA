# SGCP Main Table Candidate

更新时间：2026-07-17

本文档把当前可复现结果收束为论文主表候选。目标是避免三类混淆：把 FullPerception upper reference 当作公平主对比、用低 payload 的 Random/MWS 证明通信量降低、只给单一 SGCP 设置而不展示 AP/payload tradeoff。

当前表仍不是最终满意版本：SGCP 已能在 payload-matched selective baseline 附近取得更好的 AP@0.7，但尚未同时满足“对 Random/Greedy/FullPerception 均 AP 最高且 Mbps 最少”。后续主表需要把 Random/Greedy 改成强制使用统一带宽或统一 payload cap 的强 baseline。

## 口径

- 数据：`D:\Data\Carla\2026_07_15_01_26_56`，41 帧，20 CAV。
- 感知：OpenCOOD early-fusion checkpoint，SGCP inter-cluster late fusion evaluation path。
- 通信量：点云 upload payload；Mbps 按 41 帧、0.1 s 协作周期换算，即 `payload_bytes * 8 / 4.1 s / 1e6`。
- NS3：`spatial_diverse` 10ch (`rho_th=2/3`) 和 20ch 已完成 11 帧 request-level replay，均为 application/RLC complete。

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
4. SGCP coverage-aware 20ch reaches `0.80/0.76/0.41` at 73.98 Mbps, slightly improving AP@0.7 over the payload-matched selective baseline while retaining PPS feasibility and verified NS3 delivery.
5. SGCP target-aware PG 10ch `rho_th=3` is the low-budget main point: `0.80/0.76/0.39` at 60.62 Mbps, far below centralized FullPerception's 118.71 Mbps. It keeps the potential-game sender/subchannel allocation as the first stage and moves coverage-aware grid refinement into the scheduler itself.
6. `B_h=2` is a useful high-IoU sensitivity: `0.76/0.72/0.42` at 54.56 Mbps, with 110/110 NS3 request-level delivery over the first 11 frames. It should be discussed as a localization-quality tradeoff unless AP@0.3/0.5 recall is recovered.
7. `--max-upload-points-per-source 3000` confirms payload is tunable down to 38.07 Mbps, but current deterministic sampling loses AP. It is evidence for a communication knob, not a final algorithm row.

## Next Decisions

- Use target-aware PG `rho_th=3` as the current 10ch main row unless later forced-budget baselines overturn the table. It improves the former `spatial_diverse,rho_th=3` row from `0.79/0.76/0.38` to `0.80/0.76/0.39`, at the cost of payload rising from 57.38 Mbps to 60.62 Mbps.
- Decide whether the main table shows both SGCP 10ch and 20ch, or places 20ch in network-resource sensitivity.
- `rho_th=3` NS3 replay is now complete: 110/110 application callbacks and 110/110 RLC-complete requests over the first 11 frames, with no PHY decode failures.
- If using `B_h=2` in the main table, add a short note that it improves AP@0.7 but reduces AP@0.3/0.5; the 11-frame NS3 replay is complete, but the AP tradeoff still needs a paper-story decision.
- Before declaring SGCP "lowest Mbps", rerun Random/Greedy under forced bandwidth utilization or identical point cap. The existing Random/MWS rows use only about 18-19 Mbps and therefore are weak ablations, not fair main baselines.
