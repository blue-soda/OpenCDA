# SGCP Qualitative Case Study Draft

Date: 2026-07-19

Purpose: provide a compact appendix-ready qualitative study using existing diagnostic outputs. This is a text/table draft rather than a rendered visualization; if used in the paper, the next step is to generate BEV overlays for the same frames.

Primary sources:

- `docs/doc_workspace/SGCP/failure_diagnostics.md`
- `docs/doc_workspace/SGCP/target_grid_case_study.md`
- `docs/doc_workspace/SGCP/artifacts/failure_diag_target_aware_pg_10ch_rho3_41f/gt_objects.csv`

## Case Selection Rationale

The cases below are chosen from persistent misses where full 20-CAV early fusion detects the object but the constrained SGCP variant misses it. They cover three distinct failure modes:

- relevant target grid exists but the best sender is not scheduled;
- target grid is covered only by a less relevant cluster head;
- target grid has many points, but object-level support is still insufficient for the detector.

These cases support the paper narrative that SGCP needs both hierarchical clustering and perception-aware point-cloud scheduling, while also documenting why the current PAPG variant still has high-IoU headroom.

## Case 1: Best-View Sender Not Scheduled

| Field | Value |
| --- | --- |
| Frame / object | `000068` / object `438` |
| GT grid | `3_0` |
| Nearest CAV / nearest head | CAV 12 / head 4 |
| Cluster | `4;9;12` |
| Strong source evidence | CAV 12 has 424 points in grid `3_0`, candidate rank 1 |
| Scheduled sender in old target-aware PG | CAV 9 |
| Scheduled target-grid points from old sender | 0 in grid `3_0` |
| Final detection status | Full 20-CAV reference detects; constrained method misses |

Interpretation: a density/coverage-driven sender score can prefer a vehicle that covers many ordinary grids but contributes no points to the missed target grid. This motivates PAPG's target layer: sender/grid utility should include marginal object-support and not only aggregate grid coverage.

Paper use: show as a resource-scheduling failure case. A BEV overlay should mark CAV 12 as the missed best-view sender, CAV 9 as the scheduled sender, and grid `3_0` as the target grid.

## Case 2: Target Grid Covered by the Wrong Receiver Side

| Field | Value |
| --- | --- |
| Frame / object | `000066` / object `401` |
| GT grid | `2_0` |
| Nearest CAV / nearest head | CAV 12 / head 12 |
| Cluster | `4;7;12` |
| Strong source evidence | CAV 4 has 891 points in grid `2_0`, candidate rank 4 |
| Scheduled sender in old target-aware PG | CAV 7 |
| Scheduled target-grid points from old sender | 7 in grid `2_0` |
| Final detection status | Full 20-CAV reference detects; constrained method misses |

Interpretation: the object is not invisible and bandwidth alone is not the whole cause. The useful view exists inside the local cluster, but the scheduled sender supplies only sparse target-grid evidence. This supports the claim that the scheduler must reason about sender-view quality and receiver-side need under the same subchannel budget.

Paper use: show as a cluster-local PPS example. It is useful for explaining why "same cluster, some upload" is not enough; the selected sender and selected grids both matter.

## Case 3: Dense Grid But Weak Object-Level Support

| Field | Value |
| --- | --- |
| Frame / object | `000062` / object `337` |
| GT grid | `0_-3` |
| Nearest CAV / nearest head | CAV 1 / head 1 |
| Cluster | `1;2;8;11` |
| Strong source evidence | Head CAV 1 has 1,453 points in grid `0_-3`; CAV 8 has 138 peer-view points |
| Scheduled sender/grid issue | CAV 2/CAV 11 are scheduled but do not provide a strong peer view for `0_-3` |
| Final detection status | Full 20-CAV reference detects; constrained method misses |

Interpretation: grid-level density is too coarse for some near-body or blind-spot objects. The grid can be dense while still lacking the object shape/context needed by the detector. This is the main reason the current PAPG result is AP@0.7-limited, and it justifies the checkpoint improvement task rather than adding another ad hoc scheduler patch.

Paper use: show as the boundary case. It protects the narrative by explaining why two-layer fusion improves coverage while high-IoU localization still depends on stronger early-fusion detection and object-level point support.

## Figure Draft Recommendation

If this becomes a paper figure, use a three-panel BEV layout:

| Panel | Visual Elements | Caption Point |
| --- | --- | --- |
| A | GT box, target grid, scheduled sender, best-view sender, cluster head | Best-view sender can be missed by coverage-only scheduling. |
| B | Same-cluster candidate points and chosen upload | PPS must select the right sender/grid, not merely any cluster upload. |
| C | Dense grid and missing final box | Dense grid coverage is not equivalent to object-level detector support. |

The figure should not introduce a new metric. It should simply visualize aggregate-AP failure modes that are already quantified in `failure_diagnostics.md`.
