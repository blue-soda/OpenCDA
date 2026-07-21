# PCS / EdgeCooper Protocol Alignment

Date: 2026-07-21

This note fixes the writing and experiment boundary for FullPerception-PCS,
EdgeCooper V2V adaptation, and SGCP.

## Forward PCS Parameter Default

All future PCS experiments in this repository should use the raw-LiDAR
adaptation parameters:

- `blind_spot_min_division = 4`
- `blind_spot_adjacency_radius = 4`
- `blind_spot_min_grids = 128`

CLI equivalent:

```powershell
--pcs-blind-spot-min-division 4 --pcs-blind-spot-radius 4 --pcs-min-spot-grids 128
```

The code default in `opencda/core/clustering/algorithms/resource_allocation/pcs.py`
has been changed accordingly. Existing PCS tables generated before this change
must be treated as archived diagnostics unless they explicitly state
`div4/radius4/min128`.

## FullPerception-PCS Objective

FullPerception targets network-level collaborative perception under constrained
communication resources. Its PCS scheduler decides sensing information sharing
and communication resource allocation so that critical blind spots are
eliminated as much as possible. Therefore:

- The primary target is perception improvement under communication constraints,
  not pure Mbps minimization.
- Communication reduction is a constraint and an efficiency outcome.
- The original data unit is semantic information / neural-network features over
  blind spots, not necessarily raw LiDAR grids.
- The original scheduling model is link/resource scheduling over time/frequency
  resources; when adapted to our raw-LiDAR replay, a one-shot 100 ms per-frame
  run is stricter than a possible multi-slot PCS execution within the same
  perception frame.

For Table 1, PCS should be described as a paper-faithful PCS scheduling
adaptation to raw-LiDAR V2V replay, with all CAVs evaluated as singleton
receivers and unscheduled receivers falling back to local-only detection.

## EdgeCooper Objective

EdgeCooper is edge-assisted cooperative LiDAR perception. Its scheduler selects
complementary raw sensor data, relay/channel/packet decisions, and sends data
toward an edge server that constructs a holistic view and broadcasts detection
results. Therefore:

- The primary target is edge-level perception quality under network constraints.
- Bandwidth savings come from selecting complementary data and avoiding full
  all-to-all raw dissemination.
- Original EdgeCooper is not equivalent to asking every CAV receiver to run a
  separate V2V request plan. That V2V adaptation can duplicate the same sender
  payload across many singleton receivers and may overstate generated demand.
- Like PCS, EdgeCooper contains scheduling/packetization assumptions that can
  span multiple resource decisions within a perception frame.

For Table 1, EdgeCooper V2V should be labelled as a protocol adaptation or
V2V+ proxy unless an actual edge-server upload-once implementation is used.

## SGCP Objective

SGCP's current experiment protocol is deliberately different:

- one CARLA / offline replay frame maps to one cooperative perception cycle,
- the cycle deadline is 100 ms,
- intra-cluster raw-LiDAR scheduling is expected to complete within that cycle,
- inter-cluster box-level late fusion is charged separately as detection-box
  communication.

This makes SGCP a per-frame, deadline-aware V2V protocol. Scheduler-comparison
tables are fair only when every row shares this SGCP scaffold. Protocol-native
baseline tables must explicitly state when PCS or EdgeCooper are being adapted
into this stricter one-cycle raw-LiDAR setting.

