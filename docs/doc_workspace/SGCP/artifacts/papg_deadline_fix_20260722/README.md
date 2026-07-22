# SGCP PAPG Deadline Fix and 60/100 ms Rerun

Purpose: audit why the restored SGCP-PAPG 60 ms and 100 ms runs were identical.

## Finding

The previous 60 ms run was not a true 60 ms scheduling-capacity run. The CLI
`--communication-deadline-ms` correctly built a `ChannelModel` with the requested
deadline, but `PotentialGame.calculate_max_grids_per_rb()` passed
`Params.T_ddl`, whose default remained `0.1`, into
`ChannelModel.max_grids_per_rb()`. This overrode the channel model deadline and
made PAPG use a 100 ms grid budget even when the CLI requested 60 ms.

Fix: `opencda/tools/offline_inference.py::apply_resource_overrides()` now also
sets both `resource_allocator.time_slot` and `resource_allocator.p.T_ddl` from
`channel_model.frame_deadline_s` before `set_clusters()` computes
`max_grids_per_rb`.

## Protocol

- Dataset: `D:\Data\Carla\2026_07_15_01_26_56`
- Frames: 41 (`000060` to `000140`)
- Detector: attentive-derived early checkpoint YAML
- Resource allocation: `perception_aware_potential_game`
- Clustering: `coalition_game`
- Receiver policy: `all-cluster-heads`
- Inter-cluster late fusion: enabled
- `N_max=4`, `rho_th=3`, `head_rb_budget=2`
- Channel estimator: `ns3`
- NS3 estimator: `40 MHz`, `10` channels, `tb_size=899 bytes`,
  `slot=0.5 ms`, `MCS=28`

## Corrected Results

| Deadline | AP@0.3 | AP@0.5 | AP@0.7 | raw Mbps | late-box Mbps | total Mbps | avg selected grids | estimated frame time mean / P95 / max |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 60 ms | 0.87 | 0.76 | 0.38 | 58.44 | 0.71 | 59.15 | 36.67 | 40.82 / 42.08 / 42.38 ms |
| 100 ms | 0.87 | 0.79 | 0.37 | 61.47 | 0.71 | 62.18 | 61.67 | 42.76 / 43.89 / 44.32 ms |

Interpretation: after the fix, the 60 ms and 100 ms traces differ. The 100 ms
run recovers the old result because the old "60 ms" trace had effectively used
the default 100 ms `Params.T_ddl` for grid admission.

## Real NS3 Delay

The corrected 100 ms trace matches `../papg_100ms_budget_20260722/`, so the
already executed real NS3 replay for frame `000060` remains valid:

- planned chunks: `80`
- application callbacks delivered: `80/80`
- RLC complete requests: `80/80`
- PHY failures: `0`
- callback delay mean / P95 / max: `26.51 / 53.00 / 55.00 ms`

This means the 100 ms AP run is feasible under the paper's practical
communication-stage target of staying below roughly 60 ms in the tested frame.
