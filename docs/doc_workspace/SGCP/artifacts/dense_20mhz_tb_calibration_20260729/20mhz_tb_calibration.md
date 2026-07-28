# Dense SGCP 20 MHz NS3 TB Calibration

Date: 2026-07-29.

## Purpose

The dense-LiDAR sweep originally considered 20 MHz and 40 MHz settings with 10 target subchannels. This note records the exact NS3 feasibility check for that channel configuration.

## Finding

Exact NS3 does **not** support `20 MHz / 10 target subchannels` with `slSubchannelSize=5 PRB` in the current NR sidelink resource-pool implementation.

The failed command used:

```text
--targetSubchannels=10
--slBandwidthIn100kHz=200
--slSubchannelSize=5
--slMcs=28
--slSymbolsPerSlot=12
--slBearerActivationGuardMs=1
--nrSlZeroTimeSendDelayMs=0
```

NS3 terminated with:

```text
Invalid subchannel size in RBs : 5
```

The resource-pool factory accepts legal subchannel sizes such as `10, 15, 20, 25, 50, 75, 100` PRBs. Therefore, the paper-facing exact NS3 settings should avoid claiming `20 MHz / 10ch / 5PRB` as a valid NR sidelink configuration.

## Legal 20 MHz Probe

The legal 20 MHz alternative is:

```text
--targetSubchannels=5
--slBandwidthIn100kHz=200
--slSubchannelSize=10
```

Direct bridge probe result:

| Setting | Target subchannels | PRB/subchannel | Manual add | Manual consume | Median TB size |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 MHz legal | 5 | 10 | 2 | 48 | 912 B |

The 40 MHz sanity probe under the same 10PRB/subchannel setting produced the same median per-subchannel TB size:

| Setting | Target subchannels | PRB/subchannel | Manual add | Manual consume | Median TB size |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 MHz sanity | 10 | 10 | 2 | 48 | 912 B |

Thus, the practical difference between the legal 20 MHz and 40 MHz configurations is the number of parallel 10PRB target subchannels: 5 vs. 10. The current dense paper-facing configuration remains `40 MHz / 10 target subchannels`; the existing dense 20MHz row should be treated as a provisional logical half-bandwidth diagnostic, not an exact NS3-calibrated paper row.

## Probe Notes

The direct probe waits until after sidelink bearer activation before injecting transfer requests. With `nrSlZeroTimeSendDelayMs=0`, sending at simulator time zero can schedule packets before bearer activation and produce no manual consume events.

Artifacts:

- `run_20mhz_tb_probe.py`
- `20mhz_5ch_10prb_direct_activation_tb_summary.json`
- `40mhz_10ch_10prb_direct_activation_tb_summary.json`
- `ns3_20mhz_stderr.log`
