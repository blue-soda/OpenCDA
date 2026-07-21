# NS3 Channel Model Alignment

Last updated: 2026-07-21

## Problem

The scheduler baselines previously used mixed channel assumptions:

- PCS used `bandwidth_all / lambda_subchannels * frame_deadline`.
- SGCP/PAPG used the potential-game channel utilities and a grid budget derived from logical bandwidth.
- EdgeCooper and other selective baselines mostly used fixed member/grid budgets, with communication time estimated separately at evaluation time.
- NS3 actually admits payload by NR sidelink transport blocks. In the 2026-07-21 single-frame timing logs, each manual grant delivered about `400 bytes` per subchannel opportunity under the default `MCS=20`, `symbolsPerSlot=9`, `slSubchannelSize=10 PRBs`, numerology 1 setting.

This explains why a run that reports about `39.6 MHz` sidelink bandwidth can still exceed a 100 ms application deadline for 600-800 KB raw-LiDAR bursts: the raw PHY/BWP width is not equivalent to ideal application goodput. RLC segmentation, CAM chunking, scheduler grant cadence, PSCCH/PSSCH overhead, MCS, and per-subchannel resource selection determine the observable service rate.

## Code Changes

OpenCDA now has a shared estimator:

- `opencda/core/clustering/utils/channel_model.py`
- `ChannelModel(mode='logical' | 'ns3', ...)`

The estimator is wired into:

- PCS required-subchannel calculation.
- SGCP/PAPG grid budget and NS3-mode data-rate estimate.
- Offline `frame_comm_time_ms` calculation.
- EdgeCooper/selective baselines' communication-time estimate.
- Optional deadline-aware trimming for selective baselines via `--selective-frame-deadline-ms`.

Relevant `offline_inference.py` arguments:

```powershell
conda run -n opencda python opencda\tools\offline_inference.py `
  ... `
  --channel-estimator ns3 `
  --bandwidth-mhz 20 `
  --num-channels 10 `
  --ns3-tb-size-bytes 400 `
  --ns3-slot-duration-ms 0.5 `
  --ns3-subchannel-prbs 10 `
  --ns3-symbols-per-slot 9 `
  --ns3-mcs 20
```

Default behavior remains `--channel-estimator logical` so earlier reproduced tables are not silently rewritten.

## NS3 Capacity Controls

The NS3 co-simulation now exposes additional CLI parameters:

```bash
./ns3 run 'scratch/vanet/main.cc \
  --targetSubchannels=10 \
  --slSubchannelSize=10 \
  --slBandwidthIn100kHz=396 \
  --slMcs=20 \
  --slSymbolsPerSlot=9 \
  --slPscchRbs=10 \
  --slMaxNumPerReserve=1 \
  --slMaxTxTransNumPssch=1 \
  --slRriMs=5 \
  --slProbResourceKeep=0.8'
```

To run an explicit high-capacity diagnostic, vary only these NS3 parameters and mirror the assumed service rate in OpenCDA with `--channel-estimator ns3`:

- Increase `--slMcs` after confirming PHY decode remains reliable.
- Increase `--slSymbolsPerSlot` toward 12-14 if the sidelink pool leaves enough control resources.
- Reduce `--slPscchRbs` if control allocation is over-reserved for the scenario.
- Reduce `--slRriMs` to allow more frequent resource opportunities.
- Increase `--slMaxNumPerReserve` only if the resulting resource reuse remains conflict-free.
- For a strict 20 MHz / 10 exposed-subchannel setup, test `--slBandwidthIn100kHz=200 --slSubchannelSize=5` rather than `10 PRBs`, because 10 PRBs per subchannel requires about 39.6 MHz to expose 10 logical subchannels plus reserve.

## Interpretation

The paper-facing claim should not say "40 MHz cannot carry 60 Mbps" as a pure Shannon-rate statement. The accurate statement is:

> Under the current NR sidelink Mode-2 configuration, application-layer raw-LiDAR bursts are limited by TB size and grant cadence; 60 Mbps raw payload can exceed a 100 ms cooperative perception cycle unless the scheduler payload, chunking, and NS3 sidelink parameters are jointly calibrated.

Next experiment should run three aligned modes:

- `logical`: old paper-table estimator for reproducibility.
- `ns3-default`: `tb_size=400B`, `slot=0.5ms`, matching 2026-07-21 logs.
- `ns3-high-capacity`: explicit NS3 settings with higher MCS/symbols or smaller subchannel PRBs, followed by real chunked single-round timing.
