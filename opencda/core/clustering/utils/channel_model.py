import math


class ChannelModel:
    """Shared channel estimator for schedulers and offline diagnostics."""

    def __init__(self, mode='logical', bandwidth_mhz=20.0, num_channels=10,
                 frame_deadline_s=0.1, ns3_tb_size_bytes=400,
                 ns3_slot_duration_ms=0.5, ns3_subchannel_prbs=10,
                 ns3_symbols_per_slot=9, ns3_mcs=20):
        if mode not in ('logical', 'ns3'):
            raise ValueError('channel model mode must be logical or ns3')
        if bandwidth_mhz <= 0:
            raise ValueError('bandwidth_mhz must be positive')
        if num_channels <= 0:
            raise ValueError('num_channels must be positive')
        self.mode = mode
        self.bandwidth_mhz = float(bandwidth_mhz)
        self.num_channels = int(num_channels)
        self.frame_deadline_s = float(frame_deadline_s)
        self.ns3_tb_size_bytes = int(ns3_tb_size_bytes)
        self.ns3_slot_duration_ms = float(ns3_slot_duration_ms)
        self.ns3_subchannel_prbs = int(ns3_subchannel_prbs)
        self.ns3_symbols_per_slot = int(ns3_symbols_per_slot)
        self.ns3_mcs = int(ns3_mcs)

    @property
    def bandwidth_bps(self):
        return self.bandwidth_mhz * (10 ** 6)

    def per_channel_bps(self):
        if self.mode == 'ns3':
            slot_s = max(self.ns3_slot_duration_ms / 1000.0, 1e-9)
            return float(self.ns3_tb_size_bytes) * 8.0 / slot_s
        return self.bandwidth_bps / float(max(self.num_channels, 1))

    def payload_time_ms(self, payload_bytes, subchannels=1):
        payload_bytes = max(float(payload_bytes or 0), 0.0)
        if payload_bytes <= 0:
            return 0.0
        channels = max(int(subchannels or 1), 1)
        return (
            payload_bytes * 8.0 /
            max(self.per_channel_bps() * float(channels), 1.0) *
            1000.0)

    def payload_budget_bytes(self, deadline_ms=None, subchannels=1):
        deadline_s = (
            self.frame_deadline_s if deadline_ms is None
            else max(float(deadline_ms), 0.0) / 1000.0)
        channels = max(int(subchannels or 1), 1)
        return int(
            self.per_channel_bps() * float(channels) * deadline_s / 8.0)

    def required_subchannels(self, payload_bytes, deadline_s=None):
        deadline_s = self.frame_deadline_s if deadline_s is None else deadline_s
        budget = max(self.per_channel_bps() * float(deadline_s) / 8.0, 1.0)
        required = int(math.ceil(max(float(payload_bytes or 0), 0.0) / budget))
        return max(1, min(self.num_channels, required))

    def max_grids_per_rb(self, grid_bits, deadline_s=None):
        deadline_s = self.frame_deadline_s if deadline_s is None else deadline_s
        return int(math.floor(
            self.per_channel_bps() * float(deadline_s) /
            max(float(grid_bits or 1), 1.0)))

    def to_metadata(self):
        return {
            'channel_estimator': self.mode,
            'bandwidth_mhz': self.bandwidth_mhz,
            'num_channels': self.num_channels,
            'ns3_tb_size_bytes': self.ns3_tb_size_bytes,
            'ns3_slot_duration_ms': self.ns3_slot_duration_ms,
            'ns3_subchannel_prbs': self.ns3_subchannel_prbs,
            'ns3_symbols_per_slot': self.ns3_symbols_per_slot,
            'ns3_mcs': self.ns3_mcs,
        }


def build_channel_model(mode='logical', bandwidth_mhz=None, num_channels=None,
                        frame_deadline_s=0.1, ns3_tb_size_bytes=400,
                        ns3_slot_duration_ms=0.5,
                        ns3_subchannel_prbs=10,
                        ns3_symbols_per_slot=9, ns3_mcs=20):
    return ChannelModel(
        mode=mode or 'logical',
        bandwidth_mhz=20.0 if bandwidth_mhz is None else bandwidth_mhz,
        num_channels=10 if num_channels is None else num_channels,
        frame_deadline_s=frame_deadline_s,
        ns3_tb_size_bytes=ns3_tb_size_bytes,
        ns3_slot_duration_ms=ns3_slot_duration_ms,
        ns3_subchannel_prbs=ns3_subchannel_prbs,
        ns3_symbols_per_slot=ns3_symbols_per_slot,
        ns3_mcs=ns3_mcs)
