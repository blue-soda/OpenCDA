"""Communication statistics tracking."""

from collections import defaultdict


class CommunicationStats:
    """Tracks communication metrics and statistics."""

    def __init__(self):
        self.current_slot = {
            'try_volume': 0.0,
            'total_volume': 0.0,
            'intra_cluster': {'upload': 0.0, 'download': 0.0},
            'inter_cluster': 0.0,
            'control_overhead': 0.0,
            'collisions': 0,
            't_latency': [],
            'p_latency': [],
            'utilization': 0.0
        }
        self.history = []

    def record_transmission(self, volume, comm_type="try"):
        """Record transmission volume."""
        if comm_type == "try":
            self.current_slot['try_volume'] += volume
        else:
            self.current_slot['total_volume'] += volume

    def end_slot(self):
        """Finalize current slot and save to history."""
        self.history.append(self.current_slot.copy())
        self.current_slot = {
            'try_volume': 0.0,
            'total_volume': 0.0,
            'intra_cluster': {'upload': 0.0, 'download': 0.0},
            'inter_cluster': 0.0,
            'control_overhead': 0.0,
            'collisions': 0,
            't_latency': [],
            'p_latency': [],
            'utilization': 0.0
        }

    def get_metrics(self):
        """Get current metrics."""
        return self.current_slot.copy()
