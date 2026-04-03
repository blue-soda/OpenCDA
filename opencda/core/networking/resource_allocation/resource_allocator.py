"""Resource allocation for V2X communication."""

from collections import defaultdict
import math


class ResourceAllocator:
    """Manages subchannel allocation and interference."""

    def __init__(self, subchannel_num, subchannel_bandwidth, min_sinr_threshold):
        self.subchannel_num = subchannel_num
        self.subchannel_bandwidth = subchannel_bandwidth
        self.min_sinr_threshold = min_sinr_threshold
        self.active_allocations = defaultdict(set)

    def allocate(self, source, target, volume, subchannel_start):
        """Allocate resources for communication."""
        return True

    def check_interference(self, subchannel, source_id, target_id):
        """Check if allocation causes interference."""
        return len(self.active_allocations[subchannel]) == 0

    def release_expired(self, current_time_slot):
        """Release expired allocations."""
        for subchannel in list(self.active_allocations.keys()):
            self.active_allocations[subchannel] = {
                alloc for alloc in self.active_allocations[subchannel]
                if alloc[2] > current_time_slot
            }
