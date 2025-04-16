from abc import ABC, abstractmethod
import weakref
from typing import List, Tuple, Optional
from opencda.core.common.v2x_manager import V2XManager
from opencda.customize.core.v2x.network_manager import NetworkManager, ResourceConflictError


class Scheduler(ABC):
    """
    Abstract base class for resource scheduling algorithms.
    """

    def __init__(self, network_manager: 'NetworkManager'):
        # Use weakref to avoid circular references
        self._network_manager = weakref.ref(network_manager)

    @property
    def network_manager(self) -> Optional['NetworkManager']:
        """
        Get the NetworkManager instance (or None if it has been garbage collected).

        Returns:
            Optional[NetworkManager]: The NetworkManager instance.
        """
        return self._network_manager()

    @abstractmethod
    def schedule(self, source: 'V2XManager', target: 'V2XManager', volume: float) -> Tuple[int, int, int, bool]:
        """
        Schedule resources for a communication request.

        Args:
            source (V2XManager): The source vehicle manager.
            target (V2XManager): The target vehicle manager.
            volume (float): The data volume to transmit (in MB).

        Returns:
            Tuple[int, int, int, bool]: A tuple containing:
                - subchannel: The allocated subchannel index.
                - start_time_slot: The starting time slot for the communication.
                - end_time_slot: The ending time slot for the communication.
                - success: Whether the allocation was successful.
        """
        pass






#///////////////////////////////////////////////////////////////////////////////////////////////////

class RoundRobinScheduler(Scheduler):
    """
    Allocates resources in a round-robin fashion.
    """

    def __init__(self, network_manager: 'NetworkManager'):
        super().__init__(network_manager)
        self.next_subchannel = 0  # Start scheduling from the first subchannel

    def schedule(self, source: 'V2XManager', target: 'V2XManager', volume: float) -> Tuple[int, int, int, bool]:
        nm = self.network_manager
        if nm is None:
            return -1, -1, -1, False  # NetworkManager has been garbage collected

        subchannel = self.next_subchannel
        self.next_subchannel = (self.next_subchannel + 1) % nm.subchannels

        try:
            subchannel, start_time_slot, end_time_slot = nm.allocate_resource(source, target, volume, subchannel)
            return subchannel, start_time_slot, end_time_slot, True
        except ResourceConflictError as e:
            print(f"RoundRobinScheduler: {e}")
            return subchannel, -1, -1, False







#///////////////////////////////////////////////////////////////////////////////////////////////////

class InterferenceAwareScheduler(Scheduler):
    """
    Allocates resources while minimizing interference.
    """

    def schedule(self, source: 'V2XManager', target: 'V2XManager', volume: float) -> Tuple[int, int, int, bool]:
        nm = self.network_manager
        if nm is None:
            return -1, -1, -1, False  # NetworkManager has been garbage collected

        min_interference = float('inf')
        best_subchannel = -1

        for subchannel in range(nm.subchannels):
            try:
                interference = nm.calculate_interference(subchannel, [target])
                if interference < min_interference:
                    min_interference = interference
                    best_subchannel = subchannel
            except ValueError:
                continue

        if best_subchannel != -1:
            try:
                subchannel, start_time_slot, end_time_slot = nm.allocate_resource(source, target, volume, best_subchannel)
                return subchannel, start_time_slot, end_time_slot, True
            except ResourceConflictError as e:
                print(f"InterferenceAwareScheduler: {e}")

        return -1, -1, -1, False





#///////////////////////////////////////////////////////////////////////////////////////////////////
from opencda.customize.core.clustering.clustering_scheduler import ClusterBasedScheduler
from opencda.core.common.cav_world import CavWorld

def build_scheduler(scheduler_name):
    """
    Factory method to build a scheduler object given its name.
    Args:
    scheduler_name (str): Name of the scheduler (e.g., "RoundRobin", "Greedy", "ClusterBased").
    network_manager (NetworkManager): Required network manager.

    Returns:
        Scheduler: An instance of the corresponding Scheduler subclass.

    Raises:
        ValueError: If the scheduler name is not recognized.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'roundrobin':
        return RoundRobinScheduler(CavWorld.network_manager)
    elif scheduler_name == 'greedy':
        return InterferenceAwareScheduler(CavWorld.network_manager)
    elif scheduler_name == 'clusterbased':
        return ClusterBasedScheduler(CavWorld.network_manager)
    # elif scheduler_name == 'random':
    #     return RandomScheduler(CavWorld.network_manager)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")
