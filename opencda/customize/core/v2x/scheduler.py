from abc import ABC, abstractmethod
import weakref
from typing import List, Tuple, Optional
# from opencda.core.common.v2x_manager import V2XManager
from opencda.customize.core.v2x.network_manager import NetworkManager, ResourceConflictError


class Scheduler(ABC):
    """
    Abstract base class for resource scheduling algorithms.
    """

    def __init__(self, network_manager: 'NetworkManager', config={}):
        # Use weakref to avoid circular references
        self._network_manager = weakref.ref(network_manager)
        self.config = config

    @property
    def network_manager(self) -> Optional['NetworkManager']:
        """
        Get the NetworkManager instance (or None if it has been garbage collected).

        Returns:
            Optional[NetworkManager]: The NetworkManager instance.
        """
        return self._network_manager()

    @abstractmethod
    def schedule(self, source, target, volume: float) -> bool:
        """
        Schedule resources for a communication request.

        Args:
            source (V2XManager): The source vehicle manager.
            target (V2XManager): The target vehicle manager.
            volume (float): The data volume to transmit (in MB).

        Returns:
                success(bool): Whether the communication was successful.
        """
                #     Tuple[int, int, int, bool]: A tuple containing:
                # - subchannel: The allocated subchannel index.
                # - start_time_slot: The starting time slot for the communication.
                # - end_time_slot: The ending time slot for the communication.
        pass






#///////////////////////////////////////////////////////////////////////////////////////////////////

class RoundRobinScheduler(Scheduler):
    """
    Allocates resources in a round-robin fashion.
    """

    def __init__(self, network_manager: 'NetworkManager', config={}):
        super().__init__(network_manager, config)
        self.next_subchannel = 0  # Start scheduling from the first subchannel

    def schedule(self, source, target, volume: float) -> Tuple[int, int, int, bool]:
        nm = self.network_manager
        if nm is None:
            return False  # NetworkManager has been garbage collected
        subchannel = self.next_subchannel
        self.next_subchannel = (self.next_subchannel + 1) % nm.subchannel_num
        return nm.communicate(source, target, volume, subchannel)





#///////////////////////////////////////////////////////////////////////////////////////////////////

class InterferenceAwareScheduler(Scheduler):
    """
    Allocates resources while minimizing interference.
    """

    def schedule(self, source, target, volume: float) -> bool:
        nm = self.network_manager
        if nm is None:
            return False
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
                return nm.communicate(source, target, volume, best_subchannel)

        return False





#///////////////////////////////////////////////////////////////////////////////////////////////////
