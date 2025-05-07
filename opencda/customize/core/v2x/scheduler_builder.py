
from opencda.customize.core.v2x.scheduler import InterferenceAwareScheduler, RoundRobinScheduler
from opencda.customize.core.clustering.clustering_scheduler import ClusterBasedScheduler
from opencda.core.common.cav_world import CavWorld

def build_scheduler(scheduler_name, config={}):
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
        return ClusterBasedScheduler(CavWorld.network_manager, config)
    # elif scheduler_name == 'random':
    #     return RandomScheduler(CavWorld.network_manager)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

