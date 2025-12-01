
from opencda.customize.core.v2x.scheduler import *
from opencda.customize.core.clustering.clustering_scheduler import *

def build_scheduler(scheduler_name, cav_world, config={}):
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

    if scheduler_name == 'default':
        return DefaultScheduler(cav_world, config)
    elif scheduler_name == 'roundrobin':
        return RoundRobinScheduler(cav_world, config)
    elif scheduler_name == 'greedy':
        return InterferenceAwareScheduler(cav_world, config)
    elif scheduler_name == 'wcgc':
        return WCGCScheduler(cav_world, config)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

