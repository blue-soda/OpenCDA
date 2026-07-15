from opencda.core.networking.scheduler import Scheduler
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import math
from opencda.core.networking.network_manager import NetworkManager, ResourceConflictError
import opencda.core.networking.utils as utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from opencda.log.logger_config import logger
from random import uniform
import numpy as np
from opencda.core.networking.utils import *

from opencda.core.clustering.algorithms.resource_allocation.builder import \
    build_resource_allocator

class ClusteringScheduler(Scheduler):
    resource_allocation_algorithm = None
    def __init__(self, cav_world, config={}):
        super().__init__(cav_world, config)
        self.is_cluster_based = True
        self.use_default_subchannel = False
        self.use_default_scheduler = False
        self.channel_allocation: Dict[Tuple[int, int], int] = {}  # {(source, target): 子信道}

        if ClusteringScheduler.resource_allocation_algorithm is None:
            ra_name = config.get('resource_allocation_algorithm',
                                 config.get('resource_allocation', {}).get(
                                     'algorithm', 'potential_game'))
            ClusteringScheduler.resource_allocation_algorithm = \
                build_resource_allocator(ra_name, cav_world)
            logger.info("ClusteringScheduler resource allocation=%s", ra_name)

    def get_subchannel_allocation(self, link: Tuple[int, int]):
        if link in self.channel_allocation:
            return True, self.channel_allocation[link]
        else:
            return self.use_default_subchannel, -1
    
    def set_strategies(self, strategies):
        self.channel_allocation.update(strategies)

    def clear_strategies(self):
        self.channel_allocation.clear()

    def _get_naive_subchannel(self, source_id: int) -> int:
        subchannel_num = self.network_manager.subchannel_num
        if subchannel_num <= 0:
            return -1
        return (4 - source_id) % subchannel_num

    @staticmethod
    def run():
        ClusteringScheduler.resource_allocation_algorithm.run()

    @staticmethod
    def set_clusters(clusters):
        ClusteringScheduler.resource_allocation_algorithm.set_clusters(clusters)    
        
    def schedule(self, source, target, volume: float) -> bool:
        """执行分簇博弈子信道分配"""
        link = (source.vehicle_id, target.vehicle_id)
        success, ch = self.get_subchannel_allocation(link)
        if (not success and self.network_manager.use_ns3 and
                isinstance(ClusteringScheduler.resource_allocation_algorithm, NaiveRA)):
            ch = self._get_naive_subchannel(source.vehicle_id)
            if ch >= 0:
                logger.info(f"[DEBUG] NaiveRA inline schedule link={link}, volume={volume}, subchannel={ch}")
                success = True
        if not success:
            if self.network_manager.use_ns3:
                # Fall back to NS3 default scheduling only when no explicit allocation exists.
                logger.info(f"[DEBUG] NS3 schedule fallback to default subchannel for link={link}, volume={volume}")
                return self.network_manager.communicate(
                    source, target, volume,
                    subchannel_start=-1, subchannel_num=0
                )
            return False

        logger.info(f"[DEBUG] schedule link={link}, volume={volume}, subchannel={ch}, use_ns3={self.network_manager.use_ns3}")
        success = self.network_manager.communicate(
            source, target, volume,
            subchannel_start=ch, subchannel_num=1
        )
        return success
    
