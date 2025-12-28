from opencda.customize.core.v2x.scheduler import Scheduler
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import math
from opencda.customize.core.v2x.network_manager import NetworkManager, ResourceConflictError
import opencda.customize.core.v2x.utils as utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from opencda.log.logger_config import logger
from random import uniform
import numpy as np
from opencda.customize.core.v2x.utils import *

from opencda.customize.core.clustering.algorithm.potential_game import PotentialGame \
    as ResourceAllocationAlgorithm

class ClusteringScheduler(Scheduler):
    resource_allocation_algorithm = None
    def __init__(self, cav_world, config={}):
        super().__init__(cav_world, config)
        self.is_cluster_based = True
        self.channel_allocation: Dict[Tuple[int, int], int] = {}  # {(source, target): 子信道}

        if ClusteringScheduler.resource_allocation_algorithm is None:
            ClusteringScheduler.resource_allocation_algorithm = ResourceAllocationAlgorithm(cav_world)

    def set_strategies(self, strategies):
        self.channel_allocation.update(strategies)

    @staticmethod
    def run():
        ClusteringScheduler.resource_allocation_algorithm.run()

    @staticmethod
    def set_clusters(clusters):
        ClusteringScheduler.resource_allocation_algorithm.set_clusters(clusters)    
        
    def schedule(self, source, target, volume: float) -> bool:
        """执行分簇博弈子信道分配"""
        link = (source.vehicle_id, target.vehicle_id)
        print(f"Schedule {link} with volume {volume}")
        ch = -1
        if link not in self.channel_allocation or self.channel_allocation[link] == -1:
            print(f"Link {link} not in channel_allocation or has no allocation, allocation: {self.channel_allocation}")
            return False
        else:
            ch = self.channel_allocation[link]
        print(f"Link {link} allocated to subchannel {ch}")
        success = self.network_manager.communicate(
            source, target, volume,
            subchannel_start=ch, subchannel_num=1
        )
        return success
    
