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
    as PotentialGame
from opencda.customize.core.clustering.algorithm.pcs import PCS
from opencda.customize.core.clustering.algorithm.naive_ra import NaiveRA
from opencda.customize.core.clustering.algorithm.mws import MWS
from opencda.customize.core.clustering.algorithm.random import RandomRA

class ClusteringScheduler(Scheduler):
    resource_allocation_algorithm = None
    def __init__(self, cav_world, config={}):
        super().__init__(cav_world, config)
        self.is_cluster_based = True
        self.use_default_subchannel = False
        self.use_default_scheduler = False
        self.channel_allocation: Dict[Tuple[int, int], int] = {}  # {(source, target): 子信道}

        if ClusteringScheduler.resource_allocation_algorithm is None:
            # ClusteringScheduler.resource_allocation_algorithm = PotentialGame(cav_world)
            # ClusteringScheduler.resource_allocation_algorithm = PCS(cav_world)
            # ClusteringScheduler.resource_allocation_algorithm = MWS(cav_world)
            ClusteringScheduler.resource_allocation_algorithm = RandomRA(cav_world)
            # ClusteringScheduler.resource_allocation_algorithm = NaiveRA(cav_world)

    def get_subchannel_allocation(self, link: Tuple[int, int]):
        if link in self.channel_allocation:
            return True, self.channel_allocation[link]
        else:
            return self.use_default_subchannel, -1
    
    def set_strategies(self, strategies):
        self.channel_allocation.update(strategies)

    def clear_strategies(self):
        self.channel_allocation.clear()

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
        success, ch = self.get_subchannel_allocation(link)
        if not success:
            print(f"Link {link} not in channel_allocation or has no allocation, allocation: {self.channel_allocation}")
            return False
        print(f"Link {link} allocated to subchannel {ch}")
        success = self.network_manager.communicate(
            source, target, volume,
            subchannel_start=ch, subchannel_num=1
        )
        return success
    
