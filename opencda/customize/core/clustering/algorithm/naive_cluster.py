from opencda.customize.core.clustering.clustering_algorithm_manager import ClusteringAlgorithm
import opencda.customize.core.clustering.algorithm.common as common
from opencda.customize.core.clustering.algorithm.common import *

class NaiveCluster(ClusteringAlgorithm):
    def __init__(self, cav_world, all_in_one: bool = False, apply_grid: bool = True):
        super().__init__(cav_world)
        self.p = common.Params()
        self.coalitions = self.clusters
        self.apply_grid = apply_grid
        self.all_in_one = all_in_one

    def initialize_vehicles(self):
        if self.apply_grid:
            common.Vehicle_Grid.initialize(self.cav_world)
        else:
            common.Vehicle.initialize(self.cav_world)
        if self.all_in_one:
            self.clusters = [Cluster([vid for vid in common.global_vehicles.keys()])] # 全体车辆都在一个簇中
        else:
            self.clusters = [Cluster([vid]) for vid in common.global_vehicles.keys()] # 每个车辆单独成簇
    