from opencda.core.clustering.base import ClusteringAlgorithm
from opencda.core.clustering import utils
from opencda.core.clustering.utils import *
from opencda.core.clustering.utils import common
from opencda.core.clustering.utils.common import Cluster

class NaiveCluster(ClusteringAlgorithm):
    def __init__(self, cav_world, all_in_one: bool = False, apply_grid: bool = True):
        super().__init__(cav_world)
        self.cav_world = cav_world
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
    
    def main(self):
        # common.visualize_sens_grids(1)
        pass

    def run(self):
        self.initialize_vehicles()
        self.main()
        return self.clusters
