from opencda.customize.core.clustering.clustering_algorithm_manager import ClusterResourceAllocationAlgorithm
import opencda.customize.core.clustering.algorithm.common as common
from opencda.customize.core.clustering.algorithm.common import *
from opencda.log.logger_config import logger

class NaiveRA(ClusterResourceAllocationAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)

    def main(self):
        pass

    def update_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for vid, vm in vehicle_dict.items():
            vm.v2x_manager.scheduler.use_default_subchannel = True
            vm.perception_manager.co_manager.enable_grid = False
            vm.perception_manager.apply_late_fusion = False

