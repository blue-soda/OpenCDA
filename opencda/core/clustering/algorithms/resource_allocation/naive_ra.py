from opencda.core.clustering.base import ResourceAllocationAlgorithm
from opencda.core.clustering import utils
from opencda.core.clustering.utils import *
from opencda.log.logger_config import logger

class NaiveRA(ResourceAllocationAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)

    def main(self):
        pass

    def update_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for vid, vm in vehicle_dict.items():
            vm.v2x_manager.scheduler.use_default_subchannel = True
            vm.perception_manager.co_manager.enable_grid = False
            # vm.perception_manager.apply_late_fusion = False

