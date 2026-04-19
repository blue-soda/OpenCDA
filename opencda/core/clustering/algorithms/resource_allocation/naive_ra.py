from opencda.core.clustering.base import ResourceAllocationAlgorithm
from opencda.core.clustering import utils
from opencda.core.clustering.utils import *
from opencda.log.logger_config import logger


class NaiveRA(ResourceAllocationAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)
        self.cav_world = cav_world

    def main(self):
        pass

    def run(self):
        self.update_resource_allocation_strategy()
        return True

    def _build_channel_allocation(self, vehicle_ids, subchannel_num):
        allocation = {}
        sorted_ids = sorted(vehicle_ids)
        if subchannel_num <= 0:
            return allocation

        for target_id in sorted_ids:
            source_ids = [source_id for source_id in sorted_ids if source_id != target_id]
            for offset, source_id in enumerate(source_ids):
                allocation[(source_id, target_id)] = offset % subchannel_num
        return allocation

    def update_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        vehicle_ids = list(vehicle_dict.keys())
        subchannel_num = 0
        if vehicle_dict:
            first_vm = next(iter(vehicle_dict.values()))
            subchannel_num = first_vm.v2x_manager.scheduler.network_manager.subchannel_num

        channel_allocation = self._build_channel_allocation(vehicle_ids, subchannel_num)
        logger.info(f"NaiveRA explicit channel allocation: {channel_allocation}")

        for vid, vm in vehicle_dict.items():
            scheduler = vm.v2x_manager.scheduler
            scheduler.clear_strategies()
            scheduler.use_default_subchannel = False
            scheduler.set_strategies(channel_allocation)
            vm.perception_manager.co_manager.enable_grid = False
            vm.perception_manager.co_manager.clear_grid_selection()
            # vm.perception_manager.apply_late_fusion = False
