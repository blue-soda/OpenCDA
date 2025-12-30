import opencda.customize.core.clustering.algorithm.common as common
from opencda.log.logger_config import logger
class ClusteringAlgorithm:
    def __init__(self, cav_world):
        self.cav_world = cav_world
        self.need_init = True
        self.vehicle_nums = 0
        self.clusters = [] # list of common.Cluster
        
    def main(self):
        pass

    def run(self):
        self.main()
        self.update_cluster_states()
        return self.clusters
    
    def update_cluster_states(self):
        for cluster in self.clusters:
            print(f"cluster: {cluster.members}")
            head_id = cluster.head_id
            for member_id in cluster.members:
                if member_id not in common.global_vms:
                    logger.info(f"Vehicle {member_id} not in global_vms.")
                    continue
                member_v2x = common.global_vms[member_id]
                member_v2x.cluster_state['head_id'] = head_id
                member_v2x.cluster_state['member_ids'] = set(cluster.members)

    def initialize_vehicles(self):
        pass

    def check_initialized(self):
        if self.vehicle_nums < len(self.cav_world.vehicle_id_set):
            self.vehicle_nums = len(self.cav_world.vehicle_id_set)
            self.need_init = True
            print('initialize vehicle:', self.vehicle_nums)
        return self.need_init

    def initialize(self):
        if self.check_initialized():
            self.initialize_vehicles()
            self.need_init = False


class ResourceAllocationAlgorithm:
    def __init__(self, cav_world):
        self.cav_world = cav_world
    
    def main(self):
        pass

    def run(self):
        self.clear_resource_allocation_strategy()
        self.main()
        self.update_resource_allocation_strategy()

    def update_resource_allocation_strategy(self):
        # API to update strategy:
        # vehicle_dict = self.cav_world.get_vehicle_managers()
        # vehicle_dict[hid].perception_manager.co_manager.set_grid_selection(grids_selection) #(not necessary), grids_selection: Dict[int, List[str]]
        # vehicle_dict[hid].v2x_manager.scheduler.set_strategies(strategy) #strategy: Dict[Tuple[int, int], int]
        pass

    def clear_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for vid, vehicle_manager in vehicle_dict.items():
            vehicle_manager.v2x_manager.scheduler.clear_strategies()
            vehicle_manager.perception_manager.co_manager.clear_grid_selection()

class ClusterResourceAllocationAlgorithm(ResourceAllocationAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)
        self.clusters = []
    
    def set_clusters(self, clusters):
        if clusters is None or len(clusters) == 0:
            raise ValueError("clusters must be non-empty")
        self.clusters = clusters

    def update_resource_allocation_strategy(self):
        # API to update strategy:
        # vehicle_dict = self.cav_world.get_vehicle_managers()
        # vehicle_dict[hid].perception_manager.co_manager.set_grid_selection(grids_selection) #(not necessary), grids_selection: Dict[int, List[str]]
        # vehicle_dict[hid].v2x_manager.scheduler.set_strategies(strategy) #strategy: Dict[Tuple[int, int], int]
        pass

    def clear_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for vid, vehicle_manager in vehicle_dict.items():
            vehicle_manager.v2x_manager.scheduler.clear_strategies()
            vehicle_manager.perception_manager.co_manager.clear_grid_selection()