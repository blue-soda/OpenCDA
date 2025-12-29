class ClusteringAlgorithm:
    def __init__(self, cav_world):
        self.cav_world = cav_world
        self.need_init = True
        self.vehicle_nums = 0
        
    def run(self):
        pass
    
    def update_cluster_states(self):
        pass

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

class ClusterResourceAllocationAlgorithm:
    def __init__(self, cav_world):
        self.cav_world = cav_world
        self.clusters = []
    def run(self):
        pass
    def set_clusters(self, clusters):
        if len(clusters) == 0:
            raise ValueError("clusters must be non-empty")
        self.clusters = clusters
    def update_resource_allocation_strategy(self):
        pass
    def clear_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for vid, vehicle_manager in vehicle_dict.items():
            vehicle_manager.v2x_manager.scheduler.clear_strategies()