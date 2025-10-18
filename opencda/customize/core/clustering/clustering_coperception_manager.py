from opencda.core.sensing.perception.coperception_manager \
    import CoperceptionManager
from opencda.log.logger_config import logger

class ClusteringCoperceptionManager(CoperceptionManager):
    def __init__(self, vid, v2x_manager, coperception_libs, enable_network=False, network_manager=None):
        super().__init__(vid, v2x_manager, coperception_libs, enable_network, network_manager)
        self.communicate_inside_cluster = True

    def set_communicate_inside_cluster(self):
        self.communicate_inside_cluster = True
    
    def set_communicate_outside_cluster(self):
        self.communicate_inside_cluster = False

    def get_coperception_cavs_dict(self):
        if self.communicate_inside_cluster:
            data_inside_cluster = {}
            vms = self.v2x_manager.get_cluster_members()['members']
            for vid, vm in vms.items():
                v2x_manager = vm.v2x_manager
                data_inside_cluster.update({vid: {'v2x_manager': v2x_manager, 'vehicle_manager': vm}})
                # print(f"cluster member vid: {vid}")
            return data_inside_cluster
        else:
            all_neighbors = self.v2x_manager.cav_nearby
            cluster_members = self.v2x_manager.cluster_state['members']
            key_diff = all_neighbors.keys() - cluster_members.keys()
            data_outside_cluster = {k: all_neighbors[k] for k in key_diff}
            return data_outside_cluster
    
    def broadcast_objects_info(self, objects):
        if self.v2x_manager.is_cluster_head():
            for vid, member_data_dict in self.get_coperception_cavs_dict().items():
                member_v2x_manager = member_data_dict['v2x_manager']
                member_v2x_manager.set_buffer(source_id=self.vid)
                member_v2x_manager.set_buffer(objects=objects)


    