from opencda.core.sensing.perception.coperception_manager \
    import CoperceptionManager

class ClusteringCoperceptionManager(CoperceptionManager):
    def __init__(self, vid, v2x_manager, coperception_libs):
        super().__init__(vid, v2x_manager, coperception_libs)

    def communicate_inside_cluster(self):
        print(f"cluster head {self.v2x_manager.vehicle_id} is communicating inside cluster")
        data = {}
        if self.v2x_manager is not None:
            for vehicle_id in self.v2x_manager.cluster_state['members'].keys():
                if vehicle_id == self.v2x_manager.vehicle_id:
                    continue
                data_dict = self.v2x_manager.cav_nearby.get(vehicle_id)
                if data_dict is None:
                    print(f"member {vehicle_id} is not a neighbor")
                    continue
                data.update({str(vehicle_id): data_dict})
        return data
    
    def communicate_outside_cluster(self):
        all_neighbors = self.comunicate()
        cluster_members = self.communicate_inside_cluster()
        key_diff = all_neighbors.keys() - cluster_members.keys()
        data_outside_cluster = {k: all_neighbors[k] for k in key_diff}
        return data_outside_cluster
    
    def broadcast_inside_cluster(self, source=None, objects=None):#, results=None):
        if self.v2x_manager.is_cluster_head():
            for vid, member_data_dict in self.communicate_inside_cluster().items():
                #member_vm = member_data_dict['vehicle_manager']
                member_v2x_manager = member_data_dict['v2x_manager']
                if source:
                    member_v2x_manager.set_buffer(source=source)
                if objects:
                    member_v2x_manager.set_buffer(objects=objects)
                # if results:
                #     member_v2x_manager.set_buffer(results=results)

    