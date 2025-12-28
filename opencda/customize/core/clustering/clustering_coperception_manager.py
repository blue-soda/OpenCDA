from opencda.core.sensing.perception.coperception_manager \
    import CoperceptionManager
from opencda.log.logger_config import logger

class ClusteringCoperceptionManager(CoperceptionManager):
    def __init__(self, vid, v2x_manager, coperception_libs, enable_network=False, network_manager=None):
        super().__init__(vid, v2x_manager, coperception_libs, enable_network, network_manager)
        self.communicate_inside_cluster = True
        self.vehicle_id = vid
        self.grid_selection = {}
        
    def set_communicate_inside_cluster(self):
        self.communicate_inside_cluster = True
    
    def set_communicate_outside_cluster(self):
        self.communicate_inside_cluster = False
        
    def set_grid_selection(self, grid_selection):
        """
        设置从分簇博弈获得的点云网格选择结果
        :param grid_selection: 从clustering_game_manager.py获得的网格选择结果 {member_id: [grid_ids]}
        """
        self.grid_selection.update(grid_selection)
    
    def get_data_from_lidar(self, lidar, vehicle_id=None):
        """
        根据点云网格选择结果获取数据
        :param lidar: lidar对象
        :param vehicle_id: 可选参数，指定车辆ID
        :return: 根据网格选择结果过滤后的点云数据
        """
        if not vehicle_id:
            vehicle_id = self.vehicle_id
        
        # 检查是否有该车辆的网格选择结果
        if vehicle_id in self.grid_selection and self.grid_selection[vehicle_id]:
            # print(f"Vehicle {vehicle_id} grid selection: {self.grid_selection[vehicle_id]}")
            selected_grids = self.grid_selection[vehicle_id]
            grid_data = lidar.get_local_points_by_grid_ids(selected_grids)
            return grid_data
        # 如果是本地车辆，返回全部原始数据
        elif vehicle_id == self.vehicle_id:
            # print(f"Vehicle {vehicle_id} is the local vehicle. Returning raw data.")
            return lidar.data
        else:
        # 如果没有选择结果，则返回空数据
            # print(f"Vehicle {vehicle_id} has no grid selection. {self.grid_selection.keys()}")
            return None
    
    def get_coperception_cavs_dict(self):
        if self.communicate_inside_cluster:
            data_inside_cluster = {}
            vms = self.v2x_manager.get_cluster_member_vms()['members']
            # print(f"Cluster members: {vms.keys()}")
            for vid, vm in vms.items():
                if vid == self.vehicle_id:
                    continue
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


    