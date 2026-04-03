import math
from opencda.core.clustering.base import ClusteringAlgorithm
# 导入common模块，而不是使用*导入
from opencda.core.clustering import utils
from opencda.core.clustering.utils import *
from opencda.log.logger_config import logger
from opencda.core.clustering.utils import common
from opencda.core.clustering.utils.common import Cluster  # 使用common中的Cluster类

class CoalitionGame(ClusteringAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)
        self.cav_world = cav_world  # 保持向后兼容
        self.p = common.Params()
        self.coalitions = self.clusters

    def initialize(self):
        """Initialize the clustering algorithm."""
        self.initialize_vehicles()

    def initialize_vehicles(self):
        common.Vehicle_Grid.initialize(self.cav_world)
        common.Cluster.set_election_function(self.election)
        new_coliations = []
        for vid in common.global_vehicles.keys():
            if self.find_coalition(vid) is None:
                new_coliations.append(Cluster({vid}))
        self.coalitions.extend(new_coliations)
        # self.max_grids_per_rb = common.calculate_max_grids_per_rb(None, self.p.bandwidth_per_channel, self.p.T_ddl, self.coalitions[0].grid_bits)
        
    def election(self, members):
        from opencda.core.common.misc import compute_distance
        # global global_vehicles
        """选择位置最靠近所有成员中心的车辆作为簇头"""
        if not members:
            return None
            
        # 计算所有成员的位置中心
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        member_count = len(members)
        
        # 收集所有成员的位置
        positions = {}
        for vid in members:
            vehicle = common.global_vehicles[vid]
            location = vehicle.get_position().location
            positions[vid] = location
            sum_x += location.x
            sum_y += location.y
            sum_z += location.z
        
        # 计算中心位置
        center_x = sum_x / member_count
        center_y = sum_y / member_count
        center_z = sum_z / member_count
        
        # 创建中心位置对象（模拟carla.Location的结构）
        center_location = type('Location', (), {'x': center_x, 'y': center_y, 'z': center_z})
        
        # 计算每个成员到中心的距离，找到最近的
        min_distance = float('inf')
        closest_vid = None
        
        for vid, location in positions.items():
            distance = compute_distance(location, center_location)
            if distance < min_distance:
                min_distance = distance
                closest_vid = vid
        
        return closest_vid
    
    def stability_cost(self, vid, coalition):   
        """计算车辆与联盟的稳定性成本，基于预测位置的稳定性系数"""
        if len(coalition.members) == 0:
            return 0.0
            
        # 预测车辆在T_min_stab时间后的位置
        pred_pos = self.predict_vehicle_position(vid, coalition)
        if pred_pos is None:
            return 0.0
        
        # 计算预测位置的感知覆盖G_i^pred
        pred_sens_grids = self.compute_predicted_sens_grids(vid, pred_pos)
        if not pred_sens_grids:
            return 0.0
        
        # 计算交集G_i^int = G_i^pred ∩ G_S^req
        coalition_req_grids = coalition.req_grids
        int_grids = pred_sens_grids & coalition_req_grids
        
        # 计算稳定性系数β_i = |G_i^int| / |G_i^pred|
        beta_i = len(int_grids) / len(pred_sens_grids)
        
        # 返回稳定性系数作为稳定性成本（值越大表示稳定性越高）
        return beta_i
    
    def predict_vehicle_position(self, vid, coalition):
        """预测车辆在T_min_stab时间后的位置"""
        # 获取车辆当前位置和速度
        pos = common.get_vehicle_position(vid)
        vel = common.get_vehicle_velocity(vid)
        if pos is None or vel is None:
            return None
        
        # 计算联盟平均位置和速度
        mean_pos, mean_vel = common.compute_coalition_mean(coalition)
        if mean_pos is None or mean_vel is None:
            return None
        
        # 计算位移和速度偏差
        delta_pos = [pos[i] - mean_pos[i] for i in range(3)]
        delta_vel = [vel[i] - mean_vel[i] for i in range(3)]
        
        # 预测位置
        pred_pos = [pos[i] + delta_vel[i] * self.p.T_min_stab for i in range(3)]
        return pred_pos
    
    def compute_predicted_sens_grids(self, vid, pred_pos):
        """根据预测位置计算车辆的感知覆盖网格"""        
        vehicle_manager_dict = self.cav_world.get_vehicle_managers()
        vm = vehicle_manager_dict.get(vid)
        if not vm or not vm.is_ok:
            return set()
        
        lidar = vm.perception_manager.lidar
        grid_size = lidar.grid_size
        lidar_range = lidar.lidar_range
        
        grid_coords = lidar.generate_perception_grid_coords(grid_size, lidar_range, int(pred_pos[0]), int(pred_pos[1]))
        
        # 生成感知网格
        pred_sens_grids = set()
        for x, y in grid_coords:
                grid_id = lidar.get_point_grid_id((x, y))
                pred_sens_grids.add(grid_id)
        
        return pred_sens_grids

    def marginal_contribution(self, coalition, vid):
        if vid in coalition.members:
            logger.info(f"vehicle {vid} is already in coalition {coalition.members}.")
            return 0.0
        if vid not in common.global_vehicles:
            logger.info(f"vehicle {vid} not in global_vehicles.")
            return 0.0
        # valuable_grids = common.global_vehicles[vid].sens_grids - coalition.high_density_grids
        # new_grids = valuable_grids & coalition.req_grids
        valuable_grids = common.global_vehicles[vid].sens_grids
        valuable_grids = valuable_grids & coalition.sens_grids
        grid_gain = common.avg_grids_score(vid, valuable_grids)
        stab_diff = self.stability_cost(vid, coalition)
        gain = grid_gain * stab_diff
        return gain
    
    def current_contribution(self, coalition, vid):
        if vid not in coalition.members:
            return 0.0
        coalition.remove_member(vid)
        ret = self.marginal_contribution(coalition, vid)
        coalition.add_member(vid)
        return ret
    
    def main(self, max_iter=20):
        self.coalition_formation(max_iter)

    def find_coalition(self, vid):
        for coalition in self.coalitions:
            if vid in coalition.members:
                return coalition
        return None
    
    def check_is_ok(self):
        vehicle_manager_dict = self.cav_world.get_vehicle_managers()
        for vid, vm in vehicle_manager_dict.items():
            if not vm.is_ok and vid in common.global_vehicles:
                logger.info(f"vehicle {vid} left the map.")
                common.global_vehicles.pop(vid, None)
                current = self.find_coalition(vid)
                if current is None:
                    continue
                current.remove_member(vid)
                if current.size() == 0:
                    self.coalitions.remove(current)
            if vm.is_ok and vid not in common.global_vehicles:
                logger.info(f"vehicle {vid} join the map.")
                self.initialize_vehicles()
    
    def update_cluster_states(self):
        super().update_cluster_states()
        # vehicle_manager_dict = self.cav_world.get_vehicle_managers()
        # head_ids = set([c.head_id for c in self.coalitions])
        # for vid, vm in vehicle_manager_dict.items():
        #     vm.perception_manager.co_manager.ego_vehicle_ids = head_ids

    def ego_coalition_be_first(self):
        ego_id = self.cav_world.ego_id
        ego_coalition = self.find_coalition(ego_id)
        if ego_coalition is None:
            return
        self.coalitions.remove(ego_coalition)
        self.coalitions.insert(0, ego_coalition)

    def coalition_formation(self, max_iter=20):
        self.check_is_ok()
        self.ego_coalition_be_first()
        for _ in range(max_iter):
            logger.info("--- Coalition Formation Iteration %d ---", _+1)
            updated = False
            for vid in common.global_vehicles.keys():
                current = self.find_coalition(vid)
                if current is None:
                    logger.info(f"Vehicle {vid} is not in any coalition.")
                    continue
                current_contribution = self.current_contribution(current, vid)
                # if vid == self.cav_world.ego_id:
                #     print(f"ego's current coalition {current.members} contribution: {current_contribution:.3f}.")
                best_delta = current_contribution
                best_coalition = current
                for c in self.coalitions:
                    if c is current:
                        continue
                    if c.size() >= self.p.N_max:
                        # print(f"Vehicle {vid} cannot join coalition {c.members} due to size limit.")
                        continue
                    delta = self.marginal_contribution(c, vid)
                    if delta > best_delta:
                        best_delta = delta
                        best_coalition = c
                    # if vid == self.cav_world.ego_id:
                    #     print(f"ego's marginal contribution to coalition {c.members} is {delta:.3f}.")
                if best_coalition is not current and best_delta > current_contribution * self.p.ita:
                    logger.info(f"Vehicle {vid} moves from coalition {current.members}(contribution: {current_contribution:.3f}) to {best_coalition.members} with contribution {best_delta:.3f}.")
                    current.remove_member(vid)
                    best_coalition.add_member(vid)
                    if current.size() == 0:
                        self.coalitions.remove(current)
                    updated = True
            if not updated:
                break
        logger.info(f"Coalition formation converged in %d iterations.", _+1)
        for coalition in self.coalitions:
            logger.info(f"[{coalition.members}]")
        return self.coalitions
    def run(self):
        """Execute coalition game clustering algorithm."""
        self.initialize_vehicles()
        return self.coalition_formation()
