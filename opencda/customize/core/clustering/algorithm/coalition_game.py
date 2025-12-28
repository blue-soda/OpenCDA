import math
from opencda.customize.core.clustering.clustering_algorithm_manager import ClusteringAlgorithm
# 导入common模块，而不是使用*导入
import opencda.customize.core.clustering.algorithm.common as common
from opencda.customize.core.clustering.algorithm.common import *
from opencda.log.logger_config import logger

class CoalitionGame(ClusteringAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)
        self.p = common.Params()
        self.coalitions = []

    def initialize_vehicles(self):
        common.Vehicle_Grid.initialize(self.cav_world)
        self.coalitions = [Coalition({vid}) for vid in common.global_vehicles.keys()]
        self.max_grids_per_rb = common.calculate_max_grids_per_rb(None, self.p.bandwidth_per_channel, self.p.T_ddl, self.coalitions[0].grid_bits)
        
    def stability_cost(self, vid, coalition):   
        """计算车辆与联盟的稳定性成本，为与联盟内所有成员的时空相似性的负平均值"""
        if len(coalition.members) == 0:
            return 0.0
            
        ego_vehicle = common.global_vehicles.get(vid)
        if not ego_vehicle:
            return 0.0
            
        ego_data = ego_vehicle.summary_state()
        total_distance = 0.0
        count = 0
        
        for member_id in coalition.members:
            if member_id == vid:  # 不计算与自身的相似性
                continue
                
            neighbor_vehicle = common.global_vehicles.get(member_id)
            if not neighbor_vehicle:
                continue
                
            neighbor_data = neighbor_vehicle.summary_state()
            distance = compute_spatiotemporal_distance(ego_data, neighbor_data)
            total_distance += distance
            count += 1
        
        if count == 0:
            return 0.0
            
        average_distance = total_distance / count
        return average_distance
    
    def comm_cost(self, grid_num):
        max_grids = self.max_grids_per_rb
        
        if grid_num >= max_grids:
            return 1000.0
        
        proximity = grid_num / max_grids
        base_cost = 1
        penalty = base_cost * (proximity **2) / (1 - proximity)
        
        total_cost = penalty
        return total_cost

    def marginal_contribution(self, coalition, vid):
        if vid in coalition.members:
            logger.info(f"vehicle {vid} is already in coalition {coalition.members}.")
            return 0.0
        if vid not in common.global_vehicles:
            logger.info(f"vehicle {vid} not in global_vehicles.")
            return 0.0
        new_grids = common.global_vehicles[vid].sens_grids - coalition.high_density_grids
        new_grids = new_grids & coalition.req_grids
        grid_gain = common.avg_grids_score(vid, new_grids)
        stab_diff = self.stability_cost(vid, coalition)
        comm_diff = self.comm_cost(len(new_grids))
        # print(f"marginal_contribution: grid_gain={grid_gain}, stab_diff={stab_diff}, comm_diff={comm_diff}")
        return grid_gain - self.p.alpha * stab_diff - self.p.beta * comm_diff
    
    def current_contribution(self, coalition, vid):
        if vid not in coalition.members:
            return 0.0
        coalition.remove_member(vid)
        ret = self.marginal_contribution(coalition, vid)
        coalition.add_member(vid)
        return ret
    
    def run(self, max_iter=20):
        ret = self.coalition_formation(max_iter)
        self.update_cluster_states()
        return ret

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
    
    def update_cluster_states(self):
        for coalition in self.coalitions:
            head_id = coalition.head_id
            for member_id in coalition.members:
                if member_id not in common.global_vms:
                    logger.info(f"Vehicle {member_id} not in global_vms.")
                    continue
                member_v2x = common.global_vms[member_id]
                member_v2x.cluster_state['head_id'] = head_id
                member_v2x.cluster_state['member_ids'] = set(coalition.members)

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
                    if c.size() >= self.p.N_max: #TODO: bad if-statement
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