# cluster_algorithm.py
import math
import random
from opencda.core.clustering.base import ClusteringAlgorithm
from opencda.core.common.misc import compute_distance
from opencda.log.logger_config import logger
from opencda.core.clustering.utils import *
from opencda.core.clustering import utils

class SimilarityAlgorithm(ClusteringAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)
        self.params = {
            'd0': 50.0,         # 距离归一化参数 (单位: m)
            's0': 5.0,           # 速度归一化参数 (单位: m/s)
            'N_th': 4,           # 邻居数阈值
            'N_max': 8,          # 最大簇成员数
            'kappa': 0.05,       # 权重调节系数
            'eta_join': 0.50,    # 加入簇的阈值
            'eta_create': 0.40,  # 创建簇的阈值
            'RSSI_max': 100,     # 最大接收信号强度
            'epsilon': 0.1,      # 协同感知模型差异阈值
            'sigma': 1.0,        # 创建簇的概率调节参数
            'w1': 0.4,           # 通信质量权重
            'w2': 0.3,           # 计算能力权重
            'w3': 0.2,           # 速度一致性权重
            'T_timeout': 1.0,    # 簇头超时时间 (单位: s)
            'shadow_timeout': 0.5,  # 影子簇头超时时间 (单位: s)
            'eta_leave': 0.35,  # 离开阈值 (eta_join - delta_eta)
            'eta_elect': 0.10, #选举阈值(超过当前簇头优先级得分 + eta_elect 才能当选)
            'ego_must_be_leader': True, #ego车辆是否指定为簇头
            'apply_late_fusion': False, #是否跨簇晚期协作
            'record_all_cavs': False #统计所有车辆结果
        }
        self.clusters = []

    def initialize_vehicles(self):
        Vehicle_T.params = self.params
        Vehicle_T.initialize(self.cav_world)

    def run(self):
        all_cavs = list(common.global_vehicles.values())
        for cav in all_cavs:
            cav.read_neighbor_ids()
        for cav in all_cavs:
            cav.leave_join_create_cluster()
        for cav in all_cavs:
            cav.elect_leader()    
        for cav in all_cavs:
            cav.sync_update_cluster_state()    
        self.update_cluster_states()
        return self.clusters

    def update_cluster_states(self):
        self.clusters = []
        for vid, vm in common.global_vms.items():
            vehicle = common.global_vehicles[vid]
            vm.cluster_state['head_id'] = vehicle.cluster_state['head_id']
            vm.cluster_state['member_ids'] = vehicle.cluster_state['member_ids']
            if vehicle.is_cluster_head():
                self.clusters.append(SimpleCluster(vm.cluster_state['member_ids']))
        super().update_cluster_states()

class Vehicle_T(Vehicle_Grid):
    params = None
    def __init__(self, vid, position, speed, direction):
        super().__init__(vid, position, speed, direction)
        self.cluster_state = {
            'head_id': None,               # 当前簇头ID
            'shadow_head_id': None,        # 影子簇头ID
            'member_ids': set(),           # 簇成员ID集合
            'neighbor_ids': set(),         # 邻居ID集合
            'neighbor_data': {},           # 邻居数据缓存 {neighbor_id: data}
            'similarity_scores': {},       # 邻居相似度 {neighbor_id: score}
            'priority_score': 0.0          # 本地优先级得分
        }
        
    def read_neighbor_ids(self):
        self.cluster_state['neighbor_ids'] = common.global_vms[self.id].cav_nearby.keys()
    
    def get_cluster_head(self):
        return self.cluster_state['head_id']
    
    def is_cluster_head(self):
        return self.cluster_state['head_id'] == self.id

    def get_cluster_state(self):
        return self.cluster_state

    def leave_join_create_cluster(self):
        self._update_similarity()
        self._update_cluster_membership()
    
    def elect_leader(self):
        if self.is_cluster_head():
            self._elect_cluster_head()
    
    def sync_update_cluster_state(self):
        self._sync_cluster_state()

    # ------------------------------
    # 时空相似性计算
    # ------------------------------

    @staticmethod
    def calculate_create_probability(avg_similarity, params):
        """计算创建新簇的概率"""
        return sigmoid((params['eta_create'] - avg_similarity) / params['sigma'])

    @staticmethod
    def compute_priority_score(ego_data, cluster_avg_speed, params):
        """计算簇头优先级得分"""
        speed_consistency = 1.5 - sigmoid(abs(ego_data['speed'] - cluster_avg_speed))
        return (
            params['w1'] * ego_data['communication_quality'] +
            params['w2'] * ego_data['computing_capability'] +
            params['w3'] * speed_consistency
        )
    
    # ------------------------------
    # 分簇数据同步
    # ------------------------------
    def _sync_cluster_state(self):
        """与簇头同步簇状态，确保全簇一致"""
        if not self.cluster_state['head_id']:
            return
        head = common.global_vehicles.get(self.cluster_state['head_id'], None)
        if not head:
            return
        head_state = head.get_cluster_state()
        self._update_cluster_state(head_state)

    def _broadcast_cluster_state(self):
        """簇头广播最新状态，强制同步所有成员"""
        for mid in self.cluster_state['member_ids']:
            member = common.global_vehicles.get(mid, None)
            if member and member.vehicle_id != self.vehicle_id:
                member._update_cluster_state(self.cluster_state)

    def _update_cluster_state(self, new_state):
        """成员接收簇头的状态更新"""
        self.cluster_state['head_id'] = new_state['head_id']
        self.cluster_state['shadow_head_id'] = new_state['shadow_head_id']
        self.cluster_state['member_ids'] = set(new_state['member_ids'])
        # logger.debug(f"Vehicle {self.vehicle_id} synced cluster state: head={new_state['head_id']}, members={len(new_state['member_ids'])}")

    # ------------------------------
    # 优先级计算
    # ------------------------------
    def _compute_cluster_avg_speed(self):
        member_speeds = []
        for mid in self.cluster_state['member_ids']:
            member = common.global_vehicles.get(mid, None)
            if member:
                member_speeds.append(member.get_speed())
        
        cluster_avg_speed = sum(member_speeds) / len(member_speeds) if member_speeds else self.get_ego_speed()
        return cluster_avg_speed
    
    def _compute_local_priority(self, cluster_avg_speed=None):
        """仅在簇头选举时计算优先级"""
        if self.params['ego_must_be_leader'] and self.vehicle_id == common.global_ego_id:
            self.cluster_state['local_priority'] = float('inf')
            return
        if cluster_avg_speed is None:
            cluster_avg_speed = self._compute_cluster_avg_speed()
        self.cluster_state['local_priority'] = self.compute_priority_score({
            'speed': self.get_speed(),
            'communication_quality': common.global_vms.get(self.vid).communication_quality,
            'computing_capability': common.global_vms.get(self.vid).computing_capability
        }, cluster_avg_speed, self.params)

    # ------------------------------
    # 分簇逻辑实现
    # ------------------------------
    
    def _update_similarity(self):
        ego_data = self.summary_state()
        for nid in self.cluster_state['neighbor_ids']:
            neighbor_data = common.global_vehicles.get(nid, None).summary_state()
            self.cluster_state['similarity_scores'][nid] = common.compute_spatiotemporal_similarity(
                ego_data, neighbor_data
            )

    def _update_cluster_membership(self):
        # 离开簇逻辑
        if self.cluster_state['head_id'] and not self.is_cluster_head():
            head_similarity = self.cluster_state['similarity_scores'].get(self.cluster_state['head_id'], 0)
            if head_similarity < self.params.get('eta_leave', 0.35):
                if self.params['ego_must_be_leader'] and self.cluster_state['head_id'] == common.global_ego_id:
                    if len(self.cluster_state['member_ids']) <= self.params['N_max']:
                        logger.debug(f"{self.vehicle_id} wanted to leave ego's cluster with similarity_score {head_similarity:.3f}, but stopped")
                        return
                self._leave_cluster()
                logger.debug(f"Vehicle {self.vehicle_id} left cluster of {self.cluster_state['head_id']} due to low similarity={head_similarity:.3f}")
        
        # 加入簇逻辑
        if not self.cluster_state['head_id']:
            self._join_cluster()
        
        # 创建簇逻辑
        if not self.cluster_state['head_id']:
            self._create_cluster()

    def _join_cluster(self):
        for nid in self.cluster_state['neighbor_ids']:                
            similarity = self.cluster_state['similarity_scores'][nid]
            adjusted_threshold = self.params.get('eta_join', 0.6)
            
            if similarity > adjusted_threshold:
                head = common.global_vehicles.get(nid, None)
                if head and len(head.cluster_state['member_ids']) < self.params.get('N_max', 4):
                    self.cluster_state['head_id'] = nid
                    head.cluster_state['member_ids'].add(self.vehicle_id)
                    logger.debug(f"Vehicle {self.vehicle_id} joined cluster of {nid} with similarity={similarity:.3f}, threshold={adjusted_threshold:.3f}")
                    break

    def _create_cluster(self):
        similarity_values = list(self.cluster_state['similarity_scores'].values())
        avg_similarity = sum(similarity_values)/len(similarity_values) if similarity_values else 0
        create_prob = self.calculate_create_probability(avg_similarity, self.params)
        
        if random.random() < create_prob:
            self.cluster_state['head_id'] = self.vehicle_id
            self.cluster_state['member_ids'] = {self.vehicle_id}
            logger.debug(f"Vehicle {self.vehicle_id} created new cluster with avg_similarity={avg_similarity:.3f}, prob={create_prob:.3f}")

    def _leave_cluster(self):
        head = common.global_vehicles.get(self.cluster_state['head_id'], None)
        if head:
            head.cluster_state['member_ids'].discard(self.vehicle_id)
        
        self.cluster_state['head_id'] = None
        self.cluster_state['member_ids'].clear()
        logger.debug(f"Vehicle {self.vehicle_id} left cluster of {self.cluster_state['head_id']}")

    def _elect_cluster_head(self):
        """簇头选举"""
        priority_list = []
        cluster_avg_speed = self._compute_cluster_avg_speed()
        for mid in self.cluster_state['member_ids']:
            member = common.global_vehicles.get(mid, None)
            if not member:
                continue
            # 触发成员计算自身优先级
            member._compute_local_priority(cluster_avg_speed)
            priority_list.append((member.cluster_state['local_priority'], mid))
        
        if priority_list:
            priority_list.sort(reverse=True)
            if priority_list[0][1] != self.cluster_state['head_id']: 
                self.cluster_state['head_id'] = priority_list[0][1]
                self._broadcast_cluster_state()  # 簇头变更, 立即同步全簇状态
                logger.debug(f"Vehicle {self.vehicle_id} elected new head {self.cluster_state['head_id']}(score:{priority_list[0][0]:.3f}) ")
            if len(priority_list) > 1 and priority_list[1][1] != self.cluster_state['shadow_head_id']:
                self.cluster_state['shadow_head_id'] = priority_list[1][1]
                logger.debug(f"Vehicle {self.vehicle_id} elected new shadow head {self.cluster_state['shadow_head_id']}(score:{priority_list[1][0]:.3f}) ")

    def _force_self_as_head(self):
        """强制设置ego为簇头并同步状态"""
        self.cluster_state['head_id'] = self.vehicle_id
        self.cluster_state['member_ids'] = {self.vehicle_id}

    def _handle_out_of_range_neighbor(self, vehicle_id):
        """处理超出范围的邻居"""
        if vehicle_id == self.cluster_state['head_id']:
            self._trigger_shadow_head_takeover()
        elif vehicle_id == self.cluster_state['shadow_head_id']:
            self.cluster_state['shadow_head_id'] = None
            if self.is_cluster_head():
                logger.debug(f"Vehicle {self.vehicle_id}: shadow head {vehicle_id} went out of range, triggering takeover.")
                self._elect_cluster_head()
        elif vehicle_id in self.cluster_state['member_ids']:
            self.cluster_state['member_ids'].discard(vehicle_id)
            logger.debug(f"Vehicle {self.vehicle_id}: removed out-of-range member {vehicle_id} from cluster.")

    def _trigger_shadow_head_takeover(self):
        """影子簇头接管流程"""
        logger.debug(f"Vehicle {self.vehicle_id} triggering shadow head takeover.")
        shadow_head_id = self.cluster_state['shadow_head_id']
        if not shadow_head_id:
            self.cluster_state['head_id'] = None
            logger.debug(f"Vehicle {self.vehicle_id} has no shadow head to takeover.")
            return
        
        shadow = common.global_vehicles.get(shadow_head_id, None)
        if shadow and (shadow_head_id in self.cluster_state['neighbor_ids'] or shadow_head_id == self.vehicle_id):
            self.cluster_state['head_id'] = shadow_head_id
            shadow._broadcast_cluster_state() #簇头变更, 立即同步全簇状态
            logger.debug(f"Vehicle {self.vehicle_id} shadow head {shadow_head_id} took over as new head.")
        else:
            self.cluster_state['head_id'] = None
            self.cluster_state['shadow_head_id'] = None
            logger.debug(f"Vehicle {self.vehicle_id} shadow head {shadow_head_id} is no longer valid.")

