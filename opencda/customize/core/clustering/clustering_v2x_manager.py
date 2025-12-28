from opencda.core.common.cav_world import CavWorld
from opencda.core.common.v2x_manager import V2XManager
import math
from opencda.core.common.misc import compute_distance
import random
from opencda.log.logger_config import logger
# from pympler.asizeof import asizeof
from sys import getsizeof as asizeof
import weakref

from opencda.customize.core.clustering.algorithm.coalition_game import CoalitionGame \
    as ClusteringAlgorithm

class ClusteringV2XManager(V2XManager):
    cluster_algorithm = None
    def __init__(self, cav_world, config_yaml, vid):
        super(ClusteringV2XManager, self).__init__(cav_world, config_yaml, vid)
        self.cp_model = 'default_model'
        # ------------------------------
        # 分簇协议状态
        # ------------------------------
        self.cluster_state = {
            'head_id': None,               # 当前簇头ID
            'member_ids': set(),           # 簇成员ID集合
        }
        if ClusteringV2XManager.cluster_algorithm is None:
            ClusteringV2XManager.cluster_algorithm = ClusteringAlgorithm(self.cav_world)

    def run_algorithm(self):
        self.cluster_algorithm.initialize()
        clusters = self.cluster_algorithm.run()
        if self.enable_scheduler and hasattr(self.scheduler, 'is_cluster_based') and self.scheduler.is_cluster_based:
            self.scheduler.set_clusters(clusters)
            self.scheduler.run()

    @staticmethod
    def set_enable_scheduler(enable_scheduler):
        ClusteringV2XManager.enable_scheduler = enable_scheduler
    # ------------------------------
    # 信标数据结构
    # ------------------------------
    def beacon(self):
        """标准化信标数据结构"""
        return {
            'vehicle_id': self.vehicle_id,
            'position': self.get_ego_pos().location,
            'speed': self.get_ego_speed(),
            'direction': self.get_ego_dir(),
            'computing_capability': getattr(self, 'computing_capability', 1.0),
            'communication_quality': getattr(self, 'communication_quality', 1.0),
            'perception_model': self.cp_model,
            'cluster_head_id': self.cluster_state['head_id'],
            'cluster_member_ids': list(self.cluster_state['member_ids']),
        }


    # ------------------------------
    # 对外接口: 簇状态查询
    # ------------------------------
    def is_cluster_head(self):
        return self.cluster_state['head_id'] == self.vehicle_id

    def get_cluster_head(self):
        return self.cluster_state['head_id']

    def get_cluster_member_ids(self):
        return self.cluster_state['member_ids'].copy()
    
    def get_cluster_member_vms(self):
        """获取簇成员Vehicle_Manager实例列表"""
        print(f"self.cluster_state: {self.cluster_state}")
        # Resolve cluster head
        cluster_head_id = self.cluster_state.get('head_id', None)
        cluster_head_vm = self.cav_world.get_vehicle_manager(cluster_head_id) if cluster_head_id is not None else None

        # Resolve members
        members_vm = {}
        for vehicle_id in self.cluster_state.get('member_ids', {}):
            vm = self.cav_world.get_vehicle_manager(vehicle_id)
            members_vm[vehicle_id] = vm

        return {
            'cluster_head': cluster_head_vm,
            'members': members_vm,
        }
    
    def get_uplink_links(self):
        if not self.is_cluster_head():
            return []
        return [(mid, self.vehicle_id) for mid in self.cluster_state['member_ids'] if mid != self.vehicle_id]
    
    def get_downlink_links(self):
        if not self.is_cluster_head():
            return []
        return [(self.vehicle_id, mid) for mid in self.cluster_state['member_ids'] if mid != self.vehicle_id]
    
    def get_cluster_state(self):
        return self.cluster_state
    
    # ------------------------------
    # 对外接口: 簇状态更新
    # ------------------------------

    def search(self):
        super().search()
    
    def update_rgb(self):
        """更新簇颜色标识"""
        if not self.cluster_state['head_id']:
            self.rgb = (255, 255, 0)
            return
        hash_value = hash(f"vehicle_{self.cluster_state['head_id']}") & 0xFFFFFF
        r = int((hash_value >> 16) & 0xFF)
        g = int((hash_value >> 8) & 0xFF)
        b = int(hash_value & 0xFF)
        r = 255 if r > 127 else 0
        g = 255 if g > 127 else 0
        b = 255 if b > 127 else 0
        # print(f"Vehicle {self.vehicle_id} cluster_state: {self.cluster_state} rgb: {r}, {g}, {b}")
        self.rgb = (r, g, b)