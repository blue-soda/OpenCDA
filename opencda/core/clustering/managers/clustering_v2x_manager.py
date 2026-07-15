from opencda.core.common.cav_world import CavWorld
from opencda.core.common.v2x_manager import V2XManager
from opencda.core.common.config_manager import ConfigManager, ClusteringConfig
import math
from opencda.core.common.misc import compute_distance
import random
from opencda.log.logger_config import logger
# from pympler.asizeof import asizeof
from sys import getsizeof as asizeof
import weakref

from opencda.core.clustering.algorithms.clustering.coalition_game import CoalitionGame \
    as CoalitionGame
from opencda.core.clustering.algorithms.clustering.naive_cluster import NaiveCluster

class ClusteringV2XManager(V2XManager):
    cluster_algorithm = None
    clusters = None
    all_clusters = []
    accepted_topology_signature = None
    skipped_topology_checks = 0

    def __init__(self, cav_world, config_yaml, vid):
        super(ClusteringV2XManager, self).__init__(cav_world, config_yaml, vid)
        self.cp_model = 'default_model'

        # Load clustering configuration
        if 'config_path' in config_yaml:
            config_mgr = ConfigManager.from_yaml(config_yaml['config_path'])
            clustering_config = config_mgr.clustering
        else:
            clustering_config = ClusteringConfig()

        self.cluster_interval = clustering_config.cluster_interval
        self.enable_topology_trigger_gate = (
            clustering_config.enable_topology_trigger_gate)
        self.topology_periodic_guard = (
            clustering_config.topology_periodic_guard)
        self.cnt = self.cluster_interval

        # ------------------------------
        # 分簇协议状态
        # ------------------------------
        self.cluster_state = {
            'head_id': None,               # 当前簇头ID
            'member_ids': set(),           # 簇成员ID集合
        }
        if ClusteringV2XManager.cluster_algorithm is None:
            ClusteringV2XManager.cluster_algorithm = CoalitionGame(self.cav_world)
            # ClusteringV2XManager.cluster_algorithm = NaiveCluster(self.cav_world, all_in_one=False)

    @staticmethod
    def _location_xy(transform):
        return (transform.location.x, transform.location.y)

    @staticmethod
    def _distance_xy(a_xy, b_xy):
        dx = a_xy[0] - b_xy[0]
        dy = a_xy[1] - b_xy[1]
        return math.sqrt(dx * dx + dy * dy)

    def _topology_signature(self):
        vehicle_manager_dict = self.cav_world.get_vehicle_managers()
        vehicle_positions = {}
        communication_ranges = {}
        for vid, vm in vehicle_manager_dict.items():
            v2x_manager = getattr(vm, 'v2x_manager', None)
            if v2x_manager is None:
                continue
            ego_pos = v2x_manager.get_ego_pos()
            if ego_pos is None:
                continue
            vehicle_positions[vid] = self._location_xy(ego_pos)
            communication_ranges[vid] = getattr(
                v2x_manager,
                'communication_range',
                self.communication_range)

        neighbor_items = []
        for vid in sorted(vehicle_positions.keys()):
            neighbors = []
            for other_id in sorted(vehicle_positions.keys()):
                if other_id == vid:
                    continue
                distance = self._distance_xy(
                    vehicle_positions[vid],
                    vehicle_positions[other_id])
                if distance <= communication_ranges.get(
                        vid,
                        self.communication_range):
                    neighbors.append(other_id)
            neighbor_items.append((vid, tuple(neighbors)))
        return tuple(neighbor_items)

    def _has_head_member_failure(self):
        vehicle_manager_dict = self.cav_world.get_vehicle_managers()
        for cluster in ClusteringV2XManager.all_clusters or []:
            head_id = getattr(cluster, 'head_id', None)
            head_vm = vehicle_manager_dict.get(head_id)
            if head_vm is None or not hasattr(head_vm, 'v2x_manager'):
                return True
            head_pos = head_vm.v2x_manager.get_ego_pos()
            if head_pos is None:
                return True
            for member_id in getattr(cluster, 'members', set()):
                member_vm = vehicle_manager_dict.get(member_id)
                if member_vm is None or not hasattr(member_vm, 'v2x_manager'):
                    return True
                member_pos = member_vm.v2x_manager.get_ego_pos()
                if member_pos is None:
                    return True
                distance = compute_distance(
                    head_pos.location,
                    member_pos.location)
                communication_range = getattr(
                    head_vm.v2x_manager,
                    'communication_range',
                    self.communication_range)
                if distance > communication_range:
                    return True
        return False

    def _should_recluster(self):
        if not self.enable_topology_trigger_gate:
            return True, 'periodic'
        if self.clusters is None and not ClusteringV2XManager.all_clusters:
            return True, 'initial'

        topology_signature = self._topology_signature()
        accepted_signature = ClusteringV2XManager.accepted_topology_signature
        if accepted_signature is None:
            ClusteringV2XManager.accepted_topology_signature = topology_signature
            return True, 'initial_topology'
        if topology_signature != accepted_signature:
            return True, 'neighbor_set_change'
        if self._has_head_member_failure():
            return True, 'head_member_unreachable'

        guard = int(self.topology_periodic_guard or 0)
        if guard > 0 and \
                ClusteringV2XManager.skipped_topology_checks >= guard:
            return True, 'periodic_guard'

        return False, 'no_topology_change'

    def _sync_cluster_states(self, clusters):
        clusters = list(clusters or [])
        ClusteringV2XManager.all_clusters = clusters
        self.all_clusters = clusters

        vehicle_manager_dict = self.cav_world.get_vehicle_managers()
        for _, vm in vehicle_manager_dict.items():
            v2x_manager = getattr(vm, 'v2x_manager', None)
            if v2x_manager and hasattr(v2x_manager, 'cluster_state'):
                v2x_manager.cluster_state['head_id'] = None
                v2x_manager.cluster_state['member_ids'] = set()

        for cluster in clusters:
            head_id = getattr(cluster, 'head_id', None)
            members = set(getattr(cluster, 'members', set()))
            for member_id in members:
                vm = vehicle_manager_dict.get(member_id)
                if vm is None or not hasattr(vm, 'v2x_manager'):
                    continue
                vm.v2x_manager.cluster_state['head_id'] = head_id
                vm.v2x_manager.cluster_state['member_ids'] = members.copy()

        if clusters:
            logger.info(
                "CLUSTER_SYNC %s",
                [(getattr(cluster, 'head_id', None), sorted(list(getattr(cluster, 'members', set()))))
                 for cluster in clusters]
            )
        else:
            logger.info("CLUSTER_SYNC []")

    def run_algorithm(self):
        self.cluster_algorithm.initialize()
        self.cnt += 1
        if self.cnt >= self.cluster_interval:
            should_recluster, reason = self._should_recluster()
            if should_recluster:
                self.clusters = self.cluster_algorithm.run()
                self._sync_cluster_states(self.clusters)
                ClusteringV2XManager.accepted_topology_signature = \
                    self._topology_signature()
                ClusteringV2XManager.skipped_topology_checks = 0
                logger.info("CLUSTER_TRIGGER recluster reason=%s", reason)
            else:
                ClusteringV2XManager.skipped_topology_checks += 1
                logger.info("CLUSTER_TRIGGER skip reason=%s", reason)
                if self.clusters is None and ClusteringV2XManager.all_clusters:
                    self.clusters = list(ClusteringV2XManager.all_clusters)
            self.cnt = 0
        elif self.clusters is None and ClusteringV2XManager.all_clusters:
            self.clusters = list(ClusteringV2XManager.all_clusters)
        if self.enable_scheduler and hasattr(self.scheduler, 'is_cluster_based') and self.scheduler.is_cluster_based:
            self.scheduler.set_clusters(self.clusters)
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
        # print(f"self.cluster_state: {self.cluster_state}")
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
