from opencda.core.common.cav_world import CavWorld
from opencda.core.common.v2x_manager import V2XManager
import math
from opencda.core.common.misc import compute_distance
import random
from opencda.log.logger_config import logger
# from pympler.asizeof import asizeof
from sys import getsizeof as asizeof
from opencda.customize.core.clustering.cluster_algorithm import ClusterAlgorithm
import weakref

# 1. 簇内聚类分工，分担计算任务
# 2. 感知范围： LOS邻居并集最大
# 3. 簇内聚类做感知融合，多个聚类上传给簇头晚期融合（簇间再融合）

class ClusteringV2XManager(V2XManager):
    def __init__(self, cav_world, config_yaml, vid, vehicle_id=None, cluster_yaml=None):
        super(ClusteringV2XManager, self).__init__(cav_world, config_yaml, vid, vehicle_id)
        
        self.cp_model = 'default_model'
        self.params = cluster_yaml
        self.ego_must_be_leader = self.params.get('ego_must_be_leader', False)
        # 分簇协议状态
        self.cluster_state = {
            'head_id': None,               # 当前簇头ID
            'shadow_head_id': None,        # 影子簇头ID
            'member_ids': set(),           # 簇成员ID集合
            'neighbor_ids': set(),         # 邻居ID集合
            'neighbor_data': {},           # 邻居数据缓存 {neighbor_id: data}
            'similarity_scores': {},       # 邻居相似度 {neighbor_id: score}
            'priority_score': 0.0          # 本地优先级得分
        }

     # ------------------------------
    # 对外接口
    # ------------------------------
    def is_cluster_head(self):
        return self.cluster_state['head_id'] == self.vehicle_id

    def get_cluster_head(self):
        return self.cluster_state['head_id']

    def get_cluster_member_ids(self):
        return self.cluster_state['member_ids'].copy()
    
    def get_cluster_member_vms(self):
        """获取簇成员Vehicle_Manager实例列表"""
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
        return {
            'head_id': self.cluster_state['head_id'],
            'shadow_head_id': self.cluster_state['shadow_head_id'],
            'member_ids': self.cluster_state['member_ids'].copy(),
            'is_head': self.is_cluster_head()
        }

    # ------------------------------
    # 核心功能实现
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
            'cluster_shadow_id': self.cluster_state['shadow_head_id'],
            'cluster_member_ids': list(self.cluster_state['member_ids']),
            'RSSI': self._calc_rssi()
        }

    def search(self):
        """唯一更新邻居数据的入口"""
        vehicle_manager_dict = self.cav_world.get_vehicle_managers()
        self.cluster_state['neighbor_ids'].clear()
        self.cluster_state['neighbor_data'].clear()

        # 强制ego为簇头
        if self.ego_must_be_leader and self.vehicle_id == self.cav_world.ego_id:
            self._force_ego_as_head()

        for vid, vm in vehicle_manager_dict.items():
            vehicle_id = vm.vehicle.id
            if vehicle_id == self.vehicle_id or not vm.v2x_manager.get_ego_pos() or not vm.is_ok:
                continue
            
            distance = compute_distance(self.get_ego_pos().location, vm.v2x_manager.get_ego_pos().location)
            if distance < self.communication_range:
                # 唯一更新neighbor_data的位置
                neighbor_beacon = vm.v2x_manager.beacon()
                self.cluster_state['neighbor_ids'].add(vehicle_id)
                self.cluster_state['neighbor_data'][vehicle_id] = neighbor_beacon
                # 更新附近车辆缓存
                self.cav_nearby[vid] = {
                    'vehicle_manager': weakref.ref(vm)(),
                    'v2x_manager': weakref.ref(vm.v2x_manager)()
                }
                # 通信统计
                if self.cav_world.network_manager:
                    self.cav_world.network_manager._update_communication_stats(asizeof(neighbor_beacon), "control")
            else:
                self.cav_nearby.pop(vid, None)
                self._handle_out_of_range_neighbor(vehicle_id)

        # 执行分簇逻辑
        self._update_cluster_logic()

    def _update_cluster_logic(self):
        """分簇核心逻辑（依赖标准化的neighbor_data）"""
        self._update_similarity()
        self._update_cluster_membership()
        self._sync_cluster_state_with_neighbors()

        if self.is_cluster_head():
            self._compute_local_priority()  # 仅簇头选举时计算优先级
            self._elect_cluster_head()
            self._broadcast_cluster_state()

        self._update_rgb()

    # ------------------------------
    # 数据一致性保障
    # ------------------------------
    def _sync_cluster_state_with_neighbors(self):
        """与邻居同步簇状态，确保全簇一致"""
        if not self.cluster_state['head_id']:
            return
        
        # 获取簇头的最新状态
        head_vm = self._get_v2x_manager(self.cluster_state['head_id'])
        if not head_vm or (self.cluster_state['head_id'] not in self.cluster_state['neighbor_ids'] and not self.is_cluster_head()):
            self._trigger_shadow_head_takeover()
            return
        
        # 同步簇头的状态
        head_state = head_vm.get_cluster_state()
        self.cluster_state['head_id'] = head_state['head_id']
        self.cluster_state['shadow_head_id'] = head_state['shadow_head_id']
        self.cluster_state['member_ids'] = set(head_state['member_ids'])

    def _broadcast_cluster_state(self):
        """簇头广播最新状态，强制同步所有成员"""
        for mid in self.cluster_state['member_ids']:
            member_vm = self._get_v2x_manager(mid)
            if member_vm and member_vm.vehicle_id != self.vehicle_id:
                member_vm._update_cluster_state({
                    'head_id': self.cluster_state['head_id'],
                    'shadow_head_id': self.cluster_state['shadow_head_id'],
                    'member_ids': self.cluster_state['member_ids'].copy()
                })

    def _update_cluster_state(self, new_state):
        """成员接收簇头的状态更新"""
        self.cluster_state['head_id'] = new_state['head_id']
        self.cluster_state['shadow_head_id'] = new_state['shadow_head_id']
        self.cluster_state['member_ids'] = set(new_state['member_ids'])
        # logger.debug(f"Vehicle {self.vehicle_id} synced cluster state: head={new_state['head_id']}, members={len(new_state['member_ids'])}")

    # ------------------------------
    # 优先级计算
    # ------------------------------
    def _compute_local_priority(self):
        """仅在簇头选举时计算优先级（唯一计算入口）"""
        member_speeds = []
        for mid in self.cluster_state['member_ids']:
            member_vm = self._get_v2x_manager(mid)
            if member_vm:
                member_speeds.append(member_vm.get_ego_speed())
        
        cluster_avg_speed = sum(member_speeds) / len(member_speeds) if member_speeds else self.get_ego_speed()
        self.cluster_state['local_priority'] = ClusterAlgorithm.compute_priority_score({
            'speed': self.get_ego_speed(),
            'communication_quality': getattr(self, 'communication_quality', 1.0),
            'computing_capability': getattr(self, 'computing_capability', 1.0)
        }, cluster_avg_speed, self.params)

    # ------------------------------
    # 分簇逻辑实现
    # ------------------------------
    def _update_similarity(self):
        ego_data = {
            'position': self.get_ego_pos().location,
            'speed': self.get_ego_speed(),
            'direction': self.get_ego_dir(),
            'perception_model': self.cp_model,
            'neighbor_count': len(self.cluster_state['neighbor_ids'])
        }
        
        for nid in self.cluster_state['neighbor_ids']:
            neighbor_data = self.cluster_state['neighbor_data'][nid]
            self.cluster_state['similarity_scores'][nid] = ClusterAlgorithm.compute_spatiotemporal_similarity(
                ego_data, neighbor_data, self.params
            )

    def _update_cluster_membership(self):
        if self.ego_must_be_leader and self.vehicle_id == self.cav_world.ego_id:
            return
        
        # 离开簇逻辑
        if self.cluster_state['head_id'] and not self.is_cluster_head():
            head_similarity = self.cluster_state['similarity_scores'].get(self.cluster_state['head_id'], 0)
            if head_similarity < self.params.get('eta_leave', 0.35):
                if self.ego_must_be_leader and self.cluster_state['head_id'] == self.cav_world.ego_id:
                    if len(self.cluster_state['member_ids']) <= self.params['N_max']:
                        logger.debug(f"{self.vehicle_id} wanted to leave ego's cluster with similarity_score {head_similarity}, but stopped")
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
            neighbor_data = self.cluster_state['neighbor_data'][nid]
            if neighbor_data['cluster_head_id'] != nid:
                continue
            
            similarity = self.cluster_state['similarity_scores'][nid]
            adjusted_threshold = self.params.get('eta_join', 0.6) * (1 + neighbor_data['RSSI'] / self.params.get('RSSI_max', 100))
            
            if similarity > adjusted_threshold:
                head_vm = self._get_v2x_manager(nid)
                if head_vm and len(head_vm.cluster_state['member_ids']) < self.params.get('N_max', 4):
                    self.cluster_state['head_id'] = nid
                    head_vm.cluster_state['member_ids'].add(self.vehicle_id)
                    head_vm._broadcast_cluster_state()  # 立即同步状态
                    logger.debug(f"Vehicle {self.vehicle_id} joined cluster of {nid} with similarity={similarity:.3f}, threshold={adjusted_threshold:.3f}")
                    break

    def _create_cluster(self):
        similarity_values = list(self.cluster_state['similarity_scores'].values())
        avg_similarity = sum(similarity_values)/len(similarity_values) if similarity_values else 0
        create_prob = ClusterAlgorithm.calculate_create_probability(avg_similarity, self.params)
        
        if random.random() < create_prob:
            self.cluster_state['head_id'] = self.vehicle_id
            self.cluster_state['member_ids'] = {self.vehicle_id}
            self._broadcast_cluster_state()
            logger.debug(f"Vehicle {self.vehicle_id} created new cluster with avg_similarity={avg_similarity:.3f}, prob={create_prob:.3f}")

    def _leave_cluster(self):
        head_vm = self._get_v2x_manager(self.cluster_state['head_id'])
        if head_vm:
            head_vm.cluster_state['member_ids'].discard(self.vehicle_id)
            head_vm._broadcast_cluster_state()
        
        self.cluster_state['head_id'] = None
        self.cluster_state['member_ids'].clear()
        logger.debug(f"Vehicle {self.vehicle_id} left cluster of {self.cluster_state['head_id']}")

    def _elect_cluster_head(self):
        """簇头选举"""
        priority_list = []
        for mid in self.cluster_state['member_ids']:
            member_vm = self._get_v2x_manager(mid)
            if not member_vm:
                continue
            
            # 触发成员计算自身优先级
            member_vm._compute_local_priority()
            priority_list.append((member_vm.cluster_state['local_priority'], mid))
        
        if priority_list:
            priority_list.sort(reverse=True)
            self.cluster_state['head_id'] = priority_list[0][1]
            self.cluster_state['shadow_head_id'] = priority_list[1][1] if len(priority_list) > 1 else None
            self._broadcast_cluster_state()  # 立即同步全簇状态
            logger.debug(f"Vehicle {self.vehicle_id} elected new head {self.cluster_state['head_id']}(score:{priority_list[0][0]}) "
                         f"with shadow {self.cluster_state['shadow_head_id'] if self.cluster_state['shadow_head_id'] else 'None'}(score:{priority_list[1][0] if len(priority_list) > 1 else 'N/A'})")

    # ------------------------------
    # 辅助方法
    # ------------------------------
    def _force_ego_as_head(self):
        """强制设置ego为簇头并同步状态"""
        self.cluster_state['head_id'] = self.vehicle_id
        self.cluster_state['member_ids'] = {self.vehicle_id}
        self._broadcast_cluster_state()

    def _handle_out_of_range_neighbor(self, vehicle_id):
        """处理超出范围的邻居"""
        if vehicle_id == self.cluster_state['head_id']:
            self._trigger_shadow_head_takeover()
        elif vehicle_id == self.cluster_state['shadow_head_id']:
            self.cluster_state['shadow_head_id'] = None
            if self.is_cluster_head():
                self._elect_cluster_head()
        elif vehicle_id in self.cluster_state['member_ids']:
            self.cluster_state['member_ids'].discard(vehicle_id)
            if self.is_cluster_head():
                self._broadcast_cluster_state()
            logger.debug(f"Vehicle {self.vehicle_id} removed out-of-range member {vehicle_id} from cluster.")

    def _trigger_shadow_head_takeover(self):
        """影子簇头接管流程"""
        logger.debug(f"Vehicle {self.vehicle_id} triggering shadow head takeover.")
        shadow_head_id = self.cluster_state['shadow_head_id']
        if not shadow_head_id:
            self.cluster_state['head_id'] = None
            logger.debug(f"Vehicle {self.vehicle_id} has no shadow head to takeover.")
            return
        
        shadow_vm = self._get_v2x_manager(shadow_head_id)
        if shadow_vm and shadow_head_id in self.cluster_state['neighbor_ids']:
            self.cluster_state['head_id'] = shadow_head_id
            shadow_vm._broadcast_cluster_state()
            logger.debug(f"Vehicle {self.vehicle_id} shadow head {shadow_head_id} took over as new head.")
        else:
            self.cluster_state['head_id'] = None
            self.cluster_state['shadow_head_id'] = None
            logger.debug(f"Vehicle {self.vehicle_id} shadow head {shadow_head_id} is no longer valid.")

    def _calc_rssi(self):
        """简化的RSSI计算"""
        return 0  # 实际应根据距离和通信质量计算

    def _get_v2x_manager(self, vehicle_id):
        vm = self.cav_world.get_vehicle_manager(vehicle_id)
        return vm.v2x_manager if vm else None
    
    def _update_rgb(self):
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
        self.rgb = (r, g, b)