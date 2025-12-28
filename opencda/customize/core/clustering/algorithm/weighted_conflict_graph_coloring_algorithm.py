from typing import Dict, Set, Tuple, List
from opencda.customize.core.clustering.clustering_algorithm_manager import ClusterResourceAllocationAlgorithm
from opencda.customize.core.clustering.algorithm.common import *
import opencda.customize.core.clustering.algorithm.common as common

class WCGCAlgorithm(ClusterResourceAllocationAlgorithm):
    """
    Scheduler based on cluster-based weighted conflict graph coloring.
    """    
    def __init__(self, cav_world, config={}):
        super().__init__(cav_world)

        self.subchannels = list(range(self.network_manager.subchannel_num))
        self.sinr_threshold = config.get('sinr_threshold', 6)  # dB
        self.max_power = config.get('max_power', 20)  # dBm

        self.function_cp_value = self.f
        self.link_weights: Dict[Tuple[int, int], float] = {}  # {(source_id, target_id): 权重}
        self.conflict_graph: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)  # 冲突图邻接表
        self.channel_allocation: Dict[Tuple[int, int], int] = {}  # {(source, target): 子信道}
        self.power_allocation: Dict[Tuple[int, int], float] = {}  # {(source, target): 功率(dBm)}

        Vehicle_Grid.initialize_world(cav_world)
        Vehicle_Grid.initialize_vehicles(cav_world)

    def _collect_global_links(self) -> List[Tuple[int, int, float]]:
        """收集全网链路并计算权重"""
        global_links = []
        for cluster in self.clusters:
            hid = cluster.head_id
            for mid in cluster.members:
                # 簇内上行链路（成员→簇头）
                weight = self._calc_uplink_weight(mid, hid)
                self.link_weights[(mid, hid)] = weight
                global_links.append((mid, hid, weight))
            # # 簇间下行广播链路（簇头→邻居）暂不考虑
        return global_links
    
    def _calc_broadcast_weight(self, cluster_head_id) -> float:
        """下行链路权重: w_l = f_u( S_{i, t}^{tot\_u}, \bar \delta t)"""
        return len(common.global_vehicles.get(cluster_head_id).req_grids)
    
    def _calc_uplink_weight(self, member_id, cluster_head_id) -> float:
        """计算上行链路权重（成员→簇头）"""
        # 权重 = 数据量 x 感知贡献 x 归一化
        req_grids = common.global_vehicles.get(cluster_head_id).req_grids
        ret = avg_grids_score(member_id, common.global_vehicles.get(member_id).sens_grids & req_grids)
        print(f"Calculating uplink weight for member {member_id} to cluster head {cluster_head_id}: req_grids={len(req_grids)}, contribution={ret}")
        return ret
    
    def _build_conflict_graph(self, global_links: List[Tuple[int, int, float]]):
        """基于SINR阈值构建冲突图"""
        self.conflict_graph.clear()
        
        for i, (s1_id, t1_id, _) in enumerate(global_links):
            s1_v2x = common.global_vms.get(s1_id)
            t1_v2x = common.global_vms.get(t1_id) if t1_id != -1 else s1_v2x  # 广播目标为自身
            if not s1_v2x or not t1_v2x:
                continue
            
            for j, (s2_id, t2_id, _) in enumerate(global_links):
                if i >= j or (s1_id == s2_id and t1_id == t2_id):
                    continue
                s2_v2x = common.global_vms.get(s2_id)
                t2_v2x = common.global_vms.get(t2_id) if t2_id != -1 else s2_v2x
                if not s2_v2x or not t2_v2x:
                    continue
                
                # 判断链路(s1→t1)与(s2→t2)是否冲突
                if is_link_conflict(s1_v2x, t1_v2x, s2_v2x, t2_v2x, self.config['min_sinr_threshold']):
                    self.conflict_graph[(s1_id, t1_id)].add((s2_id, t2_id))
                    self.conflict_graph[(s2_id, t2_id)].add((s1_id, t1_id))

    def _weighted_graph_coloring(self):
        """加权冲突图着色"""
        sorted_links = sorted(self.link_weights.keys(), key=lambda x: self.link_weights[x], reverse=True)
        self.channel_allocation.clear()
        
        for link in sorted_links:
            neighbor_channels = set()
            for neighbor_link in self.conflict_graph[link]:
                if neighbor_link in self.channel_allocation:
                    neighbor_channels.add(self.channel_allocation[neighbor_link])
            available_channels = [ch for ch in self.subchannels if ch not in neighbor_channels]
            self.channel_allocation[link] = available_channels[0] if available_channels else -1

    def run(self):
        global_links = self._collect_global_links()
        self._build_conflict_graph(global_links)
        self._weighted_graph_coloring()
        # self._power_control()
        self.update_resource_allocation_strategy()
    
    def update_resource_allocation_strategy(self):
        for hid, vm in common.global_vms.items():
            vm.scheduler.set_strategies(self.channel_allocation)

    # ------------------------------
    # 功率控制
    # ------------------------------
    def _power_control(self):
        """功率控制：满足SINR阈值的最小功率"""
        self.power_allocation.clear()
        
        for (s_id, t_id), ch in self.channel_allocation.items():
            if ch == -1:
                continue
            s_v2x = common.global_vms.get(s_id)
            t_v2x = common.global_vms.get(t_id) if t_id != -1 else s_v2x
            if not s_v2x or not t_v2x:
                continue
            
            # 计算同信道干扰总和
            interf_total = 0.0
            for (s2_id, t2_id), ch2 in self.channel_allocation.items():
                if ch2 != ch or (s2_id == s_id and t2_id == t_id):
                    continue
                s2_v2x = common.global_vms.get(s2_id)
                if not s2_v2x:
                    continue
                interf_total += get_interference_contribution(s2_v2x, t_v2x)
            
            # 计算所需最小发射功率（线性域）
            sinr_linear_threshold = 10 ** (self.config['min_sinr_threshold'] / 10)
            required_signal_power = sinr_linear_threshold * (interf_total + t_v2x.noise_power)
            distance = calculate_distance(s_v2x, t_v2x)
            channel_gain = calculate_channel_gain(distance)
            min_tx_power_w = required_signal_power / channel_gain if channel_gain > 0 else s_v2x.tx_power
            
            # 限制最大功率
            max_tx_power_w = 10 ** (s_v2x.tx_power / 10) / 1000
            self.power_allocation[(s_id, t_id)] = min(min_tx_power_w, max_tx_power_w)