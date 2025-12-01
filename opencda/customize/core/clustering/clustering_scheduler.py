from opencda.customize.core.v2x.scheduler import Scheduler
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import math
from opencda.customize.core.v2x.network_manager import NetworkManager, ResourceConflictError
import opencda.customize.core.v2x.utils as utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from opencda.log.logger_config import logger
from random import uniform
import numpy as np
from opencda.customize.core.v2x.utils import *

class WCGCScheduler(Scheduler):
    """
    Scheduler based on cluster-based weighted conflict graph coloring.
    """
    def __init__(self, cav_world, config={}):
        super().__init__(cav_world, config)

        self.subchannels = list(range(self.network_manager.subchannel_num))
        self.sinr_threshold = config.get('sinr_threshold', 6)  # dB
        self.max_power = config.get('max_power', 20)  # dBm
        self.link_weights: Dict[Tuple[int, int], float] = {}  # {(source_id, target_id): 权重}
        self.conflict_graph: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)  # 冲突图邻接表
        self.channel_allocation: Dict[Tuple[int, int], int] = {}  # {(source, target): 子信道}
        self.power_allocation: Dict[Tuple[int, int], float] = {}  # {(source, target): 功率(dBm)}

    def _collect_global_links(self) -> List[Tuple[int, int, float]]:
        """收集全网链路并计算权重"""
        global_links = []
        nm = self.network_manager
        
        for vehicle_id, vm in self.cav_world.get_vehicle_managers().items():
            cluster_state = vm.v2x_manager.cluster_state
            
            # 簇内上行链路（成员→簇头）
            if cluster_state['cluster_head'] is not None and vehicle_id == cluster_state['cluster_head']:
                # 收集簇头信息
                ch_id = cluster_state['cluster_head']
                ch_v2x = nm.vehicles.get(ch_id)
                if not ch_v2x:
                    continue
                for member_id, (data_vol, g) in cluster_state['members'].items():
                    member_v2x = nm.vehicles.get(member_id)
                    if not member_v2x:
                        continue
                    # 权重 = 数据量 × 感知贡献系数 × 可用数据率（链路质量）
                    distance = calculate_distance(member_v2x, ch_v2x)
                    channel_gain = calculate_channel_gain(distance)
                    tx_power_w = 10 ** (member_v2x.tx_power / 10) / 1000
                    sinr_linear = (tx_power_w * channel_gain) / ch_v2x.noise_power
                    data_rate = calculate_available_data_rate(self.config.get('subchannel_bandwidth', 1) * 1e6, sinr_linear)
                    weight = data_vol * g * (data_rate / 1e6)  # 归一化数据率
                    self.link_weights[(member_id, ch_id)] = weight
                    global_links.append((member_id, ch_id, weight))
            
                # 簇间下行广播链路（簇头→邻居）
                weight = self._calc_broadcast_weight(vm.v2x_manager)
                self.link_weights[(vehicle_id, -1)] = weight
                global_links.append((vehicle_id, -1, weight))
        
        return global_links
    
    def _calc_broadcast_weight(self, ch_v2x) -> float:
        """计算广播链路权重"""
        cluster_state = ch_v2x.cluster_state
        avg_g = np.mean([g for _, g in cluster_state['members'].values()]) if cluster_state['members'] else 1.0
        # 广播链路权重 = 簇头优先级 × 平均贡献系数 × 覆盖邻居数
        return cluster_state['priority_score'] * avg_g * len(cluster_state['neighbors'])

    def _build_conflict_graph(self, global_links: List[Tuple[int, int, float]]):
        """基于SINR阈值构建冲突图"""
        nm = self.network_manager
        self.conflict_graph.clear()
        
        for i, (s1_id, t1_id, _) in enumerate(global_links):
            s1_v2x = nm.vehicles.get(s1_id)
            t1_v2x = nm.vehicles.get(t1_id) if t1_id != -1 else s1_v2x  # 广播目标为自身
            if not s1_v2x or not t1_v2x:
                continue
            
            for j, (s2_id, t2_id, _) in enumerate(global_links):
                if i >= j or (s1_id == s2_id and t1_id == t2_id):
                    continue
                s2_v2x = nm.vehicles.get(s2_id)
                t2_v2x = nm.vehicles.get(t2_id) if t2_id != -1 else s2_v2x
                if not s2_v2x or not t2_v2x:
                    continue
                
                # 判断链路(s1→t1)与(s2→t2)是否冲突
                if is_link_conflict(s1_v2x, t1_v2x, s2_v2x, t2_v2x, self.sinr_threshold_dB):
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

    def _power_control(self):
        """功率控制：满足SINR阈值的最小功率"""
        nm = self.network_manager
        self.power_allocation.clear()
        
        for (s_id, t_id), ch in self.channel_allocation.items():
            if ch == -1:
                continue
            s_v2x = nm.vehicles.get(s_id)
            t_v2x = nm.vehicles.get(t_id) if t_id != -1 else s_v2x
            if not s_v2x or not t_v2x:
                continue
            
            # 计算同信道干扰总和
            interf_total = 0.0
            for (s2_id, t2_id), ch2 in self.channel_allocation.items():
                if ch2 != ch or (s2_id == s_id and t2_id == t_id):
                    continue
                s2_v2x = nm.vehicles.get(s2_id)
                if not s2_v2x:
                    continue
                interf_total += get_interference_contribution(s2_v2x, t_v2x)
            
            # 计算所需最小发射功率（线性域）
            sinr_linear_threshold = 10 ** (self.sinr_threshold_dB / 10)
            required_signal_power = sinr_linear_threshold * (interf_total + t_v2x.noise_power)
            distance = calculate_distance(s_v2x, t_v2x)
            channel_gain = calculate_channel_gain(distance, s_v2x.path_loss_exponent)
            min_tx_power_w = required_signal_power / channel_gain if channel_gain > 0 else s_v2x.tx_power
            
            # 限制最大功率
            max_tx_power_w = 10 ** (s_v2x.tx_power / 10) / 1000
            self.power_allocation[(s_id, t_id)] = min(min_tx_power_w, max_tx_power_w)

    def schedule(self, source, target, volume: float) -> bool:
        """执行调度并调用NS3接口"""
        # 全局链路收集与冲突图构建
        global_links = self._collect_global_links()
        self._build_conflict_graph(global_links)
        self._weighted_graph_coloring()
        self._power_control()
        
        # 获取当前链路分配结果
        link = (source.id, target.id if target else -1)
        if link not in self.channel_allocation or self.channel_allocation[link] == -1:
            return False
        
        # 调用NS3通信接口（转换功率为dBm）
        ch = self.channel_allocation[link]
        tx_power_w = self.power_allocation.get(link, 10 ** (source.tx_power / 10) / 1000)
        tx_power_dBm = 10 * math.log10(tx_power_w * 1000) if tx_power_w > 0 else source.tx_power
        
        success = self.network_manager.communicate(
            self, source, target, volume,
            subchannel_start=ch, subchannel_num=1
        )
        return success