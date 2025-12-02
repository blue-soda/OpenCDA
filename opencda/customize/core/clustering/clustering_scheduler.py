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
    def f(v: float, t: float) -> float:
        """感知贡献函数 f(x) = 1 - e^(-x)"""
        return (1 - math.exp(-v / 1e6)) * math.exp(-t / 0.1)
    
    def __init__(self, cav_world, config={}):
        super().__init__(cav_world, config)

        self.subchannels = list(range(self.network_manager.subchannel_num))
        self.sinr_threshold = config.get('sinr_threshold', 6)  # dB
        self.max_power = config.get('max_power', 20)  # dBm

        self.function_cp_value = self.f
        self.link_weights: Dict[Tuple[int, int], float] = {}  # {(source_id, target_id): 权重}
        self.conflict_graph: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)  # 冲突图邻接表
        self.channel_allocation: Dict[Tuple[int, int], int] = {}  # {(source, target): 子信道}
        self.power_allocation: Dict[Tuple[int, int], float] = {}  # {(source, target): 功率(dBm)}

    def _collect_global_links(self) -> List[Tuple[int, int, float]]:
        """收集全网链路并计算权重"""
        global_links = []
        nm = self.network_manager
        
        for vehicle_id, vm in self.cav_world.get_vehicle_managers().items():
            if vm.v2x_manager.is_cluster_head():
                head_v2x = vm.v2x_manager
                # 簇内上行链路（成员→簇头）
                for source_id, target_id in head_v2x.get_uplink_links():
                    weight = self._calc_uplink_weight(source_id, target_id)
                    self.link_weights[(source_id, target_id)] = weight
                    global_links.append((source_id, target_id, weight))
                # 簇间下行广播链路（簇头→邻居）, 暂不考虑
                # for source_id, target_id in head_v2x.get_downlink_links():
                #     weight = self._calc_broadcast_weight(source_id)
                #     self.link_weights[(source_id, target_id)] = weight
                #     global_links.append((source_id, target_id, weight))
        
        return global_links
    
    def _calc_broadcast_weight(self, cluster_head_id) -> float:
        """下行链路权重: w_l = f_u( S_{i, t}^{tot\_u}, \bar \delta t)"""
        if len(self.data_size_infos[cluster_head_id]) == 0:
            return 0.0
        data_vol = sum([size for _, size in self.data_size_infos[cluster_head_id].items()])
        avg_delay = sum([delay for _, delay in self.data_delay_infos[cluster_head_id].items()]) / len(self.data_delay_infos[cluster_head_id])
        return self.function_cp_value(data_vol, avg_delay)
    
    def _calc_uplink_weight(self, member_id, cluster_head_id) -> float:
        """计算上行链路权重（成员→簇头）"""
        # 权重 = 数据量 x 感知贡献 x 归一化
        if cluster_head_id not in self.data_size_infos or member_id not in self.data_size_infos[cluster_head_id]:
            return 0.0
        data_vol = self.data_size_infos[cluster_head_id][member_id]
        # cluster_head = self.cav_world.get_vehicle_managers().get(cluster_head_id).v2x_manager
        # member = self.cav_world.get_vehicle_managers().get(member_id).v2x_manager
        # g = cluster_head.calculate_contribution_coefficient(member)
        g = 0.5
        print(f"Calculating uplink weight for member {member_id} to cluster head {cluster_head_id}: data_vol={data_vol}, contribution={g}")
        return data_vol * g / 1e6
    
    def _build_conflict_graph(self, global_links: List[Tuple[int, int, float]]):
        """基于SINR阈值构建冲突图"""
        nm = self.network_manager
        self.conflict_graph.clear()
        
        for i, (s1_id, t1_id, _) in enumerate(global_links):
            s1_v2x = self.cav_world.get_vehicle_managers().get(s1_id).v2x_manager
            t1_v2x = self.cav_world.get_vehicle_managers().get(t1_id).v2x_manager if t1_id != -1 else s1_v2x  # 广播目标为自身
            if not s1_v2x or not t1_v2x:
                continue
            
            for j, (s2_id, t2_id, _) in enumerate(global_links):
                if i >= j or (s1_id == s2_id and t1_id == t2_id):
                    continue
                s2_v2x = self.cav_world.get_vehicle_managers().get(s2_id).v2x_manager
                t2_v2x = self.cav_world.get_vehicle_managers().get(t2_id).v2x_manager if t2_id != -1 else s2_v2x
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

    def _power_control(self):
        """功率控制：满足SINR阈值的最小功率"""
        nm = self.network_manager
        self.power_allocation.clear()
        
        for (s_id, t_id), ch in self.channel_allocation.items():
            if ch == -1:
                continue
            s_v2x = self.cav_world.get_vehicle_managers().get(s_id).v2x_manager
            t_v2x = self.cav_world.get_vehicle_managers().get(t_id).v2x_manager if t_id != -1 else s_v2x
            if not s_v2x or not t_v2x:
                continue
            
            # 计算同信道干扰总和
            interf_total = 0.0
            for (s2_id, t2_id), ch2 in self.channel_allocation.items():
                if ch2 != ch or (s2_id == s_id and t2_id == t_id):
                    continue
                s2_v2x = self.cav_world.get_vehicle_managers().get(s2_id).v2x_manager
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

    def schedule(self, source, target, volume: float) -> bool:
        """执行调度并调用NS3接口"""
        # 全局链路收集与冲突图构建
        global_links = self._collect_global_links()
        self._build_conflict_graph(global_links)
        self._weighted_graph_coloring()
        self._power_control()
        
        # 获取当前链路分配结果
        link = (source.vehicle_id, target.vehicle_id if target else -1)
        if link not in self.channel_allocation or self.channel_allocation[link] == -1:
            return False
        
        # 调用NS3通信接口（转换功率为dBm）
        ch = self.channel_allocation[link]
        tx_power_w = self.power_allocation.get(link, 10 ** (source.tx_power / 10) / 1000)
        tx_power_dBm = 10 * math.log10(tx_power_w * 1000) if tx_power_w > 0 else source.tx_power
        
        print(f"Scheduling link {link}: channel={ch}, power={tx_power_dBm} dBm")
        success = self.network_manager.communicate(
            source, target, volume,
            subchannel_start=ch, subchannel_num=1
        )
        return success