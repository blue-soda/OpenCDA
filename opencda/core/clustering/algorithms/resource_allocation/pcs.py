from typing import Dict, List, Tuple, Set, Optional
import math
import zlib
from opencda.core.clustering.base import *
from opencda.core.clustering import utils
from opencda.core.clustering.utils import common
from opencda.core.clustering.utils import *
from opencda.core.clustering.utils.channel_model import build_channel_model
from opencda.core.networking.utils import calculate_distance
from opencda.log.logger_config import logger

class PCS(ResourceAllocationAlgorithm):
    def __init__(self, cav_world, lambda_subchannels: int = 10):
        """
        初始化PCS调度算法（新增链路自动生成与效用计算）
        :param cav_world: 自动驾驶车辆环境实例
        :param lambda_subchannels: 逻辑子信道数量（默认25）
        """
        super().__init__(cav_world)
        self.cav_world = cav_world
        self.lambda_subchannels = lambda_subchannels  # 逻辑子信道总数
        self.interference_range = 100  # 干扰范围（单位：米）
        self.link_utilities: Dict[Tuple[int, int, int], float] = {}  # 链路效用：(发送方vid, 接收方vid, 盲spot_id) -> 权重
        self.all_links: List[Tuple[int, int, int]] = []  # 所有潜在链路
        self.link_conflicts: Dict[Tuple[int, int, int], Dict[str, Set[Tuple[int, int, int]]]] = {}  # 链路冲突缓存
        self.resource_strategy: Dict[Tuple[int, int], int] = {}  # 最终调度策略：(发送方vid, 接收方vid) -> 子信道起始索引
        self.resource_sc_nums: Dict[Tuple[int, int], int] = {}  # 最终调度子信道数量：(发送方vid, 接收方vid) -> 连续子信道数量
        self.grid_selection: Dict[int, Dict[int, Set[str]]] = {}  # 网格选择：接收方vid -> 发送方vid -> 需要接收的网格ID集合
        self.grid_mAP_cache: Dict[int, Dict[int, float]] = {}  # 网格mAP缓存：vid -> grid_id -> mAP值（预计算）
        self.blind_spots_cache: Dict[Tuple[int, int, int, int], Dict[int, Set[str]]] = {}  # (vid, division, radius, min_grids) -> blind_spot_id -> grids
        self.excluded_receiver_grids: Dict[int, Set[str]] = {}
        # Raw-LiDAR PCS adaptation default.  The original paper schedules
        # semantic blind-spot features, while our replay transmits point-cloud
        # grids; larger blind-spot units avoid unrealistically tiny uploads.
        self.blind_spot_min_division = 4
        self.blind_spot_adjacency_radius = 4
        self.blind_spot_min_grids = 128
        self.min_overlap_grids = 0
        self.active_blind_spot_min_division = self.blind_spot_min_division
        self.bandwidth_all = 20.0 * (10 ** 6)
        self.time_slot = 0.1
        self.channel_model = build_channel_model(
            mode='logical',
            bandwidth_mhz=self.bandwidth_all / (10 ** 6),
            num_channels=self.lambda_subchannels,
            frame_deadline_s=self.time_slot)
        self.feature_bytes_per_grid = 1024
        self.point_bytes = 16

    @staticmethod
    def _grid_sort_key(grid_id: str):
        try:
            x_idx, y_idx = map(int, str(grid_id).split("_"))
            return (x_idx, y_idx)
        except (ValueError, IndexError):
            return (0, str(grid_id))

    @staticmethod
    def _grid_seed_key(grid_id: str):
        return zlib.crc32(str(grid_id).encode('utf-8'))

    def _get_vehicle_blind_spots(self, vid: int, min_division: int=1) -> Dict[int, Set[str]]:
        """
        获取车辆的盲 spot 集合（盲spot_id -> 对应的网格集合）
        盲spot定义：req_grids（需求范围）与high_density_grids（非盲spot区域）的差集
        """
        division = max(1, int(min_division))
        radius = max(1, int(getattr(self, 'blind_spot_adjacency_radius', 2)))
        min_grids = max(1, int(getattr(self, 'blind_spot_min_grids', 1)))
        cache_key = (int(vid), division, radius, min_grids)
        if cache_key in self.blind_spots_cache:
            return self.blind_spots_cache[cache_key]
        
        vehicle = common.global_vehicles.get(vid)
        if not vehicle:
            return {}
        
        blind_spots = {}
        blind_spot_grids = vehicle.req_grids - vehicle.high_density_grids  # 需求范围内的感知薄弱区域
        blind_spot_grids = blind_spot_grids - self.excluded_receiver_grids.get(
            int(vid),
            set())
        if not blind_spot_grids:
            return blind_spots
        
        # 对盲spot网格进行分组（相邻网格为一个盲spot）
        spot_id = 0
        unassigned_grids = blind_spot_grids.copy()
        size = len(blind_spot_grids)
        target_size = max(min_grids, int(math.ceil(size / float(division))))
        while unassigned_grids:
            start_grid = min(unassigned_grids, key=self._grid_seed_key)
            unassigned_grids.remove(start_grid)
            adjacent_grids = self._find_adjacent_grids(
                start_grid,
                unassigned_grids,
                target_size=target_size,
                radius=radius)
            blind_spot = {start_grid} | adjacent_grids
            blind_spots[spot_id] = blind_spot
            unassigned_grids -= adjacent_grids
            spot_id += 1
        self.blind_spots_cache[cache_key] = blind_spots
        return blind_spots

    def _generate_adjacent_grids(self, grid_id: str, radius: int = 2) -> Set[str]:
        """
        生成(x,y)的相邻网格（上下左右四个方向）
        """
        x, y = map(int, grid_id.split("_"))
        adjacent = set()
        radius = max(1, int(radius))
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                adjacent.add(f"{x + i}_{y + j}")
        return adjacent

    def _find_adjacent_grids(self, grid_id: str, candidate_grids: Set[str],
                             target_size: int, radius: int = 2) -> Set[str]:
        """
        查找相邻网格（适配'x_y'二维格式）
        """
        adjacent = set([grid_id])
        candidate = set([grid_id])
        target_size = max(1, int(target_size))
        flag = True
        while flag and len(adjacent) < target_size:
            flag = False
            cur_candidate = candidate.copy()
            candidate.clear()
            for grid_id in sorted(cur_candidate, key=self._grid_sort_key):
                adjacent_ids = self._generate_adjacent_grids(
                    grid_id,
                    radius=radius)
                for adj_id in sorted(adjacent_ids, key=self._grid_sort_key):
                    if adj_id in candidate_grids and adj_id not in adjacent:
                        candidate.add(adj_id)
                        adjacent.add(adj_id)
                        flag = True
                        if len(adjacent) >= target_size:
                            break
                if len(adjacent) >= target_size:
                    break
        return adjacent


    def _generate_potential_links(self, min_division: int=1, min_overlap: int=20):
        """
        生成所有潜在链路（发送方vid, 接收方vid, 盲spot_id）
        链路生成条件：
        1. 发送方的感知范围（sens_grids）覆盖接收方的某个盲spot网格
        2. 发送方与接收方在通信范围内
        """
        min_division = max(1, int(min_division))
        min_overlap = max(0, int(min_overlap))
        self.active_blind_spot_min_division = min_division
        vehicle_vids = sorted(common.global_vehicles.keys())
        for receiver_vid in vehicle_vids:
            # 获取接收方的盲spot
            receiver_blind_spots = self._get_vehicle_blind_spots(receiver_vid, min_division)
            if not receiver_blind_spots:
                continue
            
            # 查找能覆盖该盲spot的发送方
            for sender_vid in vehicle_vids:
                if int(sender_vid) == int(receiver_vid):
                    continue  # 不与自身建立链路
                
                sender = common.global_vehicles.get(sender_vid)
                if not sender:
                    continue
                
                # 检查通信范围
                if not self._is_in_communication_range(sender_vid, receiver_vid):
                    continue
                
                # 检查发送方感知范围是否覆盖接收方盲spot
                for spot_id in sorted(receiver_blind_spots):
                    spot_grids = receiver_blind_spots[spot_id]
                    overlap_grids = spot_grids & sender.sens_grids
                    # print(f"sender_vid: {sender_vid}, receiver_vid: {receiver_vid}, spot_id: {spot_id}, overlap_grids: {len(overlap_grids)}")
                    if overlap_grids and len(overlap_grids) >= min_overlap:  # 存在覆盖的网格，生成链路
                        link = (sender_vid, receiver_vid, spot_id)
                        self.all_links.append(link)

        # 去重链路
        self.all_links = list(dict.fromkeys(self.all_links))

    def _is_in_communication_range(self, sender_vid: int, receiver_vid: int) -> bool:
        """
        检查发送方与接收方是否在通信范围内
        """
        sender_vm = common.global_vms.get(sender_vid)
        receiver_vm = common.global_vms.get(receiver_vid)
        if not sender_vm or not receiver_vm:
            return False
        
        # 计算车辆间距离（使用V2XManager的距离计算方法）
        distance = calculate_distance(sender_vm, receiver_vm)
        # 通信范围：城市场景100m，高速场景200m（根据车辆速度判断场景）
        sender_vehicle = common.global_vehicles.get(sender_vid)
        speed = sender_vehicle.get_speed()  # 车辆速度（km/h）
        comm_range = 200 if speed >= 60 else 100
        return distance <= comm_range

    def _precompute_grid_mAP(self):
        """
        预计算每个网格的mAP值（U(g_h)）
        规则：基于网格中心到车辆LiDAR的距离（距离越近，mAP越高）
        适配网格ID生成规则：grid_id = f"{x_idx}_{y_idx}"，x_idx/y_idx为网格在全局坐标系的索引
        """
        for vid, vehicle in common.global_vehicles.items():
            # 获取车辆LiDAR在全局坐标系中的位置（x, y）
            vehicle_position = vehicle.get_position().location
            lidar_x = vehicle_position.x  # LiDAR全局x坐标（浮点数）
            lidar_y = vehicle_position.y  # LiDAR全局y坐标（浮点数）
            grid_size = vehicle.grid_size  # 网格尺寸（如0.8m，与生成网格时一致）
            self.grid_mAP_cache[vid] = {}
            for grid_id in vehicle.sens_grids:
                if grid_id in self.grid_mAP_cache[vid]:
                    continue
                
                # 解析网格ID，计算网格中心的全局坐标
                try:
                    x_idx, y_idx = map(int, grid_id.split("_"))
                except (ValueError, IndexError):
                    # 若网格ID格式异常，默认低mAP
                    self.grid_mAP_cache[vid][grid_id] = 0.2
                    continue
                
                # 计算网格中心坐标（全局坐标系）
                # 网格索引(x_idx, y_idx)对应的网格范围：[x_idx*grid_size, (x_idx+1)*grid_size)
                grid_center_x = x_idx * grid_size + grid_size / 2.0
                grid_center_y = y_idx * grid_size + grid_size / 2.0
                
                # 计算LiDAR到网格中心的欧式距离（2D平面距离，忽略z轴）
                distance = math.hypot(
                    grid_center_x - lidar_x,
                    grid_center_y - lidar_y
                )
                
                # 基于距离计算mAP（严格拟合论文Fig.5趋势：距离越远，精度下降越显著）
                if distance <= 20:
                    mAP = 0.9  # 近距离（≤20m）：高感知精度
                elif distance <= 40:
                    mAP = 0.7  # 中距离（20-40m）：中等感知精度
                elif distance <= 60:
                    mAP = 0.4  # 远距离（40-60m）：低感知精度
                elif distance <= 100:
                    mAP = 0.2  # 超远距离（60-100m）：极低感知精度
                else:
                    mAP = 0.1  # 超出常规感知范围：最低感知精度
                
                self.grid_mAP_cache[vid][grid_id] = mAP

    def _calculate_link_utilities(self):
        """
        计算链路效用（权重）：盲spot内所有网格的平均mAP（论文公式1）
        """
        for link in self.all_links:
            sender_vid, receiver_vid, spot_id = link
            receiver = common.global_vehicles.get(receiver_vid)
            if not receiver:
                continue
            
            # 获取该链路对应的盲spot网格
            receiver_blind_spots = self._get_vehicle_blind_spots(
                receiver_vid,
                self.active_blind_spot_min_division)
            spot_grids = receiver_blind_spots.get(spot_id, set())
            if not spot_grids:
                continue
            sender = common.global_vehicles.get(sender_vid)
            if not sender:
                continue
            covered_grids = spot_grids & sender.sens_grids
            if not covered_grids:
                continue
            
            # 计算网格平均mAP
            total_mAP = 0.0
            valid_grids = 0
            for grid_id in covered_grids:
                mAP = self.grid_mAP_cache.get(sender_vid, {}).get(grid_id, 0.0)
                total_mAP += mAP
                valid_grids += 1
            
            if valid_grids == 0:
                link_weight = 0.0
            else:
                coverage_ratio = min(
                    1.0,
                    float(len(covered_grids)) / float(max(len(spot_grids), 1)))
                link_weight = (total_mAP / valid_grids) * coverage_ratio
            
            self.link_utilities[link] = link_weight

    def _get_link_required_subchannels(self, link: Tuple[int, int, int]) -> int:
        """
        计算链路所需子信道数量（论文公式2）
        """
        sender_vid, receiver_vid, spot_id = link
        sender = common.global_vehicles.get(sender_vid)
        receiver = common.global_vehicles.get(receiver_vid)
        if not sender or not receiver:
            return 1
        
        # 获取盲spot网格集合
        receiver_blind_spots = self._get_vehicle_blind_spots(
            receiver_vid,
            self.active_blind_spot_min_division)
        spot_grids = receiver_blind_spots.get(spot_id, set())
        covered_grids = spot_grids & sender.sens_grids
        if not covered_grids:
            return 1

        payload_bytes = self._estimate_link_payload_bytes(sender,
                                                          covered_grids)
        channel_model = getattr(self, 'channel_model', None)
        if channel_model is not None:
            required_subchannels = channel_model.required_subchannels(
                payload_bytes,
                deadline_s=self.time_slot)
        else:
            total_feature_bits = payload_bytes * 8.0
            per_channel_capacity = (
                float(self.bandwidth_all) /
                float(max(self.lambda_subchannels, 1)))
            available_bits = max(
                per_channel_capacity * float(self.time_slot),
                1.0)
            required_subchannels = int(math.ceil(
                total_feature_bits / available_bits))
        return max(1, min(int(self.lambda_subchannels), required_subchannels))

    def _estimate_link_payload_bytes(self, sender, covered_grids):
        grid_points = getattr(sender, 'grid_local_points', None)
        if grid_points is not None:
            point_count = sum(
                len(grid_points.get(grid_id, []))
                for grid_id in covered_grids)
            if point_count > 0:
                return point_count * self.point_bytes
        return len(covered_grids) * self.feature_bytes_per_grid

    def _build_conflict_graph(self):
        """
        构建冲突图：计算每个链路的A类和B类冲突链路
        """
        for link in self.all_links:
            sender_q, receiver_q, spot_q = link
            conflicts = {"A": set(), "B": set()}
            
            # 计算A类冲突（共享发送方或接收方）
            for other_link in self.all_links:
                if link == other_link:
                    continue
                sender_p, receiver_p, spot_p = other_link
                if sender_q == sender_p or receiver_q == receiver_p or sender_q == receiver_p or sender_p == receiver_q:
                    conflicts["A"].add(other_link)
            
            # 计算B类冲突（同子信道+干扰范围内）
            for other_link in self.all_links:
                if link == other_link or other_link in conflicts["A"]:
                    continue
                sender_p, receiver_p, spot_p = other_link
                # 检查接收方是否在发送方干扰范围内
                sender_q_vm = common.global_vms.get(sender_q)
                receiver_p_vm = common.global_vms.get(receiver_p)
                if sender_q_vm and receiver_p_vm:
                    # if sender_q_vm.is_in_interference_range(receiver_p_vm.get_position()):
                    distance = calculate_distance(sender_q_vm, receiver_p_vm)
                    if distance <= self.interference_range:
                        conflicts["B"].add(other_link)
                # 检查对方发送方是否在当前接收方干扰范围内
                sender_p_vm = common.global_vms.get(sender_p)
                receiver_q_vm = common.global_vms.get(receiver_q)
                if sender_p_vm and receiver_q_vm:
                    # if sender_p_vm.is_in_interference_range(receiver_q_vm.get_position()):
                    distance = calculate_distance(sender_p_vm, receiver_q_vm)
                    if distance <= self.interference_range:
                        conflicts["B"].add(other_link)
            
            self.link_conflicts[link] = conflicts
            # print(f"link: {link} conflicts: {conflicts}")
            logger.info(f"link: {link} conflicts: {conflicts}")

    def _weight_splitting(self, M: List[Tuple[int, int, int]], W: Dict[Tuple[int, int, int], float]) -> Tuple[
        List[Tuple[int, int, int]], Dict[Tuple[int, int, int], float], Tuple[int, int, int]]:
        """权重拆分（Algorithm 1）"""
        if not M:
            return [], {}, None
        q_t = M[0]  # 选择第一个链路作为pivot
        M.remove(q_t)
        w_qt = W[q_t]
        c_qt = self._get_link_required_subchannels(q_t)
        W1 = {}
        M_t1 = []
        
        for p in M:
            if p in self.link_conflicts[q_t]["A"]:
                logger.info(f"link: {p} in A class conflicts with {q_t}, w2_p: {w_qt}")
                w2_p = w_qt
            elif p in self.link_conflicts[q_t]["B"]:
                c_p = self._get_link_required_subchannels(p)
                denominator = math.floor(self.lambda_subchannels / c_qt)
                w2_p = (w_qt * c_p) / denominator if denominator != 0 else 0
                logger.info(f"link: {p} in B class conflicts with {q_t}, w2_p: {w2_p}")
            else:
                w2_p = 0
                logger.info(f"link: {p} not in {q_t}'s A or B class, w2_p: {w2_p}")
            w1_p = W[p] - w2_p
            # print(f"w1_p: {w1_p} = W[p]: {W[p]} - w2_p: {w2_p}")
            logger.info(f"link: {p} w1_p: {w1_p} = W[p]: {W[p]} - w2_p: {w2_p}")
            W1[p] = w1_p
            if w1_p > 0:
                M_t1.append(p)
        
        return M_t1, W1, q_t

    def _resource_allocation(self, P_t1: Set[Tuple[int, int, int]], q_t: Tuple[int, int, int]) -> Tuple[
        Set[Tuple[int, int, int]], Optional[int]]:
        """资源分配（Algorithm 2）"""
        # 检查A类冲突
        a_conflicts = self.link_conflicts[q_t]["A"]
        if a_conflicts & P_t1:
            return P_t1, None
        
        # 计算可用子信道
        available_subchannels = set(range(self.lambda_subchannels))
        c_qt = self._get_link_required_subchannels(q_t)
        
        for p in P_t1:
            if p in self.link_conflicts[q_t]["B"]:
                sender_p, receiver_p, _ = p
                if (sender_p, receiver_p) in self.resource_strategy:
                    start_p = self.resource_strategy[(sender_p, receiver_p)]
                    c_p = self._get_link_required_subchannels(p)
                    occupied = set(range(start_p, start_p + c_p))
                    available_subchannels -= occupied
        
        # 寻找连续可用子信道段
        sorted_available = sorted(available_subchannels)
        start_idx = None
        for i in range(len(sorted_available) - c_qt + 1):
            if sorted_available[i + c_qt - 1] - sorted_available[i] == c_qt - 1:
                start_idx = sorted_available[i]
                break
        
        if start_idx is not None:
            # 分配子信道并更新网格选择
            sender_q, receiver_q, spot_id = q_t
            self.resource_strategy[(sender_q, receiver_q)] = start_idx
            self.resource_sc_nums[(sender_q, receiver_q)] = c_qt
            # 获取盲spot网格并记录
            receiver_blind_spots = self._get_vehicle_blind_spots(
                receiver_q,
                self.active_blind_spot_min_division)
            sender = common.global_vehicles.get(sender_q)
            spot_grids = receiver_blind_spots.get(spot_id, set())
            if sender:
                spot_grids = spot_grids & sender.sens_grids
            if receiver_q not in self.grid_selection:
                self.grid_selection[receiver_q] = {}
            if sender_q not in self.grid_selection[receiver_q]:
                self.grid_selection[receiver_q][sender_q] = set()
            self.grid_selection[receiver_q][sender_q].update(spot_grids)
            return P_t1 | {q_t}, start_idx
        else:
            return P_t1, None

    def _pcs_recursion(self, M: List[Tuple[int, int, int]], W: Dict[Tuple[int, int, int], float]) -> Set[Tuple[int, int, int]]:
        """PCS递归调度（Algorithm 3）"""
        if not M:
            return set()
        
        M_t1, W_t1, q_t = self._weight_splitting(M, W)
        P_t1 = self._pcs_recursion(M_t1, W_t1)
        P_t, _ = self._resource_allocation(P_t1, q_t)
        return P_t

    def main(self):
        """执行PCS调度（重写父类方法）"""

        # 初始化：生成链路并计算效用
        self._generate_potential_links(
            min_division=self.blind_spot_min_division,
            min_overlap=self.min_overlap_grids)
        self._precompute_grid_mAP()
        self._calculate_link_utilities()

        # 1. 构建冲突图
        self._build_conflict_graph()
        
        # 2. 按权重降序排序链路
        sorted_links = sorted(
            (link for link in self.all_links if link in self.link_utilities),
            key=lambda x: self.link_utilities[x],
            reverse=True)
        initial_weights = {link: self.link_utilities[link] for link in sorted_links}
        
        # 3. 执行递归调度
        self._pcs_recursion(sorted_links, initial_weights)

        # 4. 更新资源分配策略到车辆
        logger.info(f"PCS resource strategy: {self.resource_strategy}")
        for receiver_q, sender_grids_dict in self.grid_selection.items():
            for sender_q, spot_grids in sender_grids_dict.items():
                logger.info(f"receiver_q: {receiver_q}, sender_q: {sender_q}, spot_grids: {len(spot_grids)}")

        # 5. 清除缓存
        self.blind_spots_cache.clear()
        self.grid_mAP_cache.clear()

    def run(self):
        """Execute PCS-style resource allocation and write strategies back."""
        self.clear_resource_allocation_strategy()
        self.main()
        self.update_resource_allocation_strategy()
        return True

    def update_resource_allocation_strategy(self):
        """更新调度策略到车辆（重写父类API）"""
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for vid, vehicle_manager in vehicle_dict.items():
            vehicle_manager.v2x_manager.cluster_state = { 'head_id': vid, 'member_ids': set([vid]) }
        for k, v in self.resource_strategy.items():
            sender_q, receiver_q = k
            selected_grids = self.grid_selection.get(
                receiver_q, {}).get(sender_q, [])
            if not selected_grids:
                continue
            sender_vm = vehicle_dict[sender_q]
            receiver_vm = vehicle_dict[receiver_q]
            receiver_vm.perception_manager.apply_late_fusion = False
            receiver_vm.perception_manager.do_not_skip_any_cav = True
            receiver_vm.perception_manager.co_manager.set_grid_selection(
                {sender_q: selected_grids})
            receiver_vm.v2x_manager.scheduler.set_strategies({k: v})
            receiver_vm.v2x_manager.scheduler.channel_allocation_sc_nums = (
                getattr(receiver_vm.v2x_manager.scheduler,
                        'channel_allocation_sc_nums', {}))
            receiver_vm.v2x_manager.scheduler.channel_allocation_sc_nums[k] = (
                self.resource_sc_nums.get(k, 1))
            receiver_vm.v2x_manager.cluster_state['head_id'] = receiver_q
            receiver_vm.v2x_manager.cluster_state['member_ids'].add(sender_q)
            sender_vm.v2x_manager.cluster_state['head_id'] = receiver_q
            sender_vm.v2x_manager.cluster_state['member_ids'].add(receiver_q)

    def clear_resource_allocation_strategy(self):
        """清空调度策略（重写父类方法）"""
        self.all_links.clear()
        self.link_utilities.clear()
        self.resource_strategy.clear()
        self.resource_sc_nums.clear()
        self.grid_selection.clear()
        self.link_conflicts.clear()
        self.blind_spots_cache.clear()
        self.grid_mAP_cache.clear()
