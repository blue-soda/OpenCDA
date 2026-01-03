from opencda.customize.core.clustering.algorithm.common import *
from opencda.log.logger_config import logger
from opencda.customize.core.clustering.algorithm.pcs import PCS
import random

class RandomRA(PCS):
    def __init__(self, cav_world, lambda_subchannels = 10):
        super().__init__(cav_world, lambda_subchannels)

    def main(self):
        """执行随机算法调度（重写父类方法）"""

        # 初始化：生成链路并计算效用
        self._generate_potential_links(min_division=1, min_overlap=50)
        self._precompute_grid_mAP()
        self._calculate_link_utilities()

        # 1. 构建冲突图
        self._build_conflict_graph()
        
        # 2. 随机打乱链路顺序
        random_links = self.all_links.copy()
        random.shuffle(random_links)
        
        # 3. 随机选择不冲突的链路
        selected_links = set()
        available_subchannels = set(range(self.lambda_subchannels))
        
        for link in random_links:
            # 检查链路是否与已选链路有A类冲突
            has_a_conflict = False
            for selected_link in selected_links:
                if link in self.link_conflicts[selected_link]["A"] or selected_link in self.link_conflicts[link]["A"]:
                    has_a_conflict = True
                    break
            if has_a_conflict:
                continue
            
            # 计算链路所需子信道数量
            required_subchannels = self._get_link_required_subchannels(link)
            
            # 寻找连续可用子信道段
            sorted_available = sorted(available_subchannels)
            start_idx = None
            for i in range(len(sorted_available) - required_subchannels + 1):
                if sorted_available[i + required_subchannels - 1] - sorted_available[i] == required_subchannels - 1:
                    start_idx = sorted_available[i]
                    break
            
            if start_idx is not None:
                # 分配子信道
                sender_q, receiver_q, spot_id = link
                self.resource_strategy[(sender_q, receiver_q)] = start_idx
                
                # 更新网格选择
                receiver_blind_spots = self._get_vehicle_blind_spots(receiver_q, min_division=1)
                spot_grids = receiver_blind_spots.get(spot_id, set())
                if receiver_q not in self.grid_selection:
                    self.grid_selection[receiver_q] = {}
                if sender_q not in self.grid_selection[receiver_q]:
                    self.grid_selection[receiver_q][sender_q] = set()
                self.grid_selection[receiver_q][sender_q].update(spot_grids)
                
                # 更新已选链路和可用子信道
                selected_links.add(link)
                allocated_subchannels = set(range(start_idx, start_idx + required_subchannels))
                available_subchannels -= allocated_subchannels
                
                # 检查是否所有子信道都已用完
                if not available_subchannels:
                    break
        
        # 4. 更新资源分配策略到车辆
        print("Random resource strategy:", self.resource_strategy)
        print("Random grid selection:", self.grid_selection)

        # 5. 清除缓存
        self.blind_spots_cache.clear()
        self.grid_mAP_cache.clear()
