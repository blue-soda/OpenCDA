from collections import defaultdict
import math
from opencda.core.clustering.base import ResourceAllocationAlgorithm
from opencda.core.clustering import utils
from opencda.core.clustering.utils import *
from opencda.core.clustering.utils.common import Params, visualize_grid_set
from opencda.core.clustering.utils import common
from opencda.core.networking.utils import calculate_available_data_rate
from opencda.log.logger_config import logger

# 辅助函数
def dB_to_linear(dB_value):
    """将dB值转换为线性值"""
    return 10 ** (dB_value / 10.0)

def calculate_distance(vm1, vm2):
    """计算两个车辆之间的距离"""
    # 处理不同类型的对象
    if hasattr(vm1, 'localizer'):
        pos1 = vm1.localizer.get_ego_pos()
    elif hasattr(vm1, 'ego_pos') and len(vm1.ego_pos) > 0:
        # V2XManager has ego_pos deque
        pos1 = vm1.ego_pos[-1]
    else:
        raise ValueError(f"Cannot get position from object type: {type(vm1)}")

    if hasattr(vm2, 'localizer'):
        pos2 = vm2.localizer.get_ego_pos()
    elif hasattr(vm2, 'ego_pos') and len(vm2.ego_pos) > 0:
        # V2XManager has ego_pos deque
        pos2 = vm2.ego_pos[-1]
    else:
        raise ValueError(f"Cannot get position from object type: {type(vm2)}")

    return math.sqrt((pos1.location.x - pos2.location.x)**2 +
                     (pos1.location.y - pos2.location.y)**2)

def calculate_channel_gain(distance):
    """计算信道增益（路径损耗）"""
    if distance < 1:
        distance = 1
    # 使用简化的路径损耗模型
    path_loss_dB = 128.1 + 37.6 * math.log10(distance / 1000.0)  # distance in meters
    return 10 ** (-path_loss_dB / 10.0)

def get_interference_contribution(src_vm, dst_vm):
    """计算从src_vm到dst_vm的干扰功率"""
    tx_power_w = dB_to_linear(src_vm.tx_power) / 1000.0  # dBm→W
    distance = calculate_distance(src_vm, dst_vm)
    ch_gain = calculate_channel_gain(distance)
    return tx_power_w * ch_gain

def calculate_sinr_linear(signal_power, interference_power, noise_power):
    """计算SINR（线性值）"""
    return signal_power / (interference_power + noise_power)

class PotentialGame(ResourceAllocationAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)
        self.cav_world = cav_world  # 保持向后兼容
        # need Vehicle_Grid as input
        self.p = Params()
        self.strategies = {}  # {head_id: (member_id, subchannel_k, time_slot_t, [grid_ids])}
        self.grids_uploading = set()
        self.grids_ch_sens = set()
        self.grids_density = {}
        self.convergence_stats = {}
    
    def set_clusters(self, clusters):
        super().set_clusters(clusters)
        self.max_grids_per_rb = self.calculate_max_grids_per_rb()
        logger.info(f"grid_bits: {self.clusters[0].grid_bits}")
        logger.info(f"max_grids_per_rb: {self.max_grids_per_rb}")
        self.compute_grids_ch_sens()

    def main(self):
        self.channel_game()
        
    def run(self):
        self.clear_resource_allocation_strategy()
        ret = self.channel_game()
        self.update_resource_allocation_strategy()
        return ret

    def clear_resource_allocation_strategy(self):
        """Clear resource allocation strategies."""
        self.strategies = {}
        self.grids_uploading = set()
        self.grids_ch_sens = set()
        self.grids_density = {}

    def update_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for hid, links in self.strategies.items():
            for mid, k, t, grid_ids in links:
                schedule = {(mid, hid): k}
                vehicle_dict[hid].v2x_manager.scheduler.set_strategies(schedule)
                grids_selection = {mid: grid_ids}
                vehicle_dict[hid].perception_manager.co_manager.set_grid_selection(grids_selection)
                vehicle_dict[mid].perception_manager.do_not_skip_any_cav = True

    def calculate_max_grids_per_rb(self, sinr=None):
        return common.calculate_max_grids_per_rb(sinr, self.p.bandwidth_per_channel, self.p.T_ddl, self.clusters[0].grid_bits)

    def grid_utility_density(self, density, rho_th):
        return common.density_score(density, rho_th)
    
    def grid_late_utility(self, participating_clusters_density_dict):
        # 计算某网格由参与簇集合提供晚期融合的总效用
        if not participating_clusters_density_dict:
            return 0.0
        U_late = max([self.grid_utility_density(density, common.global_vehicles[hid].rho_th) for hid, density in participating_clusters_density_dict.items()])
        return U_late
    
    def grid_early_utility(self, density, hid=1):
        # 计算某网格由参与簇集合提供早期融合的总效用
        return self.grid_utility_density(density, common.global_vehicles[hid].rho_th)
        
    def get_certain_strategy(self, hid, mid=None, grid_id=None):
        for (mid, sc, t, grids) in self.strategies[hid]:
            if (mid is not None and mid == mid) or (grid_id is not None and grid_id in grids):
                return (mid, sc, t, len(grids))
        return (None, -1, -1, 0)
    
    def compute_delay(self, cluster, mid=None, grid_id=None):
        return self.p.T_ddl
        hid = cluster.head_id
        mid, subchannel_id, t, grids_num = self.get_certain_strategy(hid, mid, grid_id)
        if mid is None or grid_id is None or subchannel_id == -1: # 此前未传输过
            return self.p.T_ddl
        cache_key = (hid, mid, subchannel_id)
        data_rate = self.compute_data_rate(cluster, mid, subchannel_id)
        d_s = self.transmission_delay(data_rate, cluster, grids_num)
        return d_s

    def get_interference(self, target_vm, subchannel=None, exclude_vid=None):
        """
        计算目标车辆 target_vm 在指定子信道上的总干扰功率
        exclude_vid: 排除的车辆ID（通常为自身）
        """
        interference_power = 0.0
        for hid, links in self.strategies.items():
            for (mid, sc, t, grids) in links:
                if sc != subchannel or mid == exclude_vid:
                    continue
                vm = common.global_vms[mid]
                interference_power += get_interference_contribution(vm, target_vm)
        return interference_power
    
    def compute_data_rate(self, cluster_h, member_id, subchannel_id=-1):
        """
        计算成员 i 向簇头 h 的上行通信时延 d_S
        组成部分：
            - 传播时延: distance / c
            - 传输时延: 数据量 / 速率
        其中速率由路径损耗对应的SINR决定
        """
        # --- 估算通信速率 ---
        src_vm = common.global_vms[member_id]
        dst_vm = common.global_vms[cluster_h.head_id]

        tx_power_w = dB_to_linear(src_vm.tx_power) / 1000.0  # dBm→W
        ch_gain = calculate_channel_gain(calculate_distance(src_vm, dst_vm))
        noise_power_w = dB_to_linear(dst_vm.noise_power) / 1000.0  # dBm→W
        interference_power_w = self.get_interference(dst_vm, subchannel=subchannel_id, exclude_vid=member_id)
        sinr_linear = calculate_sinr_linear(tx_power_w * ch_gain, interference_power_w, noise_power_w)
        data_rate = calculate_available_data_rate(self.p.bandwidth_per_channel, sinr_linear)
        return data_rate
    
    @staticmethod
    def transmission_delay(data_rate, cluster, grids_num):
        grid_bits = cluster.grid_bits * grids_num
        return grid_bits / max(data_rate, 1e-9)
    
    def compute_grids_density(self, cluster):
        hid = cluster.head_id
        links = self.strategies.get(hid, None)
        grids_density = common.global_vehicles[hid].grid_density_dict.copy()
        for (mid, sc, t, grids) in links:
            for grid_id in grids:
                grids_density[grid_id] = grids_density.get(grid_id, 0.0) + common.global_vehicles[mid].grid_density_dict.get(grid_id, 0.0)
        return grids_density
    
    def get_participating_clusters(self, grid_id):
        participating_clusters = {}
        for cluster in self.clusters:
            grid_density_dict = self.compute_grids_density(cluster)
            if grid_id in grid_density_dict:
                participating_clusters[cluster.head_id] = grid_density_dict[grid_id]
                break
        return participating_clusters
    
    def bits_to_sinr(self, bits):
        data_rate = bits / self.p.T_ddl
        sinr = 2 ** (data_rate / self.p.bandwidth_per_channel) - 1
        return sinr

    def update_grids_density(self, update_self_density=True):
        if update_self_density:
            for cluster in self.clusters:
                hid = cluster.head_id
                self.grids_density[hid] = common.global_vehicles[hid].grid_density_dict

        for hid, links in self.strategies.items():        
            grids_density_hid = defaultdict(int)
            for (mid, sc, t, grids) in links:
                for grid_id in grids:
                    grids_density_hid[grid_id] += common.global_vehicles[mid].grid_density_dict.get(grid_id, 0.0)
    
    def compute_grids_uploading_inside_cluster(self, cluster):
        grids_density = self.compute_grids_density(cluster)
        grids = set([grid_id for grid_id, density in grids_density.items() if density >= common.global_vehicles[cluster.head_id].rho_th])
        return grids
    
    def compute_grids_ch_sens(self):
        self.grids_ch_sens = set()
        for cluster in self.clusters:
            head_id = cluster.head_id
            self.grids_ch_sens |= common.global_vehicles[head_id].high_density_grids

    def grid_score(self, cluster, grid_id, member_grid_density):
        participating_clusters = self.get_participating_clusters(grid_id)
        late_score = self.grid_late_utility(participating_clusters)

        current_density = 0.0
        early_score = 0.0
        if cluster.head_id in participating_clusters:
            current_density = participating_clusters[cluster.head_id]
            early_score = self.grid_early_utility(current_density, cluster.head_id)

        current_density += member_grid_density
        early_score_if_upload = self.grid_early_utility(current_density, cluster.head_id)

        gain = max(late_score, early_score_if_upload) - max(late_score, early_score)
        return gain

    def best_response(self, cluster, strategies, global_rb_used):
        """
        近似最优 best-response
        - 每个簇头最多使用 B_h 个 RB
        - 两轮调度：exclusive → reuse
        """
        h = cluster.head_id
        # B_h = self.p.N_max  # 或者单独设一个 B_h
        B_h = 1
        schedule = []

        # ========= Step 0: 准备候选 grids =========

        # 本簇可选 grids
        logger.info(f"Cluster head {h} req grids: {len(cluster.req_grids)}")
        # print("Cluster head", h, "req grids:", (cluster.req_grids))
        # candidate_grids = cluster.req_grids.copy()
        # candidate_grids -= self.grids_ch_sens 
        # candidate_grids -= self.grids_uploading
        J_req = cluster.req_grids #J_h^req
        J_eff = self.compute_grids_uploading_inside_cluster(cluster) #J_h^eff
        # 下面挑选grids扩充 J_eff
        candidate_grids = J_req - J_eff

        logger.info(f"Cluster head {h} candidate grids: {len(candidate_grids)}")
        # print("Cluster head", h, "grids_ch_sens grids:", (vehicles[h].sens_grids))
        if cluster == self.clusters[0]:
            visualize_grid_set(self.grids_uploading, title=f"All uploading high density grids")
            visualize_grid_set(candidate_grids, title=f"Cluster head {h} candidate grids")
            visualize_grid_set(J_req, title=f"Cluster requested grids")
            visualize_grid_set(J_eff, title=f"Cluster head {h} collected high density grids")
            # visualize_grid_set(self.grids_ch_sens, title=f"All cluster heads' high density grids")
        if not candidate_grids:
            return []

        # ========= Step 1: 第一轮（exclusive grid） =========
        cur_h_links = self.strategies.get(h, None)
        member_grid_map = {}
        for m in cluster.members:
            if m == h:
                continue
            grids_m = common.global_vehicles[m].sens_grids & candidate_grids
            if cur_h_links is not None:
                for (mid, sc, t, grids) in cur_h_links:
                    if mid == m:
                        grids_m -= set(grids) # 已经调度的网格都不算
                        break
            if grids_m and len(grids_m) > 0:
                member_grid_map[m] = grids_m

        member_grid_score_map = {} # 记录每个member对每个grid的score
        for m, grids_m in member_grid_map.items():
            member_grid_score_map[m] = {}
            for grid in grids_m:
                member_grid_density = common.global_vehicles[m].grid_density_dict.get(grid, 0.0)
                member_grid_score_map[m][grid] = self.grid_score(cluster, grid, member_grid_density)

        members_sorted = sorted(
            member_grid_map.keys(),
            # key=lambda m: avg_grids_score(m, member_grid_map[m]),
            key=lambda m: sum([member_grid_score_map[m][grid] for grid in member_grid_map[m]]),
            reverse=True
        )

        if cluster == self.clusters[0]:
            for m in members_sorted:
                logger.info(f"Cluster head {h} member {m} can provide candidate grids: {len(member_grid_map[m])}")
                visualize_grid_set(common.global_vehicles[m].sens_grids, title=f"Cluster head {h} member {m} sensed grids", rho_th=common.global_vehicles[m].rho_th, grid_density_dict=common.global_vehicles[m].grid_density_dict)
                visualize_grid_set(member_grid_map[m], title=f"Cluster head {h} member {m} candidate grids", rho_th=common.global_vehicles[m].rho_th, grid_density_dict=common.global_vehicles[m].grid_density_dict)

        used_rbs = 0
        for m in members_sorted:
            if used_rbs >= B_h:
                break

            # 找一个可用 RB
            for k in range(self.p.num_channels):
                if global_rb_used[(k, 0)] >= 1:
                    continue

                # 按照优先级分配 grid
                member_grids_sorted = sorted(
                    list(member_grid_map[m]),
                    key=lambda grid: member_grid_score_map[m][grid],
                    reverse=True
                )
                grids = member_grids_sorted[:self.max_grids_per_rb]
                # grids = common.global_vehicles[m].sens_grids
                if not grids:
                    continue

                schedule.append((m, k, 0, grids))
                self.grids_uploading |= set(grids)
                global_rb_used[(k, 0)] += 1
                used_rbs += 1
                logger.info(f"Cluster head {h} assign exclusive RB {(k, 0)} to member {m} grids: {grids}")
                # 更新占用
                for g in grids:
                    candidate_grids.discard(g)
                break

        # # ========= Step 2: 第二轮（自身已分配 RB 修改） =========
        # for (mid, k, t, grids) in cur_h_links:
        #     if used_rbs >= B_h:
        #         break
        #     self.strategies[h].remove((mid, k, t, grids))
        #     score_assigned = sum([self.grid_score(cluster, grid, common.global_vehicles[mid].grid_density_dict.get(grid, 0.0)) for grid in grids])
        #     replaced = False
        #     print(f"cur_h_links: Cluster head {h} member {mid} assigned score: {score_assigned}")
        #     logger.info(f"cur_h_links: Cluster head {h} member {mid} assigned score: {score_assigned}")
        #     for m in members_sorted:
        #         score_new = sum([self.grid_score(cluster, grid, common.global_vehicles[m].grid_density_dict.get(grid, 0.0)) for grid in member_grid_map[m]])
        #         print(f"cur_h_links: Cluster head {h} member {m} new score: {score_new}")
        #         logger.info(f"cur_h_links: Cluster head {h} member {m} new score: {score_new}")
        #         if score_new > score_assigned:
        #             replaced = True
        #             upload_grids = member_grid_map[m][:self.max_grids_per_rb]
        #             schedule.append((m, k, t, upload_grids))
        #             self.grids_uploading |= set(upload_grids)
        #             global_rb_used[(k, 0)] += 1
        #             used_rbs += 1
        #             print(f"cur_h_links: Cluster head {h}: member {m} take place of member {mid}, with score {score_new} > {score_assigned}")
        #             logger.info(f"cur_h_links: Cluster head {h}: member {m} take place of member {mid}, with score {score_new} > {score_assigned}")
        #             # 更新占用
        #             for g in grids:
        #                 candidate_grids.discard(g)
        #             break
        #     if not replaced:
        #         self.strategies[h].append((mid, k, t, grids))
                

                     
        # ========= Step 2: 第二轮（受保护 RB 复用） =========
        multiplex_bits = 0.10 * self.max_grids_per_rb
        sinr_min_multiplex = self.bits_to_sinr(multiplex_bits)
        logger.info(f"Cluster head {h} sinr_min_multiplex: {sinr_min_multiplex}")

        for k in range(self.p.num_channels):
            if used_rbs >= B_h:
                break

            # ---- 找该 RB 上已有的上传计划 ----
            existing_links = []
            used_grids = 0
            for hid, links in strategies.items():
                for (mid, sc, t, grids) in links:
                    if sc == k:
                        existing_links.append((mid, hid, grids))
                        used_grids += len(grids)

            # 若没有已有链路 or 已满，跳过（第一轮已经处理过）
            if not existing_links or len(existing_links) >= self.p.channel_capacity or used_grids >= self.max_grids_per_rb * 0.80:
                continue
            
            logger.info(f"Cluster head {h} consider reusing RB {(k, 0)} with existing links: {existing_links}")
            # ---- Step 2.1：初步检查已有发送方 → 当前簇头的干扰 ----

            h_vm = common.global_vms[h]

            for (mid_old, hid_old, grids_old) in existing_links: #existing_links列表只会有一个元素
                s_vm_old = common.global_vms[mid_old]
                t_vm_old = common.global_vms[hid_old]
                interf = get_interference_contribution(s_vm_old, h_vm)

                # 当前簇头接收信号功率（假设无干扰发送）
                signal_power = dB_to_linear(s_vm_old.tx_power) / 1000.0 * calculate_channel_gain(10) 
                # 10m距离下的信道增益, 用于初步筛选
                noise_power = dB_to_linear(h_vm.noise_power) / 1000.0  # dBm→W

                sinr = calculate_sinr_linear(
                    signal_power,
                    interf,
                    noise_power
                )

                logger.info(f"Cluster head {h}: existing link from {mid_old} SINR: {sinr} signal_power: {signal_power} interf: {interf}")
                if sinr < sinr_min_multiplex:
                    logger.info(f"Cluster head {h} cannot reuse RB {(k, 0)} due to insufficient SINR on existing link from {mid_old}")
                    continue
                
                max_grids_possible = math.floor(
                    calculate_available_data_rate(self.p.bandwidth_per_channel, sinr) * self.p.T_ddl / cluster.grid_bits
                )
                logger.info(f"Cluster head {h}: existing link from {mid_old} max_grids_possible: {max_grids_possible}")

                # ---- Step 2.2：选择对外簇最友好的 member, 检查是否会干扰原链路 ----
                best_m = None
                best_gain = -1

                interf_old_to_h = get_interference_contribution(s_vm_old, h_vm)

                for m, grids_m in member_grid_map.items():
                    m_vm = common.global_vms[m]
                    interf_h_to_old = get_interference_contribution(m_vm, t_vm_old)
                    if not grids_m:
                        logger.info(f"Cluster head {h} member {m} has no candidate grids.")
                        logger.info(f"candidate_grids: {len(candidate_grids)}")
                        logger.info(f"common.global_vehicles[m].sens_grids: {len(common.global_vehicles[m].sens_grids)}")
                        # visualize_grid_set(common.global_vehicles[m].sens_grids, title=f"Member {m} sensed grids")
                        continue

                    # 若接收方是当前簇头，不必判冲突
                    if hid_old == h:
                        continue

                    # 原链路所需 SINR
                    bits = len(grids_old) * cluster.grid_bits
                    sinr_min_old = self.bits_to_sinr(bits)
                    logger.info(f"Cluster head {h} checking conflict for member {m} on existing link to {hid_old} required SINR: {sinr_min_old}")

                    signal_power = dB_to_linear(m_vm.tx_power) / 1000.0 * calculate_channel_gain(calculate_distance(m_vm, t_vm_old))
                    noise_power = dB_to_linear(t_vm_old.noise_power) / 1000.0  # dBm→W
                    sinr = calculate_sinr_linear(
                        signal_power,
                        interf_h_to_old,
                        noise_power
                    )

                    logger.info(f"Cluster head {h} member {m} to existing link receiver {hid_old} SINR: {sinr}")
                    if sinr < sinr_min_old:
                        logger.info(f"SINR too low, skip.")
                        continue

                    # ---- Step 2.3: 再次精确计算外簇原链路对当前簇的干扰 ----
                    signal_power = dB_to_linear(m_vm.tx_power) / 1000.0 * calculate_channel_gain(calculate_distance(m_vm, h_vm))
                    noise_power = dB_to_linear(h_vm.noise_power) / 1000.0  # dBm→W
                    sinr = calculate_sinr_linear(
                        signal_power,
                        interf_old_to_h,
                        noise_power
                    )
                    logger.info(f"Cluster head {h} member {m} to existing link receiver {hid_old} SINR: {sinr}")
                    if sinr < sinr_min_multiplex:
                        logger.info(f"SINR too low, skip.")
                        continue

                    max_grids = self.calculate_max_grids_per_rb(sinr)
                    logger.info(f"Cluster head {h} member {m} to existing link receiver {hid_old} max_grids: {max_grids}")
                    gain = min(len(grids_m), max_grids)
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m

            if best_m is None:
                continue

            # ---- Step 2.4: 执行复用 ----
            # grids = list(
            #     vehicles[best_m].sens_grids & candidate_grids
            # )[: best_gain]
            # 按照优先级分配 grid
            member_grids_sorted = sorted(
                list(member_grid_map[best_m]),
                key=lambda grid: member_grid_score_map[best_m][grid],
                reverse=True
            )
            grids = member_grids_sorted[: best_gain]

            if not grids:
                continue

            schedule.append((best_m, k, 0, grids))
            self.grids_uploading |= set(grids)
            global_rb_used[(k, 0)] += 1
            used_rbs += 1
            logger.info(f"Cluster head {h} reuse RB {k, 0} for member {best_m} grids: {grids}")

            for g in grids:
                candidate_grids.discard(g)

        return schedule


    def channel_game(self, max_iter=20):
        self.strategies = {cluster.head_id: [] for cluster in self.clusters}
        global_rb_used = defaultdict(int) # (k_chan, t_slot) -> used_count
        total_cluster_updates = 0
        for it in range(max_iter):
            updated = False
            for cluster in self.clusters:
                h = cluster.head_id
                new_schedule = self.best_response(cluster, self.strategies, global_rb_used)
                if len(new_schedule) > 0:
                    self.strategies[h] += new_schedule
                    logger.info(f"Cluster head {h} strategy updated.")
                    updated = True
                    total_cluster_updates += 1
                else:
                    logger.info(f"Cluster head {h} strategy unchanged.")
            if not updated:
                break
        logger.info(f"Channel game converged in {it+1} iterations.")
        link_count = 0
        selected_grid_count = 0
        rb_occupancy = defaultdict(int)
        for h in self.strategies:
            for m, k, t, grids in self.strategies[h]:
                link_count += 1
                selected_grid_count += len(grids)
                rb_occupancy[(k, t)] += 1
                logger.info(f"strategy: Cluster head {h} member {m} on RB {k, t} grids: {len(grids)}")
        self.convergence_stats = {
            'iterations': it + 1,
            'cluster_updates': total_cluster_updates,
            'scheduled_links': link_count,
            'selected_grids': selected_grid_count,
            'used_rbs': len(rb_occupancy),
            'reused_rbs': sum(
                1 for occupancy in rb_occupancy.values()
                if occupancy > 1),
            'max_rb_occupancy': (
                max(rb_occupancy.values()) if rb_occupancy else 0),
            'converged': it + 1 < max_iter,
        }
        return self.strategies
