from opencda.customize.core.clustering.clustering_algorithm_manager import ClusterResourceAllocationAlgorithm
import opencda.customize.core.clustering.algorithm.common as common
from opencda.customize.core.clustering.algorithm.common import *
from opencda.log.logger_config import logger

class PotentialGame(ClusterResourceAllocationAlgorithm):
    def __init__(self, cav_world):
        super().__init__(cav_world)
        # need Vehicle_Grid as input
        self.p = Params()
        self.strategies = {}  # {head_id: (member_id, subchannel_k, time_slot_t, [grid_ids])}
        self.d_s_cache = {}  # (cluster_head_id, member_id) -> delay
        self.channel_cache_index = defaultdict(set)  # subchannel_id -> set of cache keys
        self.grids_uploading = set()
        self.grids_ch_sens = set()
    
    def set_clusters(self, clusters):
        super().set_clusters(clusters)
        self.max_grids_per_rb = self.calculate_max_grids_per_rb()
        logger.info(f"grid_bits: {self.clusters[0].grid_bits}")
        logger.info(f"max_grids_per_rb: {self.max_grids_per_rb}")
        self.compute_grids_ch_sens()

    def run(self):
        ret = self.channel_game()
        self.update_resource_allocation_strategy()
        return ret
    
    def update_resource_allocation_strategy(self):
        vehicle_dict = self.cav_world.get_vehicle_managers()
        for hid, links in self.strategies.items():
            for mid, k, t, grid_ids in links:
                schedule = {(mid, hid): k}
                vehicle_dict[hid].v2x_manager.scheduler.set_strategies(schedule)
                grids_selection = {mid: grid_ids}
                vehicle_dict[hid].perception_manager.co_manager.set_grid_selection(grids_selection)


    def calculate_max_grids_per_rb(self, sinr=None):
        return common.calculate_max_grids_per_rb(sinr, self.p.bandwidth_per_channel, self.p.T_ddl, self.clusters[0].grid_bits)

    @staticmethod
    def extract_used_channels(schedule):
        return set(sc for (_, sc, _, _) in schedule)

    def invalidate_channel_cache(self, affected_channels):
        for sc in affected_channels:
            keys = self.channel_cache_index.get(sc, set())
            for key in keys:
                self.d_s_cache.pop(key, None)
            self.channel_cache_index.pop(sc, None)

    def grid_late_utility(self, grid_id, participating_clusters, in_J_eff=False):
        # 计算某网格由参与簇集合提供晚期融合的总效用
        n = len(participating_clusters)
        if n == 0:
            return 0.0

        # soft-winner 补充项
        if in_J_eff:
            U_soft = (1.0 - self.p.bar_p)
        else:
            U_soft = 1.0
        U_soft = U_soft * (1 - (1 - self.p.bar_p) ** n) * \
            sum([self.p.bar_mu(self.compute_delay(cluster, grid_id=grid_id)) for cluster in participating_clusters]) / n
        
        # pairwise 替代项
        U_pairwise = 0.0
        if in_J_eff:
            for k, l in combinations(participating_clusters, 2):
                mu_k = self.p.bar_mu(self.compute_delay(k, grid_id=grid_id))
                mu_l = self.p.bar_mu(self.compute_delay(l, grid_id=grid_id))
                win_prob = norm.cdf((mu_k - mu_l) / math.sqrt(self.p.s * (mu_k + mu_l)))
                U_pairwise += win_prob * (mu_k - mu_l)
        
        U_Fp = n * self.p.fp_penalty
        # print(f"U_soft: {U_soft}, U_pairwise: {U_pairwise}, U_Fp: {U_Fp}")
        return U_soft + U_pairwise - U_Fp
    
    def member_early_utility(self, cluster, member_id):
        # 计算某网格由参与簇集合提供早期融合的总效用
        d_s = self.compute_delay(cluster, member_id=member_id)
        U_early = self.p.bar_lambda * self.p.bar_mu(d_s)
        return U_early
    
    def grid_early_utility(self, cluster, grid_id):
        # 计算某网格由参与簇集合提供早期融合的总效用
        d_s = self.compute_delay(cluster, grid_id=grid_id)
        U_early = self.p.bar_lambda * self.p.bar_mu(d_s)
        return U_early
        
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
        if cache_key in self.d_s_cache:
            return self.d_s_cache[cache_key]
        data_rate = self.compute_data_rate(cluster, mid, subchannel_id)
        d_s = self.transmission_delay(data_rate, cluster, grids_num)
        self.d_s_cache[cache_key] = d_s
        self.channel_cache_index[subchannel_id].add(cache_key)
        print(f"data_rate: {data_rate}, grids_num: {grids_num}, d_s: {d_s}")
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
    
    def get_participating_clusters(self, grid_id):
        participating_clusters = []
        for cluster in self.clusters:
            for (_, _, _, grids) in self.strategies.get(cluster.head_id, []):
                if grid_id in grids:
                    participating_clusters.append(cluster)
                    break
        return participating_clusters
    
    def bits_to_sinr(self, bits):
        data_rate = bits / self.p.T_ddl
        sinr = 2 ** (data_rate / self.p.bandwidth_per_channel) - 1
        return sinr

    def compute_grids_uploading(self):
        self.grids_uploading = set()
        for hid, links in self.strategies.items():
            grids_density = defaultdict(int)
            for (mid, sc, t, grids) in links:
                for grid_id in grids:
                    grids_density[grid_id] += common.global_vehicles[mid].grid_density_dict.get(grid_id, 0.0)
            for grid_id in grids_density.keys():
                grids_density[grid_id] += common.global_vehicles[hid].grid_density_dict.get(grid_id, 0.0)

            grids = set([grid_id for grid_id, density in grids_density.items() if density >= common.global_vehicles[hid].rho_th])
            self.grids_uploading |= grids
    
    def compute_grids_uploading_inside_cluster(self, cluster):
        hid = cluster.head_id
        links = self.strategies.get(hid, None)
        grids_density = common.global_vehicles[hid].grid_density_dict.copy()
        for (mid, sc, t, grids) in links:
            for grid_id in grids:
                density = grids_density.get(grid_id, 0.0)
                grids_density[grid_id] = density + common.global_vehicles[mid].grid_density_dict.get(grid_id, 0.0)
        grids = set([grid_id for grid_id, density in grids_density.items() if density >= common.global_vehicles[hid].rho_th])
        return grids
    
    def compute_grids_ch_sens(self):
        self.grids_ch_sens = set()
        for cluster in self.clusters:
            head_id = cluster.head_id
            # self.grids_ch_sens |= common.global_vehicles[head_id].sens_grids
            self.grids_ch_sens |= common.global_vehicles[head_id].high_density_grids

    def grid_score(self, cluster, grid_id):
        participating_clusters = self.get_participating_clusters(grid_id)
        late_score_upload = self.grid_late_utility(grid_id, participating_clusters, in_J_eff=True)
        # print("late_score_upload", late_score_upload)
        late_score_not_upload = self.grid_late_utility(grid_id, participating_clusters, in_J_eff=False)
        # print("late_score_not_upload", late_score_not_upload)
        early_score = self.grid_early_utility(cluster, grid_id)
        # print("early_score", early_score)
        gain = early_score + late_score_upload - late_score_not_upload
        # print("gain", gain)
        return gain

    def best_response(self, cluster, strategies, global_rb_used):
        """
        近似最优 best-response
        - 每个簇头最多使用 B_h 个 RB
        - 两轮调度：exclusive → reuse
        """
        h = cluster.head_id
        # B_h = self.p.N_max  # 或者单独设一个 B_h
        B_h = 2
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
        # 按 member 能提供的“未上传 grid 数”排序
        member_grid_map = {}
        for m in cluster.members:
            if m == h:
                continue
            grids_m = common.global_vehicles[m].sens_grids & candidate_grids
            if grids_m:
                member_grid_map[m] = grids_m

        member_grid_score_map = {} # 记录每个member对每个grid的score
        for m, grids_m in member_grid_map.items():
            member_grid_score_map[m] = {}
            for grid in grids_m:
                member_grid_score_map[m][grid] = self.grid_score(cluster, grid)

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

                logger.info(f"Cluster head {h} existing link from {mid_old} SINR: {sinr} signal_power: {signal_power} interf: {interf}")
                if sinr < sinr_min_multiplex:
                    logger.info(f"Cluster head {h} cannot reuse RB {(k, 0)} due to insufficient SINR on existing link from {mid_old}")
                    continue
                
                max_grids_possible = math.floor(
                    calculate_available_data_rate(self.p.bandwidth_per_channel, sinr) * self.p.T_ddl / cluster.grid_bits
                )
                logger.info(f"Cluster head {h} existing link from {mid_old} max_grids_possible: {max_grids_possible}")

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
                    if hid == h:
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
        for it in range(max_iter):
            updated = False
            for cluster in self.clusters:
                h = cluster.head_id
                new_schedule = self.best_response(cluster, self.strategies, global_rb_used)
                if len(new_schedule) > 0:
                     # 计算受影响子信道, 清除延迟缓存
                    # old_channels = self.extract_used_channels(self.strategies[h])
                    # new_channels = self.extract_used_channels(new_schedule)
                    # affected_channels = old_channels | new_channels
                    # self.invalidate_channel_cache(affected_channels)
                    self.strategies[h] += new_schedule
                    logger.info(f"Cluster head {h} strategy updated.")
                    updated = True
                else:
                    logger.info(f"Cluster head {h} strategy unchanged.")
            if not updated:
                break
        logger.info(f"Channel game converged in {it+1} iterations.")
        logger.info(self.strategies)
        return self.strategies