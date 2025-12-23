from collections import defaultdict
import math
from itertools import combinations

from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from opencda.core.common.cav_world import CavWorld
from opencda.customize.core.v2x.utils import *
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

global_cav_world = None  # type: CavWorld
vehicles = {}  # vid -> Vehicle object
vms = {}  # vid -> V2XManager object

class Vehicle:
    def __init__(self, vid, position, speed, sens_grids, req_grids, high_density_grids,grid_size, rho_th, grid_density_dict):
        self.id = vid
        self.get_position = position
        self.get_speed = speed
        self.sens_grids = set(sens_grids)  # J_i^{sens}
        self.req_grids = set(req_grids)  # J_i^{req}
        self.high_density_grids = set(high_density_grids)
        self.grid_size = grid_size #meter
        self.rho_th = rho_th # 点云密度阈值
        self.grid_density_dict = grid_density_dict  # grid_id -> density

def initialize_vehicles(cav_world):
    global global_cav_world, vehicles, vms
    global_cav_world = cav_world
    vehicle_manager_dict = cav_world.get_vehicle_managers()
    vms = {vid: vm.v2x_manager for vid, vm in vehicle_manager_dict.items() if vm.is_ok}
    for vid, vm in vehicle_manager_dict.items():
        if not vm.is_ok:
            continue
        position = vm.v2x_manager.get_ego_pos
        speed = vm.v2x_manager.get_ego_speed
        sens_grids = vm.perception_manager.lidar.sens_grids
        high_density_grids = vm.perception_manager.lidar.high_density_grids
        req_grids = vm.perception_manager.lidar.req_grids
        grid_size = vm.perception_manager.lidar.grid_size
        rho_th = vm.perception_manager.lidar.density_threshold
        grid_density_dict = vm.perception_manager.lidar.grid_density_dict
        vehicles[vid] = Vehicle(vid, position, speed, sens_grids, req_grids, high_density_grids, grid_size, rho_th, grid_density_dict)
    return vehicles

def density_score(density, rho_th):
    if density >= rho_th:
        return 1.0
    else:
        return density / rho_th
    
def avg_grids_score(vid, grid_set):
    vehicle = vehicles[vid]
    score = sum([density_score(vehicle.grid_density_dict.get(grid_id, 0.0), vehicle.rho_th) for grid_id in grid_set])
    return score

POINT_BIT = 128  # 每个点的比特数(XYZ+intensity, 4 * 4 * 8)
class Coalition:
    def __init__(self, members):
        self.members = set(members)
        self.req_grids = self.get_req_grids()
        self.sens_grids = self.get_sens_grids()
        self.head_id = min(self.members)
        self.grid_bits = self.compute_grid_bits()
        self.high_density_grids = self.get_high_density_grids()
        # self.member_contributions = {vid: 0.0 for vid in self.members}

    def size(self):
        return len(self.members)

    def compute_grid_bits(self):
        return vehicles[self.head_id].grid_size ** 2 * vehicles[self.head_id].rho_th * POINT_BIT  # bits

    def get_sens_grids(self):
        grids = set()
        for vid in self.members:
            grids |= vehicles[vid].sens_grids
        return grids
    
    def get_req_grids(self):
        grids = set()
        for vid in self.members:
            grids |= vehicles[vid].req_grids
        return grids
    
    def get_high_density_grids(self):
        grids = set()
        for vid in self.members:
            grids |= vehicles[vid].high_density_grids
        return grids

    def add_member(self, vid):
        self.members.add(vid)
        self.req_grids |= vehicles[vid].req_grids
        self.sens_grids |= vehicles[vid].sens_grids
        self.high_density_grids |= vehicles[vid].high_density_grids
    
    def remove_member(self, vid):
        self.members.remove(vid)
        # self.member_contributions.pop(vid, None)
        self.req_grids = self.get_req_grids()
        self.sens_grids = self.get_sens_grids()
        self.high_density_grids = self.get_high_density_grids()
        if vid == self.head_id and len(self.members) > 0:
            self.elect_head()

    def elect_head(self):
        self.head_id = min(self.members)
        return self.head_id
    
class Cluster(Coalition):
    pass

class Params:
    def __init__(self,
                 rho_th=2.0,
                 kappa=1.0,
                 gamma=0.5,
                 q_max=1.0,
                 s=0.1,
                 alpha=0.15,
                 beta=0.10,
                 delta_v_max=15.0,
                 N_max=4,
                 T_ddl=0.1,
                 ita=1.1,
                 fp_penalty=0.01,
                 bandwidth_all = 72, # MHz
                 num_channels=10,
                 num_time_slots=2
                 ):
        """
        Parameters
        ----------
        rho_th : float
            点云密度阈值 ρ_th
            - 含义：判定“该网格是否形成有效检测输出”的最小点云密度
            - 作用：用于阈值化建模，决定 bar{λ} 与 bar{μ}
            - 合理范围：1 ~ 5（点/平方米，取决于激光雷达分辨率）

        kappa : float
            密度饱和参数 κ
            - 含义：控制 λ(ρ)=λ0(1-exp(-ρ/κ)) 的饱和速度
            - 物理意义：点云密度增加带来收益的“边际递减速度”
            - 合理范围：ρ_th/2 ~ ρ_th

        gamma : float
            延迟衰减系数 γ
            - 含义：早期融合置信度随融合延迟指数衰减的速率
            - 模型位置：μ = q_max * (...) * exp(-γ d_S)
            - 合理范围：0.2 ~ 1.0

        q_max : float
            最大单网格置信度 q_max
            - 含义：理想情况下（ρ→∞，d→0）单网格可达到的最大感知置信度
            - 通常归一化到 [0,1]

        s : float
            效用方差参数 s
            - 含义：用于计算 pairwise 补充项中置信度差的标准差
            - 作用于 norm.cdf(...) 的分母
            - 合理范围：0.05 ~ 0.2

        alpha : float
            簇稳定性成本权重 α
            - 含义：联盟速度不一致性对联盟价值的惩罚权重
            - 越大 → 更偏好速度一致的小簇
            - 合理范围：0.05 ~ 0.3

        beta : float
            通信规模成本权重 β
            - 含义：簇规模增长导致通信瓶颈的惩罚权重
            - 越大 → 更严格限制簇大小
            - 合理范围：0.05 ~ 0.3

        delta_v_max : float
            最大速度差 Δv_max（归一化基准）
            - 含义：认为“完全不可接受”的簇内最大速度差
            - 用于 C_stab(S) = max(1, Δv(S)/Δv_max)
            - 单位：m/s
            - 合理范围：10 ~ 20（城市场景）

        N_max : int
            允许的最大簇规模
            - 含义：通信与调度可接受的最大车辆数
            - 超过该规模联盟价值不再增长
            - 合理范围：2 ~ 8

        T_ddl : float
            统一融合延迟 d_S
            - 含义：一个采样周期内，从数据采集到早期融合完成的总延迟最大值
            - 统一融合延迟假设下，与网格无关
            - 单位：秒
            - 合理范围：0.05 ~ 0.2
        
        fp_penalty : float
            误检惩罚项
            - 含义：对误检情况的固定惩罚，防止过度补充
            - 作用于每个网格的效用计算中
            - 合理范围：0.5 ~ 2.0
        """

        self.rho_th = rho_th
        self.kappa = kappa
        self.gamma = gamma
        self.q_max = q_max
        self.s = s
        self.alpha = alpha
        self.beta = beta
        self.delta_v_max = delta_v_max
        self.N_max = N_max
        self.T_ddl = T_ddl
        self.ita = ita
        self.fp_penalty = fp_penalty # 误检惩罚项

        # 阈值化参数
        self.bar_lambda = 1 - math.exp(-self.rho_th / self.kappa)

        self.bar_p = 1 - math.exp(-self.bar_lambda)

        # 信道参数
        self.num_channels = num_channels      # 子信道数量 K
        self.num_time_slots = num_time_slots    # 时隙数量 T
        self.channel_capacity = 2   # 每个子信道的并发容量 C
        self.bandwidth_all = bandwidth_all * (10**6)  # 总带宽 72 MHz
        self.bandwidth_per_channel = self.bandwidth_all / self.num_channels  # 每个子信道带宽

    def bar_mu(self, d_S):
            return self.q_max \
            * self.rho_th / (self.rho_th + self.kappa) \
            * math.exp(-self.gamma * d_S)

####################################################################################################


class CoalitionValuation:
    def __init__(self, params):
        self.p = params
    
    def stability_cost(self, coalition):
        speeds = [vehicles[v].get_speed() for v in coalition.members]
        if len(speeds) <= 1:
            return 1.0
        delta_v = max(speeds) - min(speeds)
        return max(1.0, delta_v / self.p.delta_v_max)
    
    def comm_cost(self, coalition):
        return max(1.0, coalition.size() / self.p.N_max)
    
    def marginal_contribution(self, coalition, vid):
        if vid in coalition.members:
            return 0.0
        # 新增网格带来的收益
        # new_grids = vehicles[vid].sens_grids - coalition.sens_grids
        new_grids = vehicles[vid].sens_grids - coalition.high_density_grids
        new_grids = new_grids & (coalition.req_grids)  # | vehicles[vid].req_grids)
        # grid_gain = len(new_grids) * self.p.bar_lambda * self.p.bar_mu(self.p.T_ddl)
        grid_gain = avg_grids_score(vid, new_grids)
        # 成本变化
        stab_diff = self.stability_cost(Coalition(coalition.members | {vid})) - self.stability_cost(coalition)
        comm_diff = self.comm_cost(Coalition(coalition.members | {vid})) - self.comm_cost(coalition)
        # print(f"vid {vid} grid_gain: {grid_gain}, stab_diff: {stab_diff}, comm_diff: {comm_diff} from coalition {coalition.members}")
        return grid_gain - self.p.alpha * stab_diff - self.p.beta * comm_diff
    
    def current_contribution(self, coalition, vid):
        if vid not in coalition.members:
            return 0.0
        coalition.remove_member(vid)
        ret = self.marginal_contribution(coalition, vid)
        coalition.add_member(vid)
        return ret

class CoalitionGame:
    def __init__(self, cav_world=None):
        initialize_vehicles(cav_world)
        self.params = Params()
        self.valuation = CoalitionValuation(self.params)
        self.coalitions = []

    def run(self, max_iter=20):
        return self.coalition_formation(max_iter)

    def check_is_ok(self):
        vehicle_manager_dict = global_cav_world.get_vehicle_managers()
        for vid, vm in vms.items():
            if not vehicle_manager_dict[vid].is_ok:
                print(f"vehicle {vid} left the map.")
                vehicles.pop(vid, None)
                for coalition in self.coalitions:
                    if vid in coalition.members:
                        current = coalition
                        current.remove_member(vid)
                        if current.size() == 0:
                            self.coalitions.remove(current)

    def coalition_formation(self, max_iter=20):
        self.check_is_ok()
        if not self.coalitions:
            print("Initializing coalitions...")
            self.coalitions = [Coalition({vid}) for vid in vehicles.keys()]
        for _ in range(max_iter):
            print("--- Coalition Formation Iteration", _+1, "---")
            updated = False
            for vid in vehicles.keys():
                current = next(c for c in self.coalitions if vid in c.members)
                current_contribution = self.valuation.current_contribution(current, vid)
                print(f"Vehicle {vid} current coalition {current.members} contribution: {current_contribution:.3f}.")
                best_delta = current_contribution
                best_coalition = current
                for c in self.coalitions:
                    if c is current:
                        continue
                    if c.size() >= self.params.N_max: #TODO: bad if-statement
                        # print(f"Vehicle {vid} cannot join coalition {c.members} due to size limit.")
                        continue
                    delta = self.valuation.marginal_contribution(c, vid)
                    if delta > best_delta:
                        best_delta = delta
                        best_coalition = c
                if best_coalition is not current and best_delta > current_contribution * self.params.ita:
                    print(f"Vehicle {vid} moves from coalition {current.members}(contribution: {current_contribution:.3f}) to {best_coalition.members} with contribution {best_delta:.3f}.")
                    current.remove_member(vid)
                    best_coalition.add_member(vid)
                    if current.size() == 0:
                        self.coalitions.remove(current)
                    updated = True
            if not updated:
                break
        print(f"Coalition formation converged in {_+1} iterations.")
        for coalition in self.coalitions:
            print(f"[{coalition.members}]")
        return self.coalitions


####################################################################################################

class Cluster:
    def __init__(self, head_id, members):
        self.head = head_id
        self.members = set(members)

class ChannelEnv:
    def __init__(self, K, T, C_k):
        """
        K: 子信道数量
        T: 时隙数量
        C_k: 每个子信道的并发容量
        """
        self.K = K
        self.T = T
        self.C_k = C_k

class ClusterUtility:
    def __init__(self, params):
        self.p = params

class PotentialGame:
    def __init__(self, cav_world=None, clusters=None):
        initialize_vehicles(cav_world)
        self.params = Params()
        self.p = self.params
        self.channel_env = ChannelEnv(self.params.num_channels, self.params.num_time_slots, self.params.channel_capacity)
        self.utility_model = ClusterUtility(self.params)
        self.clusters = clusters
        self.strategies = {}  # {head_id: (member_id, subchannel_k, time_slot_t, [grid_ids])}
        self.d_s_cache = {}  # (cluster_head_id, member_id) -> delay
        self.channel_cache_index = defaultdict(set)  # subchannel_id -> set of cache keys

        # algorithm related
        data_rate_no_interference = calculate_data_rate_with_0_interference(self.p.bandwidth_per_channel)  # 理想情况下的最大速率
        self.max_grids_per_rb = math.floor(
            data_rate_no_interference * self.p.T_ddl / self.clusters[0].grid_bits
        )
        print("grid_bits:", self.clusters[0].grid_bits)
        print("data_rate_no_interference:", data_rate_no_interference)
        print("max_grids_per_rb:", self.max_grids_per_rb)
        self.grids_uploading = set()
        self.grids_ch_sens = set()
        self.compute_grids_ch_sens()

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
        print(f"U_soft: {U_soft}, U_pairwise: {U_pairwise}, U_Fp: {U_Fp}")
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
        print(f"U_early: {U_early}")
        return U_early
        
    def compute_delay(self, cluster, member_id=None, grid_id=None):
        return self.p.T_ddl
        subchannel_id = -1
        grids_num = 0
        for (mid, sc, t, grids) in self.strategies[cluster.head_id]:
            if (member_id is not None and mid == member_id) or (grid_id is not None and grid_id in grids):
                member_id = mid
                grids_num = len(grids)
                subchannel_id = sc
                break
        cache_key = (cluster.head_id, member_id, subchannel_id)
        if cache_key in self.d_s_cache:
            return self.d_s_cache[cache_key]
        data_rate = self.compute_data_rate(cluster, member_id, subchannel_id)
        d_s = self.transmission_delay(data_rate, cluster, grids_num)
        self.d_s_cache[cache_key] = d_s
        self.channel_cache_index[subchannel_id].add(cache_key)
        return d_s
    
    def run(self, max_iter=20):
        return self.channel_game(max_iter)
    
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
                vm = vms[mid]
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
        src_vm = vms[member_id]
        dst_vm = vms[cluster_h.head_id]

        tx_power_w = dB_to_linear(src_vm.tx_power) / 1000.0  # dBm→W
        ch_gain = calculate_channel_gain(calculate_distance(src_vm, dst_vm))
        noise_power_w = dB_to_linear(dst_vm.noise_power) / 1000.0  # dBm→W
        interference_power_w = self.get_interference(dst_vm, subchannel=subchannel_id, exclude_vid=member_id)
        sinr_linear = calculate_sinr_linear(tx_power_w * ch_gain, interference_power_w, noise_power_w)
        data_rate = calculate_available_data_rate(self.params.bandwidth_per_channel, sinr_linear)
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
        data_rate = bits / self.params.T_ddl
        sinr = 2 ** (data_rate / self.params.bandwidth_per_channel) - 1
        return sinr

    def compute_grids_uploading(self):
        self.grids_uploading = set()
        for hid, links in self.strategies.items():
            grids_density = defaultdict(int)
            for (mid, sc, t, grids) in links:
                for grid_id in grids:
                    grids_density[grid_id] += vehicles[mid].grid_density_dict.get(grid_id, 0.0)
            for grid_id in grids_density.keys():
                grids_density[grid_id] += vehicles[hid].grid_density_dict.get(grid_id, 0.0)

            grids = set([grid_id for grid_id, density in grids_density.items() if density >= vehicles[hid].rho_th])
            self.grids_uploading |= grids
    
    def compute_grids_uploading_inside_cluster(self, cluster):
        hid = cluster.head_id
        links = self.strategies.get(hid, None)
        grids_density = vehicles[hid].grid_density_dict.copy()
        for (mid, sc, t, grids) in links:
            for grid_id in grids:
                density = grids_density.get(grid_id, 0.0)
                grids_density[grid_id] = density + vehicles[mid].grid_density_dict.get(grid_id, 0.0)
        grids = set([grid_id for grid_id, density in grids_density.items() if density >= vehicles[hid].rho_th])
        return grids
    
    def compute_grids_ch_sens(self):
        self.grids_ch_sens = set()
        for cluster in self.clusters:
            head_id = cluster.head_id
            # self.grids_ch_sens |= vehicles[head_id].sens_grids
            self.grids_ch_sens |= vehicles[head_id].high_density_grids

    def grid_score(self, cluster, grid_id):
        participating_clusters = self.get_participating_clusters(grid_id)
        late_score_upload = self.grid_late_utility(grid_id, participating_clusters, in_J_eff=True)
        print("late_score_upload", late_score_upload)
        late_score_not_upload = self.grid_late_utility(grid_id, participating_clusters, in_J_eff=False)
        print("late_score_not_upload", late_score_not_upload)
        early_score = self.grid_early_utility(cluster, grid_id)
        print("early_score", early_score)
        gain = early_score + late_score_upload - late_score_not_upload
        print("gain", gain)
        return gain

    def best_response(self, cluster, strategies, global_rb_used):
        """
        近似最优 best-response
        - 每个簇头最多使用 B_h 个 RB
        - 两轮调度：exclusive → reuse
        """
        h = cluster.head_id
        # B_h = self.params.N_max  # 或者单独设一个 B_h
        B_h = 2
        schedule = []

        # ========= Step 0: 准备候选 grids =========

        # 本簇可选 grids
        print("Cluster head", h, "req grids:", len(cluster.req_grids))
        # print("Cluster head", h, "req grids:", (cluster.req_grids))
        # candidate_grids = cluster.req_grids.copy()
        # candidate_grids -= self.grids_ch_sens 
        # candidate_grids -= self.grids_uploading
        J_req = cluster.req_grids #J_h^req
        J_eff = self.compute_grids_uploading_inside_cluster(cluster) #J_h^eff
        # 下面挑选grids扩充 J_eff
        candidate_grids = J_req - J_eff

        print("Cluster head", h, "candidate grids:", len(candidate_grids))
        # print("Cluster head", h, "grids_ch_sens grids:", (vehicles[h].sens_grids))
        if cluster == self.clusters[0]:
            visualize_grid_set(candidate_grids, title=f"Cluster head {h} candidate grids")
            visualize_grid_set(J_req, title=f"Cluster requested grids")
            visualize_grid_set(J_eff, title=f"Cluster head {h} collected high density grids")
            visualize_grid_set(self.grids_ch_sens, title=f"All cluster heads' high density grids")
            visualize_grid_set(self.grids_uploading, title=f"All uploading high density grids")
        if not candidate_grids:
            return []

        # ========= Step 1: 第一轮（exclusive grid） =========
        # 按 member 能提供的“未上传 grid 数”排序
        member_grid_map = {}
        for m in cluster.members:
            if m == h:
                continue
            grids_m = vehicles[m].sens_grids & candidate_grids
            if grids_m:
                member_grid_map[m] = grids_m

        members_sorted = sorted(
            member_grid_map.keys(),
            # key=lambda m: avg_grids_score(m, member_grid_map[m]),
            # 根据候选网格数*点云密度加权排序
            key=lambda m: sum([self.grid_score(cluster, grid) for grid in member_grid_map[m]]),
            reverse=True
        )

        if cluster == self.clusters[0]:
            for m in members_sorted:
                print("Cluster head", h, "member", m, "can provide candidate grids:", len(member_grid_map[m]))
                visualize_grid_set(vehicles[m].sens_grids, title=f"Cluster head {h} member {m} sensed grids", rho_th=vehicles[m].rho_th, grid_density_dict=vehicles[m].grid_density_dict)
                visualize_grid_set(member_grid_map[m], title=f"Cluster head {h} member {m} candidate grids", rho_th=vehicles[m].rho_th, grid_density_dict=vehicles[m].grid_density_dict)

        used_rbs = 0
        for m in members_sorted:
            if used_rbs >= B_h:
                break

            # 找一个可用 RB
            for k in range(self.channel_env.K):
                if global_rb_used[(k, 0)] >= 1:
                    continue

                # 分配 grid
                grids = list(member_grid_map[m])[:self.max_grids_per_rb]
                if not grids:
                    continue

                schedule.append((m, k, 0, grids))
                print("schedule:", schedule)
                self.grids_uploading |= set(grids)
                global_rb_used[(k, 0)] += 1
                used_rbs += 1
                print("Cluster head", h, "assign exclusive RB", (k, 0), "to member", m, "grids:", grids)
                # 更新占用
                for g in grids:
                    candidate_grids.discard(g)
                break

        # ========= Step 2: 第二轮（受保护 RB 复用） =========
        multiplex_bits = 0.10 * self.max_grids_per_rb
        sinr_min_multiplex = self.bits_to_sinr(multiplex_bits)
        print("Cluster head", h, "sinr_min_multiplex:", sinr_min_multiplex)

        for k in range(self.channel_env.K):
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
            if not existing_links or len(existing_links) >= self.channel_env.C_k or used_grids >= self.max_grids_per_rb * 0.80:
                continue
            
            print("Cluster head", h, "consider reusing RB", (k, 0), "with existing links:", existing_links)
            # ---- Step 2.1：检查已有发送方 → 当前簇头的干扰 ----
            feasible = True
            h_vm = vms[h]

            for (mid, hid, grids_old) in existing_links: #existing_links列表只会有一个元素
                s_vm = vms[mid]
                interf = get_interference_contribution(s_vm, h_vm)

                # 当前簇头接收信号功率（假设无干扰发送）
                signal_power = dB_to_linear(s_vm.tx_power) / 1000.0 * calculate_channel_gain(10)
                noise_power = dB_to_linear(h_vm.noise_power) / 1000.0  # dBm→W

                sinr = calculate_sinr_linear(
                    signal_power,
                    interf,
                    noise_power
                )

                print("Cluster head", h, "existing link from", mid, "SINR:", sinr, "signal_power:", signal_power, "interf:", interf)
                if sinr < sinr_min_multiplex:
                    print("Cluster head", h, "cannot reuse RB", (k, 0), "due to insufficient SINR on existing link from", mid)
                    feasible = False
                    break
                
                max_grids_possible = math.floor(
                    calculate_available_data_rate(self.p.bandwidth_per_channel, sinr) * self.p.T_ddl / cluster.grid_bits
                )
                print("Cluster head", h, "existing link from", mid, "max_grids_possible:", max_grids_possible)

                if not feasible:
                    continue

                # ---- Step 2.2：选择对外簇最友好的 member ----
                best_m = None
                best_gain = -1

                interf = get_interference_contribution(s_vm, h_vm)
                for m in cluster.members:
                    grids_m = vehicles[m].sens_grids & candidate_grids
                    if not grids_m:
                        print("Cluster head", h, "member", m, "has no candidate grids.")
                        print("candidate_grids:", len(candidate_grids))
                        print("vehicles[m].sens_grids:", len(vehicles[m].sens_grids))
                        # visualize_grid_set(vehicles[m].sens_grids, title=f"Member {m} sensed grids")
                        continue

                    m_vm = vms[m]
                    conflict = False

                    t_vm = vms[hid]

                    # 若接收方是当前簇头，不必判冲突
                    if hid == h:
                        continue

                    # 原链路所需 SINR
                    bits = len(grids_old) * cluster.grid_bits
                    sinr_min = self.bits_to_sinr(bits)
                    print("Cluster head", h, "checking conflict for member", m, "on existing link to", hid, "required SINR:", sinr_min)

                    signal_power = dB_to_linear(m_vm.tx_power) / 1000.0 * calculate_channel_gain(calculate_distance(m_vm, t_vm))
                    noise_power = dB_to_linear(t_vm.noise_power) / 1000.0  # dBm→W
                    sinr = calculate_sinr_linear(
                        signal_power,
                        interf,
                        noise_power
                    )

                    print("Cluster head", h, "member", m, "to existing link receiver", hid, "SINR:", sinr)
                    if sinr < sinr_min:
                        conflict = True
                        break

                    if conflict:
                        continue

                    gain = min(len(grids_m), max_grids_possible)
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m

            if best_m is None:
                continue

            # ---- Step 2.3：执行复用 ----
            grids = list(
                vehicles[best_m].sens_grids & candidate_grids
            )[: best_gain]

            if not grids:
                continue

            schedule.append((best_m, k, 0, grids))
            self.grids_uploading |= set(grids)
            global_rb_used[(k, 0)] += 1
            used_rbs += 1
            print("Cluster head", h, "reuse RB", (k, 0), "for member", best_m, "grids:", grids)

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
                print("grids_ch_sens:", len(self.grids_ch_sens))
                print("grids_uploading:", len(self.grids_uploading))
                print("grids_req of", h, ":", len(cluster.req_grids))
                new_schedule = self.best_response(cluster, self.strategies, global_rb_used)
                if len(new_schedule) > 0:
                     # 计算受影响子信道, 清除延迟缓存
                    # old_channels = self.extract_used_channels(self.strategies[h])
                    # new_channels = self.extract_used_channels(new_schedule)
                    # affected_channels = old_channels | new_channels
                    # self.invalidate_channel_cache(affected_channels)
                    self.strategies[h] += new_schedule
                    print(f"Cluster head {h} strategy updated.")
                    updated = True
                else:
                    print(f"Cluster head {h} strategy unchanged.")
            if not updated:
                break
        print(f"Channel game converged in {it+1} iterations.")
        for h, sched in self.strategies.items():
            print(f"Cluster {h} schedule: {sched}")
        return self.strategies
    
def visualize_grid_set(grid_set, title="Grid Visualization", rho_th=None, 
                       grid_density_dict=None, show_coordinates=False):
    """
    Visualize grid cells with optional density coloring.
    
    Args:
        grid_set (set): Set of grid indices (e.g., {"0_0", "1_-1", ...})
        title (str): Title for the plot
        rho_th (float, optional): Density threshold (values at/beyond this will be black)
        grid_density_dict (dict, optional): Dictionary mapping grid coordinates to densities
        show_coordinates (bool): Whether to show coordinate labels on occupied cells
    """
    return
    if not grid_set:
        print("Empty grid set provided")
        return
    # Parse all coordinates and find min/max values
    x_coords = []
    y_coords = []
    invalid_coords = []
    
    for coord in grid_set:
        try:
            x_str, y_str = coord.split('_')
            x = int(x_str)
            y = int(y_str)
            x_coords.append(x)
            y_coords.append(y)
        except (ValueError, AttributeError):
            invalid_coords.append(coord)
    
    if invalid_coords:
        print(f"Skipped {len(invalid_coords)} invalid coordinates (e.g., {invalid_coords[:3]})")
    
    if not x_coords:
        print("No valid coordinates to plot")
        return
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Add some padding around the data
    padding = 2
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    
    # Determine grid dimensions
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    # Initialize density array (0 means no data)
    density_array = np.zeros((width, height))  
    has_density_data = grid_density_dict is not None and rho_th is not None and rho_th > 0
    
    # Create coordinate to index mapping
    coord_to_idx = {(x, y): (x-min_x, y-min_y) for x in range(min_x, max_x+1) 
                   for y in range(min_y, max_y+1)}
    
    # Populate the density array
    for x, y in zip(x_coords, y_coords):
        if (x, y) in coord_to_idx:
            i, j = coord_to_idx[(x, y)]
            if has_density_data and f"{x}_{y}" in grid_density_dict:
                density = grid_density_dict[f"{x}_{y}"]
                if density < 0:
                    density = 0  # Clamp negative values
                density_array[i, j] = min(density, rho_th)  # Cap at rho_th
            else:
                density_array[i, j] = rho_th if rho_th is not None else 1.0  # Default to max if no density data
    # Create a grayscale colormap from white to black
    if has_density_data:
        cmap = LinearSegmentedColormap.from_list('density_cmap', ['white', 'black'])
    else:
        cmap = 'binary'  # Simple binary colormap if no density data
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each grid cell with proper coloring
    for i in range(width):
        for j in range(height):
            if density_array[i, j] > 0:  # Only plot cells that exist in grid_set
                if has_density_data:
                    # Normalize density between 0 and 1
                    norm_density = density_array[i, j] / rho_th
                    color = cmap(norm_density)
                else:
                    color = 'black' if density_array[i, j] > 0 else 'none'
                
                rect = Rectangle((min_x + i - 0.5, min_y + j - 0.5), 
                                1, 1, 
                                facecolor=color,
                                edgecolor='lightgray',
                                linewidth=0.5)
                ax.add_patch(rect)
                
                # Optionally show coordinate labels
                if show_coordinates:
                    lbl_color = 'white' if norm_density > 0.5 else 'black' if has_density_data else 'white'
                    ax.text(min_x + i, min_y + j, 
                           f"{min_x + i}_{min_y + j}", 
                           ha='center', va='center', 
                           color=lbl_color, fontsize=8)
            else:
                # Empty cell grid lines
                rect = Rectangle((min_x + i - 0.5, min_y + j - 0.5), 
                                1, 1, 
                                facecolor='none',
                                edgecolor='lightgray',
                                linewidth=0.5)
                ax.add_patch(rect)
    
    # Set axis limits with padding
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)
    
    # Set ticks at integer positions
    x_ticks = np.arange(min_x, max_x + 1)
    y_ticks = np.arange(min_y, max_y + 1)
    
    # Reduce tick density if there are too many
    if len(x_ticks) > 20:
        x_ticks = np.arange(min_x, max_x + 1, max(1, (max_x - min_x) // 10))
    if len(y_ticks) > 20:
        y_ticks = np.arange(min_y, max_y + 1, max(1, (max_y - min_y) // 10))
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Labels and title
    ax.set_xlabel("X Grid Index")
    ax.set_ylabel("Y Grid Index")
    
    title_str = title
    if rho_th is not None:
        title_str += f" (ρ_thresh = {rho_th})"
    
    ax.set_title(title_str)
    ax.grid(False)  # We're handling grid drawing ourselves
    
    # Add colorbar if we have density data
    if has_density_data:
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                  norm=plt.Normalize(vmin=0, vmax=rho_th))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('Point Cloud Density')
    
    plt.tight_layout()
    plt.show()