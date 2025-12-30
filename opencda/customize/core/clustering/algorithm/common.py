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

global_vehicles = {}  # vid -> Vehicle object
global_vms = {}  # vid -> V2XManager object
global_ego_id = 0

class Vehicle:
    def __init__(self, vid, position, speed, direction):
        self.id = vid
        self.get_position = position
        self.get_speed = speed
        self.get_direction = direction

    @staticmethod
    def initialize(cav_world):
        global global_vehicles, global_vms, global_ego_id
        global_ego_id = cav_world.ego_id
        vehicle_manager_dict = cav_world.get_vehicle_managers()
        global_vms = {vid: vm.v2x_manager for vid, vm in vehicle_manager_dict.items() if vm.is_ok}
        for vid, vm in global_vms.items() :
            if not vm.is_ok or vid in global_vehicles:
                continue
            position = vm.get_ego_pos
            speed = vm.get_ego_speed
            direction = vm.get_ego_dir
            global_vehicles[vid] = Vehicle(vid, position, speed, direction)
        return global_vehicles, global_vms

    def summary_state(self):
        return {
            'position': self.get_position().location,
            'speed': self.get_speed(),
            'direction': self.get_direction(),
        }

###########################
# Grid based vehicle class
###########################
class Vehicle_Grid(Vehicle):
    def __init__(self, vid, position, speed, direction, sens_grids, req_grids, high_density_grids,grid_size, rho_th, grid_density_dict):
        super().__init__(vid, position, speed, direction)
        self.sens_grids = sens_grids  # J_i^{sens}
        self.req_grids = req_grids  # J_i^{req}
        self.high_density_grids = high_density_grids
        self.grid_size = grid_size #meter
        self.rho_th = rho_th # 点云密度阈值
        self.grid_density_dict = grid_density_dict  # grid_id -> density

    @staticmethod
    def initialize(cav_world):
        global global_vehicles, global_vms, global_ego_id
        global_ego_id = cav_world.ego_id
        vehicle_manager_dict = cav_world.get_vehicle_managers()
        print('vehicle_manager_dict:', vehicle_manager_dict.keys())
        global_vms = {vid: vm.v2x_manager for vid, vm in vehicle_manager_dict.items() if vm.is_ok}
        for vid, vm in vehicle_manager_dict.items():
            if not vm.is_ok or vid in global_vehicles:
                continue
            position = vm.v2x_manager.get_ego_pos
            speed = vm.v2x_manager.get_ego_speed
            direction = vm.v2x_manager.get_ego_dir
            sens_grids = vm.perception_manager.lidar.sens_grids
            high_density_grids = vm.perception_manager.lidar.high_density_grids
            req_grids = vm.perception_manager.lidar.req_grids
            grid_size = vm.perception_manager.lidar.grid_size
            rho_th = vm.perception_manager.lidar.density_threshold
            grid_density_dict = vm.perception_manager.lidar.grid_density_dict
            global_vehicles[vid] = Vehicle_Grid(vid, position, speed, direction, sens_grids, req_grids, high_density_grids, grid_size, rho_th, grid_density_dict)
        return global_vehicles, global_vms
###########################
# Coalition class(Grid based)
###########################
class SimpleCluster:
    def __init__(self, members):
        self.members = set(members)
        self.head_id = min(self.members)

POINT_BIT = 128  # 每个点的比特数(XYZ+intensity, 4 * 4 * 8)
class Cluster(SimpleCluster):
    def __init__(self, members):
        self.members = set(members)
        self.req_grids = self.get_req_grids()
        self.sens_grids = self.get_sens_grids()
        self.head_id = self.elect_head()
        self.grid_bits = self.compute_grid_bits()
        self.high_density_grids = self.get_high_density_grids()

    def size(self):
        return len(self.members)

    def compute_grid_bits(self):
        return global_vehicles[self.head_id].grid_size ** 2 * global_vehicles[self.head_id].rho_th * POINT_BIT  # bits

    def get_sens_grids(self):
        grids = set()
        for vid in self.members:
            grids |= global_vehicles[vid].sens_grids
        return grids
    
    def get_req_grids(self):
        grids = set()
        for vid in self.members:
            grids |= global_vehicles[vid].req_grids
        return grids
    
    def get_high_density_grids(self):
        grids = set()
        for vid in self.members:
            grids |= global_vehicles[vid].high_density_grids
        return grids

    def add_member(self, vid):
        self.members.add(vid)
        self.req_grids |= global_vehicles[vid].req_grids
        self.sens_grids |= global_vehicles[vid].sens_grids
        self.high_density_grids |= global_vehicles[vid].high_density_grids
    
    def remove_member(self, vid):
        self.members.remove(vid)
        self.req_grids = self.get_req_grids()
        self.sens_grids = self.get_sens_grids()
        self.high_density_grids = self.get_high_density_grids()
        if vid == self.head_id and len(self.members) > 0:
            self.elect_head()

    def elect_head(self):
        self.head_id = min(self.members)
        return self.head_id
    
class Coalition(Cluster):
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
    
def density_score(density, rho_th):
    if density >= rho_th:
        return 1.0
    else:
        return density / rho_th

def avg_grids_score(vid, grid_set):
    vehicle = global_vehicles[vid]
    score = sum([density_score(vehicle.grid_density_dict.get(grid_id, 0.0), vehicle.rho_th) for grid_id in grid_set])
    return score
    
###########################
# Utility functions
###########################
def calculate_cos(direction1, direction2):
    """计算两方向向量的余弦相似度"""
    dot_product = direction1[0]*direction2[0] + direction1[1]*direction2[1] + direction1[2]*direction2[2]
    magnitude1 = math.sqrt(sum(d**2 for d in direction1))
    magnitude2 = math.sqrt(sum(d**2 for d in direction2))
    return dot_product / (magnitude1 * magnitude2 + 1e-4)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + math.exp(-x))

def compute_spatiotemporal_distance(ego_data, neighbor_data):
    """计算时空相似性"""
    # 空间距离项
    distance = compute_distance(ego_data['position'], neighbor_data['position'])
    distance_term = math.exp(distance / 40.0)  # Using default d0 value of 50m
    
    # 计算真正的速度向量（速度大小 × 标准化方向向量）
    # 将速度标量和方向向量结合成真正的速度向量
    ego_speed_vector = [ego_data['speed'] * comp for comp in ego_data['direction']]
    neighbor_speed_vector = [neighbor_data['speed'] * comp for comp in neighbor_data['direction']]
    
    # 计算速度向量之间的差异（欧几里得距离）
    velocity_diff_squared = sum((a - b) ** 2 for a, b in zip(ego_speed_vector, neighbor_speed_vector))
    velocity_diff = math.sqrt(velocity_diff_squared)
    
    # 速度相似项（基于速度向量差异）
    speed_term = math.exp(velocity_diff / 20.0)  # Using default s0 value of 10 m/s
    
    # Combine terms
    similarity = (distance_term + speed_term) / 2
    # print(f"distance: {distance:.3f}, distance_term: {distance_term:.3f}, velocity_diff: {velocity_diff:.3f}, speed_term: {speed_term:.3f}, similarity: {similarity:.3f}")
    return similarity

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

def calculate_max_grids_per_rb(sinr=None, bandwidth_per_channel=None, T_ddl=None, grid_bits=None):
    if sinr is None:
        data_rate = calculate_data_rate_with_0_interference(bandwidth_per_channel)
        # print("data_rate_no_interference:", data_rate)
    else:
        data_rate = calculate_available_data_rate(bandwidth_per_channel, sinr)
    return math.floor(data_rate * T_ddl / grid_bits)
