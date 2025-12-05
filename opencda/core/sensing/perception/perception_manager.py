# -*- coding: utf-8 -*-
"""
Perception module base.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import weakref
import sys

import carla
import cv2
import numpy as np
import open3d as o3d

from opencda.core.common.cav_world import CavWorld
import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import \
    cal_distance_angle, get_speed, get_speed_sumo
from opencda.core.sensing.perception.coperception_manager import CoperceptionManager
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion, o3d_visualizer_show_coperception, o3d_predict_bbox_to_object

from opencda.core.sensing.perception.coperception_libs import CoperceptionLibs
from collections import OrderedDict, defaultdict

from shapely.geometry import box, Point
from shapely.ops import unary_union


class CameraSensor:
    """
    Camera manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : str
        Indicates the sensor is a front or rear camera. option:
        front, left, right.

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, relative_position, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', '100')

        spawn_point = self.spawn_point_estimation(relative_position,
                                                  global_position)

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.image = None
        self.timstamp = None
        self.frame = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CameraSensor._on_rgb_image_event(
                weak_self, event))

        # camera attributes
        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position

        # this is for rsu. It utilizes global position instead of relative
        # position to the vehicle
        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35

        carla_location = carla.Location(x=carla_location.x + x,
                                        y=carla_location.y + y,
                                        z=carla_location.z + z)

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_rgb_image_event(weak_self, event):
        """CAMERA  method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        # we need to remove the alpha channel
        image = image[:, :, :3]

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp


class LidarSensor:
    """
    Lidar sensor manager.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.

    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))
        blueprint.set_attribute(
            'dropoff_general_rate', str(
                config_yaml['dropoff_general_rate']))
        blueprint.set_attribute(
            'dropoff_intensity_limit', str(
                config_yaml['dropoff_intensity_limit']))
        blueprint.set_attribute(
            'dropoff_zero_intensity', str(
                config_yaml['dropoff_zero_intensity']))
        blueprint.set_attribute(
            'noise_stddev', str(
                config_yaml['noise_stddev']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))
        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
            self.vid = str(vehicle.id)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)
            self.vid = "None"

        # frequency
        self.rotate_frequency = int(config_yaml['rotation_frequency'])
        self.world_slot_seconds = world.get_settings().fixed_delta_seconds
        self.world_frequency = int(1.0 / self.world_slot_seconds)
        self.tick = 0
        self.points_buffer = []
        # lidar data
        self.data = None
        self.timestamp = None
        self.frame = 0
        self.last_rotation_time = None
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LidarSensor._on_data_event(
                weak_self, event))

        # 网格相关参数
        self.enable_grids = config_yaml.get('enable_grids', True)
        self.lidar_range = float(config_yaml['range'])
        self.required_perception_range = self.lidar_range * 2

        self.grid_size = config_yaml.get('grid_size', 10.0)
        self.perception_grids = self._generate_perception_grids()  # {grid_id: shapely.box}
        print(f"grid_size = {self.grid_size}, Generated {len(self.perception_grids)} perception grids.")
        # 核心存储：grid_id -> 本地坐标点云列表
        self.grid_local_points = defaultdict(list)
        # grid_id -> 点云密度（点/平方米）
        self.grid_density_dict = {}
        # 点云密度阈值
        self.density_threshold = 2.0 # config_yaml['points_per_second'] * self.world_slot_seconds / (self.lidar_range * self.lidar_range * 3.14) * 2.0
        print("Density threshold set to:", self.density_threshold)
        #config_yaml.get('density_threshold', 5.0)
    
    def set_enable_grids(self, enable):
        self.enable_grids = enable
        
    def rotate_tick(self):
        self.tick += 1
        if self.tick * self.rotate_frequency >= self.world_frequency:
            self.tick = 0
            return True
        return False
    
    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar  method"""
        self = weak_self()
        if not self:
            return

        # 1. 解析原始点云（传感器本地坐标）
        frame_data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        # frame_data = np.reshape(frame_data, (int(frame_data.shape[0] / 4), 4))  # (N,4) [x,y,z,intensity]
        frame_data = frame_data.reshape(-1, 4)

        # 2. 分离本地坐标和强度，转换全局坐标
        if self.enable_grids:
            local_points = frame_data[:, :3]  # (N,3) 本地坐标
            sensor_transform = self.sensor.get_transform()
            global_points = st.lidar_local_to_global(local_points, sensor_transform)  # 调用工具函数
        
        # 3. 缓存数据
        self.points_buffer.append({
            'local': frame_data, # 本地坐标+强度 (N,4)
            'global':  global_points if self.enable_grids else None # 全局坐标 (N,3)
        })
        self.frame = event.frame
        self.timestamp = event.timestamp

        if self.rotate_tick():
            # 合并缓存数据
            local_list = [item['local'] for item in self.points_buffer]
            global_list = [item['global'] for item in self.points_buffer]
            self.local_data = np.vstack(local_list)  # 本地坐标 (N,4)
            self.global_data = np.vstack(global_list) if self.enable_grids else None  # 全局坐标 (N,3)

            self.data = self.local_data
            self.last_rotation_time = event.timestamp

            if self.enable_grids:
                self.perception_grids = self._generate_perception_grids()
                self.update_grid_local_points()
                self.update_grid_density_dict()
        
            self.points_buffer = []

    def _generate_perception_grids(self):
        """生成统一Grid ID的感知网格（全局坐标系对齐）"""
        grids = {}
        grid_size = self.grid_size
        half_range = self.required_perception_range

        # 计算Lidar在全局坐标系中的初始位置
        sensor_transform = self.sensor.get_transform()
        sensor_x = sensor_transform.location.x
        sensor_y = sensor_transform.location.y

        # 生成覆盖感知范围的网格（全局坐标系）
        start_x = int(np.floor((sensor_x - half_range) / grid_size) * grid_size)
        end_x = int(np.ceil((sensor_x + half_range) / grid_size) * grid_size)
        start_y = int(np.floor((sensor_y - half_range) / grid_size) * grid_size)
        end_y = int(np.ceil((sensor_y + half_range) / grid_size) * grid_size)

        for x in range(start_x, end_x + int(grid_size), int(grid_size)):
            for y in range(start_y, end_y + int(grid_size), int(grid_size)):
                grid_id = self.get_point_grid_id((x, y))
                grid_box = box(x, y, x + grid_size, y + grid_size)
                grids[grid_id] = grid_box

        return grids

    
    def get_point_grid_id(self, point):
        """根据全局坐标点获取统一Grid ID"""
        x, y = point[0], point[1]
        x_idx = int(np.floor(x / self.grid_size))
        y_idx = int(np.floor(y / self.grid_size))
        return f"grid_{x_idx}_{y_idx}"

    def update_grid_local_points(self):
        """更新网格-本地坐标点云映射(全局坐标找Grid ID, 存储本地坐标)"""
        if self.local_data is None or self.global_data is None:
            return
        
        self.grid_local_points.clear()
        local_points = self.local_data  # 本地坐标 (N,4)
        global_points = self.global_data[:, :3]  # 全局坐标 (N,3)
        
        # 遍历每个点：全局坐标找Grid ID，存储本地坐标
        for local_p, global_p in zip(local_points, global_points):
            grid_id = self.get_point_grid_id(global_p)
            if grid_id in self.perception_grids:
                self.grid_local_points[grid_id].append(local_p)

    def get_local_points_by_grid_ids(self, grid_id_list):
        """
        获取指定Grid ID列表的所有本地坐标点云，合并为一个数组返回
        :param grid_id_list: list[str] 目标Grid ID列表（如["grid_2_3", "grid_4_5"]）
        :return: np.ndarray 合并后的本地坐标点云，形状为 (N, 4)，无点时返回空数组
        """
        
        # 收集所有指定网格的本地点云
        merged_points = []
        grids_num = 0
        for grid_id in grid_id_list:
            # 跳过不存在的Grid ID
            if grid_id not in self.grid_local_points or len(self.grid_local_points[grid_id]) < 5:
                continue
            # 追加当前网格的本地坐标点云
            merged_points.extend(self.grid_local_points[grid_id])
            grids_num += 1
        
        # 转换为numpy数组（无点时返回空数组）
        if len(merged_points) == 0:
            print("No points found in the specified grid IDs.")
            return np.empty((0, 4), dtype=np.float32)
        
        ret = np.array(merged_points, dtype=np.float32)
        print(f"vehicle {self.vid} returns points with shape: {ret.shape} from {grids_num} grids")
        print(f"all points shape: {self.data.shape}")
        return ret
    
    def update_grid_density_dict(self):
        """计算并更新grid_id: 点云密度字典"""
        self.grid_density_dict.clear()
        # total_points = self.local_data.shape[0]
        # print("Total points in last rotation:", total_points)
        total_points = 0
        for grid_id in self.perception_grids:
            # 获取网格内本地点云数量
            local_points = self.grid_local_points.get(grid_id, [])
            point_count = len(local_points)
            total_points += point_count
            
            # 计算网格面积
            grid_box = self.perception_grids[grid_id]
            grid_area = grid_box.area
            
            # 计算密度（点/平方米）
            if grid_area <= 0 or point_count == 0:
                density = 0.0
            else:
                density = point_count / grid_area
                # print("density, point_count, grid_area", density, point_count, grid_area)
            
            self.grid_density_dict[grid_id] = density
        # print("Updated grid density dict, total points:", total_points)

    def get_grid_density(self, grid_id):
        """获取指定网格的点云密度"""
        return self.grid_density_dict.get(grid_id, 0.0)

    def get_low_density_grids(self, threshold=None):
        """
        返回所有密度低于阈值的网格字典 {grid_id: (current_density, threshold)}
        :param threshold: float 密度阈值（点/平方米），默认使用类内阈值
        :return: dict 低密网格字典
        """
        if threshold is None:
            threshold = self.density_threshold
        
        low_density_grids = {}
        for grid_id, current_density in self.grid_density_dict.items():
            if current_density < threshold:
                low_density_grids[grid_id] = (current_density, threshold)
        
        return low_density_grids
    
    def get_area_density(self, x_min, y_min, x_max, y_max):
        """
        计算任意矩形区域的平均点云密度
        :param x_min, y_min, x_max, y_max: 区域边界（全局坐标系）
        :return: 平均密度值（float）
        """
        target_area = box(x_min, y_min, x_max, y_max)
        total_points = 0
        total_area = 0
        
        # 遍历所有与目标区域相交的网格
        for grid_id, grid_box in self.perception_grids.items():
            if grid_box.intersects(target_area):
                # 计算网格与目标区域的交集面积
                intersection = grid_box.intersection(target_area)
                intersect_area = intersection.area
                
                if intersect_area <= 0:
                    continue
                
                # 累加点云数量和面积
                points = self.grid_points.get(grid_id, [])
                # 估算交集中的点数（按面积比例）
                grid_points_count = len(points)
                intersect_points = int(grid_points_count * (intersect_area / grid_box.area))
                
                total_points += intersect_points
                total_area += intersect_area
        
        if total_area <= 0:
            return 0.0
        
        # 计算平均密度
        avg_density = total_points / total_area
        return avg_density
    
    @staticmethod
    def grid_union(grid_set1, grid_set2):
        """计算两个网格集合的并集"""
        return grid_set1 | grid_set2

    @staticmethod
    def grid_intersection(grid_set1, grid_set2):
        """计算两个网格集合的交集"""
        return grid_set1 & grid_set2

    @staticmethod
    def grid_difference(grid_set1, grid_set2):
        """计算两个网格集合的差集"""
        return grid_set1 - grid_set2
    
class SemanticLidarSensor:
    """
    Semantic lidar sensor manager. This class is used when data dumping
    is needed.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.


    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.lidar.ray_cast_semantic')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None

        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: SemanticLidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        # shape:(n, 6)
        data = np.frombuffer(event.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)]))

        # (x, y, z, intensity)
        self.points = np.array([data['x'], data['y'], data['z']]).T
        self.obj_tag = np.array(data['ObjTag'])
        self.obj_idx = np.array(data['ObjIdx'])

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class PerceptionManager:
    """
    Default perception module. Currenly only used to detect vehicles.

    Parameters
    ----------
    vehicle : carla.Vehicle
        carla Vehicle, we need this to spawn sensors.

    config_yaml : dict
        Configuration dictionary for perception.

    cav_world : opencda object
        CAV World object that saves all cav information, shared ML model,
         and sumo2carla id mapping dictionary.

    data_dump : bool
        Whether dumping data, if true, semantic lidar will be spawned.

    carla_world : carla.world
        CARLA world, used for rsu.

    Attributes
    ----------
    lidar : opencda object
        Lidar sensor manager.

    rgb_camera : opencda object
        RGB camera manager.

    o3d_vis : o3d object
        Open3d point cloud visualizer.
    """

    def __init__(self, v2x_manager, localization_manager, behavior_agent, vehicle,
                 config_yaml, cav_world, data_dump=False, carla_world=None, infra_id=None,
                 enable_network=False):
        self.vehicle = vehicle
        self.carla_world = carla_world if carla_world is not None \
            else self.vehicle.get_world()
        self._map = self.carla_world.get_map()
        self.id = infra_id if infra_id is not None else vehicle.id
        self.enable_network = enable_network
        print("enable_network:", self.enable_network)

        self.activate = config_yaml['activate']
        self.camera_visualize = config_yaml['camera']['visualize']
        self.camera_num = config_yaml['camera']['num']
        self.lidar_visualize = config_yaml['lidar']['visualize']
        self.global_position = config_yaml['global_position'] \
            if 'global_position' in config_yaml else None
        self.coperception = config_yaml['coperception'] \
            if 'coperception' in config_yaml else False
        self.enable_communicate = config_yaml['enable_communicate'] \
            if 'enable_communicate' in config_yaml else False
        self.enable_show_gt = config_yaml['enable_show_gt'] \
            if 'enable_show_gt' in config_yaml else False
        self.v2x_manager = v2x_manager
        self.localization_manager = localization_manager
        self.behavior_agent = behavior_agent

        self.perception_frequency = config_yaml.get('perception_frequency', 10)
        self.world_frequency = cav_world.frequency
        self.tick = 0

        self.cav_world = weakref.ref(cav_world)()
        ml_manager = cav_world.ml_manager

        if self.activate and data_dump:
            sys.exit("When you dump data, please deactivate the "
                     "detection function for precise label.")

        if self.activate and not ml_manager:
            sys.exit(
                'If you activate the perception module, '
                'then apply_ml must be set to true in '
                'the argument parser to load the detection DL model.')
        self.ml_manager = ml_manager

        # we only spawn the camera when perception module is activated or
        # camera visualization is needed
        if 'camera' in config_yaml:
            self.rgb_camera = []
            mount_position = config_yaml['camera']['positions']
            # assert len(mount_position) == self.camera_num, \
            #     "The camera number has to be the same as the length of the" \
            #     "relative positions list"

            for i in range(self.camera_num):
                self.rgb_camera.append(
                    CameraSensor(
                        vehicle, self.carla_world, mount_position[i],
                        self.global_position))

        else:
            self.rgb_camera = None

        # we only spawn the LiDAR when perception module is activated or lidar
        # visualization is needed
        if 'lidar' in config_yaml:
            self.lidar = LidarSensor(vehicle,
                                     self.carla_world,
                                     config_yaml['lidar'],
                                     self.global_position)
            if self.lidar_visualize:
                self.o3d_vis = o3d_visualizer_init(self.id)
        else:
            self.lidar = None
            self.o3d_vis = None

        if self.activate:
            if self.coperception:
                print("coperception mode")
            else:
                print("activate mode")
        else:
            print("deactivate mode")
        # if data dump is true, semantic lidar is also spawned
        self.data_dump = data_dump
        if data_dump:
            self.semantic_lidar = SemanticLidarSensor(vehicle,
                                                      self.carla_world,
                                                      config_yaml['lidar'],
                                                      self.global_position)

        # count how many steps have been passed
        self.count = 0
        # ego position
        self.ego_pos = None

        # the dictionary contains all objects
        self.objects = {}
        # traffic light detection related
        self.traffic_thresh = config_yaml['traffic_light_thresh'] \
            if 'traffic_light_thresh' in config_yaml else 50

        # coperception libs
        self.coperception_libs = CoperceptionLibs(
            lidar=self.lidar,
            rgb_camera=self.rgb_camera,
            localization_manager=self.localization_manager,
            behavior_agent=self.behavior_agent,
            carla_world=self.carla_world,
            cav_world=self.cav_world
        )

        self.co_manager = CoperceptionManager(
            vid=self.id,
            v2x_manager=self.v2x_manager,
            coperception_libs=self.coperception_libs,
            enable_network=self.enable_network,
            network_manager=CavWorld.network_manager if self.enable_network else None,
        )

    def set_enable_grids(self, enable):
        if self.lidar is not None:
            self.lidar.set_enable_grids(enable)

    def dist(self, a):
        """
        A fast method to retrieve the obstacle distance the ego
        vehicle from the server directly.

        Parameters
        ----------
        a : carla.actor
            The obstacle vehicle.

        Returns
        -------
        distance : float
            The distance between ego and the target actor.
        """
        return a.get_location().distance(self.ego_pos.location)

    def perception_tick(self):
        self.tick += 1
        if self.tick * self.perception_frequency >= self.world_frequency:
            self.tick = 0
            # print("Perception Tick")
            return True
        return False
    
    def detect(self, ego_pos):
        """
        Detect surrounding objects. Currently only vehicle detection supported.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose.

        Returns
        -------
        objects : list
            A list that contains all detected obstacle vehicles.

        """
        self.ego_pos = ego_pos
        objects = {
            'vehicles': [],
            'traffic_lights': [],
            'is_skipped': False
        }

        if not self.activate:
            objects = self.deactivate_mode(objects)
        else:
            if self.coperception:
                objects = self.coperception_mode(objects)
            else:
                objects = self.activate_mode(objects)
        self.count += 1

        return objects

    def coperception_mode(self, objects):
        """
        Use OpenCOOD to detect objects
        """
        self.cav_world.update_global_ego_id(self.vehicle.id)
        ego_id = self.cav_world.ego_id

        if self.id != ego_id:
            objects = self.deactivate_mode(objects)
            return objects

        if self.lidar.data is None:
            return objects

        data = OrderedDict()

        ego_data = self.co_manager.prepare_data(
            cav_id=self.id,
            camera=self.rgb_camera,
            lidar=self.lidar,
            pos=self.ego_pos,
            localizer=self.localization_manager,
            agent=self.behavior_agent,
            is_ego=True
        )
        ego_data = self.co_manager.calculate_transformation(
            cav_id=self.id,
            cav_data=ego_data,
        )
        data.update(ego_data)

        if self.enable_communicate:
            data.update(self.co_manager.communicate())

        # inference
        reformat_data_dict = self.ml_manager.opencood_dataset.get_item_test(data)
        output_dict = self.ml_manager.opencood_dataset.collate_batch_test(
            [reformat_data_dict])  # should have batch size dim
        batch_data = self.ml_manager.to_device(output_dict)
        predict_box_tensor, predict_score, gt_box_tensor = self.ml_manager.inference(batch_data)
        # self.ml_manager.show_vis(pred_box_tensor, gt_box_tensor, batch_data) show predict results frame by frame
        objects = o3d_predict_bbox_to_object(objects, predict_box_tensor, self.lidar.sensor)
        # retrieve speed from server
        self.speed_retrieve(objects)
        self.transform_retrieve(objects)

        # plot the opencood inference results
        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects
        return objects

    def activate_mode(self, objects):
        """
        Use Yolov5 + Lidar fusion to detect objects.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        # retrieve current cameras and lidar data
        rgb_images = []
        for rgb_camera in self.rgb_camera:
            while rgb_camera.image is None:
                #print("no camera.image")
                continue
            rgb_images.append(
                cv2.cvtColor(
                    np.array(
                        rgb_camera.image),
                    cv2.COLOR_BGR2RGB))

        # yolo detection
        yolo_detection = self.ml_manager.object_detector(rgb_images)
        #print(yolo_detection.xyxy)
        # rgb_images for drawing
        rgb_draw_images = []

        for (i, rgb_camera) in enumerate(self.rgb_camera):
            # lidar projection
            rgb_image, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                rgb_camera.sensor, self.lidar.data, np.array(
                    rgb_camera.image))
            rgb_draw_images.append(rgb_image)

            # camera lidar fusion
            objects = o3d_camera_lidar_fusion(
                objects,
                yolo_detection.xyxy[i],
                self.lidar.data,
                projected_lidar,
                self.lidar.sensor)

            # calculate the speed. current we retrieve from the server
            # directly.
            self.speed_retrieve(objects)
            self.transform_retrieve(objects)

        if self.camera_visualize:
            for (i, rgb_image) in enumerate(rgb_draw_images):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                rgb_image = self.ml_manager.draw_2d_box(
                    yolo_detection, rgb_image, i)
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)
                cv2.imshow(
                    '%s-th camera of actor %d, perception activated' %
                    (str(i), self.id), rgb_image)
            cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects
        return objects

    def deactivate_mode(self, objects):
        """
        Object detection using server information directly.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        world = self.carla_world

        vehicle_list = world.get_actors().filter("*vehicle*")
        if self.coperception:
            thresh = 120
        else:
            thresh = 50 if not self.data_dump else 120

        vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh and
                        v.id != self.id]

        # use semantic lidar to filter out vehicles out of the range
        if self.data_dump:
            vehicle_list = self.filter_vehicle_out_sensor(vehicle_list)

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]

        objects.update({'vehicles': vehicle_list})

        if self.camera_visualize:
            while self.rgb_camera[0].image is None:
                continue

            names = ['front', 'right', 'left', 'back']

            for (i, rgb_camera) in enumerate(self.rgb_camera):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                # we only visualiz the frontal camera
                rgb_image = np.array(rgb_camera.image)
                # draw the ground truth bbx on the camera image
                rgb_image = self.visualize_3d_bbx_front_camera(objects,
                                                               rgb_image,
                                                               i)
                # resize to make it fittable to the screen
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)

                # show image using cv2
                cv2.imshow(
                    '%s camera of actor %d, perception deactivated' %
                    (names[i], self.id), rgb_image)
                cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # render the raw lidar
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)

        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def filter_vehicle_out_sensor(self, vehicle_list):
        """
        By utilizing semantic lidar, we can retrieve the objects that
        are in the lidar detection range from the server.
        This function is important for collect training data for object
        detection as it can filter out the objects out of the senor range.

        Parameters
        ----------
        vehicle_list : list
            The list contains all vehicles information retrieves from the
            server.

        Returns
        -------
        new_vehicle_list : list
            The list that filters out the out of scope vehicles.

        """
        semantic_idx = self.semantic_lidar.obj_idx
        semantic_tag = self.semantic_lidar.obj_tag

        # label 10 is the vehicle
        vehicle_idx = semantic_idx[semantic_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))

        new_vehicle_list = []
        for veh in vehicle_list:
            if veh.id in vehicle_unique_id:
                new_vehicle_list.append(veh)

        return new_vehicle_list

    def visualize_3d_bbx_front_camera(self, objects, rgb_image, camera_index):
        """
        Visualize the 3d bounding box on frontal camera image.

        Parameters
        ----------
        objects : dict
            The object dictionary.

        rgb_image : np.ndarray
            Received rgb image at current timestamp.

        camera_index : int
            Indicate the index of the current camera.

        """
        camera_transform = \
            self.rgb_camera[camera_index].sensor.get_transform()
        camera_location = \
            camera_transform.location
        camera_rotation = \
            camera_transform.rotation

        for v in objects['vehicles']:
            # we only draw the bounding box in the fov of camera
            _, angle = cal_distance_angle(
                v.get_location(), camera_location,
                camera_rotation.yaw)
            if angle < 60:
                bbx_camera = st.get_2d_bb(
                    v,
                    self.rgb_camera[camera_index].sensor,
                    camera_transform)
                cv2.rectangle(rgb_image,
                              (int(bbx_camera[0, 0]), int(bbx_camera[0, 1])),
                              (int(bbx_camera[1, 0]), int(bbx_camera[1, 1])),
                              (255, 0, 0), 2)

        return rgb_image

    def speed_retrieve(self, objects):
        """
        We don't implement any obstacle speed calculation algorithm.
        The speed will be retrieved from the server directly.

        Parameters
        ----------
        objects : dict
            The dictionary contains the objects.
        """
        if 'vehicles' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        # todo: consider the minimum distance to be safer in next version
        for v in vehicle_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_speed = get_speed(obstacle_vehicle)
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                if obstacle_speed > 0:
                    continue
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_velocity(v.get_velocity())

                    # the case where the obstacle vehicle is controled by
                    # sumo
                    if self.cav_world.sumo2carla_ids:
                        sumo_speed = \
                            get_speed_sumo(self.cav_world.sumo2carla_ids,
                                           v.id)
                        if sumo_speed > 0:
                            # todo: consider the yaw angle in the future
                            speed_vector = carla.Vector3D(sumo_speed, 0, 0)
                            obstacle_vehicle.set_velocity(speed_vector)

                    obstacle_vehicle.set_carla_id(v.id)


    def transform_retrieve(self, objects):
        """
        We don't implement any obstacle yaw calculation algorithm.
        The speed will be retrieved from the server directly.

        Parameters
        ----------
        objects : dict
            The dictionary contains the objects.
        """
        if 'vehicles' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        # todo: consider the minimum distance to be safer in next version
        for v in vehicle_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                if obstacle_vehicle.get_transform() is not None:
                    continue
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_transform(v.get_transform())
                    # the case where the obstacle vehicle is controled by
                    obstacle_vehicle.set_carla_id(v.id)

    def retrieve_traffic_lights(self, objects):
        """
        Retrieve the traffic lights nearby from the server  directly.
        Next version may consider add traffic light detection module.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all objects.

        Returns
        -------
        object : dict
            The updated dictionary.
        """
        world = self.carla_world
        tl_list = world.get_actors().filter('traffic.traffic_light*')

        vehicle_location = self.ego_pos.location
        vehicle_waypoint = self._map.get_waypoint(vehicle_location)

        activate_tl, light_trigger_location = \
            self._get_active_light(tl_list, vehicle_location, vehicle_waypoint)

        objects.update({'traffic_lights': []})

        if activate_tl is not None:
            traffic_light = TrafficLight(activate_tl,
                                         light_trigger_location,
                                         activate_tl.get_state())
            objects['traffic_lights'].append(traffic_light)
        return objects

    def _get_active_light(self, tl_list, vehicle_location, vehicle_waypoint):
        for tl in tl_list:
            object_location = \
                TrafficLight.get_trafficlight_trigger_location(tl)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != vehicle_waypoint.road_id:
                continue

            ve_dir = vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x +\
                        ve_dir.y * wp_dir.y + \
                        ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue
            while not object_waypoint.is_intersection:
                next_waypoint = object_waypoint.next(0.5)[0]
                if next_waypoint and not next_waypoint.is_intersection:
                    object_waypoint = next_waypoint
                else:
                    break

            return tl, object_waypoint.transform.location

        return None, None

    def destroy(self):
        """
        Destroy sensors.
        """
        if self.rgb_camera:
            for rgb_camera in self.rgb_camera:
                rgb_camera.sensor.destroy()

        if self.lidar:
            self.lidar.sensor.destroy()

        if self.camera_visualize:
            cv2.destroyAllWindows()

        if self.lidar_visualize:
            self.o3d_vis.destroy_window()

        if self.data_dump:
            self.semantic_lidar.sensor.destroy()
