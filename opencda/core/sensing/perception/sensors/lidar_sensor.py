"""LiDAR sensor module."""

import weakref
import numpy as np
import carla
import open3d as o3d
from .sensor_base import SensorBase


class LidarSensor(SensorBase):
    """
    Lidar sensor manager without grid system.

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
    o3d_pointcloud : o3d object
        Received point cloud, saved in o3d.Pointcloud format.
    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.
    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2]))

        super().__init__(vehicle, world, spawn_point)

        blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute('points_per_second', str(config_yaml['points_per_second']))
        blueprint.set_attribute('rotation_frequency', str(config_yaml['rotation_frequency']))
        blueprint.set_attribute('dropoff_general_rate', str(config_yaml['dropoff_general_rate']))
        blueprint.set_attribute('dropoff_intensity_limit', str(config_yaml['dropoff_intensity_limit']))
        blueprint.set_attribute('dropoff_zero_intensity', str(config_yaml['dropoff_zero_intensity']))
        blueprint.set_attribute('noise_stddev', str(config_yaml['noise_stddev']))

        if vehicle is not None:
            self.sensor = self.world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
            self.vid = str(vehicle.id)
        else:
            self.sensor = self.world.spawn_actor(blueprint, spawn_point)
            self.vid = "None"

        self.rotate_frequency = int(config_yaml['rotation_frequency'])
        self.world_slot_seconds = self.world.get_settings().fixed_delta_seconds
        self.world_frequency = int(1.0 / self.world_slot_seconds)
        self.tick = 0
        self.points_buffer = []
        self.data = None
        self.last_rotation_time = None
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LidarSensor._on_data_event(weak_self, event))

    def rotate_tick(self):
        self.tick += 1
        if self.tick * self.rotate_frequency >= self.world_frequency:
            self.tick = 0
            return True
        return False

    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar callback method"""
        self = weak_self()
        if not self:
            return

        frame_data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        frame_data = frame_data.reshape(-1, 4)

        self.points_buffer.append(frame_data)
        self.frame = event.frame
        self.timestamp = event.timestamp

        if self.rotate_tick():
            self.data = np.vstack(self.points_buffer)
            self.last_rotation_time = event.timestamp
            self.points_buffer = []

    def _on_sensor_event(self, weak_self, event):
        """Implementation of abstract method."""
        return self._on_data_event(weak_self, event)
