"""Camera sensor module."""

import weakref
import numpy as np
import carla
from .sensor_base import SensorBase


class CameraSensor(SensorBase):
    """
    Camera manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.
    world : carla.World
        The carla world object, this is for rsu.
    relative_position : tuple
        (x, y, z, yaw) relative to vehicle or global position.
    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.
    """

    def __init__(self, vehicle, world, relative_position, global_position):
        spawn_point = self.spawn_point_estimation(relative_position, global_position)
        super().__init__(vehicle, world, spawn_point)

        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', '100')

        if vehicle is not None:
            self.sensor = self.world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = self.world.spawn_actor(blueprint, spawn_point)

        self.image = None
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CameraSensor._on_rgb_image_event(weak_self, event))

        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):
        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position

        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35

        carla_location = carla.Location(
            x=carla_location.x + x,
            y=carla_location.y + y,
            z=carla_location.z + z)

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        return carla.Transform(carla_location, carla_rotation)

    @staticmethod
    def _on_rgb_image_event(weak_self, event):
        """CAMERA method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        image = image[:, :, :3]

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp

    def _on_sensor_event(self, weak_self, event):
        """Implementation of abstract method."""
        return self._on_rgb_image_event(weak_self, event)
