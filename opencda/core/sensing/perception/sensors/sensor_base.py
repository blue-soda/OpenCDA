"""Base class for all sensors."""

from abc import ABC, abstractmethod
import weakref
import carla


class SensorBase(ABC):
    """Abstract base class for CARLA sensors."""

    def __init__(self, vehicle, world, spawn_point):
        """
        Initialize sensor.

        Args:
            vehicle: CARLA vehicle to attach sensor to (None for RSU)
            world: CARLA world object
            spawn_point: carla.Transform for sensor placement
        """
        self.vehicle = vehicle
        self.world = world if world else vehicle.get_world()
        self.sensor = None
        self.timestamp = None
        self.frame = 0

    @abstractmethod
    def _on_sensor_event(self, weak_self, event):
        """
        Callback for sensor events.
        Must be implemented by subclasses.
        """
        pass

    def destroy(self):
        """Destroy the sensor."""
        if self.sensor:
            self.sensor.destroy()
            self.sensor = None
