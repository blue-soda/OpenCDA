from typing import List
import carla
from ..common.vehicle_data_logger import vehicle_data_logger

def collect_vehicle_data(vehicles: List[carla.Vehicle]) -> List[dict]:
    """Collect position and velocity data for all vehicles
    
    Args:
        vehicles: List of carla.Vehicle objects
        
    Returns:
        List of dictionaries containing vehicle data including position, velocity, heading and speed
    """
    vehicle_data = []
    
    for index, vehicle in enumerate(vehicles):
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        position = {
            "x": round(transform.location.x, 2),
            "y": round(transform.location.y, 2),
            "z": round(transform.location.z, 2)
        }

        velocity_data = {
            "x": round(velocity.x, 2),
            "y": round(velocity.y, 2),
            "z": round(velocity.z, 2)
        }

        heading = round(transform.rotation.yaw, 2)
        speed = round((velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5, 2)

        vehicle_data.append({
            "id": index,
            "carla_id": vehicle.id,
            "position": position,
            "velocity": velocity_data,
            "heading": heading,
            "speed": speed
        })

    vehicle_data_logger.log_frame(vehicle_data)

    return vehicle_data 