import carla
import random
import time
from pathlib import Path
from typing import Tuple
import shutil

from opencda.log.logger_config import logger

from typing import Tuple, Optional, List

def connect_to_carla(host: str, port: int, timeout: float, synchronous: bool = False, 
                    fixed_delta_seconds: float = 0.05) -> Tuple[Optional[carla.Client], Optional[carla.World]]:
    """
    Connect to a running Carla simulator
    
    Args:
        host: Carla server host
        port: Carla server port
        timeout: Connection timeout in seconds
        synchronous: If True, run CARLA in synchronous mode
        fixed_delta_seconds: Fixed time step for synchronous mode
        
    Returns:
        Tuple of (client, world) if connection successful, (None, None) otherwise
    """
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        
        if synchronous:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = fixed_delta_seconds
            world.apply_settings(settings)
            logger.info(f"CARLA running in synchronous mode with {fixed_delta_seconds}s fixed time step")
            
        return client, world
    except Exception as e:
        logger.error(f"Error connecting to Carla: {e}")
        return None, None
        
def spawn_vehicle(world: carla.World, vehicle_type: str = None) -> Optional[carla.Vehicle]:
    """
    Spawn a vehicle in the world
    
    Args:
        world: Carla world
        vehicle_type: Type of vehicle to spawn (e.g., 'model3')
        
    Returns:
        Vehicle actor if spawned successfully, None otherwise
    """
    try:
        blueprint_library = world.get_blueprint_library()
        
        if vehicle_type:
            vehicle_blueprint = blueprint_library.filter(vehicle_type)[0]
        else:
            vehicle_blueprint = random.choice(blueprint_library.filter('vehicle'))
            
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
        return vehicle
    except Exception as e:
        logger.error(f"Error spawning vehicle: {e}")
        return None

def spawn_vehicles(world: carla.World, num_vehicles: int, vehicle_types: Optional[List[str]]) -> List[carla.Vehicle]:
    """
    Spawn a number of vehicles in the world
    
    Args:
        world: Carla world
        num_vehicles: Number of vehicles to spawn
        vehicle_types: List of vehicle types to spawn
        
    Returns:
        List of spawned vehicles
    """
    try:
        blueprint_library = world.get_blueprint_library()
        spawned_vehicles = []

        if vehicle_types:
            vehicle_blueprints = [blueprint_library.filter(vehicle_type)[0] for vehicle_type in vehicle_types]
        else:
            vehicle_blueprints = blueprint_library.filter('*vehicle*')
        
        spawn_points = world.get_map().get_spawn_points()
        
        for i in range(num_vehicles):
            try: 
                vehicle = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
                if vehicle:
                    spawned_vehicles.append(vehicle)
            except Exception as e:
                logger.error(f"Error spawning vehicle: {e}")
                continue
                
        return spawned_vehicles
    except Exception as e:
        logger.error(f"Error spawning vehicles: {e}")
        return []
    
def set_autopilot(vehicles: List[carla.Vehicle], enable: bool = True) -> None:
    """
    Set autopilot mode for all vehicles in the world
    
    Args:
        vehicles: List of vehicles to set autopilot for
        enable: If True, enable autopilot; if False, disable
    """
    try:
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.set_autopilot(enable)
                logger.info(f"Set autopilot for vehicle {vehicle.id} to {enable}")
            else:
                logger.warning(f"Vehicle {vehicle.id} is not alive, skipping")
    except Exception as e:
        logger.error(f"Error setting autopilot: {e}")
        return None

def destroy_actors(vehicles_to_destroy: List[carla.Vehicle] = None) -> None:
    """
    Destroy specified vehicles and reset spectator camera
    
    Args:
        vehicles_to_destroy: List of vehicles to destroy. If None, destroys all vehicles.
    """
    try:
        if vehicles_to_destroy is None:
            return
        
        for vehicle in vehicles_to_destroy:
            try:
                if vehicle.is_alive:
                    vehicle.set_autopilot(False)
            except Exception as e:
                logger.warning(f"Could not disable autopilot for vehicle {vehicle.id}: {e}")
        
        for vehicle in vehicles_to_destroy:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
                    logger.info(f"Destroyed vehicle {vehicle.id}")
            except Exception as e:
                logger.warning(f"Could not destroy vehicle {vehicle.id}: {e}")
                
        logger.info("All specified vehicles destroyed successfully")
    except Exception as e:
        logger.error(f"Error during vehicle destruction: {e}")
        return None

def follow_vehicle(world: carla.World, vehicle: carla.Vehicle, stop_event=None) -> None:
    """
    Set spectator to follow a vehicle
    
    Args:
        world: Carla world object
        vehicle: Carla vehicle object
        stop_event: Threading event to signal when to stop following
    """
    spectator = world.get_spectator()
    
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-4, z=4), carla.Rotation(pitch=-25))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    spectator.set_transform(camera.get_transform())

    try:
        while True:
            try:
                if stop_event and stop_event.is_set():
                    break
                    
                spectator.set_transform(camera.get_transform())
                time.sleep(0.01)
            except KeyboardInterrupt:
                logger.info("Camera follow interrupted by user")
                break

    except Exception as e:
        logger.error(f"Error setting spectator transform: {e}")

def add_camera_to_vehicle(world: carla.World, vehicle: carla.Vehicle) -> Optional[carla.Sensor]:
    """Attach an RGB camera to *vehicle*.

    Args:
        world (carla.World): Carla world object.
        vehicle (carla.Vehicle): Vehicle to attach the camera to.

    Returns:
        carla.Sensor: The camera sensor actor.
    """

    try:
        project_root = Path(__file__).resolve().parents[2]
        temp_dir = project_root / "temp"
        camera_dir = temp_dir / "camera"

        if camera_dir.exists():
            shutil.rmtree(camera_dir) 

        camera_dir.mkdir(parents=True, exist_ok=True)

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logger.info(f"Camera attached to vehicle {vehicle.id}")

        def _save_camera(image: carla.Image) -> None:
            image.save_to_disk(str(camera_dir / f"camera_{image.frame}.png"))

        camera.listen(_save_camera)

        return camera

    except Exception as exc:
        logger.error(f"Error adding camera to vehicle: {exc}")
        return None

def destroy_sensors(sensors: List[carla.Sensor]) -> None:
    """
    Destroy all sensors in the list
    
    Args:
        sensors: List of sensors to destroy
    """
    try:
        for sensor in sensors:
            if sensor.is_alive:
                sensor.destroy()
                logger.info(f"Destroyed sensor {sensor.id}")
            else:
                logger.warning(f"Sensor {sensor.id} is not alive, skipping")
                
        logger.info("All sensors destroyed successfully")
    except Exception as e:
        logger.error(f"Error during sensor destruction: {e}")
        return None