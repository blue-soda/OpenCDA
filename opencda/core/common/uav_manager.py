# -*- coding: utf-8 -*-
"""
UAV Manager for OpenCDA
Integrates AirSim drone control into OpenCDA framework
"""

import carla
import weakref
from opencda.log.logger_config import logger
from opencda.core.sensing.perception.perception_manager import PerceptionManager
from opencda.core.sensing.localization.localization_manager import LocalizationManager
from opencda.core.common.v2x_manager import V2XManager

# Try to import airsim, but make it optional
try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AirSim not available: {e}. UAV will run in CARLA-only mode.")
    airsim = None
    AIRSIM_AVAILABLE = False


def convert_carla_to_airsim(carla_location, hover_offset):
    """Convert CARLA location (ENU) to AirSim (NED)

    Args:
        carla_location: CARLA Location in ENU coordinates
        hover_offset: Height above the vehicle in meters (positive = higher altitude)
    """
    # ENU to NED: x_enu=y_ned, y_enu=x_ned
    # NED z is up-positive (opposite to ENU where z is up-positive)
    # hover_offset adds height, so we subtract it in NED z
    ned_z = -(carla_location.z - hover_offset)
    return airsim.Vector3r(
        carla_location.y,  # NED x = ENU y
        -carla_location.x,  # NED y = -ENU x
        ned_z  # NED z = -(ENU z - hover_offset) = hover_offset - ENU_z
    )


def convert_airsim_to_carla(airsim_vector):
    """Convert AirSim (NED) to CARLA location (ENU)"""
    return carla.Location(
        x=-airsim_vector.y_val,
        y=airsim_vector.x_val,
        z=-airsim_vector.z_val
    )


class UAVManager:
    """Manages UAV in OpenCDA using AirSim"""

    def __init__(self, uav_config, base_config, world, cav_world, target_vehicle=None, destination=None):
        self.world = world
        self.cav_world = cav_world
        self.mode = uav_config.get('mode', 'tracking')
        self.target_vehicle = target_vehicle
        self.destination = destination

        # Parameters from base_config
        self.takeoff_height = base_config.get('takeoff_height', 60)
        self.hover_offset = base_config.get('hover_offset', 60)
        self.speed = base_config.get('speed', 6)
        self.update_interval = base_config.get('update_interval', 0.033)

        # V2X and sensing config
        self.v2x_config = base_config.get('v2x', {})
        self.sensing_config = base_config.get('sensing', {})
        self.carla_map = world.get_map()

        # AirSim client
        self.airsim_connected = False
        if AIRSIM_AVAILABLE:
            try:
                self.airsim_client = airsim.MultirotorClient()
                self.airsim_client.confirmConnection()
                self.airsim_client.enableApiControl(True)
                self.airsim_client.armDisarm(True)
                self.airsim_connected = True
            except Exception as e:
                logger.warning(f"AirSim connection failed: {e}. Running in CARLA-only mode.")
                self.airsim_client = None
        else:
            self.airsim_client = None
            logger.warning("AirSim not installed. Running in CARLA-only mode.")

        # CARLA actors
        self.drone_actor = None
        self.visual_marker = None
        self.takeoff_complete = False

        # Managers (initialized after spawn)
        self.v2x_manager = None
        self.localizer = None
        self.perception_manager = None

        logger.info(f"UAVManager initialized in {self.mode} mode")

    def spawn_drone(self, spawn_location, vid=999):
        """Spawn drone and initialize managers"""
        blueprint_library = self.world.get_blueprint_library()

        # Use a small vehicle instead of static prop for sensor attachment
        # drone_bp = blueprint_library.filter('vehicle.*')[0]
        drone_bp = blueprint_library.find('static.prop.shoppingcart')
        drone_bp.set_attribute('role_name', 'uav')

        drone_transform = carla.Transform(spawn_location, carla.Rotation(pitch=0, yaw=0, roll=0))
        self.drone_actor = self.world.spawn_actor(drone_bp, drone_transform)

        # Set physics to simulate hovering
        self.drone_actor.set_simulate_physics(False)

        # Spawn visual marker
        # marker_bp = blueprint_library.find('static.prop.streetbarrier')
        # marker_transform = carla.Transform(carla.Location(x=0, y=0, z=-2))
        # self.visual_marker = self.world.spawn_actor(marker_bp, marker_transform, attach_to=self.drone_actor)

        # Initialize V2X manager
        self.v2x_manager = V2XManager(self.cav_world, self.v2x_config, vid)

        # Initialize localization manager
        localization_config = self.sensing_config.get('localization', {'activate': False})
        self.localizer = LocalizationManager(self.drone_actor, localization_config, self.carla_map)

        # Initialize perception manager
        perception_config = self.sensing_config.get('perception', {})

        # Set default downward-facing LiDAR config for UAV if not provided
        if 'lidar' in perception_config:
            lidar_config = perception_config['lidar']
            # Check if global_position is set inside lidar config (UAV-specific format)
            if 'global_position' in lidar_config:
                # Move to top level for PerceptionManager
                perception_config['global_position'] = lidar_config['global_position']
            elif 'global_position' not in perception_config:
                # Default: LiDAR mounted at center, looking straight down (pitch=-90)
                # [x, y, z, roll, pitch, yaw]
                perception_config['global_position'] = [0, 0, 0, 0, -90, 0]

        self.perception_manager = PerceptionManager(
            v2x_manager=self.v2x_manager,
            localization_manager=self.localizer,
            behavior_agent=None,
            vehicle=self.drone_actor,
            config_yaml=perception_config,
            cav_world=self.cav_world,
            data_dump=False,
            carla_world=self.world
        )

        # Set AirSim pose
        if self.airsim_connected:
            airsim_pos = convert_carla_to_airsim(spawn_location, 0)
            drone_pose = airsim.Pose(airsim_pos, airsim.to_quaternion(0, 0, 0))
            self.airsim_client.simSetVehiclePose(drone_pose, True)

        logger.info(f"Drone spawned at {spawn_location}")

    def takeoff(self):
        """Initiate takeoff (non-blocking)"""
        if not self.airsim_connected:
            self.takeoff_complete = True
            return

        logger.info("UAV taking off...")
        try:
            self.airsim_client.takeoffAsync()
        except Exception as e:
            logger.warning(f"Takeoff failed: {e}")
            self.takeoff_complete = True

    def update_info(self):
        """Update drone state and perception"""
        if not self.drone_actor:
            return

        if self.airsim_connected:
            try:
                state = self.airsim_client.getMultirotorState().kinematics_estimated
                updated_location = convert_airsim_to_carla(state.position)

                if self.mode == 'tracking' and self.target_vehicle:
                    vehicle_yaw = self.target_vehicle.get_transform().rotation.yaw
                    # CARLA yaw (ENU, counterclockwise, 0=East) to NED (clockwise, 0=North)
                    # NED_yaw = -(CARLA_yaw + 90) approximately
                    drone_yaw = -(vehicle_yaw + 90)
                else:
                    drone_yaw = 0

                self.drone_actor.set_transform(carla.Transform(
                    updated_location,
                    carla.Rotation(pitch=0, yaw=drone_yaw, roll=0)
                ))
            except Exception as e:
                logger.debug(f"Failed to update drone info: {e}")

        # Update perception
        if self.localizer and self.perception_manager:
            self.localizer.localize()
            ego_pos = self.localizer.get_ego_pos()
            self.perception_manager.detect(ego_pos)

    def run_step(self):
        """Execute one step - called each tick (non-blocking)"""
        if not self.airsim_connected or not self.drone_actor:
            return

        try:
            if self.mode == 'tracking' and self.target_vehicle:
                vehicle_loc = self.target_vehicle.get_transform().location
                vehicle_yaw = self.target_vehicle.get_transform().rotation.yaw

                target_pos = convert_carla_to_airsim(vehicle_loc, self.hover_offset)
                # CARLA yaw (ENU, CCW) to NED yaw (CW): yaw_NED = -(yaw_CARLA + 90)
                ned_yaw = -(vehicle_yaw + 90)
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=ned_yaw)

                self.airsim_client.moveToPositionAsync(
                    target_pos.x_val, target_pos.y_val, target_pos.z_val,
                    self.speed, yaw_mode=yaw_mode
                )
            elif self.mode == 'navigation' and self.destination:
                target_pos = convert_carla_to_airsim(self.destination, 0)
                self.airsim_client.moveToPositionAsync(
                    target_pos.x_val, target_pos.y_val, target_pos.z_val, self.speed
                )
        except Exception as e:
            logger.debug(f"Failed to execute run_step: {e}")

    def destroy(self):
        """Cleanup"""
        if self.airsim_connected:
            try:
                logger.info("Landing UAV...")
                self.airsim_client.landAsync()
                self.airsim_client.armDisarm(False)
                self.airsim_client.enableApiControl(False)
            except Exception as e:
                logger.warning(f"Error during UAV cleanup: {e}")

        if self.perception_manager:
            self.perception_manager.destroy()
        if self.localizer:
            self.localizer.destroy()
        if self.visual_marker:
            self.visual_marker.destroy()
        if self.drone_actor:
            self.drone_actor.destroy()

        logger.info("UAV destroyed")
