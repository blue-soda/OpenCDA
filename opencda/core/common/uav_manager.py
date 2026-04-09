# -*- coding: utf-8 -*-
"""
UAV Manager for OpenCDA
Integrates AirSim drone control into OpenCDA framework
"""

import carla
import math
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
    # NED z is positive DOWN (opposite of ENU where z is positive UP)
    # To fly at altitude (CARLA z + hover_offset), we need negative NED z
    z_airsim = -(carla_location.z + hover_offset)
    return airsim.Vector3r(
        carla_location.y,  # NED x = ENU y
        -carla_location.x,  # NED y = -ENU x
        z_airsim  # NED z = -(CARLA z + hover_offset)
    )


def convert_airsim_to_carla(airsim_vector):
    """Convert AirSim (NED) to CARLA location (ENU)"""
    return carla.Location(
        x=-airsim_vector.y_val,
        y=airsim_vector.x_val,
        z=-airsim_vector.z_val
    )


class UAVManager:
    """Manages UAV in OpenCDA using AirSim or CARLA native control"""

    def __init__(self, uav_config, base_config, world, cav_world, target_vehicle=None, destination=None):
        self.world = world
        self.cav_world = cav_world
        self.mode = uav_config.get('mode', 'static')
        self.target_vehicle = target_vehicle
        self.destination = destination

        # Target ID for tracking mode (vehicle IDs start from 1)
        self.target_id = uav_config.get('target', 0)

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
                logger.info("AirSim connected successfully!")
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

        # Bounding box display parameters
        self.bounding_box_size = carla.Vector3D(2.0, 2.0, 0.5)  # Width, Length, Height
        self.bounding_box_color = carla.Color(255, 0, 0)  # Red

        # Spawn altitude for tracking/navigation modes (maintain this height)
        self.spawn_z = 0.0

        # Managers (initialized after spawn)
        self.v2x_manager = None
        self.localizer = None
        self.perception_manager = None

        # Current target position for navigation/tracking (updated each step)
        self.current_target_pos = None

        logger.info(f"UAVManager initialized in {self.mode} mode")

    def spawn_drone(self, spawn_location, vid=999):
        """Spawn drone and initialize managers"""
        blueprint_library = self.world.get_blueprint_library()

        # Use a static prop as the drone visualizer (bounding box will make it visible)
        drone_bp = blueprint_library.find('static.prop.shoppingcart')
        drone_bp.set_attribute('role_name', 'uav')

        drone_transform = carla.Transform(spawn_location, carla.Rotation(pitch=0, yaw=0, roll=0))
        self.drone_actor = self.world.spawn_actor(drone_bp, drone_transform)

        # Save spawn altitude for tracking/navigation modes
        self.spawn_z = spawn_location.z

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

        # Update drone position based on mode
        self._update_position()

        # Draw bounding box to visualize drone position
        self._draw_bounding_box()

        # Update perception
        if self.localizer and self.perception_manager:
            self.localizer.localize()
            ego_pos = self.localizer.get_ego_pos()
            self.perception_manager.detect(ego_pos)

    def _draw_bounding_box(self):
        """Draw a bounding box around the drone for visualization"""
        if not self.drone_actor:
            return

        try:
            drone_transform = self.drone_actor.get_transform()
            drone_location = drone_transform.location

            # Create bounding box at drone position
            bounding_box = carla.BoundingBox(
                drone_location,
                self.bounding_box_size
            )

            # Draw the box with rotation
            self.world.debug.draw_box(
                bounding_box,
                drone_transform.rotation,
                self.bounding_box_color,
                life_time=self.update_interval + 0.1,
                persistent_lines=True
            )
        except Exception as e:
            logger.debug(f"Failed to draw bounding box: {e}")

    def _update_position(self):
        """Update drone position based on current mode"""
        if not self.drone_actor:
            return

        if self.airsim_connected:
            try:
                state = self.airsim_client.getMultirotorState().kinematics_estimated
                airsim_pos = state.position
                updated_location = convert_airsim_to_carla(airsim_pos)

                if self.mode == 'tracking' and self.target_vehicle:
                    vehicle_yaw = self.target_vehicle.get_transform().rotation.yaw
                    drone_yaw = vehicle_yaw - 90
                else:
                    drone_yaw = 0

                self.drone_actor.set_transform(carla.Transform(
                    updated_location,
                    carla.Rotation(pitch=0, yaw=drone_yaw, roll=0)
                ))
                logger.debug(f"AirSim state: pos=({airsim_pos.x_val:.2f}, {airsim_pos.y_val:.2f}, {airsim_pos.z_val:.2f}), "
                           f"CARLA=({updated_location.x:.2f}, {updated_location.y:.2f}, {updated_location.z:.2f})")
            except Exception as e:
                logger.warning(f"Failed to update drone position from AirSim: {e}")
        else:
            # CARLA-only mode: directly update position based on mode
            self._update_position_carla()

    def _update_position_carla(self):
        """Update drone position in CARLA-only mode"""
        if not self.drone_actor or self.mode == 'static':
            return

        try:
            current_transform = self.drone_actor.get_transform()
            current_location = current_transform.location

            if self.mode == 'tracking':
                target_vehicle = self._resolve_target_vehicle()
                if target_vehicle is None:
                    logger.debug(f"UAV tracking: target vehicle {self.target_id} not found")
                    return

                vehicle_loc = target_vehicle.get_transform().location
                vehicle_yaw = target_vehicle.get_transform().rotation.yaw

                # Target position: follow vehicle in X,Y but maintain spawn altitude
                target_x = vehicle_loc.x
                target_y = vehicle_loc.y
                target_z = self.spawn_z  # Maintain initial altitude

                # Calculate direction to target (only X,Y, Z is fixed)
                dx = target_x - current_location.x
                dy = target_y - current_location.y
                distance_xy = (dx * dx + dy * dy) ** 0.5

                if distance_xy > 0.5:
                    move_speed = min(self.speed, distance_xy / self.update_interval)
                    new_location = carla.Location(
                        x=current_location.x + (dx / distance_xy) * move_speed * self.update_interval,
                        y=current_location.y + (dy / distance_xy) * move_speed * self.update_interval,
                        z=target_z  # Fixed altitude
                    )
                    new_transform = carla.Transform(
                        new_location,
                        carla.Rotation(pitch=0, yaw=vehicle_yaw, roll=0)
                    )
                else:
                    new_transform = carla.Transform(
                        carla.Location(x=current_location.x, y=current_location.y, z=target_z),
                        carla.Rotation(pitch=0, yaw=vehicle_yaw, roll=0)
                    )

                self.drone_actor.set_transform(new_transform)

            elif self.mode == 'navigation' and self.destination:
                dx = self.destination.x - current_location.x
                dy = self.destination.y - current_location.y
                distance_xy = (dx * dx + dy * dy) ** 0.5
                target_z = self.spawn_z  # Maintain initial altitude

                if distance_xy > 0.5:
                    move_speed = min(self.speed, distance_xy / self.update_interval)
                    new_location = carla.Location(
                        x=current_location.x + (dx / distance_xy) * move_speed * self.update_interval,
                        y=current_location.y + (dy / distance_xy) * move_speed * self.update_interval,
                        z=target_z  # Fixed altitude
                    )
                    yaw = math.degrees(math.atan2(dy, dx))
                    new_transform = carla.Transform(
                        new_location,
                        carla.Rotation(pitch=0, yaw=yaw, roll=0)
                    )
                    self.drone_actor.set_transform(new_transform)
        except Exception as e:
            logger.debug(f"Failed to update drone position: {e}")

    def run_step(self):
        """Execute one step - called each tick (non-blocking)"""
        # Position is updated in update_info() via _update_position()
        # This method handles AirSim-specific commands that need continuous sending
        if not self.airsim_connected or not self.drone_actor:
            return

        try:
            if self.mode == 'tracking' and self.target_vehicle:
                vehicle_loc = self.target_vehicle.get_transform().location
                vehicle_yaw = self.target_vehicle.get_transform().rotation.yaw

                target_pos = convert_carla_to_airsim(vehicle_loc, self.hover_offset)
                airsim_drone_yaw = vehicle_yaw - 90
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=airsim_drone_yaw)

                self.airsim_client.moveToPositionAsync(
                    target_pos.x_val, target_pos.y_val, target_pos.z_val,
                    self.speed, yaw_mode=yaw_mode
                )
                logger.debug(f"Tracking: vehicle=({vehicle_loc.x:.2f}, {vehicle_loc.y:.2f}, {vehicle_loc.z:.2f}) "
                           f"→ AirSim=({target_pos.x_val:.2f}, {target_pos.y_val:.2f}, {target_pos.z_val:.2f})")
            elif self.mode == 'navigation' and self.destination:
                target_pos = convert_carla_to_airsim(self.destination, 0)
                self.airsim_client.moveToPositionAsync(
                    target_pos.x_val, target_pos.y_val, target_pos.z_val, self.speed
                )
                logger.debug(f"Navigation: destination=({self.destination.x}, {self.destination.y}, {self.destination.z}) "
                           f"→ AirSim=({target_pos.x_val:.2f}, {target_pos.y_val:.2f}, {target_pos.z_val:.2f})")
        except Exception as e:
            logger.warning(f"Failed to execute AirSim run_step: {e}")

    def _resolve_target_vehicle(self):
        """Resolve target vehicle from cav_world using target_id (IDs start from 1)"""
        if self.target_vehicle is not None:
            return self.target_vehicle

        # Try to find the target vehicle in cav_world
        # target_id starts from 1, so we subtract 1 for 0-indexed list access
        if self.target_id <= 0:
            return None

        try:
            # Get the target vehicle from single_cav_list
            if hasattr(self.cav_world, 'get_vehicle_manager'):
                # Try to get by VID (vehicle ID)
                vm = self.cav_world.get_vehicle_manager(self.target_id)
                if vm:
                    return vm.vehicle
        except Exception:
            pass

        # Fallback: try to find in all active vehicles
        try:
            if hasattr(self.cav_world, 'vehicle_manager_list'):
                # vehicle IDs start from 1, list is 0-indexed
                index = self.target_id - 1
                if 0 <= index < len(self.cav_world.vehicle_manager_list):
                    return self.cav_world.vehicle_manager_list[index].vehicle
        except Exception:
            pass

        return None

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
