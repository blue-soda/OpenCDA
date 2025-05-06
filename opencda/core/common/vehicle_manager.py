# -*- coding: utf-8 -*-
"""
Basic class of CAV
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import uuid

from opencda.core.actuation.control_manager \
    import ControlManager
from opencda.core.application.platooning.platoon_behavior_agent\
    import PlatooningBehaviorAgent
from opencda.core.common.v2x_manager \
    import V2XManager
from opencda.customize.core.clustering.clustering_v2x_manager \
    import ClusteringV2XManager
from opencda.core.sensing.localization.localization_manager \
    import LocalizationManager
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager
from opencda.customize.core.clustering.clustering_perception_manager \
    import ClusteringPerceptionManager
from opencda.core.safety.safety_manager import SafetyManager
from opencda.core.plan.behavior_agent \
    import BehaviorAgent
from opencda.core.map.map_manager import MapManager
from opencda.core.common.data_dumper import DataDumper

from opencda.log.logger_config import logger

class VehicleManager(object):
    """
    A class manager to embed different modules with vehicle together.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle. We need this class to spawn our gnss and imu sensor.

    config_yaml : dict
        The configuration dictionary of this CAV.

    application : list
        The application category, currently support:['single','platoon'].

    carla_map : carla.Map
        The CARLA simulation map.

    cav_world : opencda object
        CAV World. This is used for V2X communication simulation.

    current_time : str
        Timestamp of the simulation beginning, used for data dumping.

    data_dumping : bool
        Indicates whether to dump sensor data during simulation.

    Attributes
    ----------
    v2x_manager : opencda object
        The current V2X manager.

    localizer : opencda object
        The current localization manager.

    perception_manager : opencda object
        The current V2X perception manager.

    agent : opencda object
        The current carla agent that handles the basic behavior
         planning of ego vehicle.

    controller : opencda object
        The current control manager.

    data_dumper : opencda object
        Used for dumping sensor data.
    """

    def __init__(
            self,
            vehicle,
            config_yaml,
            application,  #['single', 'coperception', 'traffic', 'cluster'， 'network]
            carla_map,
            cav_world,
            current_time='',
            data_dumping=False, 
            ):

        # an unique uuid for this vehicle
        self.vid = str(uuid.uuid1())
        self.vehicle = vehicle
        self.carla_map = carla_map
        self.cav_world = cav_world
        self.application = application
        self.is_ok = True
        # retrieve the configure for different modules
        sensing_config = config_yaml['sensing']
        map_config = config_yaml['map_manager']
        behavior_config = config_yaml['behavior']
        control_config = config_yaml['controller']
        v2x_config = config_yaml['v2x']

        self.isTrafficVehicle = 'traffic' in application
        self.enableNetwork = 'network' in application
        self.enableCluster = 'cluster' in application
        self.enableCoperception = 'coperception' in application
        self.enablePlatooning = 'platooning' in application

        # v2x module
        if self.enableCluster:
            self.v2x_manager = ClusteringV2XManager(cav_world, v2x_config, self.vid, self.vehicle.id)
        else:
            if v2x_config['network']['enabled'] and v2x_config['network']['scheduler'] == 'clusterbased':
                v2x_config['network']['scheduler'] = 'roundrobin'
                print('Warning: do not use cluster_based scheduler when clustering is not active. scheduler param has been changed to roundrobin automately.')
                logger.warning('do not use cluster_based scheduler when clustering is not active. scheduler param has been changed to roundrobin automately.')
            self.v2x_manager = V2XManager(cav_world, v2x_config, self.vid)
    
        # localization module
        self.localizer = LocalizationManager(
            vehicle, sensing_config['localization'], carla_map)
        
        if self.isTrafficVehicle:
            self.map_manager = None
            self.safety_manager = None
            self.agent = None
            self.controller = None
        else:

            # map manager
            self.map_manager = MapManager(vehicle,
                                        carla_map,
                                        map_config)
            # safety manager
            self.safety_manager = SafetyManager(vehicle=vehicle,
                                                params=config_yaml['safety_manager'])

            cav_world.update_global_ego_id(self.vehicle.id)
            # behavior agent
            if self.enablePlatooning:
                platoon_config = config_yaml['platoon']
                self.agent = PlatooningBehaviorAgent(
                    vehicle,
                    self,
                    self.v2x_manager,
                    behavior_config,
                    platoon_config,
                    carla_map)
            else:
                self.agent = BehaviorAgent(vehicle, carla_map, behavior_config, self.cav_world.ego_id)

            # Control module
            self.controller = ControlManager(control_config)

        # perception module
        # move it down here to pass in the behavior manager & localization manager
        if self.enableCluster:
            if not self.enableCoperception:
                import sys
                sys.exit(
                        'If you activate the cluster module, '
                        'then apply_cp must be set to true in'
                        'the argument parser to load the opencood manager')
            self.perception_manager = ClusteringPerceptionManager(
                v2x_manager=self.v2x_manager,
                localization_manager=self.localizer,
                behavior_agent=self.agent,
                vehicle=vehicle,
                config_yaml=sensing_config['perception'],
                cav_world=cav_world,
                data_dump=data_dumping,
                enable_network = self.enableNetwork)
        else:
            self.perception_manager = PerceptionManager(
                v2x_manager=self.v2x_manager,
                localization_manager=self.localizer,
                behavior_agent=self.agent,
                vehicle=vehicle,
                config_yaml=sensing_config['perception'],
                cav_world=cav_world,
                data_dump=data_dumping,
                enable_network = self.enableNetwork)
            
        if data_dumping:
            self.data_dumper = DataDumper(self.perception_manager,
                                          vehicle.id,
                                          save_time=current_time)
        else:
            self.data_dumper = None

        self.pre_obejcts_num = 0
        
        cav_world.update_vehicle_manager(self, self.isTrafficVehicle)


    def set_destination(
            self,
            start_location,
            end_location,
            clean=False,
            end_reset=True):
        """
        Set global route.

        Parameters
        ----------
        start_location : carla.location
            The CAV start location.

        end_location : carla.location
            The CAV destination.

        clean : bool
             Indicator of whether clean waypoint queue.

        end_reset : bool
            Indicator of whether reset the end location.

        Returns
        -------
        """

        self.agent.set_destination(
            start_location, end_location, clean, end_reset)

    def update_info(self):
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()
        ego_dir = self.localizer.get_ego_dir()
        ego_lidar = self.perception_manager.lidar
        ego_image = self.perception_manager.rgb_camera

        if not self.is_ok:
            return

        # update ego position and speed to v2x manager,
        # and then v2x manager will search the nearby cavs
        self.v2x_manager.update_info(ego_pos, ego_spd, ego_lidar, ego_image, ego_dir)
        # object detection
        objects = self.perception_manager.detect(ego_pos)
        
        if 'traffic' in self.application:
            return

        # update the ego pose for map manager
        self.map_manager.update_information(ego_pos)

        # this is required by safety manager
        safety_input = {'ego_pos': ego_pos,
                        'ego_speed': ego_spd,
                        'objects': objects,
                        'carla_map': self.carla_map,
                        'world': self.vehicle.get_world(),
                        'static_bev': self.map_manager.static_bev}
        self.safety_manager.update_info(safety_input)

        self.agent.update_information(ego_pos, ego_spd, objects)
        # pass position and speed info to controller
        self.controller.update_info(ego_pos, ego_spd)

    def run_step(self, target_speed=None):
        """
        Execute one step of navigation.
        """
        if 'traffic' in self.application:
            return
        # visualize the bev map if needed
        self.map_manager.run_step()
        target_speed, target_pos = self.agent.run_step(target_speed, self.cav_world.ego_id)
        control = self.controller.run_step(target_speed, target_pos)

        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager,
                                      self.localizer,
                                      self.agent)

        return control

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        if self.perception_manager:
            self.perception_manager.destroy()
        if self.localizer:
            self.localizer.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.map_manager:
            self.map_manager.destroy()
        if self.safety_manager:
            self.safety_manager.destroy()
