# -*- coding: utf-8 -*-
"""
Scenario testing: merging vehicle joining a platoon in the customized 2-lane freeway sorely with carla
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

import argparse
import os
import sys

import carla

from core.application.platooning.platooning_manager import PlatooningManager
from core.application.platooning.platooning_world import PlatooningWorld
from core.common.vehicle_manager import VehicleManager
from scenerio_testing.utils.yaml_utils import load_yaml
from scenerio_testing.utils.load_customized_world import load_customized_world


def arg_parse():
    parser = argparse.ArgumentParser(description="Platooning Joining Settings")
    parser.add_argument("--config_yaml", required=True, type=str, help='corresponding yaml file of the testing')

    opt = parser.parse_args()
    return opt


def main():
    try:
        opt = arg_parse()
        scenario_params = load_yaml(opt.config_yaml)
        # set simulation
        simulation_config = scenario_params['world']

        client = carla.Client('localhost', simulation_config['client_port'])
        client.set_timeout(2.0)

        # Retrieve the customized map
        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(current_path,
                                 '../assets/2lane_freeway_simplified/map_v7.4_smooth_curve.xodr')
        world = load_customized_world(xodr_path, client)
        if not world:
            sys.exit()

        # TODO: Temporary testing, we need to add a helper function for this
        all_deafault_spawn = world.get_map().get_spawn_points()
        transform_point = all_deafault_spawn[11]
        transform_point.location.x = transform_point.location.x + \
                                     0.49 * (all_deafault_spawn[2].location.x - all_deafault_spawn[11].location.x)
        transform_point.location.y = transform_point.location.y + \
                                     0.49 * (all_deafault_spawn[2].location.y - all_deafault_spawn[11].location.y)

        # used to recover the world back to async mode when the testing is done
        origin_settings = world.get_settings()
        new_settings = world.get_settings()

        if simulation_config['sync_mode']:
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = simulation_config['fixed_delta_seconds']
        world.apply_settings(new_settings)

        cav_vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz2017')

        platoon_list = []
        single_cav_list = []
        platooning_world = PlatooningWorld()

        # create platoons
        for i, platoon in enumerate(scenario_params['scenario']['platoon_list']):
            platoon_manager = PlatooningManager(platoon, platooning_world)
            for j, cav in enumerate(platoon['members']):
                spawn_transform = carla.Transform(carla.Location(x=cav['spawn_position'][0],
                                                                 y=cav['spawn_position'][1],
                                                                 z=cav['spawn_position'][2]),
                                                  carla.Rotation(pitch=0, yaw=0, roll=0))
                cav_vehicle_bp.set_attribute('color', '0, 0, 0')
                vehicle = world.spawn_actor(cav_vehicle_bp, spawn_transform)

                # create vehicle manager for each cav
                vehicle_manager = VehicleManager(vehicle, cav, ['platooning'], platooning_world)
                # add the vehicle manager to platoon
                if j == 0:
                    platoon_manager.set_lead(vehicle_manager)
                else:
                    platoon_manager.add_member(vehicle_manager, leader=False)

            world.tick()
            destination = carla.Location(x=platoon['destination'][0],
                                         y=platoon['destination'][1],
                                         z=platoon['destination'][2])

            platoon_manager.set_destination(destination)
            platoon_manager.update_member_order()
            platoon_list.append(platoon_manager)

        for i, cav in enumerate(scenario_params['scenario']['single_cav_list']):
            spawn_transform = carla.Transform(carla.Location(x=cav['spawn_position'][0],
                                                             y=cav['spawn_position'][1],
                                                             z=cav['spawn_position'][2]),
                                              carla.Rotation(pitch=0, yaw=0, roll=0))
            cav_vehicle_bp.set_attribute('color', '0, 0, 0')
            vehicle = world.spawn_actor(cav_vehicle_bp, transform_point)

            # create vehicle manager for each cav
            vehicle_manager = VehicleManager(vehicle, cav, ['platooning'], platooning_world)
            vehicle_manager.v2x_manager.set_platoon(None)

            world.tick()

            destination = carla.Location(x=cav['destination'][0],
                                         y=cav['destination'][1],
                                         z=cav['destination'][2])
            vehicle_manager.update_info(platooning_world)
            vehicle_manager.set_destination(vehicle_manager.vehicle.get_location(),
                                            destination,
                                            clean=True)

            single_cav_list.append(vehicle_manager)

        spectator = world.get_spectator()
        # run steps
        while True:
            # TODO: Consider aysnc mode later
            world.tick()
            transform = platoon_list[0].vehicle_manager_list[-1].vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                                                    carla.Rotation(pitch=-90)))
            for platoon in platoon_list:
                platoon.update_information(platooning_world)
                platoon.run_step()

            for i, single_cav in enumerate(single_cav_list):
                # this function should be added in wrapper
                if single_cav.v2x_manager.in_platoon():
                    single_cav_list.pop(i)
                else:
                    single_cav.update_info(platooning_world)
                    control = single_cav.run_step()
                    single_cav.vehicle.apply_control(control)

    finally:
        world.apply_settings(origin_settings)
        for platoon in platoon_list:
            platoon.destroy()
        for cav in single_cav_list:
            cav.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
