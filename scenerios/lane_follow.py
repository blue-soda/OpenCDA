# -*- coding: utf-8 -*-

"""Simple platooning following test
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

import carla

from core.platooning.platooning_world import PlatooningWorld
from core.platooning.platooning_manager import PlatooningManager
from core.vehicle.vehicle_manager import VehicleManager


def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Retrieve the world that is currently running
        world = client.get_world()
        origin_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # traffic manager for background traffic
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        traffic_manager.global_percentage_speed_difference(-40)

        # setup spawn points
        transform_1 = carla.Transform(carla.Location(x=51.71, y=139.53, z=1.0),
                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        transform_2 = carla.Transform(carla.Location(x=41.71, y=139.53, z=1.0),
                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        transform_3 = carla.Transform(carla.Location(x=31.71, y=139.53, z=1.0),
                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        transform_4 = carla.Transform(carla.Location(x=21.71, y=139.51, z=0.3),
                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

        # background testing traffic car
        transform_5 = carla.Transform(carla.Location(x=121.7194, y=139.51, z=0.3),
                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        transform_6 = carla.Transform(carla.Location(x=121.7194, y=142.51, z=0.3),
                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        transform_7 = carla.Transform(carla.Location(x=126.7194, y=146.51, z=0.3),
                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

        transform_destination = carla.Transform(carla.Location(x=630, y=142, z=0.3),
                                                carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

        # create the leading vehicle
        ego_vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz2017')
        # black color
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        vehicle_1 = world.spawn_actor(ego_vehicle_bp, transform_1)

        ego_vehicle_bp.set_attribute('color', '255, 255, 255')
        vehicle_2 = world.spawn_actor(ego_vehicle_bp, transform_2)

        ego_vehicle_bp.set_attribute('color', '255, 255, 255')
        vehicle_3 = world.spawn_actor(ego_vehicle_bp, transform_3)

        ego_vehicle_bp.set_attribute('color', '255, 255, 255')
        vehicle_4 = world.spawn_actor(ego_vehicle_bp, transform_4)

        world.tick()
        # vehicle 5-7 are background traffic
        ego_vehicle_bp.set_attribute('color', '0, 255, 0')
        vehicle_5 = world.spawn_actor(ego_vehicle_bp, transform_5)
        vehicle_6 = world.spawn_actor(ego_vehicle_bp, transform_6)
        vehicle_7 = world.spawn_actor(ego_vehicle_bp, transform_7)

        vehicle_5.apply_control(carla.VehicleControl(throttle=0.55))
        vehicle_5.set_autopilot(False)
        vehicle_6.apply_control(carla.VehicleControl(throttle=0.55))
        vehicle_6.set_autopilot(False)
        vehicle_7.apply_control(carla.VehicleControl(throttle=0.55))
        vehicle_7.set_autopilot(False)

        # create platooning world
        platooning_world = PlatooningWorld()

        # setup managers
        vehicle_manager_1 = VehicleManager(vehicle_1, platooning_world, sample_resolution=6.0, buffer_size=8,
                                           debug_trajectory=True, debug=False, ignore_traffic_light=True)
        vehicle_manager_2 = VehicleManager(vehicle_2, platooning_world, buffer_size=8,
                                           debug_trajectory=True, debug=False)
        vehicle_manager_3 = VehicleManager(vehicle_3, platooning_world,
                                           debug_trajectory=False, debug=False)
        vehicle_manager_4 = VehicleManager(vehicle_4, platooning_world,
                                           debug_trajectory=False, debug=False)

        platooning_manager = PlatooningManager(platooning_world)

        # set leader
        platooning_manager.set_lead(vehicle_manager_1)
        # add member
        platooning_manager.add_member(vehicle_manager_2)
        platooning_manager.add_member(vehicle_manager_3)
        platooning_manager.add_member(vehicle_manager_4)

        # set destination TODO: the spawn point may have conflict
        destination = transform_destination.location

        platooning_manager.set_destination(destination)

        while True:
            # if not world.wait_for_tick(10.0):
            #     continue
            world.tick()
            spectator = world.get_spectator()
            transform = vehicle_2.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

            platooning_manager.update_information(platooning_world)
            platooning_manager.run_step()

    finally:
        platooning_manager.destroy()
        origin_settings.synchronous_mode = False
        world.apply_settings(origin_settings)

        vehicle_5.destroy()
        vehicle_6.destroy()
        vehicle_7.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
