# -*- coding: utf-8 -*-
""" This module is used to check collision possibility """

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

from math import sin, cos
from scipy import spatial

import numpy as np

from core.agents.tools.misc import cal_distance_angle


class CollisionChecker:
    def __init__(self, circle_radius=1.5, circle_offsets=None):
        """
        Construction method
        :param circle_offsets: the offset between collision checking circle and the trajectory point
        :param circle_radius: The radius of the collision checking circle
        """

        self._circle_offsets = [-1.0, 1.0, -3.0, 3.0] if circle_offsets is None else circle_offsets
        self._circle_radius = circle_radius

    def is_in_range(self, ego_vehicle, target_vehicle, candidate_vehicle, carla_map):
        """
        Check whether there is a obstacle vehicle between target_vehicle and ego_vehicle during back_joining
        :param carla_map: carla map
        :param ego_vehicle: The vehicle trying to join platooning
        :param target_vehicle: The vehicle that is suppose to be catched
        :param candidate_vehicle: The possible obstacle vehicle blocking the ego vehicle and target vehicle
        :return:
        """
        ego_loc = ego_vehicle.get_location()
        target_loc = target_vehicle.get_location()
        candidate_loc = candidate_vehicle.get_location()

        # set the checking rectangle
        min_x, max_x = min(ego_loc.x, target_loc.x), max(ego_loc.x, target_loc.x)
        min_y, max_y = min(ego_loc.y, target_loc.y), max(ego_loc.y, target_loc.y)

        # give a small buffer of 2 meters
        if candidate_loc.x <= min_x - 2 or candidate_loc.x >= max_x + 2 or \
                candidate_loc.y <= min_y - 2 or candidate_loc.y >= max_y + 2:
            return False

        candidate_wpt = carla_map.get_waypoint(candidate_loc)
        target_wpt = carla_map.get_waypoint(target_loc)

        # if the candidate vehicle is right behind the target vehicle, then it is blocking
        if target_wpt.lane_id == candidate_wpt.lane_id:
            return True

        # if the candidate is in the same section, then we can confirm they are not in the same lane
        if target_wpt.section_id == candidate_wpt.section_id:
            return False

        # check the angle
        _, angle = cal_distance_angle(target_wpt.transform.location, candidate_wpt.transform.location,
                                      candidate_wpt.transform.rotation.yaw)

        return True if angle <= 3 else False

    def collision_circle_check(self, path_x, path_y, path_yaw, obstacle_vehicle):
        """
        Use circled collision check to see whether potential hazard on the forwarding path
        :param path_yaw: a list of yaw angles
        :param path_x: a list of x coordinates
        :param path_y: a loist of y coordinates
        :param obstacle_vehicle: potention hazard vehicle on the way
        :return:
        """
        collision_free = True

        # every step is 0.1m, so we check every 10 points
        for i in range(0, len(path_x), 10):
            ptx, pty, yaw = path_x[i], path_y[i], path_yaw[i]

            # circle_x = point_x + circle_offset*cos(yaw), circle_y = point_y + circle_offset*sin(yaw)
            circle_locations = np.zeros((len(self._circle_offsets), 2))
            circle_offsets = np.array(self._circle_offsets)
            circle_locations[:, 0] = ptx + circle_offsets * cos(yaw)
            circle_locations[:, 1] = pty + circle_offsets * sin(yaw)

            obstacle_vehicle_loc = obstacle_vehicle.get_location()
            # we need compute the four corner of the bbx
            obstacle_vehicle_bbx_array = np.array([[obstacle_vehicle_loc.x - obstacle_vehicle.bounding_box.extent.x,
                                                    obstacle_vehicle_loc.y - obstacle_vehicle.bounding_box.extent.y],
                                                   [obstacle_vehicle_loc.x - obstacle_vehicle.bounding_box.extent.x,
                                                    obstacle_vehicle_loc.y + obstacle_vehicle.bounding_box.extent.y],
                                                   [obstacle_vehicle_loc.x + obstacle_vehicle.bounding_box.extent.x,
                                                    obstacle_vehicle_loc.y - obstacle_vehicle.bounding_box.extent.y],
                                                   [obstacle_vehicle_loc.x + obstacle_vehicle.bounding_box.extent.x,
                                                    obstacle_vehicle_loc.y + obstacle_vehicle.bounding_box.extent.y]])

            # compute whether the distance between the four corners of the vehicle to the trajectory point
            collision_dists = spatial.distance.cdist(obstacle_vehicle_bbx_array, circle_locations)
            collision_dists = np.subtract(collision_dists, self._circle_radius)
            collision_free = collision_free and not np.any(collision_dists < 0)

            if not collision_free:
                break

        return collision_free
