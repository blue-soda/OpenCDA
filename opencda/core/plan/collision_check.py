# -*- coding: utf-8 -*-
""" This module is used to check collision possibility """

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import math
from math import sin, cos
from scipy import spatial

import carla
import numpy as np

from opencda.core.common.misc import cal_distance_angle, draw_prediction_bbx
from opencda.core.plan.spline import Spline2D


class CollisionChecker:
    """
    The default collision checker module.

    Parameters
    ----------
    time_ahead : float
        how many seconds we look ahead in advance for collision check.
    circle_radius : float
        The radius of the collision checking circle.
    circle_offsets : float
        The offset between collision checking circle and the trajectory point.
    """

    def __init__(self, time_ahead=3,
                 cav_world=None,
                 circle_radius=1.0,
                 circle_offsets=None):
        self._cav_world = cav_world
        self.time_ahead = time_ahead
        self._circle_offsets = [-1.0,
                                0,
                                1.0] \
            if circle_offsets is None else circle_offsets
        self._circle_radius = circle_radius

    def is_in_range(
            self,
            ego_pos,
            target_vehicle,
            candidate_vehicle,
            carla_map):
        """
        Check whether there is a obstacle vehicle between target_vehicle
        and ego_vehicle during back_joining.

        Parameters
        ----------
        carla_map : carla.map
            Carla map  of the current simulation world.

        ego_pos : carla.transform
            Ego vehicle position.

        target_vehicle : carla.vehicle
            The target vehicle that ego vehicle trying to catch up with.

        candidate_vehicle : carla.vehicle
            The possible obstacle vehicle blocking the ego vehicle
            and target vehicle.

        Returns
        -------
        detection result : boolean
        Indicator of whether the target vehicle is in range.
        """
        ego_loc = ego_pos.location
        target_loc = target_vehicle.get_location()
        candidate_loc = candidate_vehicle.get_location()

        # set the checking rectangle
        min_x, max_x = min(
            ego_loc.x, target_loc.x), max(
            ego_loc.x, target_loc.x)
        min_y, max_y = min(
            ego_loc.y, target_loc.y), max(
            ego_loc.y, target_loc.y)

        # give a small buffer of 2 meters
        if candidate_loc.x <= min_x - 2 or candidate_loc.x >= max_x + 2 or \
                candidate_loc.y <= min_y - 2 or candidate_loc.y >= max_y + 2:
            return False

        candidate_wpt = carla_map.get_waypoint(candidate_loc)
        target_wpt = carla_map.get_waypoint(target_loc)

        # if the candidate vehicle is right behind the target vehicle, then it
        # is blocking
        if target_wpt.lane_id == candidate_wpt.lane_id:
            return True

        # in case they are in the same lane but section id is different which
        # changes their id
        if target_wpt.section_id == candidate_wpt.section_id:
            return False

        # check the angle
        distance, angle = cal_distance_angle(
            target_wpt.transform.location, candidate_wpt.transform.location,
            candidate_wpt.transform.rotation.yaw)

        return True if angle <= 3 else False

    def adjacent_lane_collision_check(
            self, ego_loc, target_wpt, overtake, carla_map, world):
        """
        Generate a straight line in the adjacent lane for collision detection
        during overtake/lane change.

        Args:
            -ego_loc (carla.Location): Ego Location.
            -target_wpt (carla.Waypoint): the check point in the adjacent
             at a far distance.
            -overtake (bool): indicate whether this is an overtake or normal
             lane change behavior.
            -world (carla.World): CARLA Simulation world,
             used to draw debug lines.

        Returns:
            -rx (list): the x coordinates of the collision check line in
             the adjacent lane
            -ry (list): the y coordinates of the collision check line in
             the adjacent lane
            -ryaw (list): the yaw angle of the the collision check line in
             the adjacent lane
        """
        # we first need to consider the vehicle on the other lane in front
        if overtake:
            target_wpt_next = target_wpt.next(6)[0]
        else:
            target_wpt_next = target_wpt

        # Next we consider the vehicle behind us
        diff_x = target_wpt_next.transform.location.x - ego_loc.x
        diff_y = target_wpt_next.transform.location.y - ego_loc.y
        diff_s = np.hypot(diff_x, diff_y) + 3

        target_wpt_previous = target_wpt.previous(diff_s)
        while len(target_wpt_previous) == 0:
            diff_s -= 2
            target_wpt_previous = target_wpt.previous(diff_s)

        target_wpt_previous = target_wpt_previous[0]
        target_wpt_middle = target_wpt_previous.next(diff_s / 2)[0]

        x, y = [target_wpt_next.transform.location.x,
                target_wpt_middle.transform.location.x,
                target_wpt_previous.transform.location.x], \
            [target_wpt_next.transform.location.y,
             target_wpt_middle.transform.location.y,
             target_wpt_previous.transform.location.y]
        ds = 0.1

        sp = Spline2D(x, y)
        s = np.arange(sp.s[0], sp.s[-1], ds)

        debug_tmp = []

        # calculate interpolation points
        rx, ry, ryaw = [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            debug_tmp.append(carla.Transform(carla.Location(ix, iy, 0)))

        # draw yellow line for overtaking, white line for lane change
        # draw_trajetory_points(
        #     world, debug_tmp, color=carla.Color(
        #         255, 255, 0) if overtake else carla.Color(
        #         255, 255, 255), size=0.05, lt=0.2)

        return rx, ry, ryaw

    def collision_circle_check(
            self,
            path_x,
            path_y,
            path_yaw,
            obstacle_vehicle,
            speed,
            carla_map,
            adjacent_check=False):
        """
        Use circled collision check to see whether potential hazard on
        the forwarding path.
        Args:
            -adjacent_check (boolean): Indicator of whether do adjacent check.
             Note: always give full path for adjacent lane check.
            -speed (float): ego vehicle speed in m/s.
            -path_yaw (float): a list of yaw angles
            -path_x (list): a list of x coordinates
            -path_y (list): a list of y coordinates
            -obstacle_vehicle (carla.vehicle): potention hazard vehicle
             on the way
        Returns:
            -collision_free (boolean): Flag indicate whether the
             current range is collision free.
        """
        collision_free = True
        # detect x second ahead. in case the speed is very slow,
        # there is some minimum threshold for the check distance
        ### Look ahead how many points in the path list ###
        distance_check = min(max(int(self.time_ahead * speed / 0.1), 90),
                             len(path_x)) \
            if not adjacent_check else len(path_x)

        ### Get the 3D location of the vehicle being checked ###
        obstacle_vehicle_loc = obstacle_vehicle.get_location()
        obstacle_vehicle_yaw = \
            carla_map.get_waypoint(obstacle_vehicle_loc).transform.rotation.yaw

        # every step is 0.1m, so we check every 10 points
        for i in range(0, distance_check, 10):
            ptx, pty, yaw = path_x[i], path_y[i], path_yaw[i]

            circle_locations = np.zeros((len(self._circle_offsets), 2))
            circle_offsets = np.array(self._circle_offsets)
            circle_locations[:, 0] = ptx + circle_offsets * cos(yaw)
            circle_locations[:, 1] = pty + circle_offsets * sin(yaw)

            # calculate bbx coords under world coordinate system
            corrected_extent_x = obstacle_vehicle.bounding_box.extent.x * \
                                 math.cos(math.radians(obstacle_vehicle_yaw))
            corrected_extent_y = obstacle_vehicle.bounding_box.extent.y * \
                                 math.sin(math.radians(obstacle_vehicle_yaw))

            # we need compute the four corner of the bbx
            obstacle_vehicle_bbx_array = \
                np.array([[obstacle_vehicle_loc.x -
                           corrected_extent_x,
                           obstacle_vehicle_loc.y -
                           corrected_extent_y],
                          [obstacle_vehicle_loc.x -
                           corrected_extent_x,
                           obstacle_vehicle_loc.y +
                           corrected_extent_y],
                          [obstacle_vehicle_loc.x,
                           obstacle_vehicle_loc.y],
                          [obstacle_vehicle_loc.x +
                           corrected_extent_x,
                           obstacle_vehicle_loc.y -
                           corrected_extent_y],
                          [obstacle_vehicle_loc.x +
                           corrected_extent_x,
                           obstacle_vehicle_loc.y +
                           corrected_extent_y]])
            # compute whether the distance between the four corners of the
            # vehicle to the trajectory point
            collision_dists = spatial.distance.cdist(
                obstacle_vehicle_bbx_array, circle_locations)

            collision_dists = np.subtract(collision_dists, self._circle_radius)
            collision_free = collision_free and not np.any(collision_dists < 0)

            if not collision_free:
                break

        return collision_free

    """
    TODO: the bounding box looks bigger than it should be - needs more troubleshooting
    """
    def collision_circle_check_enable_prediction(
            self,
            path_x,
            path_y,
            path_yaw,
            obstacle_vehicle,
            vehicle_predictions,
            speed,
            prediction_scan_window,
            adjacent_check=False):
        collision_free = True
        dt = 0.05
        max_time_on_path = min(int((len(path_x) * 0.1 / max(speed, 2.0) / dt)),
                               len(vehicle_predictions))
        time_check = min(max(int(self.time_ahead / dt), int(2 / dt)),
                         max_time_on_path) \
            if not adjacent_check else max_time_on_path
        # decide to align on the time axis, need to convert the points to distances
        # 1. the distance between every two point is 0.1m
        # 2. the distance traveled is speed * time
        # 3. find the corresponding index in the original array
        # check every 1 sec -> 1 / 0.05 = 20. every 20 points for the prediction points
        interval = int(0.10 / dt)

        # align on the time axis
        for time in range(0, time_check, interval):
            # calculating for corresponding ego index
            distance_traveled = time * dt * speed
            ego_idx = int(distance_traveled // 0.1)
            # ego circle (3 points)
            ptx, pty, yaw = path_x[ego_idx], path_y[ego_idx], path_yaw[ego_idx]
            circle_locations = np.zeros((len(self._circle_offsets), 2))
            circle_offsets = np.array(self._circle_offsets)
            circle_locations[:, 0] = ptx + circle_offsets * cos(yaw)
            circle_locations[:, 1] = pty + circle_offsets * sin(yaw)
            draw_prediction_bbx(self._cav_world, ptx, pty, carla.Color(0, 255, 0))

            # if enable the scan window, examine through
            # [vehicle_predictions[i - window], vehicle_predictions[i + window]]
            # calculating surrounding vehicle
            scan_start, scan_end = max(0, time - prediction_scan_window), min(len(vehicle_predictions), time + prediction_scan_window)
            for idx in range(scan_start, scan_end):
                x, y, obstacle_vehicle_yaw = vehicle_predictions[idx]
                # obstacle_vehicle_yaw = math.radians(obstacle_vehicle_yaw)
                draw_prediction_bbx(self._cav_world, x, y)
                dx, dy = obstacle_vehicle.bounding_box.extent.x, obstacle_vehicle.bounding_box.extent.y
                vehicle_extent = np.array([dx, dy])
                corner_points = np.array([
                    [-dx, -dy],
                    [dx, -dy],
                    [-dx, dy],
                    [dx, dy]
                ])
                rotation_matrix = np.array([
                    [np.cos(obstacle_vehicle_yaw), -np.sin(obstacle_vehicle_yaw)],
                    [np.sin(obstacle_vehicle_yaw), np.cos(obstacle_vehicle_yaw)]
                ])
                rotated_points = corner_points @ rotation_matrix.T
                obstacle_vehicle_bbx_array = rotated_points + np.array([x, y])
                for pt in obstacle_vehicle_bbx_array.tolist():
                    draw_prediction_bbx(self._cav_world, pt[0], pt[1], carla.Color(0, 0, 255), z=3.0)

                collision_dists = spatial.distance.cdist(
                    obstacle_vehicle_bbx_array, circle_locations)

                collision_dists = np.subtract(collision_dists, self._circle_radius)
                collision_free = collision_free and not np.any(collision_dists < 0)

                if not collision_free:
                    break
        return collision_free
