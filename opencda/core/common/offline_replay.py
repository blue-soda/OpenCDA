# -*- coding: utf-8 -*-
"""
Lightweight offline state adapters for SGCP replay.

These classes rebuild the small subset of CavWorld/VehicleManager/V2XManager
state required by the clustering code from an OPV2V-style dumped frame.
"""

from collections import OrderedDict, defaultdict, deque
import copy
import math

import numpy as np

from opencda.opencda_carla import Location, Rotation, Transform
from opencda.core.sensing.perception import sensor_transformation as st


class OfflineLidarGrid(object):
    """Grid state reconstructed from one dumped lidar frame."""

    def __init__(self, lidar_np, lidar_pose, lidar_config=None):
        lidar_config = lidar_config or {}
        self.data = lidar_np
        self.local_data = lidar_np
        self.lidar_pose = lidar_pose
        self.lidar_range = float(lidar_config.get('range', 50.0))
        self.required_perception_range = float(
            lidar_config.get('required_perception_range',
                             self.lidar_range * 3))
        self.grid_size = float(lidar_config.get('grid_size', 10.0))
        self.density_threshold = float(
            lidar_config.get('density_threshold', 2.0))
        self.grid_local_points = defaultdict(list)
        self.grid_density_dict = {}
        self.sens_grids = set()
        self.req_grids = set()
        self.high_density_grids = set()
        self.low_density_grids = set()

        self.sensor_transform = pose_to_transform(lidar_pose)
        self.perception_grids = self._generate_perception_grids()
        self.global_data = self._local_points_to_global(lidar_np)
        self.update_grid_local_points()
        self.update_grid_density_dict()

    @staticmethod
    def generate_perception_grid_coords(grid_size, half_range, sensor_x,
                                        sensor_y):
        grid_coords = set()
        half_range = int(half_range)
        sensor_x = int(sensor_x)
        sensor_y = int(sensor_y)
        grid_size = int(grid_size)
        start_x = int((sensor_x - half_range) // grid_size) * grid_size
        end_x = int((sensor_x + half_range) // grid_size) * grid_size
        start_y = int((sensor_y - half_range) // grid_size) * grid_size
        end_y = int((sensor_y + half_range) // grid_size) * grid_size
        for x in range(start_x, end_x + grid_size, grid_size):
            for y in range(start_y, end_y + grid_size, grid_size):
                grid_coords.add((x, y))
        return grid_coords

    def _generate_perception_grids(self):
        sensor_x = int(self.sensor_transform.location.x)
        sensor_y = int(self.sensor_transform.location.y)
        grid_coords = self.generate_perception_grid_coords(
            self.grid_size,
            self.required_perception_range,
            sensor_x,
            sensor_y)

        grids = {}
        for x, y in grid_coords:
            grid_id = self.get_point_grid_id((x, y))
            grids[grid_id] = (x, y, x + self.grid_size, y + self.grid_size)
        self.req_grids = set(grids.keys())
        return grids

    def get_point_grid_id(self, point):
        x, y = int(point[0]), int(point[1])
        x_idx = int(x // self.grid_size)
        y_idx = int(y // self.grid_size)
        return '%s_%s' % (x_idx, y_idx)

    def _local_points_to_global(self, lidar_np):
        if lidar_np is None or lidar_np.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return st.lidar_local_to_global(
            lidar_np[:, :3],
            self.sensor_transform)

    def update_grid_local_points(self):
        self.grid_local_points.clear()
        self.sens_grids.clear()
        for local_point, global_point in zip(self.local_data,
                                             self.global_data):
            grid_id = self.get_point_grid_id(global_point)
            if grid_id in self.perception_grids:
                self.grid_local_points[grid_id].append(local_point)
                self.sens_grids.add(grid_id)

    def update_grid_density_dict(self):
        self.grid_density_dict.clear()
        self.high_density_grids.clear()
        self.low_density_grids.clear()
        grid_area = self.grid_size * self.grid_size
        for grid_id in self.perception_grids:
            point_count = len(self.grid_local_points.get(grid_id, []))
            density = 0.0 if point_count == 0 else point_count / grid_area
            self.grid_density_dict[grid_id] = density
            if density >= self.density_threshold:
                self.high_density_grids.add(grid_id)
            else:
                self.low_density_grids.add(grid_id)

    def get_all_points(self):
        return self.data

    def get_local_points_by_grid_ids(self, grid_id_list):
        points = []
        for grid_id in grid_id_list:
            points.extend(self.grid_local_points.get(grid_id, []))
        if not points:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def get_grid_density(self, grid_id):
        return self.grid_density_dict.get(grid_id, 0.0)


class OfflinePerceptionManager(object):
    def __init__(self, lidar):
        self.lidar = lidar
        self.co_manager = OfflineCoManager()
        self.apply_late_fusion = True
        self.do_not_skip_any_cav = False


class OfflineCoManager(object):
    def __init__(self):
        self.enable_grid = False
        self.grid_selection = {}

    def set_grid_selection(self, grid_selection):
        self.grid_selection.update(grid_selection or {})
        self.enable_grid = True

    def clear_grid_selection(self):
        self.grid_selection = {}


class OfflineNetworkManager(object):
    def __init__(self, config=None):
        config = config or {}
        self.subchannel_num = int(
            config.get('subchannel_num',
                       config.get('num_channels', 10)))
        self.use_ns3 = False


class OfflineScheduler(object):
    def __init__(self, network_manager):
        self.network_manager = network_manager
        self.channel_allocation = {}
        self.use_default_subchannel = False

    def set_strategies(self, strategies):
        self.channel_allocation.update(strategies)

    def clear_strategies(self):
        self.channel_allocation.clear()


class OfflineV2XManager(object):
    def __init__(self, cav_world, vehicle_id, params, scheduler=None):
        self.cav_world = cav_world
        self.vehicle_id = vehicle_id
        self.params = params
        self.scheduler = scheduler
        network_config = extract_network_config(cav_world.protocol)
        self.tx_power = float(network_config.get(
            'transmission_power',
            network_config.get('tx_power', 23.0)))
        self.noise_power = float(network_config.get('noise_power', -100.0))
        self.communication_range = float(network_config.get(
            'communication_range', 100.0))
        self.ego_pos = deque([self.get_ego_pos()], maxlen=100)
        self.ego_spd = deque([self.get_ego_speed()], maxlen=100)
        self.cluster_state = {
            'head_id': None,
            'member_ids': set(),
        }

    @property
    def is_ok(self):
        return True

    def get_ego_pos(self):
        return pose_to_transform(
            self.params.get('true_ego_pos') or
            self.params.get('predicted_ego_pos') or
            self.params['lidar_pose'])

    def get_ego_speed(self):
        return float(self.params.get('ego_speed', 0.0))

    def get_ego_dir(self):
        pose = (self.params.get('true_ego_pos') or
                self.params.get('predicted_ego_pos') or
                self.params['lidar_pose'])
        yaw = math.radians(float(pose[4]))
        return (math.cos(yaw), math.sin(yaw), 0.0)


class OfflineVehicleManager(object):
    def __init__(self, cav_world, vehicle_id, cav_content, lidar_config=None,
                 scheduler=None):
        self.cav_world = cav_world
        self.vehicle_id = vehicle_id
        self.is_ok = True
        self.params = cav_content['params']
        self.v2x_manager = OfflineV2XManager(
            cav_world,
            vehicle_id,
            self.params,
            scheduler=scheduler)
        self.perception_manager = OfflinePerceptionManager(
            OfflineLidarGrid(
                cav_content['lidar_np'],
                self.params['lidar_pose'],
                lidar_config=lidar_config))


class OfflineCavWorld(object):
    def __init__(self, frame, ego_id=None, protocol=None,
                 density_threshold=None):
        self.frame = frame
        self.protocol = protocol or {}
        self.ego_id = int(ego_id) if ego_id is not None else self._find_ego_id()
        self.network_manager = OfflineNetworkManager(
            extract_network_config(self.protocol))
        lidar_config = extract_vehicle_lidar_config(self.protocol)
        if density_threshold is not None:
            lidar_config = copy.deepcopy(lidar_config)
            lidar_config['density_threshold'] = density_threshold
        self._vehicle_managers = OrderedDict()
        for vehicle_id, cav_content in frame.items():
            scheduler = OfflineScheduler(self.network_manager)
            self._vehicle_managers[vehicle_id] = OfflineVehicleManager(
                self,
                vehicle_id,
                cav_content,
                lidar_config=lidar_config)
            self._vehicle_managers[vehicle_id].v2x_manager.scheduler = scheduler

    def _find_ego_id(self):
        for vehicle_id, cav_content in self.frame.items():
            if cav_content.get('ego'):
                return vehicle_id
        return next(iter(self.frame.keys()))

    def get_vehicle_managers(self):
        return self._vehicle_managers

    def get_vehicle_manager(self, vehicle_id):
        return self._vehicle_managers.get(vehicle_id)


def pose_to_transform(pose):
    """Convert OpenCDA dumped [x, y, z, roll, yaw, pitch] pose."""
    return Transform(
        Location(pose[0], pose[1], pose[2]),
        Rotation(pitch=pose[5], yaw=pose[4], roll=pose[3]))


def extract_vehicle_lidar_config(protocol):
    try:
        return protocol['vehicle_base']['sensing']['perception']['lidar']
    except (KeyError, TypeError):
        return {}


def extract_network_config(protocol):
    network_config = {}
    try:
        network_config.update(protocol.get('network', {}))
    except AttributeError:
        pass
    return network_config


def clear_sgcp_globals():
    from opencda.core.clustering.utils import common
    common.global_vehicles = {}
    common.global_vms = {}
    common.global_ego_id = 0


def apply_cluster_state(world, clusters):
    """Write cluster head/member relationships back to offline V2X managers."""
    for vm in world.get_vehicle_managers().values():
        vm.v2x_manager.cluster_state = {
            'head_id': vm.vehicle_id,
            'member_ids': set([vm.vehicle_id]),
        }
    for cluster in clusters:
        member_ids = set(int(member_id) for member_id in cluster.members)
        for member_id in member_ids:
            vm = world.get_vehicle_manager(member_id)
            if vm is None:
                continue
            vm.v2x_manager.cluster_state = {
                'head_id': int(cluster.head_id),
                'member_ids': set(member_ids),
            }


def select_sgcp_receiver_id(world, ego_cav_id=None,
                            receiver_policy='ego-cluster-head'):
    """Choose the receiver CAV for constrained SGCP perception replay."""
    ego_id = int(ego_cav_id) if ego_cav_id is not None else world.ego_id
    if receiver_policy == 'ego':
        return ego_id
    if receiver_policy != 'ego-cluster-head':
        raise ValueError('Unknown receiver policy: %s' % receiver_policy)

    ego_vm = world.get_vehicle_manager(ego_id)
    if ego_vm is None:
        return ego_id
    head_id = ego_vm.v2x_manager.cluster_state.get('head_id')
    return int(head_id) if head_id is not None else ego_id


def build_constrained_frame(frame, world, receiver_id,
                            include_unconstrained_cluster=False):
    """
    Build an OpenCOOD frame using online SGCP grid-upload semantics.

    The receiver keeps its full point cloud. Other CAVs only contribute the
    grid-selected local points stored in the receiver's co_manager. This mirrors
    CoperceptionManager.get_data_from_lidar() for offline evaluation.
    """
    receiver_id = int(receiver_id)
    receiver_vm = world.get_vehicle_manager(receiver_id)
    if receiver_vm is None:
        raise ValueError('receiver_id %s is missing from offline world' %
                         receiver_id)

    constrained = OrderedDict()
    communication_bytes = 0
    selected_grid_counts = {}

    def clone_cav(cav_id, lidar_np, is_ego):
        cav = frame[cav_id]
        cloned = OrderedDict()
        cloned['ego'] = bool(is_ego)
        cloned['time_delay'] = cav.get('time_delay', 0)
        cloned['params'] = copy.deepcopy(cav['params'])
        cloned['lidar_np'] = lidar_np
        return cloned

    receiver_lidar = frame[receiver_id]['lidar_np']
    constrained[receiver_id] = clone_cav(
        receiver_id,
        receiver_lidar,
        is_ego=True)

    co_manager = receiver_vm.perception_manager.co_manager
    grid_selection = getattr(co_manager, 'grid_selection', {}) or {}
    for sender_id, grid_ids in grid_selection.items():
        sender_id = int(sender_id)
        if sender_id == receiver_id or sender_id not in frame:
            continue
        sender_vm = world.get_vehicle_manager(sender_id)
        if sender_vm is None:
            continue
        selected_points = sender_vm.perception_manager.lidar.\
            get_local_points_by_grid_ids(grid_ids)
        if selected_points is None or selected_points.size == 0:
            continue
        constrained[sender_id] = clone_cav(
            sender_id,
            selected_points,
            is_ego=False)
        communication_bytes += int(selected_points.nbytes)
        selected_grid_counts[sender_id] = len(grid_ids)

    if include_unconstrained_cluster and not grid_selection:
        member_ids = receiver_vm.v2x_manager.cluster_state.get(
            'member_ids', set())
        for sender_id in sorted(member_ids):
            sender_id = int(sender_id)
            if sender_id == receiver_id or sender_id not in frame:
                continue
            lidar_np = frame[sender_id]['lidar_np']
            constrained[sender_id] = clone_cav(
                sender_id,
                lidar_np,
                is_ego=False)
            communication_bytes += int(lidar_np.nbytes)

    metadata = {
        'receiver_id': receiver_id,
        'source_cav_ids': list(constrained.keys()),
        'communication_bytes': communication_bytes,
        'selected_grid_counts': selected_grid_counts,
        'cluster_member_ids': sorted(
            int(member_id) for member_id in
            receiver_vm.v2x_manager.cluster_state.get('member_ids', set())),
        'channel_allocation': dict(
            getattr(receiver_vm.v2x_manager.scheduler,
                    'channel_allocation', {}) or {}),
    }
    return constrained, metadata
