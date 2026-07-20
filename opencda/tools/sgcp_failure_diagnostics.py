# -*- coding: utf-8 -*-
"""Dump SGCP protocol state for AP failure diagnosis.

This tool intentionally avoids running the detector. It rebuilds the same
offline clustering/resource-allocation state used by offline_inference, then
prints vehicle poses, cluster membership, scheduled uploads, selected grids,
and GT object coordinates. When an object diagnostics CSV is provided, GT rows
are annotated with full-reference/method match flags.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
_OPENCOOD_ROOT = os.path.join(_REPO_ROOT, 'opencood')
if _OPENCOOD_ROOT not in sys.path:
    sys.path.insert(0, _OPENCOOD_ROOT)

from opencood.utils.transformation_utils import x_to_world

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    apply_cluster_state,
    build_constrained_frame,
    clear_sgcp_globals,
)
from opencda.core.clustering.algorithms.clustering.coalition_game import (
    CoalitionGame,
)
from opencda.core.clustering.algorithms.clustering.naive_cluster import (
    NaiveCluster,
)
from opencda.core.clustering.algorithms.resource_allocation import (
    build_resource_allocator,
)
from opencda.core.clustering.utils import common
from opencda.tools.offline_inference import (
    apply_resource_overrides,
    cluster_scheduled_grid_selection,
    diversify_scheduled_grid_selection,
    extract_lidar_density_threshold,
    load_protocol,
    randomize_scheduled_grid_selection,
    select_cav_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump SGCP clustering/scheduling/GT diagnostic CSVs.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', default=None)
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='0 means all frames.')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--cav-count', type=int, default=None)
    parser.add_argument('--cav-ids', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--object-diagnostics-csv', default=None,
                        help='Optional CSV from offline_inference '
                             '--object-diagnostics-output.')
    parser.add_argument('--resource-allocation', default='potential_game')
    parser.add_argument('--clustering', default='coalition_game',
                        choices=['coalition_game', 'fixed_first_frame',
                                 'singleton', 'all_in_one'])
    parser.add_argument('--sgcp-grid-selection-mode', default='utility',
                        choices=['utility', 'random', 'spatial_diverse',
                                 'object_clustered'])
    parser.add_argument('--sgcp-grid-score-mode', default='utility',
                        choices=['utility', 'raw_density',
                                 'density_distance'])
    parser.add_argument('--sgcp-upload-mode', default='grid',
                        choices=['grid', 'head_only', 'full_cluster'])
    parser.add_argument('--rho-th', type=float, default=None)
    parser.add_argument('--num-channels', type=int, default=None)
    parser.add_argument('--bandwidth-mhz', type=float, default=None)
    parser.add_argument('--head-rb-budget', type=int, default=None)
    parser.add_argument('--print-top-misses', type=int, default=20)
    return parser.parse_args()


def cav_sort_key(cav_id):
    try:
        return (0, int(cav_id))
    except ValueError:
        return (1, str(cav_id))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_object_diagnostics(path):
    lookup = {}
    if not path:
        return lookup
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            key = (str(row.get('timestamp', '')),
                   str(row.get('gt_object_id', '')))
            if key[1] == '':
                continue
            current = lookup.get(key)
            missed = int(float(row.get('full_detected_method_missed', 0) or 0))
            method = int(float(row.get('method_matched', 0) or 0))
            full = int(float(row.get('full_reference_matched', 0) or 0))
            if current is None or missed > current['full_detected_method_missed']:
                lookup[key] = {
                    'full_reference_matched': full,
                    'method_matched': method,
                    'full_detected_method_missed': missed,
                    'method_best_iou': row.get('method_best_iou', ''),
                    'full_reference_best_iou': row.get(
                        'full_reference_best_iou', ''),
                }
    return lookup


def pose_from_vehicle_yaml(vehicle):
    location = vehicle.get('location', [0.0, 0.0, 0.0])
    angle = vehicle.get('angle', [0.0, 0.0, 0.0])
    return [
        float(location[0]),
        float(location[1]),
        float(location[2]),
        float(angle[0]),
        float(angle[1]),
        float(angle[2]),
    ]


def bbox_center_world(vehicle):
    pose = pose_from_vehicle_yaml(vehicle)
    center = vehicle.get('center', [0.0, 0.0, 0.0])
    local = np.array([
        float(center[0]),
        float(center[1]),
        float(center[2]),
        1.0,
    ])
    return np.dot(x_to_world(pose), local)[:3]


def point_to_ego(point_world, ego_lidar_pose):
    world_to_ego = np.linalg.inv(x_to_world(ego_lidar_pose))
    point = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    return np.dot(world_to_ego, point)[:3]


def vehicle_position(vm):
    pos = vm.v2x_manager.get_ego_pos().location
    return float(pos.x), float(pos.y), float(pos.z)


def vehicle_distance(world, a_id, b_id):
    vm_a = world.get_vehicle_manager(int(a_id))
    vm_b = world.get_vehicle_manager(int(b_id))
    if vm_a is None or vm_b is None:
        return ''
    ax, ay, az = vehicle_position(vm_a)
    bx, by, bz = vehicle_position(vm_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def build_world_state(frame, protocol, args, timestamp):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=args.ego_cav_id,
        protocol=protocol,
        density_threshold=args.rho_th)
    if args.clustering in ['coalition_game', 'fixed_first_frame']:
        clustering_algorithm = CoalitionGame(world)
    elif args.clustering == 'singleton':
        clustering_algorithm = NaiveCluster(world, all_in_one=False)
    elif args.clustering == 'all_in_one':
        clustering_algorithm = NaiveCluster(world, all_in_one=True)
    else:
        raise ValueError(args.clustering)
    clusters = clustering_algorithm.run()
    apply_cluster_state(world, clusters)

    allocator = build_resource_allocator(args.resource_allocation, world)
    if hasattr(allocator, 'grid_score_mode'):
        allocator.grid_score_mode = args.sgcp_grid_score_mode
    apply_resource_overrides(
        allocator,
        world,
        num_channels=args.num_channels,
        bandwidth_mhz=args.bandwidth_mhz,
        head_rb_budget=args.head_rb_budget)
    allocator.set_clusters(clusters)
    allocator.run()

    if args.sgcp_grid_selection_mode == 'random':
        randomize_scheduled_grid_selection(world, clusters, timestamp)
    elif args.sgcp_grid_selection_mode == 'spatial_diverse':
        diversify_scheduled_grid_selection(world, clusters)
    elif args.sgcp_grid_selection_mode == 'object_clustered':
        cluster_scheduled_grid_selection(world, clusters)
    return world, clusters, allocator


def cluster_id_by_member(clusters):
    mapping = {}
    for cluster_index, cluster in enumerate(clusters):
        for member in cluster.members:
            mapping[int(member)] = cluster_index
    return mapping


def selected_points_for(world, receiver_id, sender_id, grid_ids):
    sender_vm = world.get_vehicle_manager(int(sender_id))
    if sender_vm is None:
        return 0
    points = sender_vm.perception_manager.lidar.get_local_points_by_grid_ids(
        grid_ids)
    if points is None:
        return 0
    return int(points.shape[0])


def grid_centers(world, sender_id, grid_ids, limit=12):
    sender_vm = world.get_vehicle_manager(int(sender_id))
    if sender_vm is None:
        return ''
    grid_size = sender_vm.perception_manager.lidar.grid_size
    centers = []
    for grid_id in list(grid_ids)[:limit]:
        try:
            x_idx, y_idx = [int(item) for item in str(grid_id).split('_')]
        except ValueError:
            continue
        centers.append([x_idx * grid_size + grid_size / 2.0,
                        y_idx * grid_size + grid_size / 2.0])
    return json.dumps(centers)


def world_grid_id(world, point_world):
    for vm in world.get_vehicle_managers().values():
        return vm.perception_manager.lidar.get_point_grid_id(point_world)
    x_idx = int(float(point_world[0]) // 10.0)
    y_idx = int(float(point_world[1]) // 10.0)
    return '%s_%s' % (x_idx, y_idx)


def grid_point_count(world, cav_id, grid_id):
    vm = world.get_vehicle_manager(int(cav_id))
    if vm is None:
        return 0
    lidar = vm.perception_manager.lidar
    return len(lidar.grid_local_points.get(str(grid_id), []))


def grid_membership_flags(world, allocator, cav_id, grid_id):
    """Return receiver-side grid memberships used to debug PCS demands."""
    vehicle = common.global_vehicles.get(int(cav_id))
    if vehicle is None:
        return {
            'in_req': 0,
            'in_high_density': 0,
            'in_pcs_blind_spot': '',
        }
    grid_id = str(grid_id)
    in_req = int(
        grid_id in set(
            str(item) for item in getattr(vehicle, 'req_grids', set())))
    in_high_density = int(
        grid_id in set(
            str(item)
            for item in getattr(vehicle, 'high_density_grids', set())))
    in_pcs_blind_spot = ''
    if hasattr(allocator, '_get_vehicle_blind_spots'):
        try:
            min_division = getattr(
                allocator,
                'active_blind_spot_min_division',
                getattr(allocator, 'blind_spot_min_division', 1))
            blind_spots = allocator._get_vehicle_blind_spots(
                int(cav_id),
                min_division)
            in_pcs_blind_spot = int(any(
                grid_id in set(str(item) for item in spot_grids)
                for spot_grids in blind_spots.values()))
        except Exception:
            in_pcs_blind_spot = ''
    return {
        'in_req': in_req,
        'in_high_density': in_high_density,
        'in_pcs_blind_spot': in_pcs_blind_spot,
    }


def diagnose(args):
    ensure_dir(args.output_dir)
    dataset = OPV2VFrameDataset(args.dataset_root)
    scenario_id = args.scenario_id or next(iter(dataset.scenarios.keys()))
    protocol = load_protocol(dataset, scenario_id)
    timestamps = dataset.scenarios[scenario_id]['timestamps']
    if args.max_frames == 0:
        selected_timestamps = timestamps[args.start_index:]
    else:
        selected_timestamps = timestamps[
            args.start_index:args.start_index + args.max_frames]
    miss_lookup = load_object_diagnostics(args.object_diagnostics_csv)

    vehicle_rows = []
    cluster_rows = []
    schedule_rows = []
    gt_rows = []
    summary = {
        'scenario_id': scenario_id,
        'frames': len(selected_timestamps),
        'resource_allocation': args.resource_allocation,
        'grid_selection_mode': args.sgcp_grid_selection_mode,
        'num_channels': args.num_channels,
        'bandwidth_mhz': args.bandwidth_mhz,
        'rho_th': args.rho_th,
        'frame_count': 0,
        'total_payload_bytes': 0,
        'scheduled_links': 0,
        'selected_grids': 0,
        'missed_gt_count': 0,
    }
    missed_by_cluster = Counter()
    missed_by_nearest_head = Counter()
    missed_by_nearest_uploaded = Counter()
    missed_by_grid = Counter()

    for timestamp in selected_timestamps:
        frame = dataset.load_frame(
            scenario_id,
            timestamp,
            ego_cav_id=args.ego_cav_id,
            cav_ids=select_cav_ids(
                dataset,
                scenario_id,
                ego_cav_id=args.ego_cav_id,
                cav_count=args.cav_count,
                cav_ids=args.cav_ids))
        world, clusters, allocator = build_world_state(
            frame,
            protocol,
            args,
            timestamp)
        member_cluster_id = cluster_id_by_member(clusters)
        summary['frame_count'] += 1

        uploaded_sources = set()
        scheduled_sources_by_receiver = defaultdict(set)
        scheduled_receivers_by_source = defaultdict(set)
        scheduled_grids_by_link = {}

        for cluster_index, cluster in enumerate(clusters):
            head_id = int(cluster.head_id)
            head_vm = world.get_vehicle_manager(head_id)
            hx, hy, hz = vehicle_position(head_vm)
            members = sorted(int(member_id) for member_id in cluster.members)
            distances = {
                str(member_id): vehicle_distance(world, head_id, member_id)
                for member_id in members if member_id != head_id
            }
            numeric_distances = [
                value for value in distances.values()
                if isinstance(value, float)
            ]
            cluster_rows.append({
                'timestamp': timestamp,
                'cluster_index': cluster_index,
                'head_id': head_id,
                'head_x': '%.3f' % hx,
                'head_y': '%.3f' % hy,
                'member_ids': ';'.join(str(item) for item in members),
                'member_count': len(members),
                'avg_member_distance': (
                    '' if not numeric_distances else
                    '%.3f' % (sum(numeric_distances) /
                              float(len(numeric_distances)))),
                'max_member_distance': (
                    '' if not numeric_distances else
                    '%.3f' % max(numeric_distances)),
                'member_distances_json': json.dumps(distances, sort_keys=True),
            })

        for cav_id, vm in world.get_vehicle_managers().items():
            cav_id = int(cav_id)
            pos = vm.v2x_manager.get_ego_pos()
            loc = pos.location
            rot = pos.rotation
            cluster_id = member_cluster_id.get(cav_id, '')
            cluster_members = []
            cluster_head = ''
            if cluster_id != '':
                cluster = clusters[int(cluster_id)]
                cluster_head = int(cluster.head_id)
                cluster_members = sorted(int(item) for item in cluster.members)
            lidar = vm.perception_manager.lidar
            vehicle_rows.append({
                'timestamp': timestamp,
                'cav_id': cav_id,
                'x': '%.3f' % loc.x,
                'y': '%.3f' % loc.y,
                'z': '%.3f' % loc.z,
                'yaw': '%.3f' % rot.yaw,
                'speed': '%.3f' % vm.v2x_manager.get_ego_speed(),
                'cluster_index': cluster_id,
                'cluster_head': cluster_head,
                'is_cluster_head': int(cav_id == cluster_head),
                'cluster_members': ';'.join(str(item)
                                            for item in cluster_members),
                'raw_points': int(frame[cav_id]['lidar_np'].shape[0]),
                'sens_grids': len(lidar.sens_grids),
                'high_density_grids': len(lidar.high_density_grids),
            })

        for cluster in clusters:
            receiver_id = int(cluster.head_id)
            receiver_vm = world.get_vehicle_manager(receiver_id)
            if receiver_vm is None:
                continue
            metadata_frame, metadata = build_constrained_frame(
                frame,
                world,
                receiver_id,
                upload_mode=args.sgcp_upload_mode)
            del metadata_frame
            summary['total_payload_bytes'] += int(
                metadata.get('communication_bytes', 0))
            co_manager = receiver_vm.perception_manager.co_manager
            grid_selection = getattr(co_manager, 'grid_selection', {}) or {}
            channel_allocation = metadata.get('channel_allocation', {}) or {}
            for sender_id, grid_ids in sorted(grid_selection.items()):
                sender_id = int(sender_id)
                if sender_id == receiver_id:
                    continue
                point_count = selected_points_for(
                    world,
                    receiver_id,
                    sender_id,
                    grid_ids)
                byte_count = point_count * 16
                subchannel = channel_allocation.get(
                    (sender_id, receiver_id), '')
                uploaded_sources.add(sender_id)
                scheduled_sources_by_receiver[receiver_id].add(sender_id)
                scheduled_receivers_by_source[sender_id].add(receiver_id)
                scheduled_grids_by_link[(sender_id, receiver_id)] = set(
                    str(item) for item in grid_ids)
                summary['scheduled_links'] += 1
                summary['selected_grids'] += len(grid_ids)
                schedule_rows.append({
                    'timestamp': timestamp,
                    'receiver_id': receiver_id,
                    'sender_id': sender_id,
                    'subchannel': subchannel,
                    'receiver_cluster_index': member_cluster_id.get(
                        receiver_id, ''),
                    'sender_cluster_index': member_cluster_id.get(
                        sender_id, ''),
                    'distance': '%.3f' % vehicle_distance(
                        world,
                        receiver_id,
                        sender_id),
                    'selected_grid_count': len(grid_ids),
                    'selected_point_count': point_count,
                    'selected_bytes': byte_count,
                    'grid_ids_head': ';'.join(str(item)
                                              for item in list(grid_ids)[:20]),
                    'grid_centers_head_json': grid_centers(
                        world,
                        sender_id,
                        grid_ids),
                })

        ego_lidar_pose = next(
            cav['params']['lidar_pose'] for cav in frame.values()
            if cav['ego'])
        vehicles = next(iter(frame.values()))['params'].get('vehicles', {})
        head_ids = [int(cluster.head_id) for cluster in clusters]
        for object_id, vehicle in vehicles.items():
            object_id = str(object_id)
            center_world = bbox_center_world(vehicle)
            center_ego = point_to_ego(center_world, ego_lidar_pose)
            object_grid_id = world_grid_id(world, center_world)
            diag = miss_lookup.get((str(timestamp), object_id), {})
            nearest_cav = None
            nearest_distance = None
            for cav_id, vm in world.get_vehicle_managers().items():
                vx, vy, vz = vehicle_position(vm)
                dist = math.sqrt(
                    (center_world[0] - vx) ** 2 +
                    (center_world[1] - vy) ** 2 +
                    (center_world[2] - vz) ** 2)
                if nearest_distance is None or dist < nearest_distance:
                    nearest_distance = dist
                    nearest_cav = int(cav_id)
            nearest_head = None
            nearest_head_distance = None
            for head_id in head_ids:
                vm = world.get_vehicle_manager(head_id)
                hx, hy, hz = vehicle_position(vm)
                dist = math.sqrt(
                    (center_world[0] - hx) ** 2 +
                    (center_world[1] - hy) ** 2 +
                    (center_world[2] - hz) ** 2)
                if (nearest_head_distance is None or
                        dist < nearest_head_distance):
                    nearest_head_distance = dist
                    nearest_head = head_id
            sensing_cavs = []
            same_cluster_sensing_cavs = []
            object_cluster_id = member_cluster_id.get(nearest_cav, '')
            for cav_id, vm in world.get_vehicle_managers().items():
                cav_id = int(cav_id)
                lidar = vm.perception_manager.lidar
                if object_grid_id in set(str(item) for item in lidar.sens_grids):
                    sensing_cavs.append(cav_id)
                    if member_cluster_id.get(cav_id, '') == object_cluster_id:
                        same_cluster_sensing_cavs.append(cav_id)
            scheduled_covering_links = []
            scheduled_covering_point_count = 0
            nearest_head_covering_point_count = 0
            for (sender_id, receiver_id), grid_ids in scheduled_grids_by_link.items():
                if object_grid_id in grid_ids:
                    scheduled_covering_links.append((sender_id, receiver_id))
                    points_in_grid = grid_point_count(
                        world, sender_id, object_grid_id)
                    scheduled_covering_point_count += points_in_grid
                    if int(receiver_id) == int(nearest_head):
                        nearest_head_covering_point_count += points_in_grid
            nearest_covering_links = [
                link for link in scheduled_covering_links
                if int(link[0]) == int(nearest_cav)
            ]
            nearest_cav_object_grid_points = grid_point_count(
                world, nearest_cav, object_grid_id)
            nearest_head_grid_flags = grid_membership_flags(
                world,
                allocator,
                nearest_head,
                object_grid_id)
            full_missed = int(diag.get('full_detected_method_missed', 0) or 0)
            if full_missed:
                summary['missed_gt_count'] += 1
                missed_by_cluster[member_cluster_id.get(nearest_cav, '')] += 1
                missed_by_nearest_head[nearest_head] += 1
                missed_by_nearest_uploaded[int(nearest_cav in
                                               uploaded_sources)] += 1
                missed_by_grid[
                    '%s_%s' % (
                        int(math.floor(center_world[0] / 10.0)),
                        int(math.floor(center_world[1] / 10.0)))] += 1
            gt_rows.append({
                'timestamp': timestamp,
                'object_id': object_id,
                'bp_id': vehicle.get('bp_id', ''),
                'world_x': '%.3f' % center_world[0],
                'world_y': '%.3f' % center_world[1],
                'world_z': '%.3f' % center_world[2],
                'ego_x': '%.3f' % center_ego[0],
                'ego_y': '%.3f' % center_ego[1],
                'ego_z': '%.3f' % center_ego[2],
                'extent_x': vehicle.get('extent', [''])[0],
                'extent_y': vehicle.get('extent', ['', ''])[1],
                'yaw': vehicle.get('angle', ['', ''])[1],
                'object_grid_id': object_grid_id,
                'sensing_cavs': ';'.join(str(item)
                                         for item in sorted(sensing_cavs)),
                'same_cluster_sensing_cavs': ';'.join(
                    str(item) for item in sorted(same_cluster_sensing_cavs)),
                'scheduled_covering_links': ';'.join(
                    '%s>%s' % (sender, receiver)
                    for sender, receiver in sorted(scheduled_covering_links)),
                'scheduled_covering_link_count': len(scheduled_covering_links),
                'scheduled_covering_point_count':
                    scheduled_covering_point_count,
                'nearest_head_covering_point_count':
                    nearest_head_covering_point_count,
                'nearest_cav_object_grid_points':
                    nearest_cav_object_grid_points,
                'nearest_cav_selected_object_grid': int(
                    len(nearest_covering_links) > 0),
                'nearest_cav': nearest_cav,
                'nearest_cav_distance': '%.3f' % nearest_distance,
                'nearest_cav_cluster_index': member_cluster_id.get(
                    nearest_cav, ''),
                'nearest_head': nearest_head,
                'nearest_head_distance': '%.3f' % nearest_head_distance,
                'nearest_head_object_grid_in_req':
                    nearest_head_grid_flags['in_req'],
                'nearest_head_object_grid_in_high_density':
                    nearest_head_grid_flags['in_high_density'],
                'nearest_head_object_grid_in_pcs_blind_spot':
                    nearest_head_grid_flags['in_pcs_blind_spot'],
                'nearest_cav_uploaded_anywhere': int(
                    nearest_cav in uploaded_sources),
                'nearest_cav_uploaded_to_receivers': ';'.join(
                    str(item) for item in sorted(
                        scheduled_receivers_by_source.get(nearest_cav, []))),
                'full_reference_matched': diag.get(
                    'full_reference_matched', ''),
                'method_matched': diag.get('method_matched', ''),
                'full_detected_method_missed': full_missed,
                'full_reference_best_iou': diag.get(
                    'full_reference_best_iou', ''),
                'method_best_iou': diag.get('method_best_iou', ''),
            })

    summary['avg_payload_bytes_per_frame'] = (
        summary['total_payload_bytes'] / float(max(summary['frame_count'], 1)))
    summary['avg_scheduled_links_per_frame'] = (
        summary['scheduled_links'] / float(max(summary['frame_count'], 1)))
    summary['avg_selected_grids_per_link'] = (
        summary['selected_grids'] / float(max(summary['scheduled_links'], 1)))
    summary['missed_by_nearest_cluster'] = dict(missed_by_cluster)
    summary['missed_by_nearest_head'] = {
        str(key): value for key, value in missed_by_nearest_head.items()
    }
    summary['missed_nearest_cav_uploaded_anywhere'] = {
        str(key): value for key, value in missed_by_nearest_uploaded.items()
    }
    summary['top_missed_world_grids'] = missed_by_grid.most_common(
        args.print_top_misses)

    write_csv(
        os.path.join(args.output_dir, 'vehicles.csv'),
        vehicle_rows,
        [
            'timestamp', 'cav_id', 'x', 'y', 'z', 'yaw', 'speed',
            'cluster_index', 'cluster_head', 'is_cluster_head',
            'cluster_members', 'raw_points', 'sens_grids',
            'high_density_grids',
        ])
    write_csv(
        os.path.join(args.output_dir, 'clusters.csv'),
        cluster_rows,
        [
            'timestamp', 'cluster_index', 'head_id', 'head_x', 'head_y',
            'member_ids', 'member_count', 'avg_member_distance',
            'max_member_distance', 'member_distances_json',
        ])
    write_csv(
        os.path.join(args.output_dir, 'schedules.csv'),
        schedule_rows,
        [
            'timestamp', 'receiver_id', 'sender_id', 'subchannel',
            'receiver_cluster_index', 'sender_cluster_index', 'distance',
            'selected_grid_count', 'selected_point_count', 'selected_bytes',
            'grid_ids_head', 'grid_centers_head_json',
        ])
    write_csv(
        os.path.join(args.output_dir, 'gt_objects.csv'),
        gt_rows,
        [
            'timestamp', 'object_id', 'bp_id',
            'world_x', 'world_y', 'world_z',
            'ego_x', 'ego_y', 'ego_z',
            'extent_x', 'extent_y', 'yaw',
            'object_grid_id', 'sensing_cavs', 'same_cluster_sensing_cavs',
            'scheduled_covering_links', 'scheduled_covering_link_count',
            'scheduled_covering_point_count',
            'nearest_head_covering_point_count',
            'nearest_cav_object_grid_points',
            'nearest_cav_selected_object_grid',
            'nearest_cav', 'nearest_cav_distance',
            'nearest_cav_cluster_index', 'nearest_head',
            'nearest_head_distance',
            'nearest_head_object_grid_in_req',
            'nearest_head_object_grid_in_high_density',
            'nearest_head_object_grid_in_pcs_blind_spot',
            'nearest_cav_uploaded_anywhere',
            'nearest_cav_uploaded_to_receivers',
            'full_reference_matched', 'method_matched',
            'full_detected_method_missed',
            'full_reference_best_iou', 'method_best_iou',
        ])
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    args = parse_args()
    diagnose(args)


if __name__ == '__main__':
    main()
