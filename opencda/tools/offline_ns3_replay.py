# -*- coding: utf-8 -*-
"""
Replay dumped OpenCDA frames through the NS3 bridge without CARLA.

This is a smoke-test utility for validating the CARLA<->NS3 socket protocol and
time synchronization against deterministic OPV2V-style data dumps.
"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import yaml

from opencda.core.clustering.algorithms.clustering.coalition_game import (
    CoalitionGame,
)
from opencda.core.clustering.algorithms.resource_allocation.naive_ra import (
    NaiveRA,
)
from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    clear_sgcp_globals,
)
from opencda.core.networking.ns3_co_simulation.bridge.carla_ns3_bridge import (
    CarlaNs3Bridge,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Replay dumped frames through NS3 without CARLA.')
    parser.add_argument('--dataset-root', required=True,
                        help='Root folder containing scenario subfolders.')
    parser.add_argument('--scenario-id', default=None,
                        help='Scenario folder name. Defaults to the first one.')
    parser.add_argument('--ego-cav-id', default=None,
                        help='Ego CAV id for loading frame transforms.')
    parser.add_argument('--max-frames', type=int, default=3,
                        help='Number of frames to replay. Use 0 for all frames.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from within the scenario.')
    parser.add_argument('--frame-step', type=int, default=1,
                        help='Replay every Nth selected dataset frame.')
    parser.add_argument('--fixed-delta-seconds', type=float, default=None,
                        help='CARLA fixed delta. Defaults to data_protocol world.fixed_delta_seconds.')
    parser.add_argument('--packet-size', type=int, default=10000,
                        help='Bytes per intra-cluster upload request.')
    parser.add_argument('--drain-seconds', type=float, default=0.5,
                        help='Extra NS3 time to advance after the last request.')
    parser.add_argument('--ns3-host', default=None,
                        help='Override NS3 host. Defaults to bridge settings.')
    parser.add_argument('--sync-timeout', type=float, default=10.0,
                        help='Seconds to wait for each sync_ack.')
    parser.add_argument('--lgcp-upload-plan', default=None,
                        help='Optional LGCP upload_plan.csv. If set, replay '
                             'LGCP hierarchy transfers instead of SGCP cluster '
                             'requests.')
    parser.add_argument('--rsu-node-id', type=int, default=None,
                        help='Positive integer node id used for RSU targets in '
                             'LGCP plan. Defaults to max CAV id + 1.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build and print replay requests without opening '
                             'NS3 sockets.')
    return parser.parse_args()


def read_upload_plan(path):
    by_timestamp = defaultdict(list)
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            by_timestamp[row['timestamp']].append(row)
    return by_timestamp


def load_protocol(dataset, scenario_id):
    protocol_path = os.path.join(
        dataset.scenarios[scenario_id]['path'],
        'data_protocol.yaml')
    if not os.path.exists(protocol_path):
        return {}
    with open(protocol_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)


def select_timestamps(dataset, scenario_id, max_frames, start_index, frame_step):
    timestamps = dataset.scenarios[scenario_id]['timestamps']
    selected = timestamps[start_index:] if max_frames == 0 else \
        timestamps[start_index:start_index + max_frames]
    return selected[::max(1, frame_step)]


def fixed_delta_from_protocol(protocol, fallback=0.05):
    try:
        return float(protocol['world']['fixed_delta_seconds'])
    except (KeyError, TypeError, ValueError):
        return fallback


def frame_interval_seconds(timestamps, fixed_delta_seconds):
    if len(timestamps) >= 2:
        try:
            return (int(timestamps[1]) - int(timestamps[0])) * fixed_delta_seconds
        except ValueError:
            pass
    return fixed_delta_seconds


def pose_to_vehicle_state(index, vehicle_id, cav_content):
    params = cav_content['params']
    pose = (
        params.get('true_ego_pos') or
        params.get('predicted_ego_pos') or
        params['lidar_pose'])
    speed = float(params.get('ego_speed', 0.0))
    yaw = math.radians(float(pose[4]))
    velocity = {
        'x': round(speed * math.cos(yaw), 2),
        'y': round(speed * math.sin(yaw), 2),
        'z': 0.0,
    }
    return {
        'id': index,
        'carla_id': int(vehicle_id),
        'position': {
            'x': round(float(pose[0]), 2),
            'y': round(float(pose[1]), 2),
            'z': round(float(pose[2]), 2),
        },
        'velocity': velocity,
        'heading': round(float(pose[4]), 2),
        'speed': round(speed, 2),
    }


def build_world_and_requests(dataset, scenario_id, timestamp, ego_cav_id,
                             protocol, packet_size):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=ego_cav_id)
    vehicle_data = [
        pose_to_vehicle_state(index, vehicle_id, cav_content)
        for index, (vehicle_id, cav_content) in enumerate(frame.items())
    ]

    clear_sgcp_globals()
    world = OfflineCavWorld(frame, ego_id=ego_cav_id, protocol=protocol)
    clusters = CoalitionGame(world).run()
    resource_allocator = NaiveRA(world)
    resource_allocator.set_clusters(clusters)
    resource_allocator.run()

    first_vm = next(iter(world.get_vehicle_managers().values()))
    channel_allocation = dict(first_vm.v2x_manager.scheduler.channel_allocation)
    requests = []
    pkt_id = 1
    for cluster in clusters:
        head_id = int(cluster.head_id)
        for member_id in sorted(cluster.members):
            member_id = int(member_id)
            if member_id == head_id:
                continue
            request = {
                'source': member_id,
                'target': head_id,
                'size': int(packet_size),
                'pkt_id': pkt_id,
            }
            channel = channel_allocation.get((member_id, head_id))
            if channel is not None:
                request['sc_start'] = int(channel)
                request['sc_num'] = 1
            requests.append(request)
            pkt_id += 1

    return vehicle_data, requests, clusters


def resolve_rsu_node_id(dataset, scenario_id, override=None):
    if override is not None:
        if int(override) <= 0:
            raise ValueError('--rsu-node-id must be positive for NS3 replay')
        return int(override)
    positive_ids = []
    for cav_id in dataset.scenarios[scenario_id]['cav_ids']:
        try:
            value = int(cav_id)
        except ValueError:
            continue
        if value > 0:
            positive_ids.append(value)
    return max(positive_ids or [0]) + 1


def request_endpoint_to_int(value, rsu_node_id):
    value = str(value)
    if value.upper() == 'RSU':
        return int(rsu_node_id)
    return int(value)


def pose_to_replay_vehicle_state(index, vehicle_id, cav_content, rsu_node_id):
    state = pose_to_vehicle_state(index, vehicle_id, cav_content)
    if str(vehicle_id) == '-1':
        state['carla_id'] = int(rsu_node_id)
    return state


def build_lgcp_requests(upload_rows, rsu_node_id):
    requests = []
    for pkt_id, row in enumerate(upload_rows, start=1):
        requests.append({
            'source': request_endpoint_to_int(row['source_id'], rsu_node_id),
            'target': request_endpoint_to_int(row['target_id'], rsu_node_id),
            'size': int(float(row['bytes'])),
            'pkt_id': pkt_id,
            'upload_type': row.get('upload_type', ''),
        })
    return requests


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)
    scenario_id = args.scenario_id or next(iter(dataset.scenarios.keys()))
    protocol = load_protocol(dataset, scenario_id)
    lgcp_uploads = read_upload_plan(args.lgcp_upload_plan) \
        if args.lgcp_upload_plan else None
    rsu_node_id = resolve_rsu_node_id(dataset, scenario_id, args.rsu_node_id)
    timestamps = select_timestamps(
        dataset,
        scenario_id,
        args.max_frames,
        args.start_index,
        args.frame_step)
    if not timestamps:
        raise ValueError('No timestamps selected for offline NS3 replay.')

    fixed_delta_seconds = args.fixed_delta_seconds or \
        fixed_delta_from_protocol(protocol)
    frame_interval = frame_interval_seconds(timestamps, fixed_delta_seconds)

    bridge = None
    if not args.dry_run:
        bridge = CarlaNs3Bridge(ns3_host=args.ns3_host) if args.ns3_host else \
            CarlaNs3Bridge()
        bridge.sync_timeout = args.sync_timeout
        bridge.enable_time_sync(True)
        bridge.start()

    try:
        first_vehicle_data = None
        for index, timestamp in enumerate(timestamps):
            if lgcp_uploads is None:
                vehicle_data, requests, clusters = build_world_and_requests(
                    dataset,
                    scenario_id,
                    timestamp,
                    args.ego_cav_id,
                    protocol,
                    args.packet_size)
                cluster_count = len(clusters)
                request_mode = 'sgcp'
            else:
                frame = dataset.load_frame(
                    scenario_id,
                    timestamp,
                    ego_cav_id=args.ego_cav_id)
                vehicle_data = [
                    pose_to_replay_vehicle_state(
                        index, vehicle_id, cav_content, rsu_node_id)
                    for index, (vehicle_id, cav_content) in enumerate(frame.items())
                ]
                requests = build_lgcp_requests(
                    lgcp_uploads.get(timestamp, []),
                    rsu_node_id)
                cluster_count = 0
                request_mode = 'lgcp'

            if args.dry_run:
                print('frame=%s/%s timestamp=%s mode=%s vehicles=%s '
                      'requests=%s bytes=%s dry_run=true' % (
                          index + 1,
                          len(timestamps),
                          timestamp,
                          request_mode,
                          len(vehicle_data),
                          len(requests),
                          sum(item['size'] for item in requests)))
                continue

            if first_vehicle_data is None:
                first_vehicle_data = vehicle_data
                bridge.send_vehicles_num(len(first_vehicle_data))

            sim_time = index * frame_interval
            bridge.send_vehicles_position(vehicle_data)
            if not bridge.sync_with_ns3(sim_time):
                raise RuntimeError(
                    'NS3 sync failed at timestamp %s, sim_time %.3fs' %
                    (timestamp, sim_time))
            if requests:
                bridge.send_transfer_requests(requests)

            print('frame=%s/%s timestamp=%s sim_time=%.3f mode=%s vehicles=%s '
                  'clusters=%s requests=%s bytes=%s' % (
                      index + 1,
                      len(timestamps),
                      timestamp,
                      sim_time,
                      request_mode,
                      len(vehicle_data),
                      cluster_count,
                      len(requests),
                      sum(item['size'] for item in requests)))

        final_time = (len(timestamps) - 1) * frame_interval + args.drain_seconds
        if args.dry_run:
            print('offline_ns3_replay dry_run completed frames=%s' %
                  len(timestamps))
        elif args.drain_seconds > 0:
            bridge.sync_with_ns3(final_time)
            time.sleep(min(args.drain_seconds, 0.2))

        if not args.dry_run:
            print('offline_ns3_replay completed frames=%s final_sync_time=%.3f' %
                  (len(timestamps), final_time))
    finally:
        if bridge is not None:
            bridge.stop()


if __name__ == '__main__':
    main()
