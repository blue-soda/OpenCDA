# -*- coding: utf-8 -*-
"""
Replay SGCP clustering from an OPV2V-style data dump.
"""

import argparse
import os
import time

import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    apply_cluster_state,
    clear_sgcp_globals,
)
from opencda.core.clustering.algorithms.clustering.coalition_game import (
    CoalitionGame,
)
from opencda.core.clustering.algorithms.clustering.naive_cluster import (
    NaiveCluster,
)
from opencda.core.clustering.utils import common
from opencda.core.clustering.algorithms.resource_allocation import (
    build_resource_allocator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Replay SGCP clustering from dumped OPV2V-style data.')
    parser.add_argument('--dataset-root', required=True,
                        help='Root folder containing scenario subfolders.')
    parser.add_argument('--scenario-id', default=None,
                        help='Scenario folder name. Defaults to the first one.')
    parser.add_argument('--timestamp', default=None,
                        help='Frame timestamp. Defaults to the first frame.')
    parser.add_argument('--ego-cav-id', default=None,
                        help='Ego CAV id. Defaults to the first CAV.')
    parser.add_argument('--max-frames', type=int, default=1,
                        help='Number of frames to replay. Use 0 for all frames.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from within the scenario.')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print aggregate metrics for multi-frame replay.')
    parser.add_argument('--resource-allocation', default=None,
                        help='Resource allocation algorithm: potential_game, pcs, mws, random, naive.')
    parser.add_argument('--clustering', default='coalition_game',
                        choices=['coalition_game', 'singleton', 'all_in_one'],
                        help='Clustering algorithm: coalition_game, singleton, all_in_one.')
    parser.add_argument('--t-min-stab', type=float, default=None,
                        help='Override CoalitionGame Params.T_min_stab in seconds. Use 0 for no stability window.')
    return parser.parse_args()


def load_protocol(dataset, scenario_id):
    protocol_path = os.path.join(
        dataset.scenarios[scenario_id]['path'],
        'data_protocol.yaml')
    if not os.path.exists(protocol_path):
        return {}
    with open(protocol_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)


def select_frames(dataset, scenario_id, timestamp, max_frames, start_index):
    if timestamp is not None:
        return [(scenario_id, timestamp)]
    timestamps = dataset.scenarios[scenario_id]['timestamps']
    if max_frames == 0:
        selected = timestamps[start_index:]
    else:
        selected = timestamps[start_index:start_index + max_frames]
    return [(scenario_id, frame_timestamp) for frame_timestamp in selected]


def get_resource_allocation_name(protocol, override=None):
    if override:
        return override
    try:
        return protocol['resource_allocation']['algorithm']
    except (KeyError, TypeError):
        return 'potential_game'


def replay_frame(dataset, scenario_id, timestamp, ego_cav_id, protocol,
                 resource_allocation=None, t_min_stab=None,
                 clustering='coalition_game'):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=ego_cav_id)
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol)
    start_time = time.time()
    if clustering == 'coalition_game':
        clustering_algorithm = CoalitionGame(world)
    elif clustering == 'singleton':
        clustering_algorithm = NaiveCluster(world, all_in_one=False)
    elif clustering == 'all_in_one':
        clustering_algorithm = NaiveCluster(world, all_in_one=True)
    else:
        raise ValueError('Unknown clustering algorithm: %s' % clustering)
    if t_min_stab is not None and hasattr(clustering_algorithm, 'p'):
        clustering_algorithm.p.T_min_stab = t_min_stab
    clusters = clustering_algorithm.run()
    apply_cluster_state(world, clusters)
    ra_start_time = time.time()
    ra_name = get_resource_allocation_name(protocol, resource_allocation)
    resource_allocator = build_resource_allocator(ra_name, world)
    resource_allocator.set_clusters(clusters)
    resource_allocator.run()
    ra_elapsed_ms = (time.time() - ra_start_time) * 1000.0
    elapsed_ms = (time.time() - start_time) * 1000.0
    channel_allocation = {}
    first_vm = next(iter(world.get_vehicle_managers().values()))
    if first_vm.v2x_manager.scheduler is not None:
        channel_allocation = dict(
            first_vm.v2x_manager.scheduler.channel_allocation)
    cluster_summary = [
        {
            'head_id': cluster.head_id,
            'members': sorted(cluster.members),
            'size': cluster.size(),
        }
        for cluster in clusters
    ]
    return {
        'timestamp': timestamp,
        'cav_count': len(frame),
        'cluster_count': len(cluster_summary),
        'avg_cluster_size': (
            sum(item['size'] for item in cluster_summary) /
            float(len(cluster_summary) or 1)),
        'elapsed_ms': elapsed_ms,
        'resource_allocation_ms': ra_elapsed_ms,
        'resource_allocation': ra_name,
        'clustering': clustering,
        't_min_stab': (
            common.Params().T_min_stab if t_min_stab is None else t_min_stab),
        'channel_allocation_count': len(channel_allocation),
        'channel_allocation_sample': sorted(
            channel_allocation.items())[:10],
        'clusters': cluster_summary,
    }


def cluster_signature(cluster):
    return tuple(sorted(cluster['members']))


def vehicle_to_head(clusters):
    mapping = {}
    for cluster in clusters:
        for member_id in cluster['members']:
            mapping[member_id] = cluster['head_id']
    return mapping


def summarize_replay(results):
    if not results:
        return {}

    frame_count = len(results)
    avg_cluster_count = (
        sum(item['cluster_count'] for item in results) / float(frame_count))
    avg_cluster_size = (
        sum(item['avg_cluster_size'] for item in results) / float(frame_count))
    isolated_counts = [
        sum(1 for cluster in item['clusters'] if cluster['size'] == 1)
        for item in results
    ]

    reconfiguration_events = 0
    vehicle_head_changes = 0
    previous_mapping = None
    for item in results:
        mapping = vehicle_to_head(item['clusters'])
        if previous_mapping is not None:
            changed = 0
            for vehicle_id, head_id in mapping.items():
                if previous_mapping.get(vehicle_id) != head_id:
                    changed += 1
            if changed:
                reconfiguration_events += 1
                vehicle_head_changes += changed
        previous_mapping = mapping

    lifetimes = []
    active_lifetimes = {}
    previous_signatures = set()
    for item in results:
        current_signatures = set(
            cluster_signature(cluster) for cluster in item['clusters'])
        for signature in list(active_lifetimes.keys()):
            if signature not in current_signatures:
                lifetimes.append(active_lifetimes.pop(signature))
        for signature in current_signatures:
            if signature in previous_signatures:
                active_lifetimes[signature] += 1
            else:
                active_lifetimes[signature] = 1
        previous_signatures = current_signatures
    lifetimes.extend(active_lifetimes.values())

    avg_lifetime = (
        sum(lifetimes) / float(len(lifetimes))) if lifetimes else 0.0
    max_lifetime = max(lifetimes) if lifetimes else 0
    min_lifetime = min(lifetimes) if lifetimes else 0

    return {
        'frame_count': frame_count,
        'avg_cluster_count': avg_cluster_count,
        'avg_cluster_size': avg_cluster_size,
        'avg_isolated_cavs': (
            sum(isolated_counts) / float(frame_count)),
        'max_isolated_cavs': max(isolated_counts),
        'reconfiguration_events': reconfiguration_events,
        'vehicle_head_changes': vehicle_head_changes,
        'avg_cluster_lifetime_frames': avg_lifetime,
        'min_cluster_lifetime_frames': min_lifetime,
        'max_cluster_lifetime_frames': max_lifetime,
        'avg_elapsed_ms': (
            sum(item['elapsed_ms'] for item in results) / float(frame_count)),
        'avg_resource_allocation_ms': (
            sum(item['resource_allocation_ms'] for item in results) /
            float(frame_count)),
    }


def print_frame_result(index, frame_total, scenario_id, result):
    print(
        'frame=%s/%s scenario=%s timestamp=%s cavs=%s clusters=%s '
        'avg_cluster_size=%.2f channel_allocations=%s '
        'elapsed_ms=%.2f ra_ms=%.2f' % (
            index,
            frame_total,
            scenario_id,
            result['timestamp'],
            result['cav_count'],
            result['cluster_count'],
            result['avg_cluster_size'],
            result['channel_allocation_count'],
            result['elapsed_ms'],
            result['resource_allocation_ms']))
    print('  channel_sample=%s' % (
        result['channel_allocation_sample'],))
    for cluster in result['clusters']:
        print('  head=%s members=%s' % (
            cluster['head_id'],
            cluster['members']))


def print_summary(summary):
    print('summary frames=%s avg_clusters=%.2f avg_cluster_size=%.2f '
          'avg_isolated_cavs=%.2f max_isolated_cavs=%s' % (
              summary['frame_count'],
              summary['avg_cluster_count'],
              summary['avg_cluster_size'],
              summary['avg_isolated_cavs'],
              summary['max_isolated_cavs']))
    print('summary reconfiguration_events=%s vehicle_head_changes=%s' % (
        summary['reconfiguration_events'],
        summary['vehicle_head_changes']))
    print('summary cluster_lifetime_frames avg=%.2f min=%s max=%s' % (
        summary['avg_cluster_lifetime_frames'],
        summary['min_cluster_lifetime_frames'],
        summary['max_cluster_lifetime_frames']))
    print('summary runtime_ms avg_total=%.2f avg_ra=%.2f' % (
        summary['avg_elapsed_ms'],
        summary['avg_resource_allocation_ms']))


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)
    scenario_id = args.scenario_id or next(iter(dataset.scenarios.keys()))
    protocol = load_protocol(dataset, scenario_id)
    frames = select_frames(
        dataset,
        scenario_id,
        args.timestamp,
        args.max_frames,
        args.start_index)

    results = []
    for index, (sid, timestamp) in enumerate(frames, start=1):
        result = replay_frame(
            dataset,
            sid,
            timestamp,
            args.ego_cav_id,
            protocol,
            resource_allocation=args.resource_allocation,
            t_min_stab=args.t_min_stab,
            clustering=args.clustering)
        results.append(result)
        if not args.summary_only:
            print_frame_result(index, len(frames), sid, result)

    if len(results) > 1 or args.summary_only:
        print_summary(summarize_replay(results))


if __name__ == '__main__':
    main()
