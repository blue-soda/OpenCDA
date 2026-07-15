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
    parser.add_argument('--n-max', type=int, default=None,
                        help='Override CoalitionGame Params.N_max.')
    parser.add_argument('--rho-th', type=float, default=None,
                        help='Override lidar density_threshold / rho_th.')
    parser.add_argument('--cav-count', type=int, default=None,
                        help='Use the first N CAVs in numeric order, keeping ego included.')
    parser.add_argument('--cav-ids', default=None,
                        help='Comma-separated CAV ids to replay, e.g. 1,2,3.')
    parser.add_argument('--num-channels', type=int, default=None,
                        help='Override SGCP resource allocation channel count.')
    parser.add_argument('--bandwidth-mhz', type=float, default=None,
                        help='Override SGCP total bandwidth in MHz.')
    parser.add_argument('--trigger-speed-source', default='pose_delta',
                        choices=['pose_delta', 'dump'],
                        help='Speed source for offline relative-speed trigger.')
    parser.add_argument('--trigger-frame-interval-sec', type=float,
                        default=0.1,
                        help='Frame interval in seconds for pose_delta speed.')
    parser.add_argument('--trigger-relative-speed-threshold', type=float,
                        default=5.0,
                        help='Relative speed threshold. Unit is m/s for '
                             'pose_delta and km/h for dump ego_speed.')
    parser.add_argument('--print-topology-events', action='store_true',
                        help='Print per-transition topology trigger details.')
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


def cav_sort_key(cav_id):
    try:
        return (0, int(cav_id))
    except ValueError:
        return (1, str(cav_id))


def select_cav_ids(dataset, scenario_id, ego_cav_id=None, cav_count=None,
                   cav_ids=None):
    scenario_cav_ids = sorted(
        dataset.scenarios[scenario_id]['cav_ids'],
        key=cav_sort_key)
    if cav_ids:
        selected = [item.strip() for item in cav_ids.split(',')
                    if item.strip()]
    elif cav_count is not None:
        if cav_count <= 0:
            raise ValueError('--cav-count must be positive')
        selected = scenario_cav_ids[:cav_count]
    else:
        return None

    if ego_cav_id is not None:
        ego_id = str(ego_cav_id)
        if ego_id not in selected:
            selected = [ego_id] + [item for item in selected
                                   if item != ego_id]
            if cav_count is not None:
                selected = selected[:cav_count]
    return selected


def get_resource_allocation_name(protocol, override=None):
    if override:
        return override
    try:
        return protocol['resource_allocation']['algorithm']
    except (KeyError, TypeError):
        return 'potential_game'


def extract_lidar_density_threshold(protocol):
    try:
        return float(
            protocol['vehicle_base']['sensing']['perception']['lidar'].get(
                'density_threshold', 2.0))
    except (AttributeError, KeyError, TypeError):
        return 2.0


def apply_resource_overrides(resource_allocator, world, num_channels=None,
                             bandwidth_mhz=None):
    if num_channels is not None:
        if num_channels <= 0:
            raise ValueError('--num-channels must be positive')
        world.network_manager.subchannel_num = int(num_channels)
    if not hasattr(resource_allocator, 'p'):
        return
    if num_channels is not None:
        resource_allocator.p.num_channels = int(num_channels)
    if bandwidth_mhz is not None:
        if bandwidth_mhz <= 0:
            raise ValueError('--bandwidth-mhz must be positive')
        resource_allocator.p.bandwidth_all = float(bandwidth_mhz) * (10 ** 6)
    if num_channels is not None or bandwidth_mhz is not None:
        resource_allocator.p.bandwidth_per_channel = (
            resource_allocator.p.bandwidth_all /
            resource_allocator.p.num_channels)


def transform_xy(transform):
    return (float(transform.location.x), float(transform.location.y))


def distance_xy(a_xy, b_xy):
    dx = a_xy[0] - b_xy[0]
    dy = a_xy[1] - b_xy[1]
    return (dx * dx + dy * dy) ** 0.5


def collect_topology_state(world):
    vehicle_states = {}
    vms = world.get_vehicle_managers()
    communication_range = 0.0
    for vehicle_id, vm in vms.items():
        pos = transform_xy(vm.v2x_manager.get_ego_pos())
        speed = float(vm.v2x_manager.get_ego_speed())
        communication_range = max(
            communication_range,
            float(getattr(vm.v2x_manager, 'communication_range', 0.0)))
        vehicle_states[int(vehicle_id)] = {
            'position_xy': pos,
            'speed': speed,
        }

    neighbor_sets = {}
    vehicle_ids = sorted(vehicle_states.keys())
    for vehicle_id in vehicle_ids:
        neighbors = set()
        for other_id in vehicle_ids:
            if other_id == vehicle_id:
                continue
            distance = distance_xy(
                vehicle_states[vehicle_id]['position_xy'],
                vehicle_states[other_id]['position_xy'])
            if communication_range <= 0.0 or distance <= communication_range:
                neighbors.add(other_id)
        neighbor_sets[vehicle_id] = sorted(neighbors)

    return {
        'communication_range': communication_range,
        'vehicles': vehicle_states,
        'neighbor_sets': neighbor_sets,
    }


def cluster_head_unreachable(cluster, topology_state):
    vehicles = topology_state['vehicles']
    communication_range = topology_state['communication_range']
    if communication_range <= 0.0:
        return False
    head_id = int(cluster['head_id'])
    if head_id not in vehicles:
        return True
    head_xy = vehicles[head_id]['position_xy']
    for member_id in cluster['members']:
        member_id = int(member_id)
        if member_id not in vehicles:
            return True
        if distance_xy(head_xy, vehicles[member_id]['position_xy']) > \
                communication_range:
            return True
    return False


def cluster_relative_speed_risk(cluster, topology_state, speed_threshold,
                                speed_key='trigger_speed'):
    if speed_threshold is None or speed_threshold <= 0:
        return False
    vehicles = topology_state['vehicles']
    member_ids = [int(member_id) for member_id in cluster['members']
                  if int(member_id) in vehicles and
                  speed_key in vehicles[int(member_id)]]
    for index, vehicle_id in enumerate(member_ids):
        speed = vehicles[vehicle_id][speed_key]
        for other_id in member_ids[index + 1:]:
            if abs(speed - vehicles[other_id][speed_key]) >= speed_threshold:
                return True
    return False


def add_pose_delta_speeds(previous_topology, current_topology,
                          frame_interval_sec):
    if frame_interval_sec is None or frame_interval_sec <= 0:
        return
    previous_vehicles = previous_topology['vehicles']
    for vehicle_id, current_state in current_topology['vehicles'].items():
        previous_state = previous_vehicles.get(vehicle_id)
        if previous_state is None:
            continue
        distance = distance_xy(
            previous_state['position_xy'],
            current_state['position_xy'])
        current_state['pose_delta_speed'] = distance / frame_interval_sec


def replay_frame(dataset, scenario_id, timestamp, ego_cav_id, protocol,
                 resource_allocation=None, t_min_stab=None,
                 clustering='coalition_game', n_max=None, rho_th=None,
                 cav_ids=None, num_channels=None, bandwidth_mhz=None):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=ego_cav_id,
        cav_ids=cav_ids)
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=rho_th)
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
    if n_max is not None and hasattr(clustering_algorithm, 'p'):
        clustering_algorithm.p.N_max = n_max
    clusters = clustering_algorithm.run()
    apply_cluster_state(world, clusters)
    topology_state = collect_topology_state(world)
    ra_start_time = time.time()
    ra_name = get_resource_allocation_name(protocol, resource_allocation)
    resource_allocator = build_resource_allocator(ra_name, world)
    apply_resource_overrides(
        resource_allocator,
        world,
        num_channels=num_channels,
        bandwidth_mhz=bandwidth_mhz)
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
        'n_max': common.Params().N_max if n_max is None else n_max,
        'rho_th': (extract_lidar_density_threshold(protocol)
                   if rho_th is None else rho_th),
        'num_channels': (
            getattr(resource_allocator.p, 'num_channels', None)
            if hasattr(resource_allocator, 'p')
            else world.network_manager.subchannel_num),
        'bandwidth_mhz': (
            getattr(resource_allocator.p, 'bandwidth_all', 0.0) / (10 ** 6)
            if hasattr(resource_allocator, 'p') else None),
        'channel_allocation_count': len(channel_allocation),
        'channel_allocation_sample': sorted(
            channel_allocation.items())[:10],
        'clusters': cluster_summary,
        'topology_state': topology_state,
    }


def cluster_signature(cluster):
    return tuple(sorted(cluster['members']))


def vehicle_to_head(clusters):
    mapping = {}
    for cluster in clusters:
        for member_id in cluster['members']:
            mapping[member_id] = cluster['head_id']
    return mapping


def evaluate_topology_trigger(previous_result, current_result,
                              relative_speed_threshold=5.0,
                              speed_source='pose_delta',
                              frame_interval_sec=0.1):
    previous_topology = previous_result['topology_state']
    current_topology = current_result['topology_state']
    add_pose_delta_speeds(
        previous_topology,
        current_topology,
        frame_interval_sec)
    previous_vehicle_ids = set(previous_topology['vehicles'].keys())
    current_vehicle_ids = set(current_topology['vehicles'].keys())
    trigger_types = set()

    if previous_vehicle_ids != current_vehicle_ids:
        trigger_types.add('cav_set_change')
        trigger_types.add('hard_failure')

    shared_vehicle_ids = previous_vehicle_ids & current_vehicle_ids
    for vehicle_id in shared_vehicle_ids:
        previous_neighbors = set(
            previous_topology['neighbor_sets'].get(vehicle_id, []))
        current_neighbors = set(
            current_topology['neighbor_sets'].get(vehicle_id, []))
        if previous_neighbors != current_neighbors:
            trigger_types.add('neighbor_set_change')
            break

    for cluster in current_result['clusters']:
        if cluster_head_unreachable(cluster, current_topology):
            trigger_types.add('head_member_unreachable')
            trigger_types.add('hard_failure')
            break

    speed_key = 'pose_delta_speed' if speed_source == 'pose_delta' else 'speed'
    for cluster in current_result['clusters']:
        if cluster_relative_speed_risk(
                cluster,
                current_topology,
                relative_speed_threshold,
                speed_key=speed_key):
            trigger_types.add('relative_speed_risk')
            break

    previous_mapping = vehicle_to_head(previous_result['clusters'])
    current_mapping = vehicle_to_head(current_result['clusters'])
    changed_vehicle_ids = []
    for vehicle_id, head_id in current_mapping.items():
        if previous_mapping.get(vehicle_id) != head_id:
            changed_vehicle_ids.append(vehicle_id)

    return {
        'timestamp': current_result['timestamp'],
        'triggered': bool(trigger_types),
        'trigger_types': sorted(trigger_types),
        'speed_source': speed_source,
        'actual_reconfiguration': bool(changed_vehicle_ids),
        'vehicle_head_changes': len(changed_vehicle_ids),
    }


def summarize_topology_triggers(results, relative_speed_threshold=5.0,
                                speed_source='pose_delta',
                                frame_interval_sec=0.1):
    events = []
    for previous_result, current_result in zip(results, results[1:]):
        events.append(evaluate_topology_trigger(
            previous_result,
            current_result,
            relative_speed_threshold=relative_speed_threshold,
            speed_source=speed_source,
            frame_interval_sec=frame_interval_sec))

    trigger_type_counts = {}
    for event in events:
        for trigger_type in event['trigger_types']:
            trigger_type_counts[trigger_type] = (
                trigger_type_counts.get(trigger_type, 0) + 1)

    triggered_events = sum(1 for event in events if event['triggered'])
    actual_reconfiguration_events = sum(
        1 for event in events if event['actual_reconfiguration'])
    trigger_and_reconfig_events = sum(
        1 for event in events
        if event['triggered'] and event['actual_reconfiguration'])
    reconfig_without_trigger_events = sum(
        1 for event in events
        if not event['triggered'] and event['actual_reconfiguration'])
    trigger_without_reconfig_events = sum(
        1 for event in events
        if event['triggered'] and not event['actual_reconfiguration'])

    return {
        'transition_count': len(events),
        'triggered_events': triggered_events,
        'actual_reconfiguration_events': actual_reconfiguration_events,
        'trigger_and_reconfig_events': trigger_and_reconfig_events,
        'reconfig_without_trigger_events': reconfig_without_trigger_events,
        'trigger_without_reconfig_events': trigger_without_reconfig_events,
        'trigger_type_counts': trigger_type_counts,
        'events': events,
    }


def summarize_replay(results, relative_speed_threshold=5.0,
                     speed_source='pose_delta',
                     frame_interval_sec=0.1):
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
        'topology_triggers': summarize_topology_triggers(
            results,
            relative_speed_threshold=relative_speed_threshold,
            speed_source=speed_source,
            frame_interval_sec=frame_interval_sec),
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
    topology_summary = summary.get('topology_triggers', {})
    print('summary topology_triggers transitions=%s triggered=%s '
          'actual_reconfig=%s matched=%s reconfig_without_trigger=%s '
          'trigger_without_reconfig=%s' % (
              topology_summary.get('transition_count', 0),
              topology_summary.get('triggered_events', 0),
              topology_summary.get('actual_reconfiguration_events', 0),
              topology_summary.get('trigger_and_reconfig_events', 0),
              topology_summary.get('reconfig_without_trigger_events', 0),
              topology_summary.get('trigger_without_reconfig_events', 0)))
    print('summary topology_trigger_types=%s' % (
        topology_summary.get('trigger_type_counts', {}),))


def print_topology_events(summary):
    topology_summary = summary.get('topology_triggers', {})
    for index, event in enumerate(topology_summary.get('events', []),
                                  start=1):
        print('topology_event=%s timestamp=%s triggered=%s types=%s '
              'actual_reconfig=%s vehicle_head_changes=%s' % (
                  index,
                  event['timestamp'],
                  event['triggered'],
                  event['trigger_types'],
                  event['actual_reconfiguration'],
                  event['vehicle_head_changes']))


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
            clustering=args.clustering,
            n_max=args.n_max,
            rho_th=args.rho_th,
            cav_ids=select_cav_ids(
                dataset,
                sid,
                ego_cav_id=args.ego_cav_id,
                cav_count=args.cav_count,
                cav_ids=args.cav_ids),
            num_channels=args.num_channels,
            bandwidth_mhz=args.bandwidth_mhz)
        results.append(result)
        if not args.summary_only:
            print_frame_result(index, len(frames), sid, result)

    if len(results) > 1 or args.summary_only:
        summary = summarize_replay(
            results,
            relative_speed_threshold=args.trigger_relative_speed_threshold,
            speed_source=args.trigger_speed_source,
            frame_interval_sec=args.trigger_frame_interval_sec)
        print_summary(summary)
        if args.print_topology_events:
            print_topology_events(summary)


if __name__ == '__main__':
    main()
