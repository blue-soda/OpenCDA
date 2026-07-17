# -*- coding: utf-8 -*-
"""Analyze why SGCP schedulers miss specific GT object grids."""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
_OPENCOOD_ROOT = os.path.join(_REPO_ROOT, 'opencood')
if _OPENCOOD_ROOT not in sys.path:
    sys.path.insert(0, _OPENCOOD_ROOT)

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    apply_cluster_state,
    clear_sgcp_globals,
)
from opencda.core.clustering.algorithms.clustering.coalition_game import (
    CoalitionGame,
)
from opencda.core.clustering.algorithms.resource_allocation import (
    build_resource_allocator,
)
from opencda.tools.offline_inference import (
    apply_resource_overrides,
    load_protocol,
    select_cav_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rank missed-object grids inside SGCP scheduler choices.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--failure-gt-csv', required=True,
                        help='gt_objects.csv from sgcp_failure_diagnostics.')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--resource-allocation',
                        default='target_aware_potential_game')
    parser.add_argument('--rho-th', type=float, default=3.0)
    parser.add_argument('--num-channels', type=int, default=10)
    parser.add_argument('--bandwidth-mhz', type=float, default=20.0)
    parser.add_argument('--max-objects', type=int, default=8,
                        help='Analyze top-N persistent missed objects.')
    parser.add_argument('--max-rows-per-object', type=int, default=3)
    parser.add_argument('--object-ids', default=None,
                        help='Optional comma-separated object ids.')
    return parser.parse_args()


def read_failure_rows(path):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows
        if str(row.get('full_detected_method_missed', '0')) == '1'
    ]


def selected_failure_rows(rows, object_ids=None, max_objects=8,
                          max_rows_per_object=3):
    if object_ids:
        wanted = [item.strip() for item in object_ids.split(',')
                  if item.strip()]
    else:
        wanted = [
            object_id for object_id, _ in
            Counter(row['object_id'] for row in rows).most_common(max_objects)
        ]
    selected = []
    for object_id in wanted:
        object_rows = [row for row in rows if row['object_id'] == object_id]
        selected.extend(object_rows[:max_rows_per_object])
    return selected


def cluster_id_by_member(clusters):
    mapping = {}
    for cluster_index, cluster in enumerate(clusters):
        for member in cluster.members:
            mapping[int(member)] = cluster_index
    return mapping


def point_count(world, cav_id, grid_id):
    vm = world.get_vehicle_manager(int(cav_id))
    if vm is None:
        return 0
    return len(vm.perception_manager.lidar.grid_local_points.get(grid_id, []))


def density(world, cav_id, grid_id):
    vm = world.get_vehicle_manager(int(cav_id))
    if vm is None:
        return 0.0
    return vm.perception_manager.lidar.get_grid_density(grid_id)


def candidate_grids(allocator, cluster, member_id):
    if hasattr(allocator, 'refinement_candidates'):
        return set(allocator.refinement_candidates(cluster, member_id))
    base_candidates = allocator.candidate_grids_for_cluster(cluster)
    return set(allocator.member_candidate_grids(
        cluster,
        member_id,
        base_candidates,
        current_head_links=allocator.strategies.get(cluster.head_id, [])))


def sorted_candidate_grids(allocator, cluster, member_id, candidates):
    scores = {}
    for grid_id in candidates:
        scores[grid_id] = allocator.grid_score(
            cluster,
            grid_id,
            density(allocator.cav_world, member_id, grid_id))
    sorted_grids = allocator.sort_member_grids(
        cluster,
        member_id,
        candidates,
        scores,
        max(1, len(candidates)))
    if len(sorted_grids) != len(candidates):
        seen = set(sorted_grids)
        rest = sorted(
            [grid for grid in candidates if grid not in seen],
            key=lambda grid: (scores.get(grid, 0.0), str(grid)),
            reverse=True)
        sorted_grids = list(sorted_grids) + rest
    return sorted_grids, scores


def build_world(dataset, scenario_id, timestamp, args, protocol):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=args.ego_cav_id,
        cav_ids=select_cav_ids(dataset, scenario_id,
                               ego_cav_id=args.ego_cav_id))
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=args.ego_cav_id,
        protocol=protocol,
        density_threshold=args.rho_th)
    clustering = CoalitionGame(world)
    clusters = clustering.run()
    apply_cluster_state(world, clusters)
    allocator = build_resource_allocator(args.resource_allocation, world)
    apply_resource_overrides(
        allocator,
        world,
        num_channels=args.num_channels,
        bandwidth_mhz=args.bandwidth_mhz)
    allocator.set_clusters(clusters)
    allocator.run()
    return world, clusters, allocator


def analyze(args):
    dataset = OPV2VFrameDataset(args.dataset_root)
    protocol = load_protocol(dataset, args.scenario_id)
    misses = read_failure_rows(args.failure_gt_csv)
    selected = selected_failure_rows(
        misses,
        object_ids=args.object_ids,
        max_objects=args.max_objects,
        max_rows_per_object=args.max_rows_per_object)
    by_timestamp = defaultdict(list)
    for row in selected:
        by_timestamp[row['timestamp']].append(row)

    output_rows = []
    for timestamp, miss_rows in sorted(by_timestamp.items()):
        world, clusters, allocator = build_world(
            dataset,
            args.scenario_id,
            timestamp,
            args,
            protocol)
        member_cluster = cluster_id_by_member(clusters)
        cluster_by_head = {
            int(cluster.head_id): cluster for cluster in clusters
        }
        for miss in miss_rows:
            object_grid = miss['object_grid_id']
            nearest_head = int(miss['nearest_head'])
            cluster = cluster_by_head.get(nearest_head)
            if cluster is None:
                continue
            head_vm = world.get_vehicle_manager(nearest_head)
            selected_by_sender = getattr(
                head_vm.perception_manager.co_manager,
                'grid_selection',
                {}) or {}
            for member_id in sorted(int(item) for item in cluster.members):
                candidates = set()
                sorted_grids = []
                scores = {}
                rank = ''
                if member_id != nearest_head:
                    candidates = candidate_grids(allocator, cluster, member_id)
                    sorted_grids, scores = sorted_candidate_grids(
                        allocator,
                        cluster,
                        member_id,
                        candidates)
                    if object_grid in sorted_grids:
                        rank = sorted_grids.index(object_grid) + 1
                selected_grids = [
                    str(item) for item in selected_by_sender.get(member_id, [])
                ]
                selected_grid_count = len(selected_grids)
                selected_min_rank = ''
                if member_id != nearest_head and selected_grids:
                    selected_ranks = [
                        sorted_grids.index(grid) + 1
                        for grid in selected_grids
                        if grid in sorted_grids
                    ]
                    if selected_ranks:
                        selected_min_rank = max(selected_ranks)
                output_rows.append({
                    'timestamp': timestamp,
                    'object_id': miss['object_id'],
                    'bp_id': miss.get('bp_id', ''),
                    'object_grid_id': object_grid,
                    'world_x': miss.get('world_x', ''),
                    'world_y': miss.get('world_y', ''),
                    'nearest_cav': miss.get('nearest_cav', ''),
                    'nearest_head': nearest_head,
                    'cluster_index': member_cluster.get(member_id, ''),
                    'cluster_members': ';'.join(
                        str(item) for item in sorted(cluster.members)),
                    'member_id': member_id,
                    'is_head': int(member_id == nearest_head),
                    'member_point_count': point_count(
                        world,
                        member_id,
                        object_grid),
                    'member_density': '%.6f' % density(
                        world,
                        member_id,
                        object_grid),
                    'candidate': int(object_grid in candidates),
                    'rank': rank,
                    'score': (
                        '' if object_grid not in scores else
                        '%.6f' % scores[object_grid]),
                    'candidate_count': len(candidates),
                    'scheduled_to_head': int(member_id in selected_by_sender),
                    'selected_object_grid': int(object_grid in selected_grids),
                    'selected_grid_count': selected_grid_count,
                    'selected_min_rank': selected_min_rank,
                    'selected_grids_head': ';'.join(selected_grids[:20]),
                    'full_reference_best_iou': miss.get(
                        'full_reference_best_iou', ''),
                    'method_best_iou': miss.get('method_best_iou', ''),
                    'nearest_head_covering_point_count': miss.get(
                        'nearest_head_covering_point_count', ''),
                    'nearest_cav_object_grid_points': miss.get(
                        'nearest_cav_object_grid_points', ''),
                })
    fieldnames = [
        'timestamp', 'object_id', 'bp_id', 'object_grid_id',
        'world_x', 'world_y', 'nearest_cav', 'nearest_head',
        'cluster_index', 'cluster_members', 'member_id', 'is_head',
        'member_point_count', 'member_density', 'candidate', 'rank',
        'score', 'candidate_count', 'scheduled_to_head',
        'selected_object_grid', 'selected_grid_count', 'selected_min_rank',
        'selected_grids_head', 'full_reference_best_iou', 'method_best_iou',
        'nearest_head_covering_point_count',
        'nearest_cav_object_grid_points',
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)),
                exist_ok=True)
    with open(args.output_csv, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)
    print('wrote %s rows to %s' % (len(output_rows), args.output_csv))


def main():
    analyze(parse_args())


if __name__ == '__main__':
    main()
