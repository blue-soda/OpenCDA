# -*- coding: utf-8 -*-
"""
Run OpenCOOD inference from an OPV2V-style data dump.
"""

import argparse
import csv
import math
import os
import json
import random
from collections import defaultdict

import numpy as np
from omegaconf import OmegaConf
import yaml

from opencood.utils import common_utils

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    apply_cluster_state,
    build_constrained_frame,
    clear_sgcp_globals,
    select_sgcp_receiver_id,
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
from opencda.core.ml_libs.opencood_manager import OpenCOODManager


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run OpenCOOD inference from dumped OPV2V-style data.')
    parser.add_argument('--dataset-root', required=True,
                        help='Root folder containing scenario subfolders.')
    parser.add_argument('--scenario-id', default=None,
                        help='Scenario folder name. Defaults to the first one.')
    parser.add_argument('--timestamp', default=None,
                        help='Frame timestamp. Defaults to the first frame.')
    parser.add_argument('--ego-cav-id', default=None,
                        help='Ego CAV folder id. Defaults to the first CAV.')
    parser.add_argument('--fusion-method', default=None,
                        help='Override coperception fusion method.')
    parser.add_argument('--coperception-yaml', default=None,
                        help='Path to enable_coperception.yaml.')
    parser.add_argument('--max-frames', type=int, default=1,
                        help='Number of frames to test. Use 0 for all frames.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from within the scenario.')
    parser.add_argument('--sgcp-constrained', action='store_true',
                        help='Run SGCP clustering/resource allocation and evaluate the constrained uploaded frame.')
    parser.add_argument('--resource-allocation', default='potential_game',
                        help='SGCP resource allocation algorithm for constrained inference.')
    parser.add_argument('--clustering', default='coalition_game',
                        choices=['coalition_game', 'fixed_first_frame',
                                 'singleton', 'all_in_one'],
                        help='Clustering algorithm for SGCP constrained inference.')
    parser.add_argument('--sgcp-receiver-policy',
                        choices=['ego', 'ego-cluster-head',
                                 'all-cluster-heads',
                                 'all-scheduled-receivers'],
                        default='ego-cluster-head',
                        help='Receiver for constrained perception. all-cluster-heads evaluates every cluster head per frame; all-scheduled-receivers evaluates receivers that actually receive scheduled uploads.')
    parser.add_argument('--sgcp-inter-cluster-late-fusion',
                        action='store_true',
                        help='Late-fuse predictions from all SGCP cluster heads into the requested ego pose and submit one AP sample per frame.')
    parser.add_argument('--sgcp-late-nms-thresh', type=float, default=0.15,
                        help='NMS IoU threshold for SGCP inter-cluster late '
                             'fusion. Defaults to the previous value 0.15.')
    parser.add_argument('--sgcp-upload-mode', default='grid',
                        choices=['grid', 'head_only', 'full_cluster'],
                        help='Upload mode for SGCP constrained replay. grid uses PPS-selected grids; head_only keeps only each receiver; full_cluster uploads all cluster member point clouds for protocol probes.')
    parser.add_argument('--sgcp-grid-selection-mode', default='utility',
                        choices=['utility', 'random', 'spatial_diverse',
                                 'object_clustered'],
                        help='Grid selection mode for SGCP scheduled links. '
                             'random/spatial_diverse/object_clustered keep '
                             'scheduled links and grid counts but replace '
                             'grids with deterministic candidate choices.')
    parser.add_argument('--sgcp-grid-score-mode', default='utility',
                        choices=['utility', 'raw_density',
                                 'density_distance'],
                        help='Grid scoring mode used by potential_game before optional grid selection replacement.')
    parser.add_argument('--sgcp-coverage-fallback', default='none',
                        choices=['none', 'persistent',
                                 'quality_persistent'],
                        help='Optional SGCP member-coverage fallback probe. '
                             'persistent uses coverage history only; '
                             'quality_persistent additionally requires the '
                             'candidate member to have comparable historical '
                             'detector-quality proxy. Both reuse the same '
                             'subchannel and grid budget. Defaults to none.')
    parser.add_argument('--t-min-stab', type=float, default=None,
                        help='Override CoalitionGame Params.T_min_stab in seconds. Use 0 for no stability window.')
    parser.add_argument('--n-max', type=int, default=None,
                        help='Override CoalitionGame Params.N_max.')
    parser.add_argument('--rho-th', type=float, default=None,
                        help='Override lidar density_threshold / rho_th.')
    parser.add_argument('--cav-count', type=int, default=None,
                        help='Use the first N CAVs in numeric order, keeping ego included.')
    parser.add_argument('--cav-ids', default=None,
                        help='Comma-separated CAV ids to evaluate, e.g. 1,2,3.')
    parser.add_argument('--num-channels', type=int, default=None,
                        help='Override SGCP resource allocation channel count.')
    parser.add_argument('--bandwidth-mhz', type=float, default=None,
                        help='Override SGCP total bandwidth in MHz.')
    parser.add_argument('--head-rb-budget', type=int, default=None,
                        help='Override PotentialGame per-head RB budget B_h. '
                             'Defaults to 1 to preserve the original SGCP '
                             'protocol.')
    parser.add_argument('--max-upload-points-per-source', type=int,
                        default=None,
                        help='Optional deterministic point budget for each '
                             'uploaded source CAV after grid/full-cluster '
                             'selection. This keeps scheduling semantics '
                             'unchanged while probing payload/AP tradeoff.')
    parser.add_argument('--selective-sharing-baseline', default=None,
                        choices=['random', 'nearest', 'density',
                                 'greedy_density', 'communication_aware',
                                 'fullperception_rsu',
                                 'fullperception_decentralized',
                                 'edgecooper'],
                        help='Run a selective-sharing or RSU/edge-assisted baseline instead of SGCP PPS.')
    parser.add_argument('--selective-member-budget', type=int, default=2,
                        help='Maximum uploaded non-head members per receiver for selective baseline.')
    parser.add_argument('--selective-grid-budget', type=int, default=87,
                        help='Maximum selected grids per receiver for selective baseline.')
    parser.add_argument('--ns3-link-quality-csv', default=None,
                        help='Optional rlc_by_request.csv from ns3_log_eval. '
                             'When set, communication_aware selective sharing '
                             'uses per-link RLC complete ratio instead of only '
                             'a distance proxy.')
    parser.add_argument('--sgcp-trace-output', default=None,
                        help='Optional CSV path for per-receiver SGCP protocol trace.')
    parser.add_argument('--object-diagnostics-output', default=None,
                        help='Optional CSV path for per-GT object miss diagnostics. '
                             'When set, each evaluated sample is compared '
                             'against a full-20CAV reference from the same '
                             'frame.')
    parser.add_argument('--object-diagnostics-iou', type=float, default=0.5,
                        help='IoU threshold used by object diagnostics. '
                             'Defaults to 0.5.')
    return parser.parse_args()


def load_coperception_params(yaml_path, fusion_method=None):
    if yaml_path is None:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
        yaml_path = os.path.join(
            repo_root,
            'opencda/scenario_testing/config_yaml/enable_coperception.yaml')

    params = OmegaConf.load(yaml_path)['enable_coperception']['coperception']
    params = OmegaConf.to_container(params, resolve=True)
    if fusion_method is not None:
        params['fusion_method'] = fusion_method
    return params


def load_protocol(dataset, scenario_id):
    protocol_path = os.path.join(
        dataset.scenarios[scenario_id]['path'],
        'data_protocol.yaml')
    if not os.path.exists(protocol_path):
        return {}
    with open(protocol_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)


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


def extract_lidar_density_threshold(protocol):
    try:
        return float(
            protocol['vehicle_base']['sensing']['perception']['lidar'].get(
                'density_threshold', 2.0))
    except (AttributeError, KeyError, TypeError):
        return 2.0


def apply_resource_overrides(resource_allocator, world, num_channels=None,
                             bandwidth_mhz=None, head_rb_budget=None):
    if num_channels is not None:
        if num_channels <= 0:
            raise ValueError('--num-channels must be positive')
        world.network_manager.subchannel_num = int(num_channels)
        if hasattr(resource_allocator, 'lambda_subchannels'):
            resource_allocator.lambda_subchannels = int(num_channels)
    if bandwidth_mhz is not None and hasattr(resource_allocator, 'bandwidth_all'):
        if bandwidth_mhz <= 0:
            raise ValueError('--bandwidth-mhz must be positive')
        resource_allocator.bandwidth_all = float(bandwidth_mhz) * (10 ** 6)
    if hasattr(resource_allocator, 'time_slot'):
        resource_allocator.time_slot = float(
            getattr(world.network_manager, 'time_slot', 0.1))
    if not hasattr(resource_allocator, 'p'):
        return
    if num_channels is not None:
        resource_allocator.p.num_channels = int(num_channels)
    if bandwidth_mhz is not None:
        if bandwidth_mhz <= 0:
            raise ValueError('--bandwidth-mhz must be positive')
        resource_allocator.p.bandwidth_all = float(bandwidth_mhz) * (10 ** 6)
    if head_rb_budget is not None:
        if head_rb_budget <= 0:
            raise ValueError('--head-rb-budget must be positive')
        resource_allocator.p.head_rb_budget = int(head_rb_budget)
    if num_channels is not None or bandwidth_mhz is not None:
        resource_allocator.p.bandwidth_per_channel = (
            resource_allocator.p.bandwidth_all /
            resource_allocator.p.num_channels)


def candidate_grids_for_sender(head_vm, sender_vm):
    head_lidar = head_vm.perception_manager.lidar
    sender_lidar = sender_vm.perception_manager.lidar
    weak_head_grids = head_lidar.req_grids - head_lidar.high_density_grids
    candidates = sender_lidar.sens_grids & weak_head_grids
    if not candidates:
        candidates = sender_lidar.sens_grids
    return candidates


def receiver_blind_grids(head_vm):
    head_lidar = head_vm.perception_manager.lidar
    blind_grids = head_lidar.req_grids - head_lidar.high_density_grids
    if not blind_grids:
        blind_grids = head_lidar.req_grids
    return blind_grids


def randomize_scheduled_grid_selection(world, clusters, timestamp):
    """Replace selected grids while preserving SGCP scheduled links/counts."""
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = getattr(co_manager, 'grid_selection', {}) or {}
        randomized = {}
        for sender_id, grid_ids in current_selection.items():
            sender_id = int(sender_id)
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            candidates = sorted(candidate_grids_for_sender(head_vm, sender_vm))
            if not candidates:
                continue
            count = min(len(grid_ids), len(candidates))
            rng = random.Random('%s-%s-%s' % (timestamp, head_id, sender_id))
            randomized[sender_id] = rng.sample(candidates, count)
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(randomized)


def grid_center_from_id(grid_id, grid_size):
    try:
        x_idx, y_idx = [int(item) for item in str(grid_id).split('_')]
    except (TypeError, ValueError):
        return None
    return (
        x_idx * grid_size + grid_size / 2.0,
        y_idx * grid_size + grid_size / 2.0,
    )


def squared_distance(point_a, point_b):
    return (
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2)


def select_spatially_diverse_grids(head_vm, sender_vm, candidates, count):
    if count <= 0 or not candidates:
        return []
    lidar = sender_vm.perception_manager.lidar
    grid_size = lidar.grid_size
    remaining = set(candidates)
    selected = []

    def density(grid_id):
        return lidar.get_grid_density(grid_id)

    first = max(
        remaining,
        key=lambda grid_id: (density(grid_id), str(grid_id)))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < count:
        selected_centers = [
            grid_center_from_id(grid_id, grid_size)
            for grid_id in selected
        ]
        selected_centers = [
            center for center in selected_centers if center is not None
        ]

        def diversity_score(grid_id):
            center = grid_center_from_id(grid_id, grid_size)
            if center is None or not selected_centers:
                min_distance = 0.0
            else:
                min_distance = min(
                    squared_distance(center, selected_center)
                    for selected_center in selected_centers)
            return (density(grid_id) + 1e-6) * (1.0 + min_distance / 10000.0)

        best_grid = max(
            remaining,
            key=lambda grid_id: (diversity_score(grid_id), str(grid_id)))
        selected.append(best_grid)
        remaining.remove(best_grid)
    return selected


def grid_index_from_id(grid_id):
    try:
        return tuple(int(item) for item in str(grid_id).split('_'))
    except (TypeError, ValueError):
        return None


def grid_l1_distance(grid_a, grid_b):
    index_a = grid_index_from_id(grid_a)
    index_b = grid_index_from_id(grid_b)
    if index_a is None or index_b is None:
        return 999999
    return abs(index_a[0] - index_b[0]) + abs(index_a[1] - index_b[1])


def select_object_clustered_grids(head_vm, sender_vm, candidates, count):
    """Select compact high-density grid patches as object-level proxies."""
    if count <= 0 or not candidates:
        return []
    lidar = sender_vm.perception_manager.lidar
    remaining = set(candidates)
    selected = []

    def density(grid_id):
        return lidar.get_grid_density(grid_id)

    while remaining and len(selected) < count:
        if not selected:
            best_grid = max(
                remaining,
                key=lambda grid_id: (density(grid_id), str(grid_id)))
        else:
            best_grid = max(
                remaining,
                key=lambda grid_id: (
                    density(grid_id) /
                    (1.0 + min(grid_l1_distance(grid_id, selected_grid)
                               for selected_grid in selected)),
                    density(grid_id),
                    str(grid_id)))
        selected.append(best_grid)
        remaining.remove(best_grid)
    return selected


def diversify_scheduled_grid_selection(world, clusters):
    """Replace selected grids with deterministic density-aware spatial cover."""
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = getattr(co_manager, 'grid_selection', {}) or {}
        diversified = {}
        for sender_id, grid_ids in current_selection.items():
            sender_id = int(sender_id)
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            candidates = sorted(candidate_grids_for_sender(head_vm, sender_vm))
            if not candidates:
                continue
            count = min(len(grid_ids), len(candidates))
            diversified[sender_id] = select_spatially_diverse_grids(
                head_vm,
                sender_vm,
                candidates,
                count)
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(diversified)


def cluster_scheduled_grid_selection(world, clusters):
    """Replace selected grids with compact high-density object proxies."""
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = getattr(co_manager, 'grid_selection', {}) or {}
        clustered = {}
        for sender_id, grid_ids in current_selection.items():
            sender_id = int(sender_id)
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            candidates = sorted(candidate_grids_for_sender(head_vm, sender_vm))
            if not candidates:
                continue
            count = min(len(grid_ids), len(candidates))
            clustered[sender_id] = select_object_clustered_grids(
                head_vm,
                sender_vm,
                candidates,
                count)
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(clustered)


def quality_ratio(stat):
    gt_sum = float(stat.get('quality_gt_sum', 0))
    if gt_sum <= 0:
        return None
    return float(stat.get('quality_pred_sum', 0)) / gt_sum


def apply_persistent_coverage_fallback(world, clusters, coverage_state,
                                       quality_aware=False):
    """Swap in repeatedly unscheduled members without changing link count.

    This is a diagnostic/algorithm probe for the 10ch case: global channel
    capacity is already saturated, so a coverage repair must reuse an existing
    subchannel rather than adding another request.
    """
    if coverage_state is None:
        return 0
    replacements = 0
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = {
            int(sender_id): list(grid_ids)
            for sender_id, grid_ids in (
                getattr(co_manager, 'grid_selection', {}) or {}).items()
        }
        if not current_selection:
            continue
        scheduler = head_vm.v2x_manager.scheduler
        channel_allocation = getattr(scheduler, 'channel_allocation', {})
        non_head_members = [
            int(member_id) for member_id in sorted(cluster.members)
            if int(member_id) != head_id
        ]
        unscheduled = [
            member_id for member_id in non_head_members
            if member_id not in current_selection
        ]
        if not unscheduled:
            continue

        def member_deficit(member_id):
            stat = coverage_state.get(member_id, {})
            return (
                int(stat.get('unscheduled_frames', 0)) -
                int(stat.get('uploaded_frames', 0)),
                int(stat.get('unscheduled_frames', 0)),
                -int(stat.get('uploaded_frames', 0)),
                -member_id,
            )

        candidate_id = max(unscheduled, key=member_deficit)
        candidate_deficit = member_deficit(candidate_id)
        if candidate_deficit[0] < 2:
            continue

        def scheduled_score(member_id):
            stat = coverage_state.get(member_id, {})
            grid_count = len(current_selection.get(member_id, []))
            return (
                int(stat.get('uploaded_frames', 0)) -
                int(stat.get('unscheduled_frames', 0)),
                grid_count,
                -member_id,
            )

        replaced_id = max(current_selection.keys(), key=scheduled_score)
        if member_deficit(replaced_id)[0] >= candidate_deficit[0]:
            continue
        if quality_aware:
            candidate_stat = coverage_state.get(candidate_id, {})
            replaced_stat = coverage_state.get(replaced_id, {})
            candidate_quality = quality_ratio(candidate_stat)
            replaced_quality = quality_ratio(replaced_stat)
            if (candidate_quality is None or
                    int(candidate_stat.get('quality_rows', 0)) < 2):
                continue
            if (replaced_quality is not None and
                    candidate_quality < 0.9 * replaced_quality):
                continue
            if candidate_quality < 0.25:
                continue
        replaced_grids = current_selection.get(replaced_id, [])
        if not replaced_grids:
            continue
        candidate_vm = world.get_vehicle_manager(candidate_id)
        if candidate_vm is None:
            continue
        candidates = sorted(candidate_grids_for_sender(head_vm, candidate_vm))
        if not candidates:
            continue
        grid_count = min(len(replaced_grids), len(candidates))
        new_grids = select_spatially_diverse_grids(
            head_vm,
            candidate_vm,
            candidates,
            grid_count)
        if not new_grids:
            continue
        replaced_vm = world.get_vehicle_manager(replaced_id)
        replaced_density = 0.0
        if replaced_vm is not None:
            replaced_density = sum(
                replaced_vm.perception_manager.lidar.get_grid_density(grid_id)
                for grid_id in replaced_grids)
        candidate_density = sum(
            candidate_vm.perception_manager.lidar.get_grid_density(grid_id)
            for grid_id in new_grids)
        if (replaced_density > 0 and
                candidate_density < 0.8 * replaced_density):
            continue
        old_channel = channel_allocation.pop((replaced_id, head_id), None)
        if old_channel is None:
            continue
        channel_allocation[(candidate_id, head_id)] = old_channel
        current_selection.pop(replaced_id, None)
        current_selection[candidate_id] = new_grids
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(current_selection)
        replacements += 1
    return replacements


def update_coverage_state_from_items(coverage_state, constrained_items):
    if coverage_state is None:
        return
    for _, metadata in constrained_items:
        receiver_id = int(metadata.get('receiver_id'))
        members = set(int(item) for item in metadata.get(
            'cluster_member_ids', []))
        sources = set(int(item) for item in metadata.get(
            'source_cav_ids', []))
        uploaded = sources - {receiver_id}
        for member_id in members:
            stat = coverage_state.setdefault(member_id, {
                'uploaded_frames': 0,
                'unscheduled_frames': 0,
                'fused_frames': 0,
                'quality_rows': 0,
                'quality_pred_sum': 0,
                'quality_gt_sum': 0,
            })
            if member_id in sources:
                stat['fused_frames'] += 1
            if member_id in uploaded:
                stat['uploaded_frames'] += 1
            elif member_id != receiver_id:
                stat['unscheduled_frames'] += 1


def update_coverage_quality_state(coverage_state, metadata, pred_count,
                                  gt_count):
    if coverage_state is None or metadata is None:
        return
    if gt_count is None or gt_count <= 0:
        return
    receiver_id = int(metadata.get('receiver_id'))
    source_ids = set(int(item) for item in metadata.get(
        'source_cav_ids', []))
    uploaded_ids = source_ids - {receiver_id}
    for source_id in uploaded_ids:
        stat = coverage_state.setdefault(source_id, {
            'uploaded_frames': 0,
            'unscheduled_frames': 0,
            'fused_frames': 0,
            'quality_rows': 0,
            'quality_pred_sum': 0,
            'quality_gt_sum': 0,
        })
        stat['quality_rows'] += 1
        stat['quality_pred_sum'] += int(pred_count or 0)
        stat['quality_gt_sum'] += int(gt_count or 0)


def cluster_templates_from_clusters(clusters):
    templates = []
    for cluster in clusters:
        templates.append({
            'head_id': int(cluster.head_id),
            'member_ids': sorted(int(member_id)
                                 for member_id in cluster.members),
        })
    return templates


def build_fixed_clusters(world, cluster_templates):
    common.Vehicle_Grid.initialize(world)
    clusters = []
    for template in cluster_templates:
        member_ids = [
            int(member_id)
            for member_id in template['member_ids']
            if world.get_vehicle_manager(int(member_id)) is not None
        ]
        if not member_ids:
            continue
        cluster = common.Cluster(set(member_ids))
        fixed_head = int(template['head_id'])
        if world.get_vehicle_manager(fixed_head) is not None:
            cluster.head_id = fixed_head
            cluster.grid_bits = cluster.compute_grid_bits()
        clusters.append(cluster)
    return clusters


def scheduled_receiver_ids(world, fallback_clusters=None):
    receiver_ids = set()
    for vm in world.get_vehicle_managers().values():
        scheduler = getattr(vm.v2x_manager, 'scheduler', None)
        channel_allocation = getattr(scheduler, 'channel_allocation', {})
        for link in channel_allocation.keys():
            try:
                _, target_id = link
            except (TypeError, ValueError):
                continue
            receiver_ids.add(int(target_id))
    if not receiver_ids and fallback_clusters:
        receiver_ids.update(int(cluster.head_id) for cluster in fallback_clusters)
    return sorted(receiver_ids)


def apply_sgcp_constraint(frame, protocol, ego_cav_id, resource_allocation,
                          receiver_policy, t_min_stab=None,
                          clustering='coalition_game', n_max=None,
                          rho_th=None, num_channels=None,
                          bandwidth_mhz=None, upload_mode='grid',
                          grid_selection_mode='utility',
                          grid_score_mode='utility',
                          timestamp=None,
                          fixed_cluster_templates=None,
                          head_rb_budget=None,
                          coverage_fallback='none',
                          coverage_state=None,
                          max_upload_points_per_source=None):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=rho_th)
    if clustering in ['coalition_game', 'fixed_first_frame']:
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
    if clustering == 'fixed_first_frame' and fixed_cluster_templates:
        clusters = build_fixed_clusters(world, fixed_cluster_templates)
    else:
        clusters = clustering_algorithm.run()
        if (clustering == 'fixed_first_frame' and
                fixed_cluster_templates is not None and
                not fixed_cluster_templates):
            fixed_cluster_templates.extend(
                cluster_templates_from_clusters(clusters))
    apply_cluster_state(world, clusters)
    allocator = build_resource_allocator(resource_allocation, world)
    if hasattr(allocator, 'grid_score_mode'):
        allocator.grid_score_mode = grid_score_mode
    apply_resource_overrides(
        allocator,
        world,
        num_channels=num_channels,
        bandwidth_mhz=bandwidth_mhz,
        head_rb_budget=head_rb_budget)
    allocator.set_clusters(clusters)
    allocator.run()
    if grid_selection_mode == 'random':
        randomize_scheduled_grid_selection(world, clusters, timestamp)
    elif grid_selection_mode == 'spatial_diverse':
        diversify_scheduled_grid_selection(world, clusters)
    elif grid_selection_mode == 'object_clustered':
        cluster_scheduled_grid_selection(world, clusters)
    coverage_fallback_replacements = 0
    if coverage_fallback in ['persistent', 'quality_persistent']:
        coverage_fallback_replacements = apply_persistent_coverage_fallback(
            world,
            clusters,
            coverage_state,
            quality_aware=coverage_fallback == 'quality_persistent')
    if receiver_policy == 'all-cluster-heads':
        receiver_ids = sorted(int(cluster.head_id) for cluster in clusters)
    elif receiver_policy == 'all-scheduled-receivers':
        receiver_ids = scheduled_receiver_ids(world, clusters)
    else:
        receiver_ids = [select_sgcp_receiver_id(
            world,
            ego_cav_id=ego_cav_id,
            receiver_policy=receiver_policy)]

    constrained_items = []
    for receiver_id in receiver_ids:
        constrained_frame, metadata = build_constrained_frame(
            frame,
            world,
            receiver_id,
            upload_mode=upload_mode,
            max_upload_points_per_source=max_upload_points_per_source)
        metadata['cluster_count'] = len(clusters)
        metadata['resource_allocation'] = resource_allocation
        metadata['clustering'] = clustering
        metadata['receiver_policy'] = receiver_policy
        metadata['t_min_stab'] = (
            common.Params().T_min_stab if t_min_stab is None else t_min_stab)
        metadata['n_max'] = (
            common.Params().N_max if n_max is None else n_max)
        metadata['rho_th'] = (
            extract_lidar_density_threshold(protocol)
            if rho_th is None else rho_th)
        metadata['num_channels'] = (
            getattr(allocator.p, 'num_channels', None)
            if hasattr(allocator, 'p')
            else world.network_manager.subchannel_num)
        metadata['head_rb_budget'] = (
            getattr(allocator.p, 'head_rb_budget', None)
            if hasattr(allocator, 'p') else None)
        metadata['bandwidth_mhz'] = (
            getattr(allocator.p, 'bandwidth_all', 0.0) / (10 ** 6)
            if hasattr(allocator, 'p') else None)
        metadata['upload_mode'] = upload_mode
        metadata['grid_selection_mode'] = grid_selection_mode
        metadata['grid_score_mode'] = grid_score_mode
        metadata['coverage_fallback'] = coverage_fallback
        metadata['coverage_fallback_replacements'] = (
            coverage_fallback_replacements)
        metadata['max_upload_points_per_source'] = (
            max_upload_points_per_source or '')
        constrained_items.append((constrained_frame, metadata))
    update_coverage_state_from_items(coverage_state, constrained_items)
    return constrained_items


def vehicle_distance(vm_a, vm_b):
    pos_a = vm_a.v2x_manager.get_ego_pos().location
    pos_b = vm_b.v2x_manager.get_ego_pos().location
    return math.sqrt(
        (pos_a.x - pos_b.x) ** 2 +
        (pos_a.y - pos_b.y) ** 2)


def load_ns3_link_quality(path):
    if not path:
        return None
    exact = defaultdict(list)
    pair = defaultdict(list)
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                source = int(row['source_node'])
                target = int(row['target_node'])
            except (KeyError, TypeError, ValueError):
                continue
            timestamp = str(row.get('timestamp', ''))
            complete = int(float(row.get('rlc_complete', 0) or 0))
            exact[(timestamp, source, target)].append(complete)
            pair[(source, target)].append(complete)
    return {
        'exact': {
            key: sum(values) / float(len(values))
            for key, values in exact.items()
        },
        'pair': {
            key: sum(values) / float(len(values))
            for key, values in pair.items()
        },
        'path': path,
    }


def ns3_link_quality(link_quality, timestamp, source_id, target_id):
    if not link_quality:
        return None
    exact_key = (str(timestamp), int(source_id), int(target_id))
    if exact_key in link_quality['exact']:
        return link_quality['exact'][exact_key]
    return link_quality['pair'].get((int(source_id), int(target_id)))


def candidate_member_ids(world, cluster, baseline_name):
    head_id = int(cluster.head_id)
    if baseline_name in ['fullperception_rsu', 'edgecooper']:
        return [
            int(member_id)
            for member_id in sorted(world.get_vehicle_managers().keys())
            if int(member_id) != head_id
        ]
    return [
        int(member_id) for member_id in sorted(cluster.members)
        if int(member_id) != head_id
    ]


def density_score_for_member(head_vm, sender_vm):
    candidate_grids = candidate_grids_for_sender(head_vm, sender_vm)
    return sum(
        sender_vm.perception_manager.lidar.get_grid_density(grid_id)
        for grid_id in candidate_grids)


def edgecooper_candidate_grids(head_vm, sender_vm):
    sender_lidar = sender_vm.perception_manager.lidar
    return sender_lidar.sens_grids & receiver_blind_grids(head_vm)


def edgecooper_grid_score(head_vm, sender_vm, grid_id, covered_grids=None):
    sender_lidar = sender_vm.perception_manager.lidar
    head_lidar = head_vm.perception_manager.lidar
    blind_grids = receiver_blind_grids(head_vm)
    sender_density = sender_lidar.get_grid_density(grid_id)
    head_density = head_lidar.get_grid_density(grid_id)
    blind_bonus = 1.0 if grid_id in blind_grids else 0.25
    novelty_bonus = 1.0
    if covered_grids is not None and grid_id in covered_grids:
        novelty_bonus = 0.35
    redundancy_penalty = min(sender_density, head_density)
    return (
        sender_density * blind_bonus * novelty_bonus -
        0.25 * redundancy_penalty)


def select_edgecooper_members(world, head_vm, members, member_budget):
    selected = []
    covered = set()
    remaining = set(members)
    while remaining and len(selected) < member_budget:
        best = None
        for member_id in sorted(remaining):
            sender_vm = world.get_vehicle_manager(member_id)
            candidate_grids = set(edgecooper_candidate_grids(
                head_vm,
                sender_vm))
            if not candidate_grids:
                continue
            complementarity = sum(
                max(0.0, edgecooper_grid_score(
                    head_vm,
                    sender_vm,
                    grid_id,
                    covered_grids=covered))
                for grid_id in candidate_grids)
            redundancy = sum(
                sender_vm.perception_manager.lidar.get_grid_density(grid_id)
                for grid_id in candidate_grids & covered)
            distance = vehicle_distance(head_vm, sender_vm)
            score = (
                complementarity / (1.0 + distance / 50.0) -
                0.35 * redundancy)
            item = (-score, distance, member_id, candidate_grids)
            if best is None or item < best:
                best = item
        if best is None:
            break
        _, _, member_id, candidate_grids = best
        selected.append(member_id)
        covered.update(candidate_grids)
        remaining.remove(member_id)
    return selected


def select_edgecooper_grids(head_vm, sender_vm, candidates, count,
                            covered_grids=None):
    if count <= 0 or not candidates:
        return []
    covered_grids = set() if covered_grids is None else set(covered_grids)
    remaining = set(candidates)
    selected = []
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda grid_id: (
                edgecooper_grid_score(
                    head_vm,
                    sender_vm,
                    grid_id,
                    covered_grids=covered_grids | set(selected)),
                sender_vm.perception_manager.lidar.get_grid_density(grid_id),
                str(grid_id)))
        selected.append(best)
        remaining.remove(best)
    return selected


def select_baseline_members(world, cluster, baseline_name, member_budget,
                            link_quality=None, timestamp=None):
    head_id = int(cluster.head_id)
    head_vm = world.get_vehicle_manager(head_id)
    members = candidate_member_ids(world, cluster, baseline_name)
    if member_budget <= 0 or not members:
        return []

    if baseline_name == 'random':
        rng = random.Random('%s_%s_%s' % (timestamp, head_id, member_budget))
        shuffled = list(members)
        rng.shuffle(shuffled)
        return shuffled[:member_budget]

    if baseline_name == 'nearest':
        scored = [
            (vehicle_distance(head_vm, world.get_vehicle_manager(member_id)),
             member_id)
            for member_id in members
        ]
        return [member_id for _, member_id in sorted(scored)[:member_budget]]

    if baseline_name == 'edgecooper':
        return select_edgecooper_members(
            world,
            head_vm,
            members,
            member_budget)

    if baseline_name in ['density', 'greedy_density', 'communication_aware',
                         'fullperception_rsu',
                         'fullperception_decentralized']:
        scored = []
        for member_id in members:
            sender_vm = world.get_vehicle_manager(member_id)
            density_sum = density_score_for_member(head_vm, sender_vm)
            if baseline_name in ['communication_aware',
                                 'fullperception_decentralized']:
                distance = vehicle_distance(head_vm, sender_vm)
                quality = ns3_link_quality(
                    link_quality,
                    timestamp,
                    member_id,
                    head_id)
                if quality is None:
                    density_sum = density_sum / (1.0 + distance / 100.0)
                else:
                    density_sum = (
                        density_sum * quality / (1.0 + distance / 100.0))
            elif baseline_name == 'fullperception_rsu':
                distance = vehicle_distance(head_vm, sender_vm)
                density_sum = density_sum / (1.0 + distance / 200.0)
            scored.append((-density_sum, member_id))
        return [member_id for _, member_id in sorted(scored)[:member_budget]]

    raise ValueError('Unknown selective baseline: %s' % baseline_name)


def assign_selective_grid_selection(world, cluster, baseline_name,
                                    member_budget, grid_budget,
                                    link_quality=None, timestamp=None):
    head_id = int(cluster.head_id)
    head_vm = world.get_vehicle_manager(head_id)
    selected_members = select_baseline_members(
        world,
        cluster,
        baseline_name,
        member_budget,
        link_quality=link_quality,
        timestamp=timestamp)
    if grid_budget <= 0 or not selected_members:
        return

    per_member_budget = max(
        1,
        int(math.ceil(grid_budget / float(len(selected_members)))))
    grid_selection = {}
    remaining = int(grid_budget)
    covered_edge_grids = set()
    for member_id in selected_members:
        if remaining <= 0:
            break
        sender_vm = world.get_vehicle_manager(member_id)
        if baseline_name == 'edgecooper':
            candidate_grids = edgecooper_candidate_grids(head_vm, sender_vm)
        else:
            candidate_grids = candidate_grids_for_sender(head_vm, sender_vm)
        if baseline_name == 'random':
            grids = list(candidate_grids)
            rng = random.Random('%s_%s_%s_%s' % (
                timestamp,
                head_id,
                member_id,
                grid_budget))
            rng.shuffle(grids)
        elif baseline_name == 'edgecooper':
            grids = select_edgecooper_grids(
                head_vm,
                sender_vm,
                candidate_grids,
                min(per_member_budget, remaining),
                covered_grids=covered_edge_grids)
        else:
            grids = sorted(
                candidate_grids,
                key=lambda grid_id: sender_vm.perception_manager.lidar.
                get_grid_density(grid_id),
                reverse=True)
        selected = grids[:min(per_member_budget, remaining)]
        if selected:
            grid_selection[member_id] = selected
            if baseline_name == 'edgecooper':
                covered_edge_grids.update(selected)
            remaining -= len(selected)
    head_vm.perception_manager.co_manager.set_grid_selection(grid_selection)


def apply_selective_sharing_baseline(frame, protocol, ego_cav_id,
                                     baseline_name, receiver_policy,
                                     member_budget, grid_budget,
                                     t_min_stab=None, clustering='coalition_game',
                                     n_max=None, rho_th=None,
                                     link_quality=None, timestamp=None,
                                     max_upload_points_per_source=None):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=rho_th)
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
    for cluster in clusters:
        assign_selective_grid_selection(
            world,
            cluster,
            baseline_name,
            member_budget,
            grid_budget,
            link_quality=link_quality,
            timestamp=timestamp)

    if receiver_policy == 'all-cluster-heads':
        receiver_ids = sorted(int(cluster.head_id) for cluster in clusters)
    else:
        receiver_ids = [select_sgcp_receiver_id(
            world,
            ego_cav_id=ego_cav_id,
            receiver_policy=receiver_policy)]

    constrained_items = []
    for receiver_id in receiver_ids:
        constrained_frame, metadata = build_constrained_frame(
            frame,
            world,
            receiver_id,
            max_upload_points_per_source=max_upload_points_per_source)
        metadata['cluster_count'] = len(clusters)
        metadata['resource_allocation'] = (
            'selective_%s' % baseline_name)
        metadata['clustering'] = clustering
        metadata['receiver_policy'] = receiver_policy
        metadata['t_min_stab'] = (
            common.Params().T_min_stab if t_min_stab is None else t_min_stab)
        metadata['n_max'] = (
            common.Params().N_max if n_max is None else n_max)
        metadata['rho_th'] = (
            extract_lidar_density_threshold(protocol)
            if rho_th is None else rho_th)
        metadata['selective_member_budget'] = member_budget
        metadata['selective_grid_budget'] = grid_budget
        metadata['ns3_link_quality_csv'] = (
            link_quality['path'] if link_quality else '')
        metadata['max_upload_points_per_source'] = (
            max_upload_points_per_source or '')
        constrained_items.append((constrained_frame, metadata))
    return constrained_items


def run_opencood_inference(manager, frame, ego_lidar_pose):
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        ego_lidar_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    batch_data = manager.to_device(output_dict)
    return manager.inference(
        batch_data,
        with_stats=False,
        return_object_ids=manager.fusion_method != 'late')


def is_empty_pillar_error(error):
    return (
        isinstance(error, RuntimeError) and
        'input.numel() == 0' in str(error))


def format_channel_allocation(channel_allocation):
    items = []
    for link, subchannel in sorted(channel_allocation.items()):
        try:
            source_id, target_id = link
            items.append('%s>%s:%s' % (source_id, target_id, subchannel))
        except (TypeError, ValueError):
            items.append('%s:%s' % (link, subchannel))
    return ';'.join(items)


def trace_row(scenario_id, timestamp, metadata, eval_frame,
              pred_count=None, gt_count=None, skipped=''):
    source_ids = [int(cav_id) for cav_id in metadata.get('source_cav_ids', [])]
    receiver_id = int(metadata.get('receiver_id'))
    selected_grid_counts = {
        int(key): int(value)
        for key, value in metadata.get('selected_grid_counts', {}).items()
    }
    channel_allocation = metadata.get('channel_allocation', {}) or {}
    scheduled_sources = set()
    for link in channel_allocation.keys():
        try:
            source_id, target_id = link
        except (TypeError, ValueError):
            continue
        if int(target_id) == receiver_id:
            scheduled_sources.add(int(source_id))
    uploaded_sources = [source_id for source_id in source_ids
                        if source_id != receiver_id]
    missing_channel_sources = [
        source_id for source_id in uploaded_sources
        if source_id not in scheduled_sources
    ]
    point_counts = {
        int(cav_id): int(cav['lidar_np'].shape[0])
        for cav_id, cav in eval_frame.items()
    }
    return {
        'scenario_id': scenario_id,
        'timestamp': timestamp,
        'receiver_id': receiver_id,
        'receiver_policy': metadata.get('receiver_policy', ''),
        'resource_allocation': metadata.get('resource_allocation', ''),
        'upload_mode': metadata.get('upload_mode', ''),
        'grid_selection_mode': metadata.get('grid_selection_mode', ''),
        'grid_score_mode': metadata.get('grid_score_mode', ''),
        'coverage_fallback': metadata.get('coverage_fallback', ''),
        'coverage_fallback_replacements': metadata.get(
            'coverage_fallback_replacements', ''),
        'clustering': metadata.get('clustering', ''),
        'cluster_count': metadata.get('cluster_count', ''),
        'cluster_member_ids': ';'.join(
            str(item) for item in metadata.get('cluster_member_ids', [])),
        'source_cav_ids': ';'.join(str(item) for item in source_ids),
        'uploaded_source_ids': ';'.join(str(item) for item in uploaded_sources),
        'selected_grid_counts_json': json.dumps(
            selected_grid_counts, sort_keys=True),
        'point_counts_json': json.dumps(point_counts, sort_keys=True),
        'communication_bytes': metadata.get('communication_bytes', 0),
        'channel_allocation': format_channel_allocation(channel_allocation),
        'missing_channel_sources': ';'.join(
            str(item) for item in missing_channel_sources),
        'pred_boxes': '' if pred_count is None else pred_count,
        'gt_boxes': '' if gt_count is None else gt_count,
        'skipped': skipped,
    }


def write_trace_csv(path, rows):
    if not path or not rows:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fieldnames = [
        'scenario_id',
        'timestamp',
        'receiver_id',
        'receiver_policy',
        'resource_allocation',
        'upload_mode',
        'grid_selection_mode',
        'grid_score_mode',
        'coverage_fallback',
        'coverage_fallback_replacements',
        'clustering',
        'cluster_count',
        'cluster_member_ids',
        'source_cav_ids',
        'uploaded_source_ids',
        'selected_grid_counts_json',
        'point_counts_json',
        'communication_bytes',
        'channel_allocation',
        'missing_channel_sources',
        'pred_boxes',
        'gt_boxes',
        'skipped',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tensor_to_numpy(value):
    if value is None:
        return None
    return common_utils.torch_tensor_to_numpy(value)


def normalize_object_ids(object_ids, count):
    if object_ids is None:
        return [''] * count
    normalized = [str(item) for item in object_ids]
    if len(normalized) < count:
        normalized.extend([''] * (count - len(normalized)))
    return normalized[:count]


def box_center(box):
    if box is None or len(box) == 0:
        return ('', '', '')
    center = np.mean(box[:, :3], axis=0)
    return ('%.4f' % center[0], '%.4f' % center[1], '%.4f' % center[2])


def match_predictions_to_gt(gt_boxes, pred_boxes, pred_scores, iou_thresh):
    """Return best and greedy matched prediction info for each canonical GT."""
    if gt_boxes is None:
        return []
    gt_boxes_np = tensor_to_numpy(gt_boxes)
    gt_count = gt_boxes_np.shape[0]
    if gt_count == 0:
        return []
    matches = [
        {
            'matched': False,
            'best_iou': 0.0,
            'best_score': '',
            'matched_score': '',
        }
        for _ in range(gt_count)
    ]
    if pred_boxes is None or pred_scores is None:
        return matches

    pred_boxes_np = tensor_to_numpy(pred_boxes)
    pred_scores_np = tensor_to_numpy(pred_scores)
    if pred_boxes_np is None or pred_scores_np is None:
        return matches
    if pred_boxes_np.shape[0] == 0:
        return matches

    pred_polygons = list(common_utils.convert_format(pred_boxes_np))
    gt_polygons = list(common_utils.convert_format(gt_boxes_np))
    score_order = np.argsort(-pred_scores_np)
    unmatched_gt = set(range(gt_count))

    for pred_index in score_order:
        if not unmatched_gt:
            break
        pred_polygon = pred_polygons[pred_index]
        ious = common_utils.compute_iou(pred_polygon, gt_polygons)
        if len(ious) == 0:
            continue
        for gt_index, iou in enumerate(ious):
            if iou > matches[gt_index]['best_iou']:
                matches[gt_index]['best_iou'] = float(iou)
                matches[gt_index]['best_score'] = (
                    '%.6f' % float(pred_scores_np[pred_index]))
        candidate_gt = max(unmatched_gt, key=lambda idx: ious[idx])
        candidate_iou = float(ious[candidate_gt])
        if candidate_iou < iou_thresh:
            continue
        matches[candidate_gt]['matched'] = True
        matches[candidate_gt]['matched_score'] = (
            '%.6f' % float(pred_scores_np[pred_index]))
        unmatched_gt.remove(candidate_gt)
    return matches


def object_diagnostic_rows(scenario_id, timestamp, sample_label, metadata,
                           canonical_gt_boxes, canonical_gt_ids,
                           reference_pred_boxes, reference_scores,
                           method_pred_boxes, method_scores, iou_thresh):
    if canonical_gt_boxes is None:
        return []
    gt_boxes_np = tensor_to_numpy(canonical_gt_boxes)
    if gt_boxes_np is None:
        return []
    gt_ids = normalize_object_ids(canonical_gt_ids, gt_boxes_np.shape[0])
    reference_matches = match_predictions_to_gt(
        canonical_gt_boxes,
        reference_pred_boxes,
        reference_scores,
        iou_thresh)
    method_matches = match_predictions_to_gt(
        canonical_gt_boxes,
        method_pred_boxes,
        method_scores,
        iou_thresh)
    rows = []
    metadata = metadata or {}
    for gt_index, gt_box in enumerate(gt_boxes_np):
        center_x, center_y, center_z = box_center(gt_box)
        ref_match = reference_matches[gt_index]
        method_match = method_matches[gt_index]
        rows.append({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'sample_label': sample_label,
            'receiver_id': metadata.get('receiver_id', ''),
            'resource_allocation': metadata.get('resource_allocation', ''),
            'clustering': metadata.get('clustering', ''),
            'upload_mode': metadata.get('upload_mode', ''),
            'grid_selection_mode': metadata.get('grid_selection_mode', ''),
            'grid_score_mode': metadata.get('grid_score_mode', ''),
            'num_channels': metadata.get('num_channels', ''),
            'bandwidth_mhz': metadata.get('bandwidth_mhz', ''),
            'communication_bytes': metadata.get('communication_bytes', ''),
            'source_cav_ids': ';'.join(
                str(item) for item in metadata.get('source_cav_ids', [])),
            'uploaded_source_ids': ';'.join(
                str(item) for item in metadata.get(
                    'source_cav_ids', [])[1:]),
            'selected_grid_counts_json': json.dumps(
                metadata.get('selected_grid_counts', {}),
                sort_keys=True),
            'gt_index': gt_index,
            'gt_object_id': gt_ids[gt_index],
            'gt_center_x': center_x,
            'gt_center_y': center_y,
            'gt_center_z': center_z,
            'full_reference_matched': int(ref_match['matched']),
            'full_reference_best_iou': '%.6f' % ref_match['best_iou'],
            'full_reference_best_score': ref_match['best_score'],
            'method_matched': int(method_match['matched']),
            'method_best_iou': '%.6f' % method_match['best_iou'],
            'method_best_score': method_match['best_score'],
            'full_detected_method_missed': int(
                ref_match['matched'] and not method_match['matched']),
        })
    return rows


def write_object_diagnostics_csv(path, rows):
    if not path or not rows:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fieldnames = [
        'scenario_id',
        'timestamp',
        'sample_label',
        'receiver_id',
        'resource_allocation',
        'clustering',
        'upload_mode',
        'grid_selection_mode',
        'grid_score_mode',
        'num_channels',
        'bandwidth_mhz',
        'communication_bytes',
        'source_cav_ids',
        'uploaded_source_ids',
        'selected_grid_counts_json',
        'gt_index',
        'gt_object_id',
        'gt_center_x',
        'gt_center_y',
        'gt_center_z',
        'full_reference_matched',
        'full_reference_best_iou',
        'full_reference_best_score',
        'method_matched',
        'method_best_iou',
        'method_best_score',
        'full_detected_method_missed',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)

    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    sgcp_summaries = []
    sgcp_trace_rows = []
    object_rows = []
    ns3_link_quality = load_ns3_link_quality(args.ns3_link_quality_csv)
    fixed_cluster_templates = []
    sgcp_coverage_state = defaultdict(dict)

    if args.timestamp is not None:
        if args.scenario_id is None:
            scenario_id = next(iter(dataset.scenarios.keys()))
        else:
            scenario_id = args.scenario_id
        frames = [(scenario_id, args.timestamp)]
    else:
        if args.scenario_id is None:
            scenario_id = next(iter(dataset.scenarios.keys()))
        else:
            scenario_id = args.scenario_id
        timestamps = dataset.scenarios[scenario_id]['timestamps']
        if args.max_frames == 0:
            selected = timestamps[args.start_index:]
        else:
            selected = timestamps[args.start_index:
                                  args.start_index + args.max_frames]
        frames = [(scenario_id, timestamp) for timestamp in selected]

    for index, (scenario_id, timestamp) in enumerate(frames, start=1):
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
        original_ego = next(cav for cav in frame.values() if cav['ego'])
        target_ego_lidar_pose = original_ego['params']['lidar_pose']
        canonical_ret = None
        if args.object_diagnostics_output:
            canonical_ret = run_opencood_inference(
                manager,
                frame,
                target_ego_lidar_pose)
        frame_items = [(frame, None)]
        late_receiver_policy = (
            args.sgcp_receiver_policy
            if args.sgcp_receiver_policy == 'all-scheduled-receivers'
            else 'all-cluster-heads')
        if args.selective_sharing_baseline is not None:
            protocol = load_protocol(dataset, scenario_id)
            frame_items = apply_selective_sharing_baseline(
                frame,
                protocol,
                args.ego_cav_id,
                args.selective_sharing_baseline,
                late_receiver_policy if args.sgcp_inter_cluster_late_fusion
                else args.sgcp_receiver_policy,
                args.selective_member_budget,
                args.selective_grid_budget,
                args.t_min_stab,
                args.clustering,
                args.n_max,
                args.rho_th,
                link_quality=ns3_link_quality,
                timestamp=timestamp,
                max_upload_points_per_source=(
                    args.max_upload_points_per_source))
        elif args.sgcp_constrained:
            protocol = load_protocol(dataset, scenario_id)
            frame_items = apply_sgcp_constraint(
                frame,
                protocol,
                args.ego_cav_id,
                args.resource_allocation,
                late_receiver_policy if args.sgcp_inter_cluster_late_fusion
                else args.sgcp_receiver_policy,
                args.t_min_stab,
                args.clustering,
                args.n_max,
                args.rho_th,
                args.num_channels,
                args.bandwidth_mhz,
                args.sgcp_upload_mode,
                args.sgcp_grid_selection_mode,
                args.sgcp_grid_score_mode,
                timestamp,
                fixed_cluster_templates=(
                    fixed_cluster_templates
                    if args.clustering == 'fixed_first_frame' else None),
                head_rb_budget=args.head_rb_budget,
                coverage_fallback=args.sgcp_coverage_fallback,
                coverage_state=sgcp_coverage_state,
                max_upload_points_per_source=(
                    args.max_upload_points_per_source))
        if args.sgcp_inter_cluster_late_fusion:
            pred_tensors = []
            pred_scores = []
            gt_tensors = []
            for receiver_index, (eval_frame, sgcp_metadata) in enumerate(
                    frame_items,
                    start=1):
                try:
                    ret = run_opencood_inference(
                        manager,
                        eval_frame,
                        target_ego_lidar_pose)
                except RuntimeError as error:
                    if not is_empty_pillar_error(error):
                        raise
                    if sgcp_metadata is not None:
                        sgcp_summaries.append(sgcp_metadata)
                        sgcp_trace_rows.append(trace_row(
                            scenario_id,
                            timestamp,
                            sgcp_metadata,
                            eval_frame,
                            skipped='empty_pillars'))
                    print('frame=%s/%s late_source=%s/%s scenario=%s '
                          'timestamp=%s receiver=%s cavs=%s skipped=%s '
                          'comm_bytes=%s' % (
                              index,
                              len(frames),
                              receiver_index,
                              len(frame_items),
                              scenario_id,
                              timestamp,
                              sgcp_metadata['receiver_id'],
                              list(eval_frame.keys()),
                              'empty_pillars',
                              sgcp_metadata['communication_bytes']))
                    continue
                pred_box_tensor, pred_score, gt_box_tensor = ret[0:3]
                if pred_box_tensor is not None and pred_score is not None:
                    pred_tensors.append(pred_box_tensor)
                    pred_scores.append(pred_score)
                if gt_box_tensor is not None:
                    gt_tensors.append(gt_box_tensor)
                if sgcp_metadata is not None:
                    update_coverage_quality_state(
                        sgcp_coverage_state,
                        sgcp_metadata,
                        0 if pred_box_tensor is None else
                        pred_box_tensor.shape[0],
                        0 if gt_box_tensor is None else
                        gt_box_tensor.shape[0])
                    sgcp_summaries.append(sgcp_metadata)
                    sgcp_trace_rows.append(trace_row(
                        scenario_id,
                        timestamp,
                        sgcp_metadata,
                        eval_frame,
                        pred_count=(
                            0 if pred_box_tensor is None else
                            pred_box_tensor.shape[0]),
                        gt_count=(
                            0 if gt_box_tensor is None else
                            gt_box_tensor.shape[0])))
                print('frame=%s/%s late_source=%s/%s scenario=%s '
                      'timestamp=%s receiver=%s cavs=%s pred_boxes=%s '
                      'gt_boxes=%s comm_bytes=%s' % (
                          index,
                          len(frames),
                          receiver_index,
                          len(frame_items),
                          scenario_id,
                          timestamp,
                          sgcp_metadata['receiver_id'],
                          list(eval_frame.keys()),
                          0 if pred_box_tensor is None else
                          pred_box_tensor.shape[0],
                          0 if gt_box_tensor is None else
                          gt_box_tensor.shape[0],
                          sgcp_metadata['communication_bytes']))

            fused_pred, fused_score = manager.naive_late_fusion(
                pred_tensors,
                pred_scores,
                iou_threshold=args.sgcp_late_nms_thresh)
            fused_gt, _ = manager.naive_late_fusion(
                gt_tensors,
                None,
                iou_threshold=args.sgcp_late_nms_thresh)
            print('sgcp_late_fusion frame=%s/%s scenario=%s timestamp=%s '
                  'sources=%s fused_pred_boxes=%s fused_gt_boxes=%s' % (
                      index,
                      len(frames),
                      scenario_id,
                      timestamp,
                      len(frame_items),
                      0 if fused_pred is None else fused_pred.shape[0],
                      0 if fused_gt is None else fused_gt.shape[0]))
            if canonical_ret is not None:
                aggregate_metadata = {
                    'receiver_id': args.ego_cav_id,
                    'resource_allocation': (
                        'selective_%s' % args.selective_sharing_baseline
                        if args.selective_sharing_baseline is not None
                        else args.resource_allocation),
                    'clustering': args.clustering,
                    'upload_mode': args.sgcp_upload_mode,
                    'grid_selection_mode': args.sgcp_grid_selection_mode,
                    'grid_score_mode': args.sgcp_grid_score_mode,
                    'num_channels': args.num_channels,
                    'bandwidth_mhz': args.bandwidth_mhz,
                    'communication_bytes': sum(
                        int((metadata or {}).get('communication_bytes', 0))
                        for _, metadata in frame_items),
                    'source_cav_ids': sorted(set(
                        source_id
                        for _, metadata in frame_items
                        for source_id in (
                            (metadata or {}).get('source_cav_ids', [])))),
                    'selected_grid_counts': {},
                }
                for _, metadata in frame_items:
                    if not metadata:
                        continue
                    for key, value in metadata.get(
                            'selected_grid_counts', {}).items():
                        aggregate_metadata['selected_grid_counts'][key] = (
                            aggregate_metadata[
                                'selected_grid_counts'].get(key, 0) +
                            value)
                object_rows.extend(object_diagnostic_rows(
                    scenario_id,
                    timestamp,
                    'inter_cluster_late_fusion',
                    aggregate_metadata,
                    canonical_ret[2],
                    canonical_ret[3] if len(canonical_ret) > 3 else None,
                    canonical_ret[0],
                    canonical_ret[1],
                    fused_pred,
                    fused_score,
                    args.object_diagnostics_iou))
            manager.submit_results(
                fused_pred,
                fused_score,
                fused_gt,
                with_stats=True,
                force=True)
            continue
        for receiver_index, (eval_frame, sgcp_metadata) in enumerate(
                frame_items,
                start=1):
            ego = next(cav for cav in eval_frame.values() if cav['ego'])
            ego_lidar_pose = ego['params']['lidar_pose']

            ret = run_opencood_inference(manager, eval_frame, ego_lidar_pose)

            pred_box_tensor, pred_score, gt_box_tensor = ret[0:3]
            pred_count = (
                0 if pred_box_tensor is None else pred_box_tensor.shape[0])
            gt_count = 0 if gt_box_tensor is None else gt_box_tensor.shape[0]
            print('frame=%s/%s receiver_sample=%s/%s scenario=%s '
                  'timestamp=%s cavs=%s' % (
                      index,
                      len(frames),
                      receiver_index,
                      len(frame_items),
                      scenario_id,
                      timestamp,
                      list(eval_frame.keys())))
            if sgcp_metadata is not None:
                update_coverage_quality_state(
                    sgcp_coverage_state,
                    sgcp_metadata,
                    pred_count,
                    gt_count)
                sgcp_summaries.append(sgcp_metadata)
                sgcp_trace_rows.append(trace_row(
                    scenario_id,
                    timestamp,
                    sgcp_metadata,
                    eval_frame,
                    pred_count=pred_count,
                    gt_count=gt_count))
                print('sgcp_constrained receiver=%s policy=%s sources=%s '
                      'clusters=%s ra=%s comm_bytes=%s selected_grids=%s' % (
                          sgcp_metadata['receiver_id'],
                          sgcp_metadata['receiver_policy'],
                          sgcp_metadata['source_cav_ids'],
                          sgcp_metadata['cluster_count'],
                          sgcp_metadata['resource_allocation'],
                          sgcp_metadata['communication_bytes'],
                          sgcp_metadata['selected_grid_counts']))
            print('fusion_method=%s pred_boxes=%s gt_boxes=%s' %
                  (coperception_params['fusion_method'], pred_count, gt_count))
            if pred_score is not None:
                print('pred_scores_shape=%s' % (tuple(pred_score.shape),))
            if canonical_ret is not None:
                sample_label = (
                    'full_reference'
                    if sgcp_metadata is None
                    else 'receiver_sample')
                object_rows.extend(object_diagnostic_rows(
                    scenario_id,
                    timestamp,
                    sample_label,
                    sgcp_metadata,
                    canonical_ret[2],
                    canonical_ret[3] if len(canonical_ret) > 3 else None,
                    canonical_ret[0],
                    canonical_ret[1],
                    pred_box_tensor,
                    pred_score,
                    args.object_diagnostics_iou))
            manager.submit_results(
                pred_box_tensor,
                pred_score,
                gt_box_tensor,
                with_stats=True,
                force=True)

    if len(frames) > 1:
        manager.evaluate_final_average_precision()
        if sgcp_summaries:
            total_comm = sum(item['communication_bytes']
                             for item in sgcp_summaries)
            avg_comm = total_comm / float(len(sgcp_summaries))
            avg_sources = sum(len(item['source_cav_ids'])
                              for item in sgcp_summaries) / \
                float(len(sgcp_summaries))
            avg_selected_grids = sum(
                sum(item.get('selected_grid_counts', {}).values())
                for item in sgcp_summaries) / float(len(sgcp_summaries))
            print('sgcp_summary frames=%s avg_comm_bytes=%.2f '
                  'total_comm_bytes=%s avg_source_cavs=%.2f '
                  'avg_selected_grids=%.2f' % (
                      len(sgcp_summaries),
                      avg_comm,
                      total_comm,
                      avg_sources,
                avg_selected_grids))
    write_trace_csv(args.sgcp_trace_output, sgcp_trace_rows)
    write_object_diagnostics_csv(
        args.object_diagnostics_output,
        object_rows)


if __name__ == '__main__':
    main()
