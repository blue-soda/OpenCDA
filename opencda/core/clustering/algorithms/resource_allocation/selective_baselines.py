# -*- coding: utf-8 -*-
"""Offline selective-sharing baseline orchestration.

This module keeps baseline scheduling out of ``opencda.tools.offline_inference``.
It only mutates each receiver's co_manager grid selection; the offline runner is
responsible for building constrained OpenCOOD samples afterwards.
"""

from collections import defaultdict
import math
import random

from opencda.core.clustering.algorithms.resource_allocation.edgecooper import (
    edgecooper_candidate_grids,
    edgecooper_global_sender_capacity,
    edgecooper_grid_score,
    select_edgecooper_grids,
    select_edgecooper_members,
)
from opencda.core.clustering.algorithms.resource_allocation.edgecooper_pmax \
    import edgecooper_base_baseline
from opencda.core.clustering.algorithms.resource_allocation.pacp_lidar import (
    pacp_lidar_candidate_grids,
    pacp_lidar_grid_score,
    select_pacp_lidar_grids,
    select_pacp_lidar_members,
)
from opencda.core.clustering.algorithms.resource_allocation.\
    selective_baseline_common import (
        EDGECOOPER_GLOBAL_COMM_RANGE_M,
        apply_receiver_grid_selection,
        candidate_grids_for_sender,
        collect_receiver_grid_selection,
        endpoint_matched_links,
        ns3_link_quality,
        vehicle_distance,
    )


def candidate_member_ids(world, cluster, baseline_name):
    baseline_name = edgecooper_base_baseline(baseline_name)
    head_id = int(cluster.head_id)
    if baseline_name in ['global_selective_proxy', 'edgecooper']:
        return [
            int(member_id)
            for member_id in sorted(world.get_vehicle_managers().keys())
            if int(member_id) != head_id
        ]
    if baseline_name in ['edgecooper_global', 'edgecooper_global_hd']:
        head_vm = world.get_vehicle_manager(head_id)
        receiver_ids = set(getattr(
            world,
            '_edgecooper_global_receiver_ids',
            set()) or set())
        feasible_members = []
        for member_id in sorted(world.get_vehicle_managers().keys()):
            member_id = int(member_id)
            if member_id == head_id:
                continue
            if (baseline_name == 'edgecooper_global_hd' and
                    member_id in receiver_ids):
                continue
            sender_vm = world.get_vehicle_manager(member_id)
            comm_range_m = float(getattr(
                world,
                '_edgecooper_global_comm_range_m',
                EDGECOOPER_GLOBAL_COMM_RANGE_M))
            if vehicle_distance(head_vm, sender_vm) <= comm_range_m:
                feasible_members.append(member_id)
        return feasible_members
    if baseline_name == 'pacp_lidar' and len(cluster.members) <= 1:
        head_vm = world.get_vehicle_manager(head_id)
        feasible_members = []
        comm_range_m = float(getattr(
            world,
            '_pacp_lidar_global_comm_range_m',
            EDGECOOPER_GLOBAL_COMM_RANGE_M))
        for member_id in sorted(world.get_vehicle_managers().keys()):
            member_id = int(member_id)
            if member_id == head_id:
                continue
            sender_vm = world.get_vehicle_manager(member_id)
            if vehicle_distance(head_vm, sender_vm) <= comm_range_m:
                feasible_members.append(member_id)
        return feasible_members
    return [
        int(member_id) for member_id in sorted(cluster.members)
        if int(member_id) != head_id
    ]


def density_score_for_member(head_vm, sender_vm):
    candidate_grids = candidate_grids_for_sender(head_vm, sender_vm)
    return sum(
        sender_vm.perception_manager.lidar.get_grid_density(grid_id)
        for grid_id in candidate_grids)


def select_baseline_members(world, cluster, baseline_name, member_budget,
                            link_quality=None, timestamp=None):
    baseline_name = edgecooper_base_baseline(baseline_name)
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

    if baseline_name in ['edgecooper', 'edgecooper_global',
                         'edgecooper_global_hd']:
        sender_loads = None
        sender_capacity = None
        if baseline_name in ['edgecooper_global', 'edgecooper_global_hd']:
            sender_loads = getattr(
                world,
                '_edgecooper_global_sender_loads',
                None)
            if sender_loads is None:
                sender_loads = {}
                world._edgecooper_global_sender_loads = sender_loads
            sender_capacity = edgecooper_global_sender_capacity(
                world,
                member_budget)
        return select_edgecooper_members(
            world,
            head_vm,
            members,
            member_budget,
            global_sender_loads=sender_loads,
            sender_capacity=sender_capacity)

    if baseline_name == 'pacp_lidar':
        return select_pacp_lidar_members(
            world,
            head_vm,
            members,
            member_budget,
            link_quality=link_quality,
            timestamp=timestamp)

    if baseline_name in ['density', 'greedy_density', 'communication_aware',
                         'global_selective_proxy',
                         'cluster_local_selective_proxy']:
        scored = []
        for member_id in members:
            sender_vm = world.get_vehicle_manager(member_id)
            density_sum = density_score_for_member(head_vm, sender_vm)
            if baseline_name in ['communication_aware',
                                 'cluster_local_selective_proxy']:
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
            elif baseline_name == 'global_selective_proxy':
                distance = vehicle_distance(head_vm, sender_vm)
                density_sum = density_sum / (1.0 + distance / 200.0)
            scored.append((-density_sum, member_id))
        return [member_id for _, member_id in sorted(scored)[:member_budget]]

    raise ValueError('Unknown selective baseline: %s' % baseline_name)


def assign_selective_grid_selection(world, cluster, baseline_name,
                                    member_budget, grid_budget,
                                    link_quality=None, timestamp=None):
    baseline_name = edgecooper_base_baseline(baseline_name)
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
        if baseline_name in ['edgecooper', 'edgecooper_global',
                             'edgecooper_global_hd']:
            candidate_grids = edgecooper_candidate_grids(head_vm, sender_vm)
        elif baseline_name == 'pacp_lidar':
            candidate_grids = pacp_lidar_candidate_grids(head_vm, sender_vm)
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
        elif baseline_name in ['edgecooper', 'edgecooper_global',
                               'edgecooper_global_hd']:
            grids = select_edgecooper_grids(
                head_vm,
                sender_vm,
                candidate_grids,
                min(per_member_budget, remaining),
                covered_grids=covered_edge_grids)
        elif baseline_name == 'pacp_lidar':
            grids = select_pacp_lidar_grids(
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
            if baseline_name in ['edgecooper', 'edgecooper_global',
                                 'edgecooper_global_hd', 'pacp_lidar']:
                covered_edge_grids.update(selected)
            remaining -= len(selected)
    head_vm.perception_manager.co_manager.set_grid_selection(grid_selection)


def trim_selective_grid_selection_to_global_deadline(
        world, clusters, baseline_name, deadline_ms, channel_model,
        max_senders_per_receiver=1):
    original = collect_receiver_grid_selection(world, clusters)
    budget_bytes = channel_model.payload_budget_bytes(
        deadline_ms=deadline_ms,
        subchannels=channel_model.num_channels)
    if deadline_ms is None or budget_bytes <= 0:
        return {
            'budget_bytes': budget_bytes,
            'admitted_bytes': 0,
            'candidate_bytes': 0,
            'admitted_links': 0,
            'candidate_links': 0,
        }

    baseline_name = edgecooper_base_baseline(baseline_name)
    link_entries = []
    candidate_bytes = 0
    for receiver_id, sender_grids in original.items():
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        for sender_id, grid_ids in sender_grids.items():
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            entries = []
            for grid_id in grid_ids:
                points = sender_vm.perception_manager.lidar.\
                    get_local_points_by_grid_ids([grid_id])
                grid_bytes = 0 if points is None else int(points.nbytes)
                if grid_bytes <= 0:
                    continue
                candidate_bytes += grid_bytes
                if baseline_name in ['edgecooper', 'edgecooper_global',
                                     'edgecooper_global_hd']:
                    score = edgecooper_grid_score(
                        receiver_vm,
                        sender_vm,
                        grid_id)
                elif baseline_name == 'pacp_lidar':
                    score = pacp_lidar_grid_score(
                        receiver_vm,
                        sender_vm,
                        grid_id)
                else:
                    score = sender_vm.perception_manager.lidar.\
                        get_grid_density(grid_id)
                entries.append((
                    -float(score),
                    -float(sender_vm.perception_manager.lidar.
                           get_grid_density(grid_id)),
                    str(grid_id),
                    grid_id,
                    grid_bytes))
            entries.sort()
            if entries:
                link_entries.append({
                    'receiver_id': int(receiver_id),
                    'sender_id': int(sender_id),
                    'entries': entries,
                    'cursor': 0,
                })

    link_entries.sort(key=lambda item: item['entries'][0][:3])
    original_link_count = len(link_entries)
    link_entries = endpoint_matched_links(
        link_entries,
        baseline_name,
        max_senders_per_receiver,
        channel_model.num_channels,
        constrained_baselines=[
            'edgecooper_global',
            'edgecooper_global_hd',
            'pacp_lidar',
        ])

    admitted = {}
    admitted_bytes = 0
    while True:
        advanced = False
        for item in link_entries:
            cursor = item['cursor']
            if cursor >= len(item['entries']):
                continue
            _, _, _, grid_id, grid_bytes = item['entries'][cursor]
            item['cursor'] += 1
            advanced = True
            if admitted_bytes + grid_bytes > budget_bytes:
                continue
            receiver_id = item['receiver_id']
            sender_id = item['sender_id']
            admitted.setdefault(receiver_id, {}).setdefault(
                sender_id, []).append(grid_id)
            admitted_bytes += grid_bytes
            if admitted_bytes >= budget_bytes:
                break
        if not advanced or admitted_bytes >= budget_bytes:
            break

    apply_receiver_grid_selection(world, clusters, admitted)
    admitted_links = sum(
        1 for sender_grids in admitted.values()
        for grid_ids in sender_grids.values() if grid_ids)
    candidate_links = len(link_entries)
    return {
        'budget_bytes': int(budget_bytes),
        'admitted_bytes': int(admitted_bytes),
        'candidate_bytes': int(candidate_bytes),
        'admitted_links': int(admitted_links),
        'candidate_links': int(candidate_links),
        'pre_matching_candidate_links': int(original_link_count),
        'max_senders_per_receiver': max(
            1,
            int(max_senders_per_receiver or 1)),
    }
