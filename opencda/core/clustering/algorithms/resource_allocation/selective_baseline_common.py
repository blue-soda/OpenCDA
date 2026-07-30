# -*- coding: utf-8 -*-
"""Shared helpers for offline selective-sharing baselines."""

from collections import defaultdict


EDGECOOPER_GLOBAL_COMM_RANGE_M = 35.0


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


def vehicle_distance(vm_a, vm_b):
    pos_a = vm_a.v2x_manager.get_ego_pos().location
    pos_b = vm_b.v2x_manager.get_ego_pos().location
    return (
        (pos_a.x - pos_b.x) ** 2 +
        (pos_a.y - pos_b.y) ** 2) ** 0.5


def grid_index_from_id(grid_id):
    try:
        return tuple(int(item) for item in str(grid_id).split('_'))
    except (TypeError, ValueError):
        return None


def ns3_link_quality(link_quality, timestamp, source_id, target_id):
    if not link_quality:
        return None
    exact_key = (str(timestamp), int(source_id), int(target_id))
    if exact_key in link_quality['exact']:
        return link_quality['exact'][exact_key]
    return link_quality['pair'].get((int(source_id), int(target_id)))


def collect_receiver_grid_selection(world, clusters):
    selection = {}
    for cluster in clusters:
        receiver_id = int(cluster.head_id)
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        grid_selection = getattr(
            receiver_vm.perception_manager.co_manager,
            'grid_selection',
            {}) or {}
        normalized = {}
        for sender_id, grid_ids in grid_selection.items():
            grid_ids = list(grid_ids or [])
            if grid_ids:
                normalized[int(sender_id)] = grid_ids
        if normalized:
            selection[receiver_id] = normalized
    return selection


def apply_receiver_grid_selection(world, clusters, selection):
    for cluster in clusters:
        receiver_id = int(cluster.head_id)
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        co_manager = receiver_vm.perception_manager.co_manager
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(selection.get(receiver_id, {}))


def endpoint_matched_links(link_entries, baseline_name,
                           max_senders_per_receiver, num_channels,
                           constrained_baselines=None):
    constrained = set(constrained_baselines or [])
    if baseline_name not in constrained:
        return list(link_entries)

    matched = []
    occupied_senders = set()
    receiver_loads = defaultdict(int)
    max_inbound = max(1, int(max_senders_per_receiver or 1))
    for item in link_entries:
        sender_id = item['sender_id']
        receiver_id = item['receiver_id']
        sender_blocked = (
            sender_id in occupied_senders or
            receiver_loads.get(sender_id, 0) > 0)
        receiver_blocked = (
            receiver_id in occupied_senders or
            receiver_loads.get(receiver_id, 0) >= max_inbound)
        if sender_blocked or receiver_blocked:
            continue
        matched.append(item)
        occupied_senders.add(sender_id)
        receiver_loads[receiver_id] += 1
        if len(matched) >= int(num_channels):
            break
    return matched
