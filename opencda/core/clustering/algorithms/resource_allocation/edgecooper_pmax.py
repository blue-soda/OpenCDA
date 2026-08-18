# -*- coding: utf-8 -*-
"""EdgeCooper Pmax-style partial point upload helpers.

The baseline scheduler in ``offline_inference`` decides which V2V links and
OpenCDA grids are worth uploading.  This module adds the EdgeCooper paper's
``Pmax`` idea as a separate raw-LiDAR adaptation: once a receiver grid already
contains enough local points, additional uploaded points from the same grid are
clipped to the residual point quota.
"""

from collections import defaultdict

from opencda.core.common.offline_replay import density_cap_points_per_grid


PMAX_SUFFIX = '_pmax'
EDGECOOPER_BASELINES = {
    'edgecooper',
    'edgecooper_global',
    'edgecooper_global_hd',
}


def is_edgecooper_pmax_baseline(baseline_name):
    return str(baseline_name or '').endswith(PMAX_SUFFIX)


def edgecooper_base_baseline(baseline_name):
    baseline_name = str(baseline_name or '')
    if is_edgecooper_pmax_baseline(baseline_name):
        return baseline_name[:-len(PMAX_SUFFIX)]
    return baseline_name


def edgecooper_baseline_names():
    names = sorted(EDGECOOPER_BASELINES)
    return names + ['%s%s' % (name, PMAX_SUFFIX) for name in names]


def resolve_pmax_density_cap_rho(explicit_rho, protocol_rho):
    if explicit_rho is not None and explicit_rho > 0:
        return float(explicit_rho)
    if protocol_rho is not None and protocol_rho > 0:
        return float(protocol_rho)
    return 2.0


def _clone_selection(selection):
    cloned = {}
    for receiver_id, sender_grids in (selection or {}).items():
        normalized = {}
        for sender_id, grid_ids in (sender_grids or {}).items():
            grid_ids = list(grid_ids or [])
            if grid_ids:
                normalized[int(sender_id)] = list(dict.fromkeys(grid_ids))
        if normalized:
            cloned[int(receiver_id)] = normalized
    return cloned


def _init_remaining(world, receiver_id, selection, density_cap_rho):
    if density_cap_rho is None or density_cap_rho <= 0:
        return None
    receiver_vm = world.get_vehicle_manager(receiver_id)
    if receiver_vm is None:
        return None
    receiver_lidar = receiver_vm.perception_manager.lidar
    max_points = density_cap_points_per_grid(receiver_lidar, density_cap_rho)
    if max_points is None:
        return None
    remaining = {}
    for grid_ids in selection.get(int(receiver_id), {}).values():
        for grid_id in grid_ids:
            if grid_id in remaining:
                continue
            local_count = len(receiver_lidar.grid_local_points.get(
                grid_id,
                []))
            remaining[grid_id] = max(0, max_points - local_count)
    return remaining


def _grid_payload_bytes(lidar, grid_id, remaining):
    point_count = len(lidar.grid_local_points.get(grid_id, []))
    if remaining is not None:
        point_count = min(point_count, max(0, int(remaining.get(grid_id, 0))))
    return int(point_count * 4 * 4)


def _consume_remaining(remaining, grid_id, payload_bytes):
    if remaining is None:
        return
    consumed_points = int(payload_bytes / (4 * 4))
    remaining[grid_id] = max(0, int(remaining.get(grid_id, 0)) -
                             consumed_points)


def _endpoint_matched_links(link_entries, baseline_name, max_senders_per_receiver,
                            num_channels):
    if baseline_name not in ['edgecooper_global', 'edgecooper_global_hd',
                             'pacp_lidar']:
        return link_entries
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


def _endpoint_matched_link_batches(link_entries, baseline_name,
                                   max_senders_per_receiver, num_channels):
    """Greedily split candidate links into orthogonal resource batches.

    A single 40 MHz / 10ch perception frame contains more than one scheduling
    instant inside the data-plane deadline.  The endpoint constraints apply to
    each orthogonal batch, not to the entire 60 ms frame.  Returning a flattened
    batch order lets the byte-budget admission below reuse the channel pool
    until the frame budget is exhausted.
    """
    if baseline_name not in ['edgecooper_global', 'edgecooper_global_hd',
                             'pacp_lidar']:
        return [list(link_entries)]

    remaining = list(link_entries)
    batches = []
    while remaining:
        batch = _endpoint_matched_links(
            remaining,
            baseline_name,
            max_senders_per_receiver,
            num_channels)
        if not batch:
            break
        batch_ids = set(id(item) for item in batch)
        batches.append(batch)
        remaining = [
            item for item in remaining
            if id(item) not in batch_ids
        ]
    return batches


def trim_pmax_selection_to_global_deadline(
        world,
        selection,
        baseline_name,
        deadline_ms,
        channel_model,
        grid_score_fn,
        max_senders_per_receiver=1,
        density_cap_rho=2.0):
    """Admit Pmax-clipped grids under one shared frame deadline.

    Args:
        grid_score_fn: callable ``(receiver_vm, sender_vm, grid_id) -> score``.

    Returns:
        tuple: ``(admitted_selection, metadata)``.
    """
    selection = _clone_selection(selection)
    base_name = edgecooper_base_baseline(baseline_name)
    budget_bytes = channel_model.payload_budget_bytes(
        deadline_ms=deadline_ms,
        subchannels=channel_model.num_channels)
    if deadline_ms is None or budget_bytes <= 0:
        return {}, {
            'budget_bytes': int(budget_bytes),
            'admitted_bytes': 0,
            'candidate_bytes': 0,
            'admitted_links': 0,
            'candidate_links': 0,
            'pre_matching_candidate_links': 0,
            'max_senders_per_receiver': max(
                1,
                int(max_senders_per_receiver or 1)),
            'edgecooper_pmax_density_cap_rho': density_cap_rho,
        }

    link_entries = []
    candidate_remaining = {}
    candidate_bytes = 0
    for receiver_id, sender_grids in selection.items():
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        candidate_remaining[receiver_id] = _init_remaining(
            world,
            receiver_id,
            selection,
            density_cap_rho)
        for sender_id, grid_ids in sender_grids.items():
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            entries = []
            for grid_id in grid_ids:
                score = float(grid_score_fn(receiver_vm, sender_vm, grid_id))
                density = float(
                    sender_vm.perception_manager.lidar.get_grid_density(
                        grid_id))
                entries.append((-score, -density, str(grid_id), grid_id))
            entries.sort()
            if entries:
                link_entries.append({
                    'receiver_id': int(receiver_id),
                    'sender_id': int(sender_id),
                    'entries': entries,
                    'cursor': 0,
                })
                remaining = candidate_remaining[receiver_id]
                lidar = sender_vm.perception_manager.lidar
                for _, _, _, grid_id in entries:
                    grid_bytes = _grid_payload_bytes(lidar, grid_id, remaining)
                    candidate_bytes += grid_bytes
                    _consume_remaining(remaining, grid_id, grid_bytes)

    link_entries.sort(key=lambda item: item['entries'][0][:3])
    pre_matching_candidate_links = len(link_entries)
    link_batches = _endpoint_matched_link_batches(
        link_entries,
        base_name,
        max_senders_per_receiver,
        channel_model.num_channels)
    link_entries = [item for batch in link_batches for item in batch]

    remaining_by_receiver = {
        receiver_id: _init_remaining(
            world,
            receiver_id,
            selection,
            density_cap_rho)
        for receiver_id in selection
    }
    admitted = {}
    admitted_bytes = 0
    while True:
        advanced = False
        for item in link_entries:
            while item['cursor'] < len(item['entries']):
                _, _, _, grid_id = item['entries'][item['cursor']]
                item['cursor'] += 1
                advanced = True
                sender_vm = world.get_vehicle_manager(item['sender_id'])
                grid_bytes = _grid_payload_bytes(
                    sender_vm.perception_manager.lidar,
                    grid_id,
                    remaining_by_receiver[item['receiver_id']])
                if grid_bytes <= 0:
                    continue
                if admitted_bytes + grid_bytes > budget_bytes:
                    continue
                admitted.setdefault(item['receiver_id'], {}).setdefault(
                    item['sender_id'], []).append(grid_id)
                admitted_bytes += grid_bytes
                _consume_remaining(
                    remaining_by_receiver[item['receiver_id']],
                    grid_id,
                    grid_bytes)
                break
            if admitted_bytes >= budget_bytes:
                break
        if not advanced or admitted_bytes >= budget_bytes:
            break

    admitted_links = sum(
        1 for sender_grids in admitted.values()
        for grid_ids in sender_grids.values() if grid_ids)
    return admitted, {
        'budget_bytes': int(budget_bytes),
        'admitted_bytes': int(admitted_bytes),
        'candidate_bytes': int(candidate_bytes),
        'admitted_links': int(admitted_links),
        'candidate_links': int(len(link_entries)),
        'candidate_batches': int(len(link_batches)),
        'pre_matching_candidate_links': int(pre_matching_candidate_links),
        'max_senders_per_receiver': max(
            1,
            int(max_senders_per_receiver or 1)),
        'edgecooper_pmax_density_cap_rho': density_cap_rho,
    }
