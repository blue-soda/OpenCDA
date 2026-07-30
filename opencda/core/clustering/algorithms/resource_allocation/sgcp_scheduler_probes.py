# -*- coding: utf-8 -*-
"""SGCP scheduler variants and diagnostic grid-replacement probes.

These helpers mutate already-built SGCP grid selections. Keeping them in the
resource-allocation package avoids hiding scheduling logic inside the generic
offline inference runner.
"""

import csv
import random
from collections import defaultdict

from opencda.core.clustering.algorithms.resource_allocation.\
    selective_baseline_common import (
        candidate_grids_for_sender,
        grid_index_from_id,
    )


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
    """Swap in repeatedly unscheduled members without changing link count."""
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


def load_sgcp_routing_hints(path):
    """Load diagnostic target-to-head routing hints."""
    if not path:
        return None
    hints = defaultdict(list)
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            timestamp = str(row.get('timestamp', ''))
            object_grid = str(row.get('object_grid_id', '')).strip()
            if not timestamp or not object_grid:
                continue
            try:
                receiver_id = int(float(row.get('nearest_head', '')))
            except (TypeError, ValueError):
                continue
            sender_value = (
                row.get('best_raw_cav_id_m0p0') or
                row.get('best_raw_cav_id_m2p0') or
                row.get('nearest_cav'))
            try:
                sender_id = int(float(sender_value))
            except (TypeError, ValueError):
                continue
            try:
                ratio = float(row.get('sgcp_full_box_point_ratio_m0p0', 1.0)
                              or 1.0)
            except (TypeError, ValueError):
                ratio = 1.0
            try:
                full_points = int(float(
                    row.get('full_reference_box_points_m0p0', 0) or 0))
            except (TypeError, ValueError):
                full_points = 0
            try:
                raw_points = int(float(
                    row.get('best_raw_cav_box_points_m0p0', 0) or 0))
            except (TypeError, ValueError):
                raw_points = 0
            hints[timestamp].append({
                'timestamp': timestamp,
                'receiver_id': receiver_id,
                'sender_id': sender_id,
                'object_grid_id': object_grid,
                'object_id': row.get('object_id', ''),
                'ratio': ratio,
                'full_points': full_points,
                'raw_points': raw_points,
                'score': (
                    (1.0 - min(max(ratio, 0.0), 1.0)) *
                    max(full_points, 1) +
                    0.2 * raw_points),
            })
    for timestamp in list(hints.keys()):
        hints[timestamp] = sorted(
            hints[timestamp],
            key=lambda item: (
                item['score'],
                item['full_points'],
                item['raw_points'],
                -item['sender_id']),
            reverse=True)
    return hints


def neighboring_grid_ids(grid_id, radius=1):
    index = grid_index_from_id(grid_id)
    if index is None:
        return [grid_id]
    gx, gy = index
    grids = []
    for dist in range(radius + 1):
        for dx in range(-dist, dist + 1):
            for dy in range(-dist, dist + 1):
                if abs(dx) + abs(dy) != dist:
                    continue
                grids.append('%d_%d' % (gx + dx, gy + dy))
    return grids


def hinted_grid_selection(head_vm, sender_vm, object_grid_id, count):
    sender_lidar = sender_vm.perception_manager.lidar
    candidates = set(candidate_grids_for_sender(head_vm, sender_vm))
    if not candidates:
        candidates = set(sender_lidar.sens_grids)
    selected = []
    for grid_id in neighboring_grid_ids(object_grid_id, radius=2):
        if grid_id in candidates and sender_lidar.get_grid_density(grid_id) > 0:
            selected.append(grid_id)
        if len(selected) >= count:
            return selected
    remaining = [
        grid for grid in candidates
        if grid not in selected and sender_lidar.get_grid_density(grid) > 0
    ]
    remaining = sorted(
        remaining,
        key=lambda grid: (
            -grid_l1_distance(grid, object_grid_id),
            sender_lidar.get_grid_density(grid),
            str(grid)),
        reverse=True)
    for grid_id in remaining:
        selected.append(grid_id)
        if len(selected) >= count:
            break
    return selected


def merge_hinted_grid_selection(head_vm, sender_vm, object_grid_id,
                                existing_grids):
    """Preserve detector context while forcing a small target-grid hint."""
    count = len(existing_grids)
    if count <= 0:
        return []
    sender_lidar = sender_vm.perception_manager.lidar
    candidates = set(candidate_grids_for_sender(head_vm, sender_vm))
    if not candidates:
        candidates = set(sender_lidar.sens_grids)
    hint_grids = []
    for grid_id in neighboring_grid_ids(object_grid_id, radius=1):
        if grid_id in candidates and sender_lidar.get_grid_density(grid_id) > 0:
            hint_grids.append(grid_id)
        if len(hint_grids) >= min(3, count):
            break
    if not hint_grids:
        return list(existing_grids)
    selected = []
    selected_set = set()
    for grid_id in hint_grids:
        if grid_id not in selected_set:
            selected.append(grid_id)
            selected_set.add(grid_id)
    preserved = sorted(
        [grid for grid in existing_grids if grid not in selected_set],
        key=lambda grid: (
            sender_lidar.get_grid_density(grid),
            str(grid)),
        reverse=True)
    for grid_id in preserved:
        selected.append(grid_id)
        selected_set.add(grid_id)
        if len(selected) >= count:
            return selected
    remaining = sorted(
        [grid for grid in candidates if grid not in selected_set],
        key=lambda grid: (
            sender_lidar.get_grid_density(grid),
            str(grid)),
        reverse=True)
    for grid_id in remaining:
        if sender_lidar.get_grid_density(grid_id) <= 0:
            continue
        selected.append(grid_id)
        if len(selected) >= count:
            break
    return selected


def apply_diagnostic_routing_hints(world, timestamp, routing_hints,
                                   max_per_frame=1):
    """Apply oracle/debug target-to-head route replacements."""
    if not routing_hints or max_per_frame <= 0:
        return 0
    hints = routing_hints.get(str(timestamp), [])
    if not hints:
        return 0
    applied = 0
    used_receivers = set()
    used_senders = set()
    for hint in hints:
        if applied >= max_per_frame:
            break
        receiver_id = int(hint['receiver_id'])
        sender_id = int(hint['sender_id'])
        if receiver_id in used_receivers or sender_id in used_senders:
            continue
        if receiver_id == sender_id:
            continue
        receiver_vm = world.get_vehicle_manager(receiver_id)
        sender_vm = world.get_vehicle_manager(sender_id)
        if receiver_vm is None or sender_vm is None:
            continue
        co_manager = receiver_vm.perception_manager.co_manager
        current_selection = {
            int(src): list(grids)
            for src, grids in (
                getattr(co_manager, 'grid_selection', {}) or {}).items()
        }
        if not current_selection:
            continue
        scheduler = receiver_vm.v2x_manager.scheduler
        channel_allocation = getattr(scheduler, 'channel_allocation', {})
        if sender_id in current_selection:
            new_grids = merge_hinted_grid_selection(
                receiver_vm,
                sender_vm,
                hint['object_grid_id'],
                current_selection[sender_id])
            if not new_grids:
                continue
            current_selection[sender_id] = new_grids
            co_manager.clear_grid_selection()
            co_manager.set_grid_selection(current_selection)
            applied += 1
            used_receivers.add(receiver_id)
            used_senders.add(sender_id)
            continue

        def replace_score(src_id):
            grids = current_selection.get(src_id, [])
            src_vm = world.get_vehicle_manager(src_id)
            if src_vm is None:
                return (float('inf'), src_id)
            density_sum = sum(
                src_vm.perception_manager.lidar.get_grid_density(grid)
                for grid in grids)
            return (density_sum, src_id)

        replaceable = [
            src_id for src_id in current_selection.keys()
            if (src_id, receiver_id) in channel_allocation
        ]
        if not replaceable:
            continue
        replaced_id = min(replaceable, key=replace_score)
        replaced_grids = current_selection.get(replaced_id, [])
        count = max(1, len(replaced_grids))
        new_grids = hinted_grid_selection(
            receiver_vm,
            sender_vm,
            hint['object_grid_id'],
            count)
        if not new_grids:
            continue
        old_channel = channel_allocation.pop((replaced_id, receiver_id), None)
        if old_channel is None:
            continue
        channel_allocation[(sender_id, receiver_id)] = old_channel
        current_selection.pop(replaced_id, None)
        current_selection[sender_id] = new_grids
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(current_selection)
        applied += 1
        used_receivers.add(receiver_id)
        used_senders.add(sender_id)
    return applied
