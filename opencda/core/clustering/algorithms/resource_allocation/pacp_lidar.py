# -*- coding: utf-8 -*-
"""PACP LiDAR-grid adaptation used by offline baseline experiments."""

from opencda.core.clustering.algorithms.resource_allocation.\
    selective_baseline_common import (
        ns3_link_quality,
        receiver_blind_grids,
        vehicle_distance,
    )


def pacp_lidar_candidate_grids(head_vm, sender_vm):
    """LiDAR adaptation of PACP BEV-match priority over raw point grids."""
    sender_lidar = sender_vm.perception_manager.lidar
    head_lidar = head_vm.perception_manager.lidar
    blind_grids = receiver_blind_grids(head_vm)
    candidates = sender_lidar.sens_grids & (blind_grids | head_lidar.req_grids)
    if not candidates:
        candidates = sender_lidar.sens_grids
    return candidates


def pacp_lidar_grid_score(head_vm, sender_vm, grid_id, covered_grids=None):
    sender_lidar = sender_vm.perception_manager.lidar
    head_lidar = head_vm.perception_manager.lidar
    covered_grids = set() if covered_grids is None else set(covered_grids)
    sender_density = sender_lidar.get_grid_density(grid_id)
    head_density = head_lidar.get_grid_density(grid_id)
    blind_bonus = 1.0 if grid_id in receiver_blind_grids(head_vm) else 0.35
    overlap_match = min(sender_density, head_density)
    complementarity = max(0.0, sender_density - head_density)
    novelty = 1.0 if grid_id not in covered_grids else 0.25
    return novelty * (
        0.55 * overlap_match +
        0.90 * blind_bonus * complementarity +
        0.20 * blind_bonus * sender_density)


def pacp_lidar_member_score(head_vm, sender_vm, covered_grids=None,
                            link_quality=None, timestamp=None):
    covered_grids = set() if covered_grids is None else set(covered_grids)
    candidates = set(pacp_lidar_candidate_grids(head_vm, sender_vm))
    if not candidates:
        return 0.0, candidates
    head_id = int(head_vm.vehicle_id)
    sender_id = int(sender_vm.vehicle_id)
    bev_match = 0.0
    complementarity = 0.0
    for grid_id in candidates:
        sender_density = sender_vm.perception_manager.lidar.get_grid_density(
            grid_id)
        head_density = head_vm.perception_manager.lidar.get_grid_density(
            grid_id)
        if grid_id in head_vm.perception_manager.lidar.req_grids:
            bev_match += min(sender_density, head_density)
        if grid_id not in covered_grids:
            complementarity += max(0.0, sender_density - head_density)
    distance = vehicle_distance(head_vm, sender_vm)
    quality = ns3_link_quality(link_quality, timestamp, sender_id, head_id)
    if quality is None:
        quality = 1.0 / (1.0 + distance / 100.0)
    score = (0.60 * bev_match + 1.00 * complementarity) * quality
    score = score / (1.0 + len(candidates) / 200.0)
    return score, candidates


def select_pacp_lidar_members(world, head_vm, members, member_budget,
                              link_quality=None, timestamp=None):
    selected = []
    covered = set()
    remaining = set(members)
    while remaining and len(selected) < member_budget:
        best = None
        for member_id in sorted(remaining):
            sender_vm = world.get_vehicle_manager(member_id)
            score, candidates = pacp_lidar_member_score(
                head_vm,
                sender_vm,
                covered_grids=covered,
                link_quality=link_quality,
                timestamp=timestamp)
            distance = vehicle_distance(head_vm, sender_vm)
            item = (-score, distance, member_id, candidates)
            if best is None or item < best:
                best = item
        if best is None or best[0] >= 0.0:
            break
        _, _, member_id, candidates = best
        selected.append(member_id)
        covered.update(candidates)
        remaining.remove(member_id)
    return selected


def select_pacp_lidar_grids(head_vm, sender_vm, candidates, count,
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
                pacp_lidar_grid_score(
                    head_vm,
                    sender_vm,
                    grid_id,
                    covered_grids=covered_grids | set(selected)),
                sender_vm.perception_manager.lidar.get_grid_density(grid_id),
                str(grid_id)))
        selected.append(best)
        remaining.remove(best)
    return selected
