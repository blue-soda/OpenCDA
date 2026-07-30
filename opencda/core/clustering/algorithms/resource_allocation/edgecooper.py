# -*- coding: utf-8 -*-
"""EdgeCooper raw-LiDAR V2V adaptation used by offline experiments."""

import math

from opencda.core.clustering.algorithms.resource_allocation.\
    selective_baseline_common import (
        receiver_blind_grids,
        vehicle_distance,
    )


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


def edgecooper_global_sender_capacity(world, member_budget):
    vehicle_count = max(1, len(world.get_vehicle_managers()))
    cluster_count = max(1, int(getattr(
        world,
        '_edgecooper_global_cluster_count',
        len(world.get_vehicle_managers()))))
    total_slots = max(1, cluster_count * max(1, member_budget))
    return max(1, int(math.ceil(total_slots / float(vehicle_count))))


def select_edgecooper_members(world, head_vm, members, member_budget,
                              global_sender_loads=None,
                              sender_capacity=None):
    selected = []
    covered = set()
    remaining = set(members)
    while remaining and len(selected) < member_budget:
        best = None
        for member_id in sorted(remaining):
            sender_load = 0
            if global_sender_loads is not None:
                sender_load = int(global_sender_loads.get(member_id, 0))
                if sender_capacity is not None and sender_load >= sender_capacity:
                    continue
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
            load_penalty = 1.0 + float(sender_load)
            score = (
                complementarity / ((1.0 + distance / 50.0) * load_penalty) -
                0.35 * redundancy)
            item = (-score, distance, member_id, candidate_grids)
            if best is None or item < best:
                best = item
        if best is None:
            break
        _, _, member_id, candidate_grids = best
        selected.append(member_id)
        if global_sender_loads is not None:
            global_sender_loads[member_id] = (
                int(global_sender_loads.get(member_id, 0)) + 1)
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
