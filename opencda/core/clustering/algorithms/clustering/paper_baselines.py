# -*- coding: utf-8 -*-
"""Paper-inspired clustering baselines for SGCP offline experiments.

The functions in this module only build cluster membership.  Head election,
raw-LiDAR scheduling, late fusion, channel estimation and detector checkpoints
remain controlled by the shared SGCP evaluation pipeline.
"""

import math
import random

from opencda.core.clustering.utils import common


def _vehicle_xy(vehicle_id):
    vehicle = common.global_vehicles[int(vehicle_id)]
    location = vehicle.get_position().location
    return float(location.x), float(location.y)


def _vehicle_velocity_xy(vehicle_id):
    vehicle = common.global_vehicles[int(vehicle_id)]
    speed = float(vehicle.get_speed())
    direction = vehicle.get_direction()
    return speed * float(direction[0]), speed * float(direction[1])


def _distance(first_id, second_id):
    x1, y1 = _vehicle_xy(first_id)
    x2, y2 = _vehicle_xy(second_id)
    return math.hypot(x1 - x2, y1 - y2)


def _relative_speed(first_id, second_id):
    vx1, vy1 = _vehicle_velocity_xy(first_id)
    vx2, vy2 = _vehicle_velocity_xy(second_id)
    return math.hypot(vx1 - vx2, vy1 - vy2)


def _direction_similarity(first_id, second_id):
    vx1, vy1 = _vehicle_velocity_xy(first_id)
    vx2, vy2 = _vehicle_velocity_xy(second_id)
    n1 = math.hypot(vx1, vy1)
    n2 = math.hypot(vx2, vy2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.5
    cosine = (vx1 * vx2 + vy1 * vy2) / max(n1 * n2, 1e-6)
    return max(0.0, min(1.0, 0.5 + 0.5 * cosine))


def _grid_quality(vehicle_id, grid_id):
    vehicle = common.global_vehicles[int(vehicle_id)]
    rho_th = max(float(getattr(vehicle, 'rho_th', 1.0)), 1e-6)
    density = float(getattr(vehicle, 'grid_density_dict', {}).get(grid_id, 0.0))
    return min(1.0, density / rho_th)


def _sensing_quality(vehicle_id):
    vehicle = common.global_vehicles[int(vehicle_id)]
    sens_grids = getattr(vehicle, 'sens_grids', set()) or set()
    if not sens_grids:
        return 0.0
    return sum(_grid_quality(vehicle_id, grid_id) for grid_id in sens_grids)


def _jaccard(first, second):
    first = set(first)
    second = set(second)
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / float(len(union))


def _center_head_id(member_ids):
    member_ids = [int(item) for item in member_ids]
    if not member_ids:
        return None
    positions = {vid: _vehicle_xy(vid) for vid in member_ids}
    center_x = sum(item[0] for item in positions.values()) / len(member_ids)
    center_y = sum(item[1] for item in positions.values()) / len(member_ids)
    return min(
        member_ids,
        key=lambda vid: (
            math.hypot(positions[vid][0] - center_x,
                       positions[vid][1] - center_y),
            vid))


def _make_cluster(member_ids):
    cluster = common.Cluster(set(int(item) for item in member_ids))
    head_id = _center_head_id(cluster.members)
    if head_id is not None:
        cluster.head_id = int(head_id)
        cluster.grid_bits = cluster.compute_grid_bits()
    return cluster


def _capacity(n_max):
    capacity = int(n_max or common.Params().N_max)
    return max(1, capacity)


def _seac_pair_score(head_id, member_id):
    """Social-aware proximity proxy used by the SeAC adaptation.

    SeAC uses SDN-assisted social similarity and physical proximity.  The
    CARLA dump has no route/social history, so we map that signal to same-frame
    direction similarity, relative speed stability, distance and sensing-field
    overlap.
    """
    head = common.global_vehicles[int(head_id)]
    member = common.global_vehicles[int(member_id)]
    distance_score = math.exp(-_distance(head_id, member_id) / 45.0)
    speed_score = math.exp(-_relative_speed(head_id, member_id) / 8.0)
    direction_score = _direction_similarity(head_id, member_id)
    interest_score = _jaccard(head.sens_grids, member.sens_grids)
    return (
        0.34 * distance_score +
        0.26 * speed_score +
        0.24 * direction_score +
        0.16 * interest_score)


def build_seac_social_adaptive_clusters(world, n_max=None, timestamp=None):
    """Build a SeAC-inspired social/mobility adaptive clustering baseline."""
    common.Vehicle_Grid.initialize(world)
    vehicle_ids = sorted(int(item) for item in common.global_vehicles)
    capacity = _capacity(n_max)
    unassigned = set(vehicle_ids)
    clusters = []
    while unassigned:
        head_id = max(
            unassigned,
            key=lambda vid: (
                sum(_seac_pair_score(vid, other)
                    for other in unassigned if other != vid),
                _sensing_quality(vid),
                -vid))
        members = [head_id]
        candidates = sorted(
            (vid for vid in unassigned if vid != head_id),
            key=lambda vid: (
                -_seac_pair_score(head_id, vid),
                _distance(head_id, vid),
                vid))
        members.extend(candidates[:capacity - 1])
        unassigned -= set(members)
        clusters.append(_make_cluster(members))
    return clusters


def _partition_objective(partition):
    """Multi-objective cluster quality for HHO-style candidate ranking."""
    total = 0.0
    for members in partition:
        if not members:
            continue
        union = set()
        quality = 0.0
        for vid in members:
            vehicle = common.global_vehicles[int(vid)]
            union |= set(vehicle.sens_grids)
            quality += _sensing_quality(vid)
        pair_scores = []
        for i, first_id in enumerate(members):
            for second_id in members[i + 1:]:
                pair_scores.append(
                    0.45 * math.exp(-_distance(first_id, second_id) / 45.0) +
                    0.35 * math.exp(-_relative_speed(first_id, second_id) / 8.0) +
                    0.20 * _direction_similarity(first_id, second_id))
        coherence = sum(pair_scores) / max(1, len(pair_scores))
        total += 0.018 * len(union) + 0.002 * quality + 6.0 * coherence
    return total


def _balanced_random_partition(vehicle_ids, capacity, seed):
    shuffled = list(vehicle_ids)
    random.Random(seed).shuffle(shuffled)
    return [
        shuffled[index:index + capacity]
        for index in range(0, len(shuffled), capacity)
    ]


def _distance_seed_partition(vehicle_ids, capacity):
    unassigned = set(vehicle_ids)
    partition = []
    while unassigned:
        head_id = min(
            unassigned,
            key=lambda vid: (
                sum(_distance(vid, other)
                    for other in unassigned if other != vid),
                vid))
        members = [head_id]
        candidates = sorted(
            (vid for vid in unassigned if vid != head_id),
            key=lambda vid: (_distance(head_id, vid), vid))
        members.extend(candidates[:capacity - 1])
        unassigned -= set(members)
        partition.append(members)
    return partition


def _improve_partition(partition, capacity, rounds=2):
    """Deterministic local transfer step after HHO-like multi-start sampling."""
    partition = [list(cluster) for cluster in partition if cluster]
    best_score = _partition_objective(partition)
    for _ in range(rounds):
        changed = False
        for source_index, source in enumerate(list(partition)):
            for vid in list(source):
                target_indices = [
                    idx for idx, target in enumerate(partition)
                    if idx != source_index and len(target) < capacity
                ]
                for target_index in target_indices:
                    trial = [list(cluster) for cluster in partition]
                    trial[source_index].remove(vid)
                    trial[target_index].append(vid)
                    trial = [cluster for cluster in trial if cluster]
                    score = _partition_objective(trial)
                    if score > best_score + 1e-6:
                        partition = trial
                        best_score = score
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return partition


def build_hho_vanet_clusters(world, n_max=None, timestamp=None):
    """Build a Harris-Hawks-Optimization-inspired VANET clustering baseline.

    The original HHO VANET paper uses a population metaheuristic over cluster
    quality terms.  This deterministic adaptation keeps that spirit through
    multi-start candidate partitions plus a small local transfer phase, using
    proximity, relative mobility and sensing coverage as the objective.
    """
    common.Vehicle_Grid.initialize(world)
    vehicle_ids = sorted(int(item) for item in common.global_vehicles)
    capacity = _capacity(n_max)
    if not vehicle_ids:
        return []
    seed_prefix = str(timestamp or 'all')
    candidates = [_distance_seed_partition(vehicle_ids, capacity)]
    for index in range(7):
        candidates.append(
            _balanced_random_partition(
                vehicle_ids,
                capacity,
                '%s:hho:%d' % (seed_prefix, index)))
    candidates = [_improve_partition(item, capacity) for item in candidates]
    best = max(candidates, key=_partition_objective)
    return [_make_cluster(members) for members in best]


def build_paper_baseline_clusters(world, clustering, n_max=None, timestamp=None):
    if clustering == 'seac_social_adaptive':
        return build_seac_social_adaptive_clusters(
            world,
            n_max=n_max,
            timestamp=timestamp)
    if clustering == 'hho_vanet':
        return build_hho_vanet_clusters(
            world,
            n_max=n_max,
            timestamp=timestamp)
    raise ValueError('Unknown paper baseline clustering: %s' % clustering)
