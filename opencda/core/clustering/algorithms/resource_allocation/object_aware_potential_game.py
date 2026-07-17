# -*- coding: utf-8 -*-
"""Object-aware SGCP potential-game scheduler.

This allocator keeps the SGCP potential-guided resource-allocation interface:
cluster heads still choose sender/subchannel/grid actions under the same RB
budget. The action utility is changed from grid-density accumulation to
object-prototype coverage. A prototype is approximated online from connected
high-density or multi-view-confirmed LiDAR grids, so the method does not depend
on ground-truth boxes.
"""

import math

from opencda.core.clustering.algorithms.resource_allocation.target_aware_potential_game import (
    TargetAwarePotentialGame,
)
from opencda.core.clustering.utils import common


class ObjectAwarePotentialGame(TargetAwarePotentialGame):
    """Potential-guided scheduler with object-prototype-aware actions."""

    def __init__(self, cav_world):
        super(ObjectAwarePotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'object_aware'
        self.prototype_density_ratio = 0.25
        self.prototype_pool_factor = 4
        self.sender_replacement_margin = 1.03
        self.sender_peak_replacement_margin = 1.02

    def run(self):
        self.clear_resource_allocation_strategy()
        self._scheduling_stage = True
        ret = self.channel_game()
        self._scheduling_stage = False
        self.refine_sender_actions()
        self.refine_grid_actions()
        self.update_resource_allocation_strategy()
        return ret

    @staticmethod
    def _grid_index(grid_id):
        try:
            return tuple(int(item) for item in str(grid_id).split('_'))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _grid_l1_distance(cls, grid_a, grid_b):
        index_a = cls._grid_index(grid_a)
        index_b = cls._grid_index(grid_b)
        if index_a is None or index_b is None:
            return 999999
        return abs(index_a[0] - index_b[0]) + abs(index_a[1] - index_b[1])

    @staticmethod
    def _vehicle_density(vehicle_id, grid_id):
        return common.global_vehicles[vehicle_id].grid_density_dict.get(
            grid_id, 0.0)

    def _density_floor(self, vehicle_id):
        vehicle = common.global_vehicles[vehicle_id]
        return max(0.05, self.prototype_density_ratio * vehicle.rho_th)

    def candidate_grids_for_cluster(self, cluster):
        """Do not discard dense head grids before member scoring.

        The selected-frame diagnostics showed several misses where the head
        already had high density in the target grid, but the detector still
        needed an external view. Keeping dense grids eligible allows the game to
        value multi-view confirmation and near-head blind-zone repair.
        """
        candidate_grids = set(cluster.req_grids)
        for (mid, sc, t, grids) in self.strategies.get(cluster.head_id, []):
            candidate_grids -= set(grids)
        return candidate_grids

    def _is_object_candidate(self, cluster, member_id, grid_id):
        member_density = self._vehicle_density(member_id, grid_id)
        if member_density <= 0:
            return False
        head_density = self._vehicle_density(cluster.head_id, grid_id)
        floor = self._density_floor(member_id)
        head_rho = max(common.global_vehicles[cluster.head_id].rho_th, 1e-6)

        if member_density >= floor and head_density < head_rho:
            return True
        if member_density >= floor and head_density >= head_rho:
            return True
        if member_density >= common.global_vehicles[member_id].rho_th:
            return True
        return False

    def member_candidate_grids(self, cluster, member_id, candidate_grids,
                               current_head_links=None):
        grids = (
            common.global_vehicles[member_id].sens_grids &
            set(candidate_grids))
        grids = {
            grid for grid in grids
            if self._vehicle_density(member_id, grid) > 0
        }
        if current_head_links is not None:
            for (mid, sc, t, selected) in current_head_links:
                if mid == member_id:
                    grids -= set(selected)
                    break
        return grids

    def refinement_candidates(self, cluster, member_id):
        base_candidates = (
            common.global_vehicles[member_id].sens_grids &
            set(cluster.req_grids))
        positive_candidates = {
            grid for grid in base_candidates
            if self._vehicle_density(member_id, grid) > 0
        }
        if positive_candidates:
            return positive_candidates
        return super(ObjectAwarePotentialGame, self).refinement_candidates(
            cluster,
            member_id)

    def _visible_member_count(self, cluster, grid_id, density_floor=0.0):
        count = 0
        for member_id in cluster.members:
            if self._vehicle_density(member_id, grid_id) > density_floor:
                count += 1
        return count

    def _best_member_density(self, cluster, member_id, grid_id):
        densities = []
        for peer_id in cluster.members:
            if peer_id == cluster.head_id:
                continue
            densities.append(self._vehicle_density(peer_id, grid_id))
        if not densities:
            return 0.0, 0.0
        densities.sort(reverse=True)
        best = densities[0]
        second = densities[1] if len(densities) > 1 else 0.0
        own = self._vehicle_density(member_id, grid_id)
        if own < best:
            return own, best
        return best, second

    def grid_score(self, cluster, grid_id, member_grid_density):
        if member_grid_density <= 0:
            return 0.0

        base_score = super(ObjectAwarePotentialGame, self).grid_score(
            cluster,
            grid_id,
            member_grid_density)
        member_vehicle = common.global_vehicles[cluster.head_id]
        rho_th = max(member_vehicle.rho_th, 1e-6)
        head_density = self._vehicle_density(cluster.head_id, grid_id)

        member_quality = self.grid_utility_density(
            member_grid_density,
            rho_th)
        weak_head_gain = max(0.0, rho_th - head_density) / rho_th
        multiview_gain = 0.0
        if head_density >= rho_th and member_grid_density >= self._density_floor(
                cluster.head_id):
            multiview_gain = 0.9 * member_quality

        visible_count = self._visible_member_count(
            cluster,
            grid_id,
            density_floor=0.0)
        own_best, second_best = self._best_member_density(
            cluster,
            cluster.head_id,
            grid_id)
        del own_best
        uniqueness_gain = 0.0
        if member_grid_density >= second_best * 1.35:
            uniqueness_gain = 0.35 * member_quality

        point_gain = 0.04 * math.log1p(
            self._point_proxy_from_density(cluster, member_grid_density))
        consensus_gain = 0.08 * min(visible_count, 4) * member_quality
        return (
            0.20 * base_score +
            0.95 * member_quality * (1.0 + weak_head_gain) +
            multiview_gain +
            uniqueness_gain +
            consensus_gain +
            point_gain)

    def _component_key(self, grid_id):
        index = self._grid_index(grid_id)
        if index is None:
            return None
        return index

    def _connected_components(self, grids):
        remaining = set(grids)
        components = []
        while remaining:
            seed = min(remaining, key=str)
            remaining.remove(seed)
            queue = [seed]
            component = [seed]
            while queue:
                current = queue.pop()
                idx = self._component_key(current)
                if idx is None:
                    continue
                cx, cy = idx
                neighbor_ids = {
                    '%d_%d' % (cx + dx, cy + dy)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                }
                for neighbor in list(remaining & neighbor_ids):
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
            components.append(component)
        return components

    def _component_score(self, component, member_grid_scores):
        if not component:
            return 0.0
        scores = [member_grid_scores.get(grid, 0.0) for grid in component]
        return max(scores) + 0.35 * sum(sorted(scores, reverse=True)[:4])

    def _score_member_for_cluster(self, cluster, member_id):
        candidates = (
            common.global_vehicles[member_id].sens_grids &
            set(cluster.req_grids))
        candidates = {
            grid for grid in candidates
            if self._vehicle_density(member_id, grid) > 0
        }
        if not candidates:
            return 0.0, 0.0, [], {}
        scores = {}
        for grid_id in candidates:
            density = self._vehicle_density(member_id, grid_id)
            scores[grid_id] = self.grid_score(cluster, grid_id, density)
        member_score = self.member_score(
            cluster,
            member_id,
            candidates,
            scores)
        peak_score = max(scores.values()) if scores else 0.0
        sorted_grids = self.sort_member_grids(
            cluster,
            member_id,
            candidates,
            scores,
            min(self.max_grids_per_rb, len(candidates)))
        return member_score, peak_score, sorted_grids, scores

    def refine_sender_actions(self):
        """Swap an occupied RB to the best object-prototype sender if needed."""
        refined = {}
        cluster_by_head = {
            int(cluster.head_id): cluster for cluster in self.clusters
        }
        for head_id, links in self.strategies.items():
            cluster = cluster_by_head.get(int(head_id))
            if cluster is None:
                refined[head_id] = links
                continue
            new_links = []
            used_members = set()
            for member_id, subchannel, time_slot, grids in links:
                current_score, current_peak, current_sorted, _ = (
                    self._score_member_for_cluster(cluster, member_id))
                best_member = member_id
                best_score = current_score
                best_peak = current_peak
                best_grids = current_sorted or grids
                for candidate_member in sorted(cluster.members):
                    if candidate_member == cluster.head_id:
                        continue
                    if candidate_member in used_members:
                        continue
                    candidate_score, candidate_peak, candidate_grids, _ = (
                        self._score_member_for_cluster(
                            cluster,
                            candidate_member))
                    if not candidate_grids:
                        continue
                    replaces_by_total = (
                        candidate_score >
                        best_score * self.sender_replacement_margin)
                    replaces_by_peak = (
                        candidate_peak >
                        best_peak * self.sender_peak_replacement_margin)
                    if replaces_by_total or replaces_by_peak:
                        best_member = candidate_member
                        best_score = candidate_score
                        best_peak = candidate_peak
                        best_grids = candidate_grids
                count = max(1, len(grids))
                new_links.append((
                    best_member,
                    subchannel,
                    time_slot,
                    list(best_grids)[:count]))
                used_members.add(best_member)
            refined[head_id] = new_links
        self.strategies = refined

    def member_score(self, cluster, member_id, member_grids,
                     member_grid_scores):
        if not member_grids:
            return 0.0
        sorted_scores = sorted(
            (member_grid_scores[grid] for grid in member_grids),
            reverse=True)
        top1 = sorted_scores[:1]
        top3 = sorted_scores[:3]
        top8 = sorted_scores[:8]
        floor = self._density_floor(member_id)
        proto_grids = [
            grid for grid in member_grids
            if self._vehicle_density(member_id, grid) >= floor
        ]
        components = self._connected_components(proto_grids)
        component_scores = sorted(
            (self._component_score(component, member_grid_scores)
             for component in components),
            reverse=True)
        unique_count = 0
        for grid in proto_grids:
            own_density = self._vehicle_density(member_id, grid)
            peer_best = 0.0
            for peer_id in cluster.members:
                if peer_id in (member_id, cluster.head_id):
                    continue
                peer_best = max(
                    peer_best,
                    self._vehicle_density(peer_id, grid))
            if own_density > max(floor, peer_best * 1.35):
                unique_count += 1

        return (
            1.90 * sum(top1) +
            0.95 * sum(top3) +
            0.22 * sum(top8) +
            0.75 * sum(component_scores[:3]) +
            0.05 * min(unique_count, self.max_grids_per_rb) +
            0.01 * len(member_grids))

    def sort_member_grids(self, cluster, member_id, member_grids,
                          member_grid_scores, max_grids):
        if max_grids <= 0:
            return []
        member_grids = set(member_grids)
        if not member_grids:
            return []

        floor = self._density_floor(member_id)
        proto_grids = {
            grid for grid in member_grids
            if self._vehicle_density(member_id, grid) >= floor
        }
        components = self._connected_components(proto_grids)
        components = sorted(
            components,
            key=lambda component: (
                self._component_score(component, member_grid_scores),
                len(component),
                str(component[0])),
            reverse=True)

        selected = []
        selected_set = set()
        for component in components:
            if len(selected) >= max_grids:
                break
            component_sorted = sorted(
                component,
                key=lambda grid: (member_grid_scores.get(grid, 0.0),
                                  self._vehicle_density(member_id, grid),
                                  str(grid)),
                reverse=True)
            for grid in component_sorted[:2]:
                if grid in selected_set or grid not in member_grids:
                    continue
                selected.append(grid)
                selected_set.add(grid)
                if len(selected) >= max_grids:
                    break

        pool_size = max(max_grids * self.prototype_pool_factor,
                        max_grids + 16)
        remaining = sorted(
            [grid for grid in member_grids if grid not in selected_set],
            key=lambda grid: (member_grid_scores.get(grid, 0.0), str(grid)),
            reverse=True)[:pool_size]

        while remaining and len(selected) < max_grids:
            if not selected:
                best_grid = remaining[0]
            else:
                best_grid = max(
                    remaining,
                    key=lambda grid: (
                        member_grid_scores.get(grid, 0.0) *
                        (1.0 + min(
                            self._grid_l1_distance(grid, selected_grid)
                            for selected_grid in selected) / 12.0),
                        str(grid)))
            selected.append(best_grid)
            selected_set.add(best_grid)
            remaining.remove(best_grid)
        return selected
