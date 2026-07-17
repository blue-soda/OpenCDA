# -*- coding: utf-8 -*-
"""Target-aware SGCP potential-game scheduler.

This variant keeps the original SGCP resource-allocation structure: cluster
heads still play a constrained best-response game over sender, subchannel and
grid actions. The utility, however, no longer treats a grid as saturated once a
single view reaches rho_th. It preserves marginal value for high-density grids
that can benefit from another CAV view, which is the main failure mode found by
the object-level diagnostics.
"""

import math

from opencda.core.clustering.algorithms.resource_allocation.potential_game import (
    PotentialGame,
)
from opencda.core.clustering.utils import common


class TargetAwarePotentialGame(PotentialGame):
    """Potential-guided scheduler with target-coverage-aware grid utility."""

    def __init__(self, cav_world):
        super(TargetAwarePotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'target_aware'
        self._scheduling_stage = False

    def run(self):
        self.clear_resource_allocation_strategy()
        self._scheduling_stage = True
        ret = self.channel_game()
        self._scheduling_stage = False
        self.refine_grid_actions()
        self.update_resource_allocation_strategy()
        return ret

    @staticmethod
    def _density(vehicle_id, grid_id):
        return common.global_vehicles[vehicle_id].grid_density_dict.get(
            grid_id, 0.0)

    @staticmethod
    def _point_proxy(vehicle_id, grid_id):
        vehicle = common.global_vehicles[vehicle_id]
        grid_area = vehicle.grid_size * vehicle.grid_size
        return TargetAwarePotentialGame._density(vehicle_id, grid_id) * grid_area

    def candidate_grids_for_cluster(self, cluster):
        """Keep target-like grids eligible even when the head has dense points.

        The original potential game removes grids whose accumulated cluster
        density already exceeds rho_th. Diagnostics showed that this can hide
        target grids where the head has many local points but still needs a
        second view for detection. This variant only removes grids that are
        already scheduled for upload to the same head in previous best-response
        updates.
        """
        if self._scheduling_stage:
            return super(TargetAwarePotentialGame,
                         self).candidate_grids_for_cluster(cluster)

        candidate_grids = set(cluster.req_grids)
        for (mid, sc, t, grids) in self.strategies.get(cluster.head_id, []):
            candidate_grids -= set(grids)
        return candidate_grids

    def grid_score(self, cluster, grid_id, member_grid_density):
        if self._scheduling_stage:
            return super(TargetAwarePotentialGame, self).grid_score(
                cluster,
                grid_id,
                member_grid_density)
        if member_grid_density <= 0:
            return 0.0

        head_id = cluster.head_id
        head_vehicle = common.global_vehicles[head_id]
        rho_th = max(float(head_vehicle.rho_th), 1e-6)
        head_density = self._density(head_id, grid_id)

        density_score = self.grid_utility_density(member_grid_density, rho_th)
        head_gap = max(0.0, rho_th - head_density) / rho_th

        # Multi-view value remains non-zero after rho_th. This is the key
        # algorithmic change: a dense single view is not assumed to be a
        # detector-sufficient target observation.
        multiview_bonus = 0.45 * density_score
        if head_density >= rho_th:
            multiview_bonus *= 1.35

        visible_members = 0
        cluster_density = head_density
        for member_id in cluster.members:
            density = self._density(member_id, grid_id)
            if density > 0:
                visible_members += 1
                cluster_density += density if member_id != head_id else 0.0

        consensus_bonus = 0.08 * min(visible_members, 4) * density_score
        raw_point_bonus = 0.03 * math.log1p(
            self._point_proxy(head_id, grid_id) +
            self._point_proxy_from_density(cluster, member_grid_density))

        base_gain = super(TargetAwarePotentialGame, self).grid_score(
            cluster,
            grid_id,
            member_grid_density)
        coverage_gain = density_score * (1.0 + head_gap)

        # Diminishing penalty prevents one extremely dense grid family from
        # consuming every selected slot while still preserving high-density
        # target regions.
        saturation_penalty = 1.0 / (1.0 + max(0.0, cluster_density - rho_th) /
                                    (6.0 * rho_th))
        saturation_penalty = max(0.65, saturation_penalty)
        return (
            0.35 * base_gain +
            (coverage_gain + multiview_bonus + consensus_bonus +
             raw_point_bonus) * saturation_penalty)

    @staticmethod
    def _point_proxy_from_density(cluster, density):
        return density * cluster.grid_bits / 128.0

    def member_score(self, cluster, member_id, member_grids,
                     member_grid_scores):
        if self._scheduling_stage:
            return super(TargetAwarePotentialGame, self).member_score(
                cluster,
                member_id,
                member_grids,
                member_grid_scores)
        if not member_grids:
            return 0.0
        sorted_scores = sorted(
            (member_grid_scores[grid] for grid in member_grids),
            reverse=True)
        top_scores = sorted_scores[:max(1, int(self.max_grids_per_rb))]
        cover_scores = sorted_scores[:max(1, int(self.max_grids_per_rb * 2))]
        member_vehicle = common.global_vehicles[member_id]
        high_density_count = sum(
            1 for grid in member_grids
            if member_vehicle.grid_density_dict.get(grid, 0.0) >=
            member_vehicle.rho_th)
        return (
            0.70 * sum(top_scores) +
            0.30 * sum(cover_scores) +
            0.03 * len(member_grids) +
            0.04 * min(high_density_count, self.max_grids_per_rb))

    def sort_member_grids(self, cluster, member_id, member_grids,
                          member_grid_scores, max_grids):
        if self._scheduling_stage:
            return super(TargetAwarePotentialGame, self).sort_member_grids(
                cluster,
                member_id,
                member_grids,
                member_grid_scores,
                max_grids)
        if max_grids <= 0:
            return []
        grid_size = common.global_vehicles[member_id].grid_size
        pool_size = max(max_grids * 3, max_grids + 12)
        remaining = sorted(
            member_grids,
            key=lambda grid: (member_grid_scores[grid], str(grid)),
            reverse=True)[:pool_size]
        selected = []

        while remaining and len(selected) < max_grids:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            selected_centers = [
                self.grid_center(grid, grid_size) for grid in selected
            ]
            selected_centers = [
                center for center in selected_centers if center is not None
            ]

            def diversified_score(grid_id):
                center = self.grid_center(grid_id, grid_size)
                if center is None or not selected_centers:
                    min_distance = 0.0
                else:
                    min_distance = min(
                        (center[0] - selected_center[0]) ** 2 +
                        (center[1] - selected_center[1]) ** 2
                        for selected_center in selected_centers)
                return (
                    member_grid_scores[grid_id] *
                    (1.0 + min_distance / 10000.0),
                    str(grid_id))

            best_grid = max(remaining, key=diversified_score)
            selected.append(best_grid)
            remaining.remove(best_grid)
        return selected

    def refinement_candidates(self, cluster, member_id):
        head_vehicle = common.global_vehicles[cluster.head_id]
        member_vehicle = common.global_vehicles[member_id]
        weak_head_grids = head_vehicle.req_grids - head_vehicle.high_density_grids
        target_like_grids = (
            member_vehicle.high_density_grids &
            head_vehicle.req_grids)
        candidates = (
            member_vehicle.sens_grids &
            (weak_head_grids | target_like_grids))
        if not candidates:
            candidates = member_vehicle.sens_grids & head_vehicle.req_grids
        if not candidates:
            candidates = member_vehicle.sens_grids
        return candidates

    def refine_grid_actions(self):
        refined = {}
        cluster_by_head = {
            int(cluster.head_id): cluster for cluster in self.clusters
        }
        for head_id, links in self.strategies.items():
            cluster = cluster_by_head.get(int(head_id))
            if cluster is None:
                refined[head_id] = links
                continue
            refined_links = []
            for member_id, subchannel, time_slot, grids in links:
                count = len(grids)
                candidates = self.refinement_candidates(cluster, member_id)
                if not candidates:
                    refined_links.append((member_id, subchannel, time_slot,
                                          grids))
                    continue
                scores = {}
                for grid_id in candidates:
                    density = common.global_vehicles[
                        member_id].grid_density_dict.get(grid_id, 0.0)
                    scores[grid_id] = self.grid_score(
                        cluster,
                        grid_id,
                        density)
                selected = self.sort_member_grids(
                    cluster,
                    member_id,
                    candidates,
                    scores,
                    min(count, len(candidates)))
                refined_links.append((member_id, subchannel, time_slot,
                                      selected or grids))
            refined[head_id] = refined_links
        self.strategies = refined
