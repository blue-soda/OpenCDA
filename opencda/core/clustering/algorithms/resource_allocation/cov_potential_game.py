# -*- coding: utf-8 -*-
"""COV utility potential-game scheduler for SGCP.

This scheduler keeps the successful two-stage PAPG resource-allocation
structure, but makes the paper-level utility explicit:

    Delta U = Delta C + Delta O + Delta V - L

where C is cluster-head observability completion, O is object/prototype
evidence, V is multi-view complementarity, and L is communication/link cost.
The implementation intentionally stays close to PAPG so that the new narrative
can be validated without perturbing unrelated protocol choices.
"""

import math

from opencda.core.clustering.algorithms.resource_allocation.perception_aware_potential_game import (
    PerceptionAwarePotentialGame,
)
from opencda.core.clustering.utils import common


class COVPotentialGame(PerceptionAwarePotentialGame):
    """Perception utility game with explicit C/O/V/L components."""

    def __init__(self, cav_world):
        super(COVPotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'cov_utility'
        self.last_utility_breakdown = {}

        # Fixed protocol coefficients.  These are deliberately not exposed as
        # experiment-facing hyperparameters; the two-stage game changes the
        # emphasis between coverage and target refinement instead.
        self.coverage_view_regularizer = 0.25
        self.object_component_bonus = 0.55

    @staticmethod
    def _vehicle_xy(vehicle_id):
        vehicle = common.global_vehicles[int(vehicle_id)]
        location = vehicle.get_position().location
        return float(location.x), float(location.y)

    def _view_geometry_gain(self, cluster, member_id, grid_id):
        """Approximate complementary viewpoint from sender/head/grid geometry."""
        grid_size = common.global_vehicles[int(member_id)].grid_size
        center = self.grid_center(grid_id, grid_size)
        if center is None:
            return 0.0
        hx, hy = self._vehicle_xy(cluster.head_id)
        mx, my = self._vehicle_xy(member_id)
        gh = (hx - center[0], hy - center[1])
        gm = (mx - center[0], my - center[1])
        norm_h = math.hypot(gh[0], gh[1])
        norm_m = math.hypot(gm[0], gm[1])
        if norm_h <= 1e-6 or norm_m <= 1e-6:
            return 0.0
        cos_angle = (gh[0] * gm[0] + gh[1] * gm[1]) / (norm_h * norm_m)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        # 0 for same direction, 1 for opposite-side complementary views.
        angle_gain = (1.0 - cos_angle) * 0.5
        baseline = math.hypot(hx - mx, hy - my)
        baseline_gain = min(1.0, baseline / 35.0)
        return angle_gain * baseline_gain

    def _link_cost(self, cluster, member_id, member_grid_density):
        """Small normalized cost term for distance and payload pressure."""
        hx, hy = self._vehicle_xy(cluster.head_id)
        mx, my = self._vehicle_xy(member_id)
        distance_cost = min(1.0, math.hypot(hx - mx, hy - my) / 100.0)
        payload_proxy = math.log1p(
            self._point_proxy_from_density(cluster, member_grid_density))
        payload_cost = min(1.0, payload_proxy / 8.0)
        return 0.08 * distance_cost + 0.03 * payload_cost

    def grid_utility_components(self, cluster, grid_id, member_id,
                                member_grid_density):
        """Return explicit C/O/V/L terms for one sender-grid action."""
        if member_grid_density <= 0:
            return {
                'coverage': 0.0,
                'object': 0.0,
                'view': 0.0,
                'cost': 0.0,
                'utility': 0.0,
            }

        head_id = int(cluster.head_id)
        head_vehicle = common.global_vehicles[head_id]
        rho_th = max(float(head_vehicle.rho_th), 1e-6)
        head_density = self._vehicle_density(head_id, grid_id)
        member_quality = self.grid_utility_density(
            member_grid_density,
            rho_th)
        head_quality = self.grid_utility_density(head_density, rho_th)
        head_gap = max(0.0, 1.0 - head_quality)

        coverage_gain = member_quality * (1.0 + head_gap)

        visible_count = self._visible_member_count(
            cluster,
            grid_id,
            density_floor=0.0)
        own_best, second_best = self._best_member_density(
            cluster,
            member_id,
            grid_id)
        del own_best
        floor = self._density_floor(member_id)
        uniqueness_gain = 0.0
        if member_grid_density >= max(floor, second_best * 1.35):
            uniqueness_gain = 0.35 * member_quality
        point_gain = 0.04 * math.log1p(
            self._point_proxy_from_density(cluster, member_grid_density))
        consensus_gain = 0.08 * min(visible_count, 4) * member_quality
        object_gain = member_quality + uniqueness_gain + point_gain

        # Multi-view is deliberately not identical to object relevance: it has
        # a same-grid confirmation term plus an explicit viewpoint-geometry
        # proxy.  This keeps the paper C/O/V decomposition defensible.
        confirmation_gain = 0.0
        if head_density > 0.0 and member_grid_density >= floor:
            confirmation_gain = 0.55 * member_quality
            if head_density >= rho_th:
                confirmation_gain *= 1.35
        geometry_gain = self._view_geometry_gain(cluster, member_id, grid_id)
        view_gain = (
            confirmation_gain +
            consensus_gain +
            0.35 * geometry_gain * member_quality)

        cost = self._link_cost(cluster, member_id, member_grid_density)
        utility = coverage_gain + object_gain + view_gain - cost
        return {
            'coverage': coverage_gain,
            'object': object_gain,
            'view': view_gain,
            'cost': cost,
            'utility': max(0.0, utility),
        }

    def grid_score(self, cluster, grid_id, member_grid_density):
        member_id = getattr(self, '_scoring_member_id', None)
        if member_id is None:
            # Fallback for inherited paths that score a grid without an active
            # sender.  Use the parent PAPG-compatible score to preserve safety.
            return super(COVPotentialGame, self).grid_score(
                cluster,
                grid_id,
                member_grid_density)
        components = self.grid_utility_components(
            cluster,
            grid_id,
            member_id,
            member_grid_density)
        self.last_utility_breakdown[(int(cluster.head_id),
                                     int(member_id),
                                     str(grid_id))] = components
        return components['utility']

    def _score_candidate(self, cluster, member_id, mode='coverage'):
        candidates = self.refinement_candidates(cluster, member_id)
        candidates = set(candidates)
        if not candidates:
            return None
        scores = {}
        component_sums = {
            'coverage': 0.0,
            'object': 0.0,
            'view': 0.0,
            'cost': 0.0,
        }
        self._scoring_member_id = int(member_id)
        try:
            for grid_id in candidates:
                density = common.global_vehicles[
                    member_id].grid_density_dict.get(grid_id, 0.0)
                components = self.grid_utility_components(
                    cluster,
                    grid_id,
                    member_id,
                    density)
                scores[grid_id] = components['utility']
                for name in component_sums:
                    component_sums[name] += components[name]
        finally:
            self._scoring_member_id = None

        selected = self.sort_member_grids(
            cluster,
            member_id,
            candidates,
            scores,
            min(self.max_grids_per_rb, len(candidates)))
        if not selected:
            return None

        selected_components = {
            'coverage': 0.0,
            'object': 0.0,
            'view': 0.0,
            'cost': 0.0,
        }
        for grid_id in selected:
            density = common.global_vehicles[
                member_id].grid_density_dict.get(grid_id, 0.0)
            components = self.grid_utility_components(
                cluster,
                grid_id,
                member_id,
                density)
            for name in selected_components:
                selected_components[name] += components[name]

        sorted_scores = sorted(scores.values(), reverse=True)
        top = sorted_scores[:max(1, min(self.max_grids_per_rb,
                                        len(sorted_scores)))]
        cover = sorted_scores[:max(1, min(self.max_grids_per_rb * 2,
                                          len(sorted_scores)))]
        peak = top[0] if top else 0.0
        components = self._connected_components([
            grid for grid in candidates
            if self._vehicle_density(member_id, grid) >=
            self._density_floor(member_id)
        ])
        component_scores = sorted(
            (self._component_score(component, scores)
             for component in components),
            reverse=True)

        object_view_score = (
            selected_components['object'] +
            selected_components['view'] +
            self.object_component_bonus * sum(component_scores[:3]))
        coverage_score = (
            0.65 * sum(top) +
            0.28 * sum(cover) +
            0.05 * min(len(candidates), self.max_grids_per_rb * 2) +
            selected_components['coverage'] -
            selected_components['cost'])

        if mode == 'target':
            score = 0.72 * object_view_score + 0.28 * coverage_score
        else:
            score = 0.68 * coverage_score + 0.32 * object_view_score
        return {
            'member_id': member_id,
            'score': score,
            'coverage_score': coverage_score,
            'object_score': object_view_score,
            'selected': selected,
            'candidate_count': len(candidates),
            'peak': peak,
            'cov_coverage': selected_components['coverage'],
            'cov_object': selected_components['object'],
            'cov_view': selected_components['view'],
            'cov_cost': selected_components['cost'],
        }
