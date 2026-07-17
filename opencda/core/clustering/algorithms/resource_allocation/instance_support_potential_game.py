# -*- coding: utf-8 -*-
"""Instance-support-aware SGCP resource scheduler.

This branch extends ``PerceptionAwarePotentialGame`` after object-level
diagnostics showed that many missed objects are not empty-grid failures. The
critical gap is often that the best raw object-supporting CAV is not uploaded
to the relevant cluster head. Without using ground-truth boxes, this scheduler
approximates that signal with compact high-density grid components, weak-head
gain, and unique-best-view gain.
"""

import math

from opencda.core.clustering.algorithms.resource_allocation.perception_aware_potential_game import (
    PerceptionAwarePotentialGame,
)
from opencda.core.clustering.utils import common


class InstanceSupportPotentialGame(PerceptionAwarePotentialGame):
    """PAPG with an explicit instance-support utility term."""

    def __init__(self, cav_world):
        super(InstanceSupportPotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'instance_support'
        self.instance_density_ratio = 0.18
        self.unique_view_margin = 1.25

    def _instance_floor(self, member_id):
        vehicle = common.global_vehicles[member_id]
        return max(0.05, self.instance_density_ratio * vehicle.rho_th)

    def _peer_best_density(self, cluster, member_id, grid_id):
        best = 0.0
        for peer_id in cluster.members:
            if peer_id in (member_id, cluster.head_id):
                continue
            best = max(best, self._vehicle_density(peer_id, grid_id))
        return best

    def _instance_grid_score(self, cluster, member_id, grid_id,
                             base_score=None):
        density = self._vehicle_density(member_id, grid_id)
        if density <= 0:
            return 0.0
        head_density = self._vehicle_density(cluster.head_id, grid_id)
        head_vehicle = common.global_vehicles[cluster.head_id]
        rho_th = max(float(head_vehicle.rho_th), 1e-6)
        member_quality = self.grid_utility_density(density, rho_th)
        head_gap = max(0.0, rho_th - head_density) / rho_th
        peer_best = self._peer_best_density(cluster, member_id, grid_id)
        unique_gain = 0.0
        if density >= max(self._instance_floor(member_id),
                          peer_best * self.unique_view_margin):
            unique_gain = member_quality
        compact_point_gain = 0.04 * math.log1p(
            self._point_proxy_from_density(cluster, density))
        base_score = 0.0 if base_score is None else base_score
        return (
            0.45 * base_score +
            0.85 * member_quality * (1.0 + 0.75 * head_gap) +
            0.42 * unique_gain +
            compact_point_gain)

    def _component_instance_score(self, cluster, member_id, component,
                                  grid_scores):
        if not component:
            return 0.0
        instance_scores = [
            self._instance_grid_score(
                cluster,
                member_id,
                grid,
                base_score=grid_scores.get(grid, 0.0))
            for grid in component
        ]
        instance_scores = sorted(instance_scores, reverse=True)
        compactness = 1.0 / (1.0 + max(0, len(component) - 4) / 8.0)
        return (
            instance_scores[0] +
            0.55 * sum(instance_scores[1:4]) +
            0.12 * sum(instance_scores[4:8])) * compactness

    def _candidate_instance_score(self, cluster, member_id, candidates,
                                  grid_scores):
        floor = self._instance_floor(member_id)
        support_grids = [
            grid for grid in candidates
            if self._vehicle_density(member_id, grid) >= floor
        ]
        if not support_grids:
            return 0.0
        components = self._connected_components(support_grids)
        component_scores = sorted(
            (self._component_instance_score(
                cluster,
                member_id,
                component,
                grid_scores)
             for component in components),
            reverse=True)
        unique_component_bonus = 0.0
        for grid in support_grids:
            density = self._vehicle_density(member_id, grid)
            peer_best = self._peer_best_density(cluster, member_id, grid)
            if density >= max(floor, peer_best * self.unique_view_margin):
                unique_component_bonus += 0.03
        return (
            sum(component_scores[:3]) +
            min(unique_component_bonus, 0.35))

    def _score_candidate(self, cluster, member_id, mode='coverage'):
        candidates = set(self.refinement_candidates(cluster, member_id))
        if not candidates:
            return None
        scores = {}
        instance_scores = {}
        for grid_id in candidates:
            density = common.global_vehicles[
                member_id].grid_density_dict.get(grid_id, 0.0)
            base_score = self.grid_score(cluster, grid_id, density)
            scores[grid_id] = base_score
            instance_scores[grid_id] = self._instance_grid_score(
                cluster,
                member_id,
                grid_id,
                base_score=base_score)
        selected = self.sort_member_grids(
            cluster,
            member_id,
            candidates,
            scores,
            min(self.max_grids_per_rb, len(candidates)))
        if not selected:
            return None

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
        object_score = peak + 0.55 * sum(component_scores[:3])
        coverage_score = (
            0.65 * sum(top) +
            0.28 * sum(cover) +
            0.05 * min(len(candidates), self.max_grids_per_rb * 2))
        instance_score = self._candidate_instance_score(
            cluster,
            member_id,
            candidates,
            scores)

        if mode == 'target':
            score = (
                0.48 * object_score +
                0.32 * instance_score +
                0.20 * coverage_score)
        else:
            score = (
                0.58 * coverage_score +
                0.25 * object_score +
                0.17 * instance_score)
        return {
            'member_id': member_id,
            'score': score,
            'coverage_score': coverage_score,
            'object_score': object_score,
            'instance_score': instance_score,
            'selected': selected,
            'candidate_count': len(candidates),
            'peak': peak,
        }

    def _best_member_link(self, cluster, mode='coverage',
                          excluded_members=None):
        excluded_members = excluded_members or set()
        best = None
        for member_id in sorted(cluster.members):
            if member_id == cluster.head_id or member_id in excluded_members:
                continue
            candidate = self._score_candidate(cluster, member_id, mode=mode)
            if candidate is None:
                continue
            if best is None:
                best = candidate
                continue
            key = (
                candidate['score'],
                candidate['instance_score'],
                candidate['object_score'],
                candidate['coverage_score'],
                -candidate['member_id'])
            best_key = (
                best['score'],
                best['instance_score'],
                best['object_score'],
                best['coverage_score'],
                -best['member_id'])
            if key > best_key:
                best = candidate
        return best

    def sort_member_grids(self, cluster, member_id, member_grids,
                          member_grid_scores, max_grids):
        if max_grids <= 0:
            return []
        member_grids = set(member_grids)
        if not member_grids:
            return []

        floor = self._instance_floor(member_id)
        support_grids = {
            grid for grid in member_grids
            if self._vehicle_density(member_id, grid) >= floor
        }
        components = self._connected_components(support_grids)
        components = sorted(
            components,
            key=lambda component: (
                self._component_instance_score(
                    cluster,
                    member_id,
                    component,
                    member_grid_scores),
                len(component),
                str(component[0]) if component else ''),
            reverse=True)

        selected = []
        selected_set = set()
        for component in components:
            if len(selected) >= max_grids:
                break
            component_sorted = sorted(
                component,
                key=lambda grid: (
                    self._instance_grid_score(
                        cluster,
                        member_id,
                        grid,
                        base_score=member_grid_scores.get(grid, 0.0)),
                    member_grid_scores.get(grid, 0.0),
                    str(grid)),
                reverse=True)
            for grid in component_sorted[:3]:
                if grid in selected_set or grid not in member_grids:
                    continue
                selected.append(grid)
                selected_set.add(grid)
                if len(selected) >= max_grids:
                    break

        remaining = sorted(
            [grid for grid in member_grids if grid not in selected_set],
            key=lambda grid: (
                member_grid_scores.get(grid, 0.0),
                self._vehicle_density(member_id, grid),
                str(grid)),
            reverse=True)[:max(max_grids * 4, max_grids + 16)]
        while remaining and len(selected) < max_grids:
            selected.append(remaining.pop(0))
        return selected
