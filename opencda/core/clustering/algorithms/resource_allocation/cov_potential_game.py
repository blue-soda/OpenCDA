# -*- coding: utf-8 -*-
"""Clean two-stage potential-game scheduler for SGCP.

This scheduler intentionally keeps the objective simple and auditable:

* coverage stage scores a sender-grid action only by ``C``;
* target/quality stage scores it only by the configured quality term;
* candidate grids are those with positive ``C + quality``.

The coalition game remains V-only by default, so cluster formation keeps the
validated multi-view grouping behavior while this scheduler cleanly separates
coverage recovery from quality refinement.  The default quality term is ``V``;
set ``OPENCDA_COV_TARGET_TERM=object`` for the clean C-then-O ablation.
"""

import os

from opencda.core.clustering.algorithms.resource_allocation.perception_aware_potential_game import (
    PerceptionAwarePotentialGame,
)
from opencda.core.clustering.utils import common


class COVPotentialGame(PerceptionAwarePotentialGame):
    """Two-stage scheduler with explicit coverage and quality utilities."""

    def __init__(self, cav_world):
        super(COVPotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'cov_utility'
        self.last_utility_breakdown = {}
        self.target_term = os.environ.get(
            'OPENCDA_COV_TARGET_TERM',
            'view').strip().lower()
        if self.target_term not in {'view', 'object'}:
            self.target_term = 'view'

    def grid_utility_components(self, cluster, grid_id, member_id,
                                member_grid_density):
        """Return C/O/V terms for one sender-grid action."""
        if member_grid_density <= 0:
            return {
                'coverage': 0.0,
                'object': 0.0,
                'view': 0.0,
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
        coverage_gain = member_quality * max(0.0, 1.0 - head_quality)
        object_gain = member_quality
        view_gain = member_quality if head_quality > 0.0 else 0.0
        quality_gain = (
            object_gain if self.target_term == 'object' else view_gain)

        return {
            'coverage': coverage_gain,
            'object': object_gain,
            'view': view_gain,
            'utility': max(0.0, coverage_gain + quality_gain),
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

    def _stage_grid_score(self, components, mode):
        if mode == 'coverage':
            return components['coverage']
        return components[self.target_term]

    def _select_stage_grids(self, cluster, member_id, candidates,
                            component_by_grid, mode):
        """Select top grids by the current stage objective only.

        The inherited object-aware selector contains connected-component and
        spatial-diversity logic.  This clean C/V scheduler deliberately avoids
        that path so the stage objectives are auditable.
        """
        del cluster
        max_grids = min(self.max_grids_per_rb, len(candidates))
        ranked = []
        for grid_id in candidates:
            components = component_by_grid[grid_id]
            stage_score = self._stage_grid_score(components, mode)
            if stage_score <= 0.0:
                continue
            ranked.append((
                stage_score,
                self._vehicle_density(member_id, grid_id),
                str(grid_id),
                grid_id))
        ranked.sort(reverse=True)
        return [grid_id for _, _, _, grid_id in ranked[:max_grids]]

    def _score_candidate(self, cluster, member_id, mode='coverage'):
        raw_candidates = set(self.refinement_candidates(cluster, member_id))
        if not raw_candidates:
            return None
        candidates = set()
        component_by_grid = {}
        self._scoring_member_id = int(member_id)
        try:
            for grid_id in raw_candidates:
                density = common.global_vehicles[
                    member_id].grid_density_dict.get(grid_id, 0.0)
                components = self.grid_utility_components(
                    cluster,
                    grid_id,
                    member_id,
                    density)
                if components['coverage'] + components['view'] <= 0.0:
                    continue
                candidates.add(grid_id)
                component_by_grid[grid_id] = components
        finally:
            self._scoring_member_id = None
        if not candidates:
            return None

        selected = self._select_stage_grids(
            cluster,
            member_id,
            candidates,
            component_by_grid,
            mode)
        if not selected:
            return None

        selected_components = {
            'coverage': 0.0,
            'object': 0.0,
            'view': 0.0,
        }
        for grid_id in selected:
            components = component_by_grid[grid_id]
            for name in selected_components:
                selected_components[name] += components[name]

        coverage_score = selected_components['coverage']
        quality_score = selected_components[self.target_term]

        if mode == 'target':
            score = quality_score
        else:
            score = coverage_score
        if score <= 0.0:
            return None
        return {
            'member_id': member_id,
            'score': score,
            'coverage_score': coverage_score,
            'object_score': quality_score,
            'selected': selected,
            'candidate_count': len(candidates),
            'peak': max(
                self._stage_grid_score(component_by_grid[grid_id], mode)
                for grid_id in candidates),
            'cov_coverage': selected_components['coverage'],
            'cov_object': selected_components['object'],
            'cov_view': selected_components['view'],
            'cov_cost': 0.0,
            'target_term': self.target_term,
        }
