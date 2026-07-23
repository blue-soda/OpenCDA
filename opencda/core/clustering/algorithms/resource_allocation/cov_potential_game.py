# -*- coding: utf-8 -*-
"""COV utility potential-game scheduler for SGCP.

The scheduler uses a paper-facing utility with the same terms as the COV
coalition game.  For a sender-grid action ``(i, g)`` toward cluster head ``h``:

    U(i, g | h) = C(i, g | h) + O(i, g) + V(i, g | h) - L(i, h, g)

where ``C`` is the head's observability gap filled by the sender, ``O`` is the
sender's object-evidence quality, ``V`` is the multi-view confirmation between
sender and head, and ``L`` is normalized communication cost.
"""

import math
import os

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
        self.active_terms = self._parse_terms(
            os.environ.get('OPENCDA_COV_SCHEDULER_TERMS',
                           'coverage,object,view,cost'))

    @staticmethod
    def _parse_terms(raw_terms):
        aliases = {
            'c': 'coverage',
            'coverage': 'coverage',
            'o': 'object',
            'object': 'object',
            'v': 'view',
            'view': 'view',
            'l': 'cost',
            'cost': 'cost',
        }
        terms = set()
        for item in str(raw_terms).replace('+', ',').split(','):
            key = item.strip().lower()
            if key:
                terms.add(aliases.get(key, key))
        if not terms:
            return {'coverage', 'object', 'view', 'cost'}
        return terms

    @staticmethod
    def _vehicle_xy(vehicle_id):
        vehicle = common.global_vehicles[int(vehicle_id)]
        location = vehicle.get_position().location
        return float(location.x), float(location.y)

    def _link_cost(self, cluster, member_id, member_grid_density):
        """Normalized communication cost for one sender-grid action."""
        hx, hy = self._vehicle_xy(cluster.head_id)
        mx, my = self._vehicle_xy(member_id)
        distance_cost = min(1.0, math.hypot(hx - mx, hy - my) / 100.0)
        head_vehicle = common.global_vehicles[int(cluster.head_id)]
        rho_th = max(float(head_vehicle.rho_th), 1e-6)
        payload_cost = self.grid_utility_density(member_grid_density, rho_th)
        grid_budget = max(1.0, float(getattr(self, 'max_grids_per_rb', 1)))
        return distance_cost * payload_cost / grid_budget

    def _compose_utility(self, components):
        utility = 0.0
        if 'coverage' in self.active_terms:
            utility += components['coverage']
        if 'object' in self.active_terms:
            utility += components['object']
        if 'view' in self.active_terms:
            utility += components['view']
        if 'cost' in self.active_terms:
            utility -= components['cost']
        return max(0.0, utility)

    def _component_score(self, component, member_grid_scores):
        """Object-region evidence as the sum of the best grids in a component."""
        if not component:
            return 0.0
        scores = sorted(
            (member_grid_scores.get(grid_id, 0.0) for grid_id in component),
            reverse=True)
        return sum(scores[:max(1, min(self.max_grids_per_rb, len(scores)))])

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
        coverage_gain = member_quality * max(0.0, 1.0 - head_quality)
        object_gain = member_quality
        view_gain = member_quality if head_quality > 0.0 else 0.0

        cost = self._link_cost(cluster, member_id, member_grid_density)
        components = {
            'coverage': coverage_gain,
            'object': object_gain,
            'view': view_gain,
            'cost': cost,
        }
        utility = self._compose_utility(components)
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
        top_scores = sorted_scores[:max(1, min(self.max_grids_per_rb,
                                               len(sorted_scores)))]
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
            selected_components['view'] -
            selected_components['cost'] +
            sum(component_scores[:3]))
        coverage_score = (
            selected_components['coverage'] -
            selected_components['cost'] +
            sum(top_scores))
        full_score = (
            selected_components['coverage'] +
            selected_components['object'] +
            selected_components['view'] -
            selected_components['cost'])

        if mode == 'target':
            score = object_view_score
        else:
            score = coverage_score
        if score <= 0.0:
            score = full_score
        return {
            'member_id': member_id,
            'score': score,
            'coverage_score': coverage_score,
            'object_score': object_view_score,
            'selected': selected,
            'candidate_count': len(candidates),
            'peak': max(scores.values()) if scores else 0.0,
            'cov_coverage': selected_components['coverage'],
            'cov_object': selected_components['object'],
            'cov_view': selected_components['view'],
            'cov_cost': selected_components['cost'],
        }
