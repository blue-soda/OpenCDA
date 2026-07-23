# -*- coding: utf-8 -*-
"""COV utility coalition game for SGCP.

The original coalition game forms stable local groups, but its utility is
written as grid-overlap score times stability.  This variant keeps the same
coalition-formation mechanics and head-election policy while rewriting the
vehicle-level marginal contribution with the same C/O/V/L perception utility
used by the new scheduler:

    Delta U_i(S) = C_i(S) + O_i(S) + V_i(S) - L_i(S)

where S is the candidate coalition.  The terms are expected utilities at the
vehicle/coalition level, while the resource scheduler later computes realized
utilities for concrete sender-grid-channel actions.
"""

import math

from opencda.core.clustering.algorithms.clustering.coalition_game import (
    CoalitionGame,
)
from opencda.core.clustering.utils import common


class COVCoalitionGame(CoalitionGame):
    """Coalition formation driven by expected C/O/V/L utility."""

    def __init__(self, cav_world):
        super(COVCoalitionGame, self).__init__(cav_world)
        self.last_utility_breakdown = {}

        # Fixed coefficients used only to aggregate normalized C/O/V/L terms.
        # They are not exposed as paper-facing hyperparameters.
        self.coverage_weight = 0.00
        self.object_weight = 0.00
        self.view_weight = 1.00
        self.cost_weight = 0.00

    @staticmethod
    def _vehicle_xy(vehicle_id):
        vehicle = common.global_vehicles[int(vehicle_id)]
        location = vehicle.get_position().location
        return float(location.x), float(location.y)

    def _mean_distance_to_coalition(self, vehicle_id, coalition):
        if not coalition.members:
            return 0.0
        vx, vy = self._vehicle_xy(vehicle_id)
        distances = []
        for member_id in coalition.members:
            mx, my = self._vehicle_xy(member_id)
            distances.append(math.hypot(vx - mx, vy - my))
        return sum(distances) / max(1, len(distances))

    @staticmethod
    def _density_quality(vehicle_id, grid_id):
        vehicle = common.global_vehicles[int(vehicle_id)]
        rho_th = max(float(vehicle.rho_th), 1e-6)
        density = float(vehicle.grid_density_dict.get(grid_id, 0.0))
        return min(1.0, density / rho_th)

    def _coalition_grid_quality(self, coalition, grid_id):
        quality = 0.0
        for member_id in coalition.members:
            quality = max(quality, self._density_quality(member_id, grid_id))
        return quality

    def _coverage_gain(self, vehicle_id, coalition):
        vehicle = common.global_vehicles[int(vehicle_id)]
        if not coalition.members:
            return len(vehicle.sens_grids) / max(1.0, len(vehicle.req_grids))
        candidate_grids = set(vehicle.sens_grids) & set(coalition.req_grids)
        if not candidate_grids:
            return 0.0
        gains = []
        for grid_id in candidate_grids:
            member_quality = self._density_quality(vehicle_id, grid_id)
            coalition_quality = self._coalition_grid_quality(
                coalition,
                grid_id)
            gains.append(member_quality * (1.0 - coalition_quality))
        return sum(gains)

    def _object_gain(self, vehicle_id, coalition):
        vehicle = common.global_vehicles[int(vehicle_id)]
        object_grids = set(vehicle.high_density_grids) & set(
            coalition.req_grids)
        if not object_grids:
            object_grids = {
                grid_id for grid_id, density in
                vehicle.grid_density_dict.items()
                if density > 0.0 and grid_id in coalition.req_grids
            }
        if not object_grids:
            return 0.0
        quality_sum = sum(
            self._density_quality(vehicle_id, grid_id)
            for grid_id in object_grids)
        coalition_relevance = min(1.0, len(object_grids) /
                                  max(1.0, len(coalition.req_grids)))
        return quality_sum + 0.25 * coalition_relevance

    def _view_gain(self, vehicle_id, coalition):
        if not coalition.members:
            return 0.0
        vehicle = common.global_vehicles[int(vehicle_id)]
        shared_grids = set(vehicle.sens_grids) & set(coalition.sens_grids)
        if not shared_grids:
            return 0.0

        overlap_score = common.avg_grids_score(vehicle_id, shared_grids)
        if overlap_score <= 0.0:
            return 0.0

        return overlap_score

    def _communication_cost(self, vehicle_id, coalition):
        if not coalition.members:
            return 0.0
        distance = self._mean_distance_to_coalition(vehicle_id, coalition)
        distance_cost = min(1.0, distance / 80.0)
        capacity_cost = coalition.size() / max(1.0, float(self.p.N_max))
        return 0.7 * distance_cost + 0.3 * capacity_cost

    def cov_marginal_utility(self, coalition, vehicle_id):
        if not coalition.members:
            return {
                'coverage': 0.0,
                'object': 0.0,
                'view': 0.0,
                'cost': 0.0,
                'stability': 1.0,
                'utility': 0.0,
            }
        coverage = self._coverage_gain(vehicle_id, coalition)
        obj = self._object_gain(vehicle_id, coalition)
        view = self._view_gain(vehicle_id, coalition)
        cost = self._communication_cost(vehicle_id, coalition)
        stability = self.stability_cost(vehicle_id, coalition)
        if coalition.members:
            stability = max(0.25, stability)
        else:
            stability = 1.0
        utility = (
            self.coverage_weight * coverage +
            self.object_weight * obj +
            self.view_weight * view -
            self.cost_weight * cost)
        utility *= stability
        return {
            'coverage': coverage,
            'object': obj,
            'view': view,
            'cost': cost,
            'stability': stability,
            'utility': utility,
        }

    def marginal_contribution(self, coalition, vid):
        if vid in coalition.members:
            return 0.0
        if vid not in common.global_vehicles:
            return 0.0
        components = self.cov_marginal_utility(coalition, vid)
        self.last_utility_breakdown[(int(vid),
                                     tuple(sorted(coalition.members)))] = (
                                         components)
        return components['utility']
