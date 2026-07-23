# -*- coding: utf-8 -*-
"""COV utility coalition game for SGCP.

This variant keeps coalition formation and head election unchanged, but writes
vehicle-to-coalition utility as the grid-level C/O/V/L utility aggregated over
candidate grids:

    U(i | S) = sum_g [C(i,g|S) + O(i,g) + V(i,g|S)] - L(i,S)

The default coalition objective uses the ``V`` term only.  This matches the
hierarchical SGCP narrative: coalition formation builds stable multi-view
groups for high-quality early fusion, while the scheduler later selects the
actual raw-LiDAR blocks using the complete C/O/V/L utility.
"""

import math
import os

from opencda.core.clustering.algorithms.clustering.coalition_game import (
    CoalitionGame,
)
from opencda.core.clustering.utils import common


class COVCoalitionGame(CoalitionGame):
    """Coalition formation driven by expected C/O/V/L utility."""

    def __init__(self, cav_world):
        super(COVCoalitionGame, self).__init__(cav_world)
        self.last_utility_breakdown = {}
        self.active_terms = self._parse_terms(
            os.environ.get('OPENCDA_COV_CLUSTER_TERMS', 'view'))

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
            return {'view'}
        return terms

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

    def _communication_cost(self, vehicle_id, coalition):
        if not coalition.members:
            return 0.0
        distance = self._mean_distance_to_coalition(vehicle_id, coalition)
        return min(1.0, distance / 100.0)

    def _grid_utility_components(self, vehicle_id, coalition, grid_id):
        vehicle_quality = self._density_quality(vehicle_id, grid_id)
        coalition_quality = self._coalition_grid_quality(coalition, grid_id)
        return {
            'coverage': vehicle_quality * max(0.0, 1.0 - coalition_quality),
            'object': vehicle_quality,
            'view': vehicle_quality if coalition_quality > 0.0 else 0.0,
        }

    def _candidate_grids(self, vehicle_id, coalition):
        vehicle = common.global_vehicles[int(vehicle_id)]
        if 'view' in self.active_terms:
            return set(vehicle.sens_grids) & set(coalition.sens_grids)
        return set(vehicle.sens_grids) & set(coalition.req_grids)

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
        return utility

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
        coverage = 0.0
        obj = 0.0
        view = 0.0
        candidate_grids = self._candidate_grids(vehicle_id, coalition)
        for grid_id in candidate_grids:
            components = self._grid_utility_components(
                vehicle_id,
                coalition,
                grid_id)
            coverage += components['coverage']
            obj += components['object']
            view += components['view']
        cost = self._communication_cost(vehicle_id, coalition)
        stability = self.stability_cost(vehicle_id, coalition)
        if coalition.members:
            stability = max(0.25, stability)
        else:
            stability = 1.0
        utility = self._compose_utility({
            'coverage': coverage,
            'object': obj,
            'view': view,
            'cost': cost,
        })
        utility *= stability
        return {
            'coverage': coverage,
            'object': obj,
            'view': view,
            'cost': cost,
            'stability': stability,
            'candidate_grids': len(candidate_grids),
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
