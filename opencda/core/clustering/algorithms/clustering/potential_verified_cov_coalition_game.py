# -*- coding: utf-8 -*-
"""Potential-verified C/V coalition game for SGCP.

This variant strengthens ``cov_coalition_game`` with an explicit local
partition-potential admission check.  Vehicles still rank migrations by the
C/V proxy utility, but a migration is committed only when it increases the
pairwise multi-view potential of the two affected coalitions.
"""

from opencda.core.clustering.algorithms.clustering.cov_coalition_game import (
    COVCoalitionGame,
)
from opencda.core.clustering.utils import common
from opencda.log.logger_config import logger


class PotentialVerifiedCOVCoalitionGame(COVCoalitionGame):
    """C/V coalition formation with exact pairwise-potential admission."""

    def __init__(self, cav_world):
        super(PotentialVerifiedCOVCoalitionGame, self).__init__(cav_world)
        self.last_potential_checks = []

    def _pairwise_view_value(self, first_id, second_id):
        """Symmetric pairwise multi-view value for two CAVs."""
        first = common.global_vehicles.get(int(first_id))
        second = common.global_vehicles.get(int(second_id))
        if first is None or second is None:
            return 0.0
        shared_grids = set(first.sens_grids) & set(second.sens_grids)
        value = 0.0
        for grid_id in shared_grids:
            value += min(
                self._density_quality(first_id, grid_id),
                self._density_quality(second_id, grid_id))
        return value

    def _coalition_view_value(self, members):
        """Pairwise multi-view potential within a coalition."""
        members = sorted(int(item) for item in members)
        if len(members) <= 1:
            return 0.0
        view_value = 0.0
        for i, first_id in enumerate(members):
            for second_id in members[i + 1:]:
                view_value += self._pairwise_view_value(first_id, second_id)
        return view_value

    def coalition_potential(self, members):
        """Coalition-level potential: pairwise multi-view value."""
        members = set(int(item) for item in members)
        if len(members) <= 1:
            return 0.0
        return self._coalition_view_value(members)

    def affected_potential_delta(self, vehicle_id, source, target):
        """Return exact potential increment for moving one vehicle."""
        vehicle_id = int(vehicle_id)
        source_before = set(int(item) for item in source.members)
        target_before = set(int(item) for item in target.members)
        source_after = source_before - {vehicle_id}
        target_after = target_before | {vehicle_id}

        before = (
            self.coalition_potential(source_before) +
            self.coalition_potential(target_before))
        after = (
            self.coalition_potential(source_after) +
            self.coalition_potential(target_after))
        return after - before, before, after

    def coalition_formation(self, max_iter=20):
        self.check_is_ok()
        self.ego_coalition_be_first()
        self.capacity_stats = {
            'full_candidate_skips': 0,
        }
        self.last_potential_checks = []
        for iteration in range(max_iter):
            logger.info(
                '--- Potential-verified C/V Coalition Iteration %d ---',
                iteration + 1)
            updated = False
            for vid in common.global_vehicles.keys():
                current = self.find_coalition(vid)
                if current is None:
                    logger.info('Vehicle %s is not in any coalition.', vid)
                    continue
                current_contribution = self.current_contribution(current, vid)
                best_candidate = None
                for coalition in list(self.coalitions):
                    if coalition is current:
                        continue
                    if coalition.size() >= self.p.N_max:
                        self.capacity_stats['full_candidate_skips'] += 1
                        continue
                    delta = self.marginal_contribution(coalition, vid)
                    proxy_accept = delta > current_contribution * self.p.ita
                    phi_delta, phi_before, phi_after = (
                        self.affected_potential_delta(
                            vid,
                            current,
                            coalition))
                    check = {
                        'vehicle_id': int(vid),
                        'source_members': sorted(int(item)
                                                 for item in current.members),
                        'target_members': sorted(int(item)
                                                 for item in
                                                 coalition.members),
                        'proxy_before': float(current_contribution),
                        'proxy_after': float(delta),
                        'phi_before': float(phi_before),
                        'phi_after': float(phi_after),
                        'phi_delta': float(phi_delta),
                        'proxy_accept': bool(proxy_accept),
                        'accepted': False,
                    }
                    self.last_potential_checks.append(check)
                    if not proxy_accept or phi_delta <= 1e-9:
                        continue
                    candidate = {
                        'coalition': coalition,
                        'proxy_after': delta,
                        'phi_delta': phi_delta,
                        'phi_before': phi_before,
                        'phi_after': phi_after,
                        'check': check,
                    }
                    if best_candidate is None:
                        best_candidate = candidate
                    elif (delta, phi_delta) > (
                            best_candidate['proxy_after'],
                            best_candidate['phi_delta']):
                        best_candidate = candidate

                if best_candidate is None:
                    continue

                best_coalition = best_candidate['coalition']
                best_delta = best_candidate['proxy_after']
                phi_delta = best_candidate['phi_delta']
                best_candidate['check']['accepted'] = True

                logger.info(
                    'Accept vehicle %s migration: %s -> %s, '
                    'proxy %.4f -> %.4f, delta_phi %.6f',
                    vid,
                    sorted(current.members),
                    sorted(best_coalition.members),
                    current_contribution,
                    best_delta,
                    phi_delta)
                current.remove_member(vid)
                best_coalition.add_member(vid)
                if current.size() == 0:
                    self.coalitions.remove(current)
                    current = None
                if current is not None and current.head_id in current.members:
                    current.grid_bits = current.compute_grid_bits()
                if best_coalition.head_id in best_coalition.members:
                    best_coalition.grid_bits = best_coalition.compute_grid_bits()
                updated = True
            if not updated:
                break

        logger.info(
            'Potential-verified C/V coalition formation converged in %d '
            'iterations.',
            iteration + 1)
        for coalition in self.coalitions:
            logger.info('[%s]', coalition.members)
        return self.coalitions
