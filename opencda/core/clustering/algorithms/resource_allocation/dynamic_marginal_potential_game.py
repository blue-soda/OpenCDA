# -*- coding: utf-8 -*-
"""Dynamic marginal-gain scheduler for SGCP.

This scheduler is a single-stage counterpart to ``cov_potential_game``.  It
scores each sender-grid action by the current receiver-side marginal evidence
gain

    min(q_i(g), 1 - Q_h^A(g)),

where ``Q_h^A(g)`` is updated after every accepted upload.  The update gives
repeated uploads to the same head/grid a diminishing return while keeping the
same communication feasibility constraints as the existing SGCP schedulers.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.cov_potential_game import (
    COVPotentialGame,
)
from opencda.core.clustering.utils import common
from opencda.log.logger_config import logger


class DynamicMarginalPotentialGame(COVPotentialGame):
    """Single-stage scheduler using dynamic receiver evidence residuals."""

    def __init__(self, cav_world):
        super(DynamicMarginalPotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'dynamic_marginal'
        self.dynamic_evidence = defaultdict(dict)

    def _head_quality(self, head_id, grid_id):
        head_vehicle = common.global_vehicles[int(head_id)]
        rho_th = max(float(head_vehicle.rho_th), 1e-6)
        return self.grid_utility_density(
            self._vehicle_density(head_id, grid_id),
            rho_th)

    def _member_quality(self, head_id, member_id, grid_id):
        head_vehicle = common.global_vehicles[int(head_id)]
        rho_th = max(float(head_vehicle.rho_th), 1e-6)
        return self.grid_utility_density(
            self._vehicle_density(member_id, grid_id),
            rho_th)

    def _current_evidence(self, head_id, grid_id):
        head_id = int(head_id)
        grid_key = str(grid_id)
        if grid_key not in self.dynamic_evidence[head_id]:
            self.dynamic_evidence[head_id][grid_key] = self._head_quality(
                head_id,
                grid_id)
        return self.dynamic_evidence[head_id][grid_key]

    def _update_evidence(self, head_id, member_id, selected_grids):
        head_id = int(head_id)
        member_id = int(member_id)
        for grid_id in selected_grids:
            current = self._current_evidence(head_id, grid_id)
            quality = self._member_quality(head_id, member_id, grid_id)
            updated = current + min(quality, max(0.0, 1.0 - current))
            self.dynamic_evidence[head_id][str(grid_id)] = min(1.0, updated)

    def _grid_marginal_score(self, head_id, member_id, grid_id):
        quality = self._member_quality(head_id, member_id, grid_id)
        if quality <= 0.0:
            return 0.0
        current = self._current_evidence(head_id, grid_id)
        return min(quality, max(0.0, 1.0 - current))

    def _score_candidate(self, cluster, member_id, mode='dynamic'):
        del mode
        raw_candidates = set(self.refinement_candidates(cluster, member_id))
        if not raw_candidates:
            return None

        head_id = int(cluster.head_id)
        ranked = []
        for grid_id in raw_candidates:
            score = self._grid_marginal_score(head_id, member_id, grid_id)
            if score <= 0.0:
                continue
            ranked.append((
                score,
                self._vehicle_density(member_id, grid_id),
                str(grid_id),
                grid_id))
        if not ranked:
            return None

        ranked.sort(reverse=True)
        selected = [
            grid_id for _, _, _, grid_id in
            ranked[:min(self.max_grids_per_rb, len(ranked))]
        ]
        if not selected:
            return None

        selected_score = sum(
            self._grid_marginal_score(head_id, member_id, grid_id)
            for grid_id in selected)
        if selected_score <= 0.0:
            return None

        return {
            'member_id': int(member_id),
            'score': selected_score,
            'coverage_score': selected_score,
            'object_score': selected_score,
            'selected': selected,
            'candidate_count': len(ranked),
            'peak': ranked[0][0],
            'dynamic_marginal': selected_score,
        }

    def _append_link(self, cluster, link, rb_occupancy, head_link_count):
        appended = super(DynamicMarginalPotentialGame, self)._append_link(
            cluster,
            link,
            rb_occupancy,
            head_link_count)
        if appended:
            self._update_evidence(
                int(cluster.head_id),
                int(link['member_id']),
                link['selected'])
        return appended

    def channel_game(self, max_iter=1):
        del max_iter
        self.dynamic_evidence = defaultdict(dict)
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()
        max_links_per_head = max(1, int(getattr(self.p, 'head_rb_budget', 1)))

        while self._next_available_channel(rb_occupancy) is not None:
            candidates = []
            for cluster in self.clusters:
                head_id = int(cluster.head_id)
                if head_link_count[head_id] >= max_links_per_head:
                    continue
                used_members = {
                    int(member_id) for member_id, _, _, _ in
                    self.strategies.get(head_id, [])
                }
                link = self._best_member_link(
                    cluster,
                    mode='dynamic',
                    excluded_members=used_members)
                if link is None:
                    continue
                link['cluster'] = cluster
                candidates.append(link)

            if not candidates:
                break

            best = max(
                candidates,
                key=lambda item: (
                    item['score'],
                    item['peak'],
                    -int(item['cluster'].head_id),
                    -int(item['member_id'])))
            if not self._append_link(best['cluster'], best, rb_occupancy,
                                     head_link_count):
                break

        link_count = 0
        selected_grid_count = 0
        for links in self.strategies.values():
            for _, _, _, grids in links:
                link_count += 1
                selected_grid_count += len(grids)
        self.convergence_stats = {
            'iterations': 1,
            'cluster_updates': len(self.clusters),
            'scheduled_links': link_count,
            'selected_grids': selected_grid_count,
            'used_rbs': sum(1 for count in rb_occupancy.values()
                            if count > 0),
            'reused_rbs': 0,
            'max_rb_occupancy': 1 if link_count else 0,
            'converged': True,
        }
        logger.info(
            'Dynamic marginal scheduler finished: links=%d grids=%d',
            link_count,
            selected_grid_count)
        return self.strategies
