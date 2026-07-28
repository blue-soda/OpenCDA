# -*- coding: utf-8 -*-
"""Two-stage dynamic marginal scheduler for SGCP probes.

The first stage uses the receiver-local dynamic marginal gain

    q_i(g) * (1 - Q_h^A(g)),

where ``Q_h^A(g)`` is updated after accepted uploads.  The second stage uses
the same multi-view refinement term as the clean C/V scheduler:

    V(i, g | h) = q_i(g), if q_h(g) > 0.

This keeps the scene-level marginal interpretation for coverage recovery while
testing whether an explicit multi-view refinement stage recovers the AP lost by
the single-stage dynamic marginal probe.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.dynamic_marginal_potential_game import (
    DynamicMarginalPotentialGame,
)
from opencda.core.clustering.utils import common
from opencda.log.logger_config import logger


class DynamicMarginalTwoStagePotentialGame(DynamicMarginalPotentialGame):
    """Dynamic marginal coverage stage followed by V refinement stage."""

    def __init__(self, cav_world):
        super(DynamicMarginalTwoStagePotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'dynamic_marginal_two_stage'

    def _grid_view_score(self, head_id, member_id, grid_id):
        member_quality = self._member_quality(head_id, member_id, grid_id)
        if member_quality <= 0.0:
            return 0.0
        head_quality = self._head_quality(head_id, grid_id)
        if head_quality <= 0.0:
            return 0.0
        return member_quality

    def _score_candidate(self, cluster, member_id, mode='dynamic'):
        if mode != 'target':
            return super(DynamicMarginalTwoStagePotentialGame,
                         self)._score_candidate(
                             cluster,
                             member_id,
                             mode='dynamic')

        raw_candidates = set(self.refinement_candidates(cluster, member_id))
        if not raw_candidates:
            return None

        head_id = int(cluster.head_id)
        ranked = []
        for grid_id in raw_candidates:
            score = self._grid_view_score(head_id, member_id, grid_id)
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

        view_score = sum(
            self._grid_view_score(head_id, member_id, grid_id)
            for grid_id in selected)
        dynamic_score = sum(
            self._grid_marginal_score(head_id, member_id, grid_id)
            for grid_id in selected)
        if view_score <= 0.0:
            return None

        return {
            'member_id': int(member_id),
            'score': view_score,
            'coverage_score': dynamic_score,
            'object_score': view_score,
            'selected': selected,
            'candidate_count': len(ranked),
            'peak': ranked[0][0],
            'dynamic_marginal': dynamic_score,
            'view_score': view_score,
        }

    def channel_game(self, max_iter=1):
        del max_iter
        self.dynamic_evidence = defaultdict(dict)
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()

        # Stage 1: give each cluster head one dynamic marginal coverage upload
        # when feasible.  This is the switch condition used by the current
        # SGCP scheduler family: after this pass, remaining RBs are used for
        # multi-view refinement.
        for cluster in sorted(self.clusters, key=lambda item: int(item.head_id)):
            if self._next_available_channel(rb_occupancy) is None:
                break
            link = self._best_member_link(cluster, mode='dynamic')
            if link is not None:
                self._append_link(cluster, link, rb_occupancy,
                                  head_link_count)

        # Stage 2: allocate remaining RBs to the strongest V refinements.
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
                    mode='target',
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
                    item['coverage_score'],
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
            'Dynamic marginal two-stage scheduler finished: links=%d grids=%d',
            link_count,
            selected_grid_count)
        return self.strategies
