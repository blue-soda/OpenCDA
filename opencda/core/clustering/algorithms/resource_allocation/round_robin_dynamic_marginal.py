# -*- coding: utf-8 -*-
"""Round-robin dynamic marginal scheduler for SGCP experiments.

Each scheduling round visits cluster heads in a deterministic order.  A head
selects its best remaining sender link according to the dynamic early-utility
increment

    q_i(g) * (1 - Q_h^A(g)),

where ``Q_h^A(g)`` is the receiver-side evidence after previously admitted
uploads in the same frame.  The scheduler keeps the existing SGCP feasibility
constraints and relies on the offline replay upload path for density-capped
point truncation.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.dynamic_marginal_potential_game import (
    DynamicMarginalPotentialGame,
)
from opencda.log.logger_config import logger


class RoundRobinDynamicMarginal(DynamicMarginalPotentialGame):
    """Dynamic marginal scheduler with deterministic head round-robin access."""

    def __init__(self, cav_world):
        super(RoundRobinDynamicMarginal, self).__init__(cav_world)
        self.grid_score_mode = 'round_robin_dynamic_marginal'

    def _grid_marginal_score(self, head_id, member_id, grid_id):
        quality = self._member_quality(head_id, member_id, grid_id)
        if quality <= 0.0:
            return 0.0
        current = self._current_evidence(head_id, grid_id)
        return quality * max(0.0, 1.0 - current)

    def channel_game(self, max_iter=1):
        del max_iter
        self.dynamic_evidence = defaultdict(dict)
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()
        max_links_per_head = max(1, int(getattr(self.p, 'head_rb_budget', 1)))
        ordered_clusters = sorted(self.clusters, key=lambda item: int(item.head_id))

        made_progress = True
        rounds = 0
        while (made_progress and
               self._next_available_channel(rb_occupancy) is not None):
            made_progress = False
            rounds += 1
            for cluster in ordered_clusters:
                if self._next_available_channel(rb_occupancy) is None:
                    break
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
                if self._append_link(cluster, link, rb_occupancy,
                                     head_link_count):
                    made_progress = True

        link_count = 0
        selected_grid_count = 0
        for links in self.strategies.values():
            for _, _, _, grids in links:
                link_count += 1
                selected_grid_count += len(grids)
        self.convergence_stats = {
            'iterations': rounds,
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
            'Round-robin dynamic marginal scheduler finished: '
            'rounds=%d links=%d grids=%d',
            rounds,
            link_count,
            selected_grid_count)
        return self.strategies
