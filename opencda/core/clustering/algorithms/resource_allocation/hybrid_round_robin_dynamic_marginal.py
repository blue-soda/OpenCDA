# -*- coding: utf-8 -*-
"""Hybrid round-robin/greedy dynamic marginal scheduler for SGCP probes.

The first pass gives each cluster head one deterministic round-robin scheduling
opportunity.  After that, all remaining subchannels are greedily assigned to
the currently best dynamic early-utility link

    q_i(g) * (1 - Q_h^A(g)).

This variant tests whether a minimal fairness pass can keep distributed
head-level coverage while allowing later resources to follow the strongest
global marginal gain.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.round_robin_dynamic_marginal import (
    RoundRobinDynamicMarginal,
)
from opencda.log.logger_config import logger


class HybridRoundRobinDynamicMarginal(RoundRobinDynamicMarginal):
    """One round-robin pass followed by global dynamic-marginal greedy."""

    def __init__(self, cav_world):
        super(HybridRoundRobinDynamicMarginal, self).__init__(cav_world)
        self.grid_score_mode = 'hybrid_round_robin_dynamic_marginal'

    def channel_game(self, max_iter=1):
        del max_iter
        self.dynamic_evidence = defaultdict(dict)
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()
        max_links_per_head = max(1, int(getattr(self.p, 'head_rb_budget', 1)))
        ordered_clusters = sorted(self.clusters, key=lambda item: int(item.head_id))

        first_pass_links = 0
        for cluster in ordered_clusters:
            if self._next_available_channel(rb_occupancy) is None:
                break
            head_id = int(cluster.head_id)
            if head_link_count[head_id] >= max_links_per_head:
                continue
            link = self._best_member_link(cluster, mode='dynamic')
            if link is None:
                continue
            if self._append_link(cluster, link, rb_occupancy,
                                 head_link_count):
                first_pass_links += 1

        greedy_rounds = 0
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
            greedy_rounds += 1

        link_count = 0
        selected_grid_count = 0
        for links in self.strategies.values():
            for _, _, _, grids in links:
                link_count += 1
                selected_grid_count += len(grids)
        self.convergence_stats = {
            'iterations': 1 + greedy_rounds,
            'cluster_updates': len(self.clusters),
            'scheduled_links': link_count,
            'selected_grids': selected_grid_count,
            'first_pass_links': first_pass_links,
            'greedy_links': greedy_rounds,
            'used_rbs': sum(1 for count in rb_occupancy.values()
                            if count > 0),
            'reused_rbs': 0,
            'max_rb_occupancy': 1 if link_count else 0,
            'converged': True,
        }
        logger.info(
            'Hybrid round-robin dynamic marginal scheduler finished: '
            'first_pass=%d greedy=%d links=%d grids=%d',
            first_pass_links,
            greedy_rounds,
            link_count,
            selected_grid_count)
        return self.strategies
