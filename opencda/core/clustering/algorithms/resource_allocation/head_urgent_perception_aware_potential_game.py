# -*- coding: utf-8 -*-
"""Head-urgent perception-aware SGCP scheduler.

This variant targets the dominant diagnostic bucket: a target grid is uploaded
somewhere, but not to the nearest/relevant cluster head. Instead of rotating
source CAVs, it keeps PAPG's link utility and gives a bounded target-layer
bonus to heads whose best remaining object-prototype candidate is strong after
the coverage layer. This preserves the potential-game narrative while focusing
the scarce second RB on the receiver most likely to benefit.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.perception_aware_potential_game import (
    PerceptionAwarePotentialGame,
)


class HeadUrgentPerceptionAwarePotentialGame(PerceptionAwarePotentialGame):
    """PAPG with receiver-side target-urgency weighting."""

    def __init__(self, cav_world):
        super(HeadUrgentPerceptionAwarePotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'head_urgent_perception_aware'
        self.head_urgency_weight = 0.18
        self.coverage_satisfaction_weight = 0.10

    @staticmethod
    def _target_base(link):
        return 0.74 * link['object_score'] + 0.26 * link['coverage_score']

    @staticmethod
    def _coverage_base(link):
        return 0.70 * link['coverage_score'] + 0.30 * link['object_score']

    def _remaining_links(self, cluster, mode, excluded_members=None):
        excluded_members = excluded_members or set()
        links = []
        for member_id in sorted(cluster.members):
            if member_id == cluster.head_id or member_id in excluded_members:
                continue
            link = self._score_candidate(cluster, member_id, mode=mode)
            if link is not None:
                links.append(link)
        return links

    def _head_urgency(self, links, coverage_score):
        if not links:
            return 0.0
        best_object = max(link['object_score'] for link in links)
        best_peak = max(link['peak'] for link in links)
        best_coverage = max(link['coverage_score'] for link in links)
        if best_object <= 0.0:
            return 0.0
        coverage_gap = 1.0 / (
            1.0 + self.coverage_satisfaction_weight *
            max(0.0, coverage_score))
        target_strength = best_object + 0.45 * best_peak + 0.18 * best_coverage
        return target_strength * coverage_gap

    def _append_link(self, cluster, link, rb_occupancy, head_link_count):
        return super(HeadUrgentPerceptionAwarePotentialGame,
                     self)._append_link(cluster, link, rb_occupancy,
                                        head_link_count)

    def channel_game(self, max_iter=1):
        del max_iter
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()
        head_coverage_score = defaultdict(float)

        # Coverage layer: close to PAPG, but choose heads by best coverage gain
        # if the number of heads ever exceeds the channel count.
        pending = []
        for cluster in sorted(self.clusters, key=lambda item: int(item.head_id)):
            links = self._remaining_links(cluster, 'coverage')
            if not links:
                continue
            best = max(
                links,
                key=lambda item: (
                    self._coverage_base(item),
                    item['coverage_score'],
                    item['object_score'],
                    -int(item['member_id'])))
            best['cluster'] = cluster
            pending.append(best)
        for link in sorted(
                pending,
                key=lambda item: (
                    self._coverage_base(item),
                    item['coverage_score'],
                    item['object_score']),
                reverse=True):
            if self._next_available_channel(rb_occupancy) is None:
                break
            if self._append_link(link['cluster'], link, rb_occupancy,
                                 head_link_count):
                head_coverage_score[int(link['cluster'].head_id)] += (
                    link['coverage_score'])

        # Target layer: prioritize heads with strong remaining target
        # prototypes, not globally repeated low-quality source rotation.
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
                links = self._remaining_links(
                    cluster,
                    'target',
                    excluded_members=used_members)
                if not links:
                    continue
                urgency = self._head_urgency(
                    links,
                    head_coverage_score[head_id])
                for link in links:
                    link['cluster'] = cluster
                    link['head_urgency'] = urgency
                    link['urgent_score'] = (
                        self._target_base(link) +
                        self.head_urgency_weight * urgency)
                    candidates.append(link)
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda item: (
                    item['urgent_score'],
                    item['head_urgency'],
                    item['object_score'],
                    item['coverage_score'],
                    -int(item['cluster'].head_id)))
            if not self._append_link(best['cluster'], best, rb_occupancy,
                                     head_link_count):
                break
            head_coverage_score[int(best['cluster'].head_id)] += (
                best['coverage_score'])

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
        return self.strategies
