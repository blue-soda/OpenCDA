# -*- coding: utf-8 -*-
"""Balanced perception-aware SGCP scheduler.

This variant keeps the PAPG two-layer mechanism, but makes source diversity an
explicit marginal term of the scheduling potential. The motivation comes from
the 10-channel diagnostics: pure target gain can repeatedly favor the same
high-scoring source CAVs, reducing the number of distinct uploaded views and
hurting recall. Here a link remains target/coverage driven, but its marginal
utility is discounted when the source has already been used in the same frame
and boosted when it brings a new high-quality view.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.perception_aware_potential_game import (
    PerceptionAwarePotentialGame,
)


class BalancedPerceptionAwarePotentialGame(PerceptionAwarePotentialGame):
    """PAPG with an explicit source-diversity marginal potential."""

    source_upload_history = Counter()

    def __init__(self, cav_world):
        super(BalancedPerceptionAwarePotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'balanced_perception_aware'
        self.repeat_source_penalty = 0.22
        self.new_source_bonus = 0.16
        self.under_served_head_bonus = 0.08
        self.history_credit_weight = 0.38

    def _source_balanced_score(self, link, mode, global_source_count,
                               head_link_count, cluster):
        member_id = int(link['member_id'])
        source_count = global_source_count[member_id]
        repeat_discount = 1.0 / (
            1.0 + self.repeat_source_penalty * float(source_count))
        source_bonus = self.new_source_bonus if source_count == 0 else 0.0
        head_bonus = (
            self.under_served_head_bonus
            if head_link_count[int(cluster.head_id)] == 0 else 0.0)
        if self.source_upload_history:
            max_history = max(self.source_upload_history.values())
            source_history = self.source_upload_history.get(member_id, 0)
            history_deficit = max(0.0, max_history - source_history)
            history_credit = (
                self.history_credit_weight *
                history_deficit / float(max_history + 1.0))
        else:
            history_credit = 0.0

        if mode == 'target':
            base = 0.78 * link['object_score'] + 0.22 * link['coverage_score']
        else:
            base = 0.72 * link['coverage_score'] + 0.28 * link['object_score']
        return (
            base * repeat_discount +
            (source_bonus + history_credit) * link['score'] +
            head_bonus)

    def _best_balanced_link(self, cluster, mode, global_source_count,
                            head_link_count, excluded_members=None):
        excluded_members = excluded_members or set()
        best = None
        best_key = None
        for member_id in sorted(cluster.members):
            if member_id == cluster.head_id or member_id in excluded_members:
                continue
            candidate = self._score_candidate(cluster, member_id, mode=mode)
            if candidate is None:
                continue
            balanced_score = self._source_balanced_score(
                candidate,
                mode,
                global_source_count,
                head_link_count,
                cluster)
            candidate['balanced_score'] = balanced_score
            key = (
                balanced_score,
                candidate['object_score'],
                candidate['coverage_score'],
                -global_source_count[int(member_id)],
                -int(member_id))
            if best is None or key > best_key:
                best = candidate
                best_key = key
        return best

    def _append_link(self, cluster, link, rb_occupancy, head_link_count,
                     global_source_count=None):
        appended = super(BalancedPerceptionAwarePotentialGame,
                         self)._append_link(cluster, link, rb_occupancy,
                                            head_link_count)
        if appended and global_source_count is not None:
            global_source_count[int(link['member_id'])] += 1
        return appended

    def channel_game(self, max_iter=1):
        del max_iter
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()
        global_source_count = Counter()

        # Coverage layer: assign one link per head, but prefer coverage links
        # that also increase distinct source views in the current frame.
        pending_heads = set(int(cluster.head_id) for cluster in self.clusters)
        cluster_by_head = {
            int(cluster.head_id): cluster for cluster in self.clusters
        }
        while pending_heads and self._next_available_channel(
                rb_occupancy) is not None:
            candidates = []
            for head_id in sorted(pending_heads):
                cluster = cluster_by_head[head_id]
                link = self._best_balanced_link(
                    cluster,
                    'coverage',
                    global_source_count,
                    head_link_count)
                if link is None:
                    continue
                link['cluster'] = cluster
                candidates.append(link)
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda item: (
                    item['balanced_score'],
                    item['coverage_score'],
                    item['object_score'],
                    -int(item['cluster'].head_id)))
            if self._append_link(best['cluster'], best, rb_occupancy,
                                 head_link_count, global_source_count):
                pending_heads.remove(int(best['cluster'].head_id))
            else:
                break

        # Target layer: fill remaining RBs with target-prototype gain while
        # applying the same source-diversity diminishing return.
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
                link = self._best_balanced_link(
                    cluster,
                    'target',
                    global_source_count,
                    head_link_count,
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
                    item['balanced_score'],
                    item['object_score'],
                    item['coverage_score'],
                    -global_source_count[int(item['member_id'])],
                    -int(item['cluster'].head_id)))
            if not self._append_link(best['cluster'], best, rb_occupancy,
                                     head_link_count, global_source_count):
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
            'distinct_sources': len(global_source_count),
            'max_source_reuse': (
                max(global_source_count.values()) if global_source_count else 0),
            'converged': True,
        }
        for member_id in global_source_count:
            self.source_upload_history[int(member_id)] += 1
        return self.strategies
