# -*- coding: utf-8 -*-
"""Quality-gated perception-aware SGCP scheduler.

This allocator is a conservative follow-up to PAPG/BPAPG. Diagnostics showed
that generic source rotation is harmful: low-frequency sources are not useful
unless they provide a high-quality target view for the relevant cluster head.
The scheduler therefore adds history/fairness credit only after a candidate
passes object-prototype quality gates.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.perception_aware_potential_game import (
    PerceptionAwarePotentialGame,
)


class QualityGatedPerceptionAwarePotentialGame(PerceptionAwarePotentialGame):
    """PAPG with target-quality-gated source protection."""

    source_upload_history = Counter()

    def __init__(self, cav_world):
        super(QualityGatedPerceptionAwarePotentialGame, self).__init__(
            cav_world)
        self.grid_score_mode = 'quality_gated_perception_aware'
        self.history_credit_weight = 0.20
        self.repeat_source_penalty = 0.10
        self.quality_gate_ratio = 0.88
        self.peak_gate_ratio = 0.85
        self.coverage_floor_ratio = 0.42

    @staticmethod
    def _link_target_base(link):
        return 0.76 * link['object_score'] + 0.24 * link['coverage_score']

    @staticmethod
    def _link_coverage_base(link):
        return 0.72 * link['coverage_score'] + 0.28 * link['object_score']

    def _passes_quality_gate(self, link, best_object_score, best_peak,
                             best_coverage_score):
        if best_object_score <= 0 or best_peak <= 0:
            return False
        object_ok = (
            link['object_score'] >= self.quality_gate_ratio *
            best_object_score)
        peak_ok = link['peak'] >= self.peak_gate_ratio * best_peak
        coverage_ok = (
            best_coverage_score <= 0 or
            link['coverage_score'] >= self.coverage_floor_ratio *
            best_coverage_score)
        return object_ok and peak_ok and coverage_ok

    def _history_credit(self, member_id):
        if not self.source_upload_history:
            return 0.0
        max_history = max(self.source_upload_history.values())
        if max_history <= 0:
            return 0.0
        history = self.source_upload_history.get(int(member_id), 0)
        return max(0.0, max_history - history) / float(max_history + 1.0)

    def _score_links_for_cluster(self, cluster, mode, global_source_count,
                                 head_link_count, excluded_members=None):
        del head_link_count
        excluded_members = excluded_members or set()
        links = []
        for member_id in sorted(cluster.members):
            if member_id == cluster.head_id or member_id in excluded_members:
                continue
            link = self._score_candidate(cluster, member_id, mode=mode)
            if link is not None:
                links.append(link)
        if not links:
            return []

        best_object = max(link['object_score'] for link in links)
        best_peak = max(link['peak'] for link in links)
        best_coverage = max(link['coverage_score'] for link in links)
        scored_links = []
        for link in links:
            member_id = int(link['member_id'])
            if mode == 'target':
                base = self._link_target_base(link)
                quality_gate = self._passes_quality_gate(
                    link,
                    best_object,
                    best_peak,
                    best_coverage)
            else:
                base = self._link_coverage_base(link)
                quality_gate = False

            repeat_discount = 1.0 / (
                1.0 + self.repeat_source_penalty *
                float(global_source_count[member_id]))
            credit = 0.0
            if quality_gate:
                credit = self.history_credit_weight * self._history_credit(
                    member_id) * link['score']
            link['quality_gate'] = int(quality_gate)
            link['balanced_score'] = base * repeat_discount + credit
            scored_links.append(link)
        return scored_links

    def _append_link(self, cluster, link, rb_occupancy, head_link_count,
                     global_source_count=None):
        appended = super(QualityGatedPerceptionAwarePotentialGame,
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
        quality_gated_links = 0

        # Coverage layer stays conservative and close to PAPG.
        for cluster in sorted(self.clusters, key=lambda item: int(item.head_id)):
            if self._next_available_channel(rb_occupancy) is None:
                break
            links = self._score_links_for_cluster(
                cluster,
                'coverage',
                global_source_count,
                head_link_count)
            if not links:
                continue
            best = max(
                links,
                key=lambda item: (
                    item['balanced_score'],
                    item['coverage_score'],
                    item['object_score'],
                    -int(item['member_id'])))
            self._append_link(cluster, best, rb_occupancy, head_link_count,
                              global_source_count)

        # Target layer may protect a low-frequency source, but only if its
        # object-prototype quality is comparable to the best local candidate.
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
                for link in self._score_links_for_cluster(
                        cluster,
                        'target',
                        global_source_count,
                        head_link_count,
                        excluded_members=used_members):
                    link['cluster'] = cluster
                    candidates.append(link)
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda item: (
                    item['balanced_score'],
                    item['quality_gate'],
                    item['object_score'],
                    item['coverage_score'],
                    -global_source_count[int(item['member_id'])],
                    -int(item['cluster'].head_id)))
            if best.get('quality_gate'):
                quality_gated_links += 1
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
            'quality_gated_links': quality_gated_links,
            'converged': True,
        }
        for member_id in global_source_count:
            self.source_upload_history[int(member_id)] += 1
        return self.strategies
