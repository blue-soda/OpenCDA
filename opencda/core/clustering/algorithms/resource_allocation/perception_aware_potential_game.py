# -*- coding: utf-8 -*-
"""Perception-aware SGCP resource scheduler.

The scheduler uses a two-layer potential-guided allocation:

1. Coverage layer: every cluster head receives one high-quality external view
   when a member is available. This keeps low-IoU recall and spatial coverage
   from collapsing under tight channel budgets.
2. Target layer: remaining RBs are assigned to object-prototype actions with
   high expected marginal detection gain.

Both layers use the same grid utility and object-prototype extraction from
``ObjectAwarePotentialGame``. The layers differ only in the potential term they
emphasize, so the mechanism remains a single perception-aware scheduling
objective rather than a collection of fallback rules.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.object_aware_potential_game import (
    ObjectAwarePotentialGame,
)
from opencda.core.clustering.utils import common
from opencda.log.logger_config import logger


class PerceptionAwarePotentialGame(ObjectAwarePotentialGame):
    """Balanced target/context scheduler for SGCP."""

    def __init__(self, cav_world):
        super(PerceptionAwarePotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'perception_aware'

    def run(self):
        self.clear_resource_allocation_strategy()
        ret = self.channel_game()
        self.update_resource_allocation_strategy()
        return ret

    def _score_candidate(self, cluster, member_id, mode='coverage'):
        candidates = self.refinement_candidates(cluster, member_id)
        candidates = set(candidates)
        if not candidates:
            return None
        scores = {}
        for grid_id in candidates:
            density = common.global_vehicles[
                member_id].grid_density_dict.get(grid_id, 0.0)
            scores[grid_id] = self.grid_score(cluster, grid_id, density)
        selected = self.sort_member_grids(
            cluster,
            member_id,
            candidates,
            scores,
            min(self.max_grids_per_rb, len(candidates)))
        if not selected:
            return None

        sorted_scores = sorted(scores.values(), reverse=True)
        top = sorted_scores[:max(1, min(self.max_grids_per_rb,
                                        len(sorted_scores)))]
        cover = sorted_scores[:max(1, min(self.max_grids_per_rb * 2,
                                          len(sorted_scores)))]
        peak = top[0] if top else 0.0
        components = self._connected_components([
            grid for grid in candidates
            if self._vehicle_density(member_id, grid) >=
            self._density_floor(member_id)
        ])
        component_scores = sorted(
            (self._component_score(component, scores)
             for component in components),
            reverse=True)
        object_score = peak + 0.55 * sum(component_scores[:3])
        coverage_score = (
            0.65 * sum(top) +
            0.28 * sum(cover) +
            0.05 * min(len(candidates), self.max_grids_per_rb * 2))

        if mode == 'target':
            score = 0.72 * object_score + 0.28 * coverage_score
        else:
            score = 0.68 * coverage_score + 0.32 * object_score
        return {
            'member_id': member_id,
            'score': score,
            'coverage_score': coverage_score,
            'object_score': object_score,
            'selected': selected,
            'candidate_count': len(candidates),
            'peak': peak,
        }

    def _best_member_link(self, cluster, mode='coverage',
                          excluded_members=None):
        excluded_members = excluded_members or set()
        best = None
        for member_id in sorted(cluster.members):
            if member_id == cluster.head_id or member_id in excluded_members:
                continue
            candidate = self._score_candidate(cluster, member_id, mode=mode)
            if candidate is None:
                continue
            if best is None:
                best = candidate
                continue
            key = (
                candidate['score'],
                candidate['object_score'],
                candidate['coverage_score'],
                -candidate['member_id'])
            best_key = (
                best['score'],
                best['object_score'],
                best['coverage_score'],
                -best['member_id'])
            if key > best_key:
                best = candidate
        return best

    def _next_available_channel(self, rb_occupancy):
        for channel_id in range(self.p.num_channels):
            if rb_occupancy[(channel_id, 0)] < 1:
                return channel_id
        return None

    def _append_link(self, cluster, link, rb_occupancy, head_link_count):
        channel_id = self._next_available_channel(rb_occupancy)
        if channel_id is None:
            return False
        head_id = int(cluster.head_id)
        self.strategies[head_id].append((
            int(link['member_id']),
            channel_id,
            0,
            list(link['selected'])))
        rb_occupancy[(channel_id, 0)] += 1
        head_link_count[head_id] += 1
        self.grids_uploading |= set(link['selected'])
        logger.info(
            'Perception-aware layer assign head=%s member=%s channel=%s '
            'score=%.4f coverage=%.4f object=%.4f grids=%d',
            head_id,
            int(link['member_id']),
            channel_id,
            link['score'],
            link['coverage_score'],
            link['object_score'],
            len(link['selected']))
        return True

    def channel_game(self, max_iter=1):
        del max_iter
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()

        # Coverage layer: one link per cluster head when possible.
        for cluster in sorted(self.clusters, key=lambda item: int(item.head_id)):
            if self._next_available_channel(rb_occupancy) is None:
                break
            link = self._best_member_link(cluster, mode='coverage')
            if link is not None:
                self._append_link(cluster, link, rb_occupancy,
                                  head_link_count)

        # Target layer: remaining RBs go to the largest object-prototype gains.
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
                    item['object_score'],
                    item['coverage_score'],
                    -int(item['cluster'].head_id)))
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
        return self.strategies
