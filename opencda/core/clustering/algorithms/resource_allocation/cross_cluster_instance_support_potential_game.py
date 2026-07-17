# -*- coding: utf-8 -*-
"""Cross-cluster instance-support-aware SGCP scheduler.

The previous instance-support probe improved the grid utility but kept sender
selection inside fixed clusters. Diagnostics showed that many persistent misses
need the best object-supporting view to reach a different, more relevant
cluster head. This scheduler keeps the same global RB budget and PAPG layering,
but allows high-value non-head CAVs within communication range to serve a
nearby cluster head even when they are not in that head's coalition.
"""

from collections import Counter, defaultdict

from opencda.core.clustering.algorithms.resource_allocation.instance_support_potential_game import (
    InstanceSupportPotentialGame,
)
from opencda.core.clustering.algorithms.resource_allocation.potential_game import (
    calculate_distance,
)
from opencda.log.logger_config import logger
from opencda.core.clustering.utils import common


class CrossClusterInstanceSupportPotentialGame(InstanceSupportPotentialGame):
    """PAPG with cross-cluster target-to-head routing."""

    def __init__(self, cav_world):
        super(CrossClusterInstanceSupportPotentialGame, self).__init__(
            cav_world)
        self.grid_score_mode = 'cross_cluster_instance_support'
        self.external_sender_penalty = 0.88
        self.external_target_penalty = 0.94
        self.external_min_instance_score = 1.15
        self.max_external_links_per_frame = 1

    def _cluster_heads(self):
        return {int(cluster.head_id) for cluster in self.clusters}

    def _communication_range(self):
        values = []
        for vm in common.global_vms.values():
            manager = getattr(vm, 'v2x_manager', vm)
            values.append(float(getattr(manager, 'communication_range',
                                        100.0)))
        return min(values) if values else 100.0

    def _candidate_sender_ids(self, cluster):
        head_id = int(cluster.head_id)
        head_vm = common.global_vms[head_id]
        cluster_heads = self._cluster_heads()
        communication_range = self._communication_range()
        candidates = []
        for sender_id in sorted(common.global_vehicles.keys()):
            sender_id = int(sender_id)
            if sender_id == head_id:
                continue
            # Keep cluster heads in receiver role to avoid half-duplex-like
            # ambiguity and to preserve the inter-cluster late-fusion hierarchy.
            if sender_id in cluster_heads:
                continue
            sender_vm = common.global_vms.get(sender_id)
            if sender_vm is None:
                continue
            if calculate_distance(sender_vm, head_vm) > communication_range:
                continue
            candidates.append(sender_id)
        return candidates

    def _score_routed_candidate(self, cluster, sender_id, mode):
        candidate = self._score_candidate(cluster, sender_id, mode=mode)
        if candidate is None:
            return None
        in_cluster = sender_id in set(int(item) for item in cluster.members)
        candidate['in_cluster'] = in_cluster
        if not in_cluster:
            if candidate.get('instance_score', 0.0) < (
                    self.external_min_instance_score):
                return None
            penalty = (
                self.external_target_penalty if mode == 'target'
                else self.external_sender_penalty)
            candidate['score'] *= penalty
            # Strong instance support should still win over weak in-cluster
            # coverage, but only through the explicit instance term.
            candidate['score'] += 0.16 * candidate.get('instance_score', 0.0)
        return candidate

    def _best_routed_link(self, cluster, mode='coverage',
                          excluded_senders=None):
        excluded_senders = excluded_senders or set()
        best = None
        for sender_id in self._candidate_sender_ids(cluster):
            if sender_id in excluded_senders:
                continue
            candidate = self._score_routed_candidate(
                cluster,
                sender_id,
                mode=mode)
            if candidate is None:
                continue
            if best is None:
                best = candidate
                continue
            key = (
                candidate['score'],
                candidate.get('instance_score', 0.0),
                candidate.get('object_score', 0.0),
                candidate.get('coverage_score', 0.0),
                1 if candidate.get('in_cluster') else 0,
                -candidate['member_id'])
            best_key = (
                best['score'],
                best.get('instance_score', 0.0),
                best.get('object_score', 0.0),
                best.get('coverage_score', 0.0),
                1 if best.get('in_cluster') else 0,
                -best['member_id'])
            if key > best_key:
                best = candidate
        return best

    def _append_link(self, cluster, link, rb_occupancy, head_link_count):
        appended = super(CrossClusterInstanceSupportPotentialGame,
                         self)._append_link(
                             cluster,
                             link,
                             rb_occupancy,
                             head_link_count)
        if appended:
            head_id = int(cluster.head_id)
            logger.info(
                'Cross-cluster instance route head=%s sender=%s '
                'in_cluster=%s instance=%.4f score=%.4f',
                head_id,
                int(link['member_id']),
                bool(link.get('in_cluster', False)),
                link.get('instance_score', 0.0),
                link.get('score', 0.0))
        return appended

    def channel_game(self, max_iter=1):
        del max_iter
        self.strategies = {int(cluster.head_id): [] for cluster in self.clusters}
        rb_occupancy = defaultdict(int)
        head_link_count = Counter()
        used_senders = set()
        external_link_count = 0

        # Coverage layer: one in-cluster link per head. Cross-cluster routing
        # is deliberately not allowed here; otherwise the stable recall layer
        # collapses into a global nearest-view selector.
        for cluster in sorted(self.clusters, key=lambda item: int(item.head_id)):
            if self._next_available_channel(rb_occupancy) is None:
                break
            link = super(CrossClusterInstanceSupportPotentialGame,
                         self)._best_member_link(
                cluster,
                mode='coverage',
                excluded_members=used_senders)
            if link is not None:
                link['in_cluster'] = True
            if link is not None and self._append_link(
                    cluster,
                    link,
                    rb_occupancy,
                    head_link_count):
                used_senders.add(int(link['member_id']))

        max_links_per_head = max(1, int(getattr(self.p, 'head_rb_budget', 1)))
        while self._next_available_channel(rb_occupancy) is not None:
            candidates = []
            for cluster in self.clusters:
                head_id = int(cluster.head_id)
                if head_link_count[head_id] >= max_links_per_head:
                    continue
                if external_link_count < self.max_external_links_per_frame:
                    link = self._best_routed_link(
                        cluster,
                        mode='target',
                        excluded_senders=used_senders)
                else:
                    link = super(CrossClusterInstanceSupportPotentialGame,
                                 self)._best_member_link(
                                     cluster,
                                     mode='target',
                                     excluded_members=used_senders)
                    if link is not None:
                        link['in_cluster'] = True
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
                    item.get('instance_score', 0.0),
                    item.get('object_score', 0.0),
                    item.get('coverage_score', 0.0),
                    -int(item['cluster'].head_id)))
            if not self._append_link(best['cluster'], best, rb_occupancy,
                                     head_link_count):
                break
            used_senders.add(int(best['member_id']))
            if not best.get('in_cluster', False):
                external_link_count += 1

        link_count = 0
        selected_grid_count = 0
        external_links = 0
        for head_id, links in self.strategies.items():
            cluster = next(
                cluster for cluster in self.clusters
                if int(cluster.head_id) == int(head_id))
            members = set(int(item) for item in cluster.members)
            for member_id, _, _, grids in links:
                link_count += 1
                selected_grid_count += len(grids)
                if int(member_id) not in members:
                    external_links += 1
        self.convergence_stats = {
            'iterations': 1,
            'cluster_updates': len(self.clusters),
            'scheduled_links': link_count,
            'selected_grids': selected_grid_count,
            'used_rbs': sum(1 for count in rb_occupancy.values()
                            if count > 0),
            'reused_rbs': 0,
            'max_rb_occupancy': 1 if link_count else 0,
            'external_links': external_links,
            'converged': True,
        }
        return self.strategies
