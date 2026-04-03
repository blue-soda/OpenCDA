"""
Abstract base class for clustering algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Set
from ..context import ClusteringContext


class Cluster:
    """Represents a vehicle cluster."""
    def __init__(self, head_id: int, members: Set[int]):
        self.head_id = head_id
        self.members = members

    def __repr__(self):
        return f"Cluster(head={self.head_id}, members={self.members})"


class ClusteringAlgorithm(ABC):
    """Base class for all clustering algorithms."""

    def __init__(self, context: ClusteringContext):
        self.context = context
        self.clusters: List[Cluster] = []

    @abstractmethod
    def run(self) -> List[Cluster]:
        """Execute clustering algorithm and return list of clusters."""
        pass

    def update_cluster_states(self):
        """Update V2X managers with cluster state information."""
        for cluster in self.clusters:
            for member_id in cluster.members:
                v2x_mgr = self.context.get_v2x_manager(member_id)
                if v2x_mgr and hasattr(v2x_mgr, 'cluster_state'):
                    v2x_mgr.cluster_state['head_id'] = cluster.head_id
                    v2x_mgr.cluster_state['member_ids'] = cluster.members.copy()
