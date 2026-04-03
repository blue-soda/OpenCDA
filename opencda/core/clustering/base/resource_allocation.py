"""
Abstract base class for resource allocation algorithms.
"""

from abc import ABC, abstractmethod
from typing import List
from .clustering_algorithm import Cluster
from ..context import ClusteringContext


class ResourceAllocationAlgorithm(ABC):
    """Base class for resource allocation algorithms."""

    def __init__(self, context: ClusteringContext):
        self.context = context
        self.clusters: List[Cluster] = []

    def set_clusters(self, clusters: List[Cluster]):
        """Set clusters for resource allocation."""
        self.clusters = clusters

    @abstractmethod
    def run(self) -> bool:
        """Execute resource allocation and return success status."""
        pass

    def clear_allocation(self):
        """Clear previous resource allocation."""
        pass
