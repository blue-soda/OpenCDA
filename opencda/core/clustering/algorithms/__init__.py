"""Clustering algorithms factory."""

from typing import Dict, Type

# Import from existing locations for now
from opencda.core.clustering.algorithms.clustering.coalition_game import CoalitionGame
from opencda.core.clustering.algorithms.clustering.naive_cluster import NaiveCluster

# Clustering algorithms registry
CLUSTERING_ALGORITHMS: Dict[str, Type] = {
    'coalition_game': CoalitionGame,
    'naive': NaiveCluster,
}

def create_clustering_algorithm(name: str, cav_world):
    """Create clustering algorithm by name."""
    if name not in CLUSTERING_ALGORITHMS:
        raise ValueError(f"Unknown clustering algorithm: {name}")
    return CLUSTERING_ALGORITHMS[name](cav_world)
