"""Grid operation utilities for clustering algorithms."""

from typing import Set, Dict, List
import numpy as np


def merge_grid_sets(grid_sets: List[Set[int]]) -> Set[int]:
    """Merge multiple grid sets into one."""
    if not grid_sets:
        return set()
    return set().union(*grid_sets)


def compute_grid_intersection(grid_sets: List[Set[int]]) -> Set[int]:
    """Compute intersection of multiple grid sets."""
    if not grid_sets:
        return set()
    result = grid_sets[0].copy()
    for gs in grid_sets[1:]:
        result &= gs
    return result


def compute_grid_density_score(grids: Dict[int, int], threshold: float) -> float:
    """Compute density score for grids above threshold."""
    if not grids:
        return 0.0
    return sum(1 for density in grids.values() if density > threshold) / len(grids)


def get_high_density_grids(grids: Dict[int, int], threshold: float) -> Set[int]:
    """Get grid IDs with density above threshold."""
    return {gid for gid, density in grids.items() if density > threshold}
