"""Vehicle query utilities for clustering algorithms."""

from typing import List, Dict, Any, Tuple
import numpy as np


def get_vehicle_position(vehicle_manager) -> Tuple[float, float, float]:
    """Get vehicle position as (x, y, z) tuple."""
    loc = vehicle_manager.localizer.get_ego_pos().location
    return (loc.x, loc.y, loc.z)


def get_vehicle_velocity(vehicle_manager) -> Tuple[float, float, float]:
    """Get vehicle velocity as (vx, vy, vz) tuple."""
    vel = vehicle_manager.localizer.get_ego_spd()
    return (vel.x, vel.y, vel.z)


def compute_distance(pos1: Tuple[float, float, float],
                     pos2: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two positions."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def get_vehicles_within_range(ego_pos: Tuple[float, float, float],
                               vehicle_managers: Dict[int, Any],
                               max_range: float) -> List[int]:
    """Get vehicle IDs within range of ego position."""
    nearby = []
    for vid, vm in vehicle_managers.items():
        pos = get_vehicle_position(vm)
        if compute_distance(ego_pos, pos) <= max_range:
            nearby.append(vid)
    return nearby
