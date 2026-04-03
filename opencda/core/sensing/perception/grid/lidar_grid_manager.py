"""LiDAR grid management system."""

import numpy as np
from typing import Dict, Set, Tuple
from collections import defaultdict
from shapely.geometry import box


class LidarGridManager:
    """Manages grid-based representation of LiDAR point clouds."""

    def __init__(self, grid_size: float = 10.0, lidar_range: float = 100.0, density_threshold: float = 2.0):
        """
        Initialize grid manager.

        Args:
            grid_size: Size of each grid cell in meters
            lidar_range: Maximum range for lidar in meters
            density_threshold: Density threshold for high/low classification
        """
        self.grid_size = grid_size
        self.lidar_range = lidar_range
        self.density_threshold = density_threshold
        self.required_perception_range = lidar_range * 3

        self.perception_grids = self._generate_perception_grids()
        self.sens_grids = set()
        self.req_grids = set()
        self.high_density_grids = set()
        self.low_density_grids = set()
        self.grid_local_points = defaultdict(list)
        self.grid_density_dict = {}

    def _generate_perception_grids(self) -> Dict[int, box]:
        """Generate perception grid cells."""
        grids = {}
        r = self.required_perception_range
        for x in np.arange(-r, r, self.grid_size):
            for y in np.arange(-r, r, self.grid_size):
                grid_id = self._point_to_grid_id(x, y)
                grids[grid_id] = box(x, y, x + self.grid_size, y + self.grid_size)
        return grids

    def _point_to_grid_id(self, x: float, y: float) -> int:
        """Convert point coordinates to grid ID."""
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        return grid_x * 10000 + grid_y

    def update_grids(self, points: np.ndarray):
        """Update grid densities from point cloud."""
        self.grid_local_points.clear()
        self.grid_density_dict.clear()

        for point in points:
            x, y = point[0], point[1]
            grid_id = self._point_to_grid_id(x, y)
            if grid_id in self.perception_grids:
                self.grid_local_points[grid_id].append(point)

        for grid_id, points_list in self.grid_local_points.items():
            density = len(points_list) / (self.grid_size * self.grid_size)
            self.grid_density_dict[grid_id] = density

        self.high_density_grids = {gid for gid, d in self.grid_density_dict.items() if d > self.density_threshold}
        self.low_density_grids = {gid for gid, d in self.grid_density_dict.items() if d <= self.density_threshold}

    def get_grid_center(self, grid_id: int) -> Tuple[float, float]:
        """Get center coordinates of a grid."""
        grid_x = grid_id // 10000
        grid_y = grid_id % 10000
        return (grid_x * self.grid_size + self.grid_size / 2, grid_y * self.grid_size + self.grid_size / 2)
