"""
Clustering context object to replace global variables.
Provides clean dependency injection for clustering algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ClusteringContext:
    """
    Clustering algorithm execution context.
    Replaces global variables (global_vehicles, global_vms, global_ego_id).

    Attributes:
        cav_world: Reference to the CAV world object
        vehicle_managers: Dictionary mapping vehicle_id -> VehicleManager
        v2x_managers: Dictionary mapping vehicle_id -> V2XManager
        ego_id: Current ego vehicle ID
    """
    cav_world: Any
    vehicle_managers: Dict[int, Any] = field(default_factory=dict)
    v2x_managers: Dict[int, Any] = field(default_factory=dict)
    ego_id: Optional[int] = None

    def initialize(self):
        """Initialize context from cav_world."""
        self.vehicle_managers = self.cav_world.get_vehicle_managers()
        self.v2x_managers = {vid: vm.v2x_manager for vid, vm in self.vehicle_managers.items()}

    def get_vehicle_manager(self, vid: int):
        """Get vehicle manager by ID."""
        return self.vehicle_managers.get(vid)

    def get_v2x_manager(self, vid: int):
        """Get V2X manager by ID."""
        return self.v2x_managers.get(vid)

    def get_all_vehicle_ids(self):
        """Get all vehicle IDs."""
        return list(self.vehicle_managers.keys())
