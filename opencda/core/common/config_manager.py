"""
Unified configuration management system for OpenCDA.
Eliminates hardcoded parameters and provides type-safe config access.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml


@dataclass
class V2XNetworkConfig:
    """V2X network communication configuration."""
    subchannel_num: int = 10
    subchannel_bandwidth: float = 0.180  # MHz
    min_sinr_threshold: float = 3.0  # dB
    time_slot: float = 0.05  # seconds
    max_vehicle_num: int = 30
    max_packet_size: int = 10000  # bytes
    reference_distance: float = 10.0  # meters
    path_loss_exponent: float = 2.0
    transmission_power: float = 23.0  # dBm
    noise_power: float = -114.0  # dBm


@dataclass
class NS3Config:
    """NS3 co-simulation configuration."""
    enable: bool = False
    bridge_port: int = 5555
    sync_interval: float = 0.05


@dataclass
class ClusteringConfig:
    """Clustering algorithm configuration."""
    algorithm: str = "coalition_game"  # coalition_game, naive, similarity
    cluster_interval: int = 4
    enable_scheduler: bool = True
    max_cluster_size: int = 10
    enable_topology_trigger_gate: bool = False
    topology_periodic_guard: int = 0


@dataclass
class ResourceAllocationConfig:
    """Resource allocation algorithm configuration."""
    algorithm: str = "potential_game"  # potential_game, pcs, mws, random
    enable: bool = True


@dataclass
class ConfigManager:
    """Central configuration manager."""
    v2x_network: V2XNetworkConfig = field(default_factory=V2XNetworkConfig)
    ns3: NS3Config = field(default_factory=NS3Config)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    resource_allocation: ResourceAllocationConfig = field(default_factory=ResourceAllocationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ConfigManager':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            v2x_network=cls._parse_v2x_network(config_dict.get('v2x_network', {})),
            ns3=cls._parse_ns3(config_dict.get('ns3', )),
            clustering=cls._parse_clustering(config_dict.get('clustering', {})),
            resource_allocation=cls._parse_resource_allocation(config_dict.get('resource_allocation', {}))
        )

    @staticmethod
    def _parse_v2x_network(config: Dict) -> V2XNetworkConfig:
        return V2XNetworkConfig(**{k: v for k, v in config.items() if k in V2XNetworkConfig.__annotations__})

    @staticmethod
    def _parse_ns3(config: Dict) -> NS3Config:
        return NS3Config(**{k: v for k, v in config.items() if k in NS3Config.__annotations__})

    @staticmethod
    def _parse_clustering(config: Dict) -> ClusteringConfig:
        return ClusteringConfig(**{k: v for k, v in config.items() if k in ClusteringConfig.__annotations__})

    @staticmethod
    def _parse_resource_allocation(config: Dict) -> ResourceAllocationConfig:
        return ResourceAllocationConfig(**{k: v for k, v in config.items() if k in ResourceAllocationConfig.__annotations__})
