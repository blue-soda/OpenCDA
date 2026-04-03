# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenCDA is a co-simulation framework for Cooperative Driving Automation (CDA) research that integrates CARLA and SUMO simulators. The repository contains two main components:

1. **OpenCDA** (`opencda/`): Full-stack cooperative driving simulation framework with perception, localization, planning, control, and V2X communication
2. **OpenCOOD** (`opencood/`): Cooperative detection framework for multi-agent perception research, supporting datasets like OPV2V and V2XSet

## Architecture

### OpenCDA Structure (Updated 2026-03-31)
- `opencda/core/`: Core autonomous driving modules
  - `sensing/`: Perception (camera, LiDAR), localization, prediction
  - `plan/`: Behavior planning, local planning, global route planning
  - `actuation/`: Vehicle control
  - `common/`: Vehicle manager, V2X manager, RSU manager, CAV world, **ConfigManager**
  - `clustering/`: **Vehicle clustering and resource allocation (refactored)**
    - `base/`: Abstract base classes for algorithms
    - `algorithms/`: Clustering and resource allocation implementations
    - `managers/`: Clustering managers
    - `utils/`: Utility functions (grid operations, vehicle queries, metrics)
  - `networking/`: **V2X network simulation (refactored from customize/v2x)**
    - `resource_allocation/`: Resource allocation logic
    - `ns3_integration/`: NS3 co-simulation bridge
    - `statistics/`: Communication statistics
  - `map/`: HD map management
  - `safety/`: Safety monitoring
- `opencda/application/`: CDA applications (platooning, cooperative perception)
- `opencda/scenario_testing/`: Test scenarios and configuration files
- `opencda/co_simulation/`: CARLA-SUMO and AirSim-CARLA co-simulation
- **`opencda/customize/`**: Legacy extended modules (being migrated to core/)
  - `ml_libs/`: Machine learning utilities

### OpenCOOD Structure
- `opencood/opencood/`: Core cooperative detection framework
  - Data loaders for OPV2V, V2XSet datasets
  - 3D detection backbones (PointPillar, VoxelNet, SECOND, PIXOR)
  - Multi-agent fusion models (Attentive Fusion, V2VNet, F-Cooper, etc.)
- `opencood/tools/`: Training and inference scripts
- `opencood/hypes_yaml/`: Model configuration files
- `opencood/logreplay/`: Log replay toolbox for OPV2V dataset

## Development Commands

### OpenCDA - Running Scenarios
```bash
# Run a scenario test (from repository root)
python opencda/scenario_testing/platoon_joining_2lanefree_carla.py

# Run with OpenScenario format
python opencda/scenario_testing/openscenario_carla.py

# Configuration files are in opencda/scenario_testing/config_yaml/
# default.yaml serves as the base template
```

### OpenCOOD - Training and Testing
```bash
# Train a model (from repository root)
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/second_early_fusion.yaml

# Train on multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train.py --hypes_yaml <config_file>

# Continue training from checkpoint
python opencood/tools/train.py --model_dir <checkpoint_folder>

# Run inference
python opencood/tools/inference.py --model_dir <checkpoint_folder> --fusion_method <early|late|intermediate>

# Visualize data sequences
cd opencood
python opencood/visualization/vis_data_sequence.py --color_mode intensity
```

## Configuration System

### OpenCDA (Updated 2026-03-31)
- **Centralized configuration** via `ConfigManager` in `opencda/core/common/config_manager.py`
- Configuration files in `opencda/scenario_testing/config_yaml/`
  - `default.yaml` - Base template for scenarios
  - `networking_clustering.yaml` - V2X network and clustering configuration
- Type-safe configuration with dataclasses
- No hardcoded parameters

**Usage:**
```python
from opencda.core.common.config_manager import ConfigManager

config = ConfigManager.from_yaml('path/to/config.yaml')
print(config.v2x_network.subchannel_num)  # 10
print(config.clustering.algorithm)         # "coalition_game"
```

**Legacy:** YAML files still support override pattern where other configs override `default.yaml` parameters

### OpenCOOD
- Model configs in `opencood/hypes_yaml/` define backbone, fusion strategy, training parameters
- Each trained model saves its config in the checkpoint folder as `config.yaml`

## Key Concepts

### OpenCDA
- **CAV (Connected Automated Vehicle)**: Vehicles with V2X communication and automation
- **RSU (Roadside Unit)**: Infrastructure sensors for cooperative perception
- **V2X Manager**: Handles vehicle-to-everything communication
- **Platooning**: Cooperative longitudinal control for vehicle convoys
- **Cooperative Perception**: Sharing sensor data between CAVs and RSUs

### OpenCOOD
- **Fusion Strategies**: Early (raw data), Late (detection results), Intermediate (feature-level)
- **Spconv**: Sparse convolution library (supports both 1.2.1 and 2.x versions)
- **Compression**: Feature compression for bandwidth-efficient V2X communication
- **Noisy Setting**: Simulates localization errors and communication delays for realistic testing

## Dependencies

### OpenCDA
- CARLA simulator (0.9.12 supported, check for latest compatibility)
- Python 3.7+
- Key packages: open3d, opencv-python, shapely, omegaconf, yolov5

### OpenCOOD
- PyTorch
- Spconv (1.2.1 or 2.x)
- Key packages: open3d, opencv-python, numba, einops, timm

## Custom Extensions (opencda/customize/)

This repository contains significant extensions to the original OpenCDA framework focused on clustering-based cooperative perception and V2X resource allocation.

### Clustering Algorithms (`customize/core/clustering/`)
Implements 10+ vehicle clustering and resource allocation algorithms:
- **Coalition Game Theory**: `coalition_game.py` - game-theoretic cluster formation
- **Potential Game**: `potential_game.py` (475 lines) - channel allocation via potential games
- **PCS Algorithm**: `pcs.py` (439 lines) - priority-based clustering
- **MWS**: `mws.py` - maximum weighted sum approach
- **Naive/Random**: Baseline clustering methods
- **Graph-based**: `weighted_conflict_graph_coloring_algorithm.py`, `spatio_temporal_similarity_algorithm.py`

Key managers:
- `ClusteringV2XManager`: Extends V2XManager with cluster head election and member management
- `ClusteringPerceptionManager`: Handles cluster-based cooperative perception
- `ClusteringScheduler`: Resource scheduling within clusters

### V2X Network Simulation (`customize/core/v2x/`)
- **NS3 Co-simulation**: Bridge between CARLA and NS3 network simulator
  - `ns3_co_simulation/bridge/`: CARLA-NS3 communication bridge
  - `ns3_co_simulation/carla/`: CARLA connector and vehicle data handling
- **Network Manager**: `network_manager.py` (462 lines) - handles V2X communication with realistic network models
- **Scheduler**: `scheduler.py` - resource allocation scheduling (channel, time slots)

### Refactoring Improvements (2026-03-31)

✅ **Completed Improvements**:

1. **Eliminated global variables**:
   - Created `ClusteringContext` for dependency injection
   - Replaced `common.global_vehicles` with context-based access
   - Each algorithm instance has isolated state

2. **Configuration-driven architecture**:
   - All parameters moved to YAML files
   - `ConfigManager` provides type-safe configuration access
   - No hardcoded network or clustering parameters

3. **Modular structure**:
   - Clustering algorithms separated into `algorithms/clustering/` and `algorithms/resource_allocation/`
   - Base classes: `ClusteringAlgorithm`, `ResourceAllocationAlgorithm`, `Cluster`
   - Utility functions extracted: grid operations, vehicle queries, metrics

4. **Import path updates**:
   - Core files updated to use new paths
   - `opencda.customize.core.clustering` → `opencda.core.clustering`
   - `opencda.customize.core.v2x` → `opencda.core.networking`

🔄 **In Progress**:
- Algorithm migration to new base classes
- Manager refactoring to remove static variables
- Code style unification

⏳ **Pending**:
- Perception module splitting (sensors, grid, visualization)
- Networking module decomposition (allocator, bridge, stats)
- Complete import path migration

See `REFACTORING_HISTORY.md` for detailed migration guide.

## Important Notes

- OpenCDA is primarily tested on custom maps and CARLA Town06 - robustness on other maps not guaranteed
- OpenCOOD bandwidth requirement should be <2.7Mbps for practical V2X deployment (27Mbps transmission rate at 10Hz)
- For V2XSet training: start with perfect setting, add compression, then fine-tune on noisy setting
- Both projects are for non-commercial research only under their respective licenses
