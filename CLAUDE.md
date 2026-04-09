# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenCDA is a co-simulation framework for Cooperative Driving Automation (CDA) research that integrates CARLA and SUMO simulators. The repository contains two main components:

1. **OpenCDA** (`opencda/`): Full-stack cooperative driving simulation framework with perception, localization, planning, control, and V2X communication
2. **OpenCOOD** (`opencood/`): Cooperative detection framework for multi-agent perception research, supporting datasets like OPV2V and V2XSet

## Architecture

### OpenCDA Structure
- `opencda/core/`: Core autonomous driving modules
  - `sensing/`: Perception (camera, LiDAR), localization, prediction
  - `plan/`: Behavior planning, local planning, global route planning
  - `actuation/`: Vehicle control
  - `common/`: Vehicle manager, V2X manager, RSU manager, UAV manager, CAV world
  - `clustering/`: Vehicle clustering and resource allocation
  - `networking/`: V2X network simulation
  - `map/`: HD map management
  - `safety/`: Safety monitoring
- `opencda/application/`: CDA applications (platooning, cooperative perception)
- `opencda/scenario_testing/`: Test scenarios and configuration files
- `opencda/co_simulation/`: CARLA-SUMO and AirSim-CARLA co-simulation

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

# Run with UAV enabled (AirSim-CARLA co-simulation)
python opencda.py -t v2x_uav_carla --apply_cp --apply_ml --debug --uav

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
```

## UAV / AirSim-CARLA Co-Simulation

### Architecture
The UAV system uses AirSim to control drone flight with CARLA for visualization:
- **AirSim**: Controls actual drone physics and flight
- **CARLA**: Visualizes drone position (updates from AirSim state)

Reference: [keshuw95/airsim_carla_co-simulation](https://github.com/keshuw95/airsim_carla_co-simulation)

### Coordinate Conversion (CARLA ENU ↔ AirSim NED)
```python
# CARLA → AirSim
x_airsim = carla_location.y
y_airsim = -carla_location.x
z_airsim = -(carla_location.z + hover_offset)  # NED z is positive DOWN

# AirSim → CARLA
carla.x = -airsim.y_val
carla.y = airsim.x_val
carla.z = -airsim.z_val
```

### Yaw Conversion
```python
airsim_drone_yaw = vehicle_yaw - 90  # For tracking mode
```

### UAV Modes
- **static**: Drone stays at spawn position
- **tracking**: Drone follows a target vehicle (uses `target` ID from config, vehicle IDs start at 1)
- **navigation**: Drone moves to a destination waypoint

### UAV Configuration
```yaml
uav_base:
  takeoff_height: 60
  hover_offset: 20   # Height above target vehicle
  speed: 6           # Movement speed m/s
  update_interval: 0.033
  sensing:
    perception:
      activate: false
      lidar:
        upper_fov: 45.0
        lower_fov: -45.0
        channels: 64
        range: 150
        global_position: [0, 0, 0, 0, -90, 0]  # pitch=-90 for downward view

uav_list:
  - mode: "tracking"
    spawn_position: [3.00, -30.31, 60, 0, 0, 0]  # x, y, z, roll, pitch, yaw
    target: 1  # Vehicle ID to track (starts from 1)

  - mode: "navigation"
    spawn_position: [0, 0, 60, 0, 0, 0]
    destination: [100.0, -50.0, 60]  # Target waypoint
```

### LiDAR FOV for UAV
For downward-facing observation from UAV:
- `upper_fov: 45.0` and `lower_fov: -45.0` gives 90° total FOV
- With `pitch: -90` in `global_position`, LiDAR points downward
- This creates a cone-shaped scan of the ground below

## Key Concepts

### OpenCDA
- **CAV (Connected Automated Vehicle)**: Vehicles with V2X communication and automation
- **RSU (Roadside Unit)**: Infrastructure sensors for cooperative perception
- **UAV (Unmanned Aerial Vehicle)**: Drone for aerial surveillance, controlled via AirSim
- **V2X Manager**: Handles vehicle-to-everything communication
- **Platooning**: Cooperative longitudinal control for vehicle convoys
- **Cooperative Perception**: Sharing sensor data between CAVs and RSUs

### OpenCOOD
- **Fusion Strategies**: Early (raw data), Late (detection results), Intermediate (feature-level)
- **Spconv**: Sparse convolution library (supports both 1.2.1 and 2.x versions)
- **Compression**: Feature compression for bandwidth-efficient V2X communication
- **Noisy Setting**: Simulates localization errors and communication delays for realistic testing

## Configuration System

### OpenCDA
- Configuration files in `opencda/scenario_testing/config_yaml/`
  - `default.yaml` - Base template for scenarios
  - `enable_uav.yaml` - UAV base configuration
  - `v2x_uav_carla.yaml` - UAV scenario configuration
- YAML configs support merge/override pattern

## Dependencies

### OpenCDA
- CARLA simulator (0.9.12 supported, check for latest compatibility)
- AirSim (for UAV support)
- Python 3.7+
- Key packages: open3d, opencv-python, shapely, omegaconf, yolov5

### OpenCOOD
- PyTorch
- Spconv (1.2.1 or 2.x)
- Key packages: open3d, opencv-python, numba, einops, timm

## Important Notes

- OpenCDA is primarily tested on custom maps and CARLA Town06 - robustness on other maps not guaranteed
- OpenCOOD bandwidth requirement should be <2.7Mbps for practical V2X deployment (27Mbps transmission rate at 10Hz)
- For V2XSet training: start with perfect setting, add compression, then fine-tune on noisy setting
- Both projects are for non-commercial research only under their respective licenses
