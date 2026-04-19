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

## NS3 Co-Simulation Debugging

### Startup Order
Use this order when debugging CARLA + NS3 co-simulation:

1. Kill stale CARLA / OpenCDA / NS3 processes.
2. Start `CarlaUE4.exe` and wait until the simulator is fully loaded.
3. Start the NS3 executable in WSL and redirect to a fresh log file.
4. Start OpenCDA with `--network`.

If OpenCDA reports either of the following:
- `Town03 is not found in your CARLA repo`
- `time-out of 10000ms while waiting for the simulator`

then assume CARLA has crashed or has not finished loading yet. In that case, kill all stale `CarlaUE4` processes and relaunch CARLA before retrying OpenCDA.

### Process Cleanup Commands

Run these from PowerShell before each clean repro:

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
wsl.exe bash -lc "fuser -k 5556/tcp 5557/tcp 2>/dev/null; pkill -f ns3.42-main-default || true"
```

If you only want to restart CARLA:

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process 'C:\Workspace\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe'
```

### NS3 Run Command
Run NS3 from PowerShell, but execute the binary inside WSL. This is the command pattern actually used during debugging:

**IMPORTANT**: Always kill stale processes on ports 5556 and 5557 before starting NS3:
```powershell
wsl.exe bash -lc "fuser -k 5556/tcp 5557/tcp 2>/dev/null || true"
```

Then start NS3 with `--carlaHost=172.26.174.149` so it connects back to OpenCDA on the correct interface:

```powershell
$ns3Log = "/home/sakakibara/Workspace/carla-ns3-co-simulation/log_run_$(Get-Date -Format yyyyMMdd_HHmmss).txt"
Start-Process powershell -ArgumentList "-NoExit","-Command","wsl.exe bash -lc 'cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 --carlaHost=172.26.174.149 > $ns3Log 2>&1'"
```

If you prefer to run it in the current terminal instead of a new window:

```powershell
wsl.exe bash -lc "cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 --carlaHost=172.26.174.149 > /home/sakakibara/Workspace/carla-ns3-co-simulation/log.txt 2>&1"
```

If you see `bind failed: Address already in use` in the NS3 output, it means a stale process is holding port 5556. Kill it with:
```powershell
wsl.exe bash -lc "fuser -k 5556/tcp 5557/tcp 2>/dev/null; pkill -f ns3.42-main-default || true"
```

Reference paths:
- `/home/sakakibara/Workspace/carla-ns3-co-simulation/ns3/vanet/main.cc`
- `/home/sakakibara/Workspace/carla-ns3-co-simulation/log.txt`

### OpenCDA Run Command
Use the actual scenario command that was used in this debug session. Run it from `C:\Workspace\OpenCDA`:

```powershell
cd C:\Workspace\OpenCDA
conda run -n opencda python opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug --network
```

Note: You must use `conda run -n opencda python` instead of bare `python` to activate the correct conda environment.

### NS3 Build / Compilation

If you modify `main.cc` or any NS3 source code, you must rebuild NS3 before running:

```bash
# From WSL
cd /home/sakakibara/Workspace/carla-ns3-co-simulation/ns-3-dev
./ns3 build
```

The compiled binary is at:
```
/home/sakakibara/Workspace/carla-ns3-co-simulation/ns-3-dev/build/scratch/vanet/ns3.42-main-default
```

### NS3 Startup (Recommended)

Start NS3 in WSL and keep it running in background with log capture:

```bash
# Kill any stale NS3 processes first
pkill -f ns3.42-main-default || true

# Start NS3 with unbuffered output
cd /home/sakakibara/Workspace/carla-ns3-co-simulation
stdbuf -oL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 2>&1
```

Wait until you see `[INFO] Waiting for Carla on port 5556...` before starting OpenCDA.

### Verify NS3 is Listening

```bash
# Check if NS3 is listening on port 5556
netstat -tlnp | grep 5556
# Should show: tcp 0 0 0.0.0.0:5556 0.0.0.0:* LISTEN <pid>/ns3.42-main-default
```

### NS3 Network Configuration

NS3 listens on `0.0.0.0:5556` (all interfaces). OpenCDA connects via the WSL host IP defined in:
- `opencda/core/networking/ns3_co_simulation/config/settings.py` → `NS3_HOST`
- Default WSL NAT IP: `172.26.174.149`

If OpenCDA cannot connect, verify:
1. `NS3_HOST` in settings.py matches WSL's IP (check with `hostname -I` in WSL)
2. Windows Firewall is not blocking port 5556
3. WSL port forwarding is working

### Logs to Inspect
- OpenCDA logs:
  - `C:\Workspace\OpenCDA\opencda\log\`
- NS3 logs:
  - `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log.txt`
  - `\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log_run_*.txt`

Do not read the full logs at once. First locate the newest log, then search by keywords.

### Find the Latest Logs

OpenCDA side:

```powershell
Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

NS3 side:

```powershell
Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

### Targeted Log Searches

OpenCDA side:

```powershell
$log = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "Connected to NS-3|sync_with_ns3: sync successful|FIRST time|AGAIN|schedule link=|send_transfer_requests|no communication_requests|uploaded its data|Received size:|waiting exceeded threshold|timeout, current_time_slot|history_try_volume|communication_requests now has" $log
```

NS3 side:

```powershell
$log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "Transfer request:|MANUAL_CMD_ADD|MANUAL_CMD_CHECK|MANUAL_LOGICAL_MAP|PSCCH_DECODE_OK|PSCCH_DECODE_FAIL|PSSCH_DECODE_OK|PSSCH_DECODE_FAIL|SCI2_DECODE_FAIL|cam_received|sync_ack|vehicles_num|vehicles_position" $log
```

If you need counts instead of raw matches:

```powershell
$log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
@("MANUAL_CMD_ADD","MANUAL_CMD_CHECK","MANUAL_LOGICAL_MAP","PSCCH_DECODE_OK","PSCCH_DECODE_FAIL","PSSCH_DECODE_OK","PSSCH_DECODE_FAIL","SCI2_DECODE_FAIL","cam_received") |
  ForEach-Object {
    $count = (rg -c $_ $log)
    "{0} = {1}" -f $_, $count
  }
```

If you need to compare OpenCDA send-side and receive-side events in one file quickly:

```powershell
$log = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "FIRST time|send_transfer_requests|Received size:|uploaded its data|history_try_volume|cp counter" $log
```

### What to Verify First
- Time sync:
  - `carla_time` and `ns3_time` should match in `sync_ack`
- Whether OpenCDA is generating `FIRST time` uploads and actual `send_transfer_requests`
- Whether NS3 receives the same batch of transfer requests and produces matching `cam_received`
- Whether the failure is in NS3 wireless delivery or in OpenCDA aggregation / state transition

### Typical Debugging Workflow

Use this template when you need to reproduce and count what happened in one co-simulation run.

#### Step 1: Clean restart

```powershell
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
wsl.exe bash -lc "pkill -f ns3.42-main-default || true"
Start-Process 'C:\Workspace\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe'
```

Wait for CARLA to finish loading, then start NS3 and OpenCDA.

#### Step 2: Start NS3 with a fresh log

```powershell
$ns3Log = "/home/sakakibara/Workspace/carla-ns3-co-simulation/log_run_$(Get-Date -Format yyyyMMdd_HHmmss).txt"
Start-Process powershell -ArgumentList "-NoExit","-Command","wsl.exe bash -lc 'cd /home/sakakibara/Workspace/carla-ns3-co-simulation && stdbuf -oL -eL ./ns-3-dev/build/scratch/vanet/ns3.42-main-default --simTime=600.0 > $ns3Log 2>&1'"
```

#### Step 3: Start OpenCDA

```powershell
cd C:\Workspace\OpenCDA
python opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug --network
```

If OpenCDA reports CARLA timeout or missing town, kill all `CarlaUE4` processes and restart CARLA before retrying.

#### Step 4: Find the newest logs after the run

```powershell
$opencdaLog = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$ns3Log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$opencdaLog
$ns3Log
```

#### Step 5: Count the four key numbers

The most useful first-pass comparison is:
- how many uploads OpenCDA tried to send
- how many transfer requests NS3 accepted
- how many packets NS3 reported as received
- how many uploads OpenCDA finally marked as completed

OpenCDA send-side and completion-side:

```powershell
$opencdaLog = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
"FIRST time = $((rg -c 'FIRST time' $opencdaLog))"
"send_transfer_requests = $((rg -c 'send_transfer_requests' $opencdaLog))"
"uploaded its data = $((rg -c 'uploaded its data' $opencdaLog))"
rg -n "FIRST time|send_transfer_requests|uploaded its data|history_try_volume|cp counter" $opencdaLog
```

NS3 receive-side:

```powershell
$ns3Log = (Get-ChildItem '\\wsl.localhost\Ubuntu\home\sakakibara\Workspace\carla-ns3-co-simulation\log*.txt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
"Transfer request = $((rg -c 'Transfer request:' $ns3Log))"
"cam_received = $((rg -c 'cam_received' $ns3Log))"
"PSCCH_DECODE_OK = $((rg -c 'PSCCH_DECODE_OK' $ns3Log))"
"PSCCH_DECODE_FAIL = $((rg -c 'PSCCH_DECODE_FAIL' $ns3Log))"
rg -n "Transfer request:|cam_received|PSCCH_DECODE_OK|PSCCH_DECODE_FAIL|PSSCH_DECODE_OK|PSSCH_DECODE_FAIL|MANUAL_LOGICAL_MAP" $ns3Log
```

#### Step 6: Interpret the mismatch

Use the counts this way:

- If `FIRST time` exists but `send_transfer_requests` is missing, the break is inside OpenCDA before requests are flushed to NS3.
- If `send_transfer_requests` exists but NS3 has no `Transfer request:`, the break is in the bridge or socket path.
- If NS3 has `Transfer request:` but `cam_received` is much lower than expected, the break is in NS3 wireless delivery or receive-side decode.
- If NS3 has enough `cam_received` but OpenCDA still does not print `uploaded its data`, the break is in OpenCDA aggregation or upload state cleanup.
- If `history_try_volume` only has the first slot non-zero, check whether there were actually later `send_transfer_requests` before blaming the wireless layer.

#### Step 7: For second-round CP failures, search only the state-machine keywords

```powershell
$opencdaLog = (Get-ChildItem C:\Workspace\OpenCDA\opencda\log\opencda_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
rg -n "preparing data from|communicate: set uploading_data|FIRST time|AGAIN|schedule link=|communication_requests now has|send_transfer_requests|no communication_requests|waiting exceeded threshold|uploaded its data" $opencdaLog
```

This is the fastest way to confirm whether the second CP round prepared data but never re-entered `schedule -> transfer_requests`.

### Known Debugging Conclusions
- A real NS3 time-sync bug existed before and was fixed in the NS3 repo. The old failure mode was `sync_ack` returning while NS3 time had already run far ahead of CARLA time.
- The OpenCDA bridge listener also had a real startup bug: it used a one-shot `accept()` timeout and could exit before NS3 connected back.
- `subchannel_start=-1` in the current NS3 path should not be trusted as a safe default during debugging. In observed runs it behaved like overlapping default allocations and caused asymmetric delivery.
- Even after forcing explicit `subchannel_start` and `subchannel_num=1`, one uplink can still stall partway through. That means the remaining bug is inside the NS3 NR sidelink scheduling / delivery path, not only in the OpenCDA fallback logic.

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
