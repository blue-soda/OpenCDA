# OpenCDA on Fedora 43 Runbook

Date: 2026-05-03
Environment:

- Host OS: Fedora 43
- Project root: `/home/sakakibara/Workspace/OpenCDA`
- Conda env: `opencda`
- Python: 3.7 in `/home/sakakibara/miniconda3/envs/opencda`
- Target command:

```bash
python ./opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug
```

## Goal

Make the target command run on this machine and record every blocking issue and fix along the way.

## Initial failure

The original run failed while importing `carla`:

```text
ImportError: libjpeg.so.8: cannot open shared object file: No such file or directory
```

## Findings so far

`cache/carla/libcarla.cpython-37m-x86_64-linux-gnu.so` depends on old ABI names not shipped by Fedora 43:

- `libjpeg.so.8`
- `libtiff.so.5`

Fedora 43 currently provides newer SONAMEs:

- `libjpeg.so.62`
- `libtiff.so.6`

The CARLA binary package bundled here is therefore not directly compatible with the host runtime.

## Validation performed

Checked the dependency list:

```bash
ldd cache/carla/libcarla.cpython-37m-x86_64-linux-gnu.so
```

Observed missing libraries:

```text
libjpeg.so.8 => not found
libtiff.so.5 => not found
```

After supplying those two libraries, a transitive dependency also appeared:

```text
libwebp.so.6 => not found
```

Using a local compatibility directory with these three libraries allowed `import carla` to succeed:

- `libjpeg.so.8`
- `libtiff.so.5`
- `libwebp.so.6`

## Current status

The target command now runs through successfully on this machine.

## Issues encountered and fixes

### 1. `carla` import failed on missing old image libraries

Original error:

```text
ImportError: libjpeg.so.8: cannot open shared object file: No such file or directory
```

Root cause:

- `cache/carla/libcarla.cpython-37m-x86_64-linux-gnu.so` was built against old SONAMEs.
- Fedora 43 only ships newer `libjpeg.so.62` and `libtiff.so.6`.

Fix applied:

- Copied compatibility libraries into `cache/carla/compat/`:
  - `libjpeg.so.8`
  - `libtiff.so.5`
  - `libwebp.so.6`
- Updated `cache/carla/__init__.py` to preload these libraries with `ctypes.CDLL(..., RTLD_GLOBAL)` before importing `libcarla`.

Validation:

```bash
/home/sakakibara/miniconda3/envs/opencda/bin/python -c "import carla; print(carla.__file__)"
```

This succeeded after the patch.

### 2. PyTorch failed because `libtorch_cpu.so` requested an executable stack

Error after the `carla` layer was fixed:

```text
ImportError: libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument
```

Root cause:

- The ELF `PT_GNU_STACK` program header in
  `/home/sakakibara/miniconda3/envs/opencda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so`
  had flags `RWE`.

Validation:

```bash
readelf -W -l /home/sakakibara/miniconda3/envs/opencda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so
```

Observed before patch:

```text
GNU_STACK ... RWE
```

Fix applied:

- Installed `execstack` to inspect ELF stack flags:

```bash
sudo dnf -y install execstack
```

- `execstack -c` could not rewrite this file because of the ELF layout, so the `PT_GNU_STACK` header was patched directly with a small Python script that changed the flags from `0x7` to `0x6`.
- Backup files were kept:
  - `libtorch_cpu.so.bak`
  - `libtorch_cpu.so.gnu_stack.bak`

Observed after patch:

```text
GNU_STACK ... RW
```

Validation:

```bash
/home/sakakibara/miniconda3/envs/opencda/bin/python -c "import torch; print(torch.__version__)"
```

This succeeded after the patch.

### 3. OpenCOOD model weights were Git LFS pointers, not real checkpoint data

Error after `torch` was fixed:

```text
_pickle.UnpicklingError: invalid load key, 'v'.
```

Root cause:

- `opencood/logs/pointpillar_early_fusion/latest.pth` was a Git LFS pointer text file starting with:

```text
version https://git-lfs.github.com/spec/v1
```

Fix applied:

- Installed Git LFS:

```bash
sudo dnf -y install git-lfs
git lfs install
git lfs pull --include='opencood/logs/pointpillar_early_fusion/latest.pth'
```

Validation:

- The file became a real binary file of about 26 MB instead of a text pointer.

### 4. `spconv` was installed for CUDA 10.2 while PyTorch used CUDA 11.3

Error after the checkpoint was fixed:

```text
ImportError: arg(): could not convert default argument 'workspace: tv::Tensor' ...
ImportError: generic_type: cannot initialize type "ExternalAllocator": an object with that name is already defined
```

Root cause:

- Environment had:
  - `torch 1.10.0+cu113`
  - `spconv-cu102 2.3.6`
- That CUDA mismatch broke the `spconv` Python bindings.

Fix applied:

```bash
/home/sakakibara/miniconda3/envs/opencda/bin/python -m pip install --force-reinstall spconv-cu113
```

Notes:

- This also updated `numpy` from `1.20.0` to `1.21.6`.
- `pip` warned that `opencda` pins `numpy==1.20.0`, but the final target command still ran successfully with `1.21.6`.

Validation:

```bash
/home/sakakibara/miniconda3/envs/opencda/bin/python - <<'PY'
import spconv
from spconv.utils import Point2VoxelCPU3d
print(spconv.__version__)
print("Point2VoxelCPU3d ok")
PY
```

This succeeded after reinstalling `spconv`.

### 5. CARLA client timed out during map switching

Observed error:

```text
time-out of 10000ms while waiting for the simulator
```

Then after increasing timeout:

```text
Failed to load Town03 from CARLA. If the map is already active, launch CARLA directly with that town or raise client_timeout.
Error: time-out of 60000ms while waiting for the simulator
```

Root cause:

- OpenCDA hardcoded `self.client.set_timeout(10.0)`.
- The scenario tried to call `load_world('Town03')` even when preloading the map in the simulator is more stable on this machine.
- The packaged CARLA assets for `Town03` do exist, so the earlier message saying the map was missing was misleading.

Fix applied:

- Updated `opencda/scenario_testing/utils/sim_api.py` to:
  - read `world.client_timeout` from config, defaulting to `10.0`
  - skip `load_world(town)` when the current map already matches the requested town
  - print a more accurate error message when loading fails
- Updated `opencda/scenario_testing/config_yaml/default.yaml` to set:

```yaml
world:
  client_timeout: 60.0
```

Runtime procedure used for the successful run:

```bash
/home/sakakibara/Programs/CARLA/CarlaUE4.sh /Game/Carla/Maps/Town03 -carla-rpc-port=2000 -carla-streaming-port=2001 -quality-level=Low
```

Validation:

```bash
/home/sakakibara/miniconda3/envs/opencda/bin/python - <<'PY'
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print(client.get_world().get_map().name)
PY
```

This reported `Town03`.

## Final successful run

The final target command:

```bash
python ./opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug
```

Completed successfully when CARLA was already running on `Town03`.

Observed successful tail output:

```text
Simulation is Over
Localization Evaluation Done.
Kinematics Evaluation Done.
Platooning Evaluation Done.
cp counter: 0
Evaluate final average precision results:
  - Fusion method: early
  - The Average Precision at IOU 0.3 is 0.00, The Average Precision at IOU 0.5 is 0.00, The Average Precision at IOU 0.7 is 0.00
Destroying all the actors...
Simulation closed
```

## Files changed in the repository

- `cache/carla/__init__.py`
- `cache/carla/compat/libjpeg.so.8`
- `cache/carla/compat/libtiff.so.5`
- `cache/carla/compat/libwebp.so.6`
- `opencda/scenario_testing/utils/sim_api.py`
- `opencda/scenario_testing/config_yaml/default.yaml`
- `docs/troubleshooting/fedora43_opencda_runbook.md`

## System-level changes made outside the repository

- Installed Fedora packages:
  - `execstack`
  - `git-lfs`
- Replaced Python packages in the `opencda` conda environment:
  - installed `spconv-cu113`
  - removed the previous active `spconv-cu102` installation during the reinstall
- Patched:
  - `/home/sakakibara/miniconda3/envs/opencda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so`

## Practical startup recipe

1. Start CARLA with `Town03` preloaded:

```bash
/home/sakakibara/Programs/CARLA/CarlaUE4.sh /Game/Carla/Maps/Town03 -carla-rpc-port=2000 -carla-streaming-port=2001 -quality-level=Low
```

2. In the `opencda` conda environment, run:

```bash
python ./opencda.py -t v2x_uav_carla --apply_ml --apply_cp --debug
```

## Residual notes

- The run emitted NumPy runtime warnings about mean of empty slice, but these did not stop the scenario.
- `opencood/opencood/utils/box_overlaps.c` and `setup.sh` were already modified in the working tree before this debugging session and were not touched here.
