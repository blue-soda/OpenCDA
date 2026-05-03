#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-opencda}"
CARLA_VERSION="${CARLA_VERSION:-0.9.11}"
INSTALL_ML="${INSTALL_ML:-1}"
TORCH_VARIANT="${TORCH_VARIANT:-cu113}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
AUTO_FIX_DIAGNOSTICS="${AUTO_FIX_DIAGNOSTICS:-1}"

log() {
  echo "[setup] $*"
}

warn() {
  echo "[setup][warn] $*" >&2
}

fail() {
  echo "[setup][error] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

activate_conda() {
  need_cmd conda
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME" || fail "Failed to activate conda environment: $CONDA_ENV_NAME"
}

ensure_conda_env() {
  log "Checking conda environment: $CONDA_ENV_NAME"
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    log "Conda environment already exists."
    return
  fi

  log "Creating conda environment from environment.yml"
  conda env create -f environment.yml || fail "Failed to create conda environment from environment.yml"
}

install_python_dependencies() {
  log "Installing core Python dependencies"
  python -m pip install --upgrade pip setuptools wheel || fail "Failed to upgrade pip/setuptools/wheel"
  python -m pip install -r requirements.txt || fail "Failed to install root requirements"

  if [[ "$INSTALL_ML" == "1" ]]; then
    log "Installing ML dependencies for PyTorch variant: $TORCH_VARIANT"
    python -m pip install \
      "torch==1.10.0+${TORCH_VARIANT}" \
      "torchvision==0.11.1+${TORCH_VARIANT}" \
      "torchaudio==0.10.0+${TORCH_VARIANT}" \
      -f "https://download.pytorch.org/whl/${TORCH_VARIANT}/torch_stable.html" \
      --no-deps || fail "Failed to install PyTorch wheels for ${TORCH_VARIANT}"
    python -m pip install -r requirements_ml.txt || fail "Failed to install ML requirements"
  else
    log "Skipping ML dependency installation because INSTALL_ML=$INSTALL_ML"
  fi
}

develop_install_opencda() {
  log "Installing OpenCDA in develop mode"
  python setup.py develop || fail "Failed to install OpenCDA in develop mode"
}

install_carla_from_egg() {
  if python -c "import carla" >/dev/null 2>&1; then
    log "carla already importable; skipping CARLA Python API installation"
    return
  fi

  [[ -n "${CARLA_HOME:-}" ]] || fail "CARLA_HOME must be set when carla is not already importable"

  local carla_egg="$CARLA_HOME/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg"
  [[ -f "$carla_egg" ]] || fail "CARLA egg not found: $carla_egg"

  local cache_dir="$ROOT_DIR/cache"
  mkdir -p "$cache_dir"

  log "Refreshing cached CARLA Python API from $carla_egg"
  rm -rf "$cache_dir/EGG-INFO" \
         "$cache_dir/carla" \
         "$cache_dir/carla-${CARLA_VERSION}-py3.7-linux-x86_64" \
         "$cache_dir"/carla-"${CARLA_VERSION}"-py3.7-linux-x86_64.egg
  cp "$carla_egg" "$cache_dir/" || fail "Failed to copy CARLA egg into cache/"
  unzip -q "$cache_dir/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg" -d "$cache_dir" || fail "Failed to unzip CARLA egg"

  if [[ -d "$cache_dir/EGG-INFO" ]]; then
    mv "$cache_dir/EGG-INFO" "$cache_dir/carla-${CARLA_VERSION}-py3.7-linux-x86_64"
  fi

  cp "$ROOT_DIR/scripts/setup.py" "$cache_dir/" || fail "Failed to copy scripts/setup.py into cache/"
  python -m pip install -e "$cache_dir" || fail "Failed to install cached CARLA package"
}

install_opencood() {
  log "Installing OpenCOOD dependencies"
  pushd "$ROOT_DIR/opencood" >/dev/null || fail "Unable to enter opencood/"
  python -m pip install -r requirements.txt || fail "Failed to install OpenCOOD requirements"
  python setup.py develop || fail "Failed to install OpenCOOD in develop mode"
  python ./opencood/utils/setup.py build_ext --inplace || fail "Failed to build OpenCOOD native extension"
  popd >/dev/null || fail "Failed to leave opencood/"
}

run_diagnostics() {
  [[ "$RUN_DIAGNOSTICS" == "1" ]] || return

  log "Running environment diagnostics"
  local args=()
  if [[ "$AUTO_FIX_DIAGNOSTICS" == "1" ]]; then
    args+=(--auto-fix)
  fi

  python "$ROOT_DIR/scripts/diagnose_opencda_env.py" "${args[@]}" || warn "Diagnostics reported unresolved issues. See output above."
}

main() {
  need_cmd python
  need_cmd unzip

  ensure_conda_env
  activate_conda

  log "Python runtime: $(python --version 2>&1)"
  install_python_dependencies
  develop_install_opencda
  install_carla_from_egg
  install_opencood
  run_diagnostics

  log "Setup completed."
  if [[ "$INSTALL_ML" == "1" ]]; then
    log "Next step: start CARLA on the target town, then run your OpenCDA scenario."
  fi
}

main "$@"
