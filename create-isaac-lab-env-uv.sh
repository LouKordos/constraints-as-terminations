#!/usr/bin/env bash
set -euxo pipefail

# Usage: ./setup_env_with_uv.sh <ENV_NAME>
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <ENV_NAME>" >&2
    exit 1
fi

ENV_NAME=$1
export OMNI_KIT_ACCEPT_EULA=Y
PROJECT_ROOT="$HOME/mamba_env_data/$ENV_NAME"

PYTHON_VERSION="3.11"
ISAACLAB_TAG="5c2ec81cb17532d32f7922dd7fcaae40d123b71a"

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] uv installation failed." >&2
        exit 1
    fi
else
    echo "[INFO] uv is already installed."
fi

# Prepare project directory
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Initialize project and explicitly set Python requirement (creates pyproject.toml)
uv init --python $PYTHON_VERSION .
uv venv

# Install Python dependencies via uv pip, torch and Isaac Sim are pinned here
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --upgrade pip 
uv pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com  
uv tool install rust-just
uv tool update-shell

# Clone and install IsaacLab
mkdir -p "$PROJECT_ROOT/isaaclab-installation"
cd isaaclab-installation
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Checkout the specific version defined at the top of script
echo "[INFO] Checking out Isaac Lab version: $ISAACLAB_TAG"
git checkout $ISAACLAB_TAG

echo "[INFO] Attempting to activate: source ${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/.venv/bin/activate" || { echo "Activation failed"; exit 1; }
export OMNI_KIT_ACCEPT_EULA=Y

echo "[INFO] Installing Isaac Lab Core and Tasks..."
uv pip install -e source/isaaclab 
uv pip install -e source/isaaclab_assets 
uv pip install -e source/isaaclab_tasks 
uv pip install -e source/isaaclab_rl
deactivate

cd "$PROJECT_ROOT" || exit 1
USER_REPO_DIR="$PROJECT_ROOT/constraints-as-terminations"
USER_REPO_URL="https://github.com/LouKordos/constraints-as-terminations.git"

echo "[INFO] Cloning user repo..."
git clone "$USER_REPO_URL" || { echo "[ERROR] Failed to clone user repo."; exit 1; }
cd "$USER_REPO_DIR" || exit 1
uv pip install --no-build-isolation --no-deps -e ./exts/cat_envs
uv pip install -r requirements.txt

cd "$PROJECT_ROOT" || exit 1
SLURM_TEMPLATE="$HOME/local-mamba-test.sbatch"
SLURM_CONFIG="$PROJECT_ROOT/slurm-config.sbatch"
if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "[WARNING] Slurm template '$SLURM_TEMPLATE' does not exist; skipping Slurm configuration."
else
    echo "[INFO] Copying Slurm configuration template..."
    cp "$SLURM_TEMPLATE" "$SLURM_CONFIG"

    echo "[INFO] Applying ENV_NAME to $SLURM_CONFIG..."
    sed -i -E "s|^[[:space:]]*(export[[:space:]]+)?ENV_NAME=.*|export ENV_NAME=${ENV_NAME}|" "$SLURM_CONFIG"

    echo "[INFO] Updating job name to '$ENV_NAME'..."
    sed -i "s/--job-name=.*/--job-name=$ENV_NAME/" "$SLURM_CONFIG"

    echo "[INFO] Slurm configuration complete."
fi

set +x
echo "-------------------------------------DONE. CHECKLIST:------------------------------------------"
echo "1. source $PROJECT_ROOT/.venv/bin/activate"
echo "2. vim $SLURM_CONFIG" # Make desired changes (if any)
echo "3. cd $USER_REPO_DIR"
echo "4. sbatch $SLURM_CONFIG"
echo "-----------------------------------------------------------------------------------------------"
