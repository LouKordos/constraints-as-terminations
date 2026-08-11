#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 ENV_NAME [--root PATH] [--repo-source URL_OR_PATH]

Create a pinned Isaac Lab environment and install LoComposition.

Options:
  --root PATH                 Parent directory for the environment.
  --repo-source URL_OR_PATH   Git URL or local repository to clone.
  -h, --help                  Show this help message.
EOF
}

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage
    exit 0
fi

ENV_NAME=$1
shift
ENV_ROOT="${LOCOMPOSITION_ENV_ROOT:-$HOME/mamba_env_data}"
REPO_SOURCE="${LOCOMPOSITION_REPO_SOURCE:-https://github.com/LouKordos/LoComposition.git}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --root requires a path." >&2
                exit 2
            fi
            ENV_ROOT=$2
            shift 2
            ;;
        --repo-source)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --repo-source requires a URL or path." >&2
                exit 2
            fi
            REPO_SOURCE=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || [ "$ENV_NAME" = "." ] || [ "$ENV_NAME" = ".." ]; then
    echo "[ERROR] Invalid environment name '$ENV_NAME'. Use letters, digits, '.', '_', or '-' without path separators." >&2
    exit 2
fi

export OMNI_KIT_ACCEPT_EULA=Y
PROJECT_ROOT="$ENV_ROOT/$ENV_NAME"
USER_REPO_DIR="$PROJECT_ROOT/LoComposition"

PYTHON_VERSION="3.11"
ISAACLAB_TAG="ddb044eb5b2300792de41e82d53b032f3632b489"

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

# Refuse to merge a new environment into an existing directory.
if [ -e "$PROJECT_ROOT" ]; then
    if [ ! -d "$PROJECT_ROOT" ] || [ -n "$(find "$PROJECT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "[ERROR] Target '$PROJECT_ROOT' already exists and is non-empty." >&2
        exit 1
    fi
fi

# Prepare project directory.
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
set -x

# Initialize project and explicitly set Python requirement (creates pyproject.toml)
uv init --python "$PYTHON_VERSION" .
uv venv

# Install Python dependencies via uv pip, torch and Isaac Sim are pinned here
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --upgrade pip 
uv pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com  
uv tool install rust-just

# Clone and install IsaacLab
mkdir -p "$PROJECT_ROOT/isaaclab-installation"
cd isaaclab-installation
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Checkout the specific version defined at the top of script
echo "[INFO] Checking out Isaac Lab version: $ISAACLAB_TAG"
git checkout "$ISAACLAB_TAG"

echo "[INFO] Attempting to activate: source ${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/.venv/bin/activate" || { echo "venv activation failed"; exit 1; }
export OMNI_KIT_ACCEPT_EULA=Y

echo "[INFO] Installing Isaac Lab Core and Tasks..."
uv pip install -e source/isaaclab 
uv pip install -e source/isaaclab_assets 
uv pip install -e source/isaaclab_tasks 
uv pip install -e source/isaaclab_rl
deactivate

cd "$PROJECT_ROOT" || exit 1

echo "[INFO] Cloning LoComposition from '$REPO_SOURCE'..."
git clone "$REPO_SOURCE" "$USER_REPO_DIR" || { echo "[ERROR] Failed to clone LoComposition."; exit 1; }
cd "$USER_REPO_DIR" || exit 1
uv pip install --no-build-isolation --no-deps -e ./exts/locomposition
uv pip install -r requirements.txt
cd "$PROJECT_ROOT" || exit 1

SLURM_TEMPLATE="$HOME/local-mamba-test.sbatch"
SLURM_CONFIG="$PROJECT_ROOT/slurm-config.sbatch"
SLURM_CONFIG_2080TI="$PROJECT_ROOT/slurm-config-2080ti.sbatch"

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "[WARNING] Slurm template '$SLURM_TEMPLATE' does not exist; skipping Slurm configuration."
else
    echo "[INFO] Copying Slurm configuration template..."
    cp "$SLURM_TEMPLATE" "$SLURM_CONFIG"

    echo "[INFO] Applying ENV_NAME to $SLURM_CONFIG..."
    sed -i -E "s|^[[:space:]]*(export[[:space:]]+)?ENV_NAME=.*|export ENV_NAME=${ENV_NAME}|" "$SLURM_CONFIG"

    echo "[INFO] Updating job name in $SLURM_CONFIG to '$ENV_NAME'..."
    sed -i -E "s|^#SBATCH[[:space:]]+--job-name=.*|#SBATCH --job-name=${ENV_NAME}|" "$SLURM_CONFIG"

    echo "[INFO] Creating 2080Ti Slurm configuration..."
    cp "$SLURM_CONFIG" "$SLURM_CONFIG_2080TI"

    echo "[INFO] Updating job name in $SLURM_CONFIG_2080TI to '${ENV_NAME}-2080ti'..."
    sed -i -E "s|^#SBATCH[[:space:]]+--job-name=.*|#SBATCH --job-name=${ENV_NAME}-2080ti|" "$SLURM_CONFIG_2080TI"

    echo "[INFO] Updating partition in $SLURM_CONFIG_2080TI to 'week'..."
    sed -i -E "s|^#SBATCH[[:space:]]+--partition=.*|#SBATCH --partition=week|" "$SLURM_CONFIG_2080TI"

    echo "[INFO] Updating GPU request in $SLURM_CONFIG_2080TI to 'gpu:2080ti:1'..."
    sed -i -E "s|^#SBATCH[[:space:]]+--gres=.*|#SBATCH --gres=gpu:2080ti:1|" "$SLURM_CONFIG_2080TI"

    echo "[INFO] Updating array in $SLURM_CONFIG_2080TI to 9 single-run jobs..."
    sed -i -E "s|^#SBATCH[[:space:]]+--array=.*|#SBATCH --array=0-8%9|" "$SLURM_CONFIG_2080TI"

    echo "[INFO] Updating RUNS_PER_NODE in $SLURM_CONFIG_2080TI to 1..."
    sed -i -E "s|^RUNS_PER_NODE=.*|RUNS_PER_NODE=1|" "$SLURM_CONFIG_2080TI"

    echo "[INFO] Slurm configurations complete."
fi

set +x
echo "-------------------------------------DONE. CHECKLIST:------------------------------------------"
echo "1. source $PROJECT_ROOT/.venv/bin/activate"
echo "2. cd $USER_REPO_DIR"
echo "3. Run a LoComposition training or evaluation command from README.md"
echo "4. If generated, review $SLURM_CONFIG before submitting a Slurm job"
echo "-----------------------------------------------------------------------------------------------"
