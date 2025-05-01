#!/usr/bin/env bash
set -euxo pipefail


# Usage: ./setup_env_with_uv.sh <ENV_NAME>
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <ENV_NAME>" >&2
    exit 1
fi


ENV_NAME=$1
OMNI_KIT_ACCEPT_EULA=Y
PYTHON_VERSION=3.10
PROJECT_ROOT="$HOME/mamba_env_data/$ENV_NAME"


# Check for uv
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it via:\n  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi


# Install Python version (if missing) and create project directory
uv python install $PYTHON_VERSION  # ([docs.astral.sh](https://docs.astral.sh/uv/guides/install-python/?utm_source=chatgpt.com))


# Prepare project directory
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"


# Initialize uv project (pyproject.toml, .venv, uv.lock)
uv init .  # ([docs.astral.sh](https://docs.astral.sh/uv/guides/projects/?utm_source=chatgpt.com))
# Pin Python version
echo "$PYTHON_VERSION" > .python-version  # ([docs.astral.sh](https://docs.astral.sh/uv/guides/projects/?utm_source=chatgpt.com))
uv venv
# Install Python dependencies via uv pip (10–100× faster than pip)
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121  # ([github.com](https://github.com/astral-sh/uv?utm_source=chatgpt.com), [pypi.org](https://pypi.org/project/uv/?utm_source=chatgpt.com))
uv pip install --upgrade pip  # ([andreagrandi.it](https://www.andreagrandi.it/posts/using-uv-to-install-python-create-virtualenv/?utm_source=chatgpt.com))
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com  # ([docs.astral.sh](https://docs.astral.sh/uv/pip/environments/?utm_source=chatgpt.com))


# Clone and install IsaacLab
mkdir -p "$PROJECT_ROOT/isaaclab-installation"
cd isaaclab-installation
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
# Run installer within uv environment
OMNI_KIT_ACCEPT_EULA=Y uv run --no-project bash isaaclab.sh --install none  # ([docs.astral.sh](https://docs.astral.sh/uv/reference/cli/?utm_source=chatgpt.com))


# Clone user repo
cd "$PROJECT_ROOT"
git clone https://github.com/LouKordos/constraints-as-terminations.git


# Fix permissions
#chmod -R 777 "$HOME/mamba_env_data/"
#chmod -R 777 "$(dirname "$(dirname "$(realpath "$0")")")"


#TODO: Create sbatch file specific to that run in mamba_env_data/$ENV_NAME, replace ENV_NAME definition in it to match $1
set +x
echo "-------------------------------------DONE. CHECKLIST:------------------------------------------"
echo "1. source .venv/bin/activate"
echo "2. change ENV_NAME in sbatch file!"
echo "3. adjust runtime, num_envs, TASK NAME in sbatch file!"
echo "3. make the changes you want inside $HOME/mamba_env_data/$ENV_NAME/constraints-as-terminations/"
echo "-----------------------------------------------------------------------------------------------"
