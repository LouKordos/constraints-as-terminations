#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 ENV_NAME [--root PATH] [--repo-source URL_OR_PATH]

Create a locked LoComposition and pinned Isaac Lab environment.

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

PROJECT_ROOT="$ENV_ROOT/$ENV_NAME"
USER_REPO_DIR="$PROJECT_ROOT/LoComposition"
ISAACLAB_DIR="$PROJECT_ROOT/isaaclab-installation/IsaacLab"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
ISAACLAB_REVISION="ddb044eb5b2300792de41e82d53b032f3632b489"

if [ -e "$PROJECT_ROOT" ]; then
    if [ ! -d "$PROJECT_ROOT" ] || [ -n "$(find "$PROJECT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "[ERROR] Target '$PROJECT_ROOT' already exists and is non-empty." >&2
        exit 1
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] uv not found; installing it for the current user..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] uv installation failed." >&2
        exit 1
    fi
else
    echo "[INFO] Using uv at $(command -v uv)."
fi

mkdir -p "$PROJECT_ROOT"

echo "[INFO] Cloning LoComposition from '$REPO_SOURCE'..."
git clone "$REPO_SOURCE" "$USER_REPO_DIR"

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export OMNI_KIT_ACCEPT_EULA=Y

uv sync --project "$USER_REPO_DIR" --frozen --no-install-package locomposition

if ! command -v just >/dev/null 2>&1; then
    echo "[INFO] Installing the just command..."
    uv tool install rust-just==1.40.0
fi

echo "[INFO] Cloning the pinned Isaac Lab source checkout..."
mkdir -p "$(dirname "$ISAACLAB_DIR")"
git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
cd "$ISAACLAB_DIR"
git checkout "$ISAACLAB_REVISION"

echo "[INFO] Installing the four supported Isaac Lab distributions editably..."
uv pip install --python "$VENV_PYTHON" -e source/isaaclab
uv pip install --python "$VENV_PYTHON" -e source/isaaclab_assets
uv pip install --python "$VENV_PYTHON" -e source/isaaclab_tasks
uv pip install --python "$VENV_PYTHON" -e source/isaaclab_rl

uv sync --project "$USER_REPO_DIR" --frozen --inexact

echo "[INFO] Checking dependency consistency and editable package locations..."
if dependency_check=$(uv pip check --python "$VENV_PYTHON" 2>&1); then
    printf '%s\n' "$dependency_check"
else
    printf '%s\n' "$dependency_check" >&2
    incompatibility_count=$(printf '%s\n' "$dependency_check" | grep -c '^The package `')
    # Isaac Sim 5.1 pins FastAPI 0.115.7, whose metadata requires Starlette <0.46;
    # the pinned Isaac Lab revision requires Starlette 0.49.1. Permit only this
    # conflict until the simulator stack is upgraded in a controlled change.
    if [[ "$dependency_check" == *"Found 1 incompatibility"* ]] \
        && [[ "$dependency_check" == *'The package `fastapi` requires `starlette'* ]] \
        && [[ "$dependency_check" == *'>=0.40.0'* ]] \
        && [[ "$dependency_check" == *'<0.46.0'* ]] \
        && [[ "$dependency_check" == *'but `0.49.1` is installed'* ]] \
        && [ "$incompatibility_count" -eq 1 ]; then
        echo "[WARNING] Allowing the known Isaac Sim/Isaac Lab metadata conflict for Starlette 0.49.1." >&2
    else
        exit 1
    fi
fi
uv pip show --python "$VENV_PYTHON" locomposition isaaclab isaaclab-assets isaaclab-tasks isaaclab-rl

SLURM_TEMPLATE="$USER_REPO_DIR/train-locomposition.sbatch"
SLURM_CONFIG="$PROJECT_ROOT/train-locomposition.sbatch"
SLURM_CONFIG_2080TI="$PROJECT_ROOT/train-locomposition-2080ti.sbatch"

escape_sed_replacement() {
    printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

escaped_env_root=$(escape_sed_replacement "$ENV_ROOT")
escaped_env_name=$(escape_sed_replacement "$ENV_NAME")

echo "[INFO] Generating the L40S Slurm job..."
sed \
    -e "s|__LOCOMPOSITION_ENV_ROOT__|${escaped_env_root}|g" \
    -e "s|__LOCOMPOSITION_ENV_NAME__|${escaped_env_name}|g" \
    "$SLURM_TEMPLATE" > "$SLURM_CONFIG"

echo "[INFO] Generating the 2080 Ti Slurm job from the same template..."
sed \
    -e "s|__LOCOMPOSITION_ENV_ROOT__|${escaped_env_root}|g" \
    -e "s|__LOCOMPOSITION_ENV_NAME__|${escaped_env_name}|g" \
    -e 's|^#SBATCH --job-name=.*|#SBATCH --job-name=locomposition-training-2080ti|' \
    -e 's|^#SBATCH --partition=.*|#SBATCH --partition=week|' \
    -e 's|^#SBATCH --gres=.*|#SBATCH --gres=gpu:2080ti:1|' \
    -e 's|^#SBATCH --array=.*|#SBATCH --array=0-8%9|' \
    -e 's|^RUNS_PER_NODE=.*|RUNS_PER_NODE=1|' \
    "$SLURM_TEMPLATE" > "$SLURM_CONFIG_2080TI"

chmod +x "$SLURM_CONFIG" "$SLURM_CONFIG_2080TI"
bash -n "$SLURM_CONFIG"
bash -n "$SLURM_CONFIG_2080TI"

echo "------------------------------------- DONE -------------------------------------"
echo "Activate: source $VENV_DIR/bin/activate"
echo "Repository: cd $USER_REPO_DIR"
echo "W&B: run 'wandb login' once per cluster account (not once per environment)."
echo "L40S: sbatch $SLURM_CONFIG"
echo "2080 Ti: sbatch $SLURM_CONFIG_2080TI"
echo "The generated jobs also accept an inherited WANDB_API_KEY override."
echo "--------------------------------------------------------------------------------"
