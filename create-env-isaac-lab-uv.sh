#!/usr/bin/env bash
set -euxo pipefail

# Usage: ./setup_env_with_uv.sh <ENV_NAME>
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <ENV_NAME>" >&2
    exit 1
fi

ENV_NAME=$1
export OMNI_KIT_ACCEPT_EULA=Y
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
uv pip install toml wandb

# Clone and install IsaacLab
mkdir -p "$PROJECT_ROOT/isaaclab-installation"
cd isaaclab-installation
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

KNOWN_GOOD_COMMIT=9be0de5f6196374fa793dc33978434542e7a40d3
echo "Checking out $KNOWN_GOOD_COMMIT"
git checkout $KNOWN_GOOD_COMMIT

####### Patch isaaclab.sh to use uv instead of pip to speed up install process
TARGET="./isaaclab.sh"
# Create a timestamped backup
BACKUP="${TARGET}.bak"

if [ ! -f "$TARGET" ]; then
    echo "[ERROR] Target script '$TARGET' not found in $ISAACLAB_DIR." >&2
    exit 1
fi

# Check if already patched (adjust check if needed)
if grep -q 'uv pip list' "$TARGET" && grep -q '# --- BEGIN Patched Loop ---' "$TARGET" && grep -q 'uv pip install -e \${ISAACLAB_PATH}/source/isaaclab_rl\[' "$TARGET" ; then
     echo "[INFO] Script '$TARGET' appears to be already patched for uv. Skipping."
else
    echo "[INFO] Creating backup: ${BACKUP}"
    cp -a "$TARGET" "$BACKUP"

    # Apply patches sequentially.
    echo "[PATCH STEP 1/5] Modifying extract_python_exe for VIRTUAL_ENV..."
    sed -i.bak1 's|^\([[:space:]]*\)if ! \[\[ -z "\${CONDA_PREFIX}" \]\]; then|\1# Check for active uv virtual environment first\n\1if [[ -n "${VIRTUAL_ENV:-}" ]]; then\n\1  local python_exe="${VIRTUAL_ENV}/bin/python"\n\1# Check if using conda if no virtual env is active\n\1elif ! [[ -z "${CONDA_PREFIX}" ]]; then|' "$TARGET" || { echo "ERROR patching extract_python_exe"; exit 1; }

    echo "[PATCH STEP 2/5] Replacing 'pip list' with 'uv pip list'..."
    sed -i.bak2 's/python -m pip list/uv pip list/g' "$TARGET" || { echo "ERROR patching pip list"; exit 1; }

    echo "[PATCH STEP 3/5] Replacing *other* 'pip install' variants with 'uv pip install'..."
    # Apply the general ${python_exe} -m pip install replacement, hoping it works elsewhere
    sed -i.bak3 's/\${python_exe}[[:space:]]\+-m[[:space:]]\+pip[[:space:]]\+install/uv pip install/g' "$TARGET" || { echo "ERROR patching pip install (variant 1)"; exit 1; }
    # Apply the standalone pip install replacement (e.g., for pre-commit)
    sed -i.bak4 's/\bpip[[:space:]]\+install\b/uv pip install/g' "$TARGET" || { echo "ERROR patching pip install (variant 2)"; exit 1; }

    echo "[PATCH STEP 4/5] Replacing extension installation loop..."
    # Define the loop text using direct assignment
    loop_text='

    # --- BEGIN Patched Loop ---
    for ext_dir in $(find -L "${ISAACLAB_PATH}/source" -mindepth 1 -maxdepth 1 -type d); do
        if [ -f "${ext_dir}/setup.py" ]; then
            echo -e "\\\\t module: ${ext_dir}";
            # Directly use uv pip install in the main shell context
            uv pip install --editable "${ext_dir}";
        fi;
    done
    # --- END Patched Loop ---
'
    TMP_LOOP_FILE=$(mktemp)
    if [[ -z "$TMP_LOOP_FILE" || ! -f "$TMP_LOOP_FILE" ]]; then echo "[ERROR] Failed mktemp"; exit 1; fi
    echo "$loop_text" > "$TMP_LOOP_FILE"
    sed -i.bak5 -e '/export -f extract_python_exe/d' \
                -e '/export -f install_isaaclab_extension/d' \
                -e '/find -L.*-exec bash -c/d' "$TARGET" || { echo "ERROR deleting old loop parts"; rm -f "$TMP_LOOP_FILE"; exit 1; }
    sed -i.bak6 "/echo \"\[INFO] Installing extensions inside the Isaac Lab repository...\"/r $TMP_LOOP_FILE" "$TARGET" || { echo "ERROR inserting new loop"; rm -f "$TMP_LOOP_FILE"; exit 1; }
    rm -f "$TMP_LOOP_FILE"

    echo "[PATCH STEP 5/5] Replacing specific framework install lines..."
    # Use sed 'c\' command to replace the *entire* line matching the pattern
    # Note: Escaping [, ], $ and " within the replacement text for sed
    # Preserve leading whitespace if possible (though 'c\' replaces the whole line) - adjust indentation manually if needed.
    sed -i.bak7 '/isaaclab_rl\[/c\            uv pip install -e ${ISAACLAB_PATH}/source/isaaclab_rl\["${framework_name}"\]' "$TARGET" || { echo "ERROR replacing isaaclab_rl line"; exit 1; }
    sed -i.bak8 '/isaaclab_mimic\[/c\            uv pip install -e ${ISAACLAB_PATH}/source/isaaclab_mimic\["${framework_name}"\]' "$TARGET" || { echo "ERROR replacing isaaclab_mimic line"; exit 1; }

    echo "[INFO] Patching complete. Verifying script integrity..."
    bash -n "$TARGET" || {
        echo "[ERROR] Patched script $TARGET has syntax errors!"
        echo "[ERROR] Restoring from backup $BACKUP..."
        cp -a "$BACKUP" "$TARGET"
        exit 1
    }
    rm -f "$TARGET".bak[1-8] # Clean up intermediate backups
    echo "[INFO] Patched script syntax check passed. Final backup remains: ${BACKUP}"
fi # End of patching block

echo "[INFO] Attempting to activate: source ${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/.venv/bin/activate" || { echo "Activation failed"; exit 1; }
export OMNI_KIT_ACCEPT_EULA=Y
bash ./isaaclab.sh --install rsl_rl
deactivate

# --- User Repo Setup ---
cd "$PROJECT_ROOT" || exit 1
USER_REPO_DIR="$PROJECT_ROOT/constraints-as-terminations"
USER_REPO_URL="https://github.com/LouKordos/constraints-as-terminations.git"

echo "[INFO] Cloning user repo..."
git clone "$USER_REPO_URL" || { echo "[ERROR] Failed to clone user repo."; exit 1; }
cd "$USER_REPO_DIR" || exit 1
uv pip install --no-build-isolation --no-deps -e ./exts/cat_envs

cd "$PROJECT_ROOT" || exit 1
SLURM_TEMPLATE="$HOME/local-mamba-test.sbatch"
SLURM_CONFIG="$PROJECT_ROOT/slurm-config.sbatch"
echo "[INFO] Copying Slurm configuration template..."
cp "$SLURM_TEMPLATE" "$SLURM_CONFIG"
sed -i -E "s|^[[:space:]]*(export[[:space:]]+)?ENV_NAME=.*|export ENV_NAME=${ENV_NAME}|" "$SLURM_CONFIG"
echo "[INFO] Updated ENV_NAME in $SLURM_CONFIG"
sed -i "s/--job-name=.*/--job-name=$ENV_NAME/" "$SLURM_CONFIG"
echo "[INFO] Updated job name in $SLURM_CONFIG"

set +x
echo "-------------------------------------DONE. CHECKLIST:------------------------------------------"
echo "1. source $PROJECT_ROOT/.venv/bin/activate"
echo "2. adjust runtime, num_envs, TASK NAME in $SLURM_CONFIG"
echo "3. make the changes you want inside $USER_REPO_DIR"
echo "4. run: sbatch $SLURM_CONFIG"
echo "-----------------------------------------------------------------------------------------------"
