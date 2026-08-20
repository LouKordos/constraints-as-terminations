#!/usr/bin/env bash

#SBATCH --job-name=locomposition-training
#SBATCH --cpus-per-task=8
#SBATCH --partition=L40Sday
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=6-23
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=loukas.kordos@uni-tuebingen.de
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

export ENV_ROOT="__LOCOMPOSITION_ENV_ROOT__"
export ENV_NAME="__LOCOMPOSITION_ENV_NAME__"
export OMNI_KIT_ACCEPT_EULA=Y

source "${ENV_ROOT}/${ENV_NAME}/.venv/bin/activate"
cd "${ENV_ROOT}/${ENV_NAME}/LoComposition"

NETRC_FILE="${NETRC:-${HOME}/.netrc}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    if [ ! -r "$NETRC_FILE" ] || ! grep -Eq '(^|[[:space:]])machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)' "$NETRC_FILE"; then
        echo "[ERROR] W&B credentials not found." >&2
        echo "Run 'wandb login' once from your cluster account, then resubmit this job." >&2
        exit 1
    fi
fi

SEEDS=(
    46
    47
    48
    49
    50
    51
    52
    53
    54
)

NUM_ENVS=7500
TASK_NAME="LoComposition-Go2-Rough-Terrain-Joint-State-History-v0"
RUNS_PER_NODE=3

START_INDEX=$((SLURM_ARRAY_TASK_ID * RUNS_PER_NODE))
END_INDEX=$((START_INDEX + RUNS_PER_NODE - 1))

echo "============================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Running seeds from index ${START_INDEX} to ${END_INDEX}"
echo "============================================================"

run_training() {
    local seed="$1"

    echo "============================================================"
    echo "Starting training with seed=${seed}"
    echo "Node: $(hostname)"
    echo "Time: $(date)"
    echo "============================================================"

    just train "${NUM_ENVS}" "${TASK_NAME}" "${seed}"

    echo "============================================================"
    echo "Finished training with seed=${seed}"
    echo "Node: $(hostname)"
    echo "Time: $(date)"
    echo "============================================================"
}

exit_code=0
active_pids=()

for seed_index in $(seq "${START_INDEX}" "${END_INDEX}"); do
    seed="${SEEDS[$seed_index]}"
    run_training "${seed}" &
    active_pids+=("$!")
done

for pid in "${active_pids[@]}"; do
    if ! wait "${pid}"; then
        exit_code=1
    fi
done

if ((exit_code == 0)); then
    echo "DONE on node $(hostname) for array task ${SLURM_ARRAY_TASK_ID}!"
else
    echo "DONE on node $(hostname), but at least one training run failed."
fi

exit "${exit_code}"
