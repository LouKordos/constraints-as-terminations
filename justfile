set shell := ["bash", "-euo", "pipefail", "-c"]

export OMNI_KIT_ACCEPT_EULA := "Y"
export CUBLAS_WORKSPACE_CONFIG := ":4096:8"
export PYTHONUNBUFFERED := "1"

train num_envs="7500" task="CaT-Go2-Rough-Terrain-v0" seed="46":
    tmpdir="${SLURM_TMPDIR:-$(pwd)/logs/tmp}"; \
    mkdir -p ./logs/clean_rl "$tmpdir" "$tmpdir/isaaclab/logs"; \
    echo "TMPDIR=$tmpdir"; \
    TMPDIR="$tmpdir" python scripts/clean_rl/train.py --task={{task}} --seed={{seed}} --headless --num_envs={{num_envs}} 2>&1 | tee "./logs/clean_rl/train-$(date +'%Y-%m-%d-%H:%M:%S').log"

_train-rsl-baseline task num_envs seed max_iterations wandb_project *flags:
    tmpdir="${SLURM_TMPDIR:-$(pwd)/logs/tmp}"; \
    mkdir -p ./logs/rsl_rl "$tmpdir" "$tmpdir/isaaclab/logs"; \
    echo "TMPDIR=$tmpdir"; \
    TMPDIR="$tmpdir" python scripts/train_rsl_rl.py \
        --task={{task}} \
        --seed={{seed}} \
        --headless \
        --num_envs={{num_envs}} \
        --max_iterations={{max_iterations}} \
        --logger=wandb \
        --log_project_name={{wandb_project}} \
        env.scene.terrain.terrain_generator.seed={{seed}} \
        env.sim.random_seed={{seed}} \
        {{flags}}

train-baseline-go2 num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_go2" *flags:
    just _train-rsl-baseline Baseline-Go2-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

train-baseline-anymal-c num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_anymal_c" *flags:
    just _train-rsl-baseline Baseline-Anymal-C-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

train-baseline-spot num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_spot" *flags:
    just _train-rsl-baseline Baseline-Spot-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

eval run_dir *flags:
    systemd-run --scope --user -p MemoryMax=45G time python scripts/eval.py --headless --run_dir={{run_dir}} {{flags}}

eval-baseline-go2 run_dir *flags:
    just eval {{run_dir}} --task=Baseline-Go2-Rough-Terrain-Play-v0 --policy_backend=rsl_rl {{flags}}

eval-baseline-anymal-c run_dir *flags:
    just eval {{run_dir}} --task=Baseline-Anymal-C-Rough-Terrain-Play-v0 --policy_backend=rsl_rl {{flags}}

eval-baseline-spot run_dir *flags:
    just eval {{run_dir}} --task=Baseline-Spot-Rough-Terrain-Play-v0 --policy_backend=rsl_rl {{flags}}

eval-all logs_root_dir num_parallel_jobs *flags:
    @# Check for GNU parallel
    @if command -v parallel >/dev/null 2>&1; then \
        find {{logs_root_dir}} -mindepth 1 -maxdepth 1 -type d -print0 | parallel --keep-order --line-buffer -0 -j {{num_parallel_jobs}} just eval {} {{flags}}; \
    else \
        >&2 echo "Warning: GNU parallel not found; running sequentially."; \
        for dir in "{{logs_root_dir}}"/*/; do just eval "$dir" {{flags}} && sleep 3; done; \
    fi

generate_plots data_file *flags:
    python scripts/generate_plots.py --data_file={{data_file}} {{flags}}
