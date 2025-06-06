export OMNI_KIT_ACCEPT_EULA := "Y"
export CUBLAS_WORKSPACE_CONFIG := ":4096:8"

train num_envs="7500":
    mkdir -p ./logs/clean_rl
    python scripts/clean_rl/train.py --task=CaT-Go2-Rough-Terrain-v0 --headless --num_envs={{num_envs}} 2>&1 | tee ./logs/clean_rl/train-$(date +"%Y-%m-%d-%H:%M:%S").log

eval run_dir *flags:
    python scripts/eval.py --task=CaT-Go2-Rough-Terrain-Play-v0 --headless --run_dir={{run_dir}} {{flags}}

eval-all logs_root_dir num_parallel_jobs:
    @# Check for GNU parallel
    @if command -v parallel >/dev/null 2>&1; then \
        find {{logs_root_dir}} -mindepth 1 -maxdepth 1 -type d -print0 | parallel --keep-order --line-buffer -0 -j {{num_parallel_jobs}} just eval {}; \
    else \
        >&2 echo "Warning: GNU parallel not found; running sequentially."; find {{logs_root_dir}} -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -n1 -I{} just eval {}; \
    fi

generate_plots data_file *flags:
    python scripts/generate_plots.py --data_file={{data_file}} {{flags}}