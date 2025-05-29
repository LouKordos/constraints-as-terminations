export OMNI_KIT_ACCEPT_EULA := "Y"
export CUBLAS_WORKSPACE_CONFIG := ":4096:8"

train:
    python scripts/clean_rl/train.py --task=CaT-Go2-Rough-Terrain-v0 --headless --num_envs=7500 2>&1 | tee ./logs/clean_rl/train-$(date +"%Y-%m-%d-%H:%M:%S").log

eval run_dir *flags:
    python scripts/eval.py --task=CaT-Go2-Rough-Terrain-Play-v0 --headless --run_dir={{run_dir}} {{flags}}

eval-all logs_root_dir num_parallel_jobs:
	@find {{logs_root_dir}} -mindepth 1 -maxdepth 1 -type d | parallel -j {{num_parallel_jobs}} just eval run_dir={}

generate_plots data_file *flags:
    python scripts/generate_plots.py --data_file={{data_file}} {{flags}}