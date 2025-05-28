export OMNI_KIT_ACCEPT_EULA := "Y"
export CUBLAS_WORKSPACE_CONFIG := ":4096:8"

train:
    python scripts/clean_rl/train.py --task=CaT-Go2-Rough-Terrain-v0 --headless --num_envs=6000

eval run_dir *flags:
    python scripts/eval.py --task=CaT-Go2-Rough-Terrain-Play-v0 --headless --run_dir={{run_dir}} {{flags}}

generate_plots data_file *flags:
    python scripts/generate_plots.py --data_file={{data_file}} {{flags}}