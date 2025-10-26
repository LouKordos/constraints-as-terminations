#!/bin/bash
set -ux pipefail

apt update && apt install -y sudo nvtop btop vim tmux git cmake build-essential curl locales ncurses-term wget unzip htop libsm6 libice6 libxt6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 libglu1-mesa-dev libglib2.0-0 libxmu6 tree ffmpeg hyperfine
echo ":set tabstop=4" >> ~/.vimrc
echo ":set shiftwidth=4" >> ~/.vimrc
echo ":set expandtab" >> ~/.vimrc

echo "set -g history-limit 200000" >> ~/.tmux.conf

sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
DEBIAN_FRONTEND=noninteractive dpkg-reconfigure --frontend=noninteractive locales
update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# swisstransfer.com cli
wget -qO- https://github.com/Blutsh/Swish/releases/download/1.0.1/swish-1.0.1-x86_64-unknown-linux-musl.tar.gz | tar xz --strip-components=1 -C /usr/local/bin --wildcards '*/swish'

# 1. Make sure there is a writable location for custom ICDs
sudo mkdir -p /etc/vulkan/icd.d                                                                                                                                                                                                                                                                                                                                                                                                   # 2. Drop a headless‑safe ICD that points to the EGL variant
sudo tee /etc/vulkan/icd.d/nvidia_headless.json >/dev/null <<'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0",
    "api_version": "1.3.204"
  }
}
EOF

curl -L micro.mamba.pm/install.sh | "${SHELL}" # Just pipe so that interactive prompts are disabled
set +u && source ~/.bashrc && set -u

###################### CUSTOMIZE #######################
export ENV_NAME=env_isaaclab_1
micromamba create -y -n $ENV_NAME python=3.10
micromamba activate $ENV_NAME

export OMNI_KIT_ALLOW_ROOT=1
export OMNI_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=Y
export HEADLESS=1
export KIT_DISABLE_NGX=1 # for DLSS etc., but can't run anyway because headless. This just silences a warning
export KIT_DISABLE_WINDOWING=1
export ENABLE_CAMERAS=1
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# experience files can be absolute path, or relative path searched in isaacsim/apps or omni/apps

# 3. Tell the loader to use *only* this file
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_headless.json
tmux
micromamba activate $ENV_NAME
# Run Isaac Sim to accept EULA
# yes | isaacsim isaacsim.exp.full.kit --no-window
# This doesn't work yet you need to manually ctrl + c when "Isaac Sim Full App is loaded" shows up.
yes | stdbuf -oL isaacsim isaacsim.exp.full.kit --no-window | stdbuf -oL sed '/Isaac Sim Full App is loaded./ q'

mkdir /isaaclab && cd /isaaclab
git clone https://github.com/isaac-sim/IsaacLab.git

cd IsaacLab
./isaaclab.sh --install # or "./isaaclab.sh -i"

# Optionally test isaac lab:
#./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless

mkdir -p /$ENV_NAME && cd /$ENV_NAME
git clone https://github.com/Loukordos/constraints-as-terminations.git && cd constraints-as-terminations
# For CaT only (run in the CaT directory):
python -m pip install -e exts/cat_envs
wandb login

python scripts/clean_rl/train.py --task=YOURTASKNAME --headless --num_envs=7500
python scripts/clean_rl/play.py --task=CaT-Go2-Rough-Terrain-v0 --headless --num_envs=1 --video
python scripts/clean_rl/eval.py --task=CaT-Go2-Rough-Terrain-v0 --headless --num_envs=1 --video --run_dir=
