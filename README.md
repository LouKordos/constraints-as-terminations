# LoComposition

### Terrain-Adaptive Energy-Efficient Quadruped Locomotion without Gait Priors

[Project page](https://sites.google.com/view/locomposition) · [Paper](https://arxiv.org/abs/2606.15896) · [Project video](https://youtu.be/byAA07ge4O0)

![LoComposition separates task specification, operational limits, gait preference, and terrain adaptation.](assets/locomposition-overview.png)

LoComposition learns efficient rough-terrain locomotion without air-time targets, contact-count objectives, foot-clearance rewards, or a prescribed gait. The training formulation gives each concern one clear role: rewards specify the task, constraints encode operational limits, mechanical-energy minimization provides a gait preference, and exteroceptive perception makes that preference terrain-aware.

## Why LoComposition

Quadruped locomotion rewards often mix command tracking, actuator limits, smoothness, foot timing, clearance, and terrain handling into one weighted objective. LoComposition separates them:

- **Task specification:** rewards track commanded planar and yaw velocity.
- **Operational limits:** [Constraints as Terminations (CaT)](https://arxiv.org/abs/2403.18765) encodes actuator, action-rate, and posture limits.
- **Gait preference:** mechanical-power minimization favors economical motion without naming a contact pattern.
- **Terrain adaptation:** a robot-centric elevation map lets the policy spend energy where obstacles require it.

The result is an energy-efficient trotting gait that emerges during training and increases clearance when the terrain requires it. CaT is the chosen constraint mechanism for our method, and LoComposition is the complete formulation built around it.

## Results at a glance

Against a conventional complex-reward locomotion baseline, LoComposition provides:

- **56% lower Cost of Transport** with comparable rough-terrain progression;
- **96% fewer operational-limit violations** under the same evaluation thresholds;
- **zero-shot deployment on a Unitree Go2**, using an online LiDAR elevation map and no hardware retraining; and
- **no explicit gait-style priors** in the final policy.

![Cost of Transport and learned contact patterns.](assets/cot-and-contact-patterns.png)

See the paper for more details. The [project page](https://sites.google.com/view/locomposition) and [project video](https://youtu.be/byAA07ge4O0) are the best place to watch the full comparison.

## Installation

The setup script creates a pinned Python 3.11 environment, installs Isaac Sim 5.1.0 and the matching Isaac Lab revision, clones LoComposition, and installs its dependencies. It requires Linux, an NVIDIA GPU with a compatible driver, Git, and enough disk space for Isaac Sim.

```bash
./create-isaac-lab-env-uv.sh locomposition
source ~/mamba_env_data/locomposition/.venv/bin/activate
cd ~/mamba_env_data/locomposition/LoComposition
```

Use `--root PATH` to place the environment elsewhere, or `--repo-source URL_OR_PATH` to install from a fork or local checkout. The script refuses to merge into a non-empty target directory.

If you already have the pinned Isaac Lab environment, install only this extension and its Python dependencies:

```bash
uv pip install --no-build-isolation --no-deps --editable ./exts/locomposition
uv pip install --requirement requirements.txt
```

## Quick start

Train the main Go2 policy:

```bash
python scripts/clean_rl/train.py \
  --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 \
  --seed=46 --headless --num_envs=7500
```

Evaluate a checkpoint and record the standard diagnostics:

```bash
python scripts/eval.py \
  --run_dir=/absolute/path/to/training/run \
  --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0 \
  --seed=46 --headless
```

Regenerate plots from a saved evaluation:

```bash
python scripts/generate_plots.py \
  --data_file=/absolute/path/to/evaluation/plots/sim_data.npz \
  --output_dir=/absolute/path/to/output/plots
```

The former `CaT-*` task IDs and `cat_envs` Python imports remain available as compatibility aliases. New scripts should use the `LoComposition-*` IDs and `locomposition` package. See the [migration guide](docs/migration.md) for the exact mapping.

## Supported robots

Each embodiment uses a separately trained policy, but the formulation and training recipe stay the same. Action scaling and actuator limits are adjusted based on the robot's size, while domain randomization, disturbances, and the energy penalty coefficient are scaled by the robot's mass ratio.

![Placeholder for the Unitree Go2 hardware demonstration.](assets/demos/go2-hardware.gif)

![Placeholder for the ANYmal C simulation demonstration.](assets/demos/anymal-c.gif)

![Placeholder for the Boston Dynamics Spot simulation demonstration.](assets/demos/spot.gif)

## Repository layout

- `exts/locomposition/`: Isaac Lab environments, robot configurations, CaT constraint manager, and CleanRL PPO implementation.
- `scripts/clean_rl/train.py`: training entry point.
- `scripts/eval.py`: checkpoint evaluation, scenario rollouts, and metric collection.
- `scripts/generate_plots.py`: plot regeneration from saved evaluation arrays.
- `sim2real/`: ROS 2 controller, elevation-map processing, launch stack, and traced policies for the Go2.
- `tests/`: naming, compatibility, setup, ROS, plotting, and metric contracts.

## Sim-to-real deployment

The Go2 stack runs policy inference at 50 Hz and republishes the latest PD target from a 500 Hz low-level loop. A Livox MID-360 and `elevation_mapping_cupy` produce the online elevation map, the LoComposition processing node converts it to the same yaw-aligned 13×11 observation used in simulation. Read [sim-to-real deployment guide](docs/sim2real.md) before setting up the hardware.

![Go2 obstacle sequences and the LiDAR elevation-mapping pipeline.](assets/sim2real-overview.png)

## Citation and attribution

If LoComposition is useful in your work, please cite:

```bibtex
@article{kordos2026locomposition,
  title   = {LoComposition: Terrain-Adaptive Energy-Efficient Quadruped Locomotion without Gait Priors},
  author  = {Kordos, Loukas and Franz, Leonard T. and Rappenecker, Simon and Hausd{\"o}rfer, Oliver and Schoellig, Angela P. and Kolev, Pavel and Martius, Georg},
  journal = {arXiv preprint arXiv:2606.15896},
  year    = {2026}
}
```

LoComposition uses **Constraints as Terminations** as the mechanism for operational-limit constraints. Please also cite the original CaT paper when using that part of the code:

```bibtex
@inproceedings{chane_sane2024cat,
  title     = {{CaT}: Constraints as Terminations for Legged Locomotion Reinforcement Learning},
  author    = {Chane-Sane, Elliot and Leziart, Pierre-Alexandre and Flayols, Thomas and Stasse, Olivier and Sou{\`e}res, Philippe and Mansard, Nicolas},
  booktitle = {2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2024}
}
```

## License

This repository does not declare one project-wide license yet, but existing source files already may have their notices (BSD-3-Clause/Apache-2.0 in the ROS controller package). Check the relevant file or package before reuse.
