# Go2 and ANYmal C Cross-Tuning Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Go2-with-ANYmal-tuning and ANYmal-with-Go2-tuning baseline tasks, validate them, and provide manual local training commands without launching production runs.

**Architecture:** Build four narrow environment subclasses on top of the existing matched train/play baselines and mutate only numerical reward weights and joint-position action scale after the receiving baseline is fully configured. Add two local RSL-RL runner subclasses that inherit the donor robot's complete PPO configuration but use crossed experiment names, register train/play tasks, and expose manual `just` recipes with distinct W&B projects.

**Tech Stack:** Python 3.11, Isaac Lab 2.3.1 configuration classes, Gymnasium registration, RSL-RL PPO, pytest, just, PyTorch, Isaac Sim headless runtime, Weights & Biases.

## Global Constraints

- Work only with Go2 and ANYmal C; Spot is out of scope.
- Go2 receives ANYmal C numerical reward weights, action scale `0.5`, and the standard non-symmetry ANYmal C PPO configuration.
- ANYmal C receives Go2 numerical reward weights, action scale `0.25`, and the standard Go2 PPO configuration.
- Go2 `undesired_contacts` remains disabled.
- ANYmal C `undesired_contacts` remains enabled at weight `-1.0` with its existing body selector.
- Reward functions, reward parameters, and robot-specific body selectors remain those of the receiving embodiment.
- The `action_rate_l2` weight remains `-0.01` in both tasks because the donor values are identical.
- Keep the receiving embodiment's asset, actuators, simulation timing, terrain, commands, observations, randomization, terminations, and curriculum unchanged.
- Use task IDs `Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0`, `Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-Play-v0`, `Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0`, and `Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-Play-v0`.
- Use W&B projects `baseline_go2_anymal_c_rewards_action_scale_ppo` and `baseline_anymal_c_go2_rewards_action_scale_ppo`.
- Production defaults are 7,500 environments, seed 46, and 30,000 iterations.
- Do not launch production training, submit Slurm jobs, or modify remote cluster state. Hand the user exact local commands after validation.
- Preserve all pre-existing user changes and do not edit the installed Isaac Lab checkout.

## File Structure

- Create `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_cross_tuning_env_cfg.py`: crossed numerical reward/action mutations and four receiving-embodiment train/play subclasses.
- Create `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/agents/rsl_rl_ppo_cfg.py`: two donor-PPO runner subclasses with crossed experiment names.
- Modify `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`: four Gym registrations and local runner entry-point constants.
- Modify `justfile`: two manual production-training recipes that reuse `_train-rsl-baseline`.
- Create `tests/test_cross_tuning_configs.py`: headless Isaac Lab configuration, receiver-boundary, PPO-transfer, and registration contracts.
- Create `tests/test_cross_tuning_justfile.py`: fast, simulator-free dry-run assertions for both manual recipes.

---

### Task 1: Crossed Environment Configurations

**Files:**
- Create: `tests/test_cross_tuning_configs.py`
- Create: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_cross_tuning_env_cfg.py`

**Interfaces:**
- Consumes: `BaselineGo2RoughEnvCfg`, `BaselineGo2RoughEnvCfg_PLAY`, `BaselineAnymalCRoughEnvCfg`, and `BaselineAnymalCRoughEnvCfg_PLAY` from the existing matched baseline modules.
- Produces: `BaselineGo2AnymalCTuningRoughEnvCfg`, `BaselineGo2AnymalCTuningRoughEnvCfg_PLAY`, `BaselineAnymalCGo2TuningRoughEnvCfg`, and `BaselineAnymalCGo2TuningRoughEnvCfg_PLAY` configuration classes.
- Produces: module constants `ANYMAL_C_REWARD_WEIGHTS` and `GO2_REWARD_WEIGHTS`, both `dict[str, float]` and intentionally excluding `undesired_contacts`.

- [ ] **Step 1: Write the failing crossed-environment contract tests**

Create `tests/test_cross_tuning_configs.py` with the following initial content:

```python
from __future__ import annotations

from copy import deepcopy

import pytest
from isaaclab.app import AppLauncher


_APP_LAUNCHER = AppLauncher(headless=True)

from cat_envs.tasks.locomotion.velocity.config.solo12.baseline_anymal_c_rough_env_cfg import (
    BaselineAnymalCRoughEnvCfg,
    BaselineAnymalCRoughEnvCfg_PLAY,
)
from cat_envs.tasks.locomotion.velocity.config.solo12.baseline_cross_tuning_env_cfg import (
    ANYMAL_C_REWARD_WEIGHTS,
    GO2_REWARD_WEIGHTS,
    BaselineAnymalCGo2TuningRoughEnvCfg,
    BaselineAnymalCGo2TuningRoughEnvCfg_PLAY,
    BaselineGo2AnymalCTuningRoughEnvCfg,
    BaselineGo2AnymalCTuningRoughEnvCfg_PLAY,
)
from cat_envs.tasks.locomotion.velocity.config.solo12.baseline_go2_rough_env_cfg import (
    BaselineGo2RoughEnvCfg,
    BaselineGo2RoughEnvCfg_PLAY,
)


EXPECTED_ANYMAL_C_WEIGHTS = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 0.5,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -1.0e-5,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.125,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}

EXPECTED_GO2_WEIGHTS = {
    "track_lin_vel_xy_exp": 1.5,
    "track_ang_vel_z_exp": 0.75,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -2.0e-4,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.01,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}


def teardown_module():
    _APP_LAUNCHER.app.close()


def _reward_weights(cfg) -> dict[str, float]:
    return {
        name: getattr(cfg.rewards, name).weight
        for name in EXPECTED_ANYMAL_C_WEIGHTS
    }


def _without_transferred_fields(crossed_cfg, receiver_cfg) -> dict:
    crossed = deepcopy(crossed_cfg.to_dict())
    receiver = receiver_cfg.to_dict()
    crossed["rewards"] = receiver["rewards"]
    crossed["actions"]["joint_pos"]["scale"] = receiver["actions"]["joint_pos"]["scale"]
    return crossed


@pytest.mark.parametrize(
    "crossed_cls,receiver_cls",
    [
        (BaselineGo2AnymalCTuningRoughEnvCfg, BaselineGo2RoughEnvCfg),
        (BaselineGo2AnymalCTuningRoughEnvCfg_PLAY, BaselineGo2RoughEnvCfg_PLAY),
    ],
)
def test_go2_receives_anymal_weights_and_action_scale(crossed_cls, receiver_cls):
    cfg = crossed_cls()
    receiver = receiver_cls()

    assert ANYMAL_C_REWARD_WEIGHTS == EXPECTED_ANYMAL_C_WEIGHTS
    assert _reward_weights(cfg) == EXPECTED_ANYMAL_C_WEIGHTS
    assert cfg.actions.joint_pos.scale == 0.5
    assert cfg.rewards.undesired_contacts is None
    assert cfg.rewards.feet_air_time.params["sensor_cfg"].body_names == ".*_foot"
    assert _without_transferred_fields(cfg, receiver) == receiver.to_dict()


@pytest.mark.parametrize(
    "crossed_cls,receiver_cls",
    [
        (BaselineAnymalCGo2TuningRoughEnvCfg, BaselineAnymalCRoughEnvCfg),
        (BaselineAnymalCGo2TuningRoughEnvCfg_PLAY, BaselineAnymalCRoughEnvCfg_PLAY),
    ],
)
def test_anymal_receives_go2_weights_and_action_scale(crossed_cls, receiver_cls):
    cfg = crossed_cls()
    receiver = receiver_cls()

    assert GO2_REWARD_WEIGHTS == EXPECTED_GO2_WEIGHTS
    assert _reward_weights(cfg) == EXPECTED_GO2_WEIGHTS
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.rewards.undesired_contacts.weight == -1.0
    assert cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names == ".*THIGH"
    assert cfg.rewards.feet_air_time.params["sensor_cfg"].body_names == ".*FOOT"
    assert _without_transferred_fields(cfg, receiver) == receiver.to_dict()
```

The `to_dict()` comparison makes the test fail if the crossed classes alter any
receiving-embodiment setting outside `rewards` and the joint-position action
scale.

- [ ] **Step 2: Run the environment tests and verify the missing module failure**

Run:

```bash
python -m pytest -q tests/test_cross_tuning_configs.py
```

Expected: collection fails with
`ModuleNotFoundError: ...baseline_cross_tuning_env_cfg`.

- [ ] **Step 3: Implement the crossed environment module**

Create
`exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_cross_tuning_env_cfg.py`:

```python
"""Crossed Go2 and ANYmal C tuning configurations.

Each class keeps the receiving embodiment's matched environment and transfers
only donor numerical reward weights and joint-position action scale. Contact
enablement and selectors remain receiver-specific.
"""

from isaaclab.utils import configclass

from .baseline_anymal_c_rough_env_cfg import (
    BaselineAnymalCRoughEnvCfg,
    BaselineAnymalCRoughEnvCfg_PLAY,
)
from .baseline_go2_rough_env_cfg import (
    BaselineGo2RoughEnvCfg,
    BaselineGo2RoughEnvCfg_PLAY,
)


ANYMAL_C_REWARD_WEIGHTS: dict[str, float] = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 0.5,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -1.0e-5,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.125,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}

GO2_REWARD_WEIGHTS: dict[str, float] = {
    "track_lin_vel_xy_exp": 1.5,
    "track_ang_vel_z_exp": 0.75,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -2.0e-4,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.01,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}


def _set_reward_weights(rewards, weights: dict[str, float]) -> None:
    for term_name, weight in weights.items():
        term = getattr(rewards, term_name)
        if term is None:
            raise ValueError(f"Cannot transfer weight for disabled reward term: {term_name}")
        term.weight = weight


def _apply_anymal_c_tuning_to_go2(cfg) -> None:
    if cfg.rewards.undesired_contacts is not None:
        raise ValueError("Crossed Go2 must retain its disabled undesired_contacts term")
    _set_reward_weights(cfg.rewards, ANYMAL_C_REWARD_WEIGHTS)
    cfg.actions.joint_pos.scale = 0.5


def _apply_go2_tuning_to_anymal_c(cfg) -> None:
    contact_term = cfg.rewards.undesired_contacts
    if contact_term is None or contact_term.weight != -1.0:
        raise ValueError("Crossed ANYmal C must retain undesired_contacts at weight -1.0")
    _set_reward_weights(cfg.rewards, GO2_REWARD_WEIGHTS)
    cfg.actions.joint_pos.scale = 0.25


@configclass
class BaselineGo2AnymalCTuningRoughEnvCfg(BaselineGo2RoughEnvCfg):
    """Matched Go2 baseline with ANYmal C reward weights and action scale."""

    def __post_init__(self):
        super().__post_init__()
        _apply_anymal_c_tuning_to_go2(self)


@configclass
class BaselineGo2AnymalCTuningRoughEnvCfg_PLAY(BaselineGo2RoughEnvCfg_PLAY):
    """Play configuration for Go2 with ANYmal C tuning."""

    def __post_init__(self):
        super().__post_init__()
        _apply_anymal_c_tuning_to_go2(self)


@configclass
class BaselineAnymalCGo2TuningRoughEnvCfg(BaselineAnymalCRoughEnvCfg):
    """Matched ANYmal C baseline with Go2 reward weights and action scale."""

    def __post_init__(self):
        super().__post_init__()
        _apply_go2_tuning_to_anymal_c(self)


@configclass
class BaselineAnymalCGo2TuningRoughEnvCfg_PLAY(BaselineAnymalCRoughEnvCfg_PLAY):
    """Play configuration for ANYmal C with Go2 tuning."""

    def __post_init__(self):
        super().__post_init__()
        _apply_go2_tuning_to_anymal_c(self)
```

- [ ] **Step 4: Run the environment tests and verify they pass**

Run:

```bash
python -m pytest -q tests/test_cross_tuning_configs.py
```

Expected: `4 passed` and clean Isaac Sim shutdown.

- [ ] **Step 5: Check formatting and commit the crossed environments**

Run:

```bash
python -m compileall -q \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_cross_tuning_env_cfg.py \
  tests/test_cross_tuning_configs.py
git diff --check
git status --short
git add \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_cross_tuning_env_cfg.py \
  tests/test_cross_tuning_configs.py
git commit -m "Add Go2 ANYmal crossed tuning environments"
```

Expected: compilation and diff checks succeed; the commit contains only the
new environment module and its contract test.

---

### Task 2: Donor PPO Runners and Gym Registration

**Files:**
- Create: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/agents/rsl_rl_ppo_cfg.py`
- Modify: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`
- Modify: `tests/test_cross_tuning_configs.py`

**Interfaces:**
- Consumes: the four crossed environment classes from Task 1.
- Consumes: upstream `AnymalCRoughPPORunnerCfg` and `UnitreeGo2RoughPPORunnerCfg`.
- Produces: `Go2AnymalCTuningPPORunnerCfg` with experiment name `go2_anymal_c_tuning_rough`.
- Produces: `AnymalCGo2TuningPPORunnerCfg` with experiment name `anymal_c_go2_tuning_rough`.
- Produces: four registered task IDs with `env_cfg_entry_point` and `rsl_rl_cfg_entry_point` strings.

- [ ] **Step 1: Extend the test with failing runner and registration contracts**

Add these imports after AppLauncher startup in
`tests/test_cross_tuning_configs.py`:

```python
import gymnasium as gym
import cat_envs.tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
```

Append:

```python
CROSS_TASKS = {
    "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0": (
        BaselineGo2AnymalCTuningRoughEnvCfg,
        AnymalCRoughPPORunnerCfg,
        "go2_anymal_c_tuning_rough",
        0.005,
    ),
    "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-Play-v0": (
        BaselineGo2AnymalCTuningRoughEnvCfg_PLAY,
        AnymalCRoughPPORunnerCfg,
        "go2_anymal_c_tuning_rough",
        0.005,
    ),
    "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0": (
        BaselineAnymalCGo2TuningRoughEnvCfg,
        UnitreeGo2RoughPPORunnerCfg,
        "anymal_c_go2_tuning_rough",
        0.01,
    ),
    "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-Play-v0": (
        BaselineAnymalCGo2TuningRoughEnvCfg_PLAY,
        UnitreeGo2RoughPPORunnerCfg,
        "anymal_c_go2_tuning_rough",
        0.01,
    ),
}


@pytest.mark.parametrize("task_id,contract", CROSS_TASKS.items())
def test_cross_task_registration_and_donor_ppo(task_id, contract):
    env_cls, donor_cls, experiment_name, entropy_coef = contract
    spec = gym.spec(task_id)
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    donor_cfg = donor_cls()

    assert isinstance(env_cfg, env_cls)
    assert spec.kwargs["env_cfg_entry_point"].endswith(f":{env_cls.__name__}")
    assert runner_cfg.algorithm.entropy_coef == entropy_coef
    assert runner_cfg.experiment_name == experiment_name

    actual_runner = runner_cfg.to_dict()
    donor_runner = donor_cfg.to_dict()
    donor_runner["experiment_name"] = experiment_name
    assert actual_runner == donor_runner
```

- [ ] **Step 2: Run the registration test and verify it fails**

Run:

```bash
python -m pytest -q tests/test_cross_tuning_configs.py
```

Expected: four existing environment tests pass and the new parametrized test
fails because the first crossed task ID is absent from the Gym registry.

- [ ] **Step 3: Implement the local donor-PPO runner classes**

Create
`exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/agents/rsl_rl_ppo_cfg.py`:

```python
"""RSL-RL runner configurations for crossed Go2 and ANYmal C tuning."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)


@configclass
class Go2AnymalCTuningPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """ANYmal C PPO parameters for the receiving Go2 embodiment."""

    experiment_name = "go2_anymal_c_tuning_rough"


@configclass
class AnymalCGo2TuningPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Go2 PPO parameters for the receiving ANYmal C embodiment."""

    experiment_name = "anymal_c_go2_tuning_rough"
```

- [ ] **Step 4: Register the four crossed task IDs**

In
`exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`,
add constants next to the existing runner constants:

```python
_GO2_ANYMAL_C_TUNING_RSL_RL_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2AnymalCTuningPPORunnerCfg"
)
_ANYMAL_C_GO2_TUNING_RSL_RL_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCGo2TuningPPORunnerCfg"
)
```

Add these registrations immediately after the matched ANYmal C registrations
and before Spot:

```python
gym.register(
    id="Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineGo2AnymalCTuningRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": _GO2_ANYMAL_C_TUNING_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-Play-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineGo2AnymalCTuningRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": _GO2_ANYMAL_C_TUNING_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineAnymalCGo2TuningRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": _ANYMAL_C_GO2_TUNING_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-Play-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineAnymalCGo2TuningRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": _ANYMAL_C_GO2_TUNING_RSL_RL_CFG,
    },
)
```

- [ ] **Step 5: Run the complete configuration and registration tests**

Run:

```bash
python -m pytest -q tests/test_cross_tuning_configs.py
```

Expected: `8 passed`; each runner is identical to its donor after normalizing
only `experiment_name`.

- [ ] **Step 6: Compile, inspect, and commit runners and registrations**

Run:

```bash
python -m compileall -q \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/agents/rsl_rl_ppo_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py \
  tests/test_cross_tuning_configs.py
git diff --check
git diff -- \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/agents/rsl_rl_ppo_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py \
  tests/test_cross_tuning_configs.py
git add \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/agents/rsl_rl_ppo_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py \
  tests/test_cross_tuning_configs.py
git commit -m "Register crossed tuning baseline tasks"
```

Expected: checks pass and the commit contains only the local runners,
registrations, and extended contracts.

---

### Task 3: Manual Training Recipes

**Files:**
- Create: `tests/test_cross_tuning_justfile.py`
- Modify: `justfile`

**Interfaces:**
- Consumes: `_train-rsl-baseline task num_envs seed max_iterations wandb_project *flags`.
- Produces: `train-baseline-go2-anymal-c-tuning num_envs seed max_iterations wandb_project *flags`.
- Produces: `train-baseline-anymal-c-go2-tuning num_envs seed max_iterations wandb_project *flags`.

- [ ] **Step 1: Write failing dry-run tests for the manual recipes**

Create `tests/test_cross_tuning_justfile.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "recipe,task,default_project",
    [
        (
            "train-baseline-go2-anymal-c-tuning",
            "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0",
            "baseline_go2_anymal_c_rewards_action_scale_ppo",
        ),
        (
            "train-baseline-anymal-c-go2-tuning",
            "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0",
            "baseline_anymal_c_go2_rewards_action_scale_ppo",
        ),
    ],
)
def test_cross_tuning_recipe_defaults(recipe, task, default_project):
    result = subprocess.run(
        ["just", "--dry-run", recipe],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expansion = result.stdout + result.stderr

    assert f"--task={task}" in expansion
    assert "--num_envs=7500" in expansion
    assert "--seed=46" in expansion
    assert "--max_iterations=30000" in expansion
    assert "--logger=wandb" in expansion
    assert f"--log_project_name={default_project}" in expansion


@pytest.mark.parametrize(
    "recipe,task",
    [
        (
            "train-baseline-go2-anymal-c-tuning",
            "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0",
        ),
        (
            "train-baseline-anymal-c-go2-tuning",
            "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0",
        ),
    ],
)
def test_cross_tuning_recipe_forwards_overrides_and_flags(recipe, task):
    result = subprocess.run(
        [
            "just",
            "--dry-run",
            recipe,
            "64",
            "47",
            "2",
            "custom_cross_project",
            "--run_name=cross_smoke",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expansion = result.stdout + result.stderr

    assert f"--task={task}" in expansion
    assert "--num_envs=64" in expansion
    assert "--seed=47" in expansion
    assert "--max_iterations=2" in expansion
    assert "--log_project_name=custom_cross_project" in expansion
    assert "--run_name=cross_smoke" in expansion
```

- [ ] **Step 2: Run the recipe tests and verify the missing-recipe failure**

Run:

```bash
python -m pytest -q tests/test_cross_tuning_justfile.py
```

Expected: four failures because `just` reports each crossed recipe as unknown.

- [ ] **Step 3: Add the two public recipes**

Add immediately after the existing matched Go2 and ANYmal C recipes in
`justfile`:

```just
train-baseline-go2-anymal-c-tuning num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_go2_anymal_c_rewards_action_scale_ppo" *flags:
    just _train-rsl-baseline Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

train-baseline-anymal-c-go2-tuning num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_anymal_c_go2_rewards_action_scale_ppo" *flags:
    just _train-rsl-baseline Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}
```

- [ ] **Step 4: Run the recipe tests and list the new interface**

Run:

```bash
python -m pytest -q tests/test_cross_tuning_justfile.py
just --list | rg 'train-baseline-(go2-anymal-c|anymal-c-go2)-tuning'
```

Expected: `4 passed` and both public recipe names appear.

- [ ] **Step 5: Check and commit the manual training interface**

Run:

```bash
git diff --check
git diff -- justfile tests/test_cross_tuning_justfile.py
git add justfile tests/test_cross_tuning_justfile.py
git commit -m "Add manual crossed baseline training recipes"
```

Expected: the commit contains only the two recipes and their simulator-free
dry-run tests.

---

### Task 4: Static and Simulator Verification

**Files:**
- Verify: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_cross_tuning_env_cfg.py`
- Verify: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/agents/rsl_rl_ppo_cfg.py`
- Verify: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`
- Verify: `justfile`
- Verify: `tests/test_cross_tuning_configs.py`
- Verify: `tests/test_cross_tuning_justfile.py`

**Interfaces:**
- Consumes: both crossed training task IDs and all contracts from Tasks 1-3.
- Produces: evidence that both environments can instantiate, reset, step, and provide finite observations/rewards on CUDA.

- [ ] **Step 1: Run the complete static and configuration test suite**

Run:

```bash
python -m compileall -q exts/cat_envs/cat_envs tests
python -m pytest -q tests/test_cross_tuning_justfile.py tests/test_generate_plots.py
python -m pytest -q tests/test_cross_tuning_configs.py
git diff --check
```

Expected: compilation succeeds; fast tests pass; crossed configuration tests
pass with a clean Isaac Sim shutdown; diff check is silent.

- [ ] **Step 2: Start, reset, and step crossed Go2**

Run:

```bash
python -c 'from isaaclab.app import AppLauncher; launcher=AppLauncher(headless=True); import gymnasium as gym, torch; import cat_envs.tasks; from isaaclab_tasks.utils import parse_env_cfg; task="Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0"; cfg=parse_env_cfg(task,device="cuda:0",num_envs=16); env=gym.make(task,cfg=cfg); obs,_=env.reset(); assert torch.isfinite(obs["policy"]).all(); actions=torch.zeros(env.action_space.shape,device=env.unwrapped.device); obs,reward,terminated,truncated,info=env.step(actions); assert torch.isfinite(obs["policy"]).all(); assert torch.isfinite(reward).all(); print(task,"obs",tuple(obs["policy"].shape),"reward",tuple(reward.shape),"action_scale",cfg.actions.joint_pos.scale); env.close(); launcher.app.close()'
```

Expected: exit 0, 12 actions, finite tensors, and printed action scale `0.5`.

- [ ] **Step 3: Start, reset, and step crossed ANYmal C**

Run:

```bash
python -c 'from isaaclab.app import AppLauncher; launcher=AppLauncher(headless=True); import gymnasium as gym, torch; import cat_envs.tasks; from isaaclab_tasks.utils import parse_env_cfg; task="Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0"; cfg=parse_env_cfg(task,device="cuda:0",num_envs=16); env=gym.make(task,cfg=cfg); obs,_=env.reset(); assert torch.isfinite(obs["policy"]).all(); actions=torch.zeros(env.action_space.shape,device=env.unwrapped.device); obs,reward,terminated,truncated,info=env.step(actions); assert torch.isfinite(obs["policy"]).all(); assert torch.isfinite(reward).all(); print(task,"obs",tuple(obs["policy"].shape),"reward",tuple(reward.shape),"action_scale",cfg.actions.joint_pos.scale); env.close(); launcher.app.close()'
```

Expected: exit 0, 12 actions, finite tensors, and printed action scale `0.25`.

- [ ] **Step 4: Inspect the final implementation boundary**

Run:

```bash
git diff HEAD~3 -- \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12 \
  justfile \
  tests/test_cross_tuning_configs.py \
  tests/test_cross_tuning_justfile.py
git status --short
```

Expected: only crossed reward weights, action scales, donor runner metadata,
registrations, recipes, and tests were added; no production process is running;
the worktree is clean.

---

### Task 5: Two-Iteration Trainer Smoke Tests and Manual Handoff

**Files:**
- Verify: `scripts/train_rsl_rl.py`
- Verify: generated ignored paths under `logs/rsl_rl/`

**Interfaces:**
- Consumes: crossed task registrations and RSL-RL runner configs.
- Produces: two finite two-iteration checkpoints/config artifacts using local TensorBoard only.
- Produces: the exact manual production commands for the user.

- [ ] **Step 1: Confirm no production trainer is already active**

Run:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'scripts/train_rsl_rl.py.*Baseline-(Go2-Anymal-C|Anymal-C-Go2)-Tuning' || true
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Expected: no crossed production trainer process. Record available GPU memory
before the smoke runs.

- [ ] **Step 2: Run a two-iteration Go2 trainer smoke test without W&B**

Run:

```bash
python scripts/train_rsl_rl.py \
  --task=Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0 \
  --seed=46 \
  --headless \
  --num_envs=64 \
  --max_iterations=2 \
  --logger=tensorboard \
  --run_name=smoke_go2_anymal_c_tuning \
  env.scene.terrain.terrain_generator.seed=46 \
  env.sim.random_seed=46
```

Expected: two iterations complete, losses and rewards remain finite, and the
saved environment config contains action scale `0.5` and the ANYmal C numerical
reward weights while `undesired_contacts` is absent.

- [ ] **Step 3: Run a two-iteration ANYmal C trainer smoke test without W&B**

Run:

```bash
python scripts/train_rsl_rl.py \
  --task=Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0 \
  --seed=46 \
  --headless \
  --num_envs=64 \
  --max_iterations=2 \
  --logger=tensorboard \
  --run_name=smoke_anymal_c_go2_tuning \
  env.scene.terrain.terrain_generator.seed=46 \
  env.sim.random_seed=46
```

Expected: two iterations complete, losses and rewards remain finite, and the
saved environment config contains action scale `0.25`, the Go2 numerical reward
weights, and ANYmal C `undesired_contacts` at `-1.0`.

- [ ] **Step 4: Verify saved smoke artifacts and final repository state**

Run:

```bash
rg -n \
  'scale: (0.5|0.25)|track_lin_vel_xy_exp|dof_torques_l2|feet_air_time|undesired_contacts|entropy_coef' \
  logs/rsl_rl/go2_anymal_c_tuning_rough \
  logs/rsl_rl/anymal_c_go2_tuning_rough
git status --short
```

Expected: both crossed log roots contain saved resolved parameters and a model
artifact; the repository remains clean because training logs are ignored.

- [ ] **Step 5: Hand off production commands without running them**

Report these exact commands to the user:

```bash
just train-baseline-go2-anymal-c-tuning
just train-baseline-anymal-c-go2-tuning
```

State that each command defaults to 7,500 environments, seed 46, and 30,000
iterations, and sends metrics to its combination-specific W&B project. Recommend
running them sequentially. If the first run fails with CUDA out-of-memory,
recommend retrying that same recipe with an explicit lower environment count,
for example:

```bash
just train-baseline-go2-anymal-c-tuning 4096
```

Do not run either production command during implementation or validation.

---

## Final Verification Checklist

- [ ] `git status --short` is clean.
- [ ] `git diff --check` is silent.
- [ ] All eight parametrized cases in `tests/test_cross_tuning_configs.py` pass.
- [ ] All four cases in `tests/test_cross_tuning_justfile.py` pass.
- [ ] Existing `tests/test_generate_plots.py` remains green.
- [ ] Both crossed tasks reset and step with finite CUDA tensors.
- [ ] Both two-iteration TensorBoard smoke trainings complete.
- [ ] No W&B production run or Slurm job was started.
- [ ] The final response names the transferred reward/action/PPO fields, the retained contact behavior, validation evidence, and both manual production commands.
