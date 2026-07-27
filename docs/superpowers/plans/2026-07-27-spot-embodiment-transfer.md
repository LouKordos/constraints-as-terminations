# Boston Dynamics Spot Embodiment Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trainable and evaluable CaT rough-terrain tasks for Boston Dynamics Spot while preserving the working Go2/ANYmal method, unchanged primary metrics, and historical Go2 evaluation behavior.

**Architecture:** Implement Spot as a narrow subclass of the working Go2 CaT configuration. Replace only the installed robot asset, names, action scale, simulation frequency, physical constraints, root mass range, and energy endpoint; inherit terrain, commands, observations/noise, tracking rewards, curricula, and termination machinery. Keep `eval.py` unchanged unless a focused runtime check proves a missing Spot requirement, because it already contains the correct Spot profile.

**Tech Stack:** Python 3.11, Isaac Lab 2.3.1, Isaac Sim 5.1, Gymnasium, PyTorch, CleanRL PPO, NumPy, pytest, YAML.

## Global Constraints

- Work only on branch `cross-embodiment`; check it before every commit and never switch branches.
- Do not modify sim-to-real files.
- Do not refactor the validated Go2/ANYmal configuration hierarchy.
- Preserve the rough-terrain generator, terrain curriculum definition, commands, 10-second episode length, observation semantics/scales/noise, tracking rewards, constraint curricula, and energy-curriculum timing.
- Preserve the definitions of achieved terrain level, cost of transport, and body-frame x/y/yaw RMS tracking error.
- Preserve unknown/pre-branch task fallback to Go2 in `eval.py`.
- Use installed `SPOT_CFG` with delayed PD hips and remotized PD knees.
- Spot physics timestep is `0.002` seconds with decimation `10`; the environment control period remains `0.02` seconds.
- Spot action scale is `0.2` rad.
- Spot constraints start at torque `80` N m, velocity `20` rad/s, acceleration `800` rad/s^2, normalized action rate `80` s^-1, foot force `800` N, absolute hip-y position `2.0` rad, relative hip-x position `0.3` rad, and standstill velocity `2.0` rad/s.
- Spot added root mass is `+/-3` kg.
- Spot energy endpoint is `0.00380`; the documented bracket is `0.00190/0.00380/0.00760`.
- Spot observation noise and training disturbances must exactly equal current ANYmal: velocity push `+/-0.25` m/s, external force `+/-10` N, external torque `+/-0.5` N m, with identical event timing.
- Do not reduce Spot's velocity bound to 12 rad/s.
- Use focused tests and simulator smokes; do not add a broad unit-test layer.
- Do not start a multi-day training run. A 64-environment, two-iteration PPO startup is the longest implementation-time training check.
- Use elevated execution for GPU/Isaac commands. Restricted-sandbox `nvidia-smi` is not valid GPU-health evidence.
- Make a focused commit after each independently reviewable deliverable.

## File Map

- Create `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py`: Spot train/play subclasses, explicit names, physical values, timing, and evaluation sensors.
- Modify `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`: register Spot train/play task IDs.
- Modify `scripts/generate_plots.py`: recognize Spot `hx/hy/kn` joint roles.
- Modify `tests/test_generate_plots.py`: focused Spot joint-layout regression check.
- Create `docs/spot_tuning_assumptions.md`: verified facts, starting assumptions, tuning order, and validation/training evidence.
- Modify `docs/anymal_c_tuning_assumptions.md`: distinguish original `60/7/300` from successful `80/12/600`.
- Validate without modifying `scripts/eval.py`, `scripts/metrics_utils.py`, `scripts/clean_rl/train.py`, and the Go2/ANYmal environment files unless a concrete failing check proves a required correction.

---

### Task 1: Add Spot Joint Plot Layout Support

**Files:**
- Modify: `tests/test_generate_plots.py`
- Modify: `scripts/generate_plots.py:583-596`

**Interfaces:**
- Consumes: `_build_joint_layout(joint_names: list[str]) -> tuple[list[int], list[int], list[int]]`.
- Produces: Spot role parsing for `hx`, `hy`, and `kn` without changing Go2/ANYmal row or column mappings.

- [ ] **Step 1: Add the focused failing Spot layout test**

Append:

```python
def test_build_joint_layout_supports_spot_mapping():
    joint_names = [
        "fl_hx", "fr_hx", "hl_hx", "hr_hx",
        "fl_hy", "fr_hy", "hl_hy", "hr_hy",
        "fl_kn", "fr_kn", "hl_kn", "hr_kn",
    ]

    _assert_complete_joint_layout(joint_names, [0, 1, 2, 3] * 3)
```

- [ ] **Step 2: Run the new test and verify the current parser fails**

Run:

```bash
pytest -q tests/test_generate_plots.py::test_build_joint_layout_supports_spot_mapping
```

Expected: FAIL with `ValueError: Could not determine joint row/col for plotting based on names`.

- [ ] **Step 3: Add only the missing Spot role synonyms**

Replace the dictionary with:

```python
JOINT_TYPE_SYNONYMS = {
    0: ("hip", "haa", "hx"),
    1: ("thigh", "hfe", "hy"),
    2: ("calf", "kfe", "kn"),
}
```

Do not change `_leg_prefixes`; it already recognizes `FL/FR/HL/HR`.

- [ ] **Step 4: Run all focused layout checks**

Run:

```bash
pytest -q tests/test_generate_plots.py
```

Expected: `3 passed`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
python -m py_compile scripts/generate_plots.py tests/test_generate_plots.py
git diff --check
git branch --show-current
```

Expected: compilation exit 0, no diff errors, branch `cross-embodiment`.

Commit:

```bash
git add scripts/generate_plots.py tests/test_generate_plots.py
git commit -m "Support Spot joint plot layout"
```

---

### Task 2: Add and Register the Spot CaT Environment

**Files:**
- Create: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py`
- Modify: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`

**Interfaces:**
- Consumes: `Go2RoughTerrainEnvCfg`, `force_hard_terrain`, installed `isaaclab_assets.robots.spot.SPOT_CFG`.
- Produces: `SPOT_JOINT_NAMES`, `SPOT_JOINT_PATTERNS`, `SPOT_FOOT_NAMES`, `SpotRoughTerrainEnvCfg`, `SpotRoughTerrainEnvCfg_PLAY`, `CaT-Spot-Rough-Terrain-v0`, and `CaT-Spot-Rough-Terrain-Play-v0`.

- [ ] **Step 1: Run a registration check proving the task is absent**

Run with elevated GPU/Isaac execution:

```bash
python -c 'from isaaclab.app import AppLauncher; launcher=AppLauncher(headless=True); import gymnasium as gym; import cat_envs.tasks.locomotion.velocity.config.solo12; assert "CaT-Spot-Rough-Terrain-v0" in gym.envs.registry; launcher.app.close()'
```

Expected before implementation: `AssertionError`.

- [ ] **Step 2: Create the complete narrow Spot configuration**

Create the file with:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CaT rough-terrain configurations for Boston Dynamics Spot."""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer import FrameTransformerCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets.robots.spot import SPOT_CFG

from .cat_go2_rough_terrain_env_cfg import Go2RoughTerrainEnvCfg, force_hard_terrain


SPOT_JOINT_NAMES = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]
SPOT_JOINT_PATTERNS = [".*_hx", ".*_hy", ".*_kn"]
SPOT_FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


def _foot_ray_caster(foot_name: str, sim_dt: float) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}",
        update_period=sim_dt,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
        debug_vis=True,
    )


@configclass
class SpotRoughTerrainEnvCfg(Go2RoughTerrainEnvCfg):
    """Shared CaT rough-terrain MDP with Spot-specific embodiment settings."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos.joint_names = list(SPOT_JOINT_NAMES)
        self.actions.joint_pos.scale = 0.2
        self.observations.policy.joint_pos_history.params["names"] = list(SPOT_JOINT_NAMES)
        self.observations.policy.joint_vel_history.params["names"] = list(SPOT_JOINT_NAMES)

        self.scene.ray_caster.prim_path = "{ENV_REGEX_NS}/Robot/body"
        self.scene.ray_caster_height_constraints.prim_path = "{ENV_REGEX_NS}/Robot/body"

        self.events.randomize_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["body"])
        self.events.randomize_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["body"])
        self.events.randomize_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
        }
        self.events.push_base_wrench.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["body"]
        )
        self.events.push_base_wrench.params["force_range"] = (-10.0, 10.0)
        self.events.push_base_wrench.params["torque_range"] = (-0.5, 0.5)

        for term_name in ("joint_torque", "joint_velocity", "joint_acceleration", "action_rate"):
            getattr(self.constraints, term_name).params["names"] = list(SPOT_JOINT_PATTERNS)
        self.constraints.joint_torque.params["limit"] = 80.0
        self.constraints.joint_velocity.params["limit"] = 20.0
        self.constraints.joint_acceleration.params["limit"] = 800.0
        self.constraints.action_rate.params["limit"] = 80.0

        self.constraints.contact.params["names"] = ["body", ".*_uleg"]
        self.constraints.foot_contact_force.params.update(limit=800.0, names=[".*_foot"])
        self.constraints.front_hfe_position.params.update(limit=2.0, names=[".*_hy"])
        self.constraints.hip_position.params["names"] = [".*_hx"]
        self.constraints.no_move.params.update(
            names=list(SPOT_JOINT_PATTERNS),
            joint_vel_limit=2.0,
        )

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["body", ".*_uleg"]
        )
        self.curriculum.power.params["end_weight"] = 0.00380

        # Preserve Spot's installed 0--8 ms actuator delay while retaining 50 Hz control.
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.ray_caster.update_period = self.sim.dt
        self.scene.ray_caster_height_constraints.update_period = self.sim.dt


@configclass
class SpotRoughTerrainEnvCfg_PLAY(SpotRoughTerrainEnvCfg):
    """One-environment disturbance-free Spot configuration for evaluation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 8
        self.apply_elevation_map_point_noise = False
        self.observations.policy.enable_corruption = True

        self.events.force_hard_terrain = EventTerm(func=force_hard_terrain, mode="startup")
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.push_robot = None
        self.events.push_base_wrench = None

        for foot_name in SPOT_FOOT_NAMES:
            setattr(
                self.scene,
                f"ray_caster_{foot_name}",
                _foot_ray_caster(foot_name, self.sim.dt),
            )
        self.scene.foot_frame_transformer = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}"
                )
                for foot_name in SPOT_FOOT_NAMES
            ],
            debug_vis=False,
        )

        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.6)
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.rewards.minimize_power = None
        self.curriculum.power = None
```

- [ ] **Step 3: Register train and play task IDs**

Add after the corresponding ANYmal registrations:

```python
gym.register(
    id="CaT-Spot-Rough-Terrain-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_spot_rough_terrain_env_cfg:SpotRoughTerrainEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)
```

and:

```python
gym.register(
    id="CaT-Spot-Rough-Terrain-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_spot_rough_terrain_env_cfg:SpotRoughTerrainEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)
```

- [ ] **Step 4: Compile and verify registration**

Run:

```bash
python -m py_compile \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py
```

Then run with elevated GPU/Isaac execution:

```bash
python -c 'from isaaclab.app import AppLauncher; launcher=AppLauncher(headless=True); import gymnasium as gym; import cat_envs.tasks.locomotion.velocity.config.solo12; assert "CaT-Spot-Rough-Terrain-v0" in gym.envs.registry; assert "CaT-Spot-Rough-Terrain-Play-v0" in gym.envs.registry; launcher.app.close()'
```

Expected: exit 0.

Do not commit until Tasks 3 and 4 have validated and documented this configuration.

---

### Task 3: Record Successful ANYmal Values and Spot Assumptions

**Files:**
- Modify: `docs/anymal_c_tuning_assumptions.md`
- Create: `docs/spot_tuning_assumptions.md`

**Interfaces:**
- Consumes: commit `3ec95de`, installed Spot facts, and the approved design values.
- Produces: current fact/assumption registers used to interpret early training and choose one-at-a-time revisions.

- [ ] **Step 1: Correct the ANYmal comparison table without erasing history**

Change the table header from:

```markdown
| Category | Parameter | Go2/reference | ANYmal C initial | Why this is the initial value | Early failure signal | Next evidence-driven candidate |
```

to:

```markdown
| Category | Parameter | Go2/reference | ANYmal C initial -> successful | Rationale/evidence | Early failure signal | Next evidence-driven candidate |
```

Update only the three successful rows:

```markdown
| Soft constraint | torque | 20 N m | 60 -> 80 N m | The initial 60 N m bound was too restrictive; commit `3ec95de` raised it to the installed 80 N m continuous limit and the user reported that this configuration learned successfully. | Applied torque/constraint probability dominates before tracking improves. | Retain 80 N m unless a controlled post-convergence tightening experiment is needed. |
| Soft constraint | joint velocity | 25 rad/s | 7.0 -> 12 rad/s | The installed 7.5 rad/s nominal speed was not a usable learning bound for this actuator-network simulation; commit `3ec95de` raised it to 12 rad/s. | Persistent velocity pressure prevents tracking/terrain progress. | Retain 12 rad/s for the demonstrated configuration. |
| Soft constraint | joint acceleration | 800 rad/s^2 | 300 -> 600 rad/s^2 | The initial value was too restrictive during exploration; commit `3ec95de` raised it to 600 rad/s^2. | Acceleration probability dominates policy updates. | Retain 600 rad/s^2 for the demonstrated configuration. |
```

Append:

```markdown
## Successful training update: 2026-07-27

The user reported that the ANYmal C configuration learned successfully after commit `3ec95de` changed torque, velocity, and acceleration constraints from `60/7/300` to `80/12/600`. No other embodiment values changed in that commit. This updates the authoritative successful configuration but does not invent unreported convergence metrics.
```

- [ ] **Step 2: Create the Spot tuning register**

Create `docs/spot_tuning_assumptions.md` with:

```markdown
# Boston Dynamics Spot Starting Values and Tuning Register

This living record separates values verified from installed Isaac Lab 2.3.1 / Isaac Sim 5.1 from learning-first assumptions. Update it after every smoke, startup, or substantive training decision. Change one parameter family at a time.

## Verified embodiment facts

| Item | Verified value | Consequence |
|---|---:|---|
| Total mass | 31.60000 kg | Runtime mass is used for CoT and energy scaling. |
| Root | `body` | Use for COM/mass randomization, wrench, sensors, and illegal contact. |
| Feet | `fl_foot`, `fr_foot`, `hl_foot`, `hr_foot` | Evaluation order is front-left, front-right, rear-left, rear-right. |
| Joint order | `fl/fr/hl/hr_hx`, then `_hy`, then `_kn` | Actions and proprioception use this exact runtime order. |
| Default pose | hx +/-0.1; front hy 0.9; hind hy 1.1; knees -1.5 rad | Exact default reset is used for play. |
| Root height | 0.5 m | Evaluation scenario height adjustment. |
| Hip actuator | delayed PD, 45 N m effort, 0--8 ms configured delay at 0.002 s physics | Keep installed model and physics period. |
| Knee actuator | delayed/remotized PD, about 31--113 N m angle-dependent torque | One 80 N m CaT term is a learning-first compromise. |
| Joint position limits | hx +/-0.7854; hy -0.8988 to 2.2951; knee -2.7929 to -0.2471 rad | The 2.0 rad hard hy bound remains inside the physical range. |
| Dimensions | 12 actions, expected 188 policy observations | Runtime smoke must verify both. |
| Timing | 0.002 s physics, decimation 10, 0.02 s control | Preserves actuator delay and cross-embodiment control rate. |

## Starting assumptions

| Category | Spot value | Why | Stop/relax signal | First candidate |
|---|---:|---|---|---|
| Action scale | 0.2 rad | Installed upstream Spot value. | Insufficient/saturated foot motion with otherwise healthy learning. | Change only after inspecting targets and joint ranges. |
| Torque | 80 N m | Hips cap at 45; knees reach about 113 near default; avoids repeating restrictive ANYmal start. | Knee torque constraint dominates before tracking. | Measure hip/knee separately before changing or splitting terms. |
| Velocity | 20 rad/s | Learning-first bound; USD 100/infinity values are not physical bounds. | Velocity constraint dominates before tracking. | Increase from evidence; do not reduce to 12 rad/s automatically. |
| Acceleration | 800 rad/s^2 | Allows delayed/remotized 500 Hz actuator transients. | Acceleration pressure prevents tracking. | 1200 rad/s^2. |
| Action rate | 80 s^-1 | Same normalized-action definition across robots. | Permanent domination/chatter mismatch. | Inspect normalized action distribution first. |
| Foot force | 800 N | Explicit learning margin for early landings. | Ordinary landings repeatedly terminate. | Increase only from contact/video evidence. |
| Absolute hip-y | 2.0 rad | Below 2.2951 physical upper limit with margin over 0.9/1.1 defaults. | Normal poses terminate. | Reassess geometry and measured distribution. |
| Relative hip-x | 0.3 rad | Same semantic style constraint and larger than 0.2 action scale. | Necessary lateral stabilization is blocked. | Inspect distribution before widening. |
| Standstill velocity | 2 rad/s | Successful ANYmal value. | Standstill term dominates. | Relax only from command-conditioned evidence. |
| Added body mass | +/-3 kg | About 9.5% of total mass, matching Go2/ANYmal proportion. | Startup instability correlates with mass bucket. | +/-1.5 kg temporarily. |
| Energy endpoint | 0.00380 | Keeps endpoint times mass near 0.120 across all robots. | Energy blocks tracking or remains irrelevant after locomotion. | 0.00190 or 0.00760. |
| Sole offset | 0.0 m | No verified Spot sole-frame offset. | Secondary foot-height metrics show bias. | Measure collision/frame geometry; primary metrics are unaffected. |

## Fixed shared settings

- Preserve CaT observation terms, scales, and noise exactly.
- Preserve ANYmal disturbances exactly: +/-0.25 m/s velocity, +/-10 N force, +/-0.5 N m torque, and identical timing.
- Preserve terrain, commands, tracking rewards, episode duration, friction, COM range, reset semantics, and curriculum timing.
- Disable both disturbance events in play/evaluation.

## First 24-hour evidence

- Finite PPO losses, action standard deviation, and action saturation.
- Episode lengths and reset fractions by source.
- Tracking and terrain-level trends.
- Per-constraint violation frequency/probability and maxima.
- Hip/knee torque distributions, velocity, acceleration, action rate, foot force, and hip-y position.
- Raw power, episode energy, and energy reward relative to tracking.
- Videos and the exact commit/seed/environment count.

Continue only when tracking, survival, and terrain progression improve without one constraint or the energy term dominating. Stop for NaN/Inf, immediate-reset loops, collapse, terrain pinned at zero with flat tracking, ordinary 800 N foot-force terminations, normal 2.0 rad hip-y terminations, or persistent soft-constraint/energy pressure blocking learning.

## Validation evidence

No Spot implementation validation has been completed yet. Record construction, simulator, PPO-startup, and evaluation evidence here without claiming convergence.
```

- [ ] **Step 3: Verify documentation**

Run:

```bash
git diff --check
rg -n '60 -> 80|7.0 -> 12|300 -> 600|80 N m|20 rad/s|800 rad/s|800 N|2.0 rad|0.00380' \
  docs/anymal_c_tuning_assumptions.md docs/spot_tuning_assumptions.md
```

Expected: every revised/starting value appears and no diff error.

Do not commit until Task 4 passes.

---

### Task 4: Validate Spot Configuration and Runtime Construction

**Files:**
- Validate: `cat_spot_rough_terrain_env_cfg.py`
- Validate: `cat_anymal_c_rough_terrain_env_cfg.py`
- Validate: `cat_go2_rough_terrain_env_cfg.py`
- Update after evidence: `docs/spot_tuning_assumptions.md`

**Interfaces:**
- Consumes: registered Spot train/play configs.
- Produces: evidence that static values, runtime entities, actuator models, sensors, diagnostics, and finite stepping match the design.

- [ ] **Step 1: Verify GPU visibility through the correct boundary**

Run elevated:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader
```

Expected: RTX 2080 Ti, driver 535.288.01, and sufficient free memory.

- [ ] **Step 2: Run static train/play and cross-embodiment assertions**

Run with elevated Isaac execution:

```bash
python -c '
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
from isaaclab_tasks.utils import parse_env_cfg
import cat_envs.tasks.locomotion.velocity.config.solo12

spot = parse_env_cfg("CaT-Spot-Rough-Terrain-v0", device="cuda:0", num_envs=1)
spot_play = parse_env_cfg("CaT-Spot-Rough-Terrain-Play-v0", device="cuda:0", num_envs=1)
anymal = parse_env_cfg("CaT-Anymal-C-Rough-Terrain-v0", device="cuda:0", num_envs=1)
go2 = parse_env_cfg("CaT-Go2-Rough-Terrain-v0", device="cuda:0", num_envs=1)

expected = [
    "fl_hx", "fr_hx", "hl_hx", "hr_hx",
    "fl_hy", "fr_hy", "hl_hy", "hr_hy",
    "fl_kn", "fr_kn", "hl_kn", "hr_kn",
]
assert spot.actions.joint_pos.joint_names == expected
assert spot.observations.policy.joint_pos_history.params["names"] == expected
assert spot.observations.policy.joint_vel_history.params["names"] == expected
assert spot.actions.joint_pos.scale == 0.2
assert spot.sim.dt == 0.002 and spot.decimation == 10
assert spot.sim.dt * spot.decimation == 0.02
assert spot.scene.contact_forces.update_period == 0.002
assert spot.scene.ray_caster.update_period == 0.002
assert spot.scene.ray_caster_height_constraints.update_period == 0.002
assert spot.scene.ray_caster.prim_path.endswith("/body")
assert spot.scene.ray_caster_height_constraints.prim_path.endswith("/body")
assert spot.constraints.joint_torque.params["limit"] == 80.0
assert spot.constraints.joint_velocity.params["limit"] == 20.0
assert spot.constraints.joint_acceleration.params["limit"] == 800.0
assert spot.constraints.action_rate.params["limit"] == 80.0
assert spot.constraints.foot_contact_force.params["limit"] == 800.0
assert spot.constraints.front_hfe_position.params["limit"] == 2.0
assert spot.constraints.front_hfe_position.params["names"] == [".*_hy"]
assert spot.constraints.hip_position.params["names"] == [".*_hx"]
assert spot.constraints.no_move.params["joint_vel_limit"] == 2.0
assert spot.curriculum.power.params["end_weight"] == 0.00380
assert spot.events.randomize_mass.params["mass_distribution_params"] == (-3.0, 3.0)

for name in ("base_ang_vel", "projected_gravity", "joint_pos_history", "joint_vel_history", "height_map"):
    assert repr(getattr(spot.observations.policy, name).noise) == repr(
        getattr(anymal.observations.policy, name).noise
    )
assert spot.events.push_robot.interval_range_s == anymal.events.push_robot.interval_range_s
assert spot.events.push_robot.params["velocity_range"] == anymal.events.push_robot.params["velocity_range"]
assert spot.events.push_base_wrench.interval_range_s == anymal.events.push_base_wrench.interval_range_s
assert spot.events.push_base_wrench.params["force_range"] == anymal.events.push_base_wrench.params["force_range"]
assert spot.events.push_base_wrench.params["torque_range"] == anymal.events.push_base_wrench.params["torque_range"]

assert spot_play.events.push_robot is None
assert spot_play.events.push_base_wrench is None
assert spot_play.observations.policy.enable_corruption is True
assert spot_play.rewards.minimize_power is None
assert spot_play.curriculum.power is None

assert go2.actions.joint_pos.scale == 0.8
assert go2.constraints.joint_torque.params["limit"] == 20.0
assert anymal.actions.joint_pos.scale == 0.5
assert anymal.constraints.joint_torque.params["limit"] == 80.0
assert anymal.constraints.joint_velocity.params["limit"] == 12.0
assert anymal.constraints.joint_acceleration.params["limit"] == 600.0
launcher.app.close()
'
```

Expected: exit 0.

- [ ] **Step 3: Run one-environment train and play simulator smokes**

Run with elevated Isaac execution:

```bash
python -c '
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
import gymnasium as gym
import torch
import cat_envs.tasks.locomotion.velocity.config.solo12
from isaaclab_tasks.utils import parse_env_cfg

for task in ("CaT-Spot-Rough-Terrain-v0", "CaT-Spot-Rough-Terrain-Play-v0"):
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset(seed=46)
    u = env.unwrapped
    robot = u.scene["robot"]
    assert u.action_manager.total_action_dim == 12
    assert obs["policy"].shape == (1, 188)
    assert u.physics_dt == 0.002 and u.step_dt == 0.02
    assert robot.data.joint_names == [
        "fl_hx", "fr_hx", "hl_hx", "hr_hx",
        "fl_hy", "fr_hy", "hl_hy", "hr_hy",
        "fl_kn", "fr_kn", "hl_kn", "hr_kn",
    ]
    assert abs(float(robot.data.default_mass.sum()) - 31.6) < 0.01
    assert type(robot.actuators["spot_hip"]).__name__ == "DelayedPDActuator"
    assert type(robot.actuators["spot_knee"]).__name__ == "RemotizedPDActuator"
    assert u.constraint_manager._diagnostic_joint_names == [".*_hx", ".*_hy", ".*_kn"]
    assert u.constraint_manager._diagnostic_foot_names == [".*_foot"]

    for _ in range(25):
        actions = torch.zeros((1, 12), device=u.device)
        obs, reward, _, _, _ = env.step(actions)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all()
        assert torch.isfinite(robot.data.applied_torque).all()
        assert torch.isfinite(robot.data.joint_vel).all()
        assert torch.isfinite(robot.data.joint_acc).all()
        assert torch.isfinite(u.constraint_manager.cat.get_probs()).all()
        assert torch.isfinite(u.constraint_manager._episode_energy_consumed).all()

    if "Play" in task:
        assert u.scene["foot_frame_transformer"].cfg.prim_path.endswith("/body")
        assert u.scene["foot_frame_transformer"].data.target_pos_w.shape[1] == 4
        assert all(name in robot.data.body_names for name in ("fl_foot", "fr_foot", "hl_foot", "hr_foot"))
    env.close()

launcher.app.close()
'
```

Expected: exit 0 with finite values for both tasks.

- [ ] **Step 4: Validate the existing evaluator against the custom Spot play config**

Run with elevated Isaac execution:

```bash
python -c '
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
import sys
sys.path.insert(0, "scripts")
import eval as evaluation
import cat_envs.tasks.locomotion.velocity.config.solo12
from isaaclab_tasks.utils import parse_env_cfg

profile = evaluation.resolve_robot_eval_profile("CaT-Spot-Rough-Terrain-Play-v0")
assert profile.name == "spot"
assert profile.root_link == "body"
assert profile.foot_links == ("fl_foot", "fr_foot", "hl_foot", "hr_foot")
assert profile.spawn_height == 0.5
assert profile.joint_role_mapping["hx"]["RL"] == "hl_hx"

cfg = parse_env_cfg("CaT-Spot-Rough-Terrain-Play-v0", device="cuda:0", num_envs=1)
evaluation.add_eval_foot_sensors_to_env_cfg(cfg, profile.foot_links, profile.root_link, cfg.sim.dt)
assert cfg.scene.foot_frame_transformer.prim_path.endswith("/body")
assert cfg.actions.joint_pos.scale == 0.2
launcher.app.close()
'
```

Expected: exit 0. Do not modify `eval.py` if this passes.

- [ ] **Step 5: Append construction evidence to the Spot register**

Replace the initial validation paragraph with exact observed results:

```markdown
## Validation evidence

On 2026-07-27, static train/play configuration checks passed for the agreed values, exact ANYmal noise/disturbance equality, and unchanged Go2/ANYmal reference values. One-environment train and play tasks each reset and completed 25 zero-action steps with finite 188-dimensional observations, rewards, constraint probabilities, applied torques, velocities, accelerations, contact telemetry, and integrated power. Runtime mass was 31.6 kg; action dimension was 12; physics/control periods were 0.002/0.02 s; the installed delayed hip and remotized knee actuators remained active; and the play frame transformer resolved `body` plus four feet. This is construction evidence, not convergence evidence.
```

Adjust the text only if actual observed values differ; do not claim a failed assertion passed.

- [ ] **Step 6: Verify and commit the environment deliverable**

Run:

```bash
python -m py_compile \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py
pytest -q tests/test_generate_plots.py
git diff --check
git branch --show-current
```

Expected: compilation exit 0, `3 passed`, clean diff check, branch `cross-embodiment`.

Commit:

```bash
git add \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py \
  docs/anymal_c_tuning_assumptions.md \
  docs/spot_tuning_assumptions.md
git commit -m "Add Spot CaT rough-terrain environment"
```

---

### Task 5: Run the Short Spot PPO Startup and Validate Saved Evaluation Metadata

**Files:**
- Update: `docs/spot_tuning_assumptions.md`
- Validate: `scripts/clean_rl/train.py`
- Validate: `scripts/eval.py`

**Interfaces:**
- Consumes: `CaT-Spot-Rough-Terrain-v0`, CleanRL runner, saved `params/env.yaml`.
- Produces: a real two-iteration checkpoint/config artifact and evidence that PPO, saved action scale, saved constraints, and evaluation resolution are usable.

- [ ] **Step 1: Run exactly the bounded PPO startup**

Run with elevated GPU/Isaac execution:

```bash
python scripts/clean_rl/train.py \
  --task=CaT-Spot-Rough-Terrain-v0 \
  --seed=46 \
  --headless \
  --num_envs=64 \
  --num_iterations=2
```

Expected:

- Environment resolves as `CaT-Spot-Rough-Terrain-v0`.
- Action dimension 12 and observation dimension 188.
- Resolved energy endpoint 0.00380.
- Two rollouts and two policy updates complete.
- No shape, entity, actuator, CUDA, NaN, or Inf error.

Do not extend the run beyond two iterations.

- [ ] **Step 2: Inspect saved config and finite training scalars**

Identify the new run directory from the command output. Check:

```bash
rg -n 'CaT-Spot|scale: 0.2|limit: 80.0|limit: 20.0|limit: 800.0|limit: 2.0|end_weight: 0.0038|mass_distribution_params' <run_dir>/params/env.yaml
```

Use TensorBoard's event accumulator:

```bash
python -c '
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
path = "<run_dir>"
ea = EventAccumulator(path)
ea.Reload()
required = [
    "charts/learning_rate",
    "losses/value_loss",
    "losses/policy_loss",
]
for tag in required:
    values = [event.value for event in ea.Scalars(tag)]
    assert values and all(math.isfinite(value) for value in values), (tag, values)
print("finite PPO scalars:", {tag: ea.Scalars(tag)[-1].value for tag in required})
'
```

If actual tag names differ, list `ea.Tags()["scalars"]`, select the corresponding actor/critic/learning-rate tags, and record their exact names. Do not treat absent tags as passing.

- [ ] **Step 3: Validate constraint distributions and termination signals**

From TensorBoard tags, extract and record:

- `Episode/MaxAppliedTorque`
- `Episode/MaxJointVel`
- `Episode/MaxJointPos`
- `Episode/MaxActionRate`
- `Episode/MaxFootContactForce`
- `Episode/EnergyConsumed`
- Per-term `Episode_Constraint_violation/*`
- Per-term `Episode_Constraint_probability/*`
- Available termination/reset tags
- Terrain-level and tracking-reward tags
- Energy curriculum weight and energy reward contribution

Require all recorded values to be finite. Interpret them as random-policy startup evidence only.

- [ ] **Step 4: Validate saved bounds and Spot evaluation profile without full evaluation**

Run:

```bash
python -c '
import sys
sys.path.insert(0, "scripts")
import eval as evaluation

bounds = evaluation.load_constraint_bounds("<run_dir>/params")
assert bounds["joint_torque"] == (-80.0, 80.0)
assert bounds["joint_velocity"] == (-20.0, 20.0)
assert bounds["joint_acceleration"] == (-800.0, 800.0)
assert bounds["action_rate"] == (-80.0, 80.0)
assert bounds["foot_contact_force"] == (0.0, 800.0)
for name in ("fl_hy", "fr_hy", "hl_hy", "hr_hy"):
    assert bounds[name] == (None, 2.0)

profile = evaluation.resolve_robot_eval_profile("CaT-Spot-Rough-Terrain-Play-v0")
assert profile.name == "spot"
assert profile.joint_role_mapping["kn"]["RR"] == "hr_kn"
print("Spot saved constraint bounds and evaluation profile passed")
'
```

Expected: exit 0.

- [ ] **Step 5: Append exact PPO-startup evidence**

Append a dated subsection to `docs/spot_tuning_assumptions.md` containing:

- Run directory, commit, seed, 64 environments, and two iterations.
- Observation/action dimensions and timing.
- Finite scalar/tag names and final values.
- Constraint maxima, violation frequencies/probabilities, termination rates, terrain/tracking signals, and energy contribution.
- A statement that two iterations characterize initialization and do not prove locomotion.
- Whether the baseline remains the recommended first substantive configuration.

Use actual values only. Do not copy the old ANYmal startup numbers.

- [ ] **Step 6: Run final focused verification**

Run:

```bash
python -m py_compile \
  scripts/generate_plots.py \
  scripts/eval.py \
  scripts/metrics_utils.py \
  scripts/clean_rl/train.py \
  exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_go2_rough_terrain_env_cfg.py
pytest -q tests/test_generate_plots.py
git diff --check
git branch --show-current
git status --short
```

Expected:

- All modules compile.
- `3 passed`.
- No diff errors.
- Branch `cross-embodiment`.
- Only the intended Spot evidence document is modified.

- [ ] **Step 7: Commit startup evidence**

```bash
git add docs/spot_tuning_assumptions.md
git commit -m "Record Spot startup validation evidence"
```

- [ ] **Step 8: Verify final repository state**

Run:

```bash
git branch --show-current
git status --short
git log --oneline -6
```

Expected: branch `cross-embodiment`, empty status, and distinct commits for design, plotting, environment, and validation evidence.

## Post-implementation handoff

Do not launch the substantive run automatically. Recommend:

```bash
just train 4096 CaT-Spot-Rough-Terrain-v0 46
```

Explain that it is the first convergence test, list the exact first-24-hour signals from `docs/spot_tuning_assumptions.md`, and provide continue/stop criteria. Evaluation after a compatible checkpoint is:

```bash
just eval /absolute/path/to/run \
  --eval_checkpoint=<checkpoint> \
  --task=CaT-Spot-Rough-Terrain-Play-v0
```

Do not claim scientific transfer until that substantive run learns stable tracking, advances through the unchanged terrain curriculum, and produces valid terrain-level, CoT, and RMS metrics.
