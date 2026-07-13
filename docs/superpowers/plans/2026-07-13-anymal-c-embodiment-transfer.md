# ANYmal C Embodiment Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trainable CaT rough-terrain ANYmal C environment and embodiment-aware Go2/ANYmal C/Spot evaluation while preserving legacy Go2 behavior and metric comparability.

**Architecture:** Keep the proven Go2 MDP as the source of shared terrain, commands, observations, rewards, curricula, timing, and termination behavior. Implement ANYmal C as a narrow subclass that replaces only the asset, names, action scale, disturbances, physical constraints, and energy endpoint. Keep `eval.py` procedural, but centralize the small set of robot geometry/name facts in one profile table and pass an explicit joint-role map to the existing metric function.

**Tech Stack:** Python 3.11, Isaac Lab 2.3.1 / Isaac Sim 5.1, Gymnasium, PyTorch, NumPy, CleanRL PPO, YAML.

## Global Constraints

- Work only on branch `cross-embodiment`; check the branch before every commit and never switch branches.
- ANYmal C training and evaluation take priority. Spot receives evaluation-profile support, but no custom Spot training environment or checkpoint claim.
- Do not modify sim-to-real files or validate sim-to-real behavior.
- Preserve Go2 defaults and the formulas/schema for achieved terrain level, cost of transport, and body-frame RMS tracking error.
- Treat a run without recognized non-Go2 metadata/task identity as legacy Go2 so pre-branch checkpoints remain evaluable.
- Use the installed Isaac Lab source and runtime as authoritative; do not substitute APIs or names from another release.
- Initial ANYmal C constraints are learning-first starting assumptions: torque 60 N m, velocity 7.0 rad/s, acceleration 300 rad/s^2, normalized action rate 80 s^-1, foot force 1000 N, standstill speed 2 rad/s.
- Initial ANYmal C disturbances are reduced: direct velocity push +/-0.25 m/s, external force +/-10 N, external torque +/-0.5 N m, and added base mass +/-5 kg.
- Initial ANYmal C power endpoint is `0.00230`; keep reward and constraint curriculum timing identical to Go2.
- Do not start a multi-day training run during implementation. Use static validation, construction/reset/step smoke tests, and a very short PPO startup run.
- Make small commits that leave the branch understandable and run focused checks after every logical change.

## File Map

- Create `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py`: ANYmal-specific subclass and play config.
- Modify `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`: register ANYmal train/play task IDs.
- Create `docs/anymal_c_tuning_assumptions.md`: living fact/assumption register and evidence-driven tuning order.
- Modify `exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py`: derive diagnostic selectors from active constraint configuration instead of task-name substrings.
- Modify `scripts/clean_rl/train.py`: optional energy endpoint override and a concise resolved embodiment/config diagnostic.
- Modify `scripts/eval.py`: robot profiles, legacy Go2 resolution, embodiment-correct sensor/body geometry, profile-safe task inference and fallback bounds.
- Modify `scripts/metrics_utils.py`: accept an explicit joint-role map while preserving the existing Go2 symmetry output keys.

---

### Task 1: Register and Validate the ANYmal C CaT Environment

**Files:**
- Create: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py`
- Modify: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`
- Create: `docs/anymal_c_tuning_assumptions.md`

**Interfaces:**
- Consumes: `Go2RoughTerrainEnvCfg`, shared Go2 scene/MDP terms, installed `isaaclab_assets.robots.anymal.ANYMAL_C_CFG`.
- Produces: `ANYMAL_C_JOINT_NAMES`, `AnymalCRoughTerrainEnvCfg`, `AnymalCRoughTerrainEnvCfg_PLAY`, `CaT-Anymal-C-Rough-Terrain-v0`, and `CaT-Anymal-C-Rough-Terrain-Play-v0`.

- [ ] **Step 1: Run a registration check that demonstrates the new task is absent**

Run after launching Isaac Sim headless:

```python
import gymnasium as gym
import cat_envs.tasks.locomotion.velocity.config.solo12
assert "CaT-Anymal-C-Rough-Terrain-v0" in gym.envs.registry
```

Expected before implementation: `AssertionError`.

- [ ] **Step 2: Create the narrow ANYmal C training subclass**

Use the installed asset and preserve the action/observation order by defining one canonical list:

```python
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from .cat_go2_rough_terrain_env_cfg import Go2RoughTerrainEnvCfg

ANYMAL_C_JOINT_NAMES = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
]
ANYMAL_C_JOINT_PATTERNS = [".*HAA", ".*HFE", ".*KFE"]
ANYMAL_C_FEET = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]

@configclass
class AnymalCRoughTerrainEnvCfg(Go2RoughTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = list(ANYMAL_C_JOINT_NAMES)
        self.actions.joint_pos.scale = 0.5
        self.observations.policy.joint_pos_history.params["names"] = list(ANYMAL_C_JOINT_NAMES)
        self.observations.policy.joint_vel_history.params["names"] = list(ANYMAL_C_JOINT_NAMES)

        for term_name in ("randomize_com", "randomize_mass", "push_base_wrench"):
            getattr(self.events, term_name).params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base"])
        self.events.randomize_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}
        self.events.push_base_wrench.params["force_range"] = (-10.0, 10.0)
        self.events.push_base_wrench.params["torque_range"] = (-0.5, 0.5)

        for term_name in ("joint_torque", "joint_velocity", "joint_acceleration", "action_rate"):
            getattr(self.constraints, term_name).params["names"] = list(ANYMAL_C_JOINT_PATTERNS)
        self.constraints.joint_torque.params["limit"] = 60.0
        self.constraints.joint_velocity.params["limit"] = 7.0
        self.constraints.joint_acceleration.params["limit"] = 300.0
        self.constraints.action_rate.params["limit"] = 80.0
        self.constraints.contact.params["names"] = ["base", ".*_THIGH"]
        self.constraints.foot_contact_force.params.update(limit=1000.0, names=[".*_FOOT"])
        self.constraints.front_hfe_position.params.update(limit=1.5, names=[".*HFE"])
        self.constraints.hip_position.params["names"] = [".*HAA"]
        self.constraints.no_move.params.update(names=list(ANYMAL_C_JOINT_PATTERNS), joint_vel_limit=2.0)
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["base", ".*_THIGH"]
        )
        self.curriculum.power.params["end_weight"] = 0.00230
```

Also update every base-body selector used by the inherited height/contact sensors if runtime inspection shows it is not already `base`; keep shared terrain, commands, observation scales/noise, rewards, timing, reset-root logic, and curricula unchanged.

- [ ] **Step 3: Implement the ANYmal C play subclass without inheriting Go2-only eval pose behavior**

```python
@configclass
class AnymalCRoughTerrainEnvCfg_PLAY(AnymalCRoughTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8
        self.apply_elevation_map_point_noise = False
        self.observations.policy.enable_corruption = True
        self.events.push_robot = None
        self.events.push_base_wrench = None
        self.events.force_hard_terrain = EventTerm(func=force_hard_terrain, mode="startup")
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
        )
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.6)
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.rewards.minimize_power = None
        self.curriculum.power = None
```

Add four one-ray foot sensors for `LF_FOOT`, `RF_FOOT`, `LH_FOOT`, `RH_FOOT` and a frame transformer sourced from `base`, matching the existing Go2 play sensor configuration apart from link names.

- [ ] **Step 4: Register the train/play tasks**

Append registrations with the existing `CaTEnv` and CleanRL runner:

```python
gym.register(
    id="CaT-Anymal-C-Rough-Terrain-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_anymal_c_rough_terrain_env_cfg:AnymalCRoughTerrainEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)
gym.register(
    id="CaT-Anymal-C-Rough-Terrain-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_anymal_c_rough_terrain_env_cfg:AnymalCRoughTerrainEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)
```

- [ ] **Step 5: Write the living tuning-assumption register**

Record two tables. The verified-facts table includes installed version, mass 52.13484 kg, base mass 26.373 kg, joint/body/foot names, 0.6 m spawn height, 80 N m continuous effort, 120 N m saturation effort, 7.5 rad/s no-load speed, and the ANYdrive LSTM actuator. The starting-assumptions table includes action scale, every constraint, energy endpoint/bracket, disturbance, reset range, contact termination, toe offset, observation/reward terms, terrain settings, rationale, early failure signal, and next candidate. Include these tuning sequences:

```text
torque: 60 -> 50 -> 40 N m only after locomotion exists
velocity: 7.0 -> approximately 6.25 rad/s
acceleration: 300 -> 200 rad/s^2
standstill speed: 2 -> 1 rad/s
energy endpoint: 0.00230, bracket with 0.00115 and 0.00461
disturbances: keep reduced until stable learning, then increase one family at a time
```

- [ ] **Step 6: Run config construction and simulator smoke checks**

Run headless with one environment and assert:

```python
assert env.action_space.shape[-1] == 12
assert obs["policy"].shape[-1] == 188
assert env.unwrapped.step_dt == 0.02
assert env.unwrapped.scene["robot"].data.joint_names == ANYMAL_C_JOINT_NAMES
assert abs(float(env.unwrapped.scene["robot"].data.default_mass.sum()) - 52.13484) < 0.01
```

Reset and step both train and play configurations for at least 25 zero/random action steps. Expected: all observations, rewards, constraint probabilities, applied torques, velocities, and accelerations are finite; entity selectors resolve to 12 joints and four feet; neither configuration throws a missing-body/joint error.

- [ ] **Step 7: Re-run Go2 construction and commit**

Construct and step `CaT-Go2-Rough-Terrain-v0` and `CaT-Go2-Rough-Terrain-Play-v0` with one environment. Expected: 12 actions, 188 observations, 0.8 action scale, and unchanged Go2 selectors/bounds.

```bash
git add docs/anymal_c_tuning_assumptions.md \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py
git commit -m "Add ANYmal C CaT rough-terrain environment"
```

---

### Task 2: Make Training Diagnostics Embodiment-Aware

**Files:**
- Modify: `exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py`
- Modify: `scripts/clean_rl/train.py`

**Interfaces:**
- Consumes: active `joint_torque` and `foot_contact_force` constraint term parameters.
- Produces: task-name-independent episode diagnostics and `--energy_end_weight FLOAT`.

- [ ] **Step 1: Demonstrate the current diagnostic selector is task-name-dependent**

Run:

```bash
rg -n '"Go2" in self\._env\.spec\.id' exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py
```

Expected before implementation: one match in `compute()`.

- [ ] **Step 2: Derive diagnostic joint/foot selectors from configured constraints**

Replace task-string selection with:

```python
term_cfgs_by_name = dict(zip(self._term_names, self._term_cfgs))
joint_names = term_cfgs_by_name["joint_torque"].params["names"]
foot_names = term_cfgs_by_name["foot_contact_force"].params["names"]
joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
feet_ids, _ = contact_sensor.find_bodies(foot_names, preserve_order=True)
```

Fail during manager construction with a descriptive error if either required diagnostic term is absent. Do not alter how max torque, action rate, velocity, force, air time, position, or integrated absolute joint power are calculated.

- [ ] **Step 3: Add an optional training energy endpoint override**

Add the parser option and apply it before configuration dumping/environment construction:

```python
parser.add_argument(
    "--energy_end_weight",
    type=float,
    default=None,
    help="Override curriculum.power.params['end_weight']; defaults to the task config.",
)

if args_cli.energy_end_weight is not None:
    if not hasattr(env_cfg, "curriculum") or env_cfg.curriculum.power is None:
        raise ValueError("--energy_end_weight requires an active curriculum.power term")
    if args_cli.energy_end_weight < 0.0:
        raise ValueError("--energy_end_weight must be non-negative")
    env_cfg.curriculum.power.params["end_weight"] = args_cli.energy_end_weight
```

Print task, asset USD, ordered action joints, action scale, power endpoint, constraint limits/selectors, disturbance ranges, step time, and episode duration. The resolved values must be dumped to `params/env.yaml` exactly as trained.

- [ ] **Step 4: Run focused static/config checks**

```bash
python -m compileall scripts/clean_rl/train.py exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py
rg -n '"Go2" in self\._env\.spec\.id' exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py
```

Expected: compilation succeeds and the second command returns no matches. Construct both robot tasks and check diagnostics resolve 12 joints and four feet.

- [ ] **Step 5: Commit**

```bash
git add scripts/clean_rl/train.py exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py
git commit -m "Make CaT training diagnostics embodiment-aware"
```

---

### Task 3: Add Backward-Compatible Evaluation Profiles

**Files:**
- Modify: `scripts/eval.py`
- Modify: `scripts/metrics_utils.py`

**Interfaces:**
- Consumes: explicit task name, saved `params/env.yaml`, runtime robot data.
- Produces: `RobotEvalProfile`, `resolve_robot_eval_profile()`, embodiment-correct foot/base geometry, and `constants["joint_role_mapping"]` for metric computation.

- [ ] **Step 1: Add a pure profile resolution check that initially fails**

The check must cover all recognized and legacy cases:

```python
assert resolve_robot_eval_profile("CaT-Go2-Rough-Terrain-Play-v0").name == "go2"
assert resolve_robot_eval_profile("CaT-Anymal-C-Rough-Terrain-Play-v0").name == "anymal_c"
assert resolve_robot_eval_profile("Isaac-Velocity-Flat-Spot-Play-v0").name == "spot"
assert resolve_robot_eval_profile("unrecognized-old-task").name == "go2"
```

Expected before implementation: `NameError`.

- [ ] **Step 2: Add the small robot profile table and resolver**

Define immutable profiles near the top of `eval.py`:

```python
@dataclass(frozen=True)
class RobotEvalProfile:
    name: str
    root_link: str
    foot_links: tuple[str, str, str, str]
    spawn_height: float
    sole_offset: float
    joint_role_mapping: dict[str, dict[str, str]]

ROBOT_EVAL_PROFILES = {
    "go2": RobotEvalProfile(
        "go2", "base", ("FL_foot", "FR_foot", "RL_foot", "RR_foot"), 0.4, 0.0228,
        {"hip_joint": {"FL": "FL_hip_joint", "FR": "FR_hip_joint", "RL": "RL_hip_joint", "RR": "RR_hip_joint"},
         "thigh_joint": {"FL": "FL_thigh_joint", "FR": "FR_thigh_joint", "RL": "RL_thigh_joint", "RR": "RR_thigh_joint"},
         "calf_joint": {"FL": "FL_calf_joint", "FR": "FR_calf_joint", "RL": "RL_calf_joint", "RR": "RR_calf_joint"}},
    ),
    "anymal_c": RobotEvalProfile(
        "anymal_c", "base", ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"), 0.6, 0.0,
        {"HAA": {"FL": "LF_HAA", "FR": "RF_HAA", "RL": "LH_HAA", "RR": "RH_HAA"},
         "HFE": {"FL": "LF_HFE", "FR": "RF_HFE", "RL": "LH_HFE", "RR": "RH_HFE"},
         "KFE": {"FL": "LF_KFE", "FR": "RF_KFE", "RL": "LH_KFE", "RR": "RH_KFE"}},
    ),
    "spot": RobotEvalProfile(
        "spot", "body", ("fl_foot", "fr_foot", "hl_foot", "hr_foot"), 0.5, 0.0,
        {"hx": {"FL": "fl_hx", "FR": "fr_hx", "RL": "hl_hx", "RR": "hr_hx"},
         "hy": {"FL": "fl_hy", "FR": "fr_hy", "RL": "hl_hy", "RR": "hr_hy"},
         "kn": {"FL": "fl_kn", "FR": "fr_kn", "RL": "hl_kn", "RR": "hr_kn"}},
    ),
}
```

Resolve `anymal` and `spot` first, resolve explicit `go2` second, and otherwise warn and return Go2. If saved future metadata is added, explicit recognized metadata outranks task text; missing metadata remains Go2.

- [ ] **Step 3: Restrict CleanRL observation-dimension task inference to legacy Go2**

Resolve the requested profile before checkpoint dimension inference. Apply the 236/558 Go2 history-task substitutions only when the resolved profile is Go2. Explicit ANYmal or Spot tasks must remain unchanged even if their input dimension overlaps Go2.

- [ ] **Step 4: Apply the profile to sensors and fixed scenarios**

Change the helper signature:

```python
def add_eval_foot_sensors_to_env_cfg(
    env_cfg, foot_links: tuple[str, str, str, str], root_link: str, sim_dt: float
):
    ...
    env_cfg.scene.foot_frame_transformer = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{root_link}",
        target_frames=[...],
        debug_vis=False,
    )
```

Use `profile.foot_links`, `profile.root_link`, and `profile.sole_offset`. Generate fixed scenario positions through a helper that replaces only the z coordinate with `profile.spawn_height`; Go2 must still produce the exact existing tensors, scenario names, commands, orientations, step counts, random segment, terrain generator, and seed behavior.

- [ ] **Step 5: Make fallback constraint loading embodiment-safe**

Restore the intended guard:

```python
if not constraint_bounds and is_upstream_go2_rough_task(task_name):
    constraint_bounds = get_hardcoded_upstream_go2_constraint_bounds()
    constraint_bounds_source = "hardcoded_upstream_go2_rough_eval_thresholds"
elif not constraint_bounds:
    constraint_bounds_source = "empty"
```

When restoring a custom hard position bound, derive matching saved joint entries from the configured term's patterns instead of checking only `RL_thigh_joint`. ANYmal loads its saved bounds; Spot receives no Go2 thresholds.

- [ ] **Step 6: Generalize gait symmetry without changing Go2 output**

Read the map with a backward-compatible default:

```python
joint_role_mapping = constants.get("joint_role_mapping", {
    "hip_joint": {leg: f"{leg}_hip_joint" for leg in ("FL", "FR", "RL", "RR")},
    "thigh_joint": {leg: f"{leg}_thigh_joint" for leg in ("FL", "FR", "RL", "RR")},
    "calf_joint": {leg: f"{leg}_calf_joint" for leg in ("FL", "FR", "RL", "RR")},
})
for dof, role_names in joint_role_mapping.items():
    concatenated_joint_positions = np.concatenate(
        [joint_positions[:, joint_mapping[role_names[leg]]] for leg in ("FL", "FR", "RL", "RR")]
    )
```

Keep comparison keys `front_left_front_right`, `rear_left_rear_right`, `front_left_rear_left`, `front_right_rear_right`, and both diagonal comparisons unchanged. Pass `profile.joint_role_mapping` in `constants_dict`.

- [ ] **Step 7: Preserve and annotate the primary metrics**

Do not modify:

```python
cost_of_transport = combined_energy[-1] / (total_robot_mass * 9.81 * distance_walked_horizontal)
linear_vel_x_rms = np.sqrt(np.mean((commanded_velocity[:, 0] - base_linear_velocity_body[:, 0]) ** 2))
linear_vel_y_rms = np.sqrt(np.mean((commanded_velocity[:, 1] - base_linear_velocity_body[:, 1]) ** 2))
yaw_rms = np.sqrt(np.mean((commanded_velocity[:, 2] - base_angular_velocity_body[:, 2]) ** 2))
```

Add `robot_profile` to saved NPZ/JSON metadata. Keep `terrain_level_summary`, `mean_terrain_level`, and `final_terrain_level` calculated from the same runtime terrain-level series; identify them as evaluation terrain levels rather than renaming them or changing semantics.

- [ ] **Step 8: Run pure/static regression checks and simulator construction checks**

```bash
python -m compileall scripts/eval.py scripts/metrics_utils.py
git diff --check
```

Run profile assertions from Step 1. Run `compute_summary_metrics` on deterministic synthetic Go2 arrays once with the mapping omitted and once with the explicit Go2 mapping; assert the returned dictionaries are equal. Construct evaluation environments/sensors for Go2, ANYmal C, and installed upstream Spot, then reset/step each and assert four feet, 12 joints, finite transforms/contact force arrays, and the correct root link.

- [ ] **Step 9: Commit**

```bash
git add scripts/eval.py scripts/metrics_utils.py
git commit -m "Support cross-embodiment locomotion evaluation"
```

---

### Task 4: Short End-to-End Startup and Regression Review

**Files:**
- Modify if evidence requires: files from Tasks 1-3
- Modify: `docs/anymal_c_tuning_assumptions.md` only when a verified runtime fact or starting value changes

**Interfaces:**
- Consumes: registered ANYmal environment, training CLI, evaluation profiles.
- Produces: startup evidence and a reviewed branch ready for the first user-run training experiment.

- [ ] **Step 1: Run a short ANYmal CleanRL startup**

Use a separate diagnostic experiment name and no video, with enough environments to exercise resets but only a few PPO iterations:

```bash
ENV_NAME=anymal_c_startup python scripts/clean_rl/train.py \
  --task CaT-Anymal-C-Rough-Terrain-v0 \
  --num_envs 64 \
  --num_iterations 2 \
  --headless
```

Expected: terrain construction, config dump, environment initialization, rollout/update, episode diagnostics when resets occur, and clean shutdown without NaN/Inf, missing selector, shape mismatch, or CUDA error. Do not interpret two iterations as convergence evidence.

- [ ] **Step 2: Verify the dumped config is the actual intended run config**

Inspect the generated `params/env.yaml`. Expected exact values: ANYmal USD, 12 ordered ANYmal joints, scale 0.5, 60/7/300/80/1000/2 constraint values, +/-0.25 velocity pushes, +/-10 N and +/-0.5 N m wrench, +/-5 kg base mass, and 0.00230 energy endpoint.

- [ ] **Step 3: Review Go2 source/default regression and baseline reference**

Confirm the Go2 config has no diff and profile resolution remains Go2 for default/unknown legacy tasks. Read the existing baseline:

```text
/local_storage1/loukas-backup/logs/clean_rl/cluster_runs/env_id_107_action_scale_sweep/constraints-as-terminations/logs/clean_rl/env_id_107_action_scale_sweep/2026-07-03-09-25-54/eval_checkpoint_21799_seed_46/metrics_summary.json
```

Record the comparison targets without launching a new 45-minute default evaluation during development: action scale 0.8, CoT `0.4196298480495525`, x RMS `0.32775482535362244`, y RMS `0.1588231772184372`, yaw RMS `0.09639634191989899`, and mean evaluation terrain level `3.0406896551724136`. A later matched checkpoint regression must use the same task, checkpoint, seed 46, random step count 4000, and CoT sweep.

- [ ] **Step 4: Review all changes for unsafe assumptions**

```bash
git branch --show-current
git status --short
git diff --check
git diff HEAD~3 --stat
rg -n "Go2|go2|FL_foot|RL_thigh_joint|Robot/base|0\.4" scripts/eval.py scripts/metrics_utils.py exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py
python -m compileall scripts exts/cat_envs/cat_envs/tasks
```

Expected: branch is `cross-embodiment`; every remaining Go2 occurrence is either an explicit Go2 profile/baseline behavior or a documented compatibility path; no ANYmal/Spot path depends on a Go2 body/joint; compilation and diff checks pass.

- [ ] **Step 5: Commit evidence-driven corrections**

Only if validation required source/doc corrections:

```bash
git add docs/anymal_c_tuning_assumptions.md scripts/clean_rl/train.py scripts/eval.py scripts/metrics_utils.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py \
  exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py
git commit -m "Validate cross-embodiment training and evaluation startup"
```

- [ ] **Step 6: Hand off one first training run with explicit 24-hour gates**

Recommend exactly:

```bash
just train 4096 CaT-Anymal-C-Rough-Terrain-v0 46
```

Inspect finite losses/rewards, episode length and termination reasons, tracking reward/error proxies, terrain-level distribution/progression, per-constraint violation frequency/probability, maximum torque/velocity/acceleration/action rate/contact force, episode energy, applied-power reward contribution, action standard deviation/saturation, and videos during the first 24 hours. Continue only if locomotion/tracking improve, terrain is not pinned at zero, episodes stop being dominated by falls/hard constraints, no soft constraint permanently dominates termination probability, energy pressure remains secondary to learning locomotion, and all signals stay finite. Stop for NaN/Inf, immediate repeated resets, terrain pinned at zero with flat tracking progress, persistent action saturation/collapse, pervasive hard-contact/force termination, or an energy/soft-constraint term that prevents tracking improvement. Use the tuning register's one-change-at-a-time candidates before another long run.
