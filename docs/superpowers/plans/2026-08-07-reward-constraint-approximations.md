# Reward Constraint Approximations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the five paper-defined soft operational CaT terms in the existing Go2 rough-terrain task with normalized, 800-iteration-curriculum reward penalties while preserving hard CaT and every other LEP setting.

**Architecture:** Five named reward wrappers reuse the existing signed constraint functions and pass their tensors through one normalization/curriculum helper. The existing environment config directly activates those rewards and deactivates the soft/legacy CaT terms without registering another task. Evaluation learns the transferred thresholds from saved reward configuration as well as constraint configuration.

**Tech Stack:** Python 3.11, PyTorch 2.7, Isaac Lab 2.x manager-based environments, Gymnasium, CleanRL PPO, pytest 9, YAML.

## Global Constraints

- Work in the existing `reward-constraint-approximations` checkout; do not create a worktree or another Gym environment.
- Keep `CaT-Go2-Rough-Terrain-v0`, action scale `0.8`, PPO settings, tracking rewards, power reward/curriculum, terrain, observations, commands, events, and randomization unchanged.
- Transfer exactly joint torque `20.0`, joint velocity `25.0`, joint acceleration `800.0`, action rate `80.0`, and base orientation `0.1`.
- Use `max(relu(c) / limit)` across joint components and no upper clipping.
- Ramp effective weights with `env.common_step_counter / 19_200`, saturating at one after 800 PPO iterations of 24 steps.
- Define low `0.1` and high `10.0` constants; select the active profile by changing one assignment, with low active initially.
- Disable `hip_position` and `no_move` instead of translating them.
- Retain the four hard `max_p=1` CaT terms and existing nonnegative total-reward clipping exactly.
- Add no generalized experiment framework or unrelated refactor.

---

### Task 1: Normalized Operational-Limit Reward Functions

**Files:**
- Modify: `exts/cat_envs/cat_envs/tasks/utils/mdp/rewards.py`
- Create: `tests/test_reward_constraint_approximations.py`

**Interfaces:**
- Consumes: existing `cat_envs.tasks.utils.cat.constraints` functions with their current signatures.
- Produces: `_normalized_constraint_penalty(env, constraint_violation, limit, curriculum_steps) -> torch.Tensor` and the five wrappers `joint_torque_limit_penalty`, `joint_velocity_limit_penalty`, `joint_acceleration_limit_penalty`, `action_rate_limit_penalty`, and `base_orientation_limit_penalty`.

- [ ] **Step 1: Write failing tests for normalization, aggregation, validation, and schedule progress**

Load `rewards.py` with lightweight Isaac Lab/package stubs so the numerical tests do not require starting Kit. Cover scalar and two-dimensional violations:

```python
class FakeEnv:
    def __init__(self, common_step_counter: int):
        self.common_step_counter = common_step_counter


def test_normalized_constraint_penalty_uses_hinge_max_and_half_progress(rewards_module):
    env = FakeEnv(common_step_counter=9_600)
    violations = torch.tensor([[-1.0, 0.0, 2.0], [4.0, 1.0, -3.0]])

    actual = rewards_module._normalized_constraint_penalty(
        env, violations, limit=2.0, curriculum_steps=19_200
    )

    torch.testing.assert_close(actual, torch.tensor([-0.5, -1.0]))


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.0), (9_600, -0.5), (19_200, -1.0), (25_000, -1.0)],
)
def test_normalized_constraint_penalty_curriculum_endpoints(rewards_module, step, expected):
    actual = rewards_module._normalized_constraint_penalty(
        FakeEnv(step), torch.tensor([2.0]), limit=2.0, curriculum_steps=19_200
    )
    torch.testing.assert_close(actual, torch.tensor([expected]))


@pytest.mark.parametrize(
    ("limit", "curriculum_steps", "message"),
    [(0.0, 19_200, "limit must be positive"), (1.0, 0, "curriculum_steps must be positive")],
)
def test_normalized_constraint_penalty_rejects_invalid_configuration(
    rewards_module, limit, curriculum_steps, message
):
    with pytest.raises(ValueError, match=message):
        rewards_module._normalized_constraint_penalty(
            FakeEnv(0), torch.tensor([1.0]), limit, curriculum_steps
        )
```

Also assert a tensor with rank other than one or two raises a clear `ValueError`.

- [ ] **Step 2: Run the focused tests and verify they fail for the missing helper**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest \
  tests/test_reward_constraint_approximations.py -q
```

Expected: FAIL because `_normalized_constraint_penalty` and the five wrappers do not exist.

- [ ] **Step 3: Implement the minimal shared helper**

Add the constraint import and helper to `rewards.py`:

```python
import cat_envs.tasks.utils.cat.constraints as constraints


def _normalized_constraint_penalty(
    env: ManagerBasedRLEnv,
    constraint_violation: torch.Tensor,
    limit: float,
    curriculum_steps: int,
) -> torch.Tensor:
    if limit <= 0.0:
        raise ValueError(f"limit must be positive, got {limit}")
    if curriculum_steps <= 0:
        raise ValueError(f"curriculum_steps must be positive, got {curriculum_steps}")
    if constraint_violation.ndim not in (1, 2):
        raise ValueError(
            "constraint_violation must have shape (num_envs,) or (num_envs, num_components), "
            f"got {tuple(constraint_violation.shape)}"
        )

    normalized_excess = torch.clamp_min(constraint_violation, 0.0) / limit
    if normalized_excess.ndim == 2:
        normalized_excess = normalized_excess.max(dim=1).values

    progress = min(max(env.common_step_counter / curriculum_steps, 0.0), 1.0)
    return -normalized_excess * progress
```

- [ ] **Step 4: Add failing delegation tests for every physical quantity**

Monkeypatch each stubbed constraint function to capture `limit`, `names`, and `asset_cfg`, return a known signed-violation tensor, then call its reward wrapper. For example:

```python
def test_joint_torque_penalty_delegates_to_constraint_quantity(rewards_module, monkeypatch):
    received = {}

    def fake_joint_torque(env, limit, names, asset_cfg):
        received.update(limit=limit, names=names, asset_cfg=asset_cfg)
        return torch.tensor([[0.0, 10.0]])

    monkeypatch.setattr(rewards_module.constraints, "joint_torque", fake_joint_torque)
    names = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
    actual = rewards_module.joint_torque_limit_penalty(
        FakeEnv(19_200), limit=20.0, names=names, curriculum_steps=19_200
    )

    torch.testing.assert_close(actual, torch.tensor([-0.5]))
    assert received["limit"] == 20.0
    assert received["names"] == names
```

Use one parametrized test with these complete cases so every wrapper is covered:

```python
joint_names = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
cases = [
    ("joint_torque_limit_penalty", "joint_torque", 20.0, joint_names, torch.tensor([[0.0, 10.0]]), -0.5),
    ("joint_velocity_limit_penalty", "joint_velocity", 25.0, joint_names, torch.tensor([[0.0, 12.5]]), -0.5),
    (
        "joint_acceleration_limit_penalty",
        "joint_acceleration",
        800.0,
        joint_names,
        torch.tensor([[0.0, 400.0]]),
        -0.5,
    ),
    ("action_rate_limit_penalty", "action_rate", 80.0, joint_names, torch.tensor([[0.0, 40.0]]), -0.5),
    ("base_orientation_limit_penalty", "base_orientation", 0.1, None, torch.tensor([0.05]), -0.5),
]
```

For each case, monkeypatch `rewards_module.constraints.<constraint_name>` with a `*args, **kwargs` capture function, call the wrapper at step 19,200, assert the raw tensor is reduced to the listed expected value, and assert `limit`, `names` when present, and default `asset_cfg` were forwarded.

- [ ] **Step 5: Run the wrapper tests and verify they fail**

Run the same focused pytest command. Expected: helper tests PASS and delegation tests FAIL because wrappers are missing.

- [ ] **Step 6: Implement all five wrappers by calling existing constraint functions**

Use signatures matching the config parameters:

```python
def joint_torque_limit_penalty(env, limit, names, curriculum_steps, asset_cfg=SceneEntityCfg("robot")):
    violation = constraints.joint_torque(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)
```

Implement these exact signatures:

```python
def joint_velocity_limit_penalty(env, limit, names, curriculum_steps, asset_cfg=SceneEntityCfg("robot")):
    violation = constraints.joint_velocity(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def joint_acceleration_limit_penalty(env, limit, names, curriculum_steps, asset_cfg=SceneEntityCfg("robot")):
    violation = constraints.joint_acceleration(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def action_rate_limit_penalty(env, limit, names, curriculum_steps, asset_cfg=SceneEntityCfg("robot")):
    violation = constraints.action_rate(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def base_orientation_limit_penalty(env, limit, curriculum_steps, asset_cfg=SceneEntityCfg("robot")):
    violation = constraints.base_orientation(env, limit=limit, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)
```

- [ ] **Step 7: Run the focused tests and confirm they pass**

Run the focused pytest command. Expected: all numerical, validation, and delegation tests PASS.

- [ ] **Step 8: Commit the reward implementation**

```bash
git add tests/test_reward_constraint_approximations.py \
  exts/cat_envs/cat_envs/tasks/utils/mdp/rewards.py
git commit -m "Add normalized operational limit rewards"
```

---

### Task 2: Transfer the Existing Go2 Task from Soft CaT to Rewards

**Files:**
- Modify: `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_go2_rough_terrain_env_cfg.py`
- Modify: `tests/test_reward_constraint_approximations.py`

**Interfaces:**
- Consumes: the five Task 1 wrapper functions and exact existing constraint selectors/thresholds.
- Produces: one directly modified `CaT-Go2-Rough-Terrain-v0` configuration with five reward penalties and exactly four retained hard CaT terms.

- [ ] **Step 1: Add failing AST/config-source tests for experiment invariants**

Parse the target config with `ast` so these checks run without Kit. Assert:

```python
def test_reward_constraint_profile_constants(config_ast):
    assert constant_value(config_ast, "SOFT_CONSTRAINT_REWARD_END_WEIGHT_LOW") == 0.1
    assert constant_value(config_ast, "SOFT_CONSTRAINT_REWARD_END_WEIGHT_HIGH") == 10.0
    assert assigned_name(config_ast, "SOFT_CONSTRAINT_REWARD_END_WEIGHT") == (
        "SOFT_CONSTRAINT_REWARD_END_WEIGHT_LOW"
    )
    assert constant_value(config_ast, "SOFT_CONSTRAINT_REWARD_CURRICULUM_STEPS") == 19_200


def test_only_hard_cat_constraints_remain(config_class_assignments):
    assert active_manager_terms("ConstraintsCfg", "ConstraintTerm") == {
        "contact", "foot_contact_force", "front_hfe_position", "upsidedown"
    }


def test_only_power_and_terrain_curricula_remain(config_class_assignments):
    assert active_manager_terms("CurriculumCfg", "CurrTerm") == {"power", "terrain_levels"}
```

Assert the five reward term names, wrapper functions, limits, selectors, selected weight constant, and `19_200` curriculum steps. Also assert tracking weights `1.0/0.5`, power end weight `0.008`, and action scale `0.8` remain unchanged.

- [ ] **Step 2: Run the config tests and verify they fail against the CaT configuration**

Run the focused test file. Expected: FAIL because the profile constants/reward terms are absent and soft CaT terms are still active.

- [ ] **Step 3: Add the low/high/selected constants and five reward terms**

Immediately before `RewardsCfg`, define:

```python
SOFT_CONSTRAINT_REWARD_END_WEIGHT_LOW = 0.1
SOFT_CONSTRAINT_REWARD_END_WEIGHT_HIGH = 10.0
SOFT_CONSTRAINT_REWARD_END_WEIGHT = SOFT_CONSTRAINT_REWARD_END_WEIGHT_LOW
SOFT_CONSTRAINT_REWARD_CURRICULUM_STEPS = 24 * 800
```

Add reward terms using their original names. The torque term is representative:

```python
joint_torque = RewTerm(
    func=rewards.joint_torque_limit_penalty,
    weight=SOFT_CONSTRAINT_REWARD_END_WEIGHT,
    params={
        "limit": 20.0,
        "names": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        "curriculum_steps": SOFT_CONSTRAINT_REWARD_CURRICULUM_STEPS,
    },
)
```

Use limits `25.0`, `800.0`, `80.0`, and `0.1` for the remaining terms and preserve their current selectors.

- [ ] **Step 4: Deactivate soft and legacy CaT terms and curricula**

Comment every line of the five transferred `ConstraintTerm` assignments plus `hip_position` and `no_move`, leaving only these active assignments in `ConstraintsCfg`:

```python
contact = ConstraintTerm(...)
foot_contact_force = ConstraintTerm(...)
front_hfe_position = ConstraintTerm(...)
upsidedown = ConstraintTerm(...)
```

Comment every line of the six `CurrTerm` assignments for joint torque, velocity, acceleration, action rate, hip position, and base orientation, leaving only these active assignments in `CurriculumCfg`:

```python
power = CurrTerm(
    func=curriculums.update_reward_weight_linear,
    params={
        "term_name": "minimize_power",
        "num_steps_from_start_step": 300000,
        "start_at_step": 0,
        "start_weight": 0.0,
        "end_weight": 0.4 * 0.02,
    },
)
terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```

- [ ] **Step 5: Run focused tests and static checks**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest \
  tests/test_reward_constraint_approximations.py -q
python3 -m compileall -q \
  exts/cat_envs/cat_envs/tasks/utils/mdp/rewards.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_go2_rough_terrain_env_cfg.py
git diff --check
```

Expected: PASS with no syntax or whitespace errors.

- [ ] **Step 6: Commit the controlled config transfer**

```bash
git add tests/test_reward_constraint_approximations.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_go2_rough_terrain_env_cfg.py
git commit -m "Replace soft CaT limits with reward penalties"
```

---

### Task 3: Preserve Evaluation Threshold Loading

**Files:**
- Modify: `scripts/eval.py`
- Modify: `tests/test_reward_constraint_approximations.py`

**Interfaces:**
- Consumes: serialized `env.yaml` with hard limits under `constraints` and transferred limits under `rewards`.
- Produces: `load_constraint_bounds(params_directory) -> dict[str, tuple[float | None, float | None]]` that merges supported bounds from both sections.

- [ ] **Step 1: Add a failing isolated parser test**

Extract `load_constraint_bounds` from `scripts/eval.py` with `ast`, compile it into a namespace containing `os`, `re`, `yaml`, and typing names, and give it a temporary environment YAML:

```python
config = {
    "actions": {"joint_pos": {"joint_names": ["FL_hip_joint", "FL_thigh_joint"]}},
    "scene": {"robot": {"init_state": {"joint_pos": {".*": 0.0}}}},
    "constraints": {
        "foot_contact_force": {
            "func": "cat_envs.tasks.utils.cat.constraints:foot_contact_force",
            "params": {"limit": 300.0, "names": [".*_foot"]},
        },
    },
    "rewards": {
        "joint_torque": {
            "func": "cat_envs.tasks.utils.mdp.rewards:joint_torque_limit_penalty",
            "params": {"limit": 20.0, "names": [".*_joint"], "curriculum_steps": 19_200},
            "weight": 0.1,
        },
        "base_orientation": {
            "func": "cat_envs.tasks.utils.mdp.rewards:base_orientation_limit_penalty",
            "params": {"limit": 0.1, "curriculum_steps": 19_200},
            "weight": 0.1,
        },
    },
}
```

Assert the result includes torque `(-20.0, 20.0)`, base orientation `(-0.1, 0.1)`, and foot contact force `(0.0, 300.0)`.

- [ ] **Step 2: Run the parser test and verify it fails**

Expected: transferred reward bounds are absent because the current parser only walks `constraints`.

- [ ] **Step 3: Update `load_constraint_bounds` to merge constraints and rewards**

Replace construction of `raw_constraints` with a merged `raw_terms` dictionary and keep the existing parsing loop over that dictionary:

```python
raw_terms = {}
for section_name in ("constraints", "rewards"):
    section_terms = cfg.get(section_name, {}) or {}
    if isinstance(section_terms, dict):
        raw_terms.update(section_terms)

for term, term_cfg in raw_terms.items():
    if not isinstance(term_cfg, dict):
        continue
    func = term_cfg.get("func", "")
    params = term_cfg.get("params", {})
    if "limit" not in params:
        continue

    limit = float(params["limit"])
    patterns = params.get("names", [])

    def expand_patterns(pats):
        out = set()
        for pat in pats:
            rx = re.compile(f"^{pat}$")
            for joint_name in joint_names:
                if rx.match(joint_name):
                    out.add(joint_name)
        return sorted(out)

    if func.endswith("joint_position_absolute_upper_bound"):
        for joint_name in expand_patterns(patterns):
            bounds[joint_name] = (None, limit)
    elif func.endswith("relative_joint_position_upper_and_lower_bound_when_moving_forward"):
        for joint_name in expand_patterns(patterns):
            base = default_pos.get(joint_name, 0.0)
            bounds[joint_name] = (base - limit, base + limit)
    elif term == "foot_contact_force":
        bounds[term] = (0.0, limit)
    else:
        bounds[term] = (-limit, limit)
```

Keep existing joint-position, relative-position, and foot-force cases. Reward terms use the original term names, so the symmetric-limit fallback produces the required operational keys.

- [ ] **Step 4: Run focused tests and verify all pass**

Run the focused pytest file and `git diff --check`. Expected: PASS.

- [ ] **Step 5: Commit evaluation compatibility**

```bash
git add scripts/eval.py tests/test_reward_constraint_approximations.py
git commit -m "Load evaluation limits from reward terms"
```

---

### Task 4: Runtime Verification and Experiment Handoff

**Files:**
- Verify: all modified files
- Inspect: generated `logs/clean_rl/reward_constraint_approximation_smoke/*/params/env.yaml`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: evidence that the existing task registers, manager terms are correct, PPO runs, and the serialized config records the selected low profile.

- [ ] **Step 1: Run the full focused test suite and compile checks from a clean process**

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest \
  tests/test_reward_constraint_approximations.py -q
python3 -m compileall -q exts/cat_envs/cat_envs scripts/eval.py
git diff --check
```

Expected: all tests PASS and both static commands exit zero.

- [ ] **Step 2: Instantiate the task with Isaac Lab and inspect active managers**

Start `AppLauncher(headless=True)`, load `CaT-Go2-Rough-Terrain-v0`, create a small environment, then assert:

```python
assert set(env.unwrapped.reward_manager.active_terms) == {
    "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "minimize_power",
    "joint_torque", "joint_velocity", "joint_acceleration", "action_rate", "base_orientation",
}
assert set(env.unwrapped.constraint_manager.active_terms) == {
    "contact", "foot_contact_force", "front_hfe_position", "upsidedown",
}
assert set(env.unwrapped.curriculum_manager.active_terms) == {"power", "terrain_levels"}
```

Close the environment and application in `finally` blocks.

- [ ] **Step 3: Run a two-iteration PPO smoke job**

Using the project venv:

```bash
ENV_NAME=reward_constraint_approximation_smoke \
WANDB_MODE=offline \
OMNICLIENT_HUB_MODE=disabled \
python scripts/clean_rl/train.py \
  --task=CaT-Go2-Rough-Terrain-v0 \
  --seed=46 \
  --headless \
  --num_envs=64 \
  --num_iterations=2 \
  --logger=tensorboard
```

Expected: exit code zero, two PPO iterations complete, and no missing-term/curriculum errors.

- [ ] **Step 4: Inspect the serialized smoke configuration**

Read the newest smoke-run `params/env.yaml` and verify:

- all five reward terms have weight `0.1`;
- all five have `curriculum_steps: 19200` and exact thresholds/selectors;
- only the four hard constraints are serialized;
- only power and terrain curricula are serialized;
- action scale remains `0.8`;
- power curriculum end weight remains `0.008`.

- [ ] **Step 5: Review the complete branch diff and status**

```bash
git diff HEAD~3 --check
git diff HEAD~3 --stat
git status --short
```

Confirm no unrelated user files changed and no generated logs are staged.

- [ ] **Step 6: Record the profile-switch handoff**

Report that low runs use:

```python
SOFT_CONSTRAINT_REWARD_END_WEIGHT = SOFT_CONSTRAINT_REWARD_END_WEIGHT_LOW
```

and high runs change only that assignment to:

```python
SOFT_CONSTRAINT_REWARD_END_WEIGHT = SOFT_CONSTRAINT_REWARD_END_WEIGHT_HIGH
```

Use distinct `ENV_NAME` values and verify each saved `env.yaml` before launching full multi-seed jobs.

---

### Task 5: Log the Effective Reward-Constraint Curriculum

**Files:**
- Modify: `exts/cat_envs/cat_envs/tasks/utils/cleanrl/ppo.py`
- Modify: `tests/test_reward_constraint_approximations.py`

**Interfaces:**
- Consumes: `env.common_step_counter`, the five active reward term configurations,
  their final `weight` values, and their `params["curriculum_steps"]` values.
- Produces: `_get_soft_constraint_reward_curriculum_state(env)`,
  `_log_soft_constraint_reward_curriculum(writer, env, iteration)`, one stdout line
  per PPO iteration, and seven `Curriculum/...` writer scalars.

- [ ] **Step 1: Add failing state and logging tests**

Load `exts/cat_envs/cat_envs/tasks/utils/cleanrl/ppo.py` directly with
`importlib.util`. Build a fake reward manager whose five terms all have weight
`0.1` and `curriculum_steps=19_200`. Assert the state helper returns progress
`0.5` and five effective weights of `0.05` at common step `9_600`, and returns
progress `1.0` and effective weights `0.1` after saturation.

Call the logging helper with a writer that records `add_scalar` calls and assert
iteration `400` produces exactly these tags:

```python
{
    "Curriculum/soft_constraint_common_step_counter": 9_600.0,
    "Curriculum/soft_constraint_progress": 0.5,
    "Curriculum/joint_torque_effective_weight": 0.05,
    "Curriculum/joint_velocity_effective_weight": 0.05,
    "Curriculum/joint_acceleration_effective_weight": 0.05,
    "Curriculum/action_rate_effective_weight": 0.05,
    "Curriculum/base_orientation_effective_weight": 0.05,
}
```

Assert every writer call uses step `400`, and use `capsys` to assert stdout
contains the iteration, common step counter, progress, and all five term names.
Also assert an environment without all five transferred terms writes and prints
nothing, preserving other tasks' behavior.

- [ ] **Step 2: Run the diagnostic tests and observe the missing-helper failure**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest -q \
  tests/test_reward_constraint_approximations.py -k soft_constraint_reward_curriculum
```

Expected: FAIL because the state and logging helpers do not yet exist.

- [ ] **Step 3: Implement the guarded state helper and logger**

In `ppo.py`, define the exact transferred term tuple and add:

```python
SOFT_CONSTRAINT_REWARD_TERM_NAMES = (
    "joint_torque",
    "joint_velocity",
    "joint_acceleration",
    "action_rate",
    "base_orientation",
)


def _get_soft_constraint_reward_curriculum_state(env):
    if not set(SOFT_CONSTRAINT_REWARD_TERM_NAMES).issubset(
        env.reward_manager.active_terms
    ):
        return None

    term_cfgs = {
        name: env.reward_manager.get_term_cfg(name)
        for name in SOFT_CONSTRAINT_REWARD_TERM_NAMES
    }
    curriculum_steps = {
        int(term_cfg.params["curriculum_steps"])
        for term_cfg in term_cfgs.values()
    }
    if len(curriculum_steps) != 1:
        raise ValueError(
            "soft constraint reward terms must share one curriculum_steps value"
        )
    curriculum_steps = curriculum_steps.pop()
    if curriculum_steps <= 0:
        raise ValueError("soft constraint reward curriculum_steps must be positive")

    common_step_counter = int(env.common_step_counter)
    progress = min(max(common_step_counter / curriculum_steps, 0.0), 1.0)
    return {
        "common_step_counter": float(common_step_counter),
        "progress": progress,
        "effective_weights": {
            name: float(term_cfg.weight) * progress
            for name, term_cfg in term_cfgs.items()
        },
    }
```

Add `_log_soft_constraint_reward_curriculum(writer, env, iteration)`. It returns
without output when the state is `None`; otherwise it writes the common counter,
progress, and five effective weights using the exact tags from Step 1, then prints
one flushed `[INFO][SoftConstraintRewardCurriculum]` line containing the same data.

Call the logger once immediately after each 24-step rollout and before PPO update
logic, using `envs.unwrapped` and the current one-based `iteration`.

- [ ] **Step 4: Run focused tests and static checks**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest -q tests
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m compileall -q \
  exts/cat_envs/cat_envs/tasks/utils/cleanrl/ppo.py tests
git diff --check
```

Expected:  all tests PASS and both static checks exit zero.

- [ ] **Step 5: Run the real two-iteration logging smoke test**

Run the existing 64-environment, two-iteration TensorBoard smoke job with a new
`ENV_NAME`. Assert stdout contains two curriculum lines. Read its TensorBoard event
file and assert all seven tags contain steps `[1, 2]`; specifically, progress is
`0.00125` at iteration 1 and `0.0025` at iteration 2, while the low-profile effective
weights are `0.000125` and `0.00025`.

- [ ] **Step 6: Commit the diagnostic implementation**

```bash
git add exts/cat_envs/cat_envs/tasks/utils/cleanrl/ppo.py \
  tests/test_reward_constraint_approximations.py
git commit -m "Log reward constraint curriculum weights"
```
