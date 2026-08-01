# Matched Isaac Lab Locomotion Baselines Implementation Plan

> **For Codex:** Execute this plan sequentially in the current workspace. Do
> not use subagents. Keep simulator checks serial, close every environment and
> AppLauncher, and verify no process/GPU leak after each runtime batch.

**Goal:** Add scientifically matched upstream Isaac Lab/RSL-RL baselines for
Go2, ANYmal C, and Spot, with train/eval recipes and comparable evaluation
metrics.

**Architecture:** Add three explicit robot-specific configuration files. Each
inherits the installed upstream robot task, preserves that task's rewards,
actions, and ordinary RSL-RL agent, and directly duplicates the approved CaT
experimental-condition overrides. Register training and play variants with
`ManagerBasedRLEnv`. Extend the existing evaluator and `justfile` without
creating a second evaluation or training framework.

**Technology:** Python, Isaac Lab manager-based environments, Gymnasium,
RSL-RL, Hydra configuration, Just, Git.

**Design reference:**
`docs/superpowers/specs/2026-08-01-matched-isaaclab-baselines-design.md`

**Testing policy:** These changes are predominantly declarative simulator
configuration. Per the user-approved testing constraints, do not add permanent
unit-test infrastructure. Use import/compile checks, configuration assertions,
one-environment simulator smoke tests, short RSL-RL startup runs, saved-config
inspection, evaluator-path checks, and diff review. If a defect requires new
nontrivial logic, add the smallest focused regression check before fixing it.

---

## Task 1: Create the explicit Go2 matched baseline

**Files:**

- Create:
  `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_go2_rough_env_cfg.py`
- Inspect:
  `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_go2_rough_terrain_env_cfg.py`
- Inspect canonical installed source with `git show HEAD:`:
  `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py`

### Step 1: Record canonical Go2 invariants

Before editing, record the canonical upstream action scale and reward term
weights from the installed checkout's committed file. Do not trust its dirty
working-tree version.

Expected invariants include:

- Action scale `0.25`, default joint offset.
- Linear/yaw tracking weights `1.5`/`0.75`.
- Go2-specific air-time, torque, acceleration, and contact overrides.

### Step 2: Define local config blocks

In the new file, define explicit robot-appropriate configuration blocks for:

- Matched terrain sensor geometry using `RayCasterCfg`, `GridPatternCfg`,
  1.0-by-0.8 m size, 0.08 m resolution, yaw alignment, and the Go2 root path.
- CaT startup/reset/interval events, copied exactly for Go2.
- CaT hard termination terms without constraints.
- Terrain-only curriculum.

Import the CaT Go2 training asset and deep-copy the CaT rough terrain
generator. Do not import or use `CaTEnv` or the CaT constraint configuration.

### Step 3: Implement the training class

Create `BaselineGo2RoughEnvCfg(UnitreeGo2RoughEnvCfg)` with a serialized field:

```python
use_deadzone_command: bool = True
```

In `__post_init__`, call the upstream parent first, then explicitly apply the
approved matched fields:

- CaT Go2 asset.
- 7,500 environments and 3 m spacing.
- CaT rough terrain, flat patches, max initial level 1, and curriculum flags.
- 10-second episodes and CaT Go2 simulation settings.
- CaT command ranges and exact deadzone generator by default.
- If `use_deadzone_command` is false, change only `class_type` to upstream
  `UniformVelocityCommand` while retaining the matched command ranges.
- CaT noise ranges, with upstream observation functions/scales/clipping.
- Remove `base_lin_vel`.
- Resize the upstream height scanner/term to 143 rays.
- Exact CaT Go2 events and hard terminations.
- Terrain-only curriculum.
- Explicitly restore/validate action scale `0.25` after all parent mutations.

Add a concise comment beside the command switch describing deadzone,
opportunistic resampling, and yaw inversion. Add a concise observation comment
describing upstream order/functions/scales and height semantics.

Do not assign to `self.rewards`; the inherited Go2 reward object must remain
untouched.

### Step 4: Implement the play class

Create `BaselineGo2RoughEnvCfg_PLAY` and explicitly apply the CaT play
contract:

- One environment and play spacing.
- Same 5-by-5 seed-dependent evaluation terrain and difficulty range.
- Terrain curriculum disabled.
- Observation corruption remains enabled.
- Training disturbances disabled explicitly.
- Training action/observation contracts unchanged.

### Step 5: Run static checks

Run:

```bash
python -m py_compile \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_go2_rough_env_cfg.py
git diff --check
```

Expected: no syntax or whitespace errors.

---

## Task 2: Create the explicit ANYmal C matched baseline

**Files:**

- Create:
  `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_anymal_c_rough_env_cfg.py`
- Inspect:
  `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py`
- Inspect:
  installed `config/anymal_c/rough_env_cfg.py`

### Step 1: Record canonical ANYmal invariants

Confirm upstream ANYmal uses `ANYMAL_C_CFG`, action scale `0.5`, and the common
upstream rough reward stack without Go2's leaf reward overrides.

### Step 2: Duplicate matched blocks explicitly

Define ANYmal-specific ray scanner, events, hard terminations, and terrain-only
curriculum in this file. Do not import the corresponding blocks from the Go2
baseline file.

Use ANYmal root/body/foot selectors, mass range, direct-push range, joint
patterns, sensor path, and CaT asset exactly as resolved in the CaT ANYmal
configuration.

### Step 3: Implement training and play classes

Create:

- `BaselineAnymalCRoughEnvCfg(AnymalCRoughEnvCfg)`
- `BaselineAnymalCRoughEnvCfg_PLAY`

Apply the same field categories as Task 1, with ANYmal-specific values.
Explicitly restore/validate action scale `0.5`. Do not assign or reweight the
inherited upstream rewards. Expose the same `use_deadzone_command` switch and
comments.

### Step 4: Run static checks

Run `py_compile` on the file and `git diff --check`.

---

## Task 3: Create the explicit Spot matched baseline

**Files:**

- Create:
  `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_spot_rough_env_cfg.py`
- Inspect:
  `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py`
- Inspect:
  installed `config/spot/flat_env_cfg.py`

### Step 1: Record canonical Spot invariants

Record Spot's specialized upstream action scale `0.2`, reward terms/weights,
observation terms/scales/clipping, 50 Hz control, and standard
`SpotFlatPPORunnerCfg`. Treat the upstream cobblestone terrain, missing height
scan, command ranges, events, and hard terminations as experimental fields to
replace.

### Step 2: Define Spot-specific matched blocks

Duplicate the approved terrain, ray scanner attached to `body`, CaT Spot
events, hard terminations, and terrain-only curriculum. Use the CaT Spot asset,
including its deterministic delayed-actuator-compatible classes and 500 Hz
physics configuration.

### Step 3: Implement training and play classes

Create:

- `BaselineSpotRoughEnvCfg(SpotFlatEnvCfg)`
- `BaselineSpotRoughEnvCfg_PLAY`

Apply all matched conditions explicitly. Add upstream `mdp.height_scan` as the
final policy observation with 143 rays and CaT raw height noise, while
retaining the upstream height function's offset/sign/clipping. Enable
observation corruption and apply CaT raw noise ranges to the existing terms,
without changing upstream scales or order. Remove base linear velocity.

Explicitly restore/validate action scale `0.2`. Never replace or mutate the
specialized Spot rewards.

The play class must explicitly disable disturbances because the installed
upstream Spot play class does not reliably do so.

### Step 4: Run static checks

Run `py_compile` on the file and `git diff --check`.

---

## Task 4: Register all baseline environments and validate pure config state

**Files:**

- Modify:
  `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py`

### Step 1: Add six Gym registrations

Use `ManagerBasedRLEnv` as the entry point. Add the six approved task IDs and
point each to its local train/play config class.

Use these external RSL-RL config entry points:

- Go2:
  `isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg`
- ANYmal C:
  `isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg`
- Spot:
  `isaaclab_tasks.manager_based.locomotion.velocity.config.spot.agents.rsl_rl_ppo_cfg:SpotFlatPPORunnerCfg`

Do not expose CleanRL entry points for the baselines.

### Step 2: Run a headless configuration audit

Launch Isaac Lab headlessly with escalated GPU access, import the local task
package, and instantiate all six configurations without creating full
environments where possible. Assert and print:

- Gym IDs and RSL-RL agent configs resolve.
- Entry point is `ManagerBasedRLEnv`.
- Action scales are 0.25/0.5/0.2.
- Reward term names/weights match freshly instantiated canonical upstream
  parents.
- Episode length is 10 seconds; control time is 0.02 seconds.
- Terrain fields and generator sub-config dictionaries match CaT.
- Training terrain max initial level is 1 and curriculum is enabled.
- Play terrain is 5-by-5 and curriculum is disabled.
- `base_lin_vel is None`.
- Height pattern produces 143 rays.
- Noise ranges match CaT while scales match upstream.
- Command ranges/class and false-switch class resolve as designed.
- Matched events and hard terminations exist; no `constraints` config exists.

Always flush diagnostic output before closing AppLauncher. Close in `finally`
where practical. Then check `ps` and `nvidia-smi`.

### Step 3: Review and commit the environment milestone

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Review all three files side by side. Commit only the environment files and
registration changes:

```bash
git add \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_go2_rough_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_anymal_c_rough_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/baseline_spot_rough_env_cfg.py \
  exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/__init__.py
git commit -m "Add matched Isaac Lab locomotion baselines"
```

---

## Task 5: Extend evaluation metadata and baseline constraint reporting

**Files:**

- Modify: `scripts/eval.py`
- Inspect: `scripts/metrics_utils.py`

### Step 1: Make matched baseline detection explicit

Add a narrow predicate for the six `Baseline-*` IDs. Do not broaden the
legacy upstream-Go2 predicate in a way that mutates the new baseline rewards.

### Step 2: Generalize evaluation-only reporting thresholds

Represent the CaT-equivalent evaluation thresholds per robot profile and use
them only when a matched baseline has no `constraints` block. Preserve current
behavior for CaT tasks and the legacy upstream Go2 task.

Record a distinct `constraint_bounds_source`, such as
`hardcoded_matched_baseline_<robot>_eval_thresholds`, so a result cannot be
mistaken for trained constraints.

Do not update environment terminations, rewards, or manager configuration from
these reporting bounds.

### Step 3: Prevent reward mutation and correct metadata

Ensure `apply_common_eval_reward_scale_if_needed` cannot match the new task
IDs. Keep upstream training reward weights intact and make the summary note
distinguish:

- Environment reward weights: upstream and unchanged.
- Separately computed raw/common tracking metrics: comparable reporting only.

### Step 4: Validate robot/evaluator routing statically

Use a focused import/config script to assert each play task:

- Resolves the correct `RobotEvalProfile`.
- Loads an RSL-RL runner config.
- Uses the right root and feet for added evaluation sensors.
- Receives the appropriate reporting thresholds.
- Does not trigger reward-scale mutation.

Run `py_compile` and `git diff --check`.

---

## Task 6: Add training and evaluation recipes

**Files:**

- Modify: `justfile`

### Step 1: Resolve the installed RSL-RL trainer without modifying Isaac Lab

Add a narrowly scoped `just` variable or recipe-local lookup that resolves the
installed Isaac Lab root from the active Python environment, then invokes:

`scripts/reinforcement_learning/rsl_rl/train.py`

Avoid embedding a second training implementation in this repository.

### Step 2: Add three training recipes

Add explicit recipes for Go2, ANYmal C, and Spot with defaults for 7,500
environments and seed 46. Accept maximum iterations and extra RSL-RL/Hydra
flags. Pass:

- Correct training task ID.
- `--headless`, `--num_envs`, `--seed`, and optional max iterations.
- `scene.terrain.terrain_generator.seed=<seed>`.
- `sim.random_seed=<seed>` if supported by the installed config schema.

Keep logs under the RSL-RL trainer's standard `logs/rsl_rl/<experiment>`
layout so `eval.py` can find `params/env.yaml`, `params/agent.yaml`, and model
checkpoints.

### Step 3: Add three evaluation recipes

Wrap the existing `eval` recipe with each baseline play task and
`--policy_backend=rsl_rl`. Pass through arbitrary evaluator flags.

### Step 4: Validate recipe expansion

Run `just --list` and dry-run/echo expansion where supported. Verify the fast
diagnostic flags remain accepted:

```text
--random_sim_step_length=0 --skip_cot_sweep
```

---

## Task 7: Run simulator and short-training diagnostics

**Files:**

- No permanent files expected, except fixes discovered by diagnostics.

### Step 1: One-environment simulator smoke tests

For each training task, serially:

1. Start AppLauncher headlessly with GPU access.
2. Parse the config with one environment and seed 46.
3. Reduce terrain rows/columns only if needed for startup latency, without
   changing the committed config.
4. `gym.make`, reset, and take at least one zero-action step.
5. Assert finite observations/rewards, 12 actions, expected observation terms,
   reward term names/weights, terrain level availability, and metric sensor
   selectors.
6. Close the environment and AppLauncher.
7. Verify no lingering process and check GPU memory.

Repeat representative play-task startup checks, especially Spot's newly added
height scanner and disturbance removal.

### Step 2: Short RSL-RL startup runs

Run each new training recipe with a small environment count (for example 64)
and two iterations. Check:

- Environment and terrain initialize.
- Runner sees the expected observation and action dimensions.
- PPO rollout/update completes without NaN/Inf.
- Saved `params/env.yaml` contains the correct asset, action scale, episode
  length, terrain seed, command class, events, and no CaT constraints.
- Saved `params/agent.yaml` is the intended standard upstream agent.
- Processes close and VRAM returns after each embodiment.

Do not interpret these runs as performance evidence.

### Step 3: Evaluation-path validation

If a compatible baseline RSL-RL checkpoint exists, run the fast evaluation
path with zero random steps and no CoT sweep. Otherwise validate task parsing,
runner config construction, sensor attachment, metric-array construction, and
checkpoint discovery up to the missing-checkpoint boundary, and report that
full policy evaluation remains pending a trained checkpoint.

The supplied historical Go2 `metrics_summary.json` is currently missing from
the mounted path. Do not launch a fresh 45-minute Go2 evaluation to compensate.

### Step 4: Fix diagnostics one defect at a time

For any failure, use systematic debugging: preserve the failure output,
identify the first incorrect resolved config/runtime value, apply the smallest
fix, rerun the focused failing check, then rerun the broader smoke batch.

---

## Task 8: Commit integration and perform final review

**Files:**

- Modify as required by Tasks 5-7.

### Step 1: Review evaluation and recipe diffs

Confirm no `sim2real` file changed and no installed Isaac Lab file was edited.
Review all `eval.py` branches for legacy CaT and Go2 regression behavior.

### Step 2: Commit integration

Stage only `scripts/eval.py` and `justfile`, plus directly related diagnostic
fixes, then commit:

```bash
git commit -m "Integrate matched baselines with training and evaluation"
```

If simulator diagnostics require nontrivial environment fixes after the
environment milestone, commit them separately with a narrow message.

### Step 3: Run final verification

Run fresh:

- `py_compile` for every changed Python file.
- Six-task registry and pure-config assertions.
- One-environment runtime smoke batch.
- Short RSL-RL startup batch or the maximum safe subset if an external runtime
  limitation is documented.
- `just --list`.
- `git diff --check`.
- `git status --short`.
- Process and GPU-memory checks.

Inspect recent commits and ensure the history contains separate design,
environment, and integration milestones.

### Step 4: Report handoff

Report:

- New task IDs and exact train/eval recipes.
- Matched versus preserved configuration decisions.
- Verification commands and outcomes.
- Smoke-run log locations.
- Any remaining need for full training/evaluation.
- Missing historical Go2 reference-file limitation.
- Final commit IDs and working-tree status.

