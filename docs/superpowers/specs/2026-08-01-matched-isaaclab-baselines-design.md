# Matched Isaac Lab Locomotion Baselines Design

**Date:** 2026-08-01

**Status:** Approved for implementation

## Purpose

Add reproducible Isaac Lab/RSL-RL rough-terrain baselines for Go2, ANYmal C,
and Spot. The baselines must isolate the locomotion-method difference from
environmental differences closely enough that cost of transport, terrain
curriculum level, and RMS velocity-tracking error can be compared to the CaT
method.

The paper has already been submitted, so reliable ablation results take
priority over creating a general environment framework. The `sim2real`
directory is out of scope.

## Goals

1. Add one matched training environment and one evaluation/play environment
   for each of Go2, ANYmal C, and Spot.
2. Match the physical plant and experimental conditions used by the
   corresponding CaT environment where they are not part of the baseline
   method.
3. Preserve the upstream baseline's action scale, reward terms, reward
   weights, observation scaling/clipping, and standard RSL-RL PPO setup.
4. Make terrain curriculum level, CoT, and RMS velocity-tracking metrics use
   equivalent definitions and evaluation scenarios across methods and robots.
5. Make training and evaluation directly runnable through `just` recipes.
6. Validate configuration resolution and simulator startup without adding a
   large unit-test surface.

## Installed Isaac Lab Baseline

The implementation is based on the installed Isaac Lab checkout at:

`/home/kordoslo/mamba_env_data/env_new_isaac_lab/isaaclab-installation/IsaacLab`

The inspected checkout identifies itself as Isaac Lab v2.3.1, package version
0.48.0, at commit `5c2ec81cb17532d32f7922dd7fcaae40d123b71a`.

The installed upstream Go2 rough configuration has uncommitted local edits.
Those edits cannot be treated as canonical upstream behavior. The audit
therefore compared the live file and `git show HEAD:` and the repository
baselines will explicitly set every experimental field they depend on. The
baseline reward and action settings remain inherited from the robot-specific
upstream classes and are also set or validated explicitly where a silent
installed-source edit would be risky.

Upstream configuration roots:

- Common rough locomotion:
  `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Go2 rough:
  `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py`
- ANYmal C rough:
  `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/rough_env_cfg.py`
- Spot baseline:
  `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py`

Spot has no upstream rough-terrain task. Its upstream `SpotFlatEnvCfg` is the
robot-specific baseline and will receive the same matched rough terrain as the
other embodiments while keeping Spot's specialized upstream rewards, action
scale, and standard agent settings.

## Environment Layout

Use three explicit files rather than a shared matching helper:

- `baseline_go2_rough_env_cfg.py`
- `baseline_anymal_c_rough_env_cfg.py`
- `baseline_spot_rough_env_cfg.py`

Each file contains a training class and a play/evaluation class. Matching
settings are intentionally duplicated so a reviewer can audit an embodiment
without following an abstraction and so robot-specific exceptions remain
visible.

Register these task pairs:

- `Baseline-Go2-Rough-Terrain-v0`
- `Baseline-Go2-Rough-Terrain-Play-v0`
- `Baseline-Anymal-C-Rough-Terrain-v0`
- `Baseline-Anymal-C-Rough-Terrain-Play-v0`
- `Baseline-Spot-Rough-Terrain-v0`
- `Baseline-Spot-Rough-Terrain-Play-v0`

The Gym registrations use `ManagerBasedRLEnv`, not `CaTEnv`, and expose the
ordinary upstream `rsl_rl_cfg_entry_point` for each embodiment. ANYmal uses
the standard non-symmetry RSL-RL configuration.

## Fairness Contract

### Settings matched to CaT

The following are experimental conditions, not intended method differences,
and must match the corresponding CaT environments.

#### Robot assets and simulation

- Use the corresponding CaT robot asset for all three robots.
- For Go2 this deliberately uses the CaT training asset with enabled
  self-collisions and its 70 Nm effort/saturation settings instead of the
  canonical upstream 23.5 Nm asset. This makes the plant identical to CaT but
  means the older Go2 upstream result is not expected to reproduce exactly.
- Use the CaT physics step, control decimation, render interval, and PhysX
  overrides. Go2 and ANYmal retain 200 Hz physics and 50 Hz control; Spot
  retains 500 Hz physics and 50 Hz control.
- Use a 10-second episode length for every embodiment and both methods.
- Match scene count and spacing in the training configuration unless a
  command-line `num_envs` override is supplied.

#### Terrain and curriculum

- Deep-copy the full CaT rough-terrain generator configuration, including
  terrain proportions, difficulty ranges, 10-by-20 layout, 8 m tiles,
  horizontal/vertical scales, cache use, and flat-patch sampling.
- Use the same `init_pos` flat patches and
  `reset_root_state_from_terrain` reset path.
- Use `max_init_terrain_level = 1` for training.
- Enable the terrain generator curriculum and importer curriculum during
  training.
- Keep only the standard `terrain_levels_vel` curriculum in the baseline.
  Do not add CaT constraint or power-weight curricula.
- Pass the selected seed to the environment, simulator, and terrain generator.
  This is required because Isaac Lab's terrain cache key is unsafe when cache
  use is enabled while the terrain seed is `None`.
- Play environments use the same seed-dependent 5-by-5 evaluation terrain,
  difficulty range, and non-curriculum setup used by the CaT play
  environments so fixed evaluation scenarios refer to the same geometry.

Terrain level is not only a terrain-generator setting. Isaac Lab promotes an
environment after it travels more than half a tile and demotes it based on
commanded speed and episode duration. Matching commands, episode length,
resets, hard failure conditions, and terrain geometry is therefore required
for level comparisons.

#### Commands

- Use the CaT command ranges: x velocity `[-0.3, 1.0]` m/s, y velocity
  `[-0.7, 0.7]` m/s, and yaw velocity `[-0.78, 0.78]` rad/s.
- Use the same 10-second nominal resampling interval, 2% standing fraction,
  and disabled heading-command mode.
- Default to the exact CaT `UniformVelocityCommandWithDeadzone` generator.
- Add a serialized `use_deadzone_command: bool = True` field to every baseline
  class. Setting it to false switches only the command implementation to
  Isaac Lab's `UniformVelocityCommand`; it does not silently restore different
  robot-specific ranges.
- A nearby comment will document that the CaT generator additionally zeros
  small three-axis commands, opportunistically resamples zero/active commands,
  and probabilistically inverts yaw commands during an episode.

#### Domain randomization, disturbances, and resets

- Match CaT startup material, mass, and center-of-mass randomization for each
  robot, including robot-specific mass ranges and body names.
- Match CaT joint and root reset distributions, including terrain-patch root
  resets, small x/y pose jitter, random yaw, zero root velocity, and the CaT
  joint reset ranges.
- Match both CaT disturbance mechanisms during training:
  - Direct planar base-velocity impulses using the CaT per-physics-step
    probability and embodiment-specific velocity ranges.
  - Repeated external base wrench disturbances with the same force, torque,
    and interval ranges.
- Retain episodic height-map offset behavior where required by the matched CaT
  configuration.
- Disable training disturbances in play/evaluation exactly as the CaT play
  configurations do. Evaluation should not add disturbances implicitly.

#### Hard resets

- Match the CaT deterministic hard terminations: timeout, illegal contact on
  the base and upper-leg bodies, and the upside-down condition.
- Do not add a `ConstraintManager`, probabilistic CaT constraint terminations,
  constraint-based reward attenuation, or CaT constraint curricula.

This makes physical failure/reset criteria equal while leaving the central
method difference intact.

#### Observations

- Remove base linear velocity from every baseline policy observation because
  it is absent from the CaT policy observation.
- Enable observation corruption for all three robots and use the CaT raw noise
  ranges, including for Spot.
- Preserve the upstream observation functions, term ordering, term scaling,
  and clipping. Observation scaling does not affect reward computation, but it
  can affect policy optimization, so it remains part of the upstream baseline.
- Add a terrain height observation to Spot.
- Use a 1.0 m by 0.8 m grid at 0.08 m resolution for every height scanner,
  yielding 13 by 11 = 143 rays. Match the CaT footprint and horizontal sensor
  placement/drift where supported.
- Use the upstream `mdp.height_scan` function and standard ray-caster approach,
  including its sign/offset convention and clipping, rather than replacing it
  with CaT's custom height-map function.

Exact total observation count is not itself a target. Comments will note the
remaining intentional differences: upstream term ordering/functions,
upstream scaling/clipping, and upstream height sign/offset semantics. Noise is
applied before scale in Isaac Lab, so matching raw noise while preserving
upstream scales is intentional and will be documented.

### Settings preserved from upstream

The following are baseline-method choices and must not be retuned to resemble
CaT.

#### Actions

- Go2 action scale: `0.25`
- ANYmal C action scale: `0.5`
- Spot action scale: `0.2`
- Preserve upstream default-joint-position offsets and action implementation.
- Explicitly inspect resolved joint names/order during smoke validation.

Using the CaT asset does not authorize changing these action scales.

#### Rewards

Preserve every upstream reward term and weight for the corresponding robot.
Do not introduce CaT's power reward, constraint reward attenuation, or common
reward reweighting.

- Go2 retains its upstream Go2-specific overrides, including 1.5 linear and
  0.75 yaw tracking weights and its upstream regularization weights.
- ANYmal C retains the common upstream rough-terrain reward configuration.
- Spot retains its specialized gait, clearance, tracking, smoothness, slip,
  base-motion, orientation, joint, and torque reward stack.

Reward Manager weights are multiplied by the environment control step, so
matching each robot's CaT control period prevents an implicit reward-scale
change. Evaluation must not mutate these training reward weights.

#### Learning algorithm

- Use the ordinary upstream RSL-RL PPO configuration registered for each
  robot.
- Use the standard non-symmetry ANYmal agent.
- Do not route baseline training through the repository CleanRL trainer; that
  path assumes CaT-specific reward and curriculum terms.

RSL-RL's randomized initial episode lengths remain part of the upstream
training algorithm rather than an environment configuration difference.

## Training Integration

Add one training recipe per embodiment to `justfile`. Recipes invoke the
installed Isaac Lab RSL-RL trainer and default to:

- Seed 46
- 7,500 environments
- The corresponding baseline training task
- Headless execution

The recipes pass the seed through both the normal RSL-RL seed option and Hydra
overrides for simulator and terrain-generator seeds. They keep useful optional
arguments such as environment count, seed, maximum iterations, and extra
trainer flags easy to override.

Do not modify the installed Isaac Lab checkout as part of this work.

## Evaluation Integration

Add one evaluation recipe per embodiment. Each recipe wraps the existing
`eval` recipe with the correct baseline play task and RSL-RL backend and passes
through additional flags.

`scripts/eval.py` already contains robot profiles for Go2, ANYmal C, and Spot,
RSL-RL loading, height-scanner discovery, action-scale loading, terrain-level
collection, CoT, and raw RMS tracking metrics. Extend and validate it so that:

1. All six baseline task IDs are registered and discoverable.
2. Each play task selects the correct root link, foot links, spawn height, and
   joint mapping.
3. The checkpoint backend resolves to RSL-RL and uses the task's ordinary
   agent entry point.
4. The saved training action scale is applied to the play environment and
   recorded in evaluation metadata.
5. Baseline reward weights are never rewritten during evaluation.
6. CoT uses the runtime randomized robot mass and actual applied torque/joint
   velocity data.
7. Terrain level and RMS x/y/yaw tracking errors use the same raw state and
   command definitions as CaT evaluation.
8. Evaluation-only constraint thresholds corresponding to the CaT embodiment
   are supplied to metric summarization for comparable violation reporting.
   These thresholds do not add constraints, rewards, or terminations to the
   baseline environment.
9. Summary metadata accurately records the task, backend, action scale,
   terrain seed/fingerprint, reward-scale policy, constraint-bound source, and
   evaluation settings.

The fast evaluation diagnostic is:

`--random_sim_step_length=0 --skip_cot_sweep`

It intentionally omits or changes random-command and CoT-sweep sections of
`metrics_summary.json`; it is not a substitute for the final quantitative
evaluation.

## Existing Go2 Reference

The requested reference path is:

`/local_storage1/loukas-backup/logs/clean_rl/cluster_runs/env_id_107_action_scale_sweep/constraints-as-terminations/logs/clean_rl/env_id_107_action_scale_sweep/2026-07-03-09-25-54/eval_checkpoint_21799_seed_46/metrics_summary.json`

That file is not currently present on the mounted filesystem. Previously
recorded reference metadata in this repository gives:

- Checkpoint: `model_21799.pt`, seed 46, action scale 0.8
- Overall CoT: `0.4196298480495525`
- RMS x tracking error: `0.32775482535362244`
- RMS y tracking error: `0.1588231772184372`
- RMS yaw tracking error: `0.09639634191989899`
- Mean evaluation terrain level: `3.0406896551724136`

Because the new Go2 baseline deliberately uses the CaT plant and the reference
uses a different method/action scale, these numbers are sanity/context values,
not an expected reproduction target. No new full Go2 baseline run is required
for implementation verification.

## Verification Strategy

Prefer focused inspection and runtime smoke validation over new hardcoded unit
tests.

1. Review each baseline file against both its CaT counterpart and canonical
   upstream parent.
2. Compile/import the new Python modules and verify all Gym IDs and RSL-RL
   entry points resolve.
3. Instantiate every configuration and print a compact fairness fingerprint:
   robot asset, action scale, reward terms/weights, episode/control timing,
   terrain hash-relevant fields, curriculum, commands, observations, events,
   resets, hard terminations, and seeds.
4. Assert 143 height rays and absence of base-linear-velocity policy input.
5. Run one-environment headless startup/reset/zero-action smoke tests for all
   three training and play families as appropriate.
6. Run short RSL-RL startup diagnostics with a small number of environments
   and iterations to catch observation/action/runner mismatches.
7. Validate the evaluation configuration and RSL-RL loading path. Run a fast
   checkpoint evaluation if a compatible local checkpoint is available.
8. Review the complete diff for inherited-field leakage, reward mutation,
   body/joint selector mistakes, and accidental `sim2real` changes.
9. After each simulator diagnostic, close the environment and app, check for
   lingering Isaac/Omniverse/Python processes, and inspect GPU memory.

## Success Conditions

- Six task IDs resolve from a clean repository import.
- All three training configs use the CaT asset and matched experimental
  conditions while preserving upstream action scales and reward stacks.
- Every play config matches the intended training action/observation contract
  and evaluation terrain contract.
- Height scans have 143 rays and use upstream height observation semantics.
- Training and play simulator smoke tests close cleanly.
- Short RSL-RL diagnostics start without configuration, observation, action,
  or reward errors.
- `eval.py` reaches the RSL-RL policy path and records the primary metrics and
  required metadata for every embodiment.
- `justfile` exposes clear train/eval commands for every embodiment.
- No unrelated or `sim2real` files change.
- The commit history separates design, environments, and integration/fixes.

## Failure Conditions

The assignment is not complete if any of the following holds:

- A baseline silently inherits a different terrain, episode length, command
  distribution, reset distribution, disturbance, plant, or hard reset rule.
- Terrain caching uses an unresolved seed or different generated geometry for
  corresponding CaT/baseline evaluations.
- An upstream action scale or reward term/weight is changed for matching
  convenience.
- Base linear velocity remains in a baseline policy observation.
- Height-map geometry differs or a custom CaT height function replaces the
  upstream baseline implementation.
- CaT constraints/reward attenuation leak into baseline training.
- Spot silently trains on its upstream cobblestone/flat task instead of the
  matched rough terrain.
- Evaluation uses wrong robot body/joint mappings, mutates baseline reward
  weights, or cannot load an RSL-RL checkpoint.
- A simulator process or material GPU/RAM allocation is left behind.
- Work depends on uncommitted modifications in the installed Isaac Lab source.

## Pitfalls and Caveats

- Python inheritance and `__post_init__` mutations can make an apparently
  small leaf config differ substantially from its source declaration.
- The locally dirty installed Go2 source can leak behavior unless every
  relevant matched field is explicitly overwritten and validated.
- Spot's source task is specialized, lacks a height scanner, replaces the
  rough terrain with cobblestone terrain, and has different observations,
  rewards, terminations, timing, and agent settings.
- The upstream Spot play configuration appears incomplete; the repository
  baseline play config must explicitly disable evaluation disturbances instead
  of relying on it.
- Command ranges alone do not match command distributions because CaT performs
  additional per-step resampling and yaw inversion.
- Curriculum level comparisons are confounded by episode length, command
  speed, hard failures, spawn sampling, and tile dimensions even when the
  curriculum term has the same name.
- Height scanners can have equal dimensions while differing in attachment,
  footprint, reference height, sign, offset, drift, noise, and clipping.
- Action scale, joint order, default offsets, actuator limits, self-collision,
  and RSL-RL action clipping jointly define the effective control interface.
- Observation noise is applied before scale. Equal raw noise ranges do not
  imply identical scaled policy noise because upstream scaling is deliberately
  retained.
- Reward terms are multiplied by the control time step. Equal written weights
  are insufficient if control periods differ.
- Startup material/mass/CoM randomization and interval-event timing have mode,
  global/local-time, and reset semantics that must be copied exactly.
- Evaluation-only constraint thresholds are reporting definitions, not
  permission to add CaT constraints to the baseline environment.
- Short smoke tests validate wiring and numerical sanity, not learning quality.
  Final conclusions still require full training and quantitative evaluation.
