# Go2 and ANYmal C Cross-Tuning Baselines Design

**Date:** 2026-08-03

**Status:** Approved for implementation, pending written-specification review

## Purpose

Add two matched reward-based baseline experiments that transfer the tunable
locomotion recipe between Unitree Go2 and ANYmal C. Each experiment keeps the
receiving robot and its physical/environmental configuration but uses the donor
robot's numerical reward weights, joint-position action scale, and standard
RSL-RL PPO hyperparameters.

The experiments test whether the upstream locomotion recipes are portable
between embodiments or whether their successful behavior depends on
embodiment-specific tuning. They are transfer tests, not single-factor
ablations: a result cannot attribute a change in performance uniquely to the
reward weights, action scale, or PPO entropy coefficient.

## Reward Audit and Scaling Conclusion

Go2 and ANYmal C inherit the same common rough-locomotion reward functions and
parameters. Their resolved configurations differ as follows:

| Reward term | ANYmal C | Go2 |
|---|---:|---:|
| `track_lin_vel_xy_exp` | `1.0` | `1.5` |
| `track_ang_vel_z_exp` | `0.5` | `0.75` |
| `lin_vel_z_l2` | `-2.0` | `-2.0` |
| `ang_vel_xy_l2` | `-0.05` | `-0.05` |
| `dof_torques_l2` | `-1.0e-5` | `-2.0e-4` |
| `dof_acc_l2` | `-2.5e-7` | `-2.5e-7` |
| `action_rate_l2` | `-0.01` | `-0.01` |
| `feet_air_time` | `0.125` | `0.01` |
| `undesired_contacts` | `-1.0` | disabled |
| `flat_orientation_l2` | `0.0` | `0.0` |
| `dof_pos_limits` | `0.0` | `0.0` |

Both tracking terms use the same squared-error exponential kernels and
`std = 0.5`. Both air-time terms use the same first-contact reward, command
gate, and `0.5` second threshold; only robot-specific foot selectors and the
weight differ. The other shared terms also use identical functions.

The nominal simulated masses are 15.019 kg for Go2 and 52.135 kg for ANYmal C,
a ratio of 3.471 and a squared ratio of 12.050. After normalizing the torque
penalty by each linear-tracking weight, Go2's relative torque coefficient is
13.333 times ANYmal's. The torque coefficient is therefore broadly consistent
with inverse squared-mass scaling because the raw penalty sums squared joint
torques. The remaining choices are not explained by mass scaling: Go2 raises
both tracking weights by 50%, reduces air-time reward by 12.5 times, disables
the contact penalty, and leaves several other penalties unchanged. The
upstream recipes required term-specific tuning rather than a single mass
multiplier.

The `action_rate_l2` weight is already `-0.01` in both recipes. Transferring it
is a no-op. The action scale is a separate MDP parameter and differs materially:
Go2 uses `0.25` rad while ANYmal C uses `0.5` rad.

The standard non-symmetry PPO configurations are identical except for entropy:
Go2 uses `entropy_coef = 0.01` and ANYmal C uses `0.005`. Experiment names are
metadata, not transferable optimization hyperparameters, and will identify the
crossed configuration. The command-line training budget remains 30,000
iterations for both.

## Experiment Matrix

### Go2 receiving ANYmal C tuning

- Robot and environment: existing matched Go2 baseline.
- Numerical reward weights: ANYmal C values for every shared term.
- Contact structure: retain the current Go2 baseline behavior;
  `undesired_contacts` remains disabled.
- Reward functions and parameters: retain Go2's resolved functions and
  robot-specific foot selectors.
- Action scale: `0.5` rad from ANYmal C.
- PPO configuration: inherit the standard non-symmetry ANYmal C rough runner,
  including `entropy_coef = 0.005`.
- Task ID:
  `Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0`.
- Play task ID:
  `Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-Play-v0`.
- W&B project: `baseline_go2_anymal_c_rewards_action_scale_ppo`.
- Runner experiment name: `go2_anymal_c_tuning_rough`.

### ANYmal C receiving Go2 tuning

- Robot and environment: existing matched ANYmal C baseline.
- Numerical reward weights: Go2 values for every shared term.
- Contact structure: retain the current ANYmal C baseline behavior;
  `undesired_contacts` remains enabled at `-1.0` with the ANYmal thigh selector.
- Reward functions and parameters: retain ANYmal C's resolved functions and
  robot-specific body selectors.
- Action scale: `0.25` rad from Go2.
- PPO configuration: inherit the standard Go2 rough runner, including
  `entropy_coef = 0.01`.
- Task ID:
  `Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0`.
- Play task ID:
  `Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-Play-v0`.
- W&B project: `baseline_anymal_c_go2_rewards_action_scale_ppo`.
- Runner experiment name: `anymal_c_go2_tuning_rough`.

The W&B names list the receiving embodiment first and the donor recipe second.
They explicitly name all three transferred categories so the runs are not
misread as reward-only experiments.

## Environment Architecture

Create one focused cross-tuning environment module next to the existing
baseline files. Define small application helpers for the ANYmal-to-Go2 and
Go2-to-ANYmal numerical mutations, then use them from explicit training and
play subclasses. The subclasses inherit their receiving embodiment's matched
baseline class, call the complete parent `__post_init__`, and apply crossed
settings last.

This preserves the receiving embodiment's:

- robot USD, initial pose, actuators, effort behavior, and body/joint names;
- simulation step, decimation, episode length, and action implementation;
- terrain generator, terrain curriculum, commands, and height scanner;
- observations and noise;
- material, mass, center-of-mass, reset, and disturbance randomization;
- terminations and contact body selectors; and
- train/play differences.

The transfer helpers change only reward weights and
`actions.joint_pos.scale`. They do not copy reward configuration objects
between robots, because that would also copy incompatible body selectors. Each
helper validates the receiving contact structure after mutation so a future
upstream or repository change cannot silently alter this experiment.

Play subclasses receive the same crossed rewards and action scale as their
training counterparts. Saved training `params/env.yaml` remains authoritative
for evaluation, while the play task supplies the correct receiving embodiment
and crossed defaults.

## PPO Architecture

Add two explicit RSL-RL runner classes in the local `agents` package:

- the Go2/ANYmal-tuning runner subclasses the upstream
  `AnymalCRoughPPORunnerCfg`;
- the ANYmal/Go2-tuning runner subclasses the upstream
  `UnitreeGo2RoughPPORunnerCfg`.

Each subclass changes only `experiment_name`. This transfers the donor's full
policy and PPO configuration without giving log directories a misleading donor
robot name. Gym registrations point both the training and play task of a cross
configuration at its crossed runner.

The transfer intentionally does not use the optional ANYmal symmetry runner,
because the current matched ANYmal baseline uses the standard non-symmetry
runner and Go2 has no corresponding symmetry configuration.

## Training Interface and Manual Handoff

Add one public `just` recipe per crossed training task. Both reuse the existing
shared RSL-RL baseline recipe and therefore expose the same arguments and
flags. Defaults are:

- 7,500 environments;
- seed 46;
- 30,000 iterations;
- W&B logging to the experiment-specific project above.

Production training must not be started automatically. After validation, hand
the user the two exact commands and recommend running them sequentially on the
local RTX 2080 Ti. The user controls when each command starts and may change
the environment count if local GPU memory requires it. No Slurm files, remote
cluster state, or existing matched-baseline launchers are changed.

## Validation

Validation must establish the transfer boundary before handing off production
commands:

1. Compile changed Python files.
2. Confirm all four task IDs register and resolve.
3. Assert the complete ordered reward contract for both crossed training and
   play configs, including unchanged contact enablement and selectors.
4. Assert action scales `0.5` for crossed Go2 and `0.25` for crossed ANYmal C.
5. Assert the receiving assets, action implementation, observation structure,
   timing, terrain, events, and terminations remain equal to their matched
   baseline counterparts.
6. Assert the crossed runner configurations match every donor PPO and policy
   field except `experiment_name`; specifically check entropy coefficients
   `0.005` and `0.01`.
7. Dry-run both public `just` recipes and verify task ID, seed, environment
   count, iteration budget, logger, and W&B project.
8. Start each crossed environment with a small environment count, reset, and
   take finite simulation steps.
9. Run a short two-iteration RSL-RL smoke training for each task with local
   TensorBoard logging so validation does not create misleading production
   W&B runs.

If 7,500 environments exceed available memory in the production run, lower the
environment count rather than changing reward, action, PPO, or environmental
settings. Record the actual count in W&B automatically through the saved
configuration.

## Scientific Interpretation

These two runs answer whether each complete numerical tuning package transfers
to the other physical embodiment while contact policy remains receiver-specific.
They do not isolate individual causes. In particular:

- reward weights, action scale, and PPO entropy change together;
- Go2 keeps no thigh-contact shaping while ANYmal keeps it;
- actuator models, nominal poses, mass randomization, and PPO observation/action
  distributions remain embodiment-dependent by design.

If attribution is needed after these runs, the minimal follow-up is a factorial
ablation that changes rewards only, action scale only, and PPO only for each
receiver. That follow-up is outside this implementation and training handoff.

## Out of Scope

- Spot analysis or configuration changes.
- Automatic production training launch.
- Slurm or `tcml-cluster` changes.
- New multi-seed sweeps.
- Transferring contact enablement or body selectors.
- Altering physical assets, actuator models, commands, terrain, randomization,
  terminations, observations, or evaluation metrics.
- Factorial attribution experiments beyond the two requested crossed runs.
