# Reward-Based Operational-Constraint Approximation Design

## Purpose

This experiment answers the reviewer's fairness criticism with a controlled ablation of the existing Go2 rough-terrain LEP task. The five paper-defined soft operational limits will use reward penalties instead of CaT continuation signals. The task, policy, PPO implementation, action scale, energy objective, perception, terrain, commands, randomization, training budget, and hard CaT behavior will otherwise remain unchanged.

This is an ablation of constraint encoding, not a reproduction of the conventional reward-shaped baseline and not an attempt to reproduce CaT's stochastic-horizon mechanism inside a reward.

## Scope Decisions

- Modify the existing `CaT-Go2-Rough-Terrain-v0` configuration directly. Do not register another environment.
- Preserve the four currently active `max_p=1` hard CaT terms exactly as implemented.
- Remove the five paper-defined soft operational terms from CaT:
  - joint torque;
  - joint velocity;
  - joint acceleration;
  - action rate;
  - base orientation.
- Disable the legacy `hip_position` and `no_move` soft CaT terms. They are not part of the five-limit paper definition and have negligible behavioral effect according to the experiment owner.
- Disable every CaT curriculum associated with a removed soft constraint.
- Keep the existing power and terrain curricula unchanged.
- Select the low or high experiment by changing one configuration constant before launching the corresponding runs. This intentionally favors a small one-off implementation over a new task or generalized command-line interface.

## Existing CaT Semantics

Each constraint function returns signed violations, with positive values representing threshold excess. CaT normalizes a positive component by a moving batch-maximum violation, maps it to a bounded probability, and takes the maximum probability across every component and term. `CaTEnv.step()` then:

1. multiplies the total reward by one minus that probability;
2. clips the resulting reward to be nonnegative; and
3. returns the probability as the PPO `done`/continuation signal.

The new soft penalties will not reproduce moving normalization, probability clipping, maximum termination probability, or fractional continuation. Because the hard CaT terms remain active, the environment will continue to use the existing nonnegative total-reward clipping and hard-CaT continuation behavior. This is intentional to minimize changes outside the five soft terms. Consequently, sufficiently large reward penalties saturate the total reward at zero rather than making it negative.

An implementation detail relevant to interpreting the retained hard terms is that `max_p=1` yields zero PPO continuation, but does not by itself call `_reset_idx()`. Actual simulator reset still depends on the regular termination manager. This experiment preserves that existing behavior rather than correcting it.

## Quantities and Thresholds

The reward functions must reuse the current constraint functions so their selectors, units, and state sources cannot drift.

| Term | Existing signed violation | Threshold |
|---|---|---:|
| Joint torque | `abs(applied_torque) - limit` | `20.0 Nm` |
| Joint velocity | `abs(joint_vel) - limit` | `25.0 rad/s` |
| Joint acceleration | `abs(joint_acc) - limit` | `800.0 rad/s^2` |
| Action rate | `abs(action - previous_action) / step_dt - limit` | `80.0 s^-1` |
| Base orientation | `norm(projected_gravity_b[:, :2]) - limit` | `0.1` |

For term (i), environment (n), and component (j), define

\[
e_{nij}=\frac{\max(0,c_{nij})}{q_i^{\max}}.
\]

Joint-wise terms reduce components using the maximum:

\[
e_{ni}=\max_j e_{nij}.
\]

Base orientation is already scalar per environment. The raw reward output is `-e_ni` multiplied by its curriculum progress. There is no upper clipping: larger excess produces a larger penalty until the environment's existing total-reward clipping saturates the combined reward at zero.

Maximum reduction is chosen because CaT currently reacts to the worst active component. A mean would dilute one severe joint violation by a factor of 12; a sum would make the four joint-wise terms structurally larger than base orientation.

## Weight Profiles

All five normalized penalties share one final coefficient to avoid introducing five additional tuning dimensions.

- Low profile: `SOFT_CONSTRAINT_REWARD_END_WEIGHT = 0.1`
- High profile: `SOFT_CONSTRAINT_REWARD_END_WEIGHT = 10.0`

The source will define named low and high constants and one active assignment so switching profiles changes one line. The selected numerical weights will appear in the serialized environment configuration.

The task-tracking reward has a maximum raw value of 1.5. For normalized excess (e), a term costs \(\lambda e\) before the common reward time-step multiplier:

| Threshold excess | Low cost | High cost |
|---:|---:|---:|
| 1% | 0.001 | 0.1 |
| 10% | 0.01 | 1.0 |
| 100% | 0.1 | 10.0 |

The low profile is deliberately weak enough that substantial violations remain cheap relative to tracking. The high profile makes modest violations comparable to the full task reward and is expected to suppress exploration once the curriculum is mature. These values bracket opposite tuning failure modes; they are not claimed to be optimal.

## Curriculum Alignment

The existing PPO rollout horizon is 24 environment steps, and W&B logs once per PPO iteration. The soft-penalty curriculum therefore spans

```text
24 environment steps/iteration * 800 iterations = 19,200 environment steps.
```

Each reward penalty will multiply its normalized violation by

\[
p(k)=\min(k/19{,}200,1),
\]

where `k` is `env.common_step_counter`. Applying progress inside the reward computation avoids the reset-driven staircase and shared-counter flaw in the current CaT curriculum. It reaches the final weight during PPO/W&B iteration 800 and stays there.

The reward term's configured weight is the final profile weight. Its function supplies the progress multiplier. This keeps the effective coefficient equal to `configured_weight * p(k)` while preserving Isaac Lab's normal per-term logging.

## Configuration Changes

In `cat_go2_rough_terrain_env_cfg.py`:

1. Define low, high, and selected final-weight constants, plus the 19,200-step curriculum duration.
2. Add five reward terms with the original names (`joint_torque`, `joint_velocity`, `joint_acceleration`, `action_rate`, and `base_orientation`), selected final weight, original thresholds, original selectors, and curriculum duration.
3. Comment out or otherwise deactivate the corresponding five `ConstraintTerm` entries.
4. Deactivate `hip_position` and `no_move`.
5. Deactivate the six CaT curriculum entries for joint torque, joint velocity, joint acceleration, action rate, hip position, and base orientation.
6. Leave hard constraints, tracking rewards, power reward, power curriculum, and terrain curriculum unchanged.

The existing task continues to use `CaTEnv` because the hard constraints remain.

## Reward Implementation

In `tasks/utils/mdp/rewards.py`:

- Add one small internal helper that validates a positive normalization threshold and curriculum duration, clamps signed violations below at zero, normalizes by the threshold, reduces vector terms with `max(dim=1)`, applies progress from `env.common_step_counter`, and returns a negative value.
- Add five named reward functions. Each named function calls its matching function in `tasks/utils/cat/constraints.py` and passes the result to the helper.
- Do not duplicate formulas for torque, velocity, acceleration, action rate, or projected gravity.

This limited helper is justified by the need to enforce identical normalization, aggregation, schedule, and sign across all five transferred constraints.

## Evaluation Compatibility

After the soft terms leave `env_cfg.constraints`, the current evaluator would load only the remaining hard limits and would not activate its all-or-nothing fallback. It would therefore lose the four supported soft-limit metrics.

`scripts/eval.py::load_constraint_bounds()` will inspect both `constraints` and `rewards` in the saved training `env.yaml`. Because the new reward term names and `limit` parameters match the original constraint terms, the evaluator will recover the same torque, velocity, acceleration, action-rate, and base-orientation thresholds. Existing handling of hard foot-force and joint-position limits remains unchanged.

The existing metric utility currently ignores base orientation even when its bound is loaded. Expanding the published evaluation metric is outside this implementation; the reward still trains against the exact base-orientation quantity and threshold.

## Validation and Error Handling

The helper will reject nonpositive thresholds and nonpositive curriculum durations. Tests will verify:

- zero penalty below and exactly at a threshold;
- normalized values above a threshold;
- maximum reduction over joint components;
- scalar base-orientation behavior;
- negative reward sign;
- progress at environment steps 0, 9,600, 19,200, and after saturation;
- each wrapper delegates to the corresponding existing constraint function;
- exact configured thresholds and joint/body selectors;
- the five soft terms and two legacy terms are absent from the active constraint configuration;
- only the four hard CaT terms remain active;
- obsolete CaT curriculum terms are inactive;
- tracking, power, action scale, PPO configuration, and other experiment settings are unchanged;
- the evaluator recovers bounds from reward entries in a representative saved configuration.

A small Isaac Lab smoke run will instantiate the existing task, inspect active manager terms, and run a short PPO training job. The smoke run must show five normalized reward penalties, four retained hard constraints, no soft CaT terms, and successful reward/curriculum execution.

## Curriculum Diagnostics

The configured reward weight in `params/env.yaml` is the final profile weight, while
the realized `Episode_Reward/*` values mix violation magnitude, curriculum progress,
reward weight, and reward time-step scaling. Neither is sufficient to directly audit
the 800-iteration ramp. The CleanRL loop will therefore emit curriculum diagnostics
once after every 24-step PPO rollout, using the same training-iteration index passed
to the W&B/TensorBoard writer.

The logger is guarded by the presence of the five transferred reward terms, so other
tasks retain their existing output. For the reward-approximation task it reads each
live reward-term configuration and `env.common_step_counter`, computes

\[
p(k)=\min(k/K,1), \qquad w_i^{\mathrm{effective}}=w_i^{\mathrm{configured}}p(k),
\]

where `K` comes from each term's serialized `curriculum_steps` parameter. It logs:

- `Curriculum/soft_constraint_common_step_counter`;
- `Curriculum/soft_constraint_progress`;
- `Curriculum/joint_torque_effective_weight`;
- `Curriculum/joint_velocity_effective_weight`;
- `Curriculum/joint_acceleration_effective_weight`;
- `Curriculum/action_rate_effective_weight`;
- `Curriculum/base_orientation_effective_weight`.

The same values are printed in one flushed stdout line per PPO iteration. The line
contains the PPO iteration, common step counter, shared progress, and all five
effective weights. This permits cluster logs and W&B to be checked independently.
For the low profile, the rollout-end effective weights must be `0.05` at iteration
400 and `0.1` at iteration 800; for the high profile they must be `5.0` and `10.0`.
The diagnostic performs no manager mutation and does not affect rewards, rollouts,
optimizer state, or scheduling.

## Run Procedure

For the low experiment:

1. Assign `SOFT_CONSTRAINT_REWARD_END_WEIGHT` to the named low constant.
2. Use a low-profile-specific `ENV_NAME`.
3. Launch the existing `CaT-Go2-Rough-Terrain-v0` training command for every requested seed.
4. Confirm the saved `env.yaml` records weight `0.1` for all five reward terms.

For the high experiment, change only the active assignment to the named high constant, use a distinct `ENV_NAME`, and repeat. Confirm the saved weight is `10.0`.

The code changes themselves do not launch the full multi-seed experiments. Full training outcomes determine whether the intended weak-penalty and exploration-suppression failure modes actually occur.

## Out of Scope

- Tuning a balanced or optimal reward coefficient.
- Reimplementing CaT probabilities inside rewards.
- Changing hard-CaT behavior.
- Correcting discrepancies in the paper's description of hard resets.
- Adding another Gym task or generalized experiment framework.
- Changing PPO optimization/update behavior, action scaling, power minimization, perception, terrain, commands, randomization, or training budget.
- Claiming experimental success from a startup smoke test alone.
