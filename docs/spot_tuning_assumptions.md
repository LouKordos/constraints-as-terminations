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
| Dimensions | 12 actions, 188 policy observations | Verified by the runtime smoke. |
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

On 2026-07-27, static train/play configuration checks passed for the agreed values, exact ANYmal observation-noise and disturbance equality, and unchanged Go2/ANYmal reference values. One-environment train and play tasks each reset and completed 25 zero-action steps in separate fresh Isaac processes with finite 188-dimensional observations, rewards, constraint probabilities, applied torques, velocities, accelerations, and integrated power. Runtime mass was 31.6 kg; action dimension was 12; physics/control periods were 0.002/0.02 s; the installed delayed hip and remotized knee actuators remained active; and the play frame transformer resolved `body` plus four feet. The existing evaluator resolved the custom play task to its Spot profile, root, foot order, 0.5 m spawn height, joint mapping, and 0.2 action scale without requiring an `eval.py` change. This is construction evidence, not convergence evidence.

The first 64-environment PPO startup exposed an Isaac Lab 2.3.1 incompatibility before rollout: its `CircularBuffer.append` advanced indexed assignment expands incorrectly on CUDA when this project enables deterministic PyTorch algorithms. A bare installed `DelayBuffer(4, 64)` reproduced the failure independently of the task. Spot now uses subclasses of the installed delayed and remotized PD actuators that replace only this buffer initialization with an elementwise mask. The 0--4 physics-step random delay, PD/remotized torque behavior, and global deterministic PPO setting are unchanged. The replacement passed heterogeneous-delay and partial-reset checks on CUDA, followed by a real 64-environment deterministic reset and finite step.

The subsequent two-iteration startup completed PPO updates but revealed a shared diagnostic issue: the initial Gym reset divided zero constraint accumulators by an episode length of zero, so all first-window constraint violation/probability logs were `NaN` even though training losses and physical signals were finite. `ConstraintManager.reset` now clamps only this logging denominator to one. A focused initial-reset check changed every affected diagnostic from `NaN` to zero; constraint computations, policy rewards, terminations, curricula, evaluator metrics, and all nonzero-length episode statistics are unchanged.
