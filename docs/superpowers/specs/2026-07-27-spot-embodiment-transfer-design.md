# Boston Dynamics Spot Embodiment Transfer Design

**Date:** 2026-07-27
**Branch:** `cross-embodiment`
**Status:** Approved design; implementation not started

## Objective

Add a trainable CaT rough-terrain environment for Boston Dynamics Spot and complete the existing Spot evaluation support. The result must test the same locomotion method already demonstrated on Unitree Go2 and ANYmal C, while changing only settings that Spot's embodiment or actuator model requires.

The primary scientific outputs remain:

1. Achieved terrain curriculum level.
2. Cost of transport.
3. Body-frame x/y/yaw RMS velocity-tracking error.

Their definitions and the rough-terrain curriculum must remain unchanged so results are directly comparable across embodiments.

## Governing requirements

- Reuse the working Go2 CaT MDP instead of creating a separate Spot locomotion implementation.
- Keep Go2 and ANYmal behavior unchanged.
- Treat current branch history as authoritative. In particular, ANYmal only learned after its torque, velocity, and acceleration constraints were relaxed from `60/7/300` to `80/12/600`.
- Use the installed Isaac Lab 2.3.1 / Isaac Sim 5.1 source and runtime as the authority for Spot names, dimensions, physical properties, actuator behavior, and APIs.
- Keep observation noise and training disturbances exactly equal to the successful ANYmal configuration.
- Preserve historical Go2 evaluation compatibility. Unknown old runs continue to default to Go2.
- Avoid sim-to-real work.
- Do not start repeated or multi-day training runs during implementation.
- Use focused source inspection, configuration checks, simulator smokes, a short PPO startup, and early-training diagnostics.
- Commit logical increments on `cross-embodiment` without switching branches.
- Record verified facts, starting assumptions, observed failures, and revised values in a living Spot tuning document.

## Alternatives considered

### Selected: narrow subclass of the Go2 CaT environment

Create a Spot-specific subclass of `Go2RoughTerrainEnvCfg`, just as ANYmal uses a narrow subclass. Preserve the common MDP and replace only embodiment-dependent asset, naming, timing, geometry, action, constraint, randomization, and energy parameters.

This gives the strongest transfer claim and minimizes regression risk.

### Rejected: subclass the ANYmal configuration

This would inherit the desired disturbances but also inherit ANYmal's uppercase naming, LSTM-actuator assumptions, action scale, physical timestep, and constraint semantics. The dependency would be misleading and fragile.

### Rejected: extract a new embodiment-neutral base class

This would be cleaner architecturally but would refactor already validated Go2 and ANYmal configurations during an ablation campaign. The scientific benefit does not justify the regression risk.

## Installed embodiment facts

The following values were verified from installed source and live runtime articulation data.

| Property | Go2 | ANYmal C | Spot |
|---|---:|---:|---:|
| Total default mass | 15.018999 kg | 52.13484 kg | 31.60000 kg |
| Root body | `base` | `base` | `body` |
| Default root height | 0.4 m | 0.6 m | 0.5 m |
| Action dimension | 12 | 12 | 12 |
| Shared control period | 0.02 s | 0.02 s | 0.02 s |
| Physics period | 0.005 s | 0.005 s | 0.002 s upstream |
| Actuator model | DC motor | ANYdrive LSTM | delayed PD hips; remotized PD knees |
| Nominal effort model | 70 N m training motor | 80 N m continuous, 120 N m saturation | 45 N m hips; angle-dependent knees |
| Installed action scale/reference | 0.8 rad CaT | 0.5 rad upstream/CaT | 0.2 rad upstream |

Spot's live runtime joint order is:

```text
fl_hx, fr_hx, hl_hx, hr_hx,
fl_hy, fr_hy, hl_hy, hr_hy,
fl_kn, fr_kn, hl_kn, hr_kn
```

Spot's live body order is:

```text
body,
fl_hip, fr_hip, hl_hip, hr_hip,
fl_uleg, fr_uleg, hl_uleg, hr_uleg,
fl_lleg, fr_lleg, hl_lleg, hr_lleg,
fl_foot, fr_foot, hl_foot, hr_foot
```

Its default joint positions are:

- Hip-x: left `+0.1`, right `-0.1` rad.
- Hip-y: front `0.9`, hind `1.1` rad.
- Knees: `-1.5` rad.

Verified soft position limits are approximately:

- Hip-x: `[-0.7854, 0.7854]` rad.
- Hip-y: `[-0.8988, 2.2951]` rad.
- Knee: `[-2.7929, -0.2471]` rad.

The hip actuator effort is capped at 45 N m. Knee torque is interpolated from installed experimental data and varies from about 31 N m near the extended limit to about 113 N m near the default pose. The USD/runtime velocity limits of 100 rad/s for hips and infinity for knees are simulation bounds, not suitable CaT safety constraints.

Spot's actuator delays are expressed in physics steps with a configured range of 0–4. The custom environment must therefore retain Spot's 0.002 s physics period and use decimation 10. This preserves the shared 0.02 s control period and the intended 0–8 ms actuator delay. Using the Go2/ANYmal 0.005 s physics period would silently turn the delay into 0–20 ms.

## Environment architecture

Create:

- `SpotRoughTerrainEnvCfg`
- `SpotRoughTerrainEnvCfg_PLAY`
- `CaT-Spot-Rough-Terrain-v0`
- `CaT-Spot-Rough-Terrain-Play-v0`

The training class subclasses `Go2RoughTerrainEnvCfg`. It replaces:

- The robot with installed `SPOT_CFG`.
- Joint/action/observation names with the explicit runtime order.
- Root and body selectors.
- Foot selectors.
- Action scale.
- Physics timestep and decimation.
- Constraint values and role mappings.
- Root mass randomization.
- Energy endpoint.

The play class subclasses the custom Spot training class rather than the installed upstream Spot play class. The installed `SpotFlatEnvCfg_PLAY` ends after a comment about removing pushes and does not actually remove them, so relying on it would make evaluation behavior version-dependent.

## Shared behavior

The following stay identical to the current CaT Go2/ANYmal method:

- Rough terrain generator, proportions, difficulty definition, and terrain curriculum.
- Initial terrain level and episode length.
- Velocity command distribution and deadzone.
- Observation term semantics, order, dimension, scaling, and noise.
- Height-map construction and noise.
- Tracking rewards and their weights/standard deviations.
- Constraint probability curricula and timing.
- Energy curriculum timing.
- Reset-root event semantics.
- Joint reset scaling of 0.95–1.05.
- Action-rate definition and bound.
- Friction randomization.
- COM randomization.
- Evaluation scenarios and primary metric definitions.

No upstream Spot gait reward, gait timing, cobblestone terrain, 48-dimensional observation layout, reset velocity distribution, or standalone Spot reward stack is imported.

## Noise and disturbance decision

Spot training uses the successful ANYmal values exactly:

| Setting | Value |
|---|---:|
| Direct x/y velocity push | `[-0.25, 0.25]` m/s |
| External force | `[-10, 10]` N |
| External torque | `[-0.5, 0.5]` N m |
| Push event timing | Identical to ANYmal |
| Wrench event timing | Identical to ANYmal |
| Observation noise | Identical terms, distributions, scales, and enablement |

Training disturbances stay active. The play/evaluation class disables both disturbance events, matching Go2 and ANYmal evaluation behavior. Evaluation observation corruption remains enabled with the common CaT noise configuration.

## Starting numerical configuration

| Category | Parameter | Go2 | Successful ANYmal C | Spot starting value |
|---|---|---:|---:|---:|
| Action | Position target scale | 0.8 rad | 0.5 rad | 0.2 rad |
| Soft constraint | Applied torque | 20 N m | 80 N m | 80 N m |
| Soft constraint | Joint velocity | 25 rad/s | 12 rad/s | 20 rad/s |
| Soft constraint | Joint acceleration | 800 rad/s^2 | 600 rad/s^2 | 800 rad/s^2 |
| Soft constraint | Normalized action rate | 80 s^-1 | 80 s^-1 | 80 s^-1 |
| Hard constraint | Foot contact force | 300 N | 1000 N | 800 N |
| Hard constraint | Flexion joint absolute position | 1.5 rad | 1.5 rad | 2.0 rad on `.*_hy` |
| Style constraint | Ab/adduction relative position | 0.3 rad | 0.3 rad | 0.3 rad on `.*_hx` |
| Standstill constraint | Joint velocity | 4 rad/s | 2 rad/s | 2 rad/s |
| Domain randomization | Added root mass | +/-1.5 kg | +/-5 kg | +/-3 kg |
| Energy reward | Curriculum endpoint | 0.008 | 0.00230 | 0.00380 |
| Simulation | Physics timestep | 0.005 s | 0.005 s | 0.002 s |
| Simulation | Decimation | 4 | 4 | 10 |

### Torque

The existing method and diagnostics use one scalar torque term. Spot hips are physically capped at 45 N m, while knee capacity depends strongly on angle. An 80 N m starting constraint cannot activate for the hips but gives the knees learning margin without using their approximately 113 N m maximum near the default pose.

Splitting hip and knee torque constraints before locomotion is learned would expand the method and diagnostics unnecessarily. Separate role-specific torque constraints are a later option only if measured distributions show that the scalar term invalidates the ablation.

### Velocity

The starting constraint is 20 rad/s. It will not be automatically tightened to 12 rad/s. Installed limits of 100 rad/s/infinity are not used. The value may be increased only if early evidence shows persistent velocity pressure preventing locomotion.

### Acceleration

The starting constraint is 800 rad/s^2. Spot's 500 Hz simulation and delayed/remotized controllers can produce larger sampled acceleration than ANYmal. If it dominates before tracking begins, the first relaxation candidate is 1200 rad/s^2. Tightening is considered only after locomotion is stable.

### Foot force

The starting hard threshold is 800 N, as explicitly selected for learning margin. It is more permissive than a strict body-mass scaling from Go2/ANYmal and is intended to prevent ordinary early landings from terminating the run.

### Hip-y position

The absolute hard bound is 2.0 rad on all `.*_hy` joints. This gives meaningful margin over the 0.9/1.1 rad default pose while remaining below the approximately 2.295 rad soft physical upper limit.

### Illegal contacts

Use `body` and `.*_uleg`. These correspond to the base and upper-leg/thigh semantics constrained on Go2 and ANYmal. Lower-leg contacts remain allowed on rough terrain. Feet are handled by the contact-force constraint.

### Mass randomization

Use +/-3 kg on `body`, about 9.5% of total mass. This matches the relative magnitude used by Go2 and ANYmal.

### Energy endpoint

The initial endpoint is 0.00380. The established configurations reveal an almost exact inverse-mass rule:

```text
Go2:       0.00800 * 15.018999 kg = 0.120152
ANYmal C:  0.00230 * 52.134840 kg = 0.119910
Spot:      0.00380 * 31.600000 kg = 0.120080
```

The documented energy bracket is:

- Lower: 0.00190.
- Baseline: 0.00380.
- Upper: 0.00760.

The bracket is used only after basic locomotion evidence distinguishes excessive energy pressure from an irrelevant energy term.

## Timing and sensor updates

The Spot subclass calls the shared Go2 post-initialization, then sets:

```text
sim.dt = 0.002
decimation = 10
sim.render_interval = 10
```

After changing `sim.dt`, every active contact/ray sensor update period must be explicitly reset to 0.002 s. This avoids inheriting the 0.005 s value assigned by the Go2 parent before the Spot override.

The 10-second episode and 50 Hz environment step remain unchanged.

## Play and evaluation configuration

The custom Spot play class:

- Uses one environment and spacing 8.
- Uses exact default Spot joint positions on reset.
- Disables direct velocity pushes and external wrench disturbances.
- Keeps common CaT observation corruption enabled.
- Adds four one-ray foot sensors for `fl_foot`, `fr_foot`, `hl_foot`, and `hr_foot`.
- Adds a frame transformer sourced from `body`.
- Uses the same forced-terrain/evaluation terrain configuration as Go2 and ANYmal.
- Disables the training power reward and its curriculum.

The existing `RobotEvalProfile` for Spot already defines:

- Root: `body`.
- Feet: `fl_foot`, `fr_foot`, `hl_foot`, `hr_foot`.
- Spawn height: 0.5 m.
- Joint roles: `hx`, `hy`, `kn`.

The zero Spot sole offset remains an explicit starting assumption. It may bias secondary foot-clearance/step-height metrics, but it does not change terrain level, CoT, or RMS tracking error.

Evaluation must use the explicit task:

```bash
just eval /absolute/path/to/run \
  --eval_checkpoint=<checkpoint> \
  --task=CaT-Spot-Rough-Terrain-Play-v0
```

The evaluator continues loading action scale and constraint bounds from the saved training `params/env.yaml`. Runtime total mass continues to determine CoT. The terrain-level and tracking-error formulas are unchanged.

## Plotting support

The current joint-layout helper recognizes Go2 and ANYmal role strings but not Spot. Extend its existing synonym sets:

- Column 0: add `hx`.
- Column 1: add `hy`.
- Column 2: add `kn`.

Add one focused Spot layout check beside the existing Go2/ANYmal checks. This is justified because the current function deterministically raises for every Spot joint name and would make otherwise successful evaluation plot generation fail after a long evaluation.

## Training and evaluation data flow

1. `just train` resolves the custom Spot Gym task.
2. The Spot subclass builds the shared CaT scene/MDP with Spot-specific names, timing, and values.
3. CleanRL receives the same semantic 188-dimensional observation vector and 12 normalized actions.
4. Constraint diagnostics derive names from active terms, so no task-name special case is required.
5. Training dumps the resolved Spot configuration, including action scale, actuator asset, constraints, disturbances, and energy endpoint.
6. `eval.py` is invoked with the custom Spot play task and the saved checkpoint.
7. Evaluation restores the saved action scale and hard constraint bounds, records physical telemetry, and computes unchanged primary metrics.
8. Plot generation maps Spot joint names into the existing four-leg/three-role layout.

## Validation strategy

### Static and configuration validation

- Confirm branch is `cross-embodiment` before each commit.
- Confirm train/play task registration.
- Assert the exact joint/action/observation order.
- Assert action scale 0.2.
- Assert physics period 0.002 s, decimation 10, and control period 0.02 s.
- Assert all active sensor periods are 0.002 s.
- Assert the agreed constraint, mass, energy, noise, and disturbance values.
- Assert Spot noise and disturbances equal the current ANYmal configuration.
- Assert Go2 and ANYmal resolved configurations remain unchanged.
- Verify Python compilation and `git diff --check`.

### Runtime construction smoke

With one environment and a small terrain generator:

- Construct/reset train and play tasks.
- Confirm 12 actions and 188 observations.
- Confirm runtime joint/body/foot names and 31.6 kg total mass.
- Confirm delayed hip and remotized knee actuator classes remain active.
- Confirm the frame transformer resolves `body` and four feet.
- Step at least 25 times.
- Require finite observations, rewards, constraint probabilities, applied torques, joint velocities, joint accelerations, contact forces, and integrated power.
- Confirm play contains neither disturbance event.

GPU and simulator commands must use the approved elevated execution boundary. `nvidia-smi` succeeds there on the RTX 2080 Ti with driver 535.288.01. The earlier restricted-sandbox `nvidia-smi` failure was not a host-driver failure. A separate optional 64-environment upstream random-response probe crashed natively and produced no usable evidence; it does not alter the design values.

### Plotting check

Run the existing focused joint-layout tests with the new Spot case. Do not expand into a broad unit-test suite.

### Short PPO startup

Run a real 64-environment, two-iteration Spot CleanRL startup:

- Record seed, commit, resolved values, and environment dimensions.
- Require completed rollouts and policy updates.
- Require finite losses, action standard deviation, rewards, and constraint telemetry.
- Record termination sources.
- Record hip and knee torque separately.
- Record velocity, acceleration, action-rate, foot-force, hip-y, and energy distributions.
- Confirm the initial energy ramp is secondary to tracking.

This startup is not evidence of locomotion convergence and must not trigger value changes solely from random-policy transients.

## First substantive training run

After all startup checks pass, run exactly:

```bash
just train 4096 CaT-Spot-Rough-Terrain-v0 46
```

This run is necessary because construction and optimizer compatibility do not demonstrate that the CaT method learns with Spot's asymmetric actuator model.

### First 24-hour signals

- Finite actor/critic losses and action standard deviation.
- Mean/min/max episode duration.
- Reset fractions by timeout, illegal contact, upside-down, foot force, and hip-y position.
- Tracking reward and available tracking-error proxies.
- Terrain-level mean/distribution and progression.
- Per-constraint violation frequency/probability and episode maxima.
- Hip versus knee torque distributions.
- Velocity, acceleration, and normalized action-rate distributions.
- Foot-force peaks and their relationship to resets.
- Raw joint power, episode energy, and energy reward contribution relative to tracking.
- Action saturation and representative videos.
- Exact commit, seed, environment count, energy endpoint, and any deviation from baseline.

### Continue criteria

Continue the run when:

- All numerical signals remain finite.
- Episode survival and tracking improve.
- Terrain levels are not pinned at zero and show progression.
- Hard termination rates decline from random-policy initialization.
- No single soft constraint permanently dominates.
- The energy term remains secondary while locomotion is being established.

### Stop criteria

Stop the run when:

- NaN/Inf appears.
- Immediate-reset loops persist.
- The policy collapses or saturates without improvement.
- Tracking stays flat and terrain remains pinned at zero.
- Ordinary landings repeatedly trigger the 800 N termination.
- Normal poses repeatedly reach the 2.0 rad hip-y termination.
- Torque, velocity, acceleration, or energy pressure clearly blocks tracking progress.

Any revision must change one parameter family at a time and record the evidence. No automatic reduction of the 20 rad/s velocity constraint is planned.

## Error handling and early failure detection

- Entity selectors must fail during construction with descriptive missing-name errors.
- Runtime assertions verify actuator classes, order, dimensions, mass, and timing before PPO startup.
- Sensor update periods are checked after the Spot timestep override.
- Play disturbances are checked explicitly because the installed upstream play configuration is incomplete.
- Saved training parameters remain the authority for evaluation action scale and hard bounds.
- A failed native simulator probe is treated as infrastructure evidence, not as a locomotion or constraint result.
- Runtime verification is never claimed from static inspection alone.

## Documentation updates during implementation

Create `docs/spot_tuning_assumptions.md` containing:

- Installed verified facts.
- Every starting value and derivation.
- The successful current ANYmal `80/12/600` values, correcting the stale original rows in the ANYmal tuning document.
- Spot-specific early failure signals and next candidates.
- Smoke/startup evidence.
- Every later training-driven revision.

Also update `docs/anymal_c_tuning_assumptions.md` so its main comparison table distinguishes the original `60/7/300` starting values from the successful `80/12/600` configuration.

## Success conditions

Implementation success requires:

- Registered Spot train/play CaT tasks.
- Correct Spot asset, names, actuator model, timing, constraints, noise, and disturbances.
- Successful finite train/play construction smokes.
- Successful short PPO startup.
- Working Spot evaluation/plot configuration without changing primary metric definitions.
- No Go2 or ANYmal configuration regression.
- Clean, incremental commits and current tuning documentation.

Scientific transfer success requires a later substantive Spot run that learns stable tracking, progresses through the unchanged terrain curriculum, and produces valid CoT and RMS metrics. The implementation phase alone cannot claim that outcome.

## Failure conditions

The design is not considered successfully implemented if:

- Spot uses the Go2/ANYmal 0.005 s physics period and changes its actuator delay.
- Observation noise or training disturbances differ from the successful ANYmal values.
- The upstream Spot reward/observation/terrain stack replaces the shared CaT method.
- Terrain-level, CoT, or RMS definitions change.
- Evaluation requires changes that break historical Go2 behavior.
- Runtime construction or short PPO startup remains non-finite or unresolved.
- A full training run is recommended without explicit first-24-hour gates.
- Sim-to-real code is modified.
