# ANYmal C Embodiment Transfer Design

**Date:** 2026-07-10

**Status:** Approved with corrections

## Objective

Demonstrate that the existing Constraints as Terminations locomotion method is not specific to the Unitree Go2 by training and evaluating an ANYbotics ANYmal C policy on the same rough-terrain problem. Preserve the Go2 path closely enough that the existing Go2 evaluation remains a valid regression reference. Prepare `scripts/eval.py` to evaluate Go2, ANYmal C, and Boston Dynamics Spot policies, while deferring a CaT Spot training environment to later work. ANYmal C construction, learning, and evaluation are the priority; Spot support must not delay getting ANYmal locomotion working.

The comparison focuses on:

- achieved terrain curriculum level during training;
- cost of transport (CoT);
- body-frame linear-x, linear-y, and yaw-rate RMS tracking error;
- energy consumption and reset/termination behavior as supporting evidence.

The `sim2real/` tree is explicitly out of scope.

## Existing-system findings

The current training entry point is already Gym-task-driven. `just train` selects `CaT-Go2-Rough-Terrain-v0`, while the recipe accepts a different task as its second positional parameter. The safest ANYmal integration therefore registers new task IDs and keeps Go2 as the default.

The current Go2 environment embeds embodiment assumptions in more than its robot asset:

- twelve explicit Go2 joint names are repeated in actions and observation terms;
- joint and body regular expressions assume Go2 `hip`, `thigh`, `calf`, `foot`, and `base` names;
- reset poses, illegal contacts, foot force constraints, action-rate constraints, and evaluation sensors use Go2 names;
- actuator limits, constraint thresholds, random mass, external force, nominal base height, and foot contact offsets have Go2 physical scales;
- evaluation chooses foot names using a Go2-versus-everything-else conditional;
- evaluation fixed scenarios use Go2's nominal base height;
- CleanRL history-task inference selects Go2 tasks directly;
- upstream fallback constraint limits are Go2-specific and are currently applied even when the task is not Go2;
- gait symmetry in `scripts/metrics_utils.py` directly indexes Go2 joint names.

The installed Isaac Lab 2.3/Isaac Sim 5.1 source provides:

- `ANYMAL_C_CFG`, using the ANYdrive 3 LSTM actuator model, a 0.6 m initial base height, `LF/LH/RF/RH` leg prefixes, `HAA/HFE/KFE` joints, uppercase `*_FOOT` bodies, and a `base` root body;
- `SPOT_CFG`, using delayed/remotized actuators, a 0.5 m initial height, `fl/fr/hl/hr` joint and foot prefixes, and a `body` root body;
- existing upstream manager-based ANYmal C rough-terrain and Spot tasks whose naming and physical parameters are authoritative references.

ANYmal C is selected rather than ANYmal D because Isaac Lab provides the actuator network for ANYmal C. ANYmal D reuses that network, adding an avoidable actuator-model caveat to the embodiment-transfer claim.

## Architectural approach

Create an ANYmal-specific environment module that subclasses the working Go2 configuration and applies narrow embodiment overrides. Do not extract or rename the existing Go2 hierarchy into a new generic framework. This minimizes the code path changed for Go2 and avoids duplicating the roughly 1,100-line configuration.

The implementation may extract a small helper only when the same operation must be performed by both Go2 and ANYmal and leaving it duplicated would risk inconsistent behavior. It must not introduce a general robot plugin framework.

Evaluation will use a three-entry robot profile mapping. A profile contains only data that `eval.py` must know before environment construction or cannot safely derive from runtime state: task aliases, base link, ordered foot links, nominal base height, foot contact offset, and joint-role mappings used by secondary gait metrics. The profile is resolved from the selected task name. Unknown tasks fail explicitly.

## ANYmal C environment

Create:

- `CaT-Anymal-C-Rough-Terrain-v0`
- `CaT-Anymal-C-Rough-Terrain-Play-v0`

Both use the existing `CaTEnv` entry point and CleanRL PPO configuration.

The training configuration inherits without modification:

- the patched `ROUGH_TERRAINS_CFG` object and all terrain proportions;
- terrain difficulty range, rows, columns, seed behavior, and cache behavior;
- `mdp.terrain_levels_vel` and its curriculum enablement logic;
- 5 ms physics timestep, decimation 4, and 20 ms policy timestep;
- command resampling and the Go2 command ranges used by the method;
- tracking reward functions and weights;
- observation ordering and scaling;
- height-map dimensions, grid, and noise model;
- CaT constraint scheduling and reward scaling semantics;
- the CleanRL network and PPO hyperparameters.

The policy remains a 12-action policy. The normal, non-history observation is expected to remain 188-dimensional because ANYmal C also has twelve controlled joints and uses the same 143-element height grid.

The ANYmal joint order follows the installed Isaac Lab convention:

```text
LF_HAA, LH_HAA, RF_HAA, RH_HAA,
LF_HFE, LH_HFE, RF_HFE, RH_HFE,
LF_KFE, LH_KFE, RF_KFE, RH_KFE
```

The same exact ordered list is used by joint actions, joint-position observations, and joint-velocity observations. Action ordering is not inferred from regular-expression sorting.

ANYmal overrides use:

- robot asset: `ANYMAL_C_CFG`;
- base body: `base`;
- feet: `LF_FOOT`, `RF_FOOT`, `LH_FOOT`, `RH_FOOT` in canonical front-left, front-right, rear-left, rear-right metric order;
- undesired-contact bodies: `base` and `.*THIGH`;
- actuated joints: `.*HAA`, `.*HFE`, and `.*KFE`;
- ab/adduction style constraint: `.*HAA`;
- flexion/extension hard-position constraint: `.*HFE`;
- action scale: `0.5`, matching the installed upstream ANYmal locomotion task;
- nominal spawn height: `0.6` m.

Physical constraint and disturbance values must not be copied blindly. Initial values are derived from the installed actuator configuration and the Go2 value's physical meaning:

- use the ANYdrive effort and velocity limits as the starting safety bounds;
- scale contact force and external push force by robot weight rather than copying Go2 Newton values;
- scale additive mass randomization to a comparable fraction of base mass;
- keep angular joint-position ranges in radians when their intended geometric limit is embodiment-independent;
- retain the normalized policy action-rate limit when its implementation operates on normalized actions rather than physical joint velocity.

The first implementation uses these audited defaults:

| Quantity | Go2 | ANYmal C initial | Rationale |
|---|---:|---:|---|
| action scale | 0.8 rad | 0.5 rad | installed upstream ANYmal locomotion setting |
| joint torque constraint | 20 N m | 60 N m | deliberately below the 80 N m continuous limit, but with learning margin before later 50/40 N m trials |
| joint velocity constraint | 25 rad/s | 7.0 rad/s | below the 7.5 rad/s ANYdrive limit, but looser than strict proportional scaling during initial learning |
| joint acceleration constraint | 800 rad/s^2 | 300 rad/s^2 | conservative learning-first value; 200 rad/s^2 is the strict actuator-speed-ratio candidate |
| normalized action-rate constraint | 80 s^-1 | 80 s^-1 | operates on normalized policy actions, not physical target velocity |
| foot contact-force constraint | 300 N | 1000 N | approximately preserves the bound-to-body-weight ratio |
| standstill joint-speed constraint | 4 rad/s | 2 rad/s | relaxed initial value; 1 rad/s is the strict actuator-speed-ratio candidate |
| added base mass | +/-1.5 kg | +/-5 kg | approximately preserves fraction of total mass and agrees with the upstream ANYmal scale |
| direct velocity push | +/-0.5 m/s | +/-0.25 m/s | reduced during initial learning |
| external force | +/-10 N | +/-10 N | retaining the Go2 absolute value makes the acceleration disturbance much weaker on ANYmal |
| external torque | +/-0.5 N m | +/-0.5 N m | retaining the Go2 absolute value makes the angular disturbance much weaker on ANYmal |

The 1.5 rad absolute HFE bound, 0.3 rad relative HAA bound, base-orientation bound, friction ranges, and 3 cm COM displacement remain unchanged initially because they are angular, normalized, or already plausible at both scales. The values above are hypotheses to validate during simulator construction and short training, not tuned final results. Any adjustment and its rationale are recorded with the experiment results.

A living `docs/anymal_c_tuning_assumptions.md` document separates verified robot facts from starting assumptions. For every tunable bound, reward weight, action scale, disturbance, reset choice, and geometry offset, it records the initial value, rationale, expected failure signal, and next candidate. It is updated whenever smoke tests or early training evidence change a value.

The play config uses the same ANYmal asset and naming overrides, disables training disturbances and the energy reward as the Go2 play config does, and creates foot ray casters and the frame transformer using ANYmal links. It must not inherit the Go2-only reset-pose function without replacing it.

## Training entry point

`scripts/clean_rl/train.py` remains generic and task-driven. Add only:

- an optional `--energy_end_weight` override that updates `curriculum.power.params["end_weight"]` before the environment is constructed and before configs are dumped;
- validation and clear logging of task name, resolved robot configuration, action scale, and energy endpoint;
- run metadata sufficient to reproduce which task and energy endpoint produced a checkpoint.

The default remains unchanged:

```bash
just train
just train 4096
```

ANYmal C is launched through the existing recipe:

```bash
just train 4096 CaT-Anymal-C-Rough-Terrain-v0
```

Energy trials may invoke `train.py` directly or extend the recipe flags only if doing so does not change the positional behavior above.

## Evaluation

`scripts/eval.py` supports these profiles:

| Profile | Base | Ordered feet | Nominal height |
|---|---|---|---:|
| Go2 | `base` | `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot` | 0.4 m |
| ANYmal C | `base` | `LF_FOOT`, `RF_FOOT`, `LH_FOOT`, `RH_FOOT` | 0.6 m |
| Spot | `body` | `fl_foot`, `fr_foot`, `hl_foot`, `hr_foot` | 0.5 m |

The profile controls foot sensors, frame transformer source, fixed-scenario spawn heights, foot-height contact offset, task diagnostics, and joint-role mappings. Common evaluation flow, buffers, video generation, scenario commands, and summary construction remain shared.

Go2 fixed-scenario names, commands, coordinates, and summary fields remain unchanged. For the common rough-terrain ANYmal environment, x/y scenario coordinates remain identical and z uses the profile height adjustment. Spot evaluation uses the installed Spot terrain and must label or omit terrain-specific scenarios that do not exist there rather than falsely calling cobblestone terrain "stairs". Common flat command scenarios and the CoT sweep remain comparable.

The installed Spot task's cobblestone terrain-level values are explicitly marked as not comparable to the common CaT rough-terrain curriculum. Operational Spot evaluation support does not count as a direct terrain-curriculum ablation until a later CaT Spot rough-terrain environment uses the same generator and `terrain_levels_vel` setup.

The existing uncommitted action-scale restoration in `eval.py` is preserved. Training action scale is loaded from `params/env.yaml` before `gym.make()` and recorded in `sim_data.npz` and `metrics_summary.json`.

CleanRL history-task inference remains Go2-specific unless an equivalent ANYmal history task is added. The normal 188-dimensional ANYmal policy relies on the explicitly supplied ANYmal play task. Evaluation must not infer an embodiment solely from checkpoint input dimension because all three quadrupeds can have twelve actions and overlapping observation dimensions.

Backward compatibility takes precedence for old runs: when no new embodiment metadata or recognized non-Go2 task is present, evaluation assumes Go2 and retains the existing Go2 task/history inference. ANYmal and Spot require an explicit recognized task or saved embodiment metadata; an old Go2 run must never start failing merely because it lacks fields introduced on `cross-embodiment`.

Upstream hardcoded Go2 constraint thresholds are used only for upstream Go2 tasks. Custom CaT Go2 and ANYmal tasks load bounds from their saved `params/env.yaml`. Spot does not silently receive Go2 bounds.

## Metrics

The following definitions do not change across embodiments:

### Terrain curriculum

Training continues to use `mdp.terrain_levels_vel`. No new terrain generator, promotion threshold, command range, or curriculum schedule is introduced for ANYmal. The achieved training curriculum level is compared from the existing training log. `metrics_summary.json` terrain fields continue to describe the terrain levels encountered during evaluation and are not relabeled as the terminal training curriculum level.

### Cost of transport

CoT remains:

```text
sum over time and joints of abs(applied torque * joint velocity) * dt
------------------------------------------------------------------------
       runtime robot mass * 9.81 * horizontal distance travelled
```

Robot mass is read from the runtime articulation. All twelve actuated joints are included. Reset/teleport repair and horizontal-distance estimation remain unchanged.

### RMS tracking error

Linear x/y RMS error compares the command against body-frame root linear velocity. Yaw RMS error compares commanded yaw rate against body-frame root angular velocity. World-frame velocity is not substituted.

### Gait symmetry

`scripts/metrics_utils.py` receives an explicit canonical leg/joint-role mapping rather than constructing Go2 names. Go2 retains the existing `hip_joint`, `thigh_joint`, and `calf_joint` summary keys and numerical computation. ANYmal maps HAA/HFE/KFE, and Spot maps hx/hy/kn. This secondary change prevents otherwise valid ANYmal and Spot evaluations from crashing after their main metrics have been collected.

## Energy minimization experiments

The Go2 linear energy curriculum endpoint remains exactly `0.008`.

For ANYmal C:

0. Use `0.00230` as the first implementation endpoint. This is the mass-normalized estimate `0.008 * 15.019 / 52.135`, which preserves expected energy-reward pressure when CoT and commanded speed are comparable. The first bracket is `0.00115`, `0.00230`, and `0.00461`.
1. Collect a matched unpenalized or initial-ramp power scale for Go2 and ANYmal C.
2. Compute a center candidate:

   ```text
   ANYmal center weight = 0.008 * Go2 mean absolute joint power / ANYmal mean absolute joint power
   ```

3. Run matched ANYmal trials at `0.5x`, `1x`, and `2x` the center candidate, using the same seed, number of environments, number of iterations, terrain curriculum, and training configuration.
4. Compare achieved terrain curriculum level, CoT, x/y/yaw RMS errors, resets, energy consumption, and the energy reward contribution.
5. Select the lowest-CoT alternative that does not materially reduce terrain progression or tracking performance. Do not select by raw reward alone.

Short trials establish stability and bracket the useful scale; they are not described as convergence proof. If runtime permits a convergence-quality run, its evidence is reported separately.

## Go2 regression reference

Do not train or run a fresh clean Go2 baseline. Use the user-provided reference:

```text
/local_storage1/loukas-backup/logs/clean_rl/cluster_runs/
env_id_107_action_scale_sweep/constraints-as-terminations/logs/clean_rl/
env_id_107_action_scale_sweep/2026-07-03-09-25-54/
eval_checkpoint_21799_seed_46/metrics_summary.json
```

Reference metadata:

- task: `CaT-Go2-Rough-Terrain-Joint-State-History-Play-v0`;
- checkpoint: `model_21799.pt`;
- seed: `46`;
- action scale: `0.8`;
- random steps: `4000`;
- total steps: `14500`;
- overall CoT: `0.4196298480495525`;
- x RMS: `0.32775482535362244`;
- y RMS: `0.1588231772184372`;
- yaw RMS: `0.09639634191989899`;
- mean evaluation terrain level: `3.0406896551724136`.

After implementation, evaluate the same checkpoint with the same task, seed, action scale, random-step count, fixed scenarios, and simulator settings. Compare the complete JSON recursively, with special attention to schema, CoT, tracking RMS, terrain values, reset steps, energy, constraint violations, and scenario blocks. Simulator nondeterminism is reported rather than hidden.

## Verification

No new unit tests are required.

Verification consists of:

- Python syntax and import checks that do not require simulator startup;
- registered-task listing for the new train and play IDs;
- config dump inspection for exact joint/body/action/constraint values;
- a small-environment, short-iteration Go2 training smoke run to ensure the default path still constructs;
- a small-environment, short-iteration ANYmal C training smoke run;
- ANYmal energy-weight trial runs;
- evaluation of an ANYmal trial checkpoint through the new play task;
- Spot environment/evaluation construction against the installed upstream Spot task and an available compatible checkpoint, if one exists;
- the matched Go2 evaluation against the user-provided July 3 baseline.

Implementation is split into small, reviewable commits. Focused syntax/config checks run after each logical change, and simulator construction/reset/step smoke tests run after the ANYmal environment and evaluation integrations rather than only at the end.

A missing compatible Spot checkpoint limits Spot verification to environment construction and metric-path validation. Lack of convergence time limits the strength of the transfer claim but does not excuse configuration or smoke-run failures.

## Success conditions

- Default Go2 training commands and effective Go2 config remain unchanged.
- Terrain generation and `mdp.terrain_levels_vel` behavior are identical between the method's Go2 and ANYmal C tasks.
- ANYmal C constructs with exactly twelve intended actions and the expected observation dimension.
- Every ANYmal joint/body selector resolves to intended entities; none resolves to an empty set.
- ANYmal short training advances without invalid contacts, NaNs, immediate reset collapse, or action/observation mismatch.
- Evaluation produces CoT, body-frame RMS errors, terrain metrics, energy, and reset information for ANYmal C.
- Evaluation constructs and reaches the same metric collection path for Spot without Go2 names or thresholds.
- The Go2 JSON schema remains compatible and differences from the July 3 reference are either absent or explained with evidence.
- The chosen ANYmal energy weight improves the energy/CoT trade-off without materially sacrificing terrain curriculum or tracking.

## Failure conditions

- ANYmal support merely imports but fails during scene/entity resolution.
- Go2 defaults, terrain curriculum, metric definitions, or output schema change silently.
- ANYmal or Spot evaluation uses Go2 feet, base height, toe offset, constraints, or joint symmetry names.
- Checkpoint embodiment is inferred only from tensor dimensions.
- Energy weights are selected without matched alternatives and terrain/tracking comparison.
- Evaluation terrain level is misrepresented as achieved training curriculum level.
- A short smoke run is presented as convergence or reliable-transfer proof.
- Changes enter `sim2real/` or introduce a broad robot framework unrelated to the three requested embodiments.

## Known caveats

- Isaac Sim rollout determinism depends on GPU, driver, PhysX, asset cache, and simulator version in addition to the configured seed.
- ANYmal C's actuator-network dynamics differ substantially from Go2's ideal/DC motor setup, so identical raw energy reward weights are not meaningful.
- CoT is cross-robot normalized by mass and distance, but actuator model fidelity still affects its absolute value.
- A single seed demonstrates operability and comparative behavior, not statistical reliability. Multiple seeds are needed for publication-strength reliability claims.
- The existing Go2 baseline is a joint-state-history policy, while the initial ANYmal task is the normal observation configuration. Results must identify this architecture difference unless an ANYmal history task is later added.
- Spot's installed environment uses a different terrain family. Its evaluation support is operational preparation; a direct method comparison awaits the later CaT Spot rough-terrain environment.
