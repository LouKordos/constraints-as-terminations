# ANYmal C Starting Values and Tuning Register

This is the living record for the first ANYmal C transfer. It separates facts verified against the installed Isaac Lab 2.3.1 runtime from learning-first assumptions. Update it whenever smoke-test or training evidence changes a value. Change one parameter family at a time unless a construction error proves several values invalid together.

## Verified embodiment facts

| Item | Installed value | Source / consequence |
|---|---:|---|
| Isaac runtime | Isaac Lab 2.3.1, Isaac Sim 5.1 | Installed environment on this machine; its APIs and assets are authoritative. |
| Asset | `ANYMAL_C_CFG`, ANYdrive 3 LSTM | `isaaclab_assets.robots.anymal`; do not replace with an ideal PD actuator for this ablation. |
| Total default mass | 52.13484 kg | Runtime articulation data. CoT uses runtime total mass, not a hardcoded value. |
| Base mass | 26.373 kg | Runtime articulation data. |
| Root body | `base` | Used for randomization, external wrench, frame transformer, and illegal contact. |
| Feet | `LF_FOOT`, `RF_FOOT`, `LH_FOOT`, `RH_FOOT` | Runtime body names, ordered as front-left, front-right, rear-left, rear-right for metrics. |
| Joint order | `LF_HAA`, `LH_HAA`, `RF_HAA`, `RH_HAA`, `LF_HFE`, `LH_HFE`, `RF_HFE`, `RH_HFE`, `LF_KFE`, `LH_KFE`, `RF_KFE`, `RH_KFE` | Runtime articulation order; actions and proprioceptive observations use this exact order. |
| Default joint pose | HAA 0; front HFE +0.4; hind HFE -0.4; front KFE -0.8; hind KFE +0.8 rad | Installed asset initial state. |
| Default spawn height | 0.6 m | Installed asset initial state. |
| Continuous actuator effort | 80 N m | Installed ANYdrive actuator configuration. This is not used as the CaT constraint. |
| Saturation effort | 120 N m | Installed ANYdrive actuator configuration. This is not used as the CaT constraint. |
| No-load actuator speed | 7.5 rad/s | Installed ANYdrive actuator configuration. The initial constraint is lower. |
| Policy dimensions | 12 actions, 188 normal observations | Must be reconfirmed by the construction smoke test. The 188 terms match Go2 semantically. |
| Control period | 0.02 s | Shared 0.005 s simulation step and decimation 4. |

## Starting assumptions

| Category | Parameter | Go2/reference | ANYmal C initial | Why this is the initial value | Early failure signal | Next evidence-driven candidate |
|---|---|---:|---:|---|---|---|
| Action | position-target scale | 0.8 rad | 0.5 rad | Installed upstream ANYmal locomotion scale; preserves the same normalized-action interface. | Policy saturates actions without tracking, or cannot generate sufficient foot motion. | Inspect target/joint range before trying 0.4 or 0.6. |
| Soft constraint | torque | 20 N m | 60 N m | Below 80 N m continuous effort, with learning margin; roughly reflects the larger robot without using actuator maximum. | Torque probability dominates before tracking improves, or applied torque sits at the constraint continuously. | Relax only if construction/runtime disproves it; after locomotion exists, tighten to 50 then 40 N m. |
| Soft constraint | joint velocity | 25 rad/s | 7.0 rad/s | Below the 7.5 rad/s no-load speed, but not the strictest proportional bound during initial learning. | Persistent velocity violations with otherwise useful motion, or actuator network saturation. | About 6.25 rad/s after stable locomotion. |
| Soft constraint | joint acceleration | 800 rad/s^2 | 300 rad/s^2 | Learning-first margin while still much lower than Go2. | Acceleration probability dominates from policy noise and prevents longer episodes. | 200 rad/s^2 after stable locomotion. |
| Soft constraint | normalized action rate | 80 s^-1 | 80 s^-1 | Defined on normalized actions per control second, so it is not scaled by robot joint speed. | Chattering and high energy without violations, or constant action-rate violations. | Reassess normalization and measured distribution before changing. |
| Hard constraint | foot contact force | 300 N | 1000 N | Approximately preserves bound/body-weight scale. | Normal stance/landing contacts terminate most episodes, or damaging impacts are never caught. | Inspect per-foot distribution; adjust by measured quantiles, not actuator effort. |
| Hard constraint | absolute HFE position | 1.5 rad | 1.5 rad | Angular geometry check remains plausible, but MUST be validated against ANYmal pose/sign convention. | Resets at nominal or ordinary swing poses. | Derive a per-joint bound from safe joint geometry. |
| Style constraint | relative HAA position while moving | 0.3 rad | 0.3 rad | Same angular role and normalized command condition. | Prevents necessary lateral stabilization on rough terrain. | Inspect HAA distributions before widening. |
| Standstill | joint speed | 4 rad/s | 2 rad/s | Relaxed first-run value while reflecting slower actuators. | Standstill environments dominate CaT probability. | 1 rad/s after locomotion/standing is stable. |
| Termination | illegal bodies | base and thighs | `base`, `.*_THIGH` | Direct semantic body-role transfer. | Ordinary knee/shank contacts wrongly continue, or thigh brushing causes excessive resets. | Change only from contact/video evidence. |
| Reset | joint pose scale | 0.95-1.05 | 0.95-1.05 train; exact defaults in play | Keeps shared train randomization and avoids Go2's eval-only crouch. | Immediate self-collision/fall on reset. | Reduce training pose spread; do not copy Go2 pose values. |
| Domain randomization | added base mass | +/-1.5 kg | +/-5 kg | Similar fraction of total mass and within upstream ANYmal scale. | Startup instability or inertia mismatch dominates early learning. | Reduce to +/-2.5 kg until locomotion exists. |
| Disturbance | direct x/y velocity push | +/-0.5 m/s | +/-0.25 m/s | Explicitly reduced for development and early learning. | Fall/reset spikes synchronized with pushes. | Disable temporarily to isolate learning; increase only after stable locomotion. |
| Disturbance | external force | +/-10 N | +/-10 N | Same absolute force is substantially weaker acceleration on heavier ANYmal. | Fall/reset spikes synchronized with wrench application. | Disable temporarily; increase one disturbance family only after stable locomotion. |
| Disturbance | external torque | +/-0.5 N m | +/-0.5 N m | Same absolute torque is substantially weaker on ANYmal. | Fall/reset spikes synchronized with wrench application. | Disable temporarily; increase only after stable locomotion. |
| Randomization | COM shift | +/-0.03 m each axis | +/-0.03 m | Plausible absolute uncertainty; intentionally unchanged initially. | Startup instability correlated with randomized environments. | Halve the range until locomotion exists. |
| Randomization | friction | 0.5-1.25 | 0.5-1.25 | Shared terrain/contact assumption and not forced by embodiment naming. | Learning separates into slipping and non-slipping modes. | Narrow temporarily, then restore after stable locomotion. |
| Energy reward | curriculum endpoint | 0.008 | 0.00230 | Starting estimate intended to give comparable optimization pressure despite larger raw power; not a final physical constant. | Energy reward overwhelms tracking/terrain learning or is numerically irrelevant. | Bracket 0.00115 and 0.00461 in short controlled runs after basic locomotion. |
| Energy reward | ramp | 0 to endpoint over 300,000 env steps | unchanged | Isolates endpoint scaling from curriculum timing. | Energy pressure arrives before any tracking behavior exists. | Change timing only after separating weight from timing effects. |
| Reward | tracking weights/std | 1.0 linear, 0.5 yaw, variance 0.25 | unchanged | Core method comparison; no robot difference proves a required change. | No tracking learning even with non-dominant constraints and energy term. | Reconsider only after actuator/action/termination causes are ruled out. |
| Observation | terms/scales/noise | 188 normal observations | unchanged semantics and scales | Maintains method comparability; names/order are replaced explicitly. | One term has implausible magnitude or observation shape/order differs. | Normalize only the offending physical term with recorded evidence. |
| Terrain | generator/curriculum | shared rough terrain, initial max level 1, 10 s episodes | unchanged | Required for direct terrain-level comparability. | Spawn/reset geometry invalid for the taller robot or no progress despite flat tracking. | Fix robot-height/reset geometry without changing curriculum definition. |
| Evaluation geometry | sole offset | Go2 0.0228 m | 0.0 m initially | No verified ANYmal toe/sole offset has been established; zero avoids copying Go2 geometry. | Foot clearance/contact-frame height is biased. | Measure collision/foot-frame vertical offset in runtime and record it here. |
| Evaluation | fixed scenario z | 0.4 m | 0.6 m | Matches the installed default root height while retaining identical x/y/orientation/scenarios. | Manual resets spawn above/below valid standing height. | Derive from default root pose plus measured terrain height. |

## First-run tuning order

1. Fix construction facts first: missing names, dimensions, reset height, self-collision, non-finite signals, or actuator/runtime errors.
2. If locomotion does not begin, inspect hard termination sources and the distributions of torque, velocity, acceleration, action rate, and contact force before changing bounds.
3. If one soft constraint dominates, relax only that constraint enough to establish locomotion; preserve the original value and evidence here.
4. If tracking begins but energy pressure prevents terrain progress, test the lower energy bracket. If energy is irrelevant and motion is wasteful, test the upper bracket.
5. Tighten actuator constraints and increase disturbances only after the baseline learns stable tracking and terrain progression.
6. Do not tune Spot values from ANYmal results. Spot needs its own physical audit when training support becomes the priority.

## First 24-hour evidence to record

- Training iteration, wall time, finite actor/critic losses, action standard deviation, and any action saturation.
- Mean/min/max episode length and the fraction of resets by timeout, illegal contact, upside-down, and each hard constraint.
- Tracking reward and available x/y/yaw tracking-error proxies.
- Mean and distribution of terrain levels over time; keep the curriculum definition unchanged.
- Per-constraint violation frequency/probability plus episode maxima for torque, joint velocity, acceleration, action rate, and foot contact force.
- Episode energy, mean absolute raw joint power, and energy reward contribution relative to tracking reward.
- Videos near startup and at representative early checkpoints.
- The exact Git commit, seed, number of environments, energy override, and any configuration deviation.

Continue the long run only when signals remain finite, tracking and episode survival improve, terrain is not pinned at level zero, no single constraint permanently dominates, and energy pressure remains secondary to establishing locomotion. Stop for NaN/Inf, repeated immediate resets, flat tracking with terrain pinned at zero, persistent policy saturation/collapse, pervasive hard-contact/force termination, or a constraint/energy term that blocks tracking progress.
