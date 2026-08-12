# LoComposition Repository Rebrand Design

**Date:** 2026-08-11

## Purpose

The repository has grown far beyond the Isaac Lab port of Constraints as
Terminations (CaT) from which it started. It now contains the LoComposition
training formulation, controlled ablations, evaluation and plotting tools,
zero-shot deployment code for a physical Unitree Go2, and embodiment-transfer
configurations for ANYmal C and Boston Dynamics Spot. The repository should
present that work as LoComposition while continuing to identify and cite CaT
as the mechanism used to encode operational limits.

This is a layered, compatibility-preserving rebrand. LoComposition becomes the
primary project and software identity. Existing commands, task names, imports,
and checkpoints remain usable where preserving them does not misrepresent a
ROS wire contract.

## Goals

- Present LoComposition clearly to researchers, engineers, hiring teams, and
  visitors who reach the repository without first reading the paper.
- Explain the high-level benefit before introducing implementation details:
  natural, energy-efficient, terrain-adaptive locomotion without explicit gait
  priors.
- Keep the README focused on the software release rather than repeating the
  paper, project website, project video, or personal blog post.
- Make LoComposition the primary name in package metadata, Python imports,
  environment IDs, commands, ROS packages, runtime labels, and documentation.
- Preserve the name CaT wherever it denotes the stochastic-termination
  algorithm, its generic environment integration, its original examples, or
  historical provenance.
- Preserve old Python imports, Gymnasium task IDs, and checkpoint workflows
  through explicit compatibility aliases.
- Move detailed ROS 2 and hardware instructions out of the README while
  keeping them easy to discover.
- Verify the complete user journey: create an environment, train a short run,
  evaluate a checkpoint, and generate plots from the evaluation output.

## Non-goals

- The GitHub repository itself will not be renamed in this branch. The owner
  will rename it to `LoComposition` after the branch is ready.
- The CaT mechanism will not be renamed or presented as a LoComposition
  algorithmic contribution.
- The rebrand will not change rewards, constraints, observations, actions,
  terrain generation, domain randomization, policy architecture, PPO behavior,
  checkpoint tensors, or evaluation metrics.
- The README will not reproduce the method derivation, complete ablation
  discussion, project-video transcript, or long-form blog narrative.
- Missing videos and GIFs will not be replaced with fabricated demonstrations.

## Public Narrative

The repository should open with one direct idea: LoComposition separates the
jobs that a locomotion policy must reconcile instead of encoding all of them in
one complex reward. Velocity tracking specifies the task, CaT constraints state
operational limits, mechanical energy minimization supplies a gait preference,
and exteroceptive perception makes that preference conditional on the terrain.
This produces efficient trotting and rough-terrain adaptation without air-time,
contact-count, foot-clearance, or other explicit gait-style priors.

The paper's headline comparison remains the primary quantitative result:
LoComposition reaches comparable terrain traversal to a conventional complex
reward formulation while reducing Cost of Transport by 56% and
operational-limit violations by 96%. The README also states that the learned
Go2 policy transfers zero-shot to physical hardware using online LiDAR-based
elevation mapping.

Additional experiments appear as concise evidence, not a second results
section. They cover terrain-dependent clearance and contact timing, greater
robustness to a 20 ms control delay, the matched reward-penalty comparison, and
embodiment transfer to ANYmal C and Spot. The embodiment claim must be precise:
the LoComposition formulation and relative training recipe remain unchanged;
action scale and operational bounds follow the actuator interface, while
disturbances, mass randomization, and energy coefficients are scaled
deterministically by robot mass. These conversions are physical normalization,
not a new reward search or gait-prior tuning pass. Each embodiment is trained
separately; this is formulation transfer, not one policy transferred between
robots.

## Naming Boundary

| Area | Primary name | Compatibility treatment |
| --- | --- | --- |
| Project and future repository | `LoComposition` | GitHub's old repository URL should redirect after the owner performs the rename. |
| Python distribution and package | `locomposition` | `cat_envs` remains as a forwarding compatibility namespace. |
| Project Gymnasium environments | `LoComposition-Go2-*`, `LoComposition-Anymal-C-*`, and `LoComposition-Spot-*` | Existing `CaT-*` IDs remain registered against the same implementations. |
| Project configuration modules | `locomposition_*_env_cfg.py` | Legacy module paths resolve through the compatibility namespace. |
| Constraint algorithm | `CaT`, `CaTEnv`, `ConstraintManager`, and `tasks/utils/cat` | Names remain unchanged and the source is explicitly attributed. |
| Original Solo12 tasks | Existing CaT task and module names | Retained and identified as upstream CaT compatibility examples. |
| ROS controller | `locomposition_controller` | A legacy `cat_controller` package forwards its existing launch entry points; C++ include/API compatibility is not promised. |
| ROS bringup | `locomposition_bringup` | A legacy `cat_bringup` package forwards `bringup.launch.py` to the new package. |
| ROS state-estimation launch package | `locomposition_state_estimation` | A legacy `cat_state_estimation` package forwards its existing launch entry points. |
| ROS elevation-map message package | `locomposition_perception_msgs` | Renamed without claiming wire compatibility: a ROS message package name is part of its type identity. The migration is documented. |
| Docker and runtime labels | LoComposition service, image-variable, image default, and container labels | The default image is `loukordos/locomposition-sim2real:latest`; `LOCOMPOSITION_SIM2REAL_IMAGE` can override it. |
| New logs and W&B projects | LoComposition task/project names | Existing log directories and saved task IDs remain readable. |

Compatibility protects meaningful user interfaces, not every incidental old
string. Local absolute paths, comments that incorrectly describe the entire
project as CaT, archive filenames, container labels, and logging defaults are
updated without aliases.

## Python Package and Environment Migration

`locomposition` becomes the canonical import namespace. Project-owned Go2,
ANYmal C, and Spot modules and their documentation use LoComposition names.
The generic constraint integration remains under `tasks/utils/cat` and keeps
the `CaTEnv` name because that class specifically changes environment stepping
and PPO return handling to implement CaT.

The `cat_envs` compatibility package forwards old import paths to the canonical
implementation without runtime deprecation noise. Both namespaces must refer
to the same class and configuration objects rather than maintaining duplicate
implementations.

New Gymnasium task registrations use `LoComposition-*`. Every currently
registered `CaT-*` Go2, ANYmal C, and Spot task remains available as a legacy
alias with the same entry point and configuration. The original
`Isaac-Velocity-CaT-*` Solo12 examples retain their historical names.

Saved checkpoints do not change. Evaluation code must accept task names stored
by old runs, recognize the new names, and resolve identical embodiment profiles
and constraint thresholds for both.

## ROS 2 Migration

Project-owned ROS packages are renamed to the LoComposition namespace,
including the elevation-map message package. Package manifests, CMake targets,
Python entry points, includes, launch files, configuration paths, dependencies,
topic type references, resource-index files, and documentation all move
together.

Launch-only compatibility wrappers are retained where forwarding is exact.
The legacy `cat_controller`, `cat_bringup`, and `cat_state_estimation` packages
contain no duplicate implementation; they depend on and include the canonical
LoComposition launches. Direct C++ header compatibility is outside the public
compatibility contract.
The old `cat_perception_msgs/ProcessedElevationMap` type is not described as
wire-compatible with
`locomposition_perception_msgs/ProcessedElevationMap`; ROS considers those
different message types even when their fields are identical. The migration
guide identifies the type change and the required source rebuild.

The deployment behavior remains unchanged: a 50 Hz TorchScript policy loop
receives normalized proprioception and a robot-centric 13 by 11 elevation map,
while a 500 Hz loop republishes the latest PD targets to the Go2. The detailed
networking, time synchronization, LiDAR, elevation-mapping, Docker, launch, and
safety instructions move to `docs/sim2real.md`.

## README Design

The README is the repository's front door, not a copy of the paper.

1. A compact header gives the full paper title and links to the project page,
   arXiv paper, videos, and code-facing documentation. The current video link
   points to the project page, which hosts the videos, until a direct video URL
   is supplied.
2. A web-optimized export of `figures/main_figure.pdf` provides the visual
   overview.
3. A short explanation introduces task reward, CaT operational limits, energy
   preference, and terrain perception in plain language.
4. A compact results block reports the paper's 56% COT and 96% violation-rate
   improvements, comparable traversal, and zero-shot physical Go2 deployment.
5. An additional-evidence block summarizes the matched penalty, delay,
   terrain-adaptation, and cross-embodiment findings without reproducing their
   derivations.
6. Installation lists the exact tested Python, Isaac Sim, Isaac Lab, PyTorch,
   and CUDA-facing package versions.
7. Quick-start commands cover environment setup, one training run, one
   evaluation, and plot generation using the primary LoComposition task ID.
8. A support matrix distinguishes physical Go2 deployment from simulated,
   separately trained ANYmal C and Spot formulation-transfer results.
9. A repository map points to training, evaluation, analysis, CaT, assets, and
   ROS code.
10. A short sim-to-real section links prominently to `docs/sim2real.md`.
11. Citation and attribution list LoComposition first, then explain that the
    operational-limit component uses CaT and cite Chane-Sane et al.
12. License and acknowledgements close the document.

The README avoids unsupported maturity claims. It describes which artifacts
are released and does not imply that every paper table can be reproduced with
one command unless the repository actually exposes that workflow.

## Documentation

`docs/sim2real.md` preserves the useful content of the current README while
reorganizing it around prerequisites, architecture, host/robot networking,
time synchronization, LiDAR setup, elevation mapping, Docker and workspace
builds, launch procedure, validation, and safety notes. Commands and paths use
the new ROS package names.

`docs/migration.md` maps old and new Python imports, Gymnasium task IDs, ROS
packages, launch files, Docker labels, and repository URLs. It explains the one
intentional ROS message-type break and provides the owner's final GitHub rename
checklist.

## Media Assets

Available paper assets are exported to stable, descriptive files under
`assets/` rather than referenced from a Downloads directory. Raster exports
must remain legible on GitHub at normal desktop width and use lossless PNG for
plots and diagrams.

- The paper overview is the README hero.
- The COT/contact-pattern figure supports the efficiency result.
- The sim-to-real montage supports the deployment section.
- The terrain contact-pattern and swing-height figures support the additional
  terrain-adaptation evidence.
- Existing failure-case images remain available for documentation but do not
  need to crowd the README.

Missing GIF and video slots use clearly labeled, repository-native placeholder
artwork with stable filenames so the final media can replace them without
rewriting the surrounding README. The implementation handoff includes an asset
checklist giving each filename, aspect ratio, crop, intended content, and link
target. No placeholder is presented as experimental evidence.

## Setup Script Design

`create-isaac-lab-env-uv.sh` keeps its unbranded filename for compatibility and
is updated to identify LoComposition and install the canonical package. Its
default remote becomes the future
LoComposition repository URL. To test the current branch before the GitHub
rename, the script accepts explicit environment-root and repository-source
overrides. A local Git clone of the completed worktree can therefore exercise
the exact branch rather than silently installing the current public `main`.

The script remains suitable for the normal remote-clone workflow and reports
the activation and repository paths using LoComposition names. Its acceptance
test uses a unique environment directory so it cannot overwrite an existing
research environment.

## Verification and Acceptance

The rebrand is complete only after fresh evidence for all of the following.

### Static and compatibility checks

- The canonical `locomposition` import and legacy `cat_envs` import both load.
- Canonical and legacy imports resolve the same CaT classes and project
  configuration objects.
- Every new LoComposition Gymnasium task and its legacy CaT alias register and
  point to equivalent entry points and configuration classes.
- Python sources compile.
- Shell, TOML, YAML, XML, CMake, and launch-file syntax checks pass where tools
  are installed.
- README relative links and asset paths exist.
- A repository-wide branding scan classifies every remaining occurrence of
  `CaT`, `cat_*`, or `constraints-as-terminations` as algorithm terminology,
  historical attribution, an original example, or an intentional compatibility
  alias.

### Fresh environment creation

Run the environment-creation script with a unique target and the completed
local worktree as its repository source. The script must create Python 3.11,
install the pinned Isaac Sim 5.1 and Isaac Lab revision, install LoComposition,
and finish its own checklist without error. This is allowed to use package
caches but not an already configured virtual environment.

### Training smoke

Start the canonical LoComposition Go2 training task with a small environment
count and two PPO iterations, offline logging, and headless rendering. Verify a
finite rollout and optimizer update, correct task/configuration reporting, and
the absence of import, registry, observation-shape, constraint, or NaN/Inf
failures.

### Evaluation smoke

Evaluate a real existing checkpoint through the canonical LoComposition play
task. Verify that the rollout completes and writes the expected metrics
summary. Separately confirm that an old saved CaT task name still resolves
through the compatibility registration.

### Plot-generation smoke

Run the repository's actual plot-generation command against the new evaluation
output. Verify exit code zero and the expected non-empty plot artifacts rather
than relying only on import or unit tests.

### Embodiment startup checks

Where the installed Isaac Lab assets permit, construct and reset the canonical
Go2, ANYmal C, and Spot train/play configurations and execute a bounded number
of finite steps. Confirm unchanged action and observation dimensions and the
expected embodiment-specific physical parameters.

The known unrestricted `pytest` collection errors caused by absent ROS
`ament_copyright`, `ament_flake8`, and `ament_pep257` packages are explicitly
ignored, as requested. Focused tests and real workflows remain required.

## Risks and Mitigations

- **Python namespace duplication:** forwarding imports could load two copies of
  a module. Compatibility tests compare object identity, not just successful
  imports.
- **Gymnasium alias drift:** registrations could point at subtly different
  configurations. Tests compare entry-point and configuration metadata.
- **ROS message rename:** identical fields do not provide wire compatibility.
  The type break is explicit, all consumers change together, and the workspace
  is rebuilt.
- **Stale checkpoint task names:** old task IDs remain registered and evaluation
  name detection accepts both prefixes.
- **Premature future URLs:** the migration guide marks the GitHub rename as the
  owner's final step; local verification uses a repository-source override.
- **README overgrowth:** long method, networking, and experiment explanations
  live in the paper, project website, blog, or focused documentation rather
  than the repository front page.
- **Unverifiable media:** missing demonstrations remain labeled placeholders
  and are listed in the final asset handoff.

## Completion Criteria

The branch is ready when LoComposition is the clear primary identity throughout
the tracked project, all intentional CaT references are technically or
historically justified, compatibility tests pass, the README and focused docs
are internally consistent, and the fresh setup/training/evaluation/plotting
workflow has been exercised with recorded commands and outcomes.
