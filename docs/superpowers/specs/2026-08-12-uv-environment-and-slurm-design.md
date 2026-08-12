# Locked uv Environment and Slurm Training Design

## Purpose

LoComposition currently spreads Python dependency ownership across the root
`pyproject.toml`, `requirements.txt`, the extension's `setup.py`, and ad-hoc
`uv pip install` commands in `create-isaac-lab-env-uv.sh`. The installer also
creates an unrelated uv project above the repository with `uv init`, so it
does not use the repository's dependency metadata, package indexes, or Isaac
Sim resolver override.

This change establishes a single repository-owned dependency manifest and
lockfile, simplifies the installer, adds repository-owned Slurm templates,
and removes embedded W&B credentials. It must preserve the simulator,
training, evaluation, plotting, task-registration, and cluster workflows that
were verified before the packaging change.

## Goals

- Make the root `pyproject.toml` the sole declaration of LoComposition's
  Python environment dependencies.
- Commit `uv.lock` and use frozen synchronization for reproducible installs.
- Preserve Python 3.11, Torch 2.7.0 with CUDA 12.8, torchvision 0.22.0,
  Isaac Sim 5.1.0, and Isaac Lab revision
  `ddb044eb5b2300792de41e82d53b032f3632b489`.
- Preserve the existing editable Isaac Lab source installation and stable
  Isaac Lab checkout path.
- Preserve every package currently listed in `requirements.txt`; dependency
  pruning is outside this change.
- Remove `requirements.txt` after all consumers use `pyproject.toml`.
- Add a tracked L40S Slurm template and generate both L40S and 2080 Ti job
  files from it during environment creation.
- Keep W&B credentials outside the repository and reusable across virtual
  environments.
- Verify behavior in a newly created environment, including real training,
  evaluation, and plot generation.

## Non-goals

- Upgrading Isaac Sim, Isaac Lab, Torch, torchvision, RSL-RL, NumPy, or Zarr.
- Removing dependencies that appear unused.
- Replacing the Isaac Lab source checkout with Isaac Lab wheels or uv's Git
  cache.
- Installing optional Isaac Lab packages such as `isaaclab_mimic`.
- Changing the LoComposition algorithm, environments, training parameters,
  evaluation metrics, or task compatibility aliases.
- Changing the intended three-training-processes-per-L40S allocation.

## Dependency ownership

### Root project

The root `pyproject.toml` becomes a uv-managed virtual environment project
named `locomposition-environment`. It is not itself a distributable Python
package:

- remove the invalid Hatch build-system declaration;
- set `tool.uv.package = false`;
- retain `requires-python = "==3.11.*"`;
- retain the explicit PyTorch CUDA 12.8 and NVIDIA package indexes;
- retain the Linux resolver restriction and Isaac Sim `pywin32` metadata
  override;
- declare the editable `locomposition` extension as a local path source at
  `exts/locomposition`;
- move every dependency from `requirements.txt` into
  `project.dependencies`, preserving existing version constraints; and
- keep the existing formatter and type-checker configuration in the same
  file.

The virtual project and the extension deliberately have different names:
`locomposition-environment` describes the complete research environment,
while the editable distribution remains `locomposition`.

### Extension package

`exts/locomposition/setup.py` remains the package definition used by the
Isaac Lab extension. It is not redundant with the root virtual project.

Add `exts/locomposition/pyproject.toml` with the same build-system declaration
used by the pinned Isaac Lab source packages: setuptools, wheel, and toml with
the `setuptools.build_meta` backend. This removes the extension's dependence
on `--no-build-isolation` and makes the local path dependency buildable by uv.

### Lock boundaries

The environment has two explicit reproducibility boundaries:

1. `uv.lock` locks LoComposition, Isaac Sim, Torch, analysis, logging,
   plotting, and other repository-declared Python dependencies.
2. The exact Isaac Lab Git revision locks the four editable Isaac Lab source
   distributions.

The four retained Isaac Lab source distributions are:

- `source/isaaclab` (`isaaclab`);
- `source/isaaclab_assets` (`isaaclab-assets`);
- `source/isaaclab_tasks` (`isaaclab-tasks`); and
- `source/isaaclab_rl` (`isaaclab-rl`).

They remain editable because `AppLauncher` resolves Isaac Lab application
files relative to its source checkout. Their own package metadata supplies
their transitive dependencies. The final uv synchronization is inexact so it
does not remove the editable Isaac Lab packages or their additional
dependencies, while still enforcing every version present in `uv.lock`.

## Installer workflow

`create-isaac-lab-env-uv.sh ENV_NAME` retains its public command and the
`--root` and `--repo-source` options. It retains the safe refusal to merge into
a non-empty target directory.

The new workflow is:

1. Validate arguments and locate or install uv.
2. Create the empty environment root.
3. Clone LoComposition into `<environment>/LoComposition`.
4. Set `UV_PROJECT_ENVIRONMENT` to `<environment>/.venv`.
5. Run a frozen uv synchronization from the LoComposition project, initially
   omitting the local `locomposition` distribution. This creates Python 3.11
   and installs the locked Torch, Isaac Sim, and project dependencies.
6. Install `rust-just==1.40.0` if `just` is not already available.
7. Clone Isaac Lab into
   `<environment>/isaaclab-installation/IsaacLab` and check out the pinned
   revision.
8. Install the four retained Isaac Lab source distributions editably into the
   environment.
9. Run a second frozen, inexact uv synchronization. This installs the
   LoComposition extension editably, restores the repository-locked versions
   of overlapping packages, and preserves the editable Isaac Lab packages.
10. Run `uv pip check` with the environment's Python and verify the installed
    LoComposition and Isaac Lab package locations.
11. Generate the L40S and 2080 Ti Slurm files from the tracked repository
    template.
12. Print activation, W&B authentication, training, and submission commands.

The installer no longer runs `uv init`, creates an unrelated
`pyproject.toml`, upgrades pip independently, installs `requirements.txt`, or
copies `$HOME/local-mamba-test.sbatch`.

## Slurm design

### Repository template

Add `train-locomposition.sbatch` at the repository root. It is the sole Slurm
training template and uses explicit substitution markers for the environment
root and environment name. It contains no credential value.

The L40S template uses:

- job name `locomposition-training`;
- eight CPUs total;
- partition `L40Sday`;
- 6 GB per CPU;
- one L40S GPU;
- the existing six-day-and-23-hour time limit;
- the existing mail settings;
- array `0-2%3`;
- seeds 46 through 54;
- 7,500 environments;
- task `LoComposition-Go2-Rough-Terrain-Joint-State-History-v0`; and
- three concurrent training processes per array task.

Each child process invokes the existing `just train` recipe. The parent waits
for every child. If any child exits nonzero, the Slurm job exits nonzero after
all children have been reaped. No new preflight checks for seed ranges,
`just`, or environment directories are added.

### Generated files

The installer writes:

- `<environment>/train-locomposition.sbatch`; and
- `<environment>/train-locomposition-2080ti.sbatch`.

The L40S file differs from the tracked template only in its resolved
environment root and name.

The 2080 Ti file is derived from the same tracked template and additionally
uses:

- job name `locomposition-training-2080ti`;
- partition `week`;
- `gpu:2080ti:1`;
- array `0-8%9`; and
- one training process per array task.

It retains seeds 46 through 54, 7,500 environments, the canonical
LoComposition task, eight CPUs, and the same failure propagation.

## W&B credentials

No W&B API key is stored in the repository, installer, generated Slurm files,
documentation examples, or tests.

The primary cluster workflow is a one-time `wandb login` for the user's
cluster account. W&B stores the credential in the user's private
`~/.netrc`, outside all virtual environments. Every subsequently created
environment and compute job under the same shared home directory reuses that
credential.

The Slurm template also accepts an inherited `WANDB_API_KEY` environment
variable as an override, but never assigns or prints it. At job start it
requires either that environment variable or a readable W&B credential in
the user's netrc file. The error message describes the one-time login without
including secret material.

The user-provided key that appeared in conversation is treated as exposed and
must be rotated outside this repository.

## Documentation

Update the README installation section so an existing compatible Isaac Lab
environment uses the repository project rather than `requirements.txt`. The
documented command targets the active environment, uses the committed lock,
and performs an inexact sync so existing editable Isaac Lab packages remain
installed.

Document the generated L40S and 2080 Ti job files and explain that W&B login
is once per cluster user account, not once per virtual environment.

## Automated tests

Packaging tests must assert that:

- the root is a non-package uv project;
- the invalid root build backend is absent;
- all former `requirements.txt` dependencies are declared in
  `pyproject.toml`;
- the extension build declaration is valid;
- `requirements.txt` and all references to it are absent;
- `uv.lock` exists and `uv lock --check` succeeds; and
- the pinned versions and package indexes remain unchanged.

Installer tests use fake `uv` and `git` executables and assert that:

- `uv init` is never called;
- frozen synchronization targets the requested external environment;
- the pinned Isaac Lab revision is checked out;
- the same four Isaac Lab packages are installed editably;
- the second synchronization is frozen and inexact;
- the LoComposition extension is installed through the project manifest; and
- both Slurm variants are generated from the tracked template with the
  expected substitutions.

Slurm tests must assert that:

- both files pass `bash -n` after substitution;
- the L40S configuration launches seeds 46, 47, and 48 for array task zero;
- the 2080 Ti configuration launches only seed 46 for array task zero;
- a child failure produces a nonzero final exit status;
- the canonical task and LoComposition repository path are used; and
- neither the tracked template nor generated files contain an assigned W&B
  API key or a key-shaped literal.

## Fresh-environment verification

After automated tests pass, create a real environment under a disposable
directory using the revised installer and the current worktree as
`--repo-source`. Retain the evidence until all comparisons are complete.

Verification in that fresh environment includes:

1. `uv lock --check` and `uv pip check`.
2. Installed-version and editable-source inspection for LoComposition and all
   four Isaac Lab packages.
3. Import and canonical/legacy task-alias verification.
4. Short finite environment rollouts for Go2, Spot, and ANYmal C.
5. A two-iteration Go2 training run.
6. Go2, Spot, and ANYmal C checkpoint evaluations with 1,000 random simulation
   steps using the same checkpoints, seeds, action scales, and fixed-scenario
   settings as the pre-change regression runs.
7. Numerical comparison of shared rollout arrays, excluding wall-clock
   inference-duration measurements.
8. Complete evaluation plot and gait-report generation with finite stored
   arrays.
9. A local execution harness for each generated Slurm variant using fake
   `just`, ensuring its process fan-out and exit handling match the design.

The change is accepted only if the fresh environment installs successfully,
training exits successfully, evaluation and plotting exit successfully, and
any numerical differences are investigated before completion is claimed.

## Final handoff

The final report lists every changed text file, explains the two lock
boundaries, reports the fresh-install and runtime evidence, identifies any
remaining warnings, and provides one copyable command for a longer Go2
training run in the newly created environment.
