# Locked uv Environment and Slurm Implementation Plan

> **For Codex:** Execute this plan with the `superpowers:executing-plans` skill. Keep the work inline, as requested by the user, and use `superpowers:test-driven-development` for each behavior change.

**Goal:** Replace LoComposition's split dependency installation with a committed `uv.lock`, preserve the pinned editable Isaac Lab checkout, add safe repository-owned Slurm jobs, and prove that a newly installed environment still trains, evaluates, and produces plots.

**Architecture:** The repository root is a non-package uv project that owns all locked runtime dependencies and installs `exts/locomposition` as an editable local source. Isaac Lab remains a separately pinned source checkout with four editable distributions because its runtime resolves application resources from the checkout. The installer performs a frozen project sync before and after the Isaac Lab installs, using an inexact final sync to retain those separately managed editables.

**Tooling:** Bash, uv, PEP 621 `pyproject.toml`, setuptools, pytest, Slurm, Isaac Sim 5.1, Isaac Lab, RSL-RL.

---

## Task 1: Move dependency ownership to `pyproject.toml` and `uv.lock`

**Files:**

- Create: `tests/test_packaging_contract.py`
- Modify: `pyproject.toml`
- Create: `exts/locomposition/pyproject.toml`
- Delete: `requirements.txt`
- Create: `uv.lock`

### Step 1: Write failing packaging-contract tests

Add tests that parse TOML with `tomllib` and assert:

- the root project is named `locomposition-environment`;
- `[tool.uv] package = false` is present;
- the root has no `[build-system]` table;
- `locomposition` is a project dependency and maps to the editable local source `exts/locomposition`;
- all dependencies formerly declared in `requirements.txt` are represented in `project.dependencies`, including the existing NumPy, Zarr, RSL-RL, Torch, torchvision, and Isaac Sim pins;
- the PyTorch CUDA 12.8 and NVIDIA indexes and the `pywin32` override remain present;
- `exts/locomposition/pyproject.toml` declares `setuptools.build_meta` with `setuptools`, `wheel`, and `toml` build requirements;
- `requirements.txt` no longer exists; and
- `uv.lock` exists.

Run:

```bash
pytest -q tests/test_packaging_contract.py
```

Expected: FAIL because the root is still a buildable Hatch project, the dependencies remain split, and neither the extension build declaration nor lockfile exists.

### Step 2: Implement the consolidated project manifest

Update `pyproject.toml` so it:

- defines the virtual environment project `locomposition-environment`;
- retains Python `==3.11.*`;
- lists `locomposition` and every dependency from the old root manifest and `requirements.txt`;
- preserves existing version pins exactly;
- removes the root Hatch build system;
- sets `tool.uv.package = false`;
- declares `locomposition = { path = "exts/locomposition", editable = true }` under `tool.uv.sources`; and
- leaves the existing index, override, formatter, and type-checker settings intact.

Add the setuptools build declaration under `exts/locomposition`, then delete `requirements.txt`.

### Step 3: Generate and validate the lock

Run:

```bash
uv lock
uv lock --check
pytest -q tests/test_packaging_contract.py
```

Expected: lock resolution succeeds and all packaging-contract tests pass.

### Step 4: Inspect the resolved pins

Confirm the lock contains the intended versions and local editable source:

```bash
rg -n 'name = "(locomposition|torch|torchvision|isaacsim|numpy|zarr|rsl-rl-lib)"|editable = "exts/locomposition"' uv.lock
```

Expected: all named distributions appear; the lock does not silently upgrade the explicitly pinned packages.

### Step 5: Commit the packaging boundary

```bash
git add pyproject.toml exts/locomposition/pyproject.toml requirements.txt uv.lock tests/test_packaging_contract.py
git commit -m "Lock LoComposition environment with uv"
```

---

## Task 2: Add the repository-owned Slurm training template

**Files:**

- Create: `train-locomposition.sbatch`
- Create: `tests/test_slurm_training.py`

### Step 1: Write failing static and execution tests

Add tests that render the template into a temporary environment and run it with fake `just`, `hostname`, and `wandb`/netrc state. Assert:

- the tracked file passes `bash -n` after placeholder substitution;
- the L40S directives request `L40Sday`, one L40S, eight CPUs, and array `0-2%3`;
- array task zero launches seeds 46, 47, and 48 with 7,500 environments and the canonical LoComposition Go2 task;
- the parent waits for all three children;
- one failed child makes the final job exit nonzero only after all children are reaped;
- a readable W&B netrc entry or inherited `WANDB_API_KEY` is accepted;
- neither credentials nor key-shaped values are assigned or printed; and
- placeholder repository paths resolve to `<root>/<environment>/LoComposition`.

Run:

```bash
pytest -q tests/test_slurm_training.py
```

Expected: FAIL because the template does not exist.

### Step 2: Implement `train-locomposition.sbatch`

Create the approved L40S job with:

- job name `locomposition-training`;
- `--cpus-per-task=8`, `--partition=L40Sday`, `--mem-per-cpu=6G`, and `--gres=gpu:L40S:1`;
- the existing time, output, email, node, and task settings;
- array `0-2%3`;
- explicit environment-root and environment-name substitution markers;
- activation of `<root>/<environment>/.venv` and `cd` into LoComposition;
- W&B credential discovery through inherited `WANDB_API_KEY` or `$NETRC`/`~/.netrc`, without assigning a key;
- seeds 46 through 54, `NUM_ENVS=7500`, the canonical task, and `RUNS_PER_NODE=3`; and
- child-PID collection and nonzero failure propagation after all waits finish.

Do not add the rejected `just`, environment-directory, or seed-range preflight checks.

### Step 3: Run the focused tests and shell parser

```bash
bash -n train-locomposition.sbatch
pytest -q tests/test_slurm_training.py
```

Expected: PASS.

### Step 4: Commit the template

```bash
git add train-locomposition.sbatch tests/test_slurm_training.py
git commit -m "Add safe LoComposition Slurm training template"
```

---

## Task 3: Rewrite the environment installer around frozen uv syncs

**Files:**

- Modify: `tests/test_setup_script.py`
- Modify: `create-isaac-lab-env-uv.sh`

### Step 1: Update the fake installer harness and write failing expectations

Make the fake `git clone` copy the project manifest, lockfile, extension metadata, and Slurm template into the fake LoComposition clone. Make fake `uv sync` create the external `.venv/bin/python` and activation file based on `UV_PROJECT_ENVIRONMENT`.

Replace obsolete assertions with tests that assert:

- LoComposition is cloned before the first sync;
- `UV_PROJECT_ENVIRONMENT` points at `<environment>/.venv`;
- the first sync is `--frozen` and omits installation of the local `locomposition` distribution;
- `uv init`, independent pip upgrades, direct Torch/Isaac Sim installs, and `-r requirements.txt` are absent;
- Isaac Lab is cloned to the stable path and checked out at `ddb044eb5b2300792de41e82d53b032f3632b489`;
- exactly the same four Isaac Lab source packages are installed editably with the new environment Python;
- the final project sync is both frozen and inexact;
- `uv pip check` targets the new environment Python;
- the local extension is installed by the project sync rather than a manual no-build-isolation command;
- both generated Slurm files exist and pass `bash -n`;
- L40S placeholders are substituted without changing its approved resources; and
- the 2080 Ti output uses `week`, `gpu:2080ti:1`, `0-8%9`, and `RUNS_PER_NODE=1` while retaining eight CPUs.

Keep the existing help, unsafe-name, missing-option, non-empty-target, and install-isolation coverage.

Run:

```bash
pytest -q tests/test_setup_script.py
```

Expected: FAIL against the old ad-hoc installer.

### Step 2: Implement the locked installer flow

Rewrite `create-isaac-lab-env-uv.sh` to:

1. retain safe argument parsing, defaults, target-directory refusal, and uv discovery;
2. create the environment root and clone LoComposition first;
3. export `UV_PROJECT_ENVIRONMENT=<environment>/.venv`;
4. run the initial frozen sync from the cloned project while omitting `locomposition`;
5. install `rust-just==1.40.0` only when `just` is unavailable;
6. clone and pin Isaac Lab at the stable checkout path;
7. install the four approved Isaac Lab source distributions editably using the new venv Python;
8. run the final `uv sync --frozen --inexact` from LoComposition;
9. run `uv pip check` and import-location diagnostics using the new environment;
10. render the L40S output from the tracked template;
11. derive and render the 2080 Ti output from that same template; and
12. print activation, one-time W&B login, local training, and `sbatch` instructions.

Do not introduce a second project, a second requirements source, a copied home-directory template, or any credential assignment.

### Step 3: Run focused installer tests

```bash
bash -n create-isaac-lab-env-uv.sh
pytest -q tests/test_setup_script.py tests/test_slurm_training.py tests/test_packaging_contract.py
```

Expected: PASS.

### Step 4: Commit the installer rewrite

```bash
git add create-isaac-lab-env-uv.sh tests/test_setup_script.py
git commit -m "Create environments from the committed uv lock"
```

---

## Task 4: Update installation, W&B, and cluster documentation

**Files:**

- Modify: `README.md`
- Modify: existing README contract test, or create `tests/test_readme_setup.py`

### Step 1: Add failing README contract checks

Assert that the README:

- no longer refers to `requirements.txt`;
- explains the committed `uv.lock` and pinned Isaac Lab revision boundaries;
- gives an inexact, frozen sync command for an already compatible Isaac Lab environment;
- documents the installer-generated L40S and 2080 Ti files;
- states that `wandb login` is a one-time per-user/shared-home operation rather than a per-environment step; and
- does not contain an assigned or key-shaped W&B secret.

Run the focused README test and confirm it fails.

### Step 2: Update the README concisely

Keep the existing layered paper/software README structure. Update only setup and cluster workflow material necessary to make the repository self-contained:

- explain what `uv.lock` controls and what the pinned Isaac Lab checkout controls;
- document fresh installation with `create-isaac-lab-env-uv.sh`;
- document how to sync an existing compatible environment without deleting editable Isaac Lab packages;
- list the generated L40S and 2080 Ti files and their intended concurrency; and
- document one-time W&B authentication without duplicating project-page, paper, or blog content.

### Step 3: Verify and commit documentation

```bash
pytest -q tests/test_readme_setup.py
rg -n 'requirements\.txt|WANDB_API_KEY[[:space:]]*=' README.md create-isaac-lab-env-uv.sh train-locomposition.sbatch
git add README.md tests/test_readme_setup.py
git commit -m "Document locked setup and cluster training"
```

Expected: tests pass and the search finds neither stale requirements instructions nor a credential assignment.

---

## Task 5: Run the repository-level automated verification

**Files:**

- Modify only if a test exposes a real regression.

### Step 1: Run lock, shell, and full unit tests

```bash
uv lock --check
bash -n create-isaac-lab-env-uv.sh
bash -n train-locomposition.sbatch
pytest -q
```

Expected: all checks pass.

### Step 2: Scan for stale install paths and secrets

```bash
rg -n 'requirements\.txt|local-mamba-test\.sbatch|constraints-as-terminations-training|export WANDB_API_KEY=' --glob '!uv.lock' .
git diff --check
```

Expected: no active stale installer/reference or secret assignment remains; historical design documents may be reviewed separately if a match is intentional.

### Step 3: Commit any test-driven corrections

If fixes were required, rerun the affected test first and then the full suite before committing them with a narrowly scoped message.

---

## Task 6: Create and inspect a genuinely fresh locked environment

**Files:**

- No repository changes expected.
- Disposable environment: `/tmp/locomposition-locked-env-smoke-20260812`

### Step 1: Run the real installer from committed source

```bash
./create-isaac-lab-env-uv.sh locomposition-locked-env-smoke-20260812 \
  --root /tmp \
  --repo-source /home/kordoslo/dev/locomposition-rename-project-and-docs
```

Expected: the cloned repository syncs from `uv.lock`; Isaac Lab is checked out at the pinned commit; the final sync and `uv pip check` pass; both Slurm variants are generated.

### Step 2: Verify lock and package provenance in the fresh clone

```bash
UV_PROJECT_ENVIRONMENT=/tmp/locomposition-locked-env-smoke-20260812/.venv \
  uv sync --project /tmp/locomposition-locked-env-smoke-20260812/LoComposition --frozen --inexact
uv pip check --python /tmp/locomposition-locked-env-smoke-20260812/.venv/bin/python
/tmp/locomposition-locked-env-smoke-20260812/.venv/bin/python -c \
  'import inspect, isaaclab, isaaclab_assets, isaaclab_tasks, locomposition; print(inspect.getfile(isaaclab)); print(inspect.getfile(isaaclab_assets)); print(inspect.getfile(isaaclab_tasks)); print(inspect.getfile(locomposition))'
```

Expected: Isaac Lab resolves from the pinned checkout and LoComposition resolves from the fresh cloned repository.

### Step 3: Verify generated Slurm variants

```bash
bash -n /tmp/locomposition-locked-env-smoke-20260812/train-locomposition.sbatch
bash -n /tmp/locomposition-locked-env-smoke-20260812/train-locomposition-2080ti.sbatch
rg -n '^#SBATCH --(job-name|cpus-per-task|partition|gres|array)|^RUNS_PER_NODE=|LoComposition' \
  /tmp/locomposition-locked-env-smoke-20260812/train-locomposition*.sbatch
```

Expected: both parse; the intended L40S and 2080 Ti resource differences are the only variant-specific changes.

---

## Task 7: Smoke-test task creation and training in the fresh environment

**Files:**

- No repository changes expected.

Run all commands from the fresh clone with its Python and `OMNI_KIT_ACCEPT_EULA=Y`. Use a disposable Matplotlib directory and disable Omniverse Hub lookups.

### Step 1: Verify task alias registration

```bash
cd /tmp/locomposition-locked-env-smoke-20260812/LoComposition
OMNI_KIT_ACCEPT_EULA=Y OMNICLIENT_HUB_MODE=disabled MPLCONFIGDIR=/tmp/locomposition-locked-mpl \
  /tmp/locomposition-locked-env-smoke-20260812/.venv/bin/python scripts/verify_task_aliases.py
```

Expected: all canonical LoComposition tasks and compatibility aliases register correctly.

### Step 2: Create and step all three robot environments

Run `scripts/smoke_task.py --headless --steps=25` for:

- `LoComposition-Go2-Rough-Terrain-Joint-State-History-v0`;
- `LoComposition-Spot-Rough-Terrain-v0`; and
- `LoComposition-Anymal-C-Rough-Terrain-v0`.

Expected: each simulator environment launches, steps 25 times, and exits cleanly. Ignore known `ament` diagnostics as requested.

### Step 3: Run real PPO training startup

```bash
cd /tmp/locomposition-locked-env-smoke-20260812/LoComposition
OMNI_KIT_ACCEPT_EULA=Y OMNICLIENT_HUB_MODE=disabled WANDB_MODE=offline \
  ENV_NAME=locomposition_locked_train_smoke \
  /tmp/locomposition-locked-env-smoke-20260812/.venv/bin/python scripts/clean_rl/train.py \
  --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 \
  --seed=46 --headless --num_envs=64 --num_iterations=2 --logger tensorboard
```

Expected: PPO initializes, performs two iterations, writes a checkpoint/log, and exits successfully.

---

## Task 8: Re-run regression evaluations and plot generation in the fresh environment

**Files:**

- No repository changes expected.
- Preserve previous reference output under `/tmp/locomposition-regression-evals.9GG7MI`.
- Stage fresh runs under a separate newly created `/tmp` directory.

### Step 1: Stage the three archived runs without altering the reference data

Copy each run's params and requested checkpoint into isolated directories:

- Go2: run `2026-05-30-00-12-00`, checkpoint `25649`;
- ANYmal C: run `2026-07-23-17-24-19`, checkpoint `29999`;
- Spot: run `2026-08-01-15-01-48`, checkpoint `19599`.

Use the archive paths supplied by the user and do not write back into them.

### Step 2: Run fresh 1,000-step-randomized evaluations

For each staged run, invoke `scripts/eval.py` with:

- the matching `--run_dir` and `--eval_checkpoint`;
- its canonical LoComposition `-Play-v0` task;
- `--random_sim_step_length=1000`;
- `--fixed_command_sim_steps=500`;
- `--seed=46`; and
- `--headless`.

Expected: all 21 evaluation scenarios complete for Go2, Spot, and ANYmal C, including generated plots, gait reports, and video artifacts.

### Step 3: Compare numerical rollouts with the retained references

Compare all common rollout arrays for the first 1,000 steps, excluding wall-clock inference durations:

- Go2 against `/tmp/locomposition-regression-evals.9GG7MI/go2/2026-05-30-00-12-00/eval_checkpoint_25649_seed_46_action_delay_0`;
- ANYmal against `/tmp/locomposition-regression-evals.9GG7MI/anymal/2026-07-23-17-24-19/eval_checkpoint_29999_seed_46_action_delay_0`; and
- Spot against `/tmp/locomposition-regression-evals.9GG7MI/spot/2026-08-01-15-01-48/eval_checkpoint_19599_seed_46_action_delay_0`.

Report exact equality where achieved; otherwise report maximum absolute/relative differences and investigate before accepting the result.

### Step 4: Verify plot outputs explicitly

List and validate the expected generated plot/report files for each run. Run the repository plot-generation tests again under the fresh environment if they can target the staged output.

Expected: plot generation succeeds in both unit-level and real-evaluation workflows.

---

## Task 9: Final verification and handoff

**Files:**

- Modify only if verification exposes a defect.

### Step 1: Apply verification-before-completion

Re-run current evidence immediately before claiming success:

```bash
uv lock --check
pytest -q
git diff --check
git status --short
git log --oneline --decorate -8
```

Also retain concise results for fresh installation, package provenance, simulator smokes, two-iteration training, all three evaluations, numerical comparison, and plot outputs.

### Step 2: Review the final diff for naming and credential safety

Inspect every modified text file and search for:

- stale `constraints-as-terminations` project branding where it is not a deliberate compatibility/path reference;
- stale `requirements.txt` installation instructions;
- embedded or assigned W&B keys; and
- accidental dependency upgrades.

### Step 3: Commit any final verified correction

Only commit after its focused test and the complete automated suite pass.

### Step 4: Hand off results and a longer-run command

The final response should:

- lead with whether the fresh locked setup works;
- list every material file changed and why;
- explain the two lock boundaries and one-time W&B login;
- summarize automated and real simulator evidence, including comparison results;
- identify any ignored `ament` warnings or other non-blocking caveats;
- list the text files the user should review; and
- provide one copy-paste command from the new environment for a longer Go2 training run, with explicit environment count, iterations, seed, task, and logger behavior.
