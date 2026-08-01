# Baseline W&B Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all three matched-baseline training recipes select W&B with a configurable embodiment-specific default project.

**Architecture:** Extend only the existing shared and public baseline recipes in `justfile`. The public recipes own their default project names, while the shared recipe forwards the project and the installed Isaac Lab RSL-RL logger flags.

**Tech Stack:** Just, Isaac Lab RSL-RL CLI, Weights & Biases.

## Global Constraints

- Default projects are exactly `baseline_go2`, `baseline_anymal_c`, and `baseline_spot`.
- Pass `--logger=wandb` and set `--log_project_name` from the public recipe's project argument.
- Keep existing three-positional-argument recipe invocations valid.
- Preserve variadic RSL-RL and Hydra flag forwarding.
- Do not modify environments, algorithms, evaluation, or sim2real.
- Use command-expansion checks rather than new unit-test infrastructure.

---

### Task 1: Enable W&B in matched-baseline training recipes

**Files:**

- Modify: `justfile:13-34`

**Interfaces:**

- Consumes: the installed trainer CLI arguments `--logger` and `--log_project_name`.
- Produces: public recipes with signature `num_envs seed max_iterations wandb_project *flags` and the existing RSL-RL log directory layout.

- [ ] **Step 1: Run the focused failing expansion check**

```bash
expansion="$(just --dry-run _train-rsl-baseline Baseline-Go2-Rough-Terrain-v0 64 46 2 --run_name wandb_verify)"
test "${expansion}" != "${expansion#*--logger=wandb}"
```

Expected: non-zero exit status because the current shared recipe does not pass `--logger=wandb`.

- [ ] **Step 2: Extend the shared recipe**

Change its signature and trainer invocation to:

```just
_train-rsl-baseline task num_envs seed max_iterations wandb_project *flags:
    # existing setup lines remain unchanged
    TMPDIR="$tmpdir" python scripts/train_rsl_rl.py \
        --task={{task}} \
        --seed={{seed}} \
        --headless \
        --num_envs={{num_envs}} \
        --max_iterations={{max_iterations}} \
        --logger=wandb \
        --log_project_name={{wandb_project}} \
        env.scene.terrain.terrain_generator.seed={{seed}} \
        env.sim.random_seed={{seed}} \
        {{flags}}
```

- [ ] **Step 3: Add embodiment defaults to the public recipes**

Use these exact signatures and forwarding calls:

```just
train-baseline-go2 num_envs="7500" seed="46" max_iterations="1500" wandb_project="baseline_go2" *flags:
    just _train-rsl-baseline Baseline-Go2-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

train-baseline-anymal-c num_envs="7500" seed="46" max_iterations="1500" wandb_project="baseline_anymal_c" *flags:
    just _train-rsl-baseline Baseline-Anymal-C-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

train-baseline-spot num_envs="7500" seed="46" max_iterations="20000" wandb_project="baseline_spot" *flags:
    just _train-rsl-baseline Baseline-Spot-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}
```

- [ ] **Step 4: Verify all default and custom expansions**

Run `just --list`, then dry-run the shared recipe once per embodiment using the corresponding default project. Also dry-run Go2 with `custom_baseline_project`.

Expected in every expansion: `--logger=wandb`, plus the matching
`--log_project_name=baseline_go2`, `baseline_anymal_c`, or `baseline_spot` value.

Expected in the Go2 custom expansion:

```text
--logger=wandb --log_project_name=custom_baseline_project
```

- [ ] **Step 5: Run final repository checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only `justfile` modified after the plan commit.

- [ ] **Step 6: Commit the implementation**

```bash
git add justfile
git commit -m "Enable W&B for baseline training"
```
