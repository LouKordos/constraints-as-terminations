# Baseline Training Iteration Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set the default maximum training budget for the matched Go2, ANYmal-C, and Spot reward-based baselines to 30,000 iterations.

**Architecture:** Keep the existing shared RSL-RL training recipe and per-embodiment public recipes. Change only the three public default argument values, then verify both default and explicit-override command expansion without starting Isaac Lab or W&B.

**Tech Stack:** `just`, shell command expansion, Git

## Global Constraints

- Default `max_iterations` must be exactly `30000` for Go2, ANYmal-C, and Spot.
- Preserve the existing `max_iterations` recipe argument and explicit override behavior.
- Do not change trainer, agent, environment, reward, observation, evaluation, or sim2real code.
- Do not start Isaac Lab, a simulator process, training, evaluation, or a W&B run.

---

### Task 1: Align the public baseline training defaults

**Files:**
- Modify: `justfile:29-36`

**Interfaces:**
- Consumes: `_train-rsl-baseline task num_envs seed max_iterations wandb_project *flags`, which forwards `max_iterations` as `--max_iterations={{max_iterations}}`.
- Produces: `train-baseline-go2`, `train-baseline-anymal-c`, and `train-baseline-spot` recipes whose omitted `max_iterations` argument resolves to `30000`, while an explicit positional argument still takes precedence.

- [ ] **Step 1: Record the failing default-expansion check**

Run:

```bash
test "$(rg -c '^train-baseline-(go2|anymal-c|spot).*max_iterations="30000"' justfile || true)" = "3"
```

Expected before the edit: exit status 1 because Go2 and ANYmal-C use `1500`
and Spot uses `20000`, so zero matching declarations are found.

- [ ] **Step 2: Change the minimal configuration values**

Replace the three recipe declarations with:

```just
train-baseline-go2 num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_go2" *flags:
    just _train-rsl-baseline Baseline-Go2-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

train-baseline-anymal-c num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_anymal_c" *flags:
    just _train-rsl-baseline Baseline-Anymal-C-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}

train-baseline-spot num_envs="7500" seed="46" max_iterations="30000" wandb_project="baseline_spot" *flags:
    just _train-rsl-baseline Baseline-Spot-Rough-Terrain-v0 {{num_envs}} {{seed}} {{max_iterations}} {{wandb_project}} {{flags}}
```

- [ ] **Step 3: Verify recipe parsing and default expansion**

Run:

```bash
just --list
for recipe in train-baseline-go2 train-baseline-anymal-c train-baseline-spot; do
    just --dry-run "$recipe"
done
```

Expected: `just --list` reports `max_iterations="30000"` for every public baseline recipe, and each dry-run expansion forwards `30000` to `_train-rsl-baseline` and then to `--max_iterations=30000`.

- [ ] **Step 4: Verify explicit override behavior**

Run:

```bash
just --dry-run train-baseline-go2 64 46 2
just --dry-run train-baseline-anymal-c 64 46 2
just --dry-run train-baseline-spot 64 46 2
```

Expected: every expansion uses `--num_envs=64`, `--seed=46`, and `--max_iterations=2`, proving that short diagnostics remain available.

- [ ] **Step 5: Review the complete change**

Run:

```bash
git diff --check
git diff -- justfile
git status --short
```

Expected: no whitespace errors; the `justfile` diff changes exactly three default values; the status contains no unexplained files.

- [ ] **Step 6: Commit the implementation**

Run:

```bash
git add justfile
git commit -m "Increase baseline training budgets"
```

Expected: one implementation commit containing only the three `justfile` default changes.
