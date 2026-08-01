# TCML Baseline Slurm Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and validate six directly submit-able baseline-training Slurm launchers on `tcml-cluster`, covering Go2, ANYmal-C, and Spot on L40S and 2080 Ti partitions.

**Architecture:** Treat the two generated Slurm files as immutable templates. Download them into a permission-restricted temporary directory, mechanically derive six embodiment-specific files, validate them locally, upload only the validated files, and then perform read-only remote verification without submitting jobs.

**Tech Stack:** Bash, Slurm, `just`, SSH/SCP, Git

## Global Constraints

- Remote root: `/home/kordos/mamba_env_data/env_id_116_reward_baselines`.
- Remote repository: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/constraints-as-terminations`.
- Seeds must be exactly `46 47 48 49 50 51 52 53 54`.
- L40S launchers must retain `--array=0-2%3` and `RUNS_PER_NODE=3`.
- 2080 Ti launchers must retain `--array=0-8%9` and `RUNS_PER_NODE=1`.
- Every launcher must use `NUM_ENVS=7500` and its existing `train-baseline-*` recipe.
- Preserve resource, mail, environment, startup-delay, failure-propagation, and W&B authentication settings from the corresponding generated template.
- Never print or embed authentication values in commands, logs, plans, or responses.
- Do not modify `slurm-config.sbatch` or `slurm-config-2080ti.sbatch`.
- Do not modify the remote Git checkout.
- Do not invoke `sbatch`, Isaac Lab, training, evaluation, or a simulator.

---

### Task 1: Prove the launchers are absent and the remote source state is safe

**Files:**
- Read: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-config.sbatch`
- Read: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-config-2080ti.sbatch`
- Read: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/constraints-as-terminations/justfile`

**Interfaces:**
- Consumes: the two generated Slurm templates and the three public baseline recipes at remote commit `42df189`.
- Produces: evidence that the six-file requirement currently fails because the destination files do not exist, plus source-template hashes for later immutability checks.

- [ ] **Step 1: Run the failing existence assertion**

Run this read-only command through an interactive SSH session, entering the
user-provided password only at the terminal prompt:

```bash
remote_root=/home/kordos/mamba_env_data/env_id_116_reward_baselines
test "$(find "$remote_root" -maxdepth 1 -type f -name 'slurm-baseline-*.sbatch' | wc -l)" -eq 6
```

Expected before implementation: exit status 1 because no matching launchers
exist.

- [ ] **Step 2: Verify the immutable sources and repository**

Run remotely without displaying complete file contents:

```bash
remote_root=/home/kordos/mamba_env_data/env_id_116_reward_baselines
repo="$remote_root/constraints-as-terminations"
test -f "$remote_root/slurm-config.sbatch"
test -f "$remote_root/slurm-config-2080ti.sbatch"
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
test "$(git -C "$repo" branch --show-current)" = cross-embodiment
for recipe in train-baseline-go2 train-baseline-anymal-c train-baseline-spot; do
    just --justfile "$repo/justfile" --show "$recipe" >/dev/null
done
sha256sum "$remote_root/slurm-config.sbatch" "$remote_root/slurm-config-2080ti.sbatch"
```

Expected: all assertions pass and two hashes are recorded without exposing
template contents.

---

### Task 2: Derive and locally validate six standalone launchers

**Files:**
- Create temporarily: `/tmp/tcml-baseline-slurm.*/slurm-baseline-go2.sbatch`
- Create temporarily: `/tmp/tcml-baseline-slurm.*/slurm-baseline-anymal-c.sbatch`
- Create temporarily: `/tmp/tcml-baseline-slurm.*/slurm-baseline-spot.sbatch`
- Create temporarily: `/tmp/tcml-baseline-slurm.*/slurm-baseline-go2-2080ti.sbatch`
- Create temporarily: `/tmp/tcml-baseline-slurm.*/slurm-baseline-anymal-c-2080ti.sbatch`
- Create temporarily: `/tmp/tcml-baseline-slurm.*/slurm-baseline-spot-2080ti.sbatch`

**Interfaces:**
- Consumes: exact copies of the two generated remote templates.
- Produces: six syntax-checked files whose only semantic differences are job identity and selected public baseline recipe.

- [ ] **Step 1: Create a restricted staging directory and download the templates**

Run locally, entering the password only at the SCP prompt:

```bash
stage_dir=$(mktemp -d /tmp/tcml-baseline-slurm.XXXXXX)
chmod 700 "$stage_dir"
scp \
  tcml-cluster:/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-config.sbatch \
  tcml-cluster:/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-config-2080ti.sbatch \
  "$stage_dir/"
```

Do not display the downloaded templates. Keep `stage_dir` for the remaining
steps and delete it after remote verification.

- [ ] **Step 2: Generate the files with one mechanical transformation**

Run locally:

```bash
make_launcher() {
    source_file="$1"
    destination_file="$2"
    job_name="$3"
    train_recipe="$4"

    cp "$source_file" "$destination_file"
    sed -i -E \
        -e "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${job_name}|" \
        -e "s|^TASK_NAME=.*|TRAIN_RECIPE=\"${train_recipe}\"|" \
        -e 's|^    just train .*|    just "${TRAIN_RECIPE}" "${NUM_ENVS}" "${seed}"|' \
        "$destination_file"
    chmod 600 "$destination_file"
}

make_launcher "$stage_dir/slurm-config.sbatch" \
    "$stage_dir/slurm-baseline-go2.sbatch" baseline-go2 train-baseline-go2
make_launcher "$stage_dir/slurm-config.sbatch" \
    "$stage_dir/slurm-baseline-anymal-c.sbatch" baseline-anymal-c train-baseline-anymal-c
make_launcher "$stage_dir/slurm-config.sbatch" \
    "$stage_dir/slurm-baseline-spot.sbatch" baseline-spot train-baseline-spot
make_launcher "$stage_dir/slurm-config-2080ti.sbatch" \
    "$stage_dir/slurm-baseline-go2-2080ti.sbatch" baseline-go2-2080ti train-baseline-go2
make_launcher "$stage_dir/slurm-config-2080ti.sbatch" \
    "$stage_dir/slurm-baseline-anymal-c-2080ti.sbatch" baseline-anymal-c-2080ti train-baseline-anymal-c
make_launcher "$stage_dir/slurm-config-2080ti.sbatch" \
    "$stage_dir/slurm-baseline-spot-2080ti.sbatch" baseline-spot-2080ti train-baseline-spot
```

- [ ] **Step 3: Validate syntax, seeds, scheduling, and recipe selection locally**

Run locally without printing complete files:

```bash
expected_seeds="46 47 48 49 50 51 52 53 54"

for file in "$stage_dir"/slurm-baseline-*.sbatch; do
    bash -n "$file"
    actual_seeds=$(sed -n '/^SEEDS=(/,/^)/p' "$file" | rg -o '[0-9]+' | paste -sd ' ' -)
    test "$actual_seeds" = "$expected_seeds"
    test "$(rg -c '^NUM_ENVS=7500$' "$file")" -eq 1
    test "$(rg -c '^TRAIN_RECIPE="train-baseline-(go2|anymal-c|spot)"$' "$file")" -eq 1
    test "$(rg -c '^    just "\$\{TRAIN_RECIPE\}" "\$\{NUM_ENVS\}" "\$\{seed\}"$' "$file")" -eq 1
done

for file in \
    "$stage_dir/slurm-baseline-go2.sbatch" \
    "$stage_dir/slurm-baseline-anymal-c.sbatch" \
    "$stage_dir/slurm-baseline-spot.sbatch"; do
    rg -q '^#SBATCH --partition=L40Sday$' "$file"
    rg -q '^#SBATCH --gres=gpu:L40S:1$' "$file"
    rg -q '^#SBATCH --array=0-2%3$' "$file"
    rg -q '^RUNS_PER_NODE=3$' "$file"
done

for file in "$stage_dir"/slurm-baseline-*-2080ti.sbatch; do
    rg -q '^#SBATCH --partition=week$' "$file"
    rg -q '^#SBATCH --gres=gpu:2080ti:1$' "$file"
    rg -q '^#SBATCH --array=0-8%9$' "$file"
    rg -q '^RUNS_PER_NODE=1$' "$file"
done

validate_delta() {
    source_file="$1"
    destination_file="$2"
    delta=$(diff -U0 "$source_file" "$destination_file" || true)
    test "$(printf '%s\n' "$delta" | grep -Ec '^[+-][^+-]')" -eq 6
    unexpected=$(printf '%s\n' "$delta" \
        | grep -E '^[+-][^+-]' \
        | grep -Ev '^[+-](#SBATCH --job-name=|TASK_NAME=|TRAIN_RECIPE=|    just )' \
        || true)
    test -z "$unexpected"
}

for file in \
    "$stage_dir/slurm-baseline-go2.sbatch" \
    "$stage_dir/slurm-baseline-anymal-c.sbatch" \
    "$stage_dir/slurm-baseline-spot.sbatch"; do
    validate_delta "$stage_dir/slurm-config.sbatch" "$file"
done

for file in "$stage_dir"/slurm-baseline-*-2080ti.sbatch; do
    validate_delta "$stage_dir/slurm-config-2080ti.sbatch" "$file"
done
```

Expected: every assertion exits successfully and no authentication value is
printed.

---

### Task 3: Upload and verify the launchers without submitting jobs

**Files:**
- Create: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-baseline-go2.sbatch`
- Create: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-baseline-anymal-c.sbatch`
- Create: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-baseline-spot.sbatch`
- Create: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-baseline-go2-2080ti.sbatch`
- Create: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-baseline-anymal-c-2080ti.sbatch`
- Create: `/home/kordos/mamba_env_data/env_id_116_reward_baselines/slurm-baseline-spot-2080ti.sbatch`

**Interfaces:**
- Consumes: the six validated staging files from Task 2.
- Produces: six directly submit-able remote Slurm files, with the original templates and remote Git checkout unchanged.

- [ ] **Step 1: Upload only the six validated launchers**

Run locally, entering the password only at the SCP prompt:

```bash
scp "$stage_dir"/slurm-baseline-*.sbatch \
    tcml-cluster:/home/kordos/mamba_env_data/env_id_116_reward_baselines/
```

- [ ] **Step 2: Verify remote syntax and safe fields**

Run remotely without displaying complete files:

```bash
remote_root=/home/kordos/mamba_env_data/env_id_116_reward_baselines
expected_seeds="46 47 48 49 50 51 52 53 54"
test "$(find "$remote_root" -maxdepth 1 -type f -name 'slurm-baseline-*.sbatch' | wc -l)" -eq 6

for file in "$remote_root"/slurm-baseline-*.sbatch; do
    bash -n "$file"
    test "$(stat -c '%a' "$file")" = 600
    actual_seeds=$(sed -n '/^SEEDS=(/,/^)/p' "$file" | grep -Eo '[0-9]+' | paste -sd ' ' -)
    test "$actual_seeds" = "$expected_seeds"
    test "$(grep -Ec '^NUM_ENVS=7500$' "$file")" -eq 1
    test "$(grep -Ec '^TRAIN_RECIPE="train-baseline-(go2|anymal-c|spot)"$' "$file")" -eq 1
done

for file in \
    "$remote_root/slurm-baseline-go2.sbatch" \
    "$remote_root/slurm-baseline-anymal-c.sbatch" \
    "$remote_root/slurm-baseline-spot.sbatch"; do
    grep -q '^#SBATCH --array=0-2%3$' "$file"
    grep -q '^RUNS_PER_NODE=3$' "$file"
done

for file in "$remote_root"/slurm-baseline-*-2080ti.sbatch; do
    grep -q '^#SBATCH --array=0-8%9$' "$file"
    grep -q '^RUNS_PER_NODE=1$' "$file"
done
```

- [ ] **Step 3: Verify final `just` command expansion**

Run remotely from the repository without executing the expanded commands:

```bash
repo=/home/kordos/mamba_env_data/env_id_116_reward_baselines/constraints-as-terminations

just --justfile "$repo/justfile" --dry-run train-baseline-go2 7500 46 2>&1 \
    | grep -q 'Baseline-Go2-Rough-Terrain-v0 7500 46 30000 baseline_go2'
just --justfile "$repo/justfile" --dry-run train-baseline-anymal-c 7500 46 2>&1 \
    | grep -q 'Baseline-Anymal-C-Rough-Terrain-v0 7500 46 30000 baseline_anymal_c'
just --justfile "$repo/justfile" --dry-run train-baseline-spot 7500 46 2>&1 \
    | grep -q 'Baseline-Spot-Rough-Terrain-v0 7500 46 30000 baseline_spot'

just --justfile "$repo/justfile" --dry-run _train-rsl-baseline \
    Baseline-Go2-Rough-Terrain-v0 7500 46 30000 baseline_go2 2>&1 \
    | grep -q -- '--max_iterations=30000.*--logger=wandb.*--log_project_name=baseline_go2'
just --justfile "$repo/justfile" --dry-run _train-rsl-baseline \
    Baseline-Anymal-C-Rough-Terrain-v0 7500 46 30000 baseline_anymal_c 2>&1 \
    | grep -q -- '--max_iterations=30000.*--logger=wandb.*--log_project_name=baseline_anymal_c'
just --justfile "$repo/justfile" --dry-run _train-rsl-baseline \
    Baseline-Spot-Rough-Terrain-v0 7500 46 30000 baseline_spot 2>&1 \
    | grep -q -- '--max_iterations=30000.*--logger=wandb.*--log_project_name=baseline_spot'
```

- [ ] **Step 4: Prove source and repository immutability**

Run remotely:

```bash
remote_root=/home/kordos/mamba_env_data/env_id_116_reward_baselines
repo="$remote_root/constraints-as-terminations"
sha256sum "$remote_root/slurm-config.sbatch" "$remote_root/slurm-config-2080ti.sbatch"
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
git -C "$repo" status --short
```

Expected: template hashes equal the values recorded in Task 1 and the Git
status is empty.

- [ ] **Step 5: Remove the credential-bearing local staging directory**

Run locally only after validating the exact path prefix:

```bash
case "$stage_dir" in
    /tmp/tcml-baseline-slurm.*) rm -r -- "$stage_dir" ;;
    *) printf 'Refusing to remove unexpected path: %s\n' "$stage_dir" >&2; exit 1 ;;
esac
```

Expected: the temporary copies are removed. No Git implementation commit is
created because the six deliverables live outside the repository; the design
and plan commits provide the local audit trail.
