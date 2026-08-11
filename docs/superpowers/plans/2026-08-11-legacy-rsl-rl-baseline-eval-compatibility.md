# Legacy RSL-RL Baseline Evaluation Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the repository's archived legacy RSL-RL baseline checkpoints without changing dependency versions or regressing the existing CleanRL Go2, ANYmal-C, and Spot paths.

**Architecture:** Add one inference-only adapter at the RSL-RL checkpoint-loading boundary in `scripts/eval.py`. The adapter reconstructs the legacy sequential actor from `model_state_dict`, validates it strictly, and exposes the evaluator's existing observation-mapping call contract; the installed RSL-RL runner remains untouched for unrelated formats.

**Tech Stack:** Python 3.11, PyTorch, Gymnasium, Isaac Lab, pytest, CUDA.

## Global Constraints

- Do not downgrade or pin RSL-RL or any other dependency.
- Do not change environment configuration, action scales, reward weights, constraints, or robot profiles.
- Do not refactor unrelated portions of `scripts/eval.py`.
- Keep CleanRL evaluation behavior unchanged.
- Ignore the entire `sim2real/` directory.
- After all implementation tests, CUDA evals, and review are complete, delete only the repository's top-level `tests/` and `docs/` trees as explicitly requested.

---

### Task 1: Legacy deterministic actor adapter

**Files:**
- Modify: `scripts/eval.py:400-445`
- Modify: `tests/test_eval_cross_embodiment_merge.py`

**Interfaces:**
- Consumes: a legacy checkpoint mapping with `model_state_dict`, an activation name, expected observation/action dimensions, and a PyTorch device.
- Produces: `build_legacy_rsl_rl_policy(checkpoint_object, *, activation_name, expected_observation_dim, expected_action_dim, device) -> torch.nn.Module` whose `forward(observations)` accepts `{"policy": tensor}`.

- [ ] **Step 1: Write the failing deterministic-inference test**

Add a synthetic two-layer legacy actor. The input and hidden values remain positive, so ELU is the identity and the hand-derived output is `24.0`.

```python
def test_legacy_rsl_rl_actor_reconstructs_deterministic_policy() -> None:
    eval_module = _load_eval_module()
    checkpoint = {
        "model_state_dict": {
            "std": torch.ones(1),
            "actor.0.weight": torch.eye(2),
            "actor.0.bias": torch.zeros(2),
            "actor.2.weight": torch.tensor([[4.0, 5.0]]),
            "actor.2.bias": torch.tensor([1.0]),
            "critic.0.weight": torch.full((1, 2), 99.0),
            "critic.0.bias": torch.full((1,), 99.0),
        }
    }

    policy = eval_module.build_legacy_rsl_rl_policy(
        checkpoint,
        activation_name="elu",
        expected_observation_dim=2,
        expected_action_dim=1,
        device=torch.device("cpu"),
    )

    output = policy({"policy": torch.tensor([[2.0, 3.0]])})
    torch.testing.assert_close(output, torch.tensor([[24.0]]))
```

- [ ] **Step 2: Write malformed-state and dimension-validation tests**

Add literal fixtures proving that an actor missing `actor.0.bias`, an actor with the wrong first-layer input size, and an unsupported activation each raise a descriptive `ValueError`. Include one regression assertion that a plain tensor state dict still resolves as `clean_rl`.

```python
def test_legacy_rsl_rl_actor_rejects_incomplete_linear_layer() -> None:
    eval_module = _load_eval_module()
    checkpoint = {"model_state_dict": {"actor.0.weight": torch.eye(2)}}

    with pytest.raises(ValueError, match="weight and bias"):
        eval_module.build_legacy_rsl_rl_policy(
            checkpoint,
            activation_name="elu",
            expected_observation_dim=2,
            expected_action_dim=2,
            device=torch.device("cpu"),
        )
```

```python
def test_legacy_rsl_rl_actor_rejects_wrong_observation_dimension() -> None:
    eval_module = _load_eval_module()
    checkpoint = {
        "model_state_dict": {
            "actor.0.weight": torch.eye(2),
            "actor.0.bias": torch.zeros(2),
        }
    }

    with pytest.raises(ValueError, match="observation dimension 3"):
        eval_module.build_legacy_rsl_rl_policy(
            checkpoint,
            activation_name="elu",
            expected_observation_dim=3,
            expected_action_dim=2,
            device=torch.device("cpu"),
        )


def test_legacy_rsl_rl_actor_rejects_unknown_activation() -> None:
    eval_module = _load_eval_module()
    checkpoint = {
        "model_state_dict": {
            "actor.0.weight": torch.eye(2),
            "actor.0.bias": torch.zeros(2),
            "actor.2.weight": torch.eye(2),
            "actor.2.bias": torch.zeros(2),
        }
    }

    with pytest.raises(ValueError, match="Unsupported legacy RSL-RL activation"):
        eval_module.build_legacy_rsl_rl_policy(
            checkpoint,
            activation_name="swish",
            expected_observation_dim=2,
            expected_action_dim=2,
            device=torch.device("cpu"),
        )


def test_cleanrl_backend_detection_remains_unchanged() -> None:
    eval_module = _load_eval_module()

    assert eval_module.detect_policy_backend_from_checkpoint({"actor.weight": torch.eye(2)}) == "clean_rl"
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
source ~/mamba_env_data/env_new_isaac_lab/.venv/bin/activate
pytest -q tests/test_eval_cross_embodiment_merge.py -k 'legacy_rsl_rl'
```

Expected: FAIL because `build_legacy_rsl_rl_policy` does not exist.

- [ ] **Step 4: Implement the minimal actor adapter**

Add a small `LegacyRslRlPolicy` wrapper and builder near the existing checkpoint-format helpers. Match only keys of the form `actor.<even index>.weight|bias`, require indices `0, 2, ...`, validate tensor ranks and adjacent dimensions, instantiate `torch.nn.Linear` plus the selected activation between layers, and load the stripped actor state strictly.

```python
class LegacyRslRlPolicy(torch.nn.Module):
    def __init__(self, actor: torch.nn.Sequential) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            policy_observation = observations["policy"]
        except (KeyError, TypeError) as exception:
            raise ValueError("Legacy RSL-RL policy requires a 'policy' observation tensor.") from exception
        return self.actor(policy_observation)
```

Support only the activation names already represented by repository agent configurations (`elu`, `relu`, `selu`, `tanh`) and reject other names. Do not load `critic.*`, `std`, or optimizer data.

- [ ] **Step 5: Run the targeted tests and verify GREEN**

Run:

```bash
source ~/mamba_env_data/env_new_isaac_lab/.venv/bin/activate
pytest -q tests/test_eval_cross_embodiment_merge.py
```

Expected: all tests pass with no new warnings.

- [ ] **Step 6: Wire only legacy checkpoints to the adapter**

In the `policy_backend == "rsl_rl"` branch, retain registry loading for activation and action clipping. When `checkpoint_object` contains `model_state_dict`, validate `agent_cfg.policy.actor_obs_normalization is False`, obtain expected dimensions from `env.observation_space["policy"].shape[-1]` and `env.action_space.shape[-1]`, build the legacy policy, and skip `RslRlVecEnvWrapper`/`OnPolicyRunner`. Preserve the current runner code in the non-legacy branch.

```python
if "model_state_dict" in checkpoint_object:
    if agent_cfg.policy.actor_obs_normalization:
        raise ValueError("Legacy RSL-RL actor adapter does not support observation normalization.")
    rsl_rl_policy = build_legacy_rsl_rl_policy(
        checkpoint_object,
        activation_name=agent_cfg.policy.activation,
        expected_observation_dim=env.observation_space["policy"].shape[-1],
        expected_action_dim=env.action_space.shape[-1],
        device=device,
    )
    rsl_rl_clip_actions = agent_cfg.clip_actions
else:
    # Existing installed-runner path remains unchanged.
```

- [ ] **Step 7: Run the first real baseline CUDA eval**

Use the already-audited temporary Go2 baseline run:

```bash
source ~/mamba_env_data/env_new_isaac_lab/.venv/bin/activate
python scripts/eval.py --headless \
  --run_dir=/tmp/cross-embodiment-eval-smoke.6DFlpB/baseline_go2 \
  --task=Baseline-Go2-Rough-Terrain-Play-v0 --policy_backend=rsl_rl \
  --random_sim_step_length=0 --fixed_scenario=stand_still \
  --fixed_command_sim_steps=100 --skip_cot_sweep \
  --num_plot_jobs_in_parallel=1 --plot_job_stagger_delay=0
```

Expected: legacy actor loads, 100 CUDA steps complete, metrics are serialized, both plot jobs exit 0, and the process exits 0.

- [ ] **Step 8: Commit the independently working adapter**

```bash
git add scripts/eval.py tests/test_eval_cross_embodiment_merge.py
git commit -m "fix: evaluate legacy RSL-RL baseline actors"
```

---

### Task 2: Priority-path and baseline runtime regression verification

**Files:**
- Verify: `scripts/eval.py`
- Verify: `tests/`
- Outputs only: `/tmp/cross-embodiment-eval-smoke.6DFlpB/`

**Interfaces:**
- Consumes: the committed legacy adapter and the six already-audited temporary run directories.
- Produces: fresh unit-test, compilation, and CUDA runtime evidence before cleanup.

- [ ] **Step 1: Run repository tests without entering `sim2real/`**

```bash
source ~/mamba_env_data/env_new_isaac_lab/.venv/bin/activate
pytest -q tests
python -m compileall -q scripts exts/cat_envs/cat_envs/tasks tests
git diff --check
```

Expected: every top-level test passes, compilation succeeds, and the Git check produces no output.

- [ ] **Step 2: Re-run the three priority CleanRL CUDA evals**

Run these exact 100-step stand-still commands from the repository root:

```bash
source ~/mamba_env_data/env_new_isaac_lab/.venv/bin/activate
python scripts/eval.py --headless \
  --run_dir=/tmp/cross-embodiment-eval-smoke.6DFlpB/custom_go2 \
  --random_sim_step_length=0 --fixed_scenario=stand_still \
  --fixed_command_sim_steps=100 --skip_cot_sweep \
  --num_plot_jobs_in_parallel=1 --plot_job_stagger_delay=0

python scripts/eval.py --headless \
  --run_dir=/tmp/cross-embodiment-eval-smoke.6DFlpB/custom_anymal \
  --task=CaT-Anymal-C-Rough-Terrain-Play-v0 \
  --random_sim_step_length=0 --fixed_scenario=stand_still \
  --fixed_command_sim_steps=100 --skip_cot_sweep \
  --num_plot_jobs_in_parallel=1 --plot_job_stagger_delay=0

python scripts/eval.py --headless \
  --run_dir=/tmp/cross-embodiment-eval-smoke.6DFlpB/custom_spot \
  --task=CaT-Spot-Rough-Terrain-Play-v0 \
  --random_sim_step_length=0 --fixed_scenario=stand_still \
  --fixed_command_sim_steps=100 --skip_cot_sweep \
  --num_plot_jobs_in_parallel=1 --plot_job_stagger_delay=0
```

Expected: each process exits 0, completes 100 steps, serializes metrics, and reports plot exit code 0 twice.

- [ ] **Step 3: Run the remaining baseline CUDA evals**

Run:

```bash
source ~/mamba_env_data/env_new_isaac_lab/.venv/bin/activate
python scripts/eval.py --headless \
  --run_dir=/tmp/cross-embodiment-eval-smoke.6DFlpB/baseline_anymal \
  --task=Baseline-Anymal-C-Rough-Terrain-Play-v0 --policy_backend=rsl_rl \
  --random_sim_step_length=0 --fixed_scenario=stand_still \
  --fixed_command_sim_steps=100 --skip_cot_sweep \
  --num_plot_jobs_in_parallel=1 --plot_job_stagger_delay=0

python scripts/eval.py --headless \
  --run_dir=/tmp/cross-embodiment-eval-smoke.6DFlpB/baseline_spot \
  --task=Baseline-Spot-Rough-Terrain-Play-v0 --policy_backend=rsl_rl \
  --random_sim_step_length=0 --fixed_scenario=stand_still \
  --fixed_command_sim_steps=100 --skip_cot_sweep \
  --num_plot_jobs_in_parallel=1 --plot_job_stagger_delay=0
```

Expected: both processes exit 0 with matching robot profiles, 188 observations, 100 completed steps, serialized metrics, and two successful plot jobs.

- [ ] **Step 4: Inspect generated summaries and logs**

For all six output directories, verify `metrics_summary.json`, `plots/sim_data.npz`, and the scenario/overall plot logs exist. Search plot logs for `[FAILED]`, `Traceback`, and non-zero exit messages; the search must return no matches.

- [ ] **Step 5: Review the production diff**

Confirm that the production diff changes only checkpoint-loading/inference code in `scripts/eval.py` and does not alter the CleanRL branch, task registrations, environment configs, rewards, constraints, or robot profiles.

---

### Task 3: Requested repository cleanup

**Files:**
- Delete: `tests/`
- Delete: `docs/`
- Do not touch: `sim2real/`

**Interfaces:**
- Consumes: completed test, runtime, and review evidence from Task 2.
- Produces: a repository with no top-level tests or docs, as requested, while preserving the production evaluator fix.

- [ ] **Step 1: Resolve and report the exact deletion targets**

Run `git ls-files tests docs` and verify every target is under top-level `tests/` or `docs/`. Confirm `git status --short` contains no unrelated user changes.

- [ ] **Step 2: Delete the requested trees**

Delete the exact tracked top-level `tests/` and `docs/` targets with Git. Do not use a broad filesystem path, wildcard, environment variable, home-directory alias, or recursive target outside these two resolved repository directories.

```bash
git rm -r -- tests docs
```

- [ ] **Step 3: Verify the final production tree**

```bash
python -m compileall -q scripts exts/cat_envs/cat_envs/tasks
git diff --cached --check
git status --short
```

Expected: production Python compiles, the staged cleanup has no whitespace errors, all listed deletions are confined to `tests/` and `docs/`, and `scripts/eval.py` remains present.

- [ ] **Step 4: Commit the requested cleanup**

```bash
git commit -m "chore: remove tests and docs"
```

- [ ] **Step 5: Final repository-state verification**

Run `git status --short --branch`, `git log -3 --oneline`, and `git show --stat --oneline HEAD`. Confirm the working tree is clean and the cleanup commit deletes only top-level `tests/` and `docs/`.
