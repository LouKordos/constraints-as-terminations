# Baseline W&B Defaults Design

**Goal:** Make every matched-baseline `just` training recipe use Weights &
Biases by default, with an embodiment-specific project name.

## Current behavior

The installed Isaac Lab RSL-RL configuration defaults to `logger: tensorboard`.
The completed baseline smoke runs therefore did not initialize W&B. The
existing CaT CleanRL agent already defaults to W&B and is unchanged by this
work.

## Approved behavior

The three public baseline recipes receive a `wandb_project` parameter after
their existing `max_iterations` parameter:

- Go2: `baseline_go2`
- ANYmal C: `baseline_anymal_c`
- Spot: `baseline_spot`

The shared `_train-rsl-baseline` recipe passes the selected value to the
installed trainer using its supported CLI arguments:

```text
--logger=wandb --log_project_name=<wandb_project>
```

Existing invocations that provide only environment count, seed, and maximum
iterations remain valid because `wandb_project` has a default in each public
recipe. Callers may choose another project through the fourth positional
argument. Existing variadic RSL-RL/Hydra flags remain supported.

## Scope and verification

Only `justfile` changes. Environment, reward, action, training algorithm,
evaluation, and sim2real behavior remain untouched.

Verification consists of `just --list`, dry-run expansion for all three public
recipes through the shared recipe, `git diff --check`, and inspection against
the installed RSL-RL CLI implementation to confirm that these arguments set
`agent_cfg.logger` and `agent_cfg.wandb_project`.
