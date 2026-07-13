# RSL-RL W&B Writer Compatibility Design

## Problem

`PPO()` currently imports and constructs the legacy RSL-RL
`WandbSummaryWriter`, then unconditionally tries to import the newer
`WandbLogWriter`. In an RSL-RL 3.3.0 environment the legacy import succeeds,
so the fallback diagnostic is not printed, but the subsequent new-writer import
fails because that module does not exist.

## Scope

Keep the production change minimal and entirely inside the existing W&B logger
branch of `PPO()` in
`exts/cat_envs/cat_envs/tasks/utils/cleanrl/ppo.py`. Do not extract a helper,
detect package versions, or refactor unrelated PPO code. Focused regression
tests may be added outside `ppo.py` without introducing other production-code
changes.

## Design

Continue trying the legacy `WandbSummaryWriter` first. Only if its module is
absent should execution print the existing fallback diagnostic and try the
newer `WandbLogWriter`. If the legacy writer is constructed successfully, do
not attempt the new import.

Preserve the version-specific initialization behavior:

- Construct `WandbSummaryWriter` with `log_dir`, `flush_secs`, and the serialized
  PPO configuration.
- Construct `WandbLogWriter` with `log_dir` and `project_name`, then call
  `store_config` when that method is available.

## Error Handling

A fallback is valid only when the requested RSL-RL writer module itself is
absent. A `ModuleNotFoundError` raised for an internal dependency, such as a
missing `wandb` installation, must propagate unchanged. Constructor and
configuration errors must also propagate unchanged.

If neither RSL-RL writer module exists, raise a clear compatibility error that
names both supported writer APIs and suggests installing W&B/RSL-RL correctly
or selecting TensorBoard.

## Testing

Add a focused regression test around the existing `PPO()` entry point without
adding production helpers. Use lightweight stand-ins for the environment,
configuration, and external writer modules so the test does not initialize
Isaac Sim or contact W&B.

The test must demonstrate that an old-only installation constructs the legacy
writer and does not attempt the new import. Additional focused cases should
cover a new-only installation and verify that an internal dependency failure is
not mistaken for a version fallback. Run a syntax check and the focused tests;
the full `just train 400` verification remains GPU-dependent.

## Non-goals

- Changing logger selection outside `ppo.py`.
- Adding a compatibility abstraction or package-version table.
- Changing W&B project naming, run naming, or metric behavior.
- Addressing the CUDA-unavailable failure observed in the sandbox.
