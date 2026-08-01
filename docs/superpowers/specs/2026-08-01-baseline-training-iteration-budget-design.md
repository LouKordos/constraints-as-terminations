# Baseline Training Iteration Budget Design

**Date:** 2026-08-01

**Status:** Awaiting written-specification review

## Purpose

Increase the default training duration of the matched Go2, ANYmal-C, and Spot
reward-based baselines so each has enough optimization budget to converge or
reach a steady-state plateau. Use the same maximum iteration count as the CaT
training configuration to avoid making final terrain progression depend on a
smaller baseline transition budget.

## Design

Change only the public baseline recipe defaults in `justfile`:

- `train-baseline-go2`: 30,000 iterations
- `train-baseline-anymal-c`: 30,000 iterations
- `train-baseline-spot`: 30,000 iterations

Keep the existing `max_iterations` recipe argument. Explicit command-line
values must continue to override the default, allowing short startup checks
and continuation experiments without another source change.

Do not change the installed upstream RSL-RL agent configurations. The shared
recipe already forwards `--max_iterations` to the trainer, so no trainer,
environment, reward, observation, evaluation, or sim2real changes are needed.

At the default 7,500 environments and 24 rollout steps, 30,000 iterations
permit up to 5.4 billion environment transitions per run. This is a maximum
budget, not evidence of convergence: training curves must still be inspected
for tracking, terrain-level, reward, and policy-statistics plateaus. A job
started before this edit retains the iteration count passed at launch.

## Verification

- `just --list` reports 30,000 for all three public recipes.
- Dry-run expansion forwards 30,000 to the shared recipe for every embodiment.
- A custom `max_iterations` value still overrides the default.
- `git diff --check` reports no whitespace errors.

No simulator startup or W&B run is required because this change affects only
command construction.
