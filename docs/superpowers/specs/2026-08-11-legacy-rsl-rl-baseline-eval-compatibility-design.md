# Legacy RSL-RL Baseline Evaluation Compatibility

## Goal

Allow the archived matched-baseline checkpoints to run through `scripts/eval.py`
without downgrading RSL-RL or changing the working CleanRL Go2, ANYmal-C, and
Spot evaluation paths.

## Context

The archived baseline checkpoints store a legacy RSL-RL `model_state_dict`
containing `actor.*`, `critic.*`, and `std` tensors. The installed RSL-RL
runner expects separate modern actor and critic configurations and checkpoint
entries. Passing a legacy checkpoint through that runner currently fails during
runner construction with `KeyError: 'actor'`.

The legacy baseline policies do not use observation normalization. Evaluation
is deterministic, so only the saved actor network is needed; critic, optimizer,
and action-distribution standard-deviation state are training-only data.

## Design

Keep compatibility at the policy-loading boundary in `scripts/eval.py`:

1. Continue identifying `model_state_dict` checkpoints as legacy RSL-RL.
2. Extract only sequential `actor.<index>.weight` and
   `actor.<index>.bias` tensors.
3. Reconstruct the actor MLP from the saved tensor shapes, inserting the
   activation declared by the registered baseline agent configuration between
   linear layers.
4. Load the extracted weights strictly and move the actor to the evaluation
   device.
5. Wrap the actor with the existing RSL-RL policy call contract: accept the
   observation mapping and evaluate its `policy` tensor deterministically.
6. Bypass the installed runner only for this recognized legacy format; leave
   the CleanRL branch and unrelated RSL-RL handling unchanged.

The adapter will be a small, independently testable unit. It will not alter
environment configuration, action scale, reward weights, constraints, robot
profiles, or installed package versions.

## Validation and Errors

The legacy loader will reject malformed or unsupported checkpoints before
simulation with clear errors for:

- missing legacy `model_state_dict` data;
- missing or non-sequential actor tensors;
- incomplete weight/bias pairs;
- input dimensions that do not match the runtime policy observation;
- output dimensions that do not match the environment action space;
- unsupported activation names.

Strict weight loading prevents partial or silently miswired policies.

## Testing

Unit tests will first reproduce the current failure boundary and then verify:

- a synthetic legacy actor produces the hand-computed deterministic output;
- critic, optimizer, and `std` state do not affect inference;
- malformed legacy actor state fails clearly;
- CleanRL backend detection and policy loading remain unchanged.

Runtime verification will use the existing virtual environment and CUDA:

- repeat short Go2, ANYmal-C, and Spot CleanRL evals to guard the priority path;
- run short matched-baseline Go2, ANYmal-C, and Spot evals with their exact
  archived checkpoints;
- require successful policy rollout, metrics serialization, and plot jobs.

## Non-Goals

- Downgrading or pinning an older RSL-RL package.
- Supporting arbitrary historical RSL-RL architectures.
- Changing baseline training.
- Refactoring unrelated portions of the already-large evaluator.
