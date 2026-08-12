"""Construct a LoComposition task and check a short finite rollout."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def assert_finite_tree(value: Any, *, label: str) -> None:
    """Assert that every numeric leaf in a nested simulator result is finite."""

    if isinstance(value, Mapping):
        for name, child in value.items():
            assert_finite_tree(child, label=f"{label}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_finite_tree(child, label=f"{label}[{index}]")
        return

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        return
    if not np.isfinite(array).all():
        raise AssertionError(f"{label} contains NaN or infinite values")


def _term_param(namespace: Any, term_name: str, parameter: str) -> Any:
    term = getattr(namespace, term_name, None)
    if term is None:
        return None
    return term.params.get(parameter)


def summarize_task_config(cfg: Any) -> dict[str, Any]:
    """Return the embodiment-specific values that must survive the rebrand."""

    constraint_names = (
        "joint_torque",
        "joint_velocity",
        "joint_acceleration",
        "action_rate",
        "base_orientation",
    )
    return {
        "action_scale": cfg.actions.joint_pos.scale,
        "mass_delta": _term_param(
            cfg.events, "randomize_mass", "mass_distribution_params"
        ),
        "push_velocity": _term_param(cfg.events, "push_robot", "velocity_range"),
        "wrench_force": _term_param(cfg.events, "push_base_wrench", "force_range"),
        "energy_end_weight": _term_param(cfg.curriculum, "power", "end_weight"),
        "constraint_limits": {
            name: _term_param(cfg.constraints, name, "limit")
            for name in constraint_names
        },
    }


def _shape_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _shape_tree(child) for name, child in value.items()}
    if hasattr(value, "shape"):
        return list(value.shape)
    return type(value).__name__


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Construct one LoComposition environment and step zero actions."
    )
    parser.add_argument("--task", required=True, help="Registered Gym task ID.")
    parser.add_argument(
        "--steps", type=int, default=25, help="Number of zero-action policy steps."
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        help="Use USD I/O instead of Fabric when constructing the environment.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    launcher = AppLauncher(args)
    environment = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import locomposition.tasks  # noqa: F401

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=1,
            use_fabric=not args.disable_fabric,
        )
        environment = gym.make(args.task, cfg=env_cfg)
        unwrapped = environment.unwrapped

        observations, _ = environment.reset()
        assert_finite_tree(observations, label="reset_observation")

        action_dimension = unwrapped.action_manager.total_action_dim
        if action_dimension != 12:
            raise AssertionError(
                f"Expected a 12-dimensional quadruped action, got {action_dimension}"
            )
        action = torch.zeros(
            (unwrapped.num_envs, action_dimension), device=unwrapped.device
        )

        for step in range(args.steps):
            observations, rewards, _, _, _ = environment.step(action)
            assert_finite_tree(
                observations, label=f"step_{step + 1}_observation"
            )
            assert_finite_tree(rewards, label=f"step_{step + 1}_reward")

        summary = {
            "task": args.task,
            "config_class": type(env_cfg).__name__,
            "action_dimension": action_dimension,
            "observation_shapes": _shape_tree(observations),
            "steps": args.steps,
            "embodiment_scaling": summarize_task_config(env_cfg),
        }
        print(json.dumps(summary, indent=2, default=str), flush=True)
    except BaseException:
        # SimulationApp.close() terminates this Isaac Sim build before Python
        # can render a pending exception, so emit it while the app is alive.
        traceback.print_exc()
        raise
    finally:
        if environment is not None:
            environment.close()
        try:
            launcher.app.close()
        except SystemExit:
            # Preserve a pending smoke-test exception instead of replacing it
            # with SimulationApp.close()'s successful SystemExit.
            pass


if __name__ == "__main__":
    main()
