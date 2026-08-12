from __future__ import annotations

import numpy as np
import pytest
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_task.py"
SPEC = importlib.util.spec_from_file_location("smoke_task", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SMOKE_TASK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE_TASK)
assert_finite_tree = SMOKE_TASK.assert_finite_tree
summarize_task_config = SMOKE_TASK.summarize_task_config


def test_assert_finite_tree_accepts_nested_numeric_observations() -> None:
    assert_finite_tree(
        {
            "policy": np.zeros((1, 12), dtype=np.float32),
            "critic": [np.ones((1, 3), dtype=np.float32)],
        },
        label="observation",
    )


def test_assert_finite_tree_reports_non_finite_leaf() -> None:
    with pytest.raises(AssertionError, match=r"observation\.policy"):
        assert_finite_tree(
            {"policy": np.array([[0.0, np.nan]], dtype=np.float32)},
            label="observation",
        )


def test_summarize_task_config_captures_embodiment_scaling() -> None:
    class Term:
        def __init__(self, params):
            self.params = params

    class Namespace:
        pass

    cfg = Namespace()
    cfg.actions = Namespace()
    cfg.actions.joint_pos = Namespace()
    cfg.actions.joint_pos.scale = 0.5
    cfg.events = Namespace()
    cfg.events.randomize_mass = Term({"mass_distribution_params": (-5.0, 5.0)})
    cfg.events.push_robot = Term({"velocity_range": {"x": (-0.3, 0.3)}})
    cfg.events.push_base_wrench = Term({"force_range": (-30.0, 30.0)})
    cfg.curriculum = Namespace()
    cfg.curriculum.power = Term({"end_weight": 0.0018})
    cfg.constraints = Namespace()
    cfg.constraints.joint_torque = Term({"limit": 80.0})
    cfg.constraints.joint_velocity = Term({"limit": 12.0})
    cfg.constraints.joint_acceleration = Term({"limit": 800.0})
    cfg.constraints.action_rate = Term({"limit": 128.0})
    cfg.constraints.base_orientation = Term({"limit": 0.1})

    assert summarize_task_config(cfg) == {
        "action_scale": 0.5,
        "mass_delta": (-5.0, 5.0),
        "push_velocity": {"x": (-0.3, 0.3)},
        "wrench_force": (-30.0, 30.0),
        "energy_end_weight": 0.0018,
        "constraint_limits": {
            "joint_torque": 80.0,
            "joint_velocity": 12.0,
            "joint_acceleration": 800.0,
            "action_rate": 128.0,
            "base_orientation": 0.1,
        },
    }
