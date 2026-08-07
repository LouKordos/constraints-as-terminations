from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REWARDS_PATH = REPOSITORY_ROOT / "exts/cat_envs/cat_envs/tasks/utils/mdp/rewards.py"


class FakeEnv:
    def __init__(self, common_step_counter: int):
        self.common_step_counter = common_step_counter


@pytest.fixture
def rewards_module(monkeypatch):
    class SceneEntityCfg:
        def __init__(self, name: str):
            self.name = name
            self.joint_ids = slice(None)

    module_stubs = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": types.ModuleType("isaaclab.assets"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.managers.manager_base": types.ModuleType("isaaclab.managers.manager_base"),
        "isaaclab.managers.manager_term_cfg": types.ModuleType("isaaclab.managers.manager_term_cfg"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
    }
    module_stubs["isaaclab.assets"].Articulation = type("Articulation", (), {})
    module_stubs["isaaclab.assets"].RigidObject = type("RigidObject", (), {})
    module_stubs["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg
    module_stubs["isaaclab.managers.manager_base"].ManagerTermBase = type("ManagerTermBase", (), {})
    module_stubs["isaaclab.managers.manager_term_cfg"].RewardTermCfg = type("RewardTermCfg", (), {})
    module_stubs["isaaclab.sensors"].ContactSensor = type("ContactSensor", (), {})
    module_stubs["isaaclab.sensors"].RayCaster = type("RayCaster", (), {})

    for name, module in module_stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    package_names = (
        "cat_envs",
        "cat_envs.tasks",
        "cat_envs.tasks.utils",
        "cat_envs.tasks.utils.cat",
    )
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    constraints_module = types.ModuleType("cat_envs.tasks.utils.cat.constraints")
    for function_name in (
        "joint_torque",
        "joint_velocity",
        "joint_acceleration",
        "action_rate",
        "base_orientation",
    ):
        setattr(constraints_module, function_name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "cat_envs.tasks.utils.cat.constraints", constraints_module)

    spec = importlib.util.spec_from_file_location("reward_constraint_test_module", REWARDS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalized_constraint_penalty_uses_hinge_max_and_half_progress(rewards_module):
    violations = torch.tensor([[-1.0, 0.0, 2.0], [4.0, 1.0, -3.0]])

    actual = rewards_module._normalized_constraint_penalty(
        FakeEnv(common_step_counter=9_600),
        violations,
        limit=2.0,
        curriculum_steps=19_200,
    )

    torch.testing.assert_close(actual, torch.tensor([-0.5, -1.0]))


def test_normalized_constraint_penalty_is_zero_at_and_below_threshold(rewards_module):
    violations = torch.tensor([[-2.0, 0.0], [-0.5, -1.0]])

    actual = rewards_module._normalized_constraint_penalty(
        FakeEnv(common_step_counter=19_200),
        violations,
        limit=2.0,
        curriculum_steps=19_200,
    )

    torch.testing.assert_close(actual, torch.zeros(2))


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.0), (9_600, -0.5), (19_200, -1.0), (25_000, -1.0)],
)
def test_normalized_constraint_penalty_curriculum_endpoints(rewards_module, step, expected):
    actual = rewards_module._normalized_constraint_penalty(
        FakeEnv(step),
        torch.tensor([2.0]),
        limit=2.0,
        curriculum_steps=19_200,
    )

    torch.testing.assert_close(actual, torch.tensor([expected]))


@pytest.mark.parametrize(
    ("limit", "curriculum_steps", "message"),
    [
        (0.0, 19_200, "limit must be positive"),
        (1.0, 0, "curriculum_steps must be positive"),
    ],
)
def test_normalized_constraint_penalty_rejects_invalid_configuration(
    rewards_module,
    limit,
    curriculum_steps,
    message,
):
    with pytest.raises(ValueError, match=message):
        rewards_module._normalized_constraint_penalty(
            FakeEnv(0),
            torch.tensor([1.0]),
            limit=limit,
            curriculum_steps=curriculum_steps,
        )


def test_normalized_constraint_penalty_rejects_unexpected_tensor_rank(rewards_module):
    with pytest.raises(ValueError, match="constraint_violation must have shape"):
        rewards_module._normalized_constraint_penalty(
            FakeEnv(0),
            torch.ones(2, 3, 4),
            limit=1.0,
            curriculum_steps=19_200,
        )


@pytest.mark.parametrize(
    (
        "wrapper_name",
        "constraint_name",
        "limit",
        "names",
        "constraint_violation",
    ),
    [
        (
            "joint_torque_limit_penalty",
            "joint_torque",
            20.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            torch.tensor([[0.0, 10.0]]),
        ),
        (
            "joint_velocity_limit_penalty",
            "joint_velocity",
            25.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            torch.tensor([[0.0, 12.5]]),
        ),
        (
            "joint_acceleration_limit_penalty",
            "joint_acceleration",
            800.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            torch.tensor([[0.0, 400.0]]),
        ),
        (
            "action_rate_limit_penalty",
            "action_rate",
            80.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            torch.tensor([[0.0, 40.0]]),
        ),
        (
            "base_orientation_limit_penalty",
            "base_orientation",
            0.1,
            None,
            torch.tensor([0.05]),
        ),
    ],
)
def test_limit_penalty_uses_matching_constraint_quantity_and_parameters(
    rewards_module,
    monkeypatch,
    wrapper_name,
    constraint_name,
    limit,
    names,
    constraint_violation,
):
    received = {}

    def fake_constraint(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return constraint_violation

    monkeypatch.setattr(rewards_module.constraints, constraint_name, fake_constraint)
    env = FakeEnv(common_step_counter=19_200)
    call_kwargs = {
        "limit": limit,
        "curriculum_steps": 19_200,
    }
    if names is not None:
        call_kwargs["names"] = names

    actual = getattr(rewards_module, wrapper_name)(env, **call_kwargs)

    torch.testing.assert_close(actual, torch.tensor([-0.5]))
    assert received["args"] == (env,)
    assert received["kwargs"]["limit"] == limit
    assert received["kwargs"]["asset_cfg"].name == "robot"
    if names is None:
        assert "names" not in received["kwargs"]
    else:
        assert received["kwargs"]["names"] == names
