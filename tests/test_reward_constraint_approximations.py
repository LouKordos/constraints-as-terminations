from __future__ import annotations

import ast
import importlib.util
import importlib
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REWARDS_PATH = REPOSITORY_ROOT / "exts/cat_envs/cat_envs/tasks/utils/mdp/rewards.py"
EVAL_PATH = REPOSITORY_ROOT / "scripts/eval.py"


class FakeEnv:
    def __init__(self, common_step_counter: int):
        self.common_step_counter = common_step_counter


def _load_function_from_source(path: Path, function_name: str):
    """Load one function without executing eval.py's simulator startup code."""
    syntax_tree = ast.parse(path.read_text())
    function_node = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function_module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "os": os,
        "re": re,
        "yaml": yaml,
    }
    exec(compile(function_module, str(path), "exec"), namespace)
    return namespace[function_name]


def test_evaluator_loads_operational_bounds_from_constraints_and_rewards(tmp_path):
    params_directory = tmp_path / "params"
    params_directory.mkdir()
    config = {
        "actions": {
            "joint_pos": {
                "joint_names": ["FL_hip_joint", "FL_thigh_joint"],
            },
        },
        "scene": {
            "robot": {
                "init_state": {
                    "joint_pos": {".*": 0.0},
                },
            },
        },
        "constraints": {
            "foot_contact_force": {
                "func": "cat_envs.tasks.utils.cat.constraints.contact_force",
                "params": {"limit": 300.0, "names": [".*_foot"]},
            },
        },
        "rewards": {
            "joint_torque": {
                "func": "cat_envs.tasks.utils.mdp.rewards.joint_torque_limit_penalty",
                "weight": 0.1,
                "params": {
                    "limit": 20.0,
                    "names": [".*_joint"],
                    "curriculum_steps": 19_200,
                },
            },
            "base_orientation": {
                "func": "cat_envs.tasks.utils.mdp.rewards.base_orientation_limit_penalty",
                "weight": 0.1,
                "params": {
                    "limit": 0.1,
                    "curriculum_steps": 19_200,
                },
            },
        },
    }
    (params_directory / "env.yaml").write_text(yaml.safe_dump(config))

    load_constraint_bounds = _load_function_from_source(EVAL_PATH, "load_constraint_bounds")
    bounds = load_constraint_bounds(str(params_directory))

    assert bounds == {
        "foot_contact_force": (0.0, 300.0),
        "joint_torque": (-20.0, 20.0),
        "base_orientation": (-0.1, 0.1),
    }


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


@pytest.fixture(scope="session")
def isaaclab_app():
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    try:
        yield launcher.app
    finally:
        launcher.app.close()


@pytest.fixture(scope="session")
def go2_config_module(isaaclab_app):
    del isaaclab_app
    return importlib.import_module(
        "cat_envs.tasks.locomotion.velocity.config.solo12.cat_go2_rough_terrain_env_cfg"
    )


@pytest.fixture(scope="session")
def go2_env_cfg(go2_config_module):
    return go2_config_module.Go2RoughTerrainEnvCfg()


def _active_term_names(config_group, term_type):
    return {
        name
        for name, value in vars(config_group).items()
        if isinstance(value, term_type)
    }


def test_go2_config_uses_selected_low_reward_profile(go2_config_module, go2_env_cfg):
    assert go2_config_module.SOFT_CONSTRAINT_REWARD_END_WEIGHT_LOW == 0.1
    assert go2_config_module.SOFT_CONSTRAINT_REWARD_END_WEIGHT_HIGH == 10.0
    assert (
        go2_config_module.SOFT_CONSTRAINT_REWARD_END_WEIGHT
        == go2_config_module.SOFT_CONSTRAINT_REWARD_END_WEIGHT_LOW
    )
    assert go2_config_module.SOFT_CONSTRAINT_REWARD_CURRICULUM_STEPS == 19_200

    expected_terms = {
        "joint_torque": (
            go2_config_module.rewards.joint_torque_limit_penalty,
            20.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        ),
        "joint_velocity": (
            go2_config_module.rewards.joint_velocity_limit_penalty,
            25.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        ),
        "joint_acceleration": (
            go2_config_module.rewards.joint_acceleration_limit_penalty,
            800.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        ),
        "action_rate": (
            go2_config_module.rewards.action_rate_limit_penalty,
            80.0,
            [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        ),
        "base_orientation": (
            go2_config_module.rewards.base_orientation_limit_penalty,
            0.1,
            None,
        ),
    }

    for term_name, (expected_func, expected_limit, expected_names) in expected_terms.items():
        term = getattr(go2_env_cfg.rewards, term_name)
        assert term.func is expected_func
        assert term.weight == 0.1
        assert term.params["limit"] == expected_limit
        assert term.params["curriculum_steps"] == 19_200
        if expected_names is None:
            assert "names" not in term.params
        else:
            assert term.params["names"] == expected_names


def test_go2_config_retains_only_hard_cat_constraints(go2_config_module, go2_env_cfg):
    assert _active_term_names(go2_env_cfg.constraints, go2_config_module.ConstraintTerm) == {
        "contact",
        "foot_contact_force",
        "front_hfe_position",
        "upsidedown",
    }
    assert go2_env_cfg.constraints.contact.max_p == 1.0
    assert go2_env_cfg.constraints.foot_contact_force.max_p == 1.0
    assert go2_env_cfg.constraints.front_hfe_position.max_p == 1.0
    assert go2_env_cfg.constraints.upsidedown.max_p == 1.0


def test_go2_config_removes_soft_cat_curricula(go2_config_module, go2_env_cfg):
    assert _active_term_names(go2_env_cfg.curriculum, go2_config_module.CurrTerm) == {
        "power",
        "terrain_levels",
    }


def test_go2_config_preserves_unrelated_lep_settings(go2_env_cfg):
    assert go2_env_cfg.actions.joint_pos.scale == 0.8
    assert go2_env_cfg.rewards.track_lin_vel_xy_exp.weight == 1.0
    assert go2_env_cfg.rewards.track_ang_vel_z_exp.weight == 0.5
    assert go2_env_cfg.rewards.minimize_power.weight == 0.0
    assert go2_env_cfg.curriculum.power.params == {
        "term_name": "minimize_power",
        "num_steps_from_start_step": 300_000,
        "start_at_step": 0,
        "start_weight": 0.0,
        "end_weight": 0.008,
    }
