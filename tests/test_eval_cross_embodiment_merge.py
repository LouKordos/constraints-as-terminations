from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
EVAL_PATH = SCRIPTS_DIR / "eval.py"


@lru_cache(maxsize=1)
def _load_eval_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("eval_cross_embodiment_merge", EVAL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exception:
        pytest.fail(f"scripts/eval.py must be syntactically resolved: {exception}")
    return module


@pytest.mark.parametrize(
    ("task_name", "expected_profile"),
    [
        ("CaT-Go2-Rough-Terrain-Play-v0", "go2"),
        ("CaT-Anymal-C-Rough-Terrain-Play-v0", "anymal_c"),
        ("CaT-Spot-Rough-Terrain-Play-v0", "spot"),
        ("legacy-task-without-embodiment-metadata", "go2"),
    ],
)
def test_eval_profile_resolution_preserves_legacy_go2_and_supports_new_robots(
    task_name: str,
    expected_profile: str,
) -> None:
    eval_module = _load_eval_module()

    assert eval_module.resolve_robot_eval_profile(task_name).name == expected_profile


@pytest.mark.parametrize(
    ("task_name", "expected_profile"),
    [
        ("Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0", "go2"),
        ("Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-Play-v0", "go2"),
        ("Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0", "anymal_c"),
        ("Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-Play-v0", "anymal_c"),
    ],
)
def test_crossed_baseline_profile_uses_the_receiver_robot(
    task_name: str,
    expected_profile: str,
) -> None:
    eval_module = _load_eval_module()

    assert eval_module.is_matched_baseline_task(task_name)
    assert eval_module.resolve_robot_eval_profile(task_name).name == expected_profile


@pytest.mark.parametrize(
    ("profile_name", "expected_bounds"),
    [
        (
            "anymal_c",
            {
                "joint_torque": (-80.0, 80.0),
                "joint_velocity": (-12.0, 12.0),
                "joint_acceleration": (-800.0, 800.0),
                "action_rate": (-128.0, 128.0),
                "foot_contact_force": (0.0, 1000.0),
            },
        ),
        (
            "spot",
            {
                "joint_torque": (-80.0, 80.0),
                "joint_velocity": (-20.0, 20.0),
                "joint_acceleration": (-800.0, 800.0),
                "action_rate": (-320.0, 320.0),
                "foot_contact_force": (0.0, 1000.0),
            },
        ),
    ],
)
def test_matched_baseline_reporting_bounds_match_tuned_cat_limits(
    profile_name: str,
    expected_bounds: dict[str, tuple[float, float]],
) -> None:
    eval_module = _load_eval_module()

    assert eval_module.get_matched_baseline_constraint_bounds(profile_name) == expected_bounds


def test_root_body_index_uses_the_profile_root_link() -> None:
    eval_module = _load_eval_module()

    assert eval_module.resolve_root_body_index(
        ["fl_uleg", "body", "fl_foot"],
        eval_module.ROBOT_EVAL_PROFILES["spot"],
    ) == 1


def test_root_body_index_rejects_an_asset_without_the_profile_root() -> None:
    eval_module = _load_eval_module()

    with pytest.raises(ValueError, match="root body 'body'"):
        eval_module.resolve_root_body_index(
            ["fl_uleg", "fl_foot"],
            eval_module.ROBOT_EVAL_PROFILES["spot"],
        )


def test_go2_scenario_height_adjustment_is_an_exact_identity() -> None:
    eval_module = _load_eval_module()
    spawn_position = torch.tensor([30.0, 30.0, 0.4])
    scenarios = [
        (
            "walk",
            torch.tensor([1.0, 0.0, 0.0]),
            (spawn_position, torch.tensor([0.0, 0.0, 0.0, 1.0])),
        )
    ]

    adjusted = eval_module.adjust_fixed_command_spawn_heights(
        scenarios,
        eval_module.ROBOT_EVAL_PROFILES["go2"],
    )

    assert adjusted is scenarios
    assert adjusted[0][2][0] is spawn_position


@pytest.mark.parametrize(("profile_name", "expected_height"), [("anymal_c", 0.6), ("spot", 0.5)])
def test_non_go2_scenario_height_adjustment_changes_only_spawn_height(
    profile_name: str,
    expected_height: float,
) -> None:
    eval_module = _load_eval_module()
    command = torch.tensor([1.0, 0.0, 0.0])
    orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])
    scenarios = [("walk", command, (torch.tensor([30.0, 30.0, 0.4]), orientation))]

    adjusted = eval_module.adjust_fixed_command_spawn_heights(
        scenarios,
        eval_module.ROBOT_EVAL_PROFILES[profile_name],
    )

    assert adjusted[0][0] == "walk"
    assert adjusted[0][1] is command
    assert adjusted[0][2][1] is orientation
    assert adjusted[0][2][0][:2].tolist() == [30.0, 30.0]
    assert adjusted[0][2][0][2].item() == pytest.approx(expected_height)


def test_cat_go2_reward_weights_remain_unchanged() -> None:
    eval_module = _load_eval_module()
    rewards = SimpleNamespace(
        track_lin_vel_xy_exp=SimpleNamespace(weight=1.5),
        track_ang_vel_z_exp=SimpleNamespace(weight=0.75),
    )
    env_cfg = SimpleNamespace(rewards=rewards)

    changed = eval_module.apply_common_eval_reward_scale_if_needed(
        env_cfg,
        task_name="CaT-Go2-Rough-Terrain-Play-v0",
        enabled=True,
    )

    assert changed is False
    assert rewards.track_lin_vel_xy_exp.weight == 1.5
    assert rewards.track_ang_vel_z_exp.weight == 0.75


def test_upstream_go2_reward_override_keeps_main_behavior() -> None:
    eval_module = _load_eval_module()
    rewards = SimpleNamespace(
        track_lin_vel_xy_exp=SimpleNamespace(weight=1.5),
        track_ang_vel_z_exp=SimpleNamespace(weight=0.75),
    )
    env_cfg = SimpleNamespace(rewards=rewards)

    changed = eval_module.apply_common_eval_reward_scale_if_needed(
        env_cfg,
        task_name="Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
        enabled=True,
    )

    assert changed is True
    assert rewards.track_lin_vel_xy_exp.weight == 1.0
    assert rewards.track_ang_vel_z_exp.weight == 0.5
