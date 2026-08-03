from __future__ import annotations

from copy import deepcopy

import pytest
from isaaclab.app import AppLauncher


_APP_LAUNCHER = AppLauncher(headless=True)

import gymnasium as gym
import cat_envs.tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from cat_envs.tasks.locomotion.velocity.config.solo12.baseline_anymal_c_rough_env_cfg import (
    BaselineAnymalCRoughEnvCfg,
    BaselineAnymalCRoughEnvCfg_PLAY,
)
from cat_envs.tasks.locomotion.velocity.config.solo12.baseline_cross_tuning_env_cfg import (
    ANYMAL_C_REWARD_WEIGHTS,
    GO2_REWARD_WEIGHTS,
    BaselineAnymalCGo2TuningRoughEnvCfg,
    BaselineAnymalCGo2TuningRoughEnvCfg_PLAY,
    BaselineGo2AnymalCTuningRoughEnvCfg,
    BaselineGo2AnymalCTuningRoughEnvCfg_PLAY,
    _apply_go2_tuning_to_anymal_c,
)
from cat_envs.tasks.locomotion.velocity.config.solo12.baseline_go2_rough_env_cfg import (
    BaselineGo2RoughEnvCfg,
    BaselineGo2RoughEnvCfg_PLAY,
)


EXPECTED_ANYMAL_C_WEIGHTS = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 0.5,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -1.0e-5,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.125,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}

EXPECTED_GO2_WEIGHTS = {
    "track_lin_vel_xy_exp": 1.5,
    "track_ang_vel_z_exp": 0.75,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -2.0e-4,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.01,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}

CROSS_TASKS = {
    "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0": (
        BaselineGo2AnymalCTuningRoughEnvCfg,
        AnymalCRoughPPORunnerCfg,
        "go2_anymal_c_tuning_rough",
        0.005,
    ),
    "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-Play-v0": (
        BaselineGo2AnymalCTuningRoughEnvCfg_PLAY,
        AnymalCRoughPPORunnerCfg,
        "go2_anymal_c_tuning_rough",
        0.005,
    ),
    "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0": (
        BaselineAnymalCGo2TuningRoughEnvCfg,
        UnitreeGo2RoughPPORunnerCfg,
        "anymal_c_go2_tuning_rough",
        0.01,
    ),
    "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-Play-v0": (
        BaselineAnymalCGo2TuningRoughEnvCfg_PLAY,
        UnitreeGo2RoughPPORunnerCfg,
        "anymal_c_go2_tuning_rough",
        0.01,
    ),
}


def _reward_weights(cfg) -> dict[str, float]:
    return {name: getattr(cfg.rewards, name).weight for name in EXPECTED_ANYMAL_C_WEIGHTS}


def _without_transferred_fields(crossed_cfg, receiver_cfg) -> dict:
    crossed = deepcopy(crossed_cfg.to_dict())
    receiver = receiver_cfg.to_dict()
    crossed["rewards"] = receiver["rewards"]
    crossed["actions"]["joint_pos"]["scale"] = receiver["actions"]["joint_pos"]["scale"]
    return crossed


@pytest.mark.parametrize(
    "crossed_cls,receiver_cls",
    [
        (BaselineGo2AnymalCTuningRoughEnvCfg, BaselineGo2RoughEnvCfg),
        (BaselineGo2AnymalCTuningRoughEnvCfg_PLAY, BaselineGo2RoughEnvCfg_PLAY),
    ],
)
def test_go2_receives_anymal_weights_and_action_scale(crossed_cls, receiver_cls):
    cfg = crossed_cls()
    receiver = receiver_cls()

    assert ANYMAL_C_REWARD_WEIGHTS == EXPECTED_ANYMAL_C_WEIGHTS
    assert _reward_weights(cfg) == EXPECTED_ANYMAL_C_WEIGHTS
    assert cfg.actions.joint_pos.scale == 0.5
    assert cfg.rewards.undesired_contacts is None
    assert cfg.rewards.feet_air_time.params["sensor_cfg"].body_names == ".*_foot"
    assert _without_transferred_fields(cfg, receiver) == receiver.to_dict()


@pytest.mark.parametrize(
    "crossed_cls,receiver_cls",
    [
        (BaselineAnymalCGo2TuningRoughEnvCfg, BaselineAnymalCRoughEnvCfg),
        (BaselineAnymalCGo2TuningRoughEnvCfg_PLAY, BaselineAnymalCRoughEnvCfg_PLAY),
    ],
)
def test_anymal_receives_go2_weights_and_action_scale(crossed_cls, receiver_cls):
    cfg = crossed_cls()
    receiver = receiver_cls()

    assert GO2_REWARD_WEIGHTS == EXPECTED_GO2_WEIGHTS
    assert _reward_weights(cfg) == EXPECTED_GO2_WEIGHTS
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.rewards.undesired_contacts.weight == -1.0
    assert cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names == ".*THIGH"
    assert cfg.rewards.feet_air_time.params["sensor_cfg"].body_names == ".*FOOT"
    assert _without_transferred_fields(cfg, receiver) == receiver.to_dict()


def test_anymal_transfer_rejects_changed_contact_selector():
    cfg = BaselineAnymalCRoughEnvCfg()
    cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*WRONG"

    with pytest.raises(ValueError, match=".*THIGH"):
        _apply_go2_tuning_to_anymal_c(cfg)


@pytest.mark.parametrize("task_id,contract", CROSS_TASKS.items())
def test_cross_task_registration_and_donor_ppo(task_id, contract):
    env_cls, donor_cls, experiment_name, entropy_coef = contract
    spec = gym.spec(task_id)
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    donor_cfg = donor_cls()

    assert isinstance(env_cfg, env_cls)
    assert spec.kwargs["env_cfg_entry_point"].endswith(f":{env_cls.__name__}")
    assert runner_cfg.algorithm.entropy_coef == entropy_coef
    assert runner_cfg.experiment_name == experiment_name

    actual_runner = runner_cfg.to_dict()
    donor_runner = donor_cfg.to_dict()
    donor_runner["experiment_name"] = experiment_name
    assert actual_runner == donor_runner
