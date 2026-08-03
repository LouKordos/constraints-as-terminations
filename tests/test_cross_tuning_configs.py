from __future__ import annotations

from copy import deepcopy

import pytest
from isaaclab.app import AppLauncher


_APP_LAUNCHER = AppLauncher(headless=True)

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


def teardown_module():
    _APP_LAUNCHER.app.close()


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
