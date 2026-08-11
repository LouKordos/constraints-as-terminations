"""Crossed Go2 and ANYmal C tuning configurations.

Each class keeps the receiving embodiment's matched environment and transfers
only donor numerical reward weights and joint-position action scale. Contact
enablement and selectors remain receiver-specific.
"""

from isaaclab.utils import configclass

from .baseline_anymal_c_rough_env_cfg import (
    BaselineAnymalCRoughEnvCfg,
    BaselineAnymalCRoughEnvCfg_PLAY,
)
from .baseline_go2_rough_env_cfg import (
    BaselineGo2RoughEnvCfg,
    BaselineGo2RoughEnvCfg_PLAY,
)


ANYMAL_C_REWARD_WEIGHTS: dict[str, float] = {
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

GO2_REWARD_WEIGHTS: dict[str, float] = {
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


def _set_reward_weights(rewards, weights: dict[str, float]) -> None:
    for term_name, weight in weights.items():
        term = getattr(rewards, term_name)
        if term is None:
            raise ValueError(f"Cannot transfer weight for disabled reward term: {term_name}")
        term.weight = weight


def _apply_anymal_c_tuning_to_go2(cfg) -> None:
    if cfg.rewards.undesired_contacts is not None:
        raise ValueError("Crossed Go2 must retain its disabled undesired_contacts term")
    _set_reward_weights(cfg.rewards, ANYMAL_C_REWARD_WEIGHTS)
    cfg.actions.joint_pos.scale = 0.5


def _apply_go2_tuning_to_anymal_c(cfg) -> None:
    contact_term = cfg.rewards.undesired_contacts
    if contact_term is None or contact_term.weight != -1.0:
        raise ValueError("Crossed ANYmal C must retain undesired_contacts at weight -1.0")
    if contact_term.params["sensor_cfg"].body_names != ".*THIGH":
        raise ValueError("Crossed ANYmal C must retain the .*THIGH undesired-contact selector")
    _set_reward_weights(cfg.rewards, GO2_REWARD_WEIGHTS)
    cfg.actions.joint_pos.scale = 0.25


@configclass
class BaselineGo2AnymalCTuningRoughEnvCfg(BaselineGo2RoughEnvCfg):
    """Matched Go2 baseline with ANYmal C reward weights and action scale."""

    def __post_init__(self):
        super().__post_init__()
        _apply_anymal_c_tuning_to_go2(self)


@configclass
class BaselineGo2AnymalCTuningRoughEnvCfg_PLAY(BaselineGo2RoughEnvCfg_PLAY):
    """Play configuration for Go2 with ANYmal C tuning."""

    def __post_init__(self):
        super().__post_init__()
        _apply_anymal_c_tuning_to_go2(self)


@configclass
class BaselineAnymalCGo2TuningRoughEnvCfg(BaselineAnymalCRoughEnvCfg):
    """Matched ANYmal C baseline with Go2 reward weights and action scale."""

    def __post_init__(self):
        super().__post_init__()
        _apply_go2_tuning_to_anymal_c(self)


@configclass
class BaselineAnymalCGo2TuningRoughEnvCfg_PLAY(BaselineAnymalCRoughEnvCfg_PLAY):
    """Play configuration for ANYmal C with Go2 tuning."""

    def __post_init__(self):
        super().__post_init__()
        _apply_go2_tuning_to_anymal_c(self)
