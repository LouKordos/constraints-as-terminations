from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

import cat_envs.tasks.utils.cat.constraints as constraints

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _normalized_constraint_penalty(
    env: ManagerBasedRLEnv,
    constraint_violation: torch.Tensor,
    limit: float,
    curriculum_steps: int,
) -> torch.Tensor:
    """Convert signed constraint violations into a scheduled normalized penalty."""
    if limit <= 0.0:
        raise ValueError(f"limit must be positive, got {limit}")
    if curriculum_steps <= 0:
        raise ValueError(f"curriculum_steps must be positive, got {curriculum_steps}")
    if constraint_violation.ndim not in (1, 2):
        raise ValueError(
            "constraint_violation must have shape (num_envs,) or (num_envs, num_components), "
            f"got {tuple(constraint_violation.shape)}"
        )

    normalized_excess = torch.clamp_min(constraint_violation, 0.0) / limit
    if normalized_excess.ndim == 2:
        normalized_excess = normalized_excess.max(dim=1).values

    curriculum_progress = min(max(env.common_step_counter / curriculum_steps, 0.0), 1.0)
    return -normalized_excess * curriculum_progress


def joint_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    violation = constraints.joint_torque(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def joint_velocity_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    violation = constraints.joint_velocity(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def joint_acceleration_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    violation = constraints.joint_acceleration(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def action_rate_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    violation = constraints.action_rate(env, limit=limit, names=names, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def base_orientation_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float,
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    violation = constraints.base_orientation(env, limit=limit, asset_cfg=asset_cfg)
    return _normalized_constraint_penalty(env, violation, limit, curriculum_steps)


def joint_power(env: ManagerBasedRLEnv, scaling_factor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vels = robot.data.joint_vel[:, asset_cfg.joint_ids]
    power = torch.sum(torch.abs(joint_torques * joint_vels), dim=1)
    return -power * scaling_factor

def squared_joint_power(env: ManagerBasedRLEnv, scaling_factor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vels = robot.data.joint_vel[:, asset_cfg.joint_ids]
    power = torch.sum(torch.square(torch.abs(joint_torques * joint_vels)), dim=1)
    return -power * scaling_factor

def squared_joint_torques(env: ManagerBasedRLEnv, scaling_factor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    squared_torque_sum = torch.sum(torch.square(torch.abs(joint_torques)), dim=1)
    print(f"env.common_step_counter={env.common_step_counter}\tscaled squared torque sum={(squared_torque_sum.mean().cpu().item() * scaling_factor):.4f}\tscaling_factor={scaling_factor}")
    return -squared_torque_sum * scaling_factor

# Eq. 4 https://arxiv.org/pdf/2403.20001
def cost_of_transport_exp(env: ManagerBasedRLEnv, scaling_factor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vels = robot.data.joint_vel[:, asset_cfg.joint_ids]
    joint_power = torch.sum(torch.abs(joint_torques * joint_vels), dim=1)

    # planar_speed = torch.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1) # XY components
    # yaw_rate = torch.abs(robot.data.root_link_ang_vel_b[:, 2])

    vel_commands = env.command_manager.get_command("base_velocity").clone()
    planar_speed_commands = torch.norm(vel_commands[:, :2], dim=1)
    yaw_rate_commands = torch.abs(vel_commands[:, 2])

    linear_vel_scale = 1000
    angular_vel_scale = 500
    denominator = linear_vel_scale * planar_speed_commands + angular_vel_scale * yaw_rate_commands
    denominator.clamp(min=1e-6) # To avoid division by zero

    # Eq. 4 https://arxiv.org/pdf/2403.20001
    return torch.exp(-joint_power/denominator)
