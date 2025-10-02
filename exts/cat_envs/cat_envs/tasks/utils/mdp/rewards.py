from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_power(env: ManagerBasedRLEnv, scaling_factor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vels = robot.data.joint_vel[:, asset_cfg.joint_ids]
    power = torch.sum(torch.abs(joint_torques * joint_vels), dim=1)
    print(f"env.common_step_counter={env.common_step_counter}\tscaled power={(power.mean().cpu().item() * scaling_factor):.4f}\tscaling_factor={scaling_factor}")
    return -power * scaling_factor

def squared_joint_power(env: ManagerBasedRLEnv, scaling_factor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vels = robot.data.joint_vel[:, asset_cfg.joint_ids]
    power = torch.sum(torch.square(torch.abs(joint_torques * joint_vels)), dim=1)
    print(f"env.common_step_counter={env.common_step_counter}\tscaled squared power={(power.mean().cpu().item() * scaling_factor):.4f}\tscaling_factor={scaling_factor}")
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