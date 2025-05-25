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
    print(f"Scaled Power={(power.mean().cpu().item() * scaling_factor):.4f}")
    return -power * scaling_factor # Negative reward since power should be minimized