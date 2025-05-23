# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# IMPORTANT NOTE: This is an absolute upper bound on joint position, not relative to default position and no lower bound, because negative constraint violations are ignored / clipped to 0
def joint_position_absolute_upper_bound(env: ManagerBasedRLEnv, limit: float, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    pos0 = robot.data.root_pos_w[0]
    x, y, z = pos0[0].item(), pos0[1].item(), pos0[2].item()
    # print(f"robot base pose of env 0: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr = torch.abs(data.joint_pos[:, joint_ids]) - limit
    return cstr

# Joint position contraint relative to default position, use this in most cases.
def relative_joint_position_upper_and_lower_bound_when_moving_forward(env: ManagerBasedRLEnv, limit: float, names: list[str], velocity_deadzone: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr = (torch.abs(data.joint_pos[:, joint_ids] - data.default_joint_pos[:, joint_ids]) - limit)
    cstr *= ((torch.abs(env.command_manager.get_command("base_velocity")[:, 1]) < velocity_deadzone).float().unsqueeze(1))
    return cstr

def joint_torque(env: ManagerBasedRLEnv, limit: float, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr = torch.abs(data.applied_torque[:, joint_ids]) - limit
    return cstr

def joint_velocity(env: ManagerBasedRLEnv, limit: float, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return torch.abs(data.joint_vel[:, joint_ids]) - limit

def joint_acceleration(env: ManagerBasedRLEnv, limit: float, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return torch.abs(data.joint_acc[:, joint_ids]) - limit

def upsidedown(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return data.projected_gravity_b[:, 2] > limit

def contact(env: ManagerBasedRLEnv, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    undesired_contact_body_ids, _ = contact_sensor.find_bodies(names, preserve_order=True)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(torch.max(torch.norm(net_contact_forces[:, :, undesired_contact_body_ids], dim=-1), dim=1,)[0] > 1.0, dim=1,)

def base_orientation(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit

# Important Note: Only punishes policy if foot actually touches the ground at some point due to compute_first_contact()
# This filters any feet that are in the air and might have very high air time if they never touch down. 
# Some experiments showed that the policy thus learns to keep the feet in the air constantly to avoid constration violation
# Also, even if that were not the case, the code still only sets a lower bound on air time, no upper limit.
def air_time_lower_bound(env: ManagerBasedRLEnv, limit: float, names: list[str], velocity_deadzone: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    feet_ids, _ = contact_sensor.find_bodies(names, preserve_order=True)
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, feet_ids]
    last_air_time = contact_sensor.data.last_air_time[:, feet_ids]

    command_more_than_limit = ((torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1) > velocity_deadzone).float().unsqueeze(1))
    cstr = (limit - last_air_time) * touchdown.float() * command_more_than_limit
    return cstr

def air_time_upper_bound(env: ManagerBasedRLEnv, limit: float, names: list[str], velocity_deadzone: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    feet_ids, _ = contact_sensor.find_bodies(names, preserve_order=True)
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, feet_ids]
    current_air_time = contact_sensor.data.current_air_time[:, feet_ids]
    command_more_than_limit = ((torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1) > velocity_deadzone).float().unsqueeze(1))
    cstr = (current_air_time - limit) * (1 - touchdown.float()) * command_more_than_limit
    return cstr

def n_foot_contact(env: ManagerBasedRLEnv, names: list[str], number_of_desired_feet: int, min_command_value: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    undesired_contact_body_ids, _ = contact_sensor.find_bodies(names, preserve_order=True)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_cstr = torch.abs((torch.max(torch.norm(net_contact_forces[:, :, undesired_contact_body_ids], dim=-1), dim=1,)[0] > 1.0).sum(1) - number_of_desired_feet)
    command_more_than_limit = (torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1) > min_command_value).float()
    return contact_cstr * command_more_than_limit

def joint_range(env: ManagerBasedRLEnv, limit: float, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return (torch.abs(data.joint_pos[:, joint_ids] - data.default_joint_pos[:, joint_ids]) - limit)

def action_rate(env: ManagerBasedRLEnv, limit: float, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return (torch.abs(env.action_manager._action[:, joint_ids] - env.action_manager._prev_action[:, joint_ids]) / env.step_dt - limit)

def foot_contact_force(env: ManagerBasedRLEnv, limit: float, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    feet_ids, _ = contact_sensor.find_bodies(names, preserve_order=True)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return (torch.max(torch.norm(net_contact_forces[:, :, feet_ids], dim=-1), dim=1)[0] - limit)

# This is NOT world frame height, but instead adjusted for terrain height below the robot.
def min_base_height_relative_to_ground(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    root_world_z = robot.data.root_link_pos_w[:, 2]
    terrain_z = env.scene["ray_caster_height_constraints"].data.ray_hits_w[:, 0, 2] # index 0 because single ray
    height_relative_to_ground = root_world_z - terrain_z
    
    # print(f"terrain_z={terrain_z[0].item()}\troot_world_z={root_world_z[0].item()}\theight_relative_to_ground={height_relative_to_ground[0].item()}")
    nonfinite_mask = ~torch.isfinite(terrain_z)
    if nonfinite_mask.any(): # Occurs on some environment resets, if this happens often there might be an issue
        print(f"ray caster for height constraint is nonfinite for {torch.count_nonzero(nonfinite_mask)} envs, reporting 0 constraint violation for these envs.")
        height_relative_to_ground.masked_fill(nonfinite_mask, limit)

    return limit - height_relative_to_ground

# This is NOT world frame height, but instead adjusted for terrain height below the robot.
def max_base_height_relative_to_ground(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    root_world_z = robot.data.root_link_pos_w[:, 2]
    terrain_z = env.scene["ray_caster_height_constraints"].data.ray_hits_w[:, 0, 2] # index 0 because single ray
    height_relative_to_ground = root_world_z - terrain_z

    # print(f"terrain_z={terrain_z[0].item()}\troot_world_z={root_world_z[0].item()}\theight_relative_to_ground={height_relative_to_ground[0].item()}")
    nonfinite_mask = ~torch.isfinite(terrain_z)
    if nonfinite_mask.any(): # Occurs on some environment resets, if this happens often there might be an issue
        print(f"ray caster for height constraint is nonfinite for {torch.count_nonzero(nonfinite_mask)} envs, reporting 0 constraint violation for these envs.")
        height_relative_to_ground.masked_fill(nonfinite_mask, limit)
        
    return height_relative_to_ground - limit

def no_move(env: ManagerBasedRLEnv, names: list[str], velocity_deadzone: float, joint_vel_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr_nomove = (torch.abs(data.joint_vel[:, joint_ids]) - joint_vel_limit) * (torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1) < velocity_deadzone).float().unsqueeze(1)
    return cstr_nomove
