# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
import isaaclab.envs.mdp as mdp
from typing import TYPE_CHECKING, List, Any, Dict
from importlib import import_module

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def joint_pos(env: ManagerBasedEnv, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset.find_joints(names, preserve_order=True)[0]]


def joint_vel(env: ManagerBasedEnv, names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset.find_joints(names, preserve_order=True)[0]]

def joint_state_history(env, names: List[str], history_len: int = 3, mode: str = "pos", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Stack the *history_len* most recent joint positions/velocities.
    Returns shape [E, history_len * num_joints] (flattened, newest first).
    """
    assert mode in ("pos", "vel"), f"Unsupported mode '{mode}'"
    key = f"_joint_hist_{mode}_{history_len}_{'_'.join(names)}"
    joint_index_key = key + "_index"
    asset: Articulation = env.scene[asset_cfg.name]

    # Initial setup
    if not hasattr(env, key):
        buf = torch.zeros(
            (env.num_envs, history_len, len(names)),
            dtype=torch.float32,
            device=env.device,
        )
        setattr(env, key, buf)

    buf: torch.Tensor = getattr(env, key)

    # Required for initial call of ObservationManager to determine shape
    ep_buf = getattr(env, "episode_length_buf", None)
    if ep_buf is None:
        return buf.reshape(env.num_envs, -1)

    if mode == "pos":
        cur = asset.data.joint_pos[:, asset.find_joints(names, preserve_order=True)[0]]
    else:  # mode == "vel"
        cur = asset.data.joint_vel[:, asset.find_joints(names, preserve_order=True)[0]]

    # Roll buffer, write newest at index 0
    buf = torch.roll(buf, shifts=1, dims=1)
    buf[:, 0, :] = cur

    # Re-initialise reset envs so all history slots equal the current state
    just_reset = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if just_reset.numel() > 0:
        buf[just_reset] = cur[just_reset].unsqueeze(1).repeat(1, history_len, 1)

    setattr(env, key, buf) # Save back (PyTorch tensors are views, but be explicit)
    return buf.reshape(env.num_envs, -1) # Flatten (newest first) as ObservationManager expects [E, ·]

def base_ang_vel_history(env, history_len: int = 3):
    """
    Stack the last <history_len> base angular-velocity readings in body frame.
    Shape → [E, history_len * 3]  (flattened, newest first).
    """
    key = f"_base_ang_vel_hist_{history_len}"
    if not hasattr(env, key):
        setattr(
            env,
            key,
            torch.zeros((env.num_envs, history_len, 3),
                        dtype=torch.float32, device=env.device),
        )
    buf: torch.Tensor = getattr(env, key)

    ep_buf = getattr(env, "episode_length_buf", None)
    if ep_buf is None: # ObservationManager uses this to check the shape
        return buf.reshape(env.num_envs, -1)

    cur = mdp.base_ang_vel(env)
    buf = torch.roll(buf, 1, 1)
    buf[:, 0] = cur

    just_reset = (ep_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if just_reset.numel():
        buf[just_reset] = cur[just_reset].unsqueeze(1).repeat(1, history_len, 1)

    setattr(env, key, buf)
    return buf.reshape(env.num_envs, -1)

def projected_gravity_history(env, history_len: int = 3):
    """
    Stack the last <history_len> projected-gravity vectors.
    Shape → [E, history_len * 3].
    """
    key = f"_proj_grav_hist_{history_len}"
    if not hasattr(env, key):
        setattr(
            env,
            key,
            torch.zeros((env.num_envs, history_len, 3),
                        dtype=torch.float32, device=env.device),
        )
    buf: torch.Tensor = getattr(env, key)

    if (ep_buf := getattr(env, "episode_length_buf", None)) is None:
        return buf.reshape(env.num_envs, -1)

    cur = mdp.projected_gravity(env) # (E, 3) 
    buf = torch.roll(buf, 1, 1)
    buf[:, 0] = cur

    just_reset = (ep_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if just_reset.numel():
        buf[just_reset] = cur[just_reset].unsqueeze(1).repeat(1, history_len, 1)

    setattr(env, key, buf)
    return buf.reshape(env.num_envs, -1)

def actions_history(env, history_len: int = 3):
    """
    Stack the last <history_len> raw policy actions.
    Shape → [E, history_len * action_dim].
    """
    key = f"_action_hist_{history_len}"
    if not hasattr(env, key):
        # action_dim available from action manager
        action_dim = env.action_manager.total_action_dim 
        setattr(
            env,
            key,
            torch.zeros((env.num_envs, history_len, action_dim),
                        dtype=torch.float32, device=env.device),
        )
    buf: torch.Tensor = getattr(env, key)

    if (ep_buf := getattr(env, "episode_length_buf", None)) is None:
        return buf.reshape(env.num_envs, -1)

    cur = mdp.last_action(env) # (E, action_dim) 
    buf = torch.roll(buf, 1, 1)
    buf[:, 0] = cur

    just_reset = (ep_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if just_reset.numel():
        buf[just_reset] = cur[just_reset].unsqueeze(1).repeat(1, history_len, 1)

    setattr(env, key, buf)
    return buf.reshape(env.num_envs, -1)

def height_map_history(env, history_len: int = 3, asset_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster")):
    """
    Stack the last <history_len> height-map grids returned by `height_map_grid`.
    Shape → [E, history_len * R] where R = # of rays in the grid.
    """
    
    # This depressing hack is required because importing it at the top of the file leads ot a circular
    # dependency. The proper way would of course have been to define this function in the observations.py
    global height_map_grid
    if "height_map_grid" not in globals():
        # replace the module path with your env-cfg’s real location
        mod = import_module("cat_envs.tasks.locomotion.velocity.config.solo12.cat_go2_rough_terrain_env_cfg")
        height_map_grid = getattr(mod, "height_map_grid")

    key = f"_height_map_hist_{history_len}_{asset_cfg.name}"
    if not hasattr(env, key):
        # Need the grid size once to allocate buffer —
        cur0 = height_map_grid(env, asset_cfg=asset_cfg) # (E, R)
        R = cur0.shape[1]
        buf0 = torch.zeros((env.num_envs, history_len, R), dtype=torch.float32, device=env.device)
        setattr(env, key, buf0)
        # Store first reading to avoid all-zero history at t=0
        buf0[:, 0] = cur0

    buf: torch.Tensor = getattr(env, key)

    if (ep_buf := getattr(env, "episode_length_buf", None)) is None:
        return buf.reshape(env.num_envs, -1)

    cur = height_map_grid(env, asset_cfg=asset_cfg) # (E, R)
    buf = torch.roll(buf, 1, 1)
    buf[:, 0] = cur

    just_reset = (ep_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if just_reset.numel():
        buf[just_reset] = cur[just_reset].unsqueeze(1).repeat(1, history_len, 1)

    setattr(env, key, buf)
    return buf.reshape(env.num_envs, -1)