# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp.curriculums as isaac_curriculums
from cat_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg, FlatPatchSamplingCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sensors.frame_transformer import FrameTransformerCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import cat_envs.tasks.utils.cat.constraints as constraints
import cat_envs.tasks.utils.cat.curriculums as curriculums
import cat_envs.tasks.utils.mdp.observations as observations

import cat_envs.tasks.utils.mdp.terminations as terminations
import cat_envs.tasks.utils.mdp.events as events
import cat_envs.tasks.utils.mdp.commands as commands
import cat_envs.tasks.utils.mdp.rewards as rewards
from functools import partial
print = partial(print, flush=True) # For cluster runs

from cat_envs.assets.odri import SOLO12_MINIMAL_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from cat_envs.assets.go2_config import UNITREE_GO2_CFG_TRAIN, UNITREE_GO2_CFG_EVAL  # isort: skip
import torch
import numpy as np
# Horrible practice to hard-code this in the env but I spent a week on trying to pass the values via hydra config or changing via train.py but it never worked.
# Right now the seed is configured here and then passed to train.py to set all the libraries
HARDCODED_SEED = 46
import random
random.seed(HARDCODED_SEED)
np.random.seed(HARDCODED_SEED)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.manual_seed(HARDCODED_SEED)
torch.cuda.manual_seed_all(HARDCODED_SEED)

def height_map_grid(env, asset_cfg: SceneEntityCfg):
    ray_hit_positions_world_frame = env.scene[asset_cfg.name].data.ray_hits_w # [E, R, 3]
    base_pose_world_frame = env.scene["robot"].data.root_pos_w # [E, 3]

    # 2) expand base_w so it lines up with hits_w
    base_expanded_to_match_shape_world_frame = base_pose_world_frame.view(-1, 1, 3).expand_as(ray_hit_positions_world_frame) # [E, R, 3]

    # 3) sanitize: any non‐finite entry → copy from base_expanded...
    #	this makes hit == base for that ray, so height=0
    non_finite_mask = ~torch.isfinite(ray_hit_positions_world_frame)
    if non_finite_mask.any():
        # clone once so we don't overwrite the original tensor in the scene
        print("NANS OR INF DURING HEIGHT MAP CALCULATION!!!")
        hits_clean = ray_hit_positions_world_frame.clone()
        hits_clean[non_finite_mask] = base_expanded_to_match_shape_world_frame[non_finite_mask]
    else:
        hits_clean = ray_hit_positions_world_frame

    # 4) compute local coordinates, then the height = z_hit - z_base
    local = hits_clean - base_expanded_to_match_shape_world_frame # [E, R, 3]
    height = local[..., 2] # [E, R]
    # height = torch.ones_like(height) * -0.33
    # print(height.mean().item())
    #test_offset = torch.ones_like(height) * 0.03
    #height += test_offset
    # height = torch.zeros_like(height)

    return height

from copy import deepcopy
seeded_rough_cfg = deepcopy(ROUGH_TERRAINS_CFG)
seeded_rough_cfg.seed = HARDCODED_SEED
seeded_rough_cfg.use_cache = True
patched_sub_terrains = {
    name: sub_cfg.replace(
        flat_patch_sampling={
            "init_pos": FlatPatchSamplingCfg(
                num_patches=4000,
                patch_radius=0.6,
                max_height_diff=0.15,
            )
        }
    )
    for name, sub_cfg in seeded_rough_cfg.sub_terrains.items()
}
seeded_rough_cfg.sub_terrains = patched_sub_terrains

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=seeded_rough_cfg,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG_TRAIN.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.0,  # will override in __post_init__
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.5)), # 0.5 m above base
        mesh_prim_paths=["/World/ground"], # Rays will only collide with meshes specified here as they need to be copied over to the GPU for calculations
        ray_alignment="yaw", # keep sensor level (no pitch/roll with body). This is a gross oversimplification but the original paper also used a grid of heights around the robot
        pattern_cfg=patterns.GridPatternCfg( # Grid pattern shoots down vertical rays to retrieve hight at each grid point. Needs adjustments to be more realistic, such as using e.g. LIDARConfig
            size=[1, 0.8], # see Fig. 3 in paper for grid layout, I tried approximating it visually here
            resolution=0.08,
            ordering="xy" # default row-major
        ),
        ray_cast_drift_range={"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
        debug_vis=True,
    )

    # DO NOT REMOVE, DO NOT USE FOR ANYTHING!
    # This is used in constraints to get relative body height to allow constraining it
    ray_caster_height_constraints = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.0,  # will override in __post_init__
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)),  # 1 m above base
        mesh_prim_paths=["/World/ground"], # Rays will only collide with meshes specified here as they need to be copied over to the GPU for calculations
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
        debug_vis=True,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        velocity_deadzone=0.1,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-0.78, 0.78)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ],
        scale=0.8,
        use_default_offset=True,
        preserve_order=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.001, n_max=0.001), scale=0.25
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.1
        )
        joint_pos = ObsTerm(
            func=observations.joint_pos,
            params={
                "names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                    "FL_thigh_joint",
                    "FR_thigh_joint",
                    "RL_thigh_joint",
                    "RR_thigh_joint",
                    "FL_calf_joint",
                    "FR_calf_joint",
                    "RL_calf_joint",
                    "RR_calf_joint",
                ]
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=observations.joint_vel,
            params={
                "names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                    "FL_thigh_joint",
                    "FR_thigh_joint",
                    "RL_thigh_joint",
                    "RR_thigh_joint",
                    "FL_calf_joint",
                    "FR_calf_joint",
                    "RL_calf_joint",
                    "RR_calf_joint",
                ]
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        
        height_map = ObsTerm(
            func=height_map_grid,
            params={"asset_cfg": SceneEntityCfg("ray_caster")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ObservationsCfgJointStateHistory(ObservationsCfg): # Inherit but redefine all terms to ensure correct order
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.001, n_max=0.001), scale=0.25
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.1
        )
        joint_pos_history = ObsTerm(
            func=observations.joint_state_history,
            params={
                "names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                    "FL_thigh_joint",
                    "FR_thigh_joint",
                    "RL_thigh_joint",
                    "RR_thigh_joint",
                    "FL_calf_joint",
                    "FR_calf_joint",
                    "RL_calf_joint",
                    "RR_calf_joint",
                ],
                "history_len": 3,
                "mode": "pos"
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel_history = ObsTerm(
            func=observations.joint_state_history,
            params={
                "names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                    "FL_thigh_joint",
                    "FR_thigh_joint",
                    "RL_thigh_joint",
                    "RR_thigh_joint",
                    "FL_calf_joint",
                    "FR_calf_joint",
                    "RL_calf_joint",
                    "RR_calf_joint",
                ],
                "history_len": 3,
                "mode": "vel"
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        
        height_map = ObsTerm(
            func=height_map_grid,
            params={"asset_cfg": SceneEntityCfg("ray_caster")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ObservationsCfgFullStateHistory(ObservationsCfg): # Inherit but redefine all terms to ensure correct order
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_ang_vel_hist = ObsTerm(
            func=observations.base_ang_vel_history,
            params={"history_len": 3},
            noise=Unoise(n_min=-0.001, n_max=0.001),
            scale=0.25,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        projected_gravity_hist = ObsTerm(
            func=observations.projected_gravity_history,
            params={"history_len": 3},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            scale=0.1,
        ) 
        joint_pos_history = ObsTerm(
            func=observations.joint_state_history,
            params={
                "names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                    "FL_thigh_joint",
                    "FR_thigh_joint",
                    "RL_thigh_joint",
                    "RR_thigh_joint",
                    "FL_calf_joint",
                    "FR_calf_joint",
                    "RL_calf_joint",
                    "RR_calf_joint",
                ],
                "history_len": 3,
                "mode": "pos"
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel_history = ObsTerm(
            func=observations.joint_state_history,
            params={
                "names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                    "FL_thigh_joint",
                    "FR_thigh_joint",
                    "RL_thigh_joint",
                    "RR_thigh_joint",
                    "FL_calf_joint",
                    "FR_calf_joint",
                    "RL_calf_joint",
                    "RR_calf_joint",
                ],
                "history_len": 3,
                "mode": "vel"
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.05,
        )
        actions_hist = ObsTerm(
            func=observations.actions_history,
            params={"history_len": 3},
            scale=1.0,
        ) 
        height_map_hist = ObsTerm(
            func=observations.height_map_history,
            params={"history_len": 3, "asset_cfg": SceneEntityCfg("ray_caster")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )    
    
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

def force_hard_terrain(env, env_ids: torch.Tensor | None):
    ti = env.scene.terrain  # runtime TerrainImporter
    # Grab sub-terrain origins: [rows, cols, 3]
    origins = ti.terrain_origins
    num_rows = origins.shape[0]
    # Compute the target row: either max_init_level or last row
    # If you set max_init_terrain_level=5 originally, row index = min(5, num_rows-1).
    row_idx = min(env.cfg.scene.terrain.max_init_terrain_level or (num_rows - 1), num_rows - 1)
    # Select just that row → shape (1, cols, 3), then flatten to (cols, 3)
    hard_origins = origins[row_idx:row_idx+1].reshape(-1, 3)
    # Reconfigure env origins so every env starts on the hardest level
    ti.configure_env_origins(hard_origins)

def force_easy_terrain(env, env_ids: torch.Tensor | None):
    ti = env.scene.terrain  # runtime TerrainImporter
    # Grab sub-terrain origins: [rows, cols, 3]
    origins = ti.terrain_origins
    row_idx = 0
    # Select just that row → shape (1, cols, 3), then flatten to (cols, 3)
    hard_origins = origins[row_idx:row_idx+1].reshape(-1, 3)
    # Reconfigure env origins so every env starts on the hardest level
    ti.configure_env_origins(hard_origins)


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 100,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-1.57, 1.57),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (-0.05, 0.05),
    #             "y": (-0.05, 0.05),
    #             "yaw": (-1.57, 1.57),
    #         },
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #             "roll": (-0.0, 0.0),
    #             "pitch": (-0.0, 0.0),
    #             "yaw": (-0.0, 0.0),
    #         },
    #     },
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (-0.05, 0.05),
        },
    )

    # set pushing every step, as only some of the environments are chosen as in the isaacgym cat version
    push_robot = EventTerm(
        # Standard push_by_setting_velocity also works, but interestingly results
        # in a different gait
        func=events.push_by_setting_velocity_with_random_envs,
        mode="interval",
        is_global_time=True,
        interval_range_s=(0.0, 0.005),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # DO NOT RENAME; USE THIS TERM FOR ALL ENERGY MINIMIZATION RELATED TASKS
    # BECAUSE IT IS AUTOMATICALLY DISABLED IN THE PLAY ENV
    minimize_power = RewTerm(
        func=rewards.joint_power,
        weight=0.0, # Updated by curriculum
        params={"scaling_factor": 1.0 } # Set to 1.0 in ppo.py anyway
    )

# Never forget to also add a curriculum term for each added constraint
# IMPORTANT NOTE: The max_p defined here is ALWAYS overwritten by the curriculum whenever
# it is enabled, so the specified value is meaningless and init_max_p defined in the curriculum terms
# is used as the target max_p after curriculum ramp-up. For the terms without curriculum, such as hard
# constraints, the values defined here are used.
@configclass
class ConstraintsCfg:
    # Safety Soft constraints
    joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25, # Overwritten by curriculum!
        params={"limit": 20.0, "names": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]},
    )
    joint_velocity = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25, # Overwritten by curriculum!
        params={"limit": 25.0, "names": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]},
    )
    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25, # Overwritten by curriculum!
        params={"limit": 800.0, "names": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]},
    )
    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25, # Overwritten by curriculum!
        params={"limit": 80.0, "names": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]},
    )

    # Safety Hard constraints
    # Knee and base
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={"names": ["base", ".*_thigh"]},
    )
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=1.0,
        params={"limit": 300.0, "names": [".*_foot"]},
    )
    front_hfe_position = ConstraintTerm(
        func=constraints.joint_position_absolute_upper_bound,
        max_p=1.0,
        params={"limit": 1.5, "names": ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"]},
    )
    upsidedown = ConstraintTerm(
        func=constraints.upsidedown, max_p=1.0, params={"limit": 0.0}
    )

    # Style constraints
    hip_position = ConstraintTerm(
        func=constraints.relative_joint_position_upper_and_lower_bound_when_moving_forward,
        max_p=0.25, # Overwritten by curriculum!
        params={"limit": 0.3, "names": [".*_hip_joint"], "velocity_deadzone": 0.1},
    )
    base_orientation = ConstraintTerm(
        func=constraints.base_orientation, max_p=0.25, params={"limit": 0.1}
    )
    # Never forget to also add a curriculum term for each added constraint
    # min_relative_base_height = ConstraintTerm(func=constraints.min_base_height_relative_to_ground, max_p=0.25, params={"limit": 0.2})
    '''
    air_time_lower_bound = ConstraintTerm(
        func=constraints.air_time_lower_bound,
        max_p=0.25, # Overwritten by curriculum!
        params={"limit": 0.1, "names": [".*_foot"], "velocity_deadzone": 0.1},
    )
    air_time_upper_bound = ConstraintTerm(
        func=constraints.air_time_upper_bound,
        max_p=0.6, # Overwritten by curriculum!
        params={"limit": 0.8, "names": [".*_foot"], "velocity_deadzone": 0.1},
    )
    '''
    no_move = ConstraintTerm(
        func=constraints.no_move,
        max_p=0.1, # Overwritten by curriculum!
        params={
            "names": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            "velocity_deadzone": 0.1,
            "joint_vel_limit": 4.0,
        },
    )
    '''
    two_foot_contact = ConstraintTerm(
        func=constraints.n_foot_contact,
        max_p=0.25, # Overwritten by curriculum!
        params={
            "names": [".*_foot"],
            "number_of_desired_feet": 2,
            "min_command_value": 0.5,
        },
    )
    '''


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base", ".*_thigh"]
            ),
            "threshold": 1.0,
        },
    )
    upside_down = DoneTerm(
        func=terminations.upside_down,
        params={
            "limit": 1,
        },
    )


def terrain_levels_with_ray_caster_refresh(env, env_ids):
    levels = mdp.terrain_levels_vel(env, env_ids)
    env.scene["ray_caster"]._initialize_warp_meshes()
    env.scene["ray_caster_height_constraints"]._initialize_warp_meshes()
    
    return levels

MAX_CURRICULUM_ITERATIONS = 5000

@configclass
class CurriculumCfg:
    # Safety Soft constraints
    joint_torque = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_velocity = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_acceleration = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_acceleration",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    action_rate = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "action_rate",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )

    # Style constraints
    hip_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "hip_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    base_orientation = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "base_orientation",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    # min_relative_base_height = CurrTerm(func=curriculums.modify_constraint_p, params={"term_name": "min_relative_base_height", "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, "init_max_p": 0.25})
    '''
    air_time_lower_bound = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "air_time_lower_bound",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    air_time_upper_bound = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "air_time_upper_bound",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.6,
        },
    )
    two_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "two_foot_contact",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    '''

    power = CurrTerm(
        func=curriculums.update_reward_weight_linear,
        params={
            "term_name": "minimize_power",
            "num_steps_from_start_step": 300000,
            "start_at_step": 0,
            "start_weight": 0.0,
            "end_weight": 0.4 * 0.02 # Instead of setting scaling_factor=0.02 because ppo.py overrides it

        }
    )

    terrain_levels = CurrTerm(func=terrain_levels_with_ray_caster_refresh)
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class Go2RoughTerrainEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=7500, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0

        # simulation settings
        self.sim.solver_type = 0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.max_position_iteration_count = 4
        self.sim.max_velocity_iteration_count = 1
        self.sim.bounce_threshold_velocity = 0.2
        self.sim.gpu_max_rigid_contact_count = 33554432
        # self.sim.device = "cpu"
        # self.sim.physx.use_gpu = False
        # self.sim.solver_type = 1
        # self.sim.num_threads = 1
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.ray_caster.update_period = self.sim.dt
        self.scene.ray_caster_height_constraints.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            mdp.terrain_levels_vel.seed = HARDCODED_SEED
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        print("Setting seed to", HARDCODED_SEED)
        self.sim.random_seed = HARDCODED_SEED
        self.seed = HARDCODED_SEED
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.seed = HARDCODED_SEED
            print(f"Terrain generator seed in env post init={self.scene.terrain.terrain_generator.seed}")

        self.sim.physx.gpu_max_rigid_patch_count = 568462

@configclass
class Go2RoughTerrainEnvCfgJointStateHistory(Go2RoughTerrainEnvCfg):
    observations: ObservationsCfgJointStateHistory = ObservationsCfgJointStateHistory()

@configclass
class Go2RoughTerrainEnvCfgFullStateHistory(Go2RoughTerrainEnvCfg):
    observations: ObservationsCfgFullStateHistory = ObservationsCfgFullStateHistory()

class Go2RoughTerrainEnvCfg_PLAY(Go2RoughTerrainEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 8
    
        # Original torque limit as specified in isaac lab example config
        self.scene.robot = UNITREE_GO2_CFG_EVAL.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        
        # Pick the hardest terrain when testing the model.
        # Technically, this is incorrect as it wlil only run after 
        # the first reset but it's good to see a baseline of it walking on flat terrain first
        self.events.force_hard_terrain = EventTerm(
            func=force_hard_terrain,
            mode="startup",  # runs once at environment startup
        )

        # Used to get foot height above terrain ("sole frame")
        self.scene.ray_caster_FL_foot = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
            update_period=self.sim.dt,
            offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)), # Starting point 1m above base, doesn't really matter
            mesh_prim_paths=["/World/ground"], # Rays will only collide with meshes specified here as they need to be copied over to the GPU for calculations
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
            debug_vis=True,
        )
        self.scene.ray_caster_FR_foot = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
            update_period=self.sim.dt,
            offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)), # Starting point 1m above base, doesn't really matter
            mesh_prim_paths=["/World/ground"], # Rays will only collide with meshes specified here as they need to be copied over to the GPU for calculations
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
            debug_vis=True,
        )
        self.scene.ray_caster_RL_foot = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
            update_period=self.sim.dt,
            offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)), # Starting point 1m above base, doesn't really matter
            mesh_prim_paths=["/World/ground"], # Rays will only collide with meshes specified here as they need to be copied over to the GPU for calculations
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
            debug_vis=True,
        )
        self.scene.ray_caster_RR_foot = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
            update_period=self.sim.dt,
            offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)), # Starting point 1m above base, doesn't really matter
            mesh_prim_paths=["/World/ground"], # Rays will only collide with meshes specified here as they need to be copied over to the GPU for calculations
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
            debug_vis=True,
        )

        self.scene.foot_frame_transformer = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FL_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FR_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_foot"),
            ],
            debug_vis=False
        )

        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.7, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.78, 0.78)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = 0
        # reduce the number of terrains to save memory and set lower difficulty range
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.6)
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable any rewards related to minimizing power during eval, since the parameters are subject to change and evals should be comparable to each other.
        # HOWEVER: It's still best practice to look at cost of transport, energy consumption and RMS error instead of rewards!
        self.rewards.minimize_power = None
        # Automatically disable any curriculum whose term_name is "minimize_power"
        for field_name in list(vars(self.curriculum)):
            term = getattr(self.curriculum, field_name)
            if isinstance(term, CurrTerm) and term.params.get("term_name") == "minimize_power":
                setattr(self.curriculum, field_name, None)

@configclass
class Go2RoughTerrainEnvCfgJointStateHistory_PLAY(Go2RoughTerrainEnvCfg_PLAY):
    observations: ObservationsCfgJointStateHistory = ObservationsCfgJointStateHistory()

@configclass
class Go2RoughTerrainEnvCfgFullStateHistory_PLAY(Go2RoughTerrainEnvCfg_PLAY):
    observations: ObservationsCfgFullStateHistory = ObservationsCfgFullStateHistory()

