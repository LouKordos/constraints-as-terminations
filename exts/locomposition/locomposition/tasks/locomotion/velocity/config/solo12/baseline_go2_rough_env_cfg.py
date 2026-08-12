# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matched upstream Isaac Lab rough-terrain baseline for Unitree Go2.

The configuration intentionally keeps the upstream Go2 reward stack, action
implementation, and action scale while matching the LoComposition experimental
conditions. It is explicit because the installed upstream Go2 source currently
contains unrelated working-tree edits and must not be a hidden dependency.
"""

from copy import deepcopy

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

import locomposition.tasks.utils.mdp.commands as cat_commands
import locomposition.tasks.utils.mdp.events as cat_events
import locomposition.tasks.utils.mdp.terminations as cat_terminations
from locomposition.assets.go2_config import UNITREE_GO2_CFG_EVAL, UNITREE_GO2_CFG_TRAIN

from .locomposition_go2_rough_terrain_env_cfg import rough_cfg


DEFAULT_MATCHED_SEED = 46


def _matched_terrain_cfg() -> TerrainImporterCfg:
    terrain_generator = deepcopy(rough_cfg)
    terrain_generator.seed = DEFAULT_MATCHED_SEED
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator,
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


def _matched_height_scanner_cfg(sim_dt: float) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=sim_dt,
        # Keep Isaac Lab's high ray origin and height_scan convention while
        # matching LoComposition's horizontal footprint, placement, and drift.
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 20.0)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            size=[1.0, 0.8],
            resolution=0.08,
            ordering="xy",
        ),
        ray_cast_drift_range={
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
        },
        debug_vis=False,
    )


@configclass
class MatchedGo2EventCfg:
    """LoComposition Go2 randomization, reset, and disturbance conditions."""

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
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "com_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.03, 0.03),
            },
        },
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (-0.05, 0.05),
        },
    )
    push_robot = EventTerm(
        func=cat_events.push_by_setting_velocity_with_random_envs,
        mode="interval",
        is_global_time=True,
        interval_range_s=(0.0, 0.005),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    push_base_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        is_global_time=False,
        interval_range_s=(0.05, 0.1),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_range": (-10.0, 10.0),
            "torque_range": (-0.5, 0.5),
        },
    )


@configclass
class MatchedGo2TerminationsCfg:
    """CaT hard resets without probabilistic CaT constraints."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_thigh"],
            ),
            "threshold": 1.0,
        },
    )
    upside_down = DoneTerm(
        func=cat_terminations.upside_down,
        params={"limit": 1},
    )


@configclass
class MatchedGo2CurriculumCfg:
    """Only the upstream terrain-level curriculum; no CaT curricula."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


def _force_hard_terrain(env, env_ids):
    del env_ids
    terrain = env.scene.terrain
    origins = terrain.terrain_origins
    row_index = min(
        env.cfg.scene.terrain.max_init_terrain_level or (origins.shape[0] - 1),
        origins.shape[0] - 1,
    )
    terrain.configure_env_origins(origins[row_index : row_index + 1].reshape(-1, 3))


def _reset_joints_to_locomposition_eval_pose(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Match the reset pose used by the LoComposition Go2 evaluation environment."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    name_to_id = {name: index for index, name in enumerate(asset.data.joint_names)}

    for joint_name in ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"):
        joint_pos[:, name_to_id[joint_name]] = 1.1
    for joint_name in ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"):
        joint_pos[:, name_to_id[joint_name]] = 0.0
    for joint_name in ("FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"):
        joint_pos[:, name_to_id[joint_name]] = -1.8

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.reset(env_ids)


def _assert_upstream_go2_reward_contract(rewards) -> None:
    expected_weights = {
        "track_lin_vel_xy_exp": 1.5,
        "track_ang_vel_z_exp": 0.75,
        "feet_air_time": 0.01,
        "dof_torques_l2": -0.0002,
        "dof_acc_l2": -2.5e-7,
    }
    for term_name, expected_weight in expected_weights.items():
        actual_weight = getattr(rewards, term_name).weight
        if actual_weight != expected_weight:
            raise ValueError(
                f"Upstream Go2 reward contract changed for {term_name}: "
                f"expected {expected_weight}, got {actual_weight}"
            )
    if rewards.undesired_contacts is not None:
        raise ValueError("Upstream Go2 reward contract changed: undesired_contacts must remain disabled")


@configclass
class BaselineGo2RoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """Upstream Go2 baseline under matched LoComposition conditions."""

    use_deadzone_command: bool = True

    def __post_init__(self):
        super().__post_init__()

        _assert_upstream_go2_reward_contract(self.rewards)

        self.seed = DEFAULT_MATCHED_SEED
        self.scene.num_envs = 7500
        self.scene.env_spacing = 3.0
        self.scene.robot = UNITREE_GO2_CFG_TRAIN.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain = _matched_terrain_cfg()

        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.random_seed = DEFAULT_MATCHED_SEED
        self.sim.solver_type = 0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.max_position_iteration_count = 24
        self.sim.max_velocity_iteration_count = 6
        self.sim.bounce_threshold_velocity = 0.2
        self.sim.gpu_max_rigid_contact_count = 33554432
        self.sim.physx.gpu_max_rigid_patch_count = 568462
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner = _matched_height_scanner_cfg(self.sim.dt)

        # Unlike the default sampler, CaT's command term applies a 3-D deadzone,
        # opportunistic per-step resampling, and random yaw inversion; its
        # override also does not enforce the parent's is_standing_env mask. The
        # switch changes only that implementation, never the matched ranges.
        self.commands.base_velocity = cat_commands.UniformVelocityCommandWithDeadzoneCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=False,
            debug_vis=True,
            velocity_deadzone=0.1,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.3, 1.0),
                lin_vel_y=(-0.7, 0.7),
                ang_vel_z=(-0.78, 0.78),
            ),
        )
        if not self.use_deadzone_command:
            self.commands.base_velocity.class_type = mdp.UniformVelocityCommand

        # Raw ObsTerm corruption matches CaT. Functions, ordering, scale,
        # clipping, and height sign/offset stay upstream; noise is applied before
        # scale. CaT's additional custom height pose jitter is not reproduced
        # because doing so would replace the requested upstream height function.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.001, n_max=0.001)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.height_scan.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.enable_corruption = True

        self.actions.joint_pos.scale = 0.25
        self.events = MatchedGo2EventCfg()
        self.terminations = MatchedGo2TerminationsCfg()
        self.curriculum = MatchedGo2CurriculumCfg()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 1


@configclass
class BaselineGo2RoughEnvCfg_PLAY(BaselineGo2RoughEnvCfg):
    """One-environment Go2 baseline for matched quantitative evaluation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.scene.robot = UNITREE_GO2_CFG_EVAL.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.observations.policy.enable_corruption = True

        self.events.force_hard_terrain = EventTerm(func=_force_hard_terrain, mode="startup")
        self.events.reset_robot_joints = EventTerm(
            func=_reset_joints_to_locomposition_eval_pose,
            mode="reset",
        )
        self.events.push_robot = None
        self.events.push_base_wrench = None

        self.scene.terrain.max_init_terrain_level = 0
        terrain_generator = self.scene.terrain.terrain_generator
        terrain_generator.difficulty_range = (0.0, 0.6)
        terrain_generator.num_rows = 5
        terrain_generator.num_cols = 5
        terrain_generator.curriculum = False
