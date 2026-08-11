# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CaT rough-terrain configurations for ANYbotics ANYmal C.

The Go2 configuration remains the source of shared task behavior. This module
only replaces settings whose meaning depends on the robot embodiment.
"""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer import FrameTransformerCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from .cat_go2_rough_terrain_env_cfg import Go2RoughTerrainEnvCfg, force_hard_terrain


# Keep actions and proprioceptive observations in the articulation's runtime order.
ANYMAL_C_JOINT_NAMES = [
    "LF_HAA",
    "LH_HAA",
    "RF_HAA",
    "RH_HAA",
    "LF_HFE",
    "LH_HFE",
    "RF_HFE",
    "RH_HFE",
    "LF_KFE",
    "LH_KFE",
    "RF_KFE",
    "RH_KFE",
]
ANYMAL_C_JOINT_PATTERNS = [".*HAA", ".*HFE", ".*KFE"]
ANYMAL_C_FOOT_NAMES = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]


def _foot_ray_caster(foot_name: str, sim_dt: float) -> RayCasterCfg:
    """Create the one-ray ground-height sensor used by evaluation metrics."""
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}",
        update_period=sim_dt,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
        debug_vis=True,
    )


@configclass
class AnymalCRoughTerrainEnvCfg(Go2RoughTerrainEnvCfg):
    """Go2 CaT task with the hard embodiment differences replaced for ANYmal C."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos.joint_names = list(ANYMAL_C_JOINT_NAMES)
        self.actions.joint_pos.scale = 0.5
        self.observations.policy.joint_pos_history.params["names"] = list(ANYMAL_C_JOINT_NAMES)
        self.observations.policy.joint_vel_history.params["names"] = list(ANYMAL_C_JOINT_NAMES)

        # ANYmal C uses the same root-link name, but set every selector explicitly so
        # changes to the Go2 config cannot silently leak robot-specific names here.
        self.events.randomize_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base"])
        self.events.randomize_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base"])
        self.events.randomize_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
        }
        self.events.push_base_wrench.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base"])
        self.events.push_base_wrench.params["force_range"] = (-10.0, 10.0)
        self.events.push_base_wrench.params["torque_range"] = (-0.5, 0.5)

        for term_name in ("joint_torque", "joint_velocity", "joint_acceleration", "action_rate"):
            getattr(self.constraints, term_name).params["names"] = list(ANYMAL_C_JOINT_PATTERNS)
        self.constraints.joint_torque.params["limit"] = 80.0
        self.constraints.joint_velocity.params["limit"] = 12.0
        self.constraints.joint_acceleration.params["limit"] = 600.0
        self.constraints.action_rate.params["limit"] = 80.0

        self.constraints.contact.params["names"] = ["base", ".*_THIGH"]
        self.constraints.foot_contact_force.params.update(limit=1000.0, names=[".*_FOOT"])
        self.constraints.front_hfe_position.params.update(limit=1.5, names=[".*HFE"])
        self.constraints.hip_position.params["names"] = [".*HAA"]
        self.constraints.hip_position.params["limit"] = 0.8
        self.constraints.no_move.params.update(
            names=list(ANYMAL_C_JOINT_PATTERNS),
            joint_vel_limit=2.0,
        )

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["base", ".*_THIGH"]
        )
        self.curriculum.power.params["end_weight"] = 0.00230


@configclass
class AnymalCRoughTerrainEnvCfg_PLAY(AnymalCRoughTerrainEnvCfg):
    """Small disturbance-free ANYmal C environment used by ``scripts/eval.py``."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 8
        self.apply_elevation_map_point_noise = False
        self.observations.policy.enable_corruption = True

        self.events.force_hard_terrain = EventTerm(func=force_hard_terrain, mode="startup")
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.push_robot = None
        self.events.push_base_wrench = None

        for foot_name in ANYMAL_C_FOOT_NAMES:
            setattr(
                self.scene,
                f"ray_caster_{foot_name}",
                _foot_ray_caster(foot_name, self.sim.dt),
            )
        self.scene.foot_frame_transformer = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}")
                for foot_name in ANYMAL_C_FOOT_NAMES
            ],
            debug_vis=False,
        )

        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.6)
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Evaluation compares physical energy and tracking metrics directly; the
        # training reward is intentionally absent, exactly as in the Go2 play cfg.
        self.rewards.minimize_power = None
        self.curriculum.power = None
