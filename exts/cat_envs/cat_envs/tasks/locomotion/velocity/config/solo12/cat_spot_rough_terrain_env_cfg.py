# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CaT rough-terrain configurations for Boston Dynamics Spot."""

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.actuators import DelayedPDActuator, RemotizedPDActuator
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer import FrameTransformerCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab_assets.robots.spot import SPOT_CFG

from .cat_go2_rough_terrain_env_cfg import Go2RoughTerrainEnvCfg, force_hard_terrain


class _DeterministicCircularBuffer(CircularBuffer):
    """Isaac Lab 2.3.1 circular buffer compatible with deterministic CUDA."""

    def append(self, data: torch.Tensor):
        if data.shape[0] != self.batch_size:
            raise ValueError(
                f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'"
            )

        data = data.to(self._device)
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty(
                (self.max_length, *data.shape),
                dtype=data.dtype,
                device=self._device,
            )

        self._pointer = (self._pointer + 1) % self.max_length
        self._buffer[self._pointer] = data

        # Isaac Lab 2.3.1 uses advanced indexed assignment here. PyTorch expands
        # that assignment incorrectly on CUDA when deterministic algorithms are
        # enabled, so initialize reset batches with an elementwise mask instead.
        is_first_push = self._num_pushes == 0
        if torch.any(is_first_push):
            first_push_mask = is_first_push.reshape(
                1,
                self.batch_size,
                *([1] * (data.ndim - 1)),
            )
            self._buffer = torch.where(first_push_mask, data.unsqueeze(0), self._buffer)
        self._num_pushes += 1


def _replace_delay_buffers(actuator: DelayedPDActuator):
    """Retain the installed delay model while replacing only its broken storage."""
    batch_size = actuator.positions_delay_buffer.batch_size
    device = actuator.positions_delay_buffer.device

    for attribute_name in (
        "positions_delay_buffer",
        "velocities_delay_buffer",
        "efforts_delay_buffer",
    ):
        delay_buffer = DelayBuffer(actuator.cfg.max_delay, batch_size, device)
        delay_buffer._circular_buffer = _DeterministicCircularBuffer(
            actuator.cfg.max_delay + 1,
            batch_size,
            device,
        )
        setattr(actuator, attribute_name, delay_buffer)


class DeterministicDelayedPDActuator(DelayedPDActuator):
    """Installed delayed PD model with deterministic-compatible buffer storage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _replace_delay_buffers(self)


class DeterministicRemotizedPDActuator(RemotizedPDActuator):
    """Installed remotized PD model with deterministic-compatible buffer storage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _replace_delay_buffers(self)


# Keep actions and proprioceptive observations in the articulation's runtime order.
SPOT_JOINT_NAMES = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]
SPOT_JOINT_PATTERNS = [".*_hx", ".*_hy", ".*_kn"]
SPOT_FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


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
class SpotRoughTerrainEnvCfg(Go2RoughTerrainEnvCfg):
    """Shared CaT rough-terrain MDP with Spot-specific embodiment settings."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SPOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={
                "spot_hip": SPOT_CFG.actuators["spot_hip"].replace(
                    class_type=DeterministicDelayedPDActuator
                ),
                "spot_knee": SPOT_CFG.actuators["spot_knee"].replace(
                    class_type=DeterministicRemotizedPDActuator
                ),
            },
        )

        self.actions.joint_pos.joint_names = list(SPOT_JOINT_NAMES)
        self.actions.joint_pos.scale = 0.2
        self.observations.policy.joint_pos_history.params["names"] = list(SPOT_JOINT_NAMES)
        self.observations.policy.joint_vel_history.params["names"] = list(SPOT_JOINT_NAMES)

        self.scene.ray_caster.prim_path = "{ENV_REGEX_NS}/Robot/body"
        self.scene.ray_caster_height_constraints.prim_path = "{ENV_REGEX_NS}/Robot/body"

        self.events.randomize_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["body"]
        )
        self.events.randomize_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["body"]
        )
        self.events.randomize_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
        }
        self.events.push_base_wrench.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["body"]
        )
        self.events.push_base_wrench.params["force_range"] = (-10.0, 10.0)
        self.events.push_base_wrench.params["torque_range"] = (-0.5, 0.5)

        for term_name in ("joint_torque", "joint_velocity", "joint_acceleration", "action_rate"):
            getattr(self.constraints, term_name).params["names"] = list(SPOT_JOINT_PATTERNS)
        self.constraints.joint_torque.params["limit"] = 80.0
        self.constraints.joint_velocity.params["limit"] = 20.0
        self.constraints.joint_acceleration.params["limit"] = 800.0
        self.constraints.action_rate.params["limit"] = 80.0

        self.constraints.contact.params["names"] = ["body", ".*_uleg"]
        self.constraints.foot_contact_force.params.update(limit=800.0, names=[".*_foot"])
        self.constraints.front_hfe_position.params.update(limit=2.0, names=[".*_hy"])
        self.constraints.hip_position.params["names"] = [".*_hx"]
        self.constraints.no_move.params.update(
            names=list(SPOT_JOINT_PATTERNS),
            joint_vel_limit=2.0,
        )

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["body", ".*_uleg"]
        )
        self.curriculum.power.params["end_weight"] = 0.00320

        # Preserve Spot's installed 0--8 ms actuator delay while retaining 50 Hz control.
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.ray_caster.update_period = self.sim.dt
        self.scene.ray_caster_height_constraints.update_period = self.sim.dt


@configclass
class SpotRoughTerrainEnvCfg_PLAY(SpotRoughTerrainEnvCfg):
    """One-environment disturbance-free Spot configuration for evaluation."""

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

        for foot_name in SPOT_FOOT_NAMES:
            setattr(
                self.scene,
                f"ray_caster_{foot_name}",
                _foot_ray_caster(foot_name, self.sim.dt),
            )
        self.scene.foot_frame_transformer = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}"
                )
                for foot_name in SPOT_FOOT_NAMES
            ],
            debug_vis=False,
        )

        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.6)
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.rewards.minimize_power = None
        self.curriculum.power = None
