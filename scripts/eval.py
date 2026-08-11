import argparse
import atexit
import os
import inspect
import glob
import json
import yaml
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Determinism
os.environ["OMNICLIENT_HUB_MODE"] = "disabled"
import torch
import time
import zmq
from functools import partial
import sys
sys.stdout.reconfigure(line_buffering=True)
print = partial(print, flush=True) # For cluster runs
import gymnasium as gym
from isaaclab.app import AppLauncher
from tqdm import tqdm
import fcntl
from queue import Queue
import subprocess
import shutil
from pathlib import Path
import os
import re
import yaml
import uuid
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
from metrics_utils import (
    build_scenario_analysis_mask,
    compute_gait_dynamics_metrics,
    compute_summary_metrics,
    summarize_metric,
)
from rebuttal_report import REBUTTAL_DYNAMICS_SCENARIOS, write_rebuttal_reports
from task_naming import is_locomposition_go2_task
eval_script_path = os.path.dirname(os.path.abspath(__file__))

CONTACT_FORCE_THRESHOLD_NEWTONS = 1.0

UPSTREAM_GO2_HARDCODED_CONSTRAINT_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "joint_torque": (-20.0, 20.0),
    "joint_velocity": (-25.0, 25.0),
    "joint_acceleration": (-800.0, 800.0),
    "action_rate": (-80.0, 80.0),
    "foot_contact_force": (0.0, 300.0),
}


def apply_policy_action_delay(
    policy_action: torch.Tensor,
    queued_action: Optional[tuple[torch.Tensor, ...]],
    delay_steps: int,
) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
    """Return the action to apply now and the action queued for the next policy step."""
    if delay_steps == 0:
        return policy_action, None
    if delay_steps not in (1, 2):
        raise ValueError(f"Only zero-, one-, or two-step action delay is supported, got {delay_steps}.")

    action_queue = list(queued_action or ())
    action_queue.append(policy_action.clone())
    if len(action_queue) <= delay_steps:
        applied_action = torch.zeros_like(policy_action)
    else:
        applied_action = action_queue.pop(0)
    return applied_action, tuple(action_queue)


def select_fixed_command_scenarios(
    scenarios: list[tuple],
    selected_scenario: Optional[str],
    selected_scenarios: Optional[tuple[str, ...]] = None,
) -> list[tuple]:
    """Optionally restrict evaluation to one named fixed-command scenario."""
    if selected_scenario is not None and selected_scenarios is not None:
        raise ValueError("Select either one fixed scenario or a named scenario set, not both.")

    if selected_scenarios is not None:
        requested = set(selected_scenarios)
        selected = [scenario for scenario in scenarios if scenario[0] in requested]
        found = {scenario[0] for scenario in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Unknown fixed scenarios: {missing}")
        return selected

    if selected_scenario is None:
        return scenarios

    selected = [scenario for scenario in scenarios if scenario[0] == selected_scenario]
    if not selected:
        raise ValueError(f"Unknown fixed scenario: {selected_scenario}")
    return selected


def compute_contact_state(
    per_foot_resultant_force: np.ndarray,
    threshold_newtons: float = CONTACT_FORCE_THRESHOLD_NEWTONS,
) -> np.ndarray:
    """Return binary foot contact using the same strict 1 N convention as training."""
    if threshold_newtons < 0.0:
        raise ValueError(f"Contact threshold must be non-negative, got {threshold_newtons}.")
    forces = np.asarray(per_foot_resultant_force)
    return (forces > threshold_newtons).astype(np.int8)


def compute_action_rate(
    applied_action: np.ndarray,
    previous_applied_action: Optional[np.ndarray],
    step_dt: float,
) -> np.ndarray:
    """Compute absolute applied-action change per second."""
    if previous_applied_action is None:
        return np.zeros_like(applied_action)
    return np.abs(applied_action - previous_applied_action) / step_dt


def compute_scenario_completion(
    scenarios: list[tuple],
    random_sim_steps: int,
    fixed_scenario_steps: int,
    automatic_reset_steps: list[int],
) -> dict[str, bool]:
    """Report whether each scenario reached its final step without an earlier reset."""
    completion = {}
    for index, (scenario_tag, *_) in enumerate(scenarios):
        start = random_sim_steps + index * fixed_scenario_steps
        expected_timeout_step = start + fixed_scenario_steps - 1
        completion[scenario_tag] = not any(
            start <= reset_step < expected_timeout_step
            for reset_step in automatic_reset_steps
        )
    return completion


MATCHED_BASELINE_CONSTRAINT_BOUNDS: Dict[
    str, Dict[str, Tuple[Optional[float], Optional[float]]]
] = {
    "go2": dict(UPSTREAM_GO2_HARDCODED_CONSTRAINT_BOUNDS),
    "anymal_c": {
        "joint_torque": (-80.0, 80.0),
        "joint_velocity": (-12.0, 12.0),
        "joint_acceleration": (-800.0, 800.0),
        "action_rate": (-128.0, 128.0),
        "foot_contact_force": (0.0, 1000.0),
    },
    "spot": {
        "joint_torque": (-80.0, 80.0),
        "joint_velocity": (-20.0, 20.0),
        "joint_acceleration": (-800.0, 800.0),
        "action_rate": (-320.0, 320.0),
        "foot_contact_force": (0.0, 1000.0),
    },
}

MATCHED_BASELINE_TASK_PROFILES = {
    "baseline-go2-rough-terrain-v0": "go2",
    "baseline-go2-rough-terrain-play-v0": "go2",
    "baseline-anymal-c-rough-terrain-v0": "anymal_c",
    "baseline-anymal-c-rough-terrain-play-v0": "anymal_c",
    "baseline-spot-rough-terrain-v0": "spot",
    "baseline-spot-rough-terrain-play-v0": "spot",
    "baseline-go2-anymal-c-tuning-rough-terrain-v0": "go2",
    "baseline-go2-anymal-c-tuning-rough-terrain-play-v0": "go2",
    "baseline-anymal-c-go2-tuning-rough-terrain-v0": "anymal_c",
    "baseline-anymal-c-go2-tuning-rough-terrain-play-v0": "anymal_c",
}
MATCHED_BASELINE_TASK_IDS = set(MATCHED_BASELINE_TASK_PROFILES)


@dataclass(frozen=True)
class RobotEvalProfile:
    """The small set of geometry/name facts evaluation cannot infer generically."""

    name: str
    root_link: str
    foot_links: tuple[str, str, str, str]
    spawn_height: float
    sole_offset: float
    joint_role_mapping: dict[str, dict[str, str]]


ROBOT_EVAL_PROFILES = {
    "go2": RobotEvalProfile(
        name="go2",
        root_link="base",
        foot_links=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        spawn_height=0.4,
        sole_offset=0.0228,
        joint_role_mapping={
            "hip_joint": {
                "FL": "FL_hip_joint", "FR": "FR_hip_joint",
                "RL": "RL_hip_joint", "RR": "RR_hip_joint",
            },
            "thigh_joint": {
                "FL": "FL_thigh_joint", "FR": "FR_thigh_joint",
                "RL": "RL_thigh_joint", "RR": "RR_thigh_joint",
            },
            "calf_joint": {
                "FL": "FL_calf_joint", "FR": "FR_calf_joint",
                "RL": "RL_calf_joint", "RR": "RR_calf_joint",
            },
        },
    ),
    "anymal_c": RobotEvalProfile(
        name="anymal_c",
        root_link="base",
        foot_links=("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"),
        spawn_height=0.6,
        sole_offset=0.0,
        joint_role_mapping={
            "HAA": {"FL": "LF_HAA", "FR": "RF_HAA", "RL": "LH_HAA", "RR": "RH_HAA"},
            "HFE": {"FL": "LF_HFE", "FR": "RF_HFE", "RL": "LH_HFE", "RR": "RH_HFE"},
            "KFE": {"FL": "LF_KFE", "FR": "RF_KFE", "RL": "LH_KFE", "RR": "RH_KFE"},
        },
    ),
    "spot": RobotEvalProfile(
        name="spot",
        root_link="body",
        foot_links=("fl_foot", "fr_foot", "hl_foot", "hr_foot"),
        spawn_height=0.5,
        sole_offset=0.0,
        joint_role_mapping={
            "hx": {"FL": "fl_hx", "FR": "fr_hx", "RL": "hl_hx", "RR": "hr_hx"},
            "hy": {"FL": "fl_hy", "FR": "fr_hy", "RL": "hl_hy", "RR": "hr_hy"},
            "kn": {"FL": "fl_kn", "FR": "fr_kn", "RL": "hl_kn", "RR": "hr_kn"},
        },
    ),
}


def resolve_robot_eval_profile(task_name: str) -> RobotEvalProfile:
    """Resolve explicit new embodiments and default unknown legacy runs to Go2."""
    task_name_lower = task_name.lower()
    matched_baseline_profile = MATCHED_BASELINE_TASK_PROFILES.get(task_name_lower)
    if matched_baseline_profile is not None:
        return ROBOT_EVAL_PROFILES[matched_baseline_profile]
    if "anymal" in task_name_lower:
        return ROBOT_EVAL_PROFILES["anymal_c"]
    if "spot" in task_name_lower:
        return ROBOT_EVAL_PROFILES["spot"]
    if "go2" in task_name_lower:
        return ROBOT_EVAL_PROFILES["go2"]

    print(
        f"[WARN] Task '{task_name}' has no recognized embodiment marker; "
        "assuming legacy Go2 for backward compatibility."
    )
    return ROBOT_EVAL_PROFILES["go2"]


def adjust_fixed_command_spawn_heights(fixed_command_scenarios, profile: RobotEvalProfile):
    """Preserve terrain-relative scenario placement while changing robot height."""
    if profile.name == "go2":
        # Preserve the exact legacy tensors and object structure for regression.
        return fixed_command_scenarios

    height_delta = profile.spawn_height - ROBOT_EVAL_PROFILES["go2"].spawn_height
    adjusted_scenarios = []
    for scenario_tag, command, (spawn_position, spawn_orientation) in fixed_command_scenarios:
        adjusted_position = spawn_position.clone()
        adjusted_position[2] += height_delta
        adjusted_scenarios.append(
            (scenario_tag, command, (adjusted_position, spawn_orientation))
        )
    return adjusted_scenarios


def resolve_root_body_index(body_names: list[str], profile: RobotEvalProfile) -> int:
    """Return the profile's root-body index or fail before collecting wrong telemetry."""
    try:
        return body_names.index(profile.root_link)
    except ValueError as exception:
        raise ValueError(
            f"Evaluation profile '{profile.name}' requires root body '{profile.root_link}', "
            f"but the articulation exposes {body_names}."
        ) from exception


def set_global_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_seed_to_env_cfg(env_cfg, seed: int):
    env_cfg.seed = seed
    env_cfg.sim.random_seed = seed

    terrain_cfg = getattr(env_cfg.scene, "terrain", None)
    terrain_generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_generator_cfg is not None:
        terrain_generator_cfg.seed = seed

    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

    velocity_mdp.terrain_levels_vel.seed = seed


def is_upstream_go2_rough_task(task_name: str) -> bool:
    """
    Detect the upstream Isaac Lab rough-terrain Unitree Go2 task family.

    This intentionally does not match LoComposition's Go2 task names, because
    those should keep using the constraint bounds saved in params/env.yaml.
    """
    task_name_lower = task_name.lower()

    if is_locomposition_go2_task(task_name):
        return False

    mentions_rough = "rough" in task_name_lower
    mentions_unitree_go2 = "unitree-go2" in task_name_lower or "unitree_go2" in task_name_lower

    return mentions_rough and mentions_unitree_go2


def is_matched_baseline_task(task_name: str) -> bool:
    """Return whether ``task_name`` is one of this repository's matched baselines."""
    return task_name.lower() in MATCHED_BASELINE_TASK_IDS


def get_matched_baseline_constraint_bounds(
    profile_name: str,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return project operational-limit thresholds for baseline summaries."""
    try:
        return dict(MATCHED_BASELINE_CONSTRAINT_BOUNDS[profile_name])
    except KeyError as exception:
        raise ValueError(f"No matched-baseline constraint bounds for profile '{profile_name}'") from exception


def get_hardcoded_upstream_go2_constraint_bounds() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Return eval-only constraint bounds for upstream Isaac Lab rough-terrain Unitree Go2 baselines.

    These bounds mirror the constraint thresholds used for the comparable
    LoComposition evaluation metrics, but only include terms already supported by
    metrics_utils.compute_summary_metrics without changing metrics_utils.py.

    Not included here by design:
    - base_orientation: user explicitly does not care about this metric here.
    - contact: user explicitly does not care about this metric here.
    - front_hfe_position: user explicitly does not care about this metric here,
      and metrics_utils.py currently does not compute per-pattern joint-position
      constraint violations from a named term without modification.
    """
    return dict(UPSTREAM_GO2_HARDCODED_CONSTRAINT_BOUNDS)


def format_constraint_bounds_for_logging(
    constraint_bounds: Dict[str, Tuple[Optional[float], Optional[float]]]
) -> str:
    if not constraint_bounds:
        return "{}"

    lines = ["{"]
    for key, (lower_bound, upper_bound) in constraint_bounds.items():
        lines.append(f"    {key}: ({lower_bound}, {upper_bound})")
    lines.append("}")
    return "\n".join(lines)


def infer_checkpoint_input_dimensions(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Return in-features of the first linear layer saved in `state_dict`.
    Assumes CleanRL naming pattern 'actor.0.weight' or similar.
    """
    for k, v in state_dict.items():
        if k.endswith(".0.weight") and v.ndim == 2:
            return v.shape[1]
    raise RuntimeError("Could not infer input dimension from checkpoint")


def load_checkpoint_for_format_detection(checkpoint_path: str) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception as exception:
        print(f"[WARN] weights_only=True checkpoint inspection failed: {exception}")
        print("[WARN] Falling back to weights_only=False for local checkpoint inspection.")
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def detect_policy_backend_from_checkpoint(checkpoint_object: dict) -> str:
    if not isinstance(checkpoint_object, dict):
        raise RuntimeError(f"Unsupported checkpoint object type: {type(checkpoint_object)}")

    if "model_state_dict" in checkpoint_object:
        return "rsl_rl"

    modern_rsl_rl_actor_keys = ("actor_state_dict", "student_state_dict")
    if any(isinstance(checkpoint_object.get(key), dict) for key in modern_rsl_rl_actor_keys):
        return "rsl_rl"

    if any(isinstance(value, torch.Tensor) for value in checkpoint_object.values()):
        return "clean_rl"

    raise RuntimeError(
        "Could not infer checkpoint backend. Expected an RSL-RL checkpoint with "
        "'model_state_dict', 'actor_state_dict', or 'student_state_dict', or a "
        "CleanRL-style plain state_dict."
    )


def extract_cleanrl_state_dict(checkpoint_object: dict) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint_object:
        raise RuntimeError("Received an RSL-RL checkpoint where a CleanRL state_dict was expected.")

    tensor_values = [value for value in checkpoint_object.values() if isinstance(value, torch.Tensor)]
    if not tensor_values:
        raise RuntimeError("CleanRL checkpoint does not appear to contain tensor parameters.")

    return checkpoint_object


class LegacyRslRlPolicy(torch.nn.Module):
    """Inference-only wrapper for actors saved by older RSL-RL versions."""

    def __init__(self, actor: torch.nn.Sequential):
        super().__init__()
        self.actor = actor

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        if "policy" not in observations:
            raise ValueError("Legacy RSL-RL policy requires a 'policy' observation group.")
        return self.actor(observations["policy"])


def build_legacy_rsl_rl_policy(
    checkpoint_object: dict,
    *,
    activation_name: str,
    expected_observation_dim: int,
    expected_action_dim: int,
    device: torch.device | str,
) -> LegacyRslRlPolicy:
    """Reconstruct a deterministic actor from a legacy RSL-RL checkpoint."""
    model_state = checkpoint_object.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Legacy RSL-RL checkpoint is missing a model_state_dict.")

    unsupported_actor_keys = [
        key
        for key in model_state
        if isinstance(key, str)
        and key.startswith("actor.")
        and re.fullmatch(r"actor\.\d+\.(weight|bias)", key) is None
    ]
    if unsupported_actor_keys:
        raise ValueError(
            "Legacy RSL-RL checkpoint contains unsupported actor parameter(s): "
            f"{sorted(unsupported_actor_keys)}."
        )

    actor_layers: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in model_state.items():
        match = re.fullmatch(r"actor\.(\d+)\.(weight|bias)", key)
        if match is not None:
            actor_layers.setdefault(int(match.group(1)), {})[match.group(2)] = value

    layer_indices = sorted(actor_layers)
    if not layer_indices:
        raise ValueError("Legacy RSL-RL checkpoint contains no actor linear layers.")
    if layer_indices != list(range(0, 2 * len(layer_indices), 2)):
        raise ValueError(
            "Legacy RSL-RL actor linear layers must use sequential indices 0, 2, 4, ...; "
            f"found {layer_indices}."
        )

    activation_factories = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh,
    }
    normalized_activation = activation_name.lower()
    if normalized_activation not in activation_factories:
        raise ValueError(
            f"Unsupported legacy RSL-RL activation '{activation_name}'. "
            f"Supported activations: {sorted(activation_factories)}."
        )

    modules: list[torch.nn.Module] = []
    previous_output_dim = None
    for layer_position, layer_index in enumerate(layer_indices):
        layer_state = actor_layers[layer_index]
        if set(layer_state) != {"weight", "bias"}:
            raise ValueError(
                f"Legacy RSL-RL actor layer {layer_index} must contain both weight and bias."
            )

        weight = layer_state["weight"]
        bias = layer_state["bias"]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"Legacy RSL-RL actor layer {layer_index} weight must be a matrix.")
        if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
            raise ValueError(f"Legacy RSL-RL actor layer {layer_index} bias must be a vector.")

        output_dim, input_dim = weight.shape
        if bias.shape[0] != output_dim:
            raise ValueError(
                f"Legacy RSL-RL actor layer {layer_index} bias has dimension {bias.shape[0]}, "
                f"but its weight has {output_dim} outputs."
            )
        if previous_output_dim is not None and input_dim != previous_output_dim:
            raise ValueError(
                f"Legacy RSL-RL actor layer {layer_index} expects {input_dim} inputs, "
                f"but the previous layer outputs {previous_output_dim}."
            )

        modules.append(torch.nn.Linear(input_dim, output_dim))
        if layer_position < len(layer_indices) - 1:
            modules.append(activation_factories[normalized_activation]())
        previous_output_dim = output_dim

    first_input_dim = actor_layers[layer_indices[0]]["weight"].shape[1]
    if first_input_dim != expected_observation_dim:
        raise ValueError(
            f"Legacy RSL-RL actor expects observation dimension {first_input_dim}, "
            f"but the environment provides observation dimension {expected_observation_dim}."
        )
    if previous_output_dim != expected_action_dim:
        raise ValueError(
            f"Legacy RSL-RL actor outputs action dimension {previous_output_dim}, "
            f"but the environment requires action dimension {expected_action_dim}."
        )

    actor = torch.nn.Sequential(*modules)
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in model_state.items()
        if re.fullmatch(r"actor\.\d+\.(weight|bias)", key)
    }
    actor.load_state_dict(actor_state, strict=True)
    return LegacyRslRlPolicy(actor).to(device).eval()


def build_rsl_rl_runner_cfg_dict(agent_cfg) -> dict:
    runner_cfg_dict = agent_cfg.to_dict()

    obs_groups = runner_cfg_dict.get("obs_groups")
    if not isinstance(obs_groups, dict) or not obs_groups:
        runner_cfg_dict["obs_groups"] = {
            "actor": ["policy"],
            "critic": ["policy"],
        }
    else:
        if "actor" not in runner_cfg_dict["obs_groups"]:
            runner_cfg_dict["obs_groups"]["actor"] = ["policy"]
        if "critic" not in runner_cfg_dict["obs_groups"]:
            runner_cfg_dict["obs_groups"]["critic"] = ["policy"]

    return runner_cfg_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Play an RL agent with detailed logging.")
    parser.add_argument("--run_dir", type=str, required=True, help="ABSOLUTE path to directory containing model checkpoints and params.")
    parser.add_argument("--eval_checkpoint", type=str, default=None, help="Optionally specify the model save checkpoint number instead of automatically using the last saved one.")
    parser.add_argument("--random_sim_step_length", type=int, default=4000, help="Number of steps to run with random commands and spawn points. Standardized tests like standing and walking forward will always run.")
    parser.add_argument(
        "--fixed_command_sim_steps",
        type=int,
        default=500,
        help=(
            "Steps per standardized fixed-command scenario. Keep the default for reported metrics; "
            "shorter values are intended only for evaluation-pipeline smoke tests."
        ),
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. If you change this, hell will break loose")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-Play-v0", help="Name of the task/environment.")
    parser.add_argument(
        "--use_training_go2_config",
        action="store_true",
        help="Temporarily evaluate LoComposition Go2 tasks with UNITREE_GO2_CFG_TRAIN instead of the PLAY task's eval robot config.",
    )
    parser.add_argument("--policy_backend", choices=["auto", "clean_rl", "rsl_rl"], default="auto", help="Policy checkpoint backend. Use auto unless debugging.")
    parser.add_argument("--agent_entry_point", type=str, default="rsl_rl_cfg_entry_point", help="Gym registry entry point key for the RSL-RL agent config.")
    parser.add_argument("--downscale_upstream_go2_tracking_rewards", action=argparse.BooleanOptionalAction, default=True, help="Evaluate upstream Go2 tracking rewards with the custom-env common scale: 1.5->1.0 and 0.75->0.5.")
    parser.add_argument("--foot_vel_height_threshold", type=float, default=0.02, help="Maximum foot height to include in the foot-velocity-vs-height plot.")
    parser.add_argument("--num_plot_jobs_in_parallel", type=int, default=2, help="Number of plot generation jobs to run in parallel.")
    parser.add_argument("--plot_job_stagger_delay", type=int, default=10, help="Delay in seconds between starting each plot generation job in a parallel batch.")
    parser.add_argument("--delay_joints", type=int, default=0, help="Latency steps for joint pos/vel.")
    parser.add_argument("--delay_imu", type=int, default=0, help="Latency steps for IMU (ang vel/gravity).")
    parser.add_argument("--delay_action_history", type=int, default=0, help="Latency steps for action history.")
    parser.add_argument("--delay_height_map", type=int, default=0, help="Latency steps for height map.")
    parser.add_argument("--delay_actions", type=int, choices=(0, 1, 2), default=0, help="Policy steps by which execution of actions is delayed.")
    parser.add_argument("--fixed_scenario", type=str, default=None, help="Run only the named fixed-command scenario.")
    parser.add_argument(
        "--rebuttal_scenarios_only",
        action="store_true",
        default=False,
        help="Run only the four fixed scenarios used by the gait-dynamics rebuttal report.",
    )
    parser.add_argument("--skip_cot_sweep", action="store_true", default=False, help="Turn off 0.2m/s increment forward walking on flat terrain that is used for Cost of Transport estimation")
    # Note that changing the seed will change terrain config and thus the fixed eval command scenarios, as well as random commands in the beginning!
    parser.add_argument("--seed", type=int, required=False, default=46, help="Seed for numpy, torch, env, terrain, terrain generator etc.. Good seeds for eval are 44, 46, 49")

    sys.path.insert(0, os.path.join(eval_script_path, "clean_rl"))
    import cli_args  # isort: skip
    cli_args.add_clean_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    arguments = parser.parse_args()
    if arguments.random_sim_step_length < 0:
        parser.error("--random_sim_step_length must be non-negative")
    if arguments.fixed_command_sim_steps <= 0:
        parser.error("--fixed_command_sim_steps must be positive")
    arguments.enable_cameras = True  # Video
    return arguments


def get_latest_checkpoint(run_directory: str) -> str:
    pattern = os.path.join(run_directory, "model_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {run_directory}")

    def extract_index(path: str) -> int:
        name = os.path.basename(path)
        num_str = name.split("_")[-1].split(".")[0]
        return int(num_str)

    latest = max(files, key=extract_index)
    return latest

def apply_training_action_scale(env_cfg, params_directory: str):
    """
    Load actions.joint_pos.scale from the saved training env config and apply it
    to the eval env cfg before gym.make() constructs the action manager.
    """
    yaml_file = os.path.join(params_directory, "env.yaml")
    text = open(yaml_file).read()
    text = re.sub(r"!!python\S*", "", text)
    cfg = yaml.safe_load(text)

    try:
        training_action_scale = cfg["actions"]["joint_pos"]["scale"]
    except KeyError as exception:
        raise RuntimeError(
            f"Could not find actions.joint_pos.scale in training env config: {yaml_file}"
        ) from exception

    if isinstance(training_action_scale, (int, float)):
        training_action_scale = float(training_action_scale)
    elif isinstance(training_action_scale, dict):
        training_action_scale = {
            str(pattern): float(scale)
            for pattern, scale in training_action_scale.items()
        }
    else:
        raise TypeError(
            "Unsupported actions.joint_pos.scale type in training env config. "
            f"Expected float/int or dict, got {type(training_action_scale)}: {training_action_scale!r}"
        )

    eval_action_scale_before = env_cfg.actions.joint_pos.scale
    env_cfg.actions.joint_pos.scale = training_action_scale

    print("[INFO] Loaded action scale from training params/env.yaml")
    print(f"[INFO] Training actions.joint_pos.scale = {training_action_scale}")
    print(f"[INFO] Eval actions.joint_pos.scale before override = {eval_action_scale_before}")
    print(f"[INFO] Eval actions.joint_pos.scale after override = {env_cfg.actions.joint_pos.scale}")

    return training_action_scale, eval_action_scale_before

def load_constraint_bounds(params_directory: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Returns a dict mapping each constraint key (either a global term
    like 'joint_torque' or an individual joint name) to a (lb, ub) tuple.
    - joint_position    -> (None, limit)
    - joint_position_when_moving_forward -> (default-limit, default+limit)
    - foot_contact_force -> (0, limit)
    - everything else   -> (-limit, +limit)
    """
    yaml_file = os.path.join(params_directory, 'env.yaml')
    text = open(yaml_file).read()
    # strip Python tags
    text = re.sub(r'!!python\S*', '', text)
    cfg = yaml.safe_load(text)

    # 1) gather all joint names
    joint_names: List[str] = cfg['actions']['joint_pos']['joint_names']

    # 2) build default_pos[joint_name] from the init_state patterns
    default_pos: Dict[str, float] = {}
    init_jpos = cfg['scene']['robot']['init_state']['joint_pos']
    for pattern, default in init_jpos.items():
        regex = re.compile(f"^{pattern}$")
        for jn in joint_names:
            if regex.match(jn):
                default_pos[jn] = float(default)

    # 3) walk through constraints
    raw_constraints = cfg.get('constraints', {})
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for term, term_cfg in raw_constraints.items():
        if not isinstance(term_cfg, dict):
            continue
        func = term_cfg.get('func', '')
        params = term_cfg.get('params', {})
        if 'limit' not in params:
            continue

        limit = float(params['limit'])
        # patterns that name joints in this constraint
        patterns = params.get('names', [])

        # helper: expand any list of patterns into the actual joint names
        def expand_patterns(pats):
            out = set()
            for pat in pats:
                rx = re.compile(f"^{pat}$")
                for jn in joint_names:
                    if rx.match(jn):
                        out.add(jn)
            return sorted(out)

        # 3a) joint_position -> only upper bound
        if func.endswith('joint_position_absolute_upper_bound'):
            joints = expand_patterns(patterns)
            for jn in joints:
                bounds[jn] = (None, limit)

        # 3b) joint_position_when_moving_forward -> relative bound about default
        elif func.endswith('relative_joint_position_upper_and_lower_bound_when_moving_forward'):
            joints = expand_patterns(patterns)
            for jn in joints:
                base = default_pos.get(jn, 0.0)
                bounds[jn] = (base - limit, base + limit)

        # 3c) foot_contact_force -> only positive
        elif term == 'foot_contact_force':
            bounds[term] = (0.0, limit)

        # 3d) everything else -> symmetric +/- limit
        else:
            bounds[term] = (-limit, limit)

    return bounds


def resolve_constraint_bounds_for_eval(
    params_directory: str,
    env_cfg,
    task_name: str,
) -> tuple[Dict[str, Tuple[Optional[float], Optional[float]]], str]:
    """
    Resolve the constraint bounds used by metrics_utils.compute_summary_metrics.

    Priority:
    1. LoComposition envs with operational limits encoded through CaT:
       load the saved training/eval bounds from params/env.yaml.
    2. Repository matched baselines without custom constraints:
       use embodiment-specific CaT thresholds for reporting only.
    3. Upstream rough Unitree Go2 envs without custom constraints:
       use hardcoded eval-only bounds so their metrics_summary.json contains
       comparable constraint_violations_percent entries.
    4. Everything else:
       return an empty bounds dict.

    This function deliberately does not modify metrics_utils.py. It only supplies
    the same kind of constraint_bounds dictionary that metrics_utils already expects.
    """
    constraint_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    constraint_bounds_source = "empty"

    if hasattr(env_cfg, "constraints") and env_cfg.constraints is not None:
        try:
            constraint_bounds = load_constraint_bounds(params_directory)
            constraint_bounds_source = "params/env.yaml"

            if hasattr(env_cfg.constraints, "foot_contact_force") and "foot_contact_force" in constraint_bounds:
                env_cfg.constraints.foot_contact_force.params["limit"] = constraint_bounds["foot_contact_force"][1]

            if hasattr(env_cfg.constraints, "front_hfe_position"):
                # This hard term accepts one scalar even though the saved metrics
                # bounds are expanded per matching joint. Use the least restrictive
                # matching upper bound so eval never invents a tighter termination.
                position_term = env_cfg.constraints.front_hfe_position
                position_patterns = position_term.params.get("names", [])
                matching_upper_bounds = [
                    upper_bound
                    for joint_name, (_, upper_bound) in constraint_bounds.items()
                    if upper_bound is not None
                    and any(re.fullmatch(pattern, joint_name) for pattern in position_patterns)
                ]
                if matching_upper_bounds:
                    position_term.params["limit"] = max(matching_upper_bounds)

            if constraint_bounds:
                print("[INFO] Loaded eval constraint bounds from params/env.yaml:")
                print(format_constraint_bounds_for_logging(constraint_bounds))
            else:
                print("[WARN] params/env.yaml was parsed, but no usable constraint bounds were found.")

        except Exception as exception:
            print(f"[WARN] Could not load/apply constraint bounds from params/env.yaml. Reason: {exception}")
            constraint_bounds = {}
            constraint_bounds_source = "failed_params/env.yaml"
    else:
        print("[INFO] env_cfg has no custom constraints block. Constraint-bound loading skipped.")

    if not constraint_bounds and is_matched_baseline_task(task_name):
        profile = resolve_robot_eval_profile(task_name)
        constraint_bounds = get_matched_baseline_constraint_bounds(profile.name)
        constraint_bounds_source = f"hardcoded_matched_baseline_{profile.name}_eval_thresholds"
        print(
            "[INFO] Using LoComposition eval-only operational-limit bounds for "
            f"matched {profile.name} baseline reporting:"
        )
        print(format_constraint_bounds_for_logging(constraint_bounds))
    elif not constraint_bounds and is_upstream_go2_rough_task(task_name):
        constraint_bounds = get_hardcoded_upstream_go2_constraint_bounds()
        constraint_bounds_source = "hardcoded_upstream_go2_rough_eval_thresholds"
        print("[INFO] Using hardcoded eval constraint bounds for upstream rough Unitree Go2 task:")
        print(format_constraint_bounds_for_logging(constraint_bounds))
    elif not constraint_bounds:
        constraint_bounds_source = "empty"
        print("[WARNING] No constraint bounds will be used for constraint_violations_percent.")

    return constraint_bounds, constraint_bounds_source


def build_generate_plots_command(
    generate_plots_script_path: str,
    sim_data_file_path: str,
    output_dir: str,
    job_params: Dict[str, Any],
    foot_vel_height_threshold: float,
) -> List[str]:
    """Build a plot command using the interpreter that is running eval.py."""
    command = [
        sys.executable,
        generate_plots_script_path,
        "--data_file",
        sim_data_file_path,
        "--output_dir",
        output_dir,
        "--start_step",
        str(job_params["start_step"]),
        "--end_step",
        str(job_params["end_step"]),
        "--foot_vel_height_threshold",
        str(foot_vel_height_threshold),
    ]
    if job_params.get("rebuttal_plots", False):
        command.append("--rebuttal_plots")
    return command


def run_generate_plots_parallel(
    plot_jobs: List[Dict[str, Any]],
    plots_directory: str,
    sim_data_file_path: str,
    foot_vel_height_threshold: float,
    num_parallel: int,
    stagger_delay: int,
) -> Dict[str, int]:
    """
    Launch generate_plots.py for all jobs in plot_jobs.
    It runs them in batches of `num_parallel`, with a `stagger_delay` between each launch.
    It then waits for all jobs to complete, providing status updates.
    NOTE: MEMORY LIMIT IS DEPRECATED AND DIRECTLY CONTROLLED IN JUSTFILE USING systemd-run
    """
    generate_plots_script_path = os.path.join(eval_script_path, "generate_plots.py")
    running_procs = []  # List of (proc, subdir, log_file_handle)
    return_codes = {}  # subdir -> rc
    job_queue = plot_jobs[:]

    while job_queue or running_procs:
        # Start new jobs if there's capacity
        while job_queue and len(running_procs) < num_parallel:
            job_params = job_queue.pop(0)
            subdir = job_params["subdir"]
            output_dir = os.path.join(plots_directory, subdir)
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(output_dir, f"generate_plots_{subdir}.log")
            cmd = build_generate_plots_command(
                generate_plots_script_path=generate_plots_script_path,
                sim_data_file_path=sim_data_file_path,
                output_dir=output_dir,
                job_params=job_params,
                foot_vel_height_threshold=foot_vel_height_threshold,
            )
            print(f"[INFO] Spawning plot generation for '{subdir}' (log -> {log_path}), command={' '.join(cmd)}")
            log_file = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            running_procs.append((proc, subdir, log_file))

            if job_queue:
                print(f"[INFO] Waiting {stagger_delay} seconds before starting next job...")
                time.sleep(stagger_delay)

        # Check for completed processes
        for i in range(len(running_procs) - 1, -1, -1):
            proc, subdir, log_file = running_procs[i]
            rc = proc.poll()
            if rc is not None:
                print(f"[INFO] Plot generation for '{subdir}' finished with exit code {rc}.")
                log_file.close()
                return_codes[subdir] = rc
                running_procs.pop(i)

        if running_procs:
            time.sleep(1)  # Poll every second if there are still running processes

    print("[INFO] All plot generation jobs finished.")

    if any(rc != 0 for rc in return_codes.values()):
        print("[WARN] At least one generate_plots.py run returned a non-zero exit code.")
    return return_codes


def ensure_tex_env():
    script_path = os.path.join(os.path.dirname(__file__), "check_and_install_tinytex_for_plots.sh")
    try:
        # Runs with bash — will print status to stdout/stderr
        subprocess.run(["bash", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: TeX environment setup failed (exit code {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)


def scene_entity_exists(env, entity_name: str) -> bool:
    try:
        env.scene[entity_name]
        return True
    except Exception:
        return False


def add_eval_foot_sensors_to_env_cfg(
    env_cfg,
    foot_links: tuple[str, str, str, str] | list[str],
    root_link: str,
    sim_dt: float,
):
    from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
    from isaaclab.sensors.frame_transformer import FrameTransformerCfg

    for link_name in foot_links:
        sensor_name = f"ray_caster_{link_name}"
        if not hasattr(env_cfg.scene, sensor_name):
            setattr(
                env_cfg.scene,
                sensor_name,
                RayCasterCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
                    update_period=sim_dt,
                    offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1)),
                    mesh_prim_paths=["/World/ground"],
                    ray_alignment="yaw",
                    pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
                    debug_vis=True,
                ),
            )

    if not hasattr(env_cfg.scene, "foot_frame_transformer"):
        env_cfg.scene.foot_frame_transformer = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{root_link}",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}")
                for link_name in foot_links
            ],
            debug_vis=False
        )


def get_height_map_sequence(env, asset_cfg):
    robot = env.scene["robot"]
    base_pose_world_frame = robot.data.root_pos_w.clone()

    sensor_name_candidates = []
    if scene_entity_exists(env, asset_cfg.name):
        sensor_name_candidates.append(asset_cfg.name)
    if scene_entity_exists(env, "height_scanner"):
        sensor_name_candidates.append("height_scanner")

    if not sensor_name_candidates:
        return torch.empty((env.num_envs, 0), device=base_pose_world_frame.device)

    sensor = env.scene[sensor_name_candidates[0]]
    ray_hit_positions_world_frame = sensor.data.ray_hits_w.clone()

    base_expanded_to_match_shape_world_frame = base_pose_world_frame.view(-1, 1, 3).expand_as(ray_hit_positions_world_frame)
    non_finite_mask = ~torch.isfinite(ray_hit_positions_world_frame)
    if non_finite_mask.any():
        hits_clean = ray_hit_positions_world_frame.clone()
        hits_clean[non_finite_mask] = base_expanded_to_match_shape_world_frame[non_finite_mask]
    else:
        hits_clean = ray_hit_positions_world_frame

    local = hits_clean - base_expanded_to_match_shape_world_frame
    return local[..., 2]


def maybe_unscale_cat_reward(reward: torch.Tensor, terminated: torch.Tensor, policy_backend: str) -> torch.Tensor:
    if policy_backend != "clean_rl":
        return reward

    if not torch.is_tensor(terminated):
        return reward

    if not torch.is_floating_point(terminated):
        return reward

    denominator = 1.0 - terminated
    if torch.all(denominator <= 0.0):
        return reward

    safe_denominator = torch.clamp(denominator, min=1e-6)
    return reward / safe_denominator


def compute_common_tracking_rewards(
    commanded_velocity: np.ndarray,
    base_linear_velocity_body: np.ndarray,
    base_angular_velocity_body: np.ndarray,
) -> tuple[float, float, float]:
    std_squared = 0.25

    lin_error_squared = float(np.sum((commanded_velocity[:2] - base_linear_velocity_body[:2]) ** 2))
    yaw_error_squared = float((commanded_velocity[2] - base_angular_velocity_body[2]) ** 2)

    track_lin_vel_xy_exp_common = float(np.exp(-lin_error_squared / std_squared))
    track_ang_vel_z_exp_common = float(0.5 * np.exp(-yaw_error_squared / std_squared))
    track_total_common = track_lin_vel_xy_exp_common + track_ang_vel_z_exp_common

    return track_lin_vel_xy_exp_common, track_ang_vel_z_exp_common, track_total_common


def add_eval_only_metrics_to_summary(
    metrics: dict[str, Any],
    mask: np.ndarray,
    eval_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    enriched_metrics = dict(metrics)

    def masked_values(name: str) -> np.ndarray:
        values = np.asarray(eval_arrays[name])
        return values[mask]

    for metric_name in (
        "track_lin_vel_xy_exp_common_weight",
        "track_ang_vel_z_exp_common_weight",
        "track_vel_exp_total_common_weight",
    ):
        values = masked_values(metric_name)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            enriched_metrics[f"mean_{metric_name}"] = None
            enriched_metrics[f"cumulative_{metric_name}"] = None
        else:
            enriched_metrics[f"mean_{metric_name}"] = float(finite_values.mean())
            enriched_metrics[f"cumulative_{metric_name}"] = float(finite_values.sum())

    terrain_levels = masked_values("terrain_level")
    finite_terrain_levels = terrain_levels[np.isfinite(terrain_levels)]
    if finite_terrain_levels.size == 0:
        enriched_metrics["terrain_level_summary"] = None
        enriched_metrics["mean_terrain_level"] = None
        enriched_metrics["final_terrain_level"] = None
    else:
        enriched_metrics["terrain_level_summary"] = summarize_metric(finite_terrain_levels.tolist())
        enriched_metrics["mean_terrain_level"] = float(finite_terrain_levels.mean())
        enriched_metrics["final_terrain_level"] = float(finite_terrain_levels[-1])

    return enriched_metrics


def apply_common_eval_reward_scale_if_needed(env_cfg, task_name: str, enabled: bool) -> bool:
    if not enabled:
        return False

    # Matched baselines must retain the exact upstream training reward stack.
    # Their comparable tracking metrics are computed separately from raw state.
    if is_matched_baseline_task(task_name):
        return False

    task_name_lower = task_name.lower()
    if "isaac-velocity-rough-unitree-go2" not in task_name_lower:
        return False

    if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "track_lin_vel_xy_exp"):
        env_cfg.rewards.track_lin_vel_xy_exp.weight = 1.0

    if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "track_ang_vel_z_exp"):
        env_cfg.rewards.track_ang_vel_z_exp.weight = 0.5

    print("[INFO] Applied common eval reward scale for upstream Go2: track_lin_vel_xy_exp=1.0, track_ang_vel_z_exp=0.5")
    return True


def get_current_terrain_level(env) -> float:
    try:
        return float(env.scene.terrain.terrain_levels[0].detach().cpu().item())
    except Exception:
        return float("nan")


def maximum_numeric_value(nested_value: Any) -> Optional[float]:
    """Return the largest finite numeric leaf in nested dictionaries/lists."""
    values: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric):
                values.append(numeric)

    visit(nested_value)
    return max(values) if values else None


def main():
    args = parse_arguments()
    args.run_dir = os.path.abspath(args.run_dir)
    robot_profile = resolve_robot_eval_profile(args.task)

    seed = args.seed
    set_global_seed(seed)
    print(f"[INFO] Using eval seed={seed}")

    if args.eval_checkpoint is None:
        checkpoint_path = get_latest_checkpoint(args.run_dir)
    else:
        checkpoint_path = os.path.join(args.run_dir, f"model_{args.eval_checkpoint}.pt")
        if not os.path.isfile(checkpoint_path):
            print(f"ERROR: checkpoint file does not exist, exiting: {checkpoint_path}")
            exit(1)

    print(f"[INFO] Loading model from: {checkpoint_path}")
    checkpoint_object = load_checkpoint_for_format_detection(checkpoint_path)
    detected_policy_backend = detect_policy_backend_from_checkpoint(checkpoint_object)
    policy_backend = detected_policy_backend if args.policy_backend == "auto" else args.policy_backend
    print(f"[INFO] Detected policy backend={detected_policy_backend}, selected policy backend={policy_backend}")

    model_state = None
    if policy_backend == "clean_rl":
        model_state = extract_cleanrl_state_dict(checkpoint_object)
        observation_dim = infer_checkpoint_input_dimensions(model_state)
        if robot_profile.name == "go2":
            if observation_dim == 236:
                args.task = "LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0"
            elif observation_dim == 558:
                args.task = "LoComposition-Go2-Rough-Terrain-Full-State-History-Play-v0"
        print(f"Observation dimension={observation_dim}, selected task={args.task}")
    else:
        print(f"[INFO] RSL-RL checkpoint selected, using task={args.task}")

    robot_profile = resolve_robot_eval_profile(args.task)
    print(f"[INFO] Resolved evaluation robot profile={robot_profile.name}")

    if not "play" in args.task.lower():
        input("\n\n-------------------------------------------------------------------\nKeyword 'Play' not found in task name, are you sure you are using the correct task/environment?\n-------------------------------------------------------------------\n\n")

    # Launch Isaac Lab environment
    args.device = "cuda"  # Using CPU Increases iterations/sec in some cases and reduces VRAM usage which allows parallel runs. Replace with "cuda" if you want pure GPU, because on more recent GPUs this might be faster depending on your hardware
    args.disable_fabric = True if args.device == "cpu" else False
    if "--/rtx/verifyDriverVersion/enabled=false" not in sys.argv:  # Needed because of overflow bug when checking nvidia driver version with minor version > 255
        sys.argv.append("--/rtx/verifyDriverVersion/enabled=false")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    ffmpeg_process = None
    cleanup_complete = False

    def cleanup_runtime_resources():
        """Release evaluator subprocess, environment, and Kit even after exceptions."""
        nonlocal cleanup_complete
        if cleanup_complete:
            return
        cleanup_complete = True

        if ffmpeg_process is not None and ffmpeg_process.poll() is None:
            try:
                if ffmpeg_process.stdin is not None and not ffmpeg_process.stdin.closed:
                    ffmpeg_process.stdin.close()
                ffmpeg_process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                ffmpeg_process.terminate()
                try:
                    ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ffmpeg_process.kill()
                    ffmpeg_process.wait()

        if env is not None:
            try:
                env.close()
            except Exception as exception:
                print(f"[WARN] Failed to close evaluation environment: {exception}")

        try:
            simulation_app.close()
        except Exception as exception:
            print(f"[WARN] Failed to close Isaac Sim application: {exception}")

    atexit.register(cleanup_runtime_resources)
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
    from isaaclab.envs.mdp.observations import root_quat_w
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    # Register LoComposition Gymnasium environments before parse_env_cfg() calls gym.spec(args.task).
    import locomposition.tasks.locomotion.velocity.config.solo12  # noqa: F401

    print(f"ISAACLAB_NUCLEUS_DIR={ISAACLAB_NUCLEUS_DIR}")

    if args.task not in gym.envs.registry:
        matching_registered_tasks = sorted(
            task_id
            for task_id in gym.envs.registry.keys()
            if any(
                name in task_id.lower()
                for name in (
                    "locomposition",
                    "cat-go2",
                    "unitree-go2",
                    "anymal",
                    "spot",
                )
            )
        )
        raise RuntimeError(
            f"Task '{args.task}' is not registered after importing the LoComposition task package. "
            f"Matching registered tasks: {matching_registered_tasks}"
        )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    if args.use_training_go2_config:
        if not is_locomposition_go2_task(args.task):
            raise ValueError("--use_training_go2_config requires a LoComposition Go2 task.")
        from locomposition.assets.go2_config import UNITREE_GO2_CFG_TRAIN

        env_cfg.scene.robot = UNITREE_GO2_CFG_TRAIN.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print("[INFO] Replaced the PLAY robot config with UNITREE_GO2_CFG_TRAIN for this evaluation.")
    eval_reward_scale_override_applied = apply_common_eval_reward_scale_if_needed(
        env_cfg=env_cfg,
        task_name=args.task,
        enabled=args.downscale_upstream_go2_tracking_rewards,
    )
    action_scale, eval_action_scale_before_override = apply_training_action_scale(
        env_cfg=env_cfg,
        params_directory=os.path.join(args.run_dir, "params"),
    )

    # Inject Latency
    if hasattr(env_cfg.observations, "policy"):
        p = env_cfg.observations.policy

        # Helper to safely set latency if the term exists AND accepts the argument
        def set_latency(term_name, val):
            if hasattr(p, term_name):
                term = getattr(p, term_name)
                try:
                    sig = inspect.signature(term.func)
                except ValueError:
                    print(f"[WARN] Could not inspect signature for {term_name}, skipping latency.")
                    return

                if "latency" in sig.parameters:
                    if term.params is None:
                        term.params = {}
                    term.params["latency"] = val
                    print(f"[INFO] Set latency for {term_name} to {val}")
                else:
                    print(f"[INFO] Skipping latency for {term_name} (Function '{term.func.__name__}' does not accept it)")

        set_latency("joint_pos_history", args.delay_joints)
        set_latency("joint_vel_history", args.delay_joints)
        set_latency("base_ang_vel", args.delay_imu)
        set_latency("projected_gravity", args.delay_imu)
        set_latency("actions", args.delay_action_history)
        set_latency("height_map", args.delay_height_map)

    apply_seed_to_env_cfg(env_cfg, seed)
    set_global_seed(seed)

    print(f"seed={seed}")
    print(f"env_cfg.seed={getattr(env_cfg, 'seed', None)}")
    print(f"env_cfg.sim.random_seed={getattr(env_cfg.sim, 'random_seed', None)}")

    terrain_cfg = getattr(env_cfg.scene, "terrain", None)
    terrain_generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_generator_cfg is not None:
        print(f"terrain_generator.seed={getattr(terrain_generator_cfg, 'seed', None)}")

    # Viewer setup
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (0.0, -3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.sim.render.rendering_mode = "quality"
    step_dt = env_cfg.sim.dt * env_cfg.decimation  # Physics run at higher frequency, action is applied `decimation` physics-steps, but video uses env steps as unit
    frame_width = 1920
    frame_height = 1080
    frame_rate = int(round(1.0 / step_dt))
    env_cfg.viewer.resolution = (frame_width, frame_height)
    device = torch.device(args.device)

    foot_links = list(robot_profile.foot_links)
    foot_labels = ['front left', 'front right', 'rear left', 'rear right']
    add_eval_foot_sensors_to_env_cfg(
        env_cfg,
        foot_links=robot_profile.foot_links,
        root_link=robot_profile.root_link,
        sim_dt=env_cfg.sim.dt,
    )

    run_path = Path(args.run_dir).resolve()
    run_name = run_path.name
    env_name = run_path.parent.name
    task_name = args.task

    eval_condition_suffix = f"action_delay_{args.delay_actions}"
    if args.fixed_scenario is not None:
        eval_condition_suffix += f"_scenario_{args.fixed_scenario}"
    elif args.rebuttal_scenarios_only:
        eval_condition_suffix += "_rebuttal_scenarios"
    eval_base_dir = os.path.join(
        str(run_path),
        f"eval_checkpoint_{os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]}_seed_{seed}_{eval_condition_suffix}"
    )
    print(f"eval_base_dir={eval_base_dir}, env_name={env_name}, run_name={run_name}, task_name={task_name}")

    # Create output directories
    plots_directory = os.path.join(eval_base_dir, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    # The 500-step default exactly fills one 10 s episode. The CLI override is
    # diagnostic-only and lets checkpoint/eval integration be tested quickly.
    fixed_command_sim_steps = args.fixed_command_sim_steps
    fixed_command_scenarios = [  # Scenario positions depend on seed!
        ("stand_still", torch.tensor([0.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_stairs_up", torch.tensor([1, 0.0, 0.0], device=device), (torch.tensor([-8, 16, -0.1], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("pure_spin", torch.tensor([0.0, 0.0, 0.5], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("walk_x_flat_terrain_0.4mps", torch.tensor([0.4, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("walk_x_flat_terrain_1.0mps", torch.tensor([1.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("walk_x_flat_terrain_1.6mps", torch.tensor([1.6, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_x_uneven_terrain", torch.tensor([0.5, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_x_uneven_terrain", torch.tensor([1.0, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_diagonal_uneven_terrain", torch.tensor([1.0, 1.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_diagonal_turning_uneven_terrain", torch.tensor([0.8, 0.5, 0.0], device=device), (torch.tensor([0.0, 3.0, 0.4], device=device), torch.tensor([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)], dtype=torch.float32, device=device))),
        ("medium_walk_diagonal_random_steps", torch.tensor([0.7, 0.5, 0.0], device=device), (torch.tensor([2.0, 5.0, 0.4], device=device), torch.tensor([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)], dtype=torch.float32, device=device))),
    ]

    if not args.skip_cot_sweep:
        fixed_command_scenarios.extend([
            ("cot_sweep_walk_x_flat_terrain_0.2", torch.tensor([0.2, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_0.4", torch.tensor([0.4, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_0.6", torch.tensor([0.6, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_0.8", torch.tensor([0.8, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_1.0", torch.tensor([1.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_1.2", torch.tensor([1.2, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_1.4", torch.tensor([1.4, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_1.6", torch.tensor([1.6, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_1.8", torch.tensor([1.8, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
            ("cot_sweep_walk_x_flat_terrain_2.0", torch.tensor([2.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ])
    else:
        print("Skipping Cost of Transport sweep scenarios")

    fixed_command_scenarios = select_fixed_command_scenarios(
        fixed_command_scenarios,
        args.fixed_scenario,
        REBUTTAL_DYNAMICS_SCENARIOS if args.rebuttal_scenarios_only else None,
    )
    fixed_command_scenarios = adjust_fixed_command_spawn_heights(
        fixed_command_scenarios,
        robot_profile,
    )

    # See main training loop for detailed explanation, but in summary, hard constraints terminate the environment or at least return terminated = 1
    # which results in zero reward and can't be recovered by rescaling. Thus, we update the constraint limits based on the loaded values from
    # the training environment. Only contact force and thigh position limit are updated because other values are handled more robustly by the rescaling in the loop.
    # This allows more constraints to be added or removed without leading to issues in this eval script.
    #
    # For upstream Isaac Lab rough Unitree Go2 baselines, there is usually no custom constraints block in the env config.
    # In that case, we still populate constraint_bounds with fixed eval thresholds so metrics_utils.compute_summary_metrics()
    # computes comparable constraint_violations_percent entries without changing metrics_utils.py.
    constraint_bounds, constraint_bounds_source = resolve_constraint_bounds_for_eval(
        params_directory=os.path.join(args.run_dir, "params"),
        env_cfg=env_cfg,
        task_name=args.task,
    )

    total_sim_steps = args.random_sim_step_length + len(fixed_command_scenarios) * fixed_command_sim_steps
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

    print(f"[INFO] Runtime env actions.joint_pos.scale = {env.unwrapped.cfg.actions.joint_pos.scale}")
    print(f"[SEED CHECK] runtime env seed={getattr(env.unwrapped.cfg, 'seed', None)}")
    print(f"[SEED CHECK] runtime sim seed={getattr(env.unwrapped.cfg.sim, 'random_seed', None)}")
    runtime_terrain_generator = getattr(env.unwrapped.cfg.scene.terrain, "terrain_generator", None)
    if runtime_terrain_generator is not None:
        print(f"[SEED CHECK] runtime terrain seed={getattr(runtime_terrain_generator, 'seed', None)}")

    def get_render_frames(env):
        if args.enable_cameras is not None and args.enable_cameras == False:
            return []

        raw_render_output = env.render()

        if raw_render_output is None:
            return []

        if isinstance(raw_render_output, list):
            return raw_render_output

        if isinstance(raw_render_output, tuple):
            return list(raw_render_output)

        if isinstance(raw_render_output, np.ndarray):
            return [raw_render_output]

        raise TypeError(f"Unexpected render output type: {type(raw_render_output)}")

    actor_with_rms = None
    rsl_rl_policy = None
    rsl_rl_policy_reset = None
    rsl_rl_clip_actions = None
    rsl_rl_env_for_runner = None

    if policy_backend == "clean_rl":
        from locomposition.tasks.utils.cleanrl.ppo import Agent
        from locomposition.tasks.utils.cleanrl.ppo import ActorWithRMS

        policy_agent = Agent(env).to(device)
        policy_agent.load_state_dict(model_state)
        actor_with_rms = ActorWithRMS(policy_agent)

    elif policy_backend == "rsl_rl":
        agent_cfg = load_cfg_from_registry(args.task, args.agent_entry_point)
        agent_cfg.device = args.device
        if hasattr(agent_cfg, "seed"):
            agent_cfg.seed = seed

        print(f"[INFO] Loading RSL-RL checkpoint from: {checkpoint_path}")
        if "model_state_dict" in checkpoint_object:
            if getattr(agent_cfg.policy, "actor_obs_normalization", False):
                raise ValueError(
                    "Legacy RSL-RL actor observation normalization is not supported because "
                    "the legacy checkpoint does not store compatible normalization state."
                )
            rsl_rl_policy = build_legacy_rsl_rl_policy(
                checkpoint_object,
                activation_name=agent_cfg.policy.activation,
                expected_observation_dim=env.observation_space["policy"].shape[-1],
                expected_action_dim=env.action_space.shape[-1],
                device=env.unwrapped.device,
            )
            print("[INFO] Reconstructed deterministic actor from legacy RSL-RL model_state_dict.")
        else:
            rsl_rl_env_for_runner = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner_cfg_dict = build_rsl_rl_runner_cfg_dict(agent_cfg)

            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(rsl_rl_env_for_runner, runner_cfg_dict, log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(rsl_rl_env_for_runner, runner_cfg_dict, log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name}")

            runner.load(checkpoint_path)
            rsl_rl_policy = runner.get_inference_policy(device=env.unwrapped.device)
            rsl_rl_policy_reset = getattr(rsl_rl_policy, "reset", None)

        rsl_rl_clip_actions = getattr(agent_cfg, "clip_actions", None)

    else:
        raise ValueError(f"Unsupported policy_backend={policy_backend}")

    robot = env.unwrapped.scene["robot"]
    base_body_index = resolve_root_body_index(robot.data.body_names, robot_profile)
    base_body_name = robot.data.body_names[base_body_index]
    print(f"[INFO] Recording base COM acceleration from body index {base_body_index}: {base_body_name}")
    vel_term = env.unwrapped.command_manager.get_term("base_velocity")

    def set_fixed_velocity_command(vec: torch.Tensor):
        # VERY dirty hack but this is just an eval script so it's ok. Basically don't allow the sampler to overwrite fixed commands
        vel_term.cfg.resampling_time_range = (1000000, 1000000)
        vel_term.cfg.heading_command = False
        vel_term._update_command = lambda *_, **__: None
        vel_term.vel_command_b = vec.repeat(env.unwrapped.num_envs, 1)

    def teleport_robot(pos_xyz, quat_xyzw):
        pose = torch.cat([pos_xyz, quat_xyzw]).unsqueeze(0)
        robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=device))
        env.unwrapped.scene.write_data_to_sim()

    joint_names = env.unwrapped.scene["robot"].data.joint_names

    # Initialize buffers for recording
    joint_positions_buffer = []
    joint_velocities_buffer = []
    contact_forces_buffer = []
    foot_contact_force_world_history_buffer = []
    joint_torques_buffer = []
    joint_accelerations_buffer = []
    action_rate_buffer = []
    base_position_buffer = []
    base_orientation_buffer = []
    base_linear_velocity_buffer = []
    base_angular_velocity_buffer = []
    base_linear_velocity_body_buffer = []
    base_angular_velocity_body_buffer = []
    base_linear_acceleration_world_buffer = []
    base_angular_acceleration_world_buffer = []
    base_linear_acceleration_body_buffer = []
    base_angular_acceleration_body_buffer = []
    commanded_velocity_buffer = []
    contact_state_buffer = []
    height_map_buffer = []
    foot_positions_world_frame_buffer = []
    foot_velocities_world_frame_buffer = []
    foot_velocities_body_frame_buffer = []
    foot_positions_body_frame_buffer = []
    foot_positions_contact_frame_buffer = []  # Height above terrain, also called sole frame
    distance_increment_buffer = []
    terrain_level_buffer = []
    track_lin_vel_xy_exp_common_weight_buffer = []
    track_ang_vel_z_exp_common_weight_buffer = []
    track_vel_exp_total_common_weight_buffer = []

    reward_buffer = []
    manual_reset_steps = []
    automatic_reset_steps = []
    inference_durations = []

    observations, info = env.reset(seed=seed)
    policy_observation = observations['policy']
    queued_action = None
    previous_applied_action = None

    video_output_path = os.path.join(eval_base_dir, f"{os.path.basename(eval_base_dir)}_run_{os.path.basename(args.run_dir)}.mp4")
    frame_storage_interval = 1
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frame_width}x{frame_height}", "-framerate", str(frame_rate),
        "-i", "pipe:0",
        "-c:v", "libx265", "-pix_fmt", "yuv420p", "-preset", "slow",
        "-movflags", "+use_metadata_tags", "-metadata", f"env_name={env_name}",
        video_output_path
    ]
    ffmpeg_process_log_path = os.path.join(eval_base_dir, "ffmpeg_encode.log")
    with open(ffmpeg_process_log_path, "w") as ffmpeg_process_logfile:
        print(f"Starting ffmpeg process={' '.join(ffmpeg_cmd)}")
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=ffmpeg_process_logfile, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, bufsize=4 * 1024 * 1024)

    CPP_INFERENCE = False  # Allows testing C++ inference in devcontainer (test_pytorch_policy.cpp) in simulation
    if CPP_INFERENCE:
        print("NOTE: C++ inference selected, connecting to 127.0.0.1:5555 via zmq...")
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.connect("tcp://127.0.0.1:5555")

    for t in tqdm(range(total_sim_steps)):
        # These should only ever run at the end of eval / after random sampling because set_fixed_velocity_command breaks random command sampling!
        if t >= args.random_sim_step_length and (t - args.random_sim_step_length) % fixed_command_sim_steps == 0:
            scenario = fixed_command_scenarios[int(t - args.random_sim_step_length) // fixed_command_sim_steps]
            fixed_command = scenario[1]
            spawn_point_pos, spawn_point_quat = scenario[2]
            print(f"Resetting env and setting fixed command + spawn point for scenario={scenario}...")
            obs, _ = env.reset()
            manual_reset_steps.append(t)
            teleport_robot(spawn_point_pos, spawn_point_quat)
            set_fixed_velocity_command(fixed_command)
            policy_observation = obs["policy"]
            queued_action = None
            previous_applied_action = None

        # Distance must be estimated from the pre-step state because Isaac Lab resets terminated envs
        # inside env.step(). That means post-step state buffers are contaminated on reset steps.
        scene_robot_data_pre_step = env.unwrapped.scene["robot"].data
        pre_step_root_linear_velocity_w = getattr(
            scene_robot_data_pre_step,
            "root_link_lin_vel_w",
            scene_robot_data_pre_step.root_lin_vel_w,
        )[0]
        pre_step_horizontal_speed = torch.linalg.norm(pre_step_root_linear_velocity_w[:2]).item()
        distance_increment_buffer.append(float(pre_step_horizontal_speed * step_dt))

        with torch.no_grad():
            inference_start = time.perf_counter_ns()
            # action, _, _, _, _ = policy_agent.get_action_and_value(policy_agent.obs_rms(policy_observation, update=False), use_deterministic_policy=True)
            if CPP_INFERENCE:
                socket.send(policy_observation.cpu().numpy().tobytes())
                action_np = np.frombuffer(socket.recv(), dtype=np.float32).reshape(1, 12).copy()
                action = torch.from_numpy(action_np)
            elif policy_backend == "clean_rl":
                action = actor_with_rms(policy_observation)
            elif policy_backend == "rsl_rl":
                action = rsl_rl_policy({"policy": policy_observation})
                if rsl_rl_clip_actions is not None:
                    action = torch.clamp(action, -rsl_rl_clip_actions, rsl_rl_clip_actions)
            else:
                raise ValueError(f"Unsupported policy_backend={policy_backend}")

            inference_end = time.perf_counter_ns()
            inference_duration_us = (inference_end - inference_start) / 1e+3
            inference_durations.append(inference_duration_us)
            # print(f"Inference took {inference_duration_us:.4f}us")

        applied_action, queued_action = apply_policy_action_delay(
            action, queued_action, args.delay_actions
        )
        step_tuple = env.step(applied_action)
        # print(step_tuple)
        next_observation, reward, terminated, truncated, info = step_tuple

        if policy_backend == "rsl_rl" and rsl_rl_policy_reset is not None:
            dones = torch.logical_or(terminated.bool(), truncated.bool())
            rsl_rl_policy_reset(dones)

        # Idea of constraints as terminations is to scale reward by max constraint violation, but since policies
        # being tested here might differ in the constraints they were trained with, it makes sense to remove this scaling
        # while evaluating a policy. This division just reverts the scaling done in cat_env.py and thus should yield the
        # "raw" rewards collected in the environment.
        # If this were omitted, a policy trained with e.g. higher joint position constraint limits would result in very low
        # rewards in this eval environment as it always calculates based on the limits specified in the local code, not what
        # the policy was trained with. Additionally, any hard constraints (max_p=1.0) need to be manually increased earlier
        # in the setup because those terminate the environment and thus break the reproducibility and the if check below.
        reward = maybe_unscale_cat_reward(reward, terminated, policy_backend)

        # Because of CaT, terminated is actually a nonzero probability instead of a boolean, so we have to check for resets this way
        reset_this_step = env.unwrapped.episode_length_buf[0].item() == 0
        if reset_this_step:
            automatic_reset_steps.append(t)
            queued_action = None
            # continue  # Skip reset iteration because it's just wrong

        if t % frame_storage_interval == 0:
            raw_frames = get_render_frames(env)
            for frame in raw_frames:
                assert frame.shape[:2] == (frame_height, frame_width), "Returned frame does not match specified dimension."
                ffmpeg_process.stdin.write(frame.tobytes())

        scene_robot_data = env.unwrapped.scene['robot'].data

        joint_positions = scene_robot_data.joint_pos[0].cpu().numpy()
        joint_velocities = scene_robot_data.joint_vel[0].cpu().numpy()
        joint_positions_buffer.append(joint_positions)
        joint_velocities_buffer.append(joint_velocities)

        contact_sensors = env.unwrapped.scene['contact_forces']
        feet_ids, _ = contact_sensors.find_bodies(foot_links, preserve_order=True)
        net_forces = contact_sensors.data.net_forces_w_history
        forces_history = net_forces[0].cpu()[:, feet_ids, :]
        foot_contact_force_world_history_buffer.append(forces_history.numpy().copy())
        force_magnitudes = torch.norm(forces_history, dim=-1)
        max_per_foot = force_magnitudes.max(dim=0)[0].numpy()
        contact_forces_buffer.append(max_per_foot)

        torques = scene_robot_data.applied_torque[0].cpu().numpy()
        joint_torques_buffer.append(torques)

        accelerations = scene_robot_data.joint_acc[0].cpu().numpy()
        joint_accelerations_buffer.append(accelerations)

        applied_action_np = applied_action.detach().cpu().numpy().copy()
        action_rate = compute_action_rate(
            applied_action_np, previous_applied_action, step_dt
        )
        action_rate_buffer.append(action_rate)
        previous_applied_action = None if reset_this_step else applied_action_np

        base_world_position = scene_robot_data.root_link_pos_w[0].cpu().numpy()  # world-frame for env 0
        # origin = env.unwrapped.scene.terrain.env_origins[0].cpu().numpy()  # terrain origin for env 0
        # relative_position = world_position + origin  # position relative to terrain
        # quat_xyzw = scene_data.root_quat_w[0]
        # quat_wxyz = torch.cat([quat_xyzw[3:], quat_xyzw[:3]])  # reorder to (w,x,y,z)
        quat_wxyz = root_quat_w(env.unwrapped, make_quat_unique=True, asset_cfg=SceneEntityCfg("robot"))
        roll_t, pitch_t, yaw_t = euler_xyz_from_quat(quat_wxyz)

        # Isaaclab returns 0 to 2pi euler angles, so small negative angles wrap around to ~2pi. Rescale accordingly
        def convert_to_signed_angle(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        # IMPORTANT: THESE ANGLES ARE NOT UNWRAPPED YET, HAPPENS AFTER THE FULL ROLLOUT
        roll = convert_to_signed_angle(roll_t.cpu().numpy().item())
        pitch = convert_to_signed_angle(pitch_t.cpu().numpy().item())
        yaw = convert_to_signed_angle(yaw_t.cpu().numpy().item())
        # print(f"UNWRAPPED roll={roll}\tUNWRAPPED pitch={pitch}")

        base_position_buffer.append(base_world_position)
        base_orientation_buffer.append([yaw, pitch, roll])

        linear_velocity_w = scene_robot_data.root_lin_vel_w[0]
        angular_velocity_w = scene_robot_data.root_ang_vel_w[0]
        # The returned quat is (w, x, y, z) which is what quat_apply_inverse expects.
        linear_velocity_b = quat_apply_inverse(quat_wxyz, linear_velocity_w.unsqueeze(0)).squeeze()
        angular_velocity_b = quat_apply_inverse(quat_wxyz, angular_velocity_w.unsqueeze(0)).squeeze()
        base_linear_velocity_buffer.append(linear_velocity_w.cpu().numpy())
        base_angular_velocity_buffer.append(angular_velocity_w.cpu().numpy())
        base_linear_velocity_body_buffer.append(linear_velocity_b.cpu().numpy())
        base_angular_velocity_body_buffer.append(angular_velocity_b.cpu().numpy())

        base_com_acceleration_world = scene_robot_data.body_com_acc_w[0, base_body_index]
        linear_acceleration_world = base_com_acceleration_world[:3]
        angular_acceleration_world = base_com_acceleration_world[3:]
        linear_acceleration_body = quat_apply_inverse(
            quat_wxyz, linear_acceleration_world.unsqueeze(0)
        ).squeeze()
        angular_acceleration_body = quat_apply_inverse(
            quat_wxyz, angular_acceleration_world.unsqueeze(0)
        ).squeeze()
        base_linear_acceleration_world_buffer.append(linear_acceleration_world.cpu().numpy().copy())
        base_angular_acceleration_world_buffer.append(angular_acceleration_world.cpu().numpy().copy())
        base_linear_acceleration_body_buffer.append(linear_acceleration_body.cpu().numpy().copy())
        base_angular_acceleration_body_buffer.append(angular_acceleration_body.cpu().numpy().copy())

        current_commanded_velocity = env.unwrapped.command_manager.get_command("base_velocity").clone()
        commanded_velocity_buffer.append(current_commanded_velocity)  # three components: lin_vel_x, lin_vel_y, ang_vel_z

        track_lin_vel_xy_exp_common, track_ang_vel_z_exp_common, track_vel_exp_total_common = compute_common_tracking_rewards(
            commanded_velocity=current_commanded_velocity[0].detach().cpu().numpy(),
            base_linear_velocity_body=linear_velocity_b.detach().cpu().numpy(),
            base_angular_velocity_body=angular_velocity_b.detach().cpu().numpy(),
        )
        track_lin_vel_xy_exp_common_weight_buffer.append(track_lin_vel_xy_exp_common)
        track_ang_vel_z_exp_common_weight_buffer.append(track_ang_vel_z_exp_common)
        track_vel_exp_total_common_weight_buffer.append(track_vel_exp_total_common)

        terrain_level_buffer.append(get_current_terrain_level(env.unwrapped))

        contact_state = compute_contact_state(max_per_foot)
        contact_state_buffer.append(contact_state)

        height_map_sequence = get_height_map_sequence(env.unwrapped, SceneEntityCfg(name="ray_caster")).cpu().numpy()
        height_map_buffer.append(height_map_sequence[0])

        # This will not account for terrain height, i.e. if the robot is standing on a 1m obstacle, the world foot height will be 1m.
        foot_positions_world_t = torch.stack([scene_robot_data.body_link_pos_w[0, scene_robot_data.body_names.index(link)] for link in foot_links])
        foot_positions_world = foot_positions_world_t.cpu().numpy()
        foot_positions_world_frame_buffer.append(foot_positions_world)

        foot_velocities_world_t = torch.stack([scene_robot_data.body_link_vel_w[0, scene_robot_data.body_names.index(link), :3] for link in foot_links])
        quat_expanded = quat_wxyz.expand(4, -1)
        foot_velocities_body_t = quat_apply_inverse(quat_expanded, foot_velocities_world_t)
        foot_velocities_world_frame_buffer.append(foot_velocities_world_t.cpu().numpy())
        foot_velocities_body_frame_buffer.append(foot_velocities_body_t.cpu().numpy())

        if scene_entity_exists(env.unwrapped, "foot_frame_transformer"):
            foot_positions_body = env.unwrapped.scene["foot_frame_transformer"].data.target_pos_source[0].cpu().numpy()
        else:
            base_position_t = scene_robot_data.root_link_pos_w[0]
            foot_positions_body_t = quat_apply_inverse(quat_expanded, foot_positions_world_t - base_position_t)
            foot_positions_body = foot_positions_body_t.cpu().numpy()
        # print(f"foot_positions_body={foot_positions_body}")
        foot_positions_body_frame_buffer.append(foot_positions_body)

        # Calculate foot height above ground using raycaster sensor in each foot (sole/contact frame)
        # Apply a robot-specific sole offset only when it has been audited.
        # Unknown offsets remain zero instead of inheriting Go2 geometry.
        foot_com_toe_tip_offset = robot_profile.sole_offset
        terrain_offset_feet = np.array([
            [
                0,
                0,
                env.unwrapped.scene[f"ray_caster_{link_name}"].data.ray_hits_w[:, 0, 2].cpu().item() + foot_com_toe_tip_offset
            ]
            for link_name in foot_links
        ])
        # print(f"terrain_z_feet={terrain_offset_feet}")
        foot_positions_contact_frame = foot_positions_world - terrain_offset_feet
        # print(f"foot_positions_contact_frame={foot_positions_contact_frame}")
        foot_positions_contact_frame_buffer.append(foot_positions_contact_frame)

        reward_buffer.append(reward.mean().item())
        policy_observation = next_observation['policy']

    print("Sim loop done, converting and saving recorded sim data...")

    manual_reset_steps = sorted(set(int(step) for step in manual_reset_steps))
    automatic_reset_steps = sorted(set(int(step) for step in automatic_reset_steps))
    all_reset_steps = sorted(set(manual_reset_steps + automatic_reset_steps))

    # Convert buffers to numpy arrays
    time_indices = np.arange(total_sim_steps)
    sim_times = time_indices * step_dt
    reset_times = [i * step_dt for i in all_reset_steps]
    manual_reset_times = [i * step_dt for i in manual_reset_steps]
    automatic_reset_times = [i * step_dt for i in automatic_reset_steps]
    print("All reset times: ", reset_times)
    reward_array = np.array(reward_buffer)
    inference_durations_us_array = np.array(inference_durations)
    joint_positions_array = np.vstack(joint_positions_buffer)
    joint_velocities_array = np.vstack(joint_velocities_buffer)
    joint_torques_array = np.vstack(joint_torques_buffer)
    joint_accelerations_array = np.vstack(joint_accelerations_buffer)
    action_rate_array = np.vstack(action_rate_buffer)
    contact_forces_array = np.stack(contact_forces_buffer)
    foot_contact_force_world_history_array = np.stack(foot_contact_force_world_history_buffer)
    base_position_array = np.vstack(base_position_buffer)
    print(np.vstack(base_orientation_buffer).shape)
    base_orientation_array = np.unwrap(np.vstack(base_orientation_buffer), axis=0)
    print(base_orientation_array.shape)
    base_linear_velocity_array = np.vstack(base_linear_velocity_buffer)
    base_angular_velocity_array = np.vstack(base_angular_velocity_buffer)
    base_linear_velocity_body_array = np.vstack(base_linear_velocity_body_buffer)
    base_angular_velocity_body_array = np.vstack(base_angular_velocity_body_buffer)
    base_linear_acceleration_world_array = np.vstack(base_linear_acceleration_world_buffer)
    base_angular_acceleration_world_array = np.vstack(base_angular_acceleration_world_buffer)
    base_linear_acceleration_body_array = np.vstack(base_linear_acceleration_body_buffer)
    base_angular_acceleration_body_array = np.vstack(base_angular_acceleration_body_buffer)
    contact_state_array = np.vstack(contact_state_buffer)
    commanded_velocity_array = np.vstack([cv.cpu().numpy() if isinstance(cv, torch.Tensor) else np.asarray(cv) for cv in commanded_velocity_buffer])
    terrain_level_array = np.asarray(terrain_level_buffer, dtype=np.float64)
    track_lin_vel_xy_exp_common_weight_array = np.asarray(track_lin_vel_xy_exp_common_weight_buffer, dtype=np.float64)
    track_ang_vel_z_exp_common_weight_array = np.asarray(track_ang_vel_z_exp_common_weight_buffer, dtype=np.float64)
    track_vel_exp_total_common_weight_array = np.asarray(track_vel_exp_total_common_weight_buffer, dtype=np.float64)
    foot_velocities_world_frame_array = np.array(foot_velocities_world_frame_buffer)
    foot_velocities_body_frame_array = np.array(foot_velocities_body_frame_buffer)
    foot_positions_world_frame_array = np.array(foot_positions_world_frame_buffer)
    foot_positions_body_frame_array = np.array(foot_positions_body_frame_buffer)
    foot_positions_contact_frame_array = np.array(foot_positions_contact_frame_buffer)
    raw_power_array = joint_torques_array * joint_velocities_array
    total_robot_mass = float(env.unwrapped.scene["robot"].data.default_mass.sum().item())

    repaired_power_array = raw_power_array.copy()
    for reset_step in automatic_reset_steps:
        if 0 <= reset_step < repaired_power_array.shape[0]:
            if reset_step > 0:
                repaired_power_array[reset_step, :] = repaired_power_array[reset_step - 1, :]
            else:
                repaired_power_array[reset_step, :] = 0.0

    # Hybrid walked-distance estimator:
    # - normal timesteps: use horizontal world-frame position difference
    # - reset/teleport contaminated timesteps: use pre-step horizontal speed * dt
    # This avoids counting respawn teleports while still using actual positions whenever they are valid.
    pre_step_distance_increment_array = np.asarray(distance_increment_buffer, dtype=np.float64)
    distance_increment_array = np.zeros(total_sim_steps, dtype=np.float64)

    if total_sim_steps > 0:
        distance_increment_array[0] = pre_step_distance_increment_array[0]

    if total_sim_steps > 1:
        distance_increment_array[1:] = np.linalg.norm(np.diff(base_position_array[:, :2], axis=0), axis=1)

    for reset_step in all_reset_steps:
        if 0 <= reset_step < total_sim_steps:
            distance_increment_array[reset_step] = pre_step_distance_increment_array[reset_step]

    # Safety guard for any unmarked teleport / corrupted sample. 0.06m for 20ms is ~3m/s, so we fall back to a constant
    MAX_REASONABLE_HORIZONTAL_DISTANCE_PER_STEP_METERS = 0.06
    invalid_distance_mask = ~np.isfinite(distance_increment_array) | (distance_increment_array > MAX_REASONABLE_HORIZONTAL_DISTANCE_PER_STEP_METERS)
    distance_increment_array[invalid_distance_mask] = pre_step_distance_increment_array[invalid_distance_mask]

    np_data_file = os.path.join(plots_directory, "sim_data.npz")
    np.savez(
        np_data_file,
        env_name=env_name,
        run_name=run_name,
        task_name=task_name,
        robot_profile=robot_profile.name,
        action_scale=np.array(action_scale, dtype=object),
        eval_action_scale_before_override=np.array(eval_action_scale_before_override, dtype=object),
        sim_times=sim_times,
        reset_times=np.array(reset_times),
        manual_reset_times=np.array(manual_reset_times),
        automatic_reset_times=np.array(automatic_reset_times),
        reward_array=reward_array,
        inference_durations_us_array=inference_durations_us_array,
        inference_device=args.device,
        joint_positions_array=joint_positions_array,
        joint_velocities_array=joint_velocities_array,
        joint_torques_array=joint_torques_array,
        joint_accelerations_array=joint_accelerations_array,
        action_rate_array=action_rate_array,
        contact_forces_array=contact_forces_array,
        foot_contact_force_world_history_array=foot_contact_force_world_history_array,
        contact_force_threshold_newtons=CONTACT_FORCE_THRESHOLD_NEWTONS,
        base_position_array=base_position_array,
        base_orientation_array=base_orientation_array,
        base_linear_velocity_array=base_linear_velocity_array,
        base_angular_velocity_array=base_angular_velocity_array,
        base_linear_velocity_body_array=base_linear_velocity_body_array,
        base_angular_velocity_body_array=base_angular_velocity_body_array,
        base_linear_acceleration_world_array=base_linear_acceleration_world_array,
        base_angular_acceleration_world_array=base_angular_acceleration_world_array,
        base_linear_acceleration_body_array=base_linear_acceleration_body_array,
        base_angular_acceleration_body_array=base_angular_acceleration_body_array,
        base_acceleration_body_name=np.array(base_body_name),
        commanded_velocity_array=commanded_velocity_array,
        contact_state_array=contact_state_array,
        height_map_array=np.array(height_map_buffer),
        foot_velocities_world_frame_array=foot_velocities_world_frame_array,
        foot_velocities_body_frame_array=foot_velocities_body_frame_array,
        foot_positions_world_frame_array=foot_positions_world_frame_array,
        foot_positions_body_frame_array=foot_positions_body_frame_array,
        foot_positions_contact_frame_array=foot_positions_contact_frame_array,
        raw_power_array=raw_power_array,
        power_array=repaired_power_array,
        distance_increment_array=distance_increment_array,
        terrain_level_array=terrain_level_array,
        track_lin_vel_xy_exp_common_weight_array=track_lin_vel_xy_exp_common_weight_array,
        track_ang_vel_z_exp_common_weight_array=track_ang_vel_z_exp_common_weight_array,
        track_vel_exp_total_common_weight_array=track_vel_exp_total_common_weight_array,
        foot_labels=np.array(foot_labels),
        joint_names=np.array(joint_names),
        total_robot_mass=total_robot_mass,
        constraint_bounds=np.array(constraint_bounds, dtype=object),
        constraint_bounds_source=np.array(constraint_bounds_source),
        action_delay_steps=args.delay_actions,
        selected_fixed_scenario=np.array(args.fixed_scenario, dtype=object),
    )

    arrays_dict = {
        "joint_positions": joint_positions_array,
        "joint_velocities": joint_velocities_array,
        "joint_torques": joint_torques_array,
        "joint_accelerations": joint_accelerations_array,
        "action_rate": action_rate_array,
        "contact_forces": contact_forces_array,
        "foot_contact_force_world_history": foot_contact_force_world_history_array,
        "base_position": base_position_array,
        "base_orientation": base_orientation_array,
        "base_linear_velocity": base_linear_velocity_array,
        "base_angular_velocity": base_angular_velocity_array,
        "base_linear_velocity_body": base_linear_velocity_body_array,
        "base_angular_velocity_body": base_angular_velocity_body_array,
        "base_linear_acceleration_world": base_linear_acceleration_world_array,
        "base_angular_acceleration_world": base_angular_acceleration_world_array,
        "base_linear_acceleration_body": base_linear_acceleration_body_array,
        "base_angular_acceleration_body": base_angular_acceleration_body_array,
        "commanded_velocity": commanded_velocity_array,
        "contact_state": contact_state_array,
        "foot_velocities_world_frame": foot_velocities_world_frame_array,
        "foot_velocities_body_frame": foot_velocities_body_frame_array,
        "foot_positions_world_frame": foot_positions_world_frame_array,
        "foot_positions_body": foot_positions_body_frame_array,
        "foot_positions_contact_frame": foot_positions_contact_frame_array,
        "power_array": repaired_power_array,
        "distance_increment": distance_increment_array,
        "reward": reward_array,
    }

    eval_only_arrays = {
        "terrain_level": terrain_level_array,
        "track_lin_vel_xy_exp_common_weight": track_lin_vel_xy_exp_common_weight_array,
        "track_ang_vel_z_exp_common_weight": track_ang_vel_z_exp_common_weight_array,
        "track_vel_exp_total_common_weight": track_vel_exp_total_common_weight_array,
    }

    constants_dict = {
        "step_dt": step_dt,
        "joint_names": joint_names,
        "foot_labels": foot_labels,
        "constraint_bounds": constraint_bounds,
        "total_robot_mass": total_robot_mass,
        "joint_role_mapping": robot_profile.joint_role_mapping,
    }

    # --- masks ---
    T = total_sim_steps
    all_indices = np.arange(T)
    random_timestep_mask = all_indices < args.random_sim_step_length

    scenario_masks = {}
    for k, (scenario_tag, *_) in enumerate(fixed_command_scenarios):
        start = args.random_sim_step_length + k * fixed_command_sim_steps
        end = start + fixed_command_sim_steps  # exclusive
        scenario_masks[scenario_tag] = (all_indices >= start) & (all_indices < end)

    overall_metrics = add_eval_only_metrics_to_summary(
        compute_summary_metrics(np.ones(T, bool), manual_reset_steps, automatic_reset_steps, arrays_dict, constants_dict),
        np.ones(T, bool),
        eval_only_arrays,
    )
    random_metrics = None
    if random_timestep_mask.any():
        random_metrics = add_eval_only_metrics_to_summary(
            compute_summary_metrics(random_timestep_mask, manual_reset_steps, automatic_reset_steps, arrays_dict, constants_dict),
            random_timestep_mask,
            eval_only_arrays,
        )
    scenario_metrics = {
        tag: add_eval_only_metrics_to_summary(
            compute_summary_metrics(msk, manual_reset_steps, automatic_reset_steps, arrays_dict, constants_dict),
            msk,
            eval_only_arrays,
        )
        for tag, msk in scenario_masks.items()
    }

    rebuttal_scenario_metrics: dict[str, Any] = {}
    warmup_seconds = 1.0
    warmup_steps = int(round(warmup_seconds / step_dt))
    for scenario_index, (scenario_tag, *_) in enumerate(fixed_command_scenarios):
        if scenario_tag not in REBUTTAL_DYNAMICS_SCENARIOS:
            continue

        scenario_start = args.random_sim_step_length + scenario_index * fixed_command_sim_steps
        scenario_end = scenario_start + fixed_command_sim_steps
        dynamics_mask, window_metadata = build_scenario_analysis_mask(
            total_steps=T,
            scenario_start=scenario_start,
            scenario_end=scenario_end,
            warmup_steps=warmup_steps,
            automatic_reset_steps=automatic_reset_steps,
        )
        window_metadata.update(
            {
                "warmup_seconds": warmup_seconds,
                "valid_duration_seconds": float(window_metadata["sample_count"] * step_dt),
            }
        )
        gait_dynamics = compute_gait_dynamics_metrics(
            mask=dynamics_mask,
            data_arrays=arrays_dict,
            constants=constants_dict,
            data_fidelity={
                "base_acceleration": "direct",
                "vertical_grf": "direct_vector_history",
            },
        )
        general_metrics = scenario_metrics[scenario_tag]
        maximum_constraint_violation = general_metrics.get("maximum_constraint_violation")
        if not isinstance(maximum_constraint_violation, dict):
            maximum_constraint_violation = {}
        achieved_velocity = base_linear_velocity_body_array[dynamics_mask]
        mean_achieved_velocity = (
            achieved_velocity.mean(axis=0).tolist() if achieved_velocity.shape[0] > 0 else None
        )
        rebuttal_scenario_metrics[scenario_tag] = {
            "analysis_window": window_metadata,
            "gait_dynamics": gait_dynamics,
            "context": {
                "mean_achieved_base_linear_velocity_body_m_s": mean_achieved_velocity,
                "base_linear_velocity_x_rms_error": general_metrics.get("base_linear_velocity_x_rms_error"),
                "base_linear_velocity_x_mean_abs_error": general_metrics.get(
                    "base_linear_velocity_x_mean_abs_error"
                ),
                "base_linear_velocity_y_rms_error": general_metrics.get("base_linear_velocity_y_rms_error"),
                "base_linear_velocity_y_mean_abs_error": general_metrics.get(
                    "base_linear_velocity_y_mean_abs_error"
                ),
                "base_angular_velocity_z_rms_error": general_metrics.get("base_angular_velocity_z_rms_error"),
                "base_angular_velocity_z_mean_abs_error": general_metrics.get(
                    "base_angular_velocity_z_mean_abs_error"
                ),
                "cost_of_transport": general_metrics.get("cost_of_transport"),
                "max_operational_limit_violation_frequency_percent": maximum_numeric_value(
                    general_metrics.get("constraint_violations_percent", {})
                ),
                "max_operational_limit_violation_percent": maximum_constraint_violation.get(
                    "normalized_excess_percent"
                ),
                "max_operational_limit_violation_constraint": maximum_constraint_violation.get(
                    "constraint"
                ),
                "max_operational_limit_violation_element": maximum_constraint_violation.get(
                    "element"
                ),
                "max_operational_limit_violation_absolute_excess": maximum_constraint_violation.get(
                    "absolute_excess"
                ),
                "maximum_constraint_violation": maximum_constraint_violation or None,
            },
        }
    scenario_completion = compute_scenario_completion(
        scenarios=fixed_command_scenarios,
        random_sim_steps=args.random_sim_step_length,
        fixed_scenario_steps=fixed_command_sim_steps,
        automatic_reset_steps=automatic_reset_steps,
    )

    summary_metrics = dict(overall_metrics)  # start with overall block
    # metrics per segment / fixed command scenario
    summary_metrics["random_simulation_steps_metrics"] = random_metrics
    summary_metrics["fixed_command_scenarios_metrics"] = scenario_metrics
    summary_metrics["fixed_command_scenarios_gait_dynamics_metrics"] = rebuttal_scenario_metrics
    summary_metrics.update({
        "env_name": env_name,
        "run_name": run_name,
        "task_name": task_name,
        "robot_profile": robot_profile.name,
        "run_dir": str(run_path),
        "eval_base_dir": eval_base_dir,
        "random_sim_steps": args.random_sim_step_length,
        "fixed_command_sim_steps": fixed_command_sim_steps,
        "total_sim_steps": total_sim_steps,
        "seed": env_cfg.seed,
        "action_scale": action_scale,
        "eval_action_scale_before_override": eval_action_scale_before_override,
        "used_checkpoint_path": checkpoint_path,
        "policy_backend": policy_backend,
        "detected_policy_backend": detected_policy_backend,
        "fixed_command_scenarios": fixed_command_scenarios,
        "fixed_command_scenario_completion": scenario_completion,
        "manual_reset_steps": manual_reset_steps,
        "automatic_reset_steps": automatic_reset_steps,
        "action_delay_steps": args.delay_actions,
        "selected_fixed_scenario": args.fixed_scenario,
        "rebuttal_scenarios_only": args.rebuttal_scenarios_only,
        "rebuttal_dynamics_scenarios": REBUTTAL_DYNAMICS_SCENARIOS,
        "rebuttal_dynamics_warmup_seconds": warmup_seconds,
        "contact_force_threshold_newtons": CONTACT_FORCE_THRESHOLD_NEWTONS,
        "base_acceleration_body_name": base_body_name,
        "constraint_bounds": constraint_bounds,
        "constraint_bounds_source": constraint_bounds_source,
        "hardcoded_upstream_go2_constraint_bounds": UPSTREAM_GO2_HARDCODED_CONSTRAINT_BOUNDS,
        "hardcoded_upstream_go2_constraint_bounds_used": (
            constraint_bounds_source == "hardcoded_upstream_go2_rough_eval_thresholds"
        ),
        "matched_baseline_constraint_bounds": MATCHED_BASELINE_CONSTRAINT_BOUNDS,
        "matched_baseline_constraint_bounds_used": constraint_bounds_source.startswith(
            "hardcoded_matched_baseline_"
        ),
        "eval_tracking_reward_common_scale": {
            "track_lin_vel_xy_exp_weight": 1.0,
            "track_ang_vel_z_exp_weight": 0.5,
            "std_squared": 0.25,
            "environment_reward_weights_modified": eval_reward_scale_override_applied,
            "note": (
                "These common tracking metrics are computed separately from raw command/state data. "
                "Matched baseline environment reward weights remain upstream and unchanged; only the "
                "legacy upstream Go2 task may be reweighted when its compatibility flag is enabled."
            ),
        },
        "downscale_upstream_go2_tracking_rewards": args.downscale_upstream_go2_tracking_rewards,
    })

    summary_path = os.path.join(eval_base_dir, "metrics_summary.json")
    with open(summary_path, 'w') as summary_file:
        json.dump(summary_metrics, summary_file, indent=4, default=lambda o: o.tolist())  # lambda for torch/numpy tensor conversion or any other nested objects
    # print(json.dumps(summary_metrics, indent=4, default=lambda o: o.tolist()))

    raw_frames = get_render_frames(env)
    for frame in raw_frames:
        ffmpeg_process.stdin.write(frame.tobytes())
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    plot_jobs = []
    for k, (scenario_tag, *_rest) in enumerate(fixed_command_scenarios):
        start = args.random_sim_step_length + k * fixed_command_sim_steps
        end = start + fixed_command_sim_steps
        subdir = f"scenario_{scenario_tag}"
        plot_jobs.append(
            {
                "start_step": start,
                "end_step": end,
                "subdir": subdir,
                "rebuttal_plots": scenario_tag in REBUTTAL_DYNAMICS_SCENARIOS,
            }
        )

    if args.random_sim_step_length > 0:
        plot_jobs.append({"start_step": 0, "end_step": args.random_sim_step_length, "subdir": "random_simulation_steps"})
    plot_jobs.append({"start_step": 0, "end_step": total_sim_steps, "subdir": "overall"})

    plot_status = run_generate_plots_parallel(
        plot_jobs=plot_jobs,
        plots_directory=plots_directory,
        sim_data_file_path=np_data_file,
        foot_vel_height_threshold=args.foot_vel_height_threshold,
        num_parallel=args.num_plot_jobs_in_parallel,
        stagger_delay=args.plot_job_stagger_delay
    )
    report_paths = write_rebuttal_reports(
        summary_metrics=summary_metrics,
        output_dir=eval_base_dir,
        plot_status=plot_status,
    )
    print("[INFO] Wrote rebuttal gait-dynamics reports:")
    for report_format, report_path in report_paths.items():
        print(f"[INFO]   {report_format}: {report_path}")

    cleanup_runtime_resources()
    atexit.unregister(cleanup_runtime_resources)


if __name__ == "__main__":
    main()
