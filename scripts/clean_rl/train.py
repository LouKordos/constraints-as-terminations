import argparse
import sys
import os
from datetime import datetime
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") # Determinism
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
from functools import partial
sys.stdout.reconfigure(line_buffering=True)
print = partial(print, flush=True) # For cluster runs

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description="Train an RL agent with CleanRL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=46, help="Seed used for the environment"
)
parser.add_argument(
    "--num_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--energy_end_weight",
    type=float,
    default=None,
    help="Override curriculum.power.params['end_weight']; defaults to the task configuration.",
)
cli_args.add_clean_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import (DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.hydra import hydra_task_config
from locomposition.tasks.utils.cleanrl.ppo import PPO
import locomposition.tasks  # noqa: F401

from os import environ
import random
import numpy as np
import torch

from pxr import Usd, UsdGeom, Tf
import hashlib
def _first_mesh_with_indices(prim):
    """DFS until we find a UsdGeom.Mesh that actually has indices."""
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        if m.GetFaceVertexIndicesAttr().HasAuthoredValueOpinion():
            return m
    for child in prim.GetChildren():
        hit = _first_mesh_with_indices(child)
        if hit:
            return hit
    return None

def _hash_mesh(mesh: UsdGeom.Mesh):
    pts  = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    idx  = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(
                       Usd.TimeCode.Default()), dtype=np.int32)
    return hashlib.sha1(pts.tobytes() + idx.tobytes()).hexdigest()

def _hash_heightfield(prim):
    # PhysX height-field data live on a custom attribute:
    data_attr = prim.GetAttribute("physxHeightField:data")
    if not data_attr or not data_attr.HasAuthoredValueOpinion():
        raise RuntimeError("Height-field has no data attribute")
    hf = np.asarray(data_attr.Get(), dtype=np.int16)
    return hashlib.sha1(hf.tobytes()).hexdigest()

def get_ground_hash(env):
    stage  = env.unwrapped.scene.stage
    root   = stage.GetPrimAtPath("/World/ground")
    
    mesh = _first_mesh_with_indices(root)
    if mesh:
        return _hash_mesh(mesh)
    
    # fall-back: maybe this is a PhysX height-field
    if root.HasAPI(Tf.Type.FindByName("PhysxHeightField")):
        return _hash_heightfield(root)
    
    raise ValueError(f"No usable mesh or height-field found under {root.GetPath()}")


def print_resolved_training_config(env_cfg, task_name: str) -> None:
    """Print the embodiment-dependent values that most often invalidate a long run."""
    robot_cfg = env_cfg.scene.robot
    usd_path = getattr(getattr(robot_cfg, "spawn", None), "usd_path", None)
    power_term = getattr(getattr(env_cfg, "curriculum", None), "power", None)
    power_end_weight = None if power_term is None else power_term.params.get("end_weight")

    print("[INFO] Resolved training embodiment/configuration:")
    print(f"[INFO] task={task_name}")
    print(f"[INFO] robot_usd={usd_path}")
    print(f"[INFO] action_joint_names={env_cfg.actions.joint_pos.joint_names}")
    print(f"[INFO] action_scale={env_cfg.actions.joint_pos.scale}")
    print(f"[INFO] control_step_dt={env_cfg.sim.dt * env_cfg.decimation}")
    print(f"[INFO] episode_length_s={env_cfg.episode_length_s}")
    print(f"[INFO] energy_end_weight={power_end_weight}")

    constraints_cfg = getattr(env_cfg, "constraints", None)
    if constraints_cfg is not None:
        for term_name in (
            "joint_torque",
            "joint_velocity",
            "joint_acceleration",
            "action_rate",
            "foot_contact_force",
            "front_hfe_position",
            "hip_position",
            "no_move",
        ):
            term_cfg = getattr(constraints_cfg, term_name, None)
            if term_cfg is not None:
                print(f"[INFO] constraint.{term_name}={term_cfg.params}")

    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None:
        for term_name in ("randomize_mass", "push_robot", "push_base_wrench"):
            term_cfg = getattr(events_cfg, term_name, None)
            if term_cfg is not None:
                print(f"[INFO] event.{term_name}={term_cfg.params}")

@hydra_task_config(args_cli.task, "clean_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg,):

    if environ.get("ENV_NAME") is None:
        print("\n\n----------------------------------------------------------------------------------")
        print("ERROR: Please set ENV_NAME environment variable before running this script, exiting.")
        print("----------------------------------------------------------------------------------")
        exit(1)

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_clean_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs)
    agent_cfg.num_iterations = (
        args_cli.num_iterations
        if args_cli.num_iterations is not None
        else agent_cfg.num_iterations
    )

    if args_cli.energy_end_weight is not None:
        power_term = getattr(getattr(env_cfg, "curriculum", None), "power", None)
        if power_term is None:
            raise ValueError("--energy_end_weight requires an active curriculum.power term")
        if args_cli.energy_end_weight < 0.0:
            raise ValueError("--energy_end_weight must be non-negative")
        power_term.params["end_weight"] = args_cli.energy_end_weight

    seed = args_cli.seed
    agent_cfg.seed = seed
    env_cfg.seed = seed
    env_cfg.sim.random_seed = seed

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.seed = seed

    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
    velocity_mdp.terrain_levels_vel.seed = seed
    set_global_seed(seed)

    print(f"[INFO] Using seed={seed}")
    print(f"[INFO] agent_cfg.seed={agent_cfg.seed}")
    print(f"[INFO] env_cfg.seed={env_cfg.seed}")
    print(f"[INFO] env_cfg.sim.random_seed={env_cfg.sim.random_seed}")
    if env_cfg.scene.terrain.terrain_generator is not None:
        print(f"[INFO] terrain_generator.seed={env_cfg.scene.terrain.terrain_generator.seed}")

    print_resolved_training_config(env_cfg, args_cli.task)

    env_cfg.sim.device = (args_cli.device if args_cli.device is not None else env_cfg.sim.device)

    # Follow robot with viewport
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name  = "robot"
    env_cfg.viewer.eye        = (0.0, -5.0, 5.0)
    env_cfg.viewer.lookat     = (0.0,  0.0, 0.5)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "clean_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    print("Terrain hash:", get_ground_hash(env))
    if env.unwrapped.scene.terrain.cfg.terrain_type != "plane" and get_ground_hash(env) != "e3f8594b1c2755f00290cebc3d98598721063bd0":
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print("!!!!!!Unexpected terrain hash!!!!!!")
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        #sys.exit(1)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos_train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    PPO(env, agent_cfg, log_dir)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
