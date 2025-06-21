import argparse
import os
import glob
import json
import yaml
import numpy as np
import torch
import time
from functools import partial
import sys
sys.stdout.reconfigure(line_buffering=True)
print = partial(print, flush=True) # For cluster runs
import gymnasium as gym
from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import fcntl
from queue import Queue
import subprocess
import os
import re
import yaml
from typing import Dict, Tuple, Optional, List, Any
from metrics_utils import compute_summary_metrics

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Determinism
os.environ["OMNICLIENT_HUB_MODE"] = "disabled"

eval_script_path = os.path.dirname(os.path.abspath(__file__))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Play an RL agent with detailed logging.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="ABSOLUTE path to directory containing model checkpoints and params.")
    parser.add_argument("--eval_checkpoint", type=str, default=None,
                        help="Optionally specify the model save checkpoint number instead of automatically using the last saved one.")
    parser.add_argument("--random_sim_step_length", type=int, default=4000,
                        help="Number of steps to run with random commands and spawn points. Standardized tests like standing and walking forward will always run.")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, required=True,
                        help="Name of the task/environment.")
    # Good seeds for eval: 44, 46, 49
    # DEPRECATED: Hardcoded seed in env config is used
    # parser.add_argument("--seed", type=int, required=False, default=46, help="Seed for numpy, torch, env, terrain, terrain generator etc.. Good seeds for eval are 44, 46, 49")

    sys.path.insert(0, os.path.join(eval_script_path, "clean_rl"))
    import cli_args  # isort: skip
    cli_args.add_clean_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    arguments = parser.parse_args()
    arguments.enable_cameras = True # Video
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

def load_constraint_bounds(params_directory: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Returns a dict mapping each constraint key (either a global term
    like 'joint_torque' or an individual joint name) to a (lb, ub) tuple.
    - joint_position    → (None, limit)
    - joint_position_when_moving_forward → (default-limit, default+limit)
    - foot_contact_force → (0, limit)
    - everything else   → (-limit, +limit)
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

        # 3a) joint_position → only upper bound
        if func.endswith('joint_position_absolute_upper_bound'):
            joints = expand_patterns(patterns)
            for jn in joints:
                bounds[jn] = (None, limit)

        # 3b) joint_position_when_moving_forward → relative bound about default
        elif func.endswith('relative_joint_position_upper_and_lower_bound_when_moving_forward'):
            joints = expand_patterns(patterns)
            for jn in joints:
                base = default_pos.get(jn, 0.0)
                bounds[jn] = (base - limit, base + limit)

        # 3c) foot_contact_force → only positive
        elif term == 'foot_contact_force':
            bounds[term] = (0.0, limit)

        # 3d) everything else → symmetric ±limit
        else:
            bounds[term] = (-limit, limit)

    return bounds

def run_generate_plots(start_step: int, end_step: int, subdir: str, plots_directory: str, sim_data_file_path: str, memory_limit_gb: Optional[float] = 40) -> int:
    """
    Launch generate_plots.py for [start_step, end_step) and block until it finishes. Return the subprocess' return-code.
    """
    output_dir = os.path.join(plots_directory, subdir)
    os.makedirs(output_dir, exist_ok=True)
    generate_plots_script_path = os.path.join(eval_script_path, "generate_plots.py")

    log_path = os.path.join(plots_directory, f"generate_plots_{subdir}.log")
    cmd = [
        "python",
        generate_plots_script_path,
        "--data_file", sim_data_file_path,
        "--output_dir", output_dir,
        "--start_step", str(start_step),
        "--end_step", str(end_step),
        # "--interactive",
    ]

    preexec: Optional[callable] = None
    if memory_limit_gb is not None:
        limit_bytes = int(memory_limit_gb * (1024 ** 3))
        def _set_mem_limit():
            import resource
            # RLIMIT_AS sets the maximum address space (virtual memory) for the process
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        preexec = _set_mem_limit

    print(f"[INFO] Spawning: {' '.join(cmd)}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=preexec)
        proc.wait()
        print(f"[INFO]  generate_plots.py for '{subdir}' exited with code {proc.returncode}  (log → {log_path})")
        return proc.returncode

def main():
    args = parse_arguments()
    args.run_dir = os.path.abspath(args.run_dir)

    if not "play" in args.task.lower():
        input("\n\n-------------------------------------------------------------------\n\
              Keyword 'Play' not found in task name, are you sure you are using the correct task/environment?\n" \
              "-------------------------------------------------------------------\n\n")

    if args.eval_checkpoint is None:
        checkpoint_path = get_latest_checkpoint(args.run_dir)
    else:
        checkpoint_path = os.path.join(args.run_dir, f"model_{args.eval_checkpoint}.pt")
        if not os.path.isfile(checkpoint_path):
            print(f"ERROR: checkpoint file does not exist, exiting: {checkpoint_path}")
            exit(1)

    print(f"[INFO] Loading model from: {checkpoint_path}")
    model_state = torch.load(checkpoint_path, weights_only=True)

    # Launch Isaac Lab environment
    args.device = "cuda" # Using CPU Increases iterations/sec in some cases and reduces VRAM usage which allows parallel runs. Replace with "cuda" if you want pure GPU, because on more recent GPUs this might be faster depending on your hardware
    args.disable_fabric = True if args.device == "cpu" else False
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    from isaaclab.utils.dict import print_dict
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from cat_envs.tasks.utils.cleanrl.ppo import Agent
    from cat_envs.tasks.locomotion.velocity.config.solo12.cat_go2_rough_terrain_env_cfg import height_map_grid
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab.envs.mdp.observations import root_quat_w
    from isaaclab.managers import SceneEntityCfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)

    # Seeding
    import random
    random.seed(env_cfg.seed)
    np.random.seed(env_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(env_cfg.seed)
    torch.cuda.manual_seed_all(env_cfg.seed)

    # Viewer setup
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (0.0, -3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.sim.render.rendering_mode = "quality"
    step_dt = env_cfg.sim.dt * env_cfg.decimation # Physics run at higher frequency, action is applied `decimation` physics-steps, but video uses env steps as unit
    frame_width = 1920
    frame_height = 1080
    frame_rate = int(round(1.0 / step_dt))
    env_cfg.viewer.resolution = (frame_width, frame_height)
    device = torch.device(args.device)

    eval_base_dir = os.path.join(args.run_dir, f"eval_checkpoint_{os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]}_seed_{env_cfg.seed}")
    env_name = os.path.basename(os.path.dirname(args.run_dir.rstrip("/")))
    print(f"eval_base_dir={eval_base_dir}, env_name={env_name}")

    # Create output directories
    plots_directory = os.path.join(eval_base_dir, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    fixed_command_sim_steps = 500 # If you want to increase this you also need to increase episode length otherwise env will reset mid-way
    fixed_command_scenarios = [ # Scenario positions depend on seed!
        ("fast_walk_stairs_up", torch.tensor([1, 0.0, 0.0], device=device), (torch.tensor([-8, 16, -0.1], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("stand_still", torch.tensor([0.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("pure_spin", torch.tensor([0.0, 0.0, 0.5], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_x_flat_terrain", torch.tensor([0.1, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_y_flat_terrain", torch.tensor([0.0, 0.1, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_x_flat_terrain", torch.tensor([0.5, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_y_flat_terrain", torch.tensor([0.0, 0.5, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_x_flat_terrain", torch.tensor([1.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_y_flat_terrain", torch.tensor([0.0, 1.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("very_fast_walk_x_flat_terrain", torch.tensor([2.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_x_uneven_terrain", torch.tensor([0.1, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_y_uneven_terrain", torch.tensor([0.0, 0.1, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_x_uneven_terrain", torch.tensor([0.5, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_y_uneven_terrain", torch.tensor([0.0, 0.5, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_x_uneven_terrain", torch.tensor([1.0, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("very_fast_walk_x_uneven_terrain", torch.tensor([2.0, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        # ("fast_walk_y_uneven_terrain", torch.tensor([0.0, 1.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_diagonal_uneven_terrain", torch.tensor([0.5, 0.5, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_diagonal_uneven_terrain", torch.tensor([1.0, 1.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("medium_walk_diagonal_turning_uneven_terrain", torch.tensor([0.8, 0.5, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([np.cos(np.pi/8), 0.0, 0.0, np.sin(np.pi/8)], dtype=torch.float32, device=device))),
    ]

    # See main training loop for detailed explanation, but in summary, hard constraints terminate the environment or at least return terminated = 1
    # which results in zero reward and can't be recovered by rescaling. Thus, we update the constraint limits based on the loaded values from
    # the training environment. Only contact force and thigh position limit are updated because other values are handled more robustly by the rescaling in the loop.
    # This allows more constraints to be added or removed without leading to issues in this eval script.
    constraint_bounds = load_constraint_bounds(os.path.join(args.run_dir, 'params'))
    env_cfg.constraints.foot_contact_force.params["limit"] = constraint_bounds["foot_contact_force"][1]
    env_cfg.constraints.front_hfe_position.params["limit"] = constraint_bounds["RL_thigh_joint"][1]

    total_sim_steps = args.random_sim_step_length + len(fixed_command_scenarios) * fixed_command_sim_steps
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    video_configuration = {
        "pop_frames": True, # env.render() is called periodically in the sim loop to store the frames on disk and thus reduce memory usage. pop_frames clears the in-memory buffer after calling render()
        "reset_clean": False, # Since sim loop resets the env for the hardcoded scenarios, the frames should be preserved on env.reset()
    }
    env = gym.wrappers.RenderCollection(env, **video_configuration)

    policy_agent = Agent(env).to(device)
    policy_agent.load_state_dict(model_state)

    robot = env.unwrapped.scene["robot"]
    vel_term = env.unwrapped.command_manager.get_term("base_velocity")

    def set_fixed_velocity_command(vec:torch.Tensor):
        # VERY dirty hack but this is just an eval script so it's ok. Basically don't allow the sampler to overwrite fixed commands
        vel_term.cfg.resampling_time_range = (1000000, 1000000)
        vel_term.cfg.heading_command = False
        vel_term._update_command = lambda *_, **__: None 
        vel_term.vel_command_b = vec.repeat(env.unwrapped.num_envs, 1)

    def teleport_robot(pos_xyz, quat_xyzw):
        pose = torch.cat([pos_xyz, quat_xyzw]).unsqueeze(0)
        robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([0],device=device))
        env.unwrapped.scene.write_data_to_sim()

    joint_names = env.unwrapped.scene["robot"].data.joint_names
    foot_links = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'] if "go2" in args.task.lower() else ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
    foot_labels = ['front left','front right','rear left','rear right']

    # Initialize buffers for recording
    joint_positions_buffer = []
    joint_velocities_buffer = []
    contact_forces_buffer = []
    joint_torques_buffer = []
    joint_accelerations_buffer = []
    action_rate_buffer = []
    base_position_buffer = []
    base_orientation_buffer = []
    base_linear_velocity_buffer = []
    base_angular_velocity_buffer = []
    commanded_velocity_buffer = []
    contact_state_buffer = []
    height_map_buffer = []
    foot_positions_world_frame_buffer = []
    foot_velocities_world_frame_buffer = []
    foot_positions_body_frame_buffer = []
    foot_positions_contact_frame_buffer = [] # Height above terrain, also called sole frame

    reward_buffer = []
    reset_steps = []

    observations, info = env.reset()
    policy_observation = observations['policy']
    previous_action = None

    video_output_path = os.path.join(eval_base_dir, f"{os.path.basename(eval_base_dir)}_run_{os.path.basename(args.run_dir)}.mp4")
    frame_storage_interval = 1
    ffmpeg_cmd = ["ffmpeg", "-y", "-hwaccel", "cuda", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frame_width}x{frame_height}", "-framerate", str(frame_rate), "-i", "pipe:0", "-c:v", "hevc_nvenc", "-preset", "slow", "-movflags", "+use_metadata_tags", "-metadata", f"env_name={env_name}", video_output_path]
    ffmpeg_process_log_path = os.path.join(eval_base_dir, "ffmpeg_encode.log")
    with open(ffmpeg_process_log_path, "w") as ffmpeg_process_logfile:
        print(f"Starting ffmpeg process={' '.join(ffmpeg_cmd)}")
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=ffmpeg_process_logfile, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, bufsize=4*1024*1024)

    for t in tqdm(range(total_sim_steps)):
        # These should only ever run at the end of eval / after random sampling because set_fixed_velocity_command breaks random command sampling!
        if t >= args.random_sim_step_length and (t-args.random_sim_step_length) % fixed_command_sim_steps == 0:
            scenario = fixed_command_scenarios[int(t-args.random_sim_step_length) // fixed_command_sim_steps]
            fixed_command = scenario[1]
            spawn_point_pos, spawn_point_quat = scenario[2]
            print(f"Resetting env and setting fixed command + spawn point for scenario={scenario}...")
            obs, _ = env.reset()
            reset_steps.append(t)
            teleport_robot(spawn_point_pos, spawn_point_quat)
            set_fixed_velocity_command(fixed_command)
            policy_observation = obs["policy"]

        with torch.no_grad():
            action, _, _, _, _ = policy_agent.get_action_and_value(policy_agent.obs_rms(policy_observation, update=False), use_deterministic_policy=True)
        step_tuple = env.step(action)
        # print(step_tuple)
        next_observation, reward, terminated, truncated, info = step_tuple

        # Idea of constraints as terminations is to scale reward by max constraint violation, but since policies
        # being tested here might differ in the constraints they were trained with, it makes sense to remove this scaling
        # while evaluating a policy. This division just reverts the scaling done in cat_env.py and thus should yield the
        # "raw" rewards collected in the environment.
        # If this were omitted, a policy trained with e.g. higher joint position constraint limits would result in very low
        # rewards in this eval environment as it always calculates based on the limits specified in the local code, not what
        # the policy was trained with. Additionally, any hard constraints (max_p=1.0) need to be manually increased earlier
        # in the setup because those terminate the environment and thus break the reproducibility and the if check below.
        if terminated != 1.0:
            # print(f"reward_before_scaling={reward}")
            reward /= (1.0 - terminated)
            # print(f"reward_after_scaling={reward}\tterminated={terminated}")
        
        # Because of CaT, terminated is actually a nonzero probability instead of a boolean, so we have to check for resets this way
        if env.unwrapped.episode_length_buf[0].item() == 0 and t > 0:
            reset_steps.append(t)
            # continue # Skip reset iteration because it's just wrong

        if t % frame_storage_interval == 0:
            raw_frames = env.render()
            for frame in raw_frames:
                assert frame.shape[:2] == (frame_height, frame_width), "Returned frame does not match specified dimension."
                ffmpeg_process.stdin.write(frame.tobytes())

        scene_robot_data = env.unwrapped.scene['robot'].data

        joint_positions = scene_robot_data.joint_pos[0].cpu().numpy()
        joint_velocities = scene_robot_data.joint_vel[0].cpu().numpy()
        joint_positions_buffer.append(joint_positions)
        joint_velocities_buffer.append(joint_velocities)

        contact_sensors = env.unwrapped.scene['contact_forces']
        feet_ids,_ = contact_sensors.find_bodies(foot_links, preserve_order=True)
        net_forces = contact_sensors.data.net_forces_w_history
        forces_history = net_forces[0].cpu()[:, feet_ids, :]
        force_magnitudes = torch.norm(forces_history,dim=-1)
        max_per_foot = force_magnitudes.max(dim=0)[0].numpy()
        contact_forces_buffer.append(max_per_foot)

        torques = scene_robot_data.applied_torque[0].cpu().numpy()
        joint_torques_buffer.append(torques)

        accelerations = scene_robot_data.joint_acc[0].cpu().numpy()
        joint_accelerations_buffer.append(accelerations)

        action_np = action.cpu().numpy()
        if previous_action is None:
            action_rate = np.zeros_like(action_np)
        else:
            action_rate = np.abs(action_np - previous_action)
        action_rate_buffer.append(action_rate)
        previous_action = action_np

        base_world_position = scene_robot_data.root_link_pos_w[0].cpu().numpy() # world-frame for env 0
        # origin = env.unwrapped.scene.terrain.env_origins[0].cpu().numpy() # terrain origin for env 0
        # relative_position = world_position + origin # position relative to terrain
        # quat_xyzw = scene_data.root_quat_w[0]
        # quat_wxyz = torch.cat([quat_xyzw[3:], quat_xyzw[:3]]) # reorder to (w,x,y,z)
        quat_wxyz = root_quat_w(env.unwrapped, make_quat_unique=True, asset_cfg=SceneEntityCfg("robot"))
        roll_t, pitch_t, yaw_t = euler_xyz_from_quat(quat_wxyz)
        # Isaaclab returns 0 to 2pi euler angles, so small negative angles wrap around to ~2pi. Rescale accordingly
        def convert_to_signed_angle(a): return (a + np.pi) % (2*np.pi) - np.pi

        # IMPORTANT: THESE ANGLES ARE NOT UNWRAPPED YET, HAPPENS AFTER THE FULL ROLLOUT
        roll = convert_to_signed_angle(roll_t.cpu().numpy().item())
        pitch = convert_to_signed_angle(pitch_t.cpu().numpy().item())
        yaw = convert_to_signed_angle(yaw_t.cpu().numpy().item())
        # print(f"UNWRAPPED roll={roll}\tUNWRAPPED pitch={pitch}")

        base_position_buffer.append(base_world_position)
        base_orientation_buffer.append([yaw, pitch, roll])

        linear_velocity = scene_robot_data.root_lin_vel_w[0].cpu().numpy()
        angular_velocity = scene_robot_data.root_ang_vel_w[0].cpu().numpy()
        base_linear_velocity_buffer.append(linear_velocity)
        base_angular_velocity_buffer.append(angular_velocity)

        commanded_velocity_buffer.append(env.unwrapped.command_manager.get_command("base_velocity").clone()) # three components: lin_vel_x, lin_vel_y, ang_vel_z
        contact_state = (max_per_foot > 0).astype(int)
        contact_state_buffer.append(contact_state)

        height_map_sequence = height_map_grid(env.unwrapped, SceneEntityCfg(name="ray_caster")).cpu().numpy()
        height_map_buffer.append(height_map_sequence[0])
        
        # This will not account for terrain height, i.e. if the robot is standing on a 1m obstacle, the world foot height will be 1m.
        foot_positions_world = np.stack([scene_robot_data.body_link_pos_w[0, scene_robot_data.body_names.index(link)].cpu().numpy() for link in foot_links])
        foot_positions_world_frame_buffer.append(foot_positions_world)
        
        foot_velocities_world = np.stack([scene_robot_data.body_link_vel_w[0, scene_robot_data.body_names.index(link), :3].cpu().numpy() for link in foot_links])
        foot_velocities_world_frame_buffer.append(foot_velocities_world)
        
        foot_positions_body = env.unwrapped.scene["foot_frame_transformer"].data.target_pos_source[0].cpu().numpy()
        # print(f"foot_positions_body={foot_positions_body}")
        foot_positions_body_frame_buffer.append(foot_positions_body)

        # Calculate foot height above ground using raycaster sensor in each foot (sole/contact frame)
        foot_com_toe_tip_offset = 0.023 # This makes swing height more intuitive, without the offset, standing still reports a ~0.0234m swing height
        terrain_offset_feet = np.array([[0, 0, env.unwrapped.scene[f"ray_caster_{link_name}"].data.ray_hits_w[:, 0, 2].cpu().item() + foot_com_toe_tip_offset] for link_name in foot_links])
        # print(f"terrain_z_feet={terrain_offset_feet}")
        foot_positions_contact_frame = foot_positions_world - terrain_offset_feet
        # print(f"foot_positions_contact_frame={foot_positions_contact_frame}")
        foot_positions_contact_frame_buffer.append(foot_positions_contact_frame)

        reward_buffer.append(reward.mean().item())
        policy_observation = next_observation['policy']

    print("Sim loop done, converting and saving recorded sim data...")
    # Convert buffers to numpy arrays
    time_indices = np.arange(total_sim_steps)
    sim_times = time_indices * step_dt
    reset_times = [i * step_dt for i in reset_steps]
    print("Reset times: ", reset_times)
    reward_array = np.array(reward_buffer)
    joint_positions_array = np.vstack(joint_positions_buffer)
    joint_velocities_array = np.vstack(joint_velocities_buffer)
    joint_torques_array = np.vstack(joint_torques_buffer)
    joint_accelerations_array = np.vstack(joint_accelerations_buffer)
    action_rate_array = np.vstack(action_rate_buffer)
    contact_forces_array = np.stack(contact_forces_buffer)
    base_position_array = np.vstack(base_position_buffer)
    print(np.vstack(base_orientation_buffer).shape)
    base_orientation_array = np.unwrap(np.vstack(base_orientation_buffer), axis=0)
    print(base_orientation_array.shape)
    base_linear_velocity_array = np.vstack(base_linear_velocity_buffer)
    base_angular_velocity_array = np.vstack(base_angular_velocity_buffer)
    contact_state_array = np.vstack(contact_state_buffer)
    commanded_velocity_array = np.vstack([cv.cpu().numpy() if isinstance(cv, torch.Tensor) else np.asarray(cv) for cv in commanded_velocity_buffer])
    foot_velocities_world_frame_array = np.array(foot_velocities_world_frame_buffer)
    foot_positions_world_frame_array = np.array(foot_positions_world_frame_buffer)
    foot_positions_body_frame_array = np.array(foot_positions_body_frame_buffer)
    foot_positions_contact_frame_array = np.array(foot_positions_contact_frame_buffer)
    power_array = joint_torques_array * joint_velocities_array
    total_robot_mass = float(env.unwrapped.scene["robot"].data.default_mass.sum().item())

    np_data_file = os.path.join(plots_directory, "sim_data.npz")
    np.savez(
        np_data_file,
        sim_times=sim_times,
        reset_times=np.array(reset_times),
        reward_array=reward_array,
        joint_positions_array=joint_positions_array,
        joint_velocities_array=joint_velocities_array,
        joint_torques_array=joint_torques_array,
        joint_accelerations_array=joint_accelerations_array,
        action_rate_array=action_rate_array,
        contact_forces_array=contact_forces_array,
        base_position_array=base_position_array,
        base_orientation_array=base_orientation_array,
        base_linear_velocity_array=base_linear_velocity_array,
        base_angular_velocity_array=base_angular_velocity_array,
        commanded_velocity_array=commanded_velocity_array,
        contact_state_array=contact_state_array,
        height_map_array=np.array(height_map_buffer),
        foot_velocities_world_frame_array=foot_velocities_world_frame_array,
        foot_positions_world_frame_array=foot_positions_world_frame_array,
        foot_positions_body_frame_array=foot_positions_body_frame_array,
        foot_positions_contact_frame_array=foot_positions_contact_frame_array,
        power_array=power_array,
        foot_labels=np.array(foot_labels),
        joint_names=np.array(joint_names),
        total_robot_mass=total_robot_mass,
        constraint_bounds=np.array(constraint_bounds, dtype=object),
    )

    arrays_dict = {
        "joint_positions"      : joint_positions_array,
        "joint_velocities"     : joint_velocities_array,
        "joint_torques"        : joint_torques_array,
        "joint_accelerations"  : joint_accelerations_array,
        "action_rate"          : action_rate_array,
        "contact_forces"       : contact_forces_array,
        "base_position"        : base_position_array,
        "base_orientation"     : base_orientation_array,
        "base_linear_velocity" : base_linear_velocity_array,
        "base_angular_velocity": base_angular_velocity_array,
        "commanded_velocity"   : commanded_velocity_array,
        "contact_state"        : contact_state_array,
        "foot_velocities_world_frame": foot_velocities_world_frame_array,
        "foot_positions_world_frame" : foot_positions_world_frame_array,
        "foot_positions_body"  : foot_positions_body_frame_array,
        "foot_positions_contact_frame": foot_positions_contact_frame_array,
        "power_array"          : power_array,
        "reward"               : reward_array,
    }

    constants_dict = {
        "step_dt"          : step_dt,
        "joint_names"      : joint_names,
        "foot_labels"      : foot_labels,
        "constraint_bounds": constraint_bounds,
        "total_robot_mass" : total_robot_mass
    }

    # --- masks ---
    T            = total_sim_steps
    all_indices  = np.arange(T)
    random_timestep_mask  = all_indices < args.random_sim_step_length

    scenario_masks = {}
    for k, (scenario_tag, *_ ) in enumerate(fixed_command_scenarios):
        start = args.random_sim_step_length + k * fixed_command_sim_steps
        end   = start + fixed_command_sim_steps     # exclusive
        scenario_masks[scenario_tag] = (all_indices >= start) & (all_indices < end)

    overall_metrics  = compute_summary_metrics(np.ones(T, bool), reset_steps, arrays_dict, constants_dict)
    random_metrics   = compute_summary_metrics(random_timestep_mask, reset_steps, arrays_dict, constants_dict)
    scenario_metrics = {
        tag: compute_summary_metrics(msk, reset_steps, arrays_dict, constants_dict)
        for tag, msk in scenario_masks.items()
    }

    summary_metrics = dict(overall_metrics) # start with overall block
    # metrics per segment / fixed command scenario
    summary_metrics["random_simulation_steps_metrics"] = random_metrics
    summary_metrics["fixed_command_scenarios_metrics"] = scenario_metrics
    summary_metrics.update({
        "random_sim_steps"        : args.random_sim_step_length,
        "total_sim_steps"         : total_sim_steps,
        "seed"                    : env_cfg.seed,
        "used_checkpoint_path"    : checkpoint_path,
        "fixed_command_scenarios" : fixed_command_scenarios,
    })

    summary_path = os.path.join(eval_base_dir, "metrics_summary.json")
    with open(summary_path, 'w') as summary_file:
        json.dump(summary_metrics, summary_file, indent=4, default=lambda o: o.tolist()) # lambda for torch/numpy tensor conversion or any other nested objects
    # print(json.dumps(summary_metrics, indent=4, default=lambda o: o.tolist()))

    raw_frames = env.render()
    for frame in raw_frames:
        ffmpeg_process.stdin.write(frame.tobytes())
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    for k, (scenario_tag, *_rest) in enumerate(fixed_command_scenarios):
        start = args.random_sim_step_length + k * fixed_command_sim_steps
        end = start + fixed_command_sim_steps # End is exclusive
        subdir = f"scenario_{scenario_tag}"
        run_generate_plots(start, end, subdir, plots_directory=plots_directory, sim_data_file_path=np_data_file)
    random_rc = run_generate_plots(start_step=0, end_step=args.random_sim_step_length, subdir="random_simulation_steps", plots_directory=plots_directory, sim_data_file_path=np_data_file)
    overall_rc = run_generate_plots(start_step=0, end_step=total_sim_steps, subdir="overall", plots_directory=plots_directory, sim_data_file_path=np_data_file)

    if any(rc != 0 for rc in (overall_rc, random_rc)):
        print("[WARN] At least one generate_plots.py run returned a non-zero exit code.")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
	main()