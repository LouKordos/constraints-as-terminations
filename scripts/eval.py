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
import threading
import subprocess
import os
import re
import yaml
from typing import Dict, Tuple, Optional, List

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Determinism

eval_script_path = os.path.dirname(os.path.abspath(__file__))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Play an RL agent with detailed logging.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="ABSOLUTE path to directory containing model checkpoints and params.")
    parser.add_argument("--eval_checkpoint", type=str, default=None,
                        help="Optionally specify the model save checkpoint number instead of automatically using the last saved one.")
    parser.add_argument("--random_sim_step_length", type=int, default=4000,
                        help="Number of steps to run with random commands and spawn points. Standardized tests like standing and walking forward will always run.")
    parser.add_argument("--disable_fabric", action="store_true", default=False,
                        help="Disable fabric and use USD I/O operations.")
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

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric
    )
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
    trajectories_directory = os.path.join(eval_base_dir, "trajectories")
    os.makedirs(plots_directory, exist_ok=True)
    os.makedirs(trajectories_directory, exist_ok=True)

    fixed_command_sim_steps = 600
    fixed_command_scenarios = [
        ("stand_still", torch.tensor([0.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("pure_spin", torch.tensor([0.0, 0.0, 0.5], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_x_flat_terrain", torch.tensor([0.1, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_y_flat_terrain", torch.tensor([0.0, 0.1, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_x_flat_terrain", torch.tensor([1.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_y_flat_terrain", torch.tensor([0.0, 1.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("walk_x_uneven_terrain", torch.tensor([0.5, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("walk_y_uneven_terrain", torch.tensor([0.0, 0.5, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_x_uneven_terrain", torch.tensor([1.0, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        # ("fast_walk_y_uneven_terrain", torch.tensor([0.0, 1.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_diagonal_uneven_terrain", torch.tensor([1.0, 1.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0],  device=device))),
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
    print_dict(video_configuration, nesting=4)
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

    # --- joint → leg-row + column index ---------------------------------
    # recognised prefixes for the four legs
    _leg_prefixes = [
        ("FL_",),                     	# 0 = front-left
        ("FR_",),                     	# 1 = front-right
        ("RL_", "HL_"),               	# 2 = rear/​hind-left
        ("RR_", "HR_"),               	# 3 = rear/​hind-right
    ]

    # -- accepted substrings for each joint "column" --------
    JOINT_TYPE_SYNONYMS = {
        0: ("hip",  "haa"),      	# 0th column  = hip  / HAA  (ab-ad)
        1: ("thigh","hfe"),      	# 1st column  = thigh/ HFE  (flex-ext)
        2: ("calf", "kfe"),      	# 2nd column  = calf / KFE  (knee flex-ext)
    }

    def _column_from_name(jname: str) -> int | None:
        """
        Return 0,1,2 depending on which set of synonyms the name matches, else None.
        """
        low = jname.lower()
        for col, keys in JOINT_TYPE_SYNONYMS.items():
            if any(k in low for k in keys):
                return col
        raise ValueError("Could not determine joint row/col for plotting based on names")
    
    # build two look-up tables
    leg_row  	= [None] * len(joint_names) # index 0-3
    leg_col  	= [None] * len(joint_names) # index 0-2
    foot_from_joint = [None] * len(joint_names)

    for j, name in enumerate(joint_names):
        # find which leg
        for row, prefixes in enumerate(_leg_prefixes):
            if any(name.startswith(p) for p in prefixes):
                leg_row[j] = row
                foot_from_joint[j] = row # same index as contact_state columns
                break
        # find column inside that leg
        leg_col[j] = _column_from_name(name)

    print("joint names: ", joint_names)
    print("leg_row:", leg_row)
    print("leg_col:", leg_col)
    print("foot_from_joint:", foot_from_joint)

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
    foot_positions_buffer = []

    cumulative_unscaled_raw_reward = 0.0
    reset_steps = []

    observations, info = env.reset()
    policy_observation = observations['policy']
    previous_action = None

    video_output_path = os.path.join(eval_base_dir, f"{os.path.basename(eval_base_dir)}_run_{os.path.basename(args.run_dir)}.mp4")
    frame_storage_interval = 1
    ffmpeg_cmd = ["ffmpeg", "-y", "-hwaccel", "cuda", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frame_width}x{frame_height}", "-framerate", str(frame_rate), "-i", "pipe:0", "-c:v", "hevc_nvenc", "-preset", "slow", "-movflags", "+use_metadata_tags", "-metadata", f"env_name={env_name}", video_output_path]
    ffmpeg_process_log_path = os.path.join(eval_base_dir, "ffmpeg_encode.log")
    with open(ffmpeg_process_log_path, "w") as ffmpeg_process_logfile:
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=ffmpeg_process_logfile, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, bufsize=4*1024*1024)
    # enlarge kernel pipe so the writer thread gets big, contiguous chunks
    # fcntl.fcntl(ffmpeg_proc.stdin, fcntl.F_SETPIPE_SZ, 16*1024*1024)    
    frame_q = Queue(maxsize=frame_storage_interval + 2)
    def frame_writer(q, pipe):
        while True:
            buf = q.get()
            if buf is None:
                break
            pipe.write(buf)
        pipe.close()
    threading.Thread(target=frame_writer, args=(frame_q, ffmpeg_process.stdin), daemon=True).start()

    for t in tqdm(range(total_sim_steps)):
        # These should only ever run at the end of eval / after random sampling because set_fixed_velocity_command breaks random command sampling!
        if t >= args.random_sim_step_length and t % fixed_command_sim_steps == 0:
            scenario = fixed_command_scenarios[int(t-args.random_sim_step_length) // fixed_command_sim_steps]
            fixed_command = scenario[1]
            spawn_point_pos, spawn_point_quat = scenario[2]
            print(f"Resetting env and setting fixed command + spawn point for scenario={scenario}...")
            obs, _ = env.reset()
            teleport_robot(spawn_point_pos, spawn_point_quat)
            set_fixed_velocity_command(fixed_command)
            policy_observation = obs["policy"]

        with torch.no_grad():
            action, _, _, _ = policy_agent.get_action_and_value(policy_agent.obs_rms(policy_observation, update=False))
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
        
        # print(f"terminated={terminated}, truncated={truncated}")
        # Because of CaT, terminated is actually a nonzero probability instead of a boolean, always remember this
        if env.unwrapped.episode_length_buf[0].item() == 0 and t > 0:
            reset_steps.append(t)

        if t % frame_storage_interval == 0:
            raw_frames = env.render()
            for frame in raw_frames:
                frame_q.put(frame.tobytes())

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

        world_position = scene_robot_data.root_link_pos_w[0].cpu().numpy() # world-frame for env 0
        # origin = env.unwrapped.scene.terrain.env_origins[0].cpu().numpy() # terrain origin for env 0
        # relative_position = world_position + origin # position relative to terrain
        # quat_xyzw = scene_data.root_quat_w[0]
        # quat_wxyz = torch.cat([quat_xyzw[3:], quat_xyzw[:3]]) # reorder to (w,x,y,z)
        quat_wxyz = root_quat_w(env.unwrapped, make_quat_unique=True, asset_cfg=SceneEntityCfg("robot"))
        roll_t, pitch_t, yaw_t = euler_xyz_from_quat(quat_wxyz)
        # Isaaclab returns 0 to 2pi euler angles, so small negative angles wrap around to ~2pi.
        # Rescale to [-pi, pi] for better plotting
        def convert_to_signed_angle(a): return (a + np.pi) % (2*np.pi) - np.pi

        # IMPORTANT: THESE ANGLES ARE NOT UNWRAPPED YET, HAPPENS AFTER THE FULL ROLLOUT
        roll = convert_to_signed_angle(roll_t.cpu().numpy().item())
        pitch = convert_to_signed_angle(pitch_t.cpu().numpy().item())
        yaw = convert_to_signed_angle(yaw_t.cpu().numpy().item())
        # print(f"UNWRAPPED roll={roll}\tUNWRAPPED pitch={pitch}")

        base_position_buffer.append(world_position)
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
        
        foot_positions = np.stack([
            scene_robot_data.body_link_pos_w[0, scene_robot_data.body_names.index(link)].cpu().numpy()
            for link in foot_links
        ])
        # Important: This does not account for body rotation, use a rotation/transformation matrix for proper body frame transformation
        foot_positions_buffer.append(foot_positions - world_position)

        cumulative_unscaled_raw_reward += reward.mean().item()
        policy_observation = next_observation['policy']

    print("Sim loop done, converting and saving recorded sim data...")

    # Convert buffers to numpy arrays
    time_indices = np.arange(total_sim_steps)
    sim_times = time_indices * step_dt
    reset_times = [i * step_dt for i in reset_steps]
    print("Reset times: ", reset_times)
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

    # --- energy & cost-of-transport ------------------------------------
    # instantaneous joint power: torque (Nm) * angular velocity (rad/s)
    power_array = joint_torques_array * joint_velocities_array  # shape (T, J)
    # cumulative per-joint energy (J) via trapezoidal/integral: ∑ |P| * dt
    energy_per_joint = np.cumsum(np.abs(power_array), axis=0) * step_dt  # shape (T, J)
    # cumulative total energy (J) across all joints
    combined_energy = np.cumsum(np.abs(power_array).sum(axis=1)) * step_dt  # shape (T,)
    # 1) Compute raw displacements
    raw_distance = np.linalg.norm(np.diff(base_position_array, axis=0), axis=1)
    # 2) Build a mask of “real” steps
    mask = np.ones_like(raw_distance, dtype=bool)
    # Zero-out after environment resets
    for reset_step in reset_steps:
        # reset_step is the time-step index at which reset happened
        # the jump appears at displacement index reset_step-1
        if reset_step > 0 and reset_step-1 < len(mask):
            mask[reset_step-1] = False
    # Zero-out after teleports
    for k in range(len(fixed_command_scenarios)):
        t0 = args.random_sim_step_length + k * fixed_command_sim_steps
        if t0 > 0 and t0-1 < len(mask):
            mask[t0-1] = False
    # 3) Sum only the valid displacements
    true_distance = float(raw_distance[mask].sum())
    # robot mass (kg): you can pull from your env if available, else hard-code/supply
    total_robot_mass = float(env.unwrapped.scene["robot"].data.default_mass.sum().item())

    # instantaneous cost of transport (dimensionless): P_total / (m g v)
    instantaneous_speed = np.linalg.norm(base_linear_velocity_array[:, :2], axis=1)
    # avoid division by zero
    cost_of_transport_time_series = combined_energy.copy() # placeholder
    with np.errstate(divide='ignore', invalid='ignore'):
        cost_of_transport_time_series = (np.abs(power_array).sum(axis=1) / (total_robot_mass * 9.81 * instantaneous_speed + 1e-12))

    # average cost of transport over the whole run
    mean_cost_of_transport = float(np.nanmean(cost_of_transport_time_series))

    target_lin_vel_x = commanded_velocity_array[:, 0]
    actual_lin_vel_x = base_linear_velocity_array[:, 0]
    lin_vel_x_error = target_lin_vel_x - actual_lin_vel_x
    lin_vel_x_rms = np.sqrt(np.mean(lin_vel_x_error**2))

    target_lin_vel_y = commanded_velocity_array[:, 1]
    actual_lin_vel_y = base_linear_velocity_array[:, 1]
    lin_vel_y_error = target_lin_vel_y - actual_lin_vel_y
    lin_vel_y_rms = np.sqrt(np.mean(lin_vel_y_error**2))

    target_yaw_rate = commanded_velocity_array[:, 2]
    actual_yaw_rate = base_angular_velocity_array[:, 2]
    yaw_rate_error = target_yaw_rate - actual_yaw_rate
    ang_vel_z_rms = np.sqrt(np.mean(yaw_rate_error**2))

    # Compute constraint violations
    print("Constraint bounds:", constraint_bounds)
    constraint_violations_percent = {}
    constraint_violation_data_map = {
        'joint_velocity': joint_velocities_array,
        'joint_torque': joint_torques_array,
        'joint_acceleration': joint_accelerations_array,
        'action_rate': action_rate_array,
        'foot_contact_force': contact_forces_array.reshape(total_sim_steps, -1).mean(axis=1),
        'joint_position': joint_positions_array, # shape (T, J)
        # air_time: 1 if foot is in the air, 0 if in contact → shape (T, F)
        'air_time': (1 - contact_state_array).astype(float)
    }

    metrics = {
        'position': joint_positions_array,
        'velocity':	joint_velocities_array,
        'acceleration': joint_accelerations_array,
        'torque': joint_torques_array,
        'action_rate': action_rate_array,
        'energy': energy_per_joint,
        'power': power_array,
    }

    for term, (lb, ub) in constraint_bounds.items():
        metric = constraint_violation_data_map.get(term)
        if metric is None:
            # skip globals not in map
            print(f"Skipping metric for term={term}")
            continue

        # build a violation mask: same shape as metric
        #  - True wherever metric > ub
        #  - True wherever metric < lb (only if lb is not None)
        if metric.ndim == 2:
            # per‐joint: shape (T, J)
            above_ub = (ub is not None) * (metric > ub)
            below_lb = (lb is not None) * (metric < lb)
            violation_mask = above_ub | below_lb

            # percent over time for each joint
            violation_percentage = violation_mask.mean(axis=0) * 100
            constraint_violations_percent[term] = dict(zip(joint_names, violation_percentage.tolist()))
        else:
            # 1D time-series
            above_ub = (ub is not None) and (metric > ub)
            below_lb = (lb is not None) and (metric < lb)
            violation_mask = above_ub | below_lb
            constraint_violations_percent[term] = float(violation_mask.mean() * 100)

    per_joint_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for j, jn in enumerate(joint_names):
        per_joint_summary[jn] = {}
        for metric_name, data in metrics.items():
            # data is shape (T, J); select column j → shape (T,)
            joint_column = data[:, j]
            per_joint_summary[jn][metric_name] = {
                'mean': float(joint_column.mean()),
                'min': float(joint_column.min()),
                'max': float(joint_column.max()),
                'median': float(np.median(joint_column)),
                '90th_percentile': float(np.percentile(joint_column, 90)),
                '99th_percentile': float(np.percentile(joint_column, 99)),
                'stddev': float(np.std(joint_column))
            }

    # --- build per‐foot air‐time summaries ----------------------------------
    # contact_state_array: shape (T, F); 1=in contact, 0=in air
    air_segments_per_foot: Dict[str, List[float]] = {label: [] for label in foot_labels}

    for i, label in enumerate(foot_labels):
        in_contact = contact_state_array[:, i].astype(bool)
        start = None
        segments: List[Tuple[int,int]] = []
        # find all [start, end) intervals where in_contact==False
        for t, c in enumerate(in_contact):
            if not c and start is None:
                start = t
            elif c and start is not None:
                segments.append((start, t))
                start = None
        # if still airborne at end
        if start is not None:
            segments.append((start, len(in_contact)))
        # convert to durations in seconds
        durations = [(end - begin) * step_dt for begin, end in segments]
        air_segments_per_foot[label] = durations

    # now compute summary stats for each foot
    air_time_summary: Dict[str, Dict[str, float]] = {}
    for label, durations in air_segments_per_foot.items():
        arr = np.array(durations, dtype=np.float64)
        if arr.size == 0:
            # no air‐time segments: fill zeros
            air_time_summary[label] = {k: 0.0 for k in 
                ('mean','min','max','median','90th_percentile', '99th_percentile', 'stddev')}
        else:
            air_time_summary[label] = {
                'mean':            float(arr.mean()),
                'min':             float(arr.min()),
                'max':             float(arr.max()),
                'median':          float(np.median(arr)),
                '90th_percentile': float(np.percentile(arr, 90)),
                '99th_percentile': float(np.percentile(arr, 99)),
                'stddev': float(np.std(arr))
            }

    # --- build contact-forces summary ------------------------------------
    # contact_forces_array is shape (T, F)
    contact_force_summary: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(foot_labels):
        joint_column = contact_forces_array[:, i]
        contact_force_summary[label] = {
            'mean':             float(joint_column.mean()),
            'min':              float(joint_column.min()),
            'max':              float(joint_column.max()),
            'median':           float(np.median(joint_column)),
            '90th_percentile':  float(np.percentile(joint_column, 90)),
            '99th_percentile':  float(np.percentile(joint_column, 99)),
            'stddev': float(np.std(joint_column))
        }

    # Save summary metrics
    summary_metrics = {
        'cumulative_unscaled_raw_reward': cumulative_unscaled_raw_reward,
        'cumulative_reward_divided_by_cost_of_transport': cumulative_unscaled_raw_reward / mean_cost_of_transport,
        'cumulative_reward_divided_by_cost_of_transport_and_sim_time': cumulative_unscaled_raw_reward / (mean_cost_of_transport * total_sim_steps * step_dt),
        'base_linear_velocity_x_rms_error': float(lin_vel_x_rms),
        'base_linear_velocity_y_rms_error': float(lin_vel_y_rms),
        'base_angular_velocity_z_rms_error': float(ang_vel_z_rms),
        'per_joint_summary': per_joint_summary,
        'air_time_seconds_per_foot': air_time_summary,
        'contact_force_summary': contact_force_summary,
        'energy_consumption_per_joint': {
            jn: float(energy_per_joint[-1, j])
            for j, jn in enumerate(joint_names)
        },
        'total_energy_consumption': float(combined_energy[-1]),
        'mean_cost_of_transport': mean_cost_of_transport,
        'constraint_violations_percent': constraint_violations_percent,
        'fixed_command_scenarios': fixed_command_scenarios,
        'random_sim_steps': args.random_sim_step_length,
        'total_sim_steps': total_sim_steps,
        'seed': env_cfg.seed,
        'used_checkpoint_path': checkpoint_path
    }
    summary_path = os.path.join(eval_base_dir, "metrics_summary.json")
    with open(summary_path, 'w') as summary_file:
        json.dump(summary_metrics, summary_file, indent=4, default=lambda o: o.tolist()) # lambda for torch tensor conversion or any other nested objects
    print(json.dumps(summary_metrics, indent=4, default=lambda o: o.tolist()))

    np_data_file = os.path.join(plots_directory, "sim_data.npz")
    np.savez(
        np_data_file,
        sim_times=sim_times,
        reset_times=np.array(reset_times),
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
        foot_positions_array=np.array(foot_positions_buffer),
        combined_energy=combined_energy,
        energy_per_joint=energy_per_joint,
        power_array=power_array,
        cost_of_transport_time_series=cost_of_transport_time_series,
        foot_labels=np.array(foot_labels),
        leg_row=np.array(leg_row),
        leg_col=np.array(leg_col),
        joint_names=np.array(joint_names),
        air_segments_per_foot=np.array(air_segments_per_foot, dtype=object),
        constraint_bounds=np.array(constraint_bounds, dtype=object),
    )

    print("Starting plot generation...")
    plot_process_log_path  = os.path.join(eval_base_dir, "generate_plots.log")
    generate_plots_script_path = os.path.join(eval_script_path, "generate_plots.py")
    plot_cmd = [
        "python", generate_plots_script_path,
        "--data", np_data_file,
        "--interactive"
    ]
    with open(plot_process_log_path, "w") as plot_process_logfile:
        plot_proc = subprocess.Popen(plot_cmd, stdout=plot_process_logfile, stderr=subprocess.STDOUT)

    # Write trajectories to JSONL
    trajectories_path = os.path.join(trajectories_directory, 'trajectory.jsonl')
    with open(trajectories_path, 'w') as traj_file:
        for idx in range(total_sim_steps):
            record = {
                'step': int(idx),
                'joint_positions': joint_positions_buffer[idx].tolist(),
                'joint_velocities': joint_velocities_buffer[idx].tolist(),
                'joint_accelerations': joint_accelerations_buffer[idx].tolist(),
                'action_rate': action_rate_buffer[idx].tolist(),
                'contact_forces': contact_forces_buffer[idx].tolist(),
                'joint_torques': joint_torques_buffer[idx].tolist(),
                'base_position': base_position_buffer[idx].tolist(),
                'base_orientation': base_orientation_buffer[idx],
                'base_linear_velocity': base_linear_velocity_buffer[idx].tolist(),
                'base_angular_velocity': base_angular_velocity_buffer[idx].tolist(),
                'commanded_velocity': (commanded_velocity_buffer[idx].cpu().numpy().tolist() if isinstance(commanded_velocity_buffer[idx], torch.Tensor) else (commanded_velocity_buffer[idx].tolist() if isinstance(commanded_velocity_buffer[idx], np.ndarray) else commanded_velocity_buffer[idx])),
                'contact_state': contact_state_buffer[idx].tolist(),
                'height_map': height_map_buffer[idx].tolist(),
                'foot_positions': foot_positions_buffer[idx].tolist(),
            }
            traj_file.write(json.dumps(record, indent=4, default=lambda o: o.tolist()) + "\n")

    start = time.time()
    raw_frames = env.render()
    for frame in raw_frames:
        frame_q.put(frame.tobytes())
    end = time.time()
    
    frame_q.put(None)
    ffmpeg_process.wait()

    plot_proc.wait()
    print(f"[INFO] Plot generation exited with return code {plot_proc.returncode}. See generate_plots.log below:")
    with open(plot_process_log_path, "r") as plot_process_logfile:
        print(plot_process_logfile.read())

    env.close()
    simulation_app.close()

if __name__ == "__main__":
	main()