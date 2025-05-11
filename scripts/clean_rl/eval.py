import argparse
import os
import glob
import json
import yaml
import numpy as np
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Disable interactive display so saves don't pop up windows immediately
plt.ioff()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Play an RL agent with detailed logging.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="ABSOLUTE path to directory containing model checkpoints and params.")
    parser.add_argument("--eval_checkpoint", type=str, default=None,
                        help="Optionally specify the model save checkpoint number instead of automatically using the last saved one.")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Record videos during playback.")
    parser.add_argument("--random_sim_step_length", type=int, default=4000,
                        help="Number of steps to run with random commands and spawn points. Standardized tests like standing and walking forward will always run.")
    parser.add_argument("--disable_fabric", action="store_true", default=False,
                        help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, required=True,
                        help="Name of the task/environment.")
    # Good seeds for eval: 44, 46, 49
    parser.add_argument("--seed", type=int, required=False, default=46, help="Seed for numpy, torch, env, terrain, terrain generator etc.. Good seeds for eval are 44, 46, 49")
    import cli_args  # isort: skip
    cli_args.add_clean_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    arguments = parser.parse_args()
    if arguments.video:
        arguments.enable_cameras = True
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

import os
import re
import yaml

def load_constraint_limits(params_directory: str, joint_names: list[str]) -> dict:
    """
    Load all constraint limits from env.yaml.

    - Scalar constraints (e.g. torque, velocity) stay as {constraint_name: limit}.
    - Joint-position constraints are expanded into per-joint entries:
        {joint_name: limit, …}
    """
    yaml_file = os.path.join(params_directory, 'env.yaml')
    raw_text = open(yaml_file, 'r').read()

    # Strip out any Python-specific tags (!!python/…)
    cleaned_text = re.sub(r'!!python\S*', '', raw_text)
    config = yaml.safe_load(cleaned_text)

    limits = {}
    constraints_cfg = config.get('constraints', config.get('scene', {}).get('constraints', {}))

    JOINT_FUNCS = {
        'cat_envs.tasks.utils.cat.constraints:joint_position',
        'cat_envs.tasks.utils.cat.constraints:joint_position_when_moving_forward'
    }

    for term_name, term_config in constraints_cfg.items():
        if not isinstance(term_config, dict):
            continue

        func = term_config.get('func', '')
        params = term_config.get('params', {})
        if not isinstance(params, dict) or 'limit' not in params:
            continue

        limit = params['limit']
        names = params.get('names', [])

        # If this is one the joint position constraints, expand per name
        if func in JOINT_FUNCS:
            # treat each pattern in names[] as a regex and expand it
            for pattern in names:
                regex = re.compile(f"^{pattern}$")
                for jn in joint_names:
                    if regex.match(jn):
                        limits[jn] = limit
        else:
            limits[term_name] = float(limit)

    return limits

def create_height_map_animation(height_map_sequence: np.ndarray, foot_positions_sequence: np.ndarray, output_path: str, fps: int = 30, sensor=None):
    """
    Create and save an animation of the height map over time,
    reshaping the 1D ray output into the 2D grid based on the sensor's pattern_cfg.
    """
    if sensor is None:
        raise ValueError("RayCaster sensor instance must be provided to determine grid dimensions.")

    # Get the ray start positions for the first environment
    ray_starts = sensor.ray_starts[0].cpu().numpy()  # shape: (R, 3)
    x_coords = ray_starts[:, 0]
    y_coords = ray_starts[:, 1]
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    Nx = len(unique_x)
    Ny = len(unique_y)
    ordering = sensor.cfg.pattern_cfg.ordering

    fig, ax = plt.subplots()

    # Initial frame
    frame0 = height_map_sequence[0]
    if ordering == 'xy':
        grid0 = frame0.reshape((Ny, Nx))
    else:  # 'yx'
        grid0 = frame0.reshape((Nx, Ny)).T
    heatmap = ax.imshow(grid0, origin='lower')

    scatter = ax.scatter(
        foot_positions_sequence[0][:, 0],
        foot_positions_sequence[0][:, 1],
        c='red',
        s=20
    )
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Terrain Height')

    def animate_frame(frame_index):
        frame = height_map_sequence[frame_index]
        if ordering == 'xy':
            grid = frame.reshape((Ny, Nx))
        else:
            grid = frame.reshape((Nx, Ny)).T
        heatmap.set_data(grid)
        scatter.set_offsets(foot_positions_sequence[frame_index][:, :2])
        return heatmap, scatter

    animation_obj = animation.FuncAnimation(
        fig,
        animate_frame,
        frames=len(height_map_sequence),
        blit=True
    )
    animation_obj.save(output_path, fps=fps)
    plt.close()

def plot_gait_diagram(contact_states: np.ndarray, sim_times: np.ndarray, reset_times: list[float], foot_labels: list[str], output_path: str, spacing: float = 1.0) -> plt.Figure:
    T, F = contact_states.shape
    assert sim_times.shape[0] == T, "sim_times length must match contact_states"

    fig, ax = plt.subplots(figsize=(12, F * 1.2))
    ax.set_xlabel('Time [s]')
    ax.set_title('Gait Diagram with Air Times')

    for reset_time in reset_times:
            ax.axvline(x=reset_time, linestyle=":", linewidth=1, color="orange", label='reset' if reset_time == reset_times[0] else None)

    for i, label in enumerate(foot_labels):
        y0 = i * spacing
        in_contact = contact_states[:, i].astype(bool)

        # Contact segments
        contact_segs = []
        start = None
        for t, c in enumerate(in_contact):
            if c and start is None:
                start = t
            elif not c and start is not None:
                contact_segs.append((start, t))
                start = None
        if start is not None:
            contact_segs.append((start, T))

        # Plot contact
        for s, e in contact_segs:
            ax.fill_between(sim_times[s:e], y0, y0 + spacing * 0.8, step='post', alpha=0.8, label=label if s == contact_segs[0][0] else None)
            t_start = sim_times[s]
            t_end   = sim_times[e - 1]
            duration = t_end - t_start
            t_mid = 0.5 * (t_start + t_end)
            y_text = y0 + spacing * 0.4
            ax.text(t_mid, y_text, f"{duration:.3f}s", ha='center', va='center', color='white', fontsize=12)

        # Air segments
        air_segs = []
        if contact_segs and contact_segs[0][0] > 0:
            air_segs.append((0, contact_segs[0][0]))
        for (s0, e0), (s1, e1) in zip(contact_segs, contact_segs[1:]):
            air_segs.append((e0, s1))
        if contact_segs and contact_segs[-1][1] < T:
            air_segs.append((contact_segs[-1][1], T))

        # Annotate durations
        for a, b in air_segs:
            t_start = sim_times[a]
            t_end   = sim_times[b - 1]
            duration = t_end - t_start
            t_mid = 0.5 * (t_start + t_end)
            ax.text(t_mid, y0 + spacing * 0.4, f"{duration:.3f}s", ha='center', va='center', fontsize=12)

    ax.set_yticks([i * spacing for i in range(F)])
    ax.set_yticklabels(foot_labels)
    ax.set_ylim(-spacing * 0.5, (F - 1) * spacing + spacing)
    ax.grid(axis='x', linestyle=':')
    ax.legend(loc='upper right', ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    return fig

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

    eval_base_dir = os.path.join(args.run_dir, f"eval_checkpoint_{os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]}_seed_{args.seed}")

    # Create output directories
    plots_directory = os.path.join(eval_base_dir, "plots")
    trajectories_directory = os.path.join(eval_base_dir, "trajectories")
    os.makedirs(plots_directory, exist_ok=True)
    os.makedirs(trajectories_directory, exist_ok=True)

    print(f"[INFO] Loading model from: {checkpoint_path}")
    log_parent = os.path.dirname(checkpoint_path)
    model_state = torch.load(checkpoint_path, weights_only=True)

    # Launch Isaac Lab environment
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
    env_cfg.seed = args.seed
    env_cfg.scene.terrain.terrain_generator.seed = env_cfg.seed
    env_cfg.scene.terrain.seed = env_cfg.seed
    # Viewer setup
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (0.0, -3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.sim.render.rendering_mode = "quality"
    env_cfg.viewer.resolution = (1920, 1080)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step_dt = env_cfg.sim.dt * env_cfg.decimation # Physics run at higher frequency, action is applied `decimation` physics-steps, but video uses env steps as unit
    random_sim_end = args.random_sim_step_length * step_dt # end of random-policy phase
    fixed_command_sim_steps = 500
    fixed_command_sim_time = fixed_command_sim_steps * step_dt

    fixed_command_scenarios = [
        ("stand_still", torch.tensor([0.0, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("pure_spin", torch.tensor([0.0, 0.0, 0.5], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_x_flat_terrain", torch.tensor([0.1, 0.0, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("slow_walk_y_flat_terrain", torch.tensor([0.0, 0.1, 0.0], device=device), (torch.tensor([30, 30.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_x_uneven_terrain", torch.tensor([0.5, 0.0, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
        ("fast_walk_y_uneven_terrain", torch.tensor([0.0, 0.5, 0.0], device=device), (torch.tensor([0, 0.0, 0.4], device=device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))),
    ]

    total_sim_steps = args.random_sim_step_length + len(fixed_command_scenarios) * fixed_command_sim_steps
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        video_configuration = {
            "video_folder": eval_base_dir,
            "name_prefix": os.path.basename(eval_base_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": total_sim_steps - 1, # One more to stop video recording and generate video before plot generation
            "disable_logger": True,
        }
        print_dict(video_configuration, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_configuration)

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
    constraint_limits = load_constraint_limits(os.path.join(args.run_dir, 'params'), joint_names)

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
    leg_row  	= [None] * len(joint_names)   # index 0-3
    leg_col  	= [None] * len(joint_names)   # index 0-2
    foot_from_joint = [None] * len(joint_names)

    for j, name in enumerate(joint_names):
        # find which leg
        for row, prefixes in enumerate(_leg_prefixes):
            if any(name.startswith(p) for p in prefixes):
                leg_row[j] = row
                foot_from_joint[j] = row        	# same index as contact_state columns
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

    cumulative_reward = 0.0
    reset_steps = []

    observations, info = env.reset()
    policy_observation = observations['policy']
    previous_action = None

    for t in tqdm(range(total_sim_steps)):
        if t == total_sim_steps - len(fixed_command_scenarios) - 1:
            print("Saving video, sim and code execution will freeze for a while.")

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
        # print(f"terminated={terminated}, truncated={truncated}")
        # Because of CaT, terminated is actually a nonzero probability instead of a boolean, always remember this
        if env.unwrapped.episode_length_buf[0].item() == 0 and t > 0:
            reset_steps.append(t)

        scene_data = env.unwrapped.scene['robot'].data

        joint_positions = scene_data.joint_pos[0].cpu().numpy()
        joint_velocities = scene_data.joint_vel[0].cpu().numpy()
        joint_positions_buffer.append(joint_positions)
        joint_velocities_buffer.append(joint_velocities)

        contact_sensors = env.unwrapped.scene['contact_forces']
        feet_ids,_ = contact_sensors.find_bodies(foot_links, preserve_order=True)
        net_forces = contact_sensors.data.net_forces_w_history
        forces_history = net_forces[0].cpu()[:, feet_ids, :]
        force_magnitudes = torch.norm(forces_history,dim=-1)
        max_per_foot = force_magnitudes.max(dim=0)[0].numpy()
        contact_forces_buffer.append(max_per_foot)

        torques = scene_data.applied_torque[0].cpu().numpy()
        joint_torques_buffer.append(torques)

        accelerations = scene_data.joint_acc[0].cpu().numpy()
        joint_accelerations_buffer.append(accelerations)

        action_np = action.cpu().numpy()
        if previous_action is None:
            action_rate = np.zeros_like(action_np)
        else:
            action_rate = np.abs(action_np - previous_action)
        action_rate_buffer.append(action_rate)
        previous_action = action_np

        world_position = scene_data.root_pos_w[0].cpu().numpy() # world-frame for env 0
        origin = env.unwrapped.scene.terrain.env_origins[0].cpu().numpy() # terrain origin for env 0
        relative_position = world_position + origin # position relative to terrain
        # quat_xyzw = scene_data.root_quat_w[0]
        # quat_wxyz = torch.cat([quat_xyzw[3:], quat_xyzw[:3]]) # reorder to (w,x,y,z)
        quat_wxyz = root_quat_w(env.unwrapped, make_quat_unique=True, asset_cfg=SceneEntityCfg("robot"))
        roll_t, pitch_t, yaw_t = euler_xyz_from_quat(quat_wxyz)
        roll = roll_t.cpu().numpy().item()
        pitch = pitch_t.cpu().numpy().item()
        yaw = yaw_t.cpu().numpy().item()
        # roll = np.unwrap(roll_t.cpu().numpy(), axis=0).item()
        # pitch = np.unwrap(pitch_t.cpu().numpy(), axis=0).item()
        # yaw = np.unwrap(yaw_t.cpu().numpy(), axis=0).item()
        base_position_buffer.append(relative_position)
        base_orientation_buffer.append([yaw, pitch, roll])

        linear_velocity = scene_data.root_lin_vel_w[0].cpu().numpy()
        angular_velocity = scene_data.root_ang_vel_w[0].cpu().numpy()
        base_linear_velocity_buffer.append(linear_velocity)
        base_angular_velocity_buffer.append(angular_velocity)

        commanded_velocity_buffer.append(env.unwrapped.command_manager.get_command("base_velocity").clone()) # three components: lin_vel_x, lin_vel_y, ang_vel_z
        contact_state = (max_per_foot > 0).astype(int)
        contact_state_buffer.append(contact_state)

        height_map_sequence = height_map_grid(env.unwrapped, SceneEntityCfg(name="ray_caster")).cpu().numpy()
        height_map_buffer.append(height_map_sequence[0])
        
        foot_positions = np.stack([
            scene_data.body_link_pos_w[0, scene_data.body_names.index(link)].cpu().numpy()
            for link in foot_links
        ])
        # Important: This does not account for body rotation, use a rotation/transformation matrix for proper body frame transformation
        foot_positions_buffer.append(foot_positions - world_position)

        cumulative_reward += reward.mean().item()
        policy_observation = next_observation['policy']

    print("Converting and saving recorded sim data...")

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
    base_orientation_array = np.vstack(base_orientation_buffer)
    base_linear_velocity_array = np.vstack(base_linear_velocity_buffer)
    base_angular_velocity_array = np.vstack(base_angular_velocity_buffer)
    contact_state_array = np.vstack(contact_state_buffer)
    commanded_velocity_array = np.vstack([cv.cpu().numpy() if isinstance(cv, torch.Tensor) else np.asarray(cv) for cv in commanded_velocity_buffer])

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
    print("Constraint limits:", constraint_limits)
    violations_percent = {}
    for term, limit in constraint_limits.items():
        data_map = {
            'joint_velocity': 	joint_velocities_array,
            'joint_torque':   	joint_torques_array,
            'joint_acceleration': joint_accelerations_array,
            'action_rate':    	action_rate_array,
            'foot_contact_force': contact_forces_array.reshape(total_sim_steps, -1).mean(axis=1),
        }

        for term, limit in constraint_limits.items():
            metric = data_map.get(term)
            if metric is None:
                continue

            # vector metrics → percent per joint
            if metric.ndim == 2:
                percent_per_joint = (metric > limit).mean(axis=0) * 100
                violations_percent[term] = dict(zip(joint_names, percent_per_joint.tolist()))
            else:
                violations_percent[term] = float((metric > limit).mean() * 100)

    # Save summary metrics
    summary_metrics = {
        'cumulative_reward': cumulative_reward,
        'base_linear_velocity_x_rms_error': float(lin_vel_x_rms),
        'base_linear_velocity_y_rms_error': float(lin_vel_y_rms),
        'base_angular_velocity_z_rms_error': float(ang_vel_z_rms),
        'violations_percent': violations_percent,
        'fixed_command_scenarios': fixed_command_scenarios,
        'random_sim_steps': args.random_sim_step_length,
        'total_sim_steps': total_sim_steps,
        'seed': args.seed
    }
    summary_path = os.path.join(eval_base_dir, "metrics_summary.txt")
    with open(summary_path, 'w') as summary_file:
        json.dump(summary_metrics, summary_file, indent=2, default=lambda o: o.tolist()) # lambda for torch tensor conversion or any other nested objects
    print(json.dumps(summary_metrics, indent=2, default=lambda o: o.tolist()))

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
            traj_file.write(json.dumps(record) + "\n")

    # Individual plots for each metric
    figs = []
    linewidth = 1

    print("Data saved, starting plot generation...")

    def draw_limit(ax, term):
        limit = constraint_limits.get(term)

        if limit is None:
            print(f"Constraint limit for {term} is None, cannot plot the limit.")
            return

        label_str = f"{term}_limit={limit}"
        existing_labels = ax.get_legend_handles_labels()[1]

        if label_str not in existing_labels:
            ax.axhline(limit, linestyle='--', linewidth=1, color='red', label=label_str)
            if term != "foot_contact_force": # Those are only positive
                ax.axhline(-limit, linestyle='--', linewidth=1, color='red')
        else:
            ax.axhline(limit, linestyle='--', linewidth=1, color='red')
            ax.axhline(-limit, linestyle='--', linewidth=1, color='red')

    def draw_resets(ax):
        for reset_time in reset_times:
            ax.axvline(x=reset_time, linestyle=":", linewidth=1, color="orange", label='reset' if reset_time == reset_times[0] else None)

    # ----------------------------------------
    # 1) Foot contact‐force per‐foot in 2×2 grid
    # ----------------------------------------
    foot_labels = ['front left','front right','rear left','rear right']
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, contact_forces_array[:, i], label='force', linewidth=linewidth)
        draw_limit(ax, "foot_contact_force")
        draw_resets(ax)
        # identify contiguous contact intervals
        in_contact = contact_state_array[:, i].astype(bool)
        segments = []
        start_idx = None
        for idx, val in enumerate(in_contact):
            if val and start_idx is None:
                start_idx = idx
            elif not val and start_idx is not None:
                segments.append((start_idx, idx))
                start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(sim_times)))

        # shade each interval across the full y-axis
        first = True
        for s, e in segments:
            ax.axvspan(
                sim_times[s],
                sim_times[e-1],
                facecolor='gray',
                alpha=0.3,
                label='in contact' if first else None
            )
            first = False

        ax.set_title(foot_labels[i])
        ax.set_ylabel('Force [N]')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_directory, 'foot_contact_force_each.pdf'), dpi=600)

    # ------------------------------------------------
    # 2) Joint metrics: 4×3 grid and overview per metric
    # ------------------------------------------------
    metrics = {
        'positions': 	joint_positions_array,
        'velocities':	joint_velocities_array,
        'accelerations': joint_accelerations_array,
        'torques':   	joint_torques_array,
        'action_rates': action_rate_array
    }

    metric_to_constraint_term_mapping = {
        'positions': 	None,
        'velocities':	'joint_velocity',
        'accelerations': 'joint_acceleration',
        'torques':   	'joint_torque',
        'action_rates': 'action_rate'
    }

    # helper: map joint index → foot index (0=FL,1=FR,2=RL,3=RR)
    foot_from_joint = []
    for name in joint_names:
        if   name.startswith('FL_'): foot_from_joint.append(0)
        elif name.startswith('FR_'): foot_from_joint.append(1)
        elif name.startswith('RL_') or name.startswith('HL_'): foot_from_joint.append(2)
        elif name.startswith('RR_') or name.startswith('HR_'): foot_from_joint.append(3)
        else:                    	 foot_from_joint.append(None) # unlikely

    def get_leg_linestyle(joint_name):
            if joint_name.startswith("FL"):
                return "solid"
            elif joint_name.startswith("FR"):
                return "dotted"
            elif joint_name.startswith("RL") or name.startswith("HL"):
                return "dashed"
            elif joint_name.startswith("RR") or name.startswith("HR"):
                return "dashdot"

    for name, data in metrics.items():
        # 4×3 grid of separate joint plots
        fig, axes = plt.subplots(4, 3, sharex=True, figsize=(18, 12))
        for j in range(len(joint_names)):
            row, col = leg_row[j], leg_col[j]
            if row is None or col is None:
                raise ValueError("Could not determine joint row/col for plotting based on names")
            ax = axes[row, col]
            ax.plot(sim_times, data[:, j], linewidth=linewidth)

            if name == 'positions':
                # use the joint’s own limit
                draw_limit(ax, joint_names[j])
            else:
                draw_limit(ax, metric_to_constraint_term_mapping[name])
            draw_resets(ax)

            foot_idx = foot_from_joint[j]
            if foot_idx is not None: # shade when that foot is in contact
                in_contact = contact_state_array[:, foot_idx].astype(bool)
                # contiguous segments for that foot
                start = None
                for t, val in enumerate(in_contact):
                    if val and start is None:  start = t
                    if (not val or t == len(in_contact)-1) and start is not None:
                        end = t if not val else t+1
                        ax.axvspan(sim_times[start], sim_times[end-1],
                                    facecolor='gray', alpha=0.5)
                        start = None
            ax.set_title(joint_names[j])
            ax.set_ylabel("Joint " + (name.replace('_', ' ')[:-1] if name != 'velocities' else 'velocity'))
        axes[-1, 0].set_xlabel('Time / s')
        fig.tight_layout()
        fig.savefig(os.path.join(plots_directory, f'joint_{name}_grid.pdf'), dpi=600)

        # overview: all joints in one plot
        fig, ax = plt.subplots(figsize=(12, 6))
        for j in range(data.shape[1]):
            ax.plot(sim_times, data[:, j], label=joint_names[j], linewidth=linewidth, linestyle=get_leg_linestyle(joint_names[j]))
        if name == 'positions': # Handle each joint position limit separetely
            for jn in joint_names:
                draw_limit(ax, jn)
        else:
            draw_limit(ax, metric_to_constraint_term_mapping[name])
        draw_resets(ax)
        ax.set_xlabel('Time / s')
        ax.set_ylabel("Joint " + (name.replace('_', ' ')[:-1] if name != 'velocities' else 'velocity'))
        ax.legend(loc='upper right', ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_directory, f'joint_{name}_overview.pdf'), dpi=600)

    # Combined subplots for base position, orientation, linear and angular velocity
    FIGSIZE = (16, 9)
    fig_bp, axes_bp = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, axis_label in enumerate(['X', 'Y', 'Z']):
        axes_bp[i].plot(sim_times, base_position_array[:, i], label=f'position_{axis_label}', linewidth=linewidth)
        draw_resets(axes_bp[i])
        axes_bp[i].set_ylabel(f'Position {axis_label}')
        axes_bp[i].legend()
        axes_bp[i].grid(True)
    axes_bp[0].set_title('Base Position Subplots')
    axes_bp[-1].set_xlabel('Time / s')
    fig_bp.tight_layout()
    fig_bp.savefig(os.path.join(plots_directory, 'base_position_subplots.pdf'), dpi=600)
    figs.append(fig_bp)

    fig_bo, axes_bo = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, orient_label in enumerate(['Yaw', 'Pitch', 'Roll']):
        axes_bo[i].plot(sim_times, base_orientation_array[:, i], label=orient_label, linewidth=linewidth)
        draw_resets(axes_bo[i])
        axes_bo[i].set_ylabel(orient_label)
        axes_bo[i].legend()
        axes_bo[i].grid(True)
    axes_bo[0].set_title('Base Orientation Subplots')
    axes_bo[-1].set_xlabel('Time / s')
    fig_bo.tight_layout()
    fig_bo.savefig(os.path.join(plots_directory, 'base_orientation_subplots.pdf'), dpi=600)
    figs.append(fig_bo)

    fig_blv, axes_blv = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate(['VX', 'VY', 'VZ']):
        axes_blv[i].plot(sim_times, base_linear_velocity_array[:, i], label=vel_label, linewidth=linewidth)
        if vel_label != "VZ": # Command format for UniformVelocityCommandCfg as used here is lin_vel_x, lin_vel_y, and ang_vel_z, so the third component belongs into another plot
            axes_blv[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f"cmd_{vel_label}", linewidth=linewidth, color="black")
        draw_resets(axes_blv[i])
        axes_blv[i].set_ylabel(vel_label)
        axes_blv[i].legend()
        axes_blv[i].grid(True)
    axes_blv[0].set_title('Base Linear Velocity Subplots')
    axes_blv[-1].set_xlabel('Time / s')
    fig_blv.tight_layout()
    fig_blv.savefig(os.path.join(plots_directory, 'base_linear_velocity_subplots.pdf'), dpi=600)
    figs.append(fig_blv)

    fig_bav, axes_bav = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate(['WX', 'WY', 'WZ']):
        axes_bav[i].plot(sim_times, base_angular_velocity_array[:, i], label=vel_label, linewidth=linewidth)
        if i == 2: # Only plot angular velocity around z, this is is the only angular velocity that is actually controlled.
            axes_bav[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f"cmd_{vel_label}", linewidth=linewidth, color="black")
        draw_resets(axes_bav[i])
        axes_bav[i].set_ylabel(vel_label)
        axes_bav[i].legend()
        axes_bav[i].grid(True)
    axes_bav[0].set_title('Base Angular Velocity Subplots')
    axes_bav[-1].set_xlabel('Time / s')
    fig_bav.tight_layout()
    fig_bav.savefig(os.path.join(plots_directory, 'base_angular_velocity_subplots.pdf'), dpi=600)
    figs.append(fig_bav)

    fig_overview, overview_axes = plt.subplots(2, 2, figsize=(20, 16))
    cats = [base_position_array, base_orientation_array,
            base_linear_velocity_array, base_angular_velocity_array]
    titles = ['Base Position', 'Base Orientation', 'Base Linear Velocity', 'Base Angular Velocity']
    labels = [['X', 'Y', 'Z'], ['Yaw', 'Pitch', 'Roll'], ['VX', 'VY', 'VZ'], ['WX', 'WY', 'WZ']]
    for ax, data_array, title, axis_labels in zip(overview_axes.flatten(), cats, titles, labels):
        for i, lbl in enumerate(axis_labels):
            ax.plot(sim_times, data_array[:, i], label=lbl, linewidth=linewidth)
            draw_resets(ax)

            # Wrong, need to only plot commanded_velocity_array[2] for angular velociy in angular vel plot and [0:2] 
            # if title == "Base Linear Velocity":
            #     ax.plot(sim_times, commanded_velocity_array[:, i], linestyle="--", label=f"cmd_{lbl}", linewidth=linewidth, color="black")
            # elif title == "Base Angular Velocity":
            #     ax.plot(sim_times, commanded_velocity_array[:, i+3], linestyle='--', label=f"cmd_{lbl}", linewidth=linewidth, color="black")
        ax.set_title(title)
        ax.set_xlabel('Time / s')
        ax.legend()
        ax.grid(True)
    fig_overview.tight_layout()
    fig_overview.savefig(os.path.join(plots_directory, 'base_overview.pdf'), dpi=600)
    figs.append(fig_overview)

    # Height map animation and gait diagram
    # create_height_map_animation(np.array(height_map_buffer), np.array(foot_positions_buffer), os.path.join(plots_directory, 'height_map.mp4'), sensor=env.unwrapped.scene["ray_caster"])
    figs.append(plot_gait_diagram(np.array(contact_state_buffer), sim_times, reset_times, foot_labels, os.path.join(plots_directory, 'gait_diagram.pdf'), spacing=1.0))

    plt.ion()
    plt.show(block=True)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
	main()