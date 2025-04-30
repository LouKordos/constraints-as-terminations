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


# Enable interactive mode for non-blocking display
plt.ion()




def quaternion_to_euler_angles(quaternion: np.ndarray) -> tuple[float, float, float]:
	"""
	Convert a quaternion (x, y, z, w) to Euler angles (yaw, pitch, roll).
	Returns (yaw, pitch, roll) in radians.
	"""
	x, y, z, w = quaternion
	# Roll (x-axis rotation)
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll = np.arctan2(t0, t1)
	# Pitch (y-axis rotation)
	t2 = +2.0 * (w * y - z * x)
	t2 = np.clip(t2, -1.0, 1.0)
	pitch = np.arcsin(t2)
	# Yaw (z-axis rotation)
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw = np.arctan2(t3, t4)
	return yaw, pitch, roll




def parse_arguments():
	parser = argparse.ArgumentParser(description="Play an RL agent with detailed logging.")
	parser.add_argument("--run_dir", type=str, required=True,
                    	help="ABSOLUTE path to directory containing model checkpoints and params.")
	parser.add_argument("--video", action="store_true", default=False,
                    	help="Record videos during playback.")
	parser.add_argument("--video_length", type=int, default=4000,
                    	help="Length of the recorded video (in steps).")
	parser.add_argument("--disable_fabric", action="store_true", default=False,
                    	help="Disable fabric and use USD I/O operations.")
	parser.add_argument("--num_envs", type=int, default=1,
                    	help="Number of environments to simulate.")
	parser.add_argument("--task", type=str, required=True,
                    	help="Name of the task/environment.")
	import cli_args  # isort: skip
	cli_args.add_clean_rl_args(parser)
	AppLauncher.add_app_launcher_args(parser)
	arguments = parser.parse_args()
	if arguments.video:
    	arguments.enable_cameras = True
	return arguments




def get_latest_checkpoint(run_directory: str) -> str:
	checkpoint_files = sorted(glob.glob(os.path.join(run_directory, "*.pt")))
	if not checkpoint_files:
    	raise FileNotFoundError(f"No checkpoint files found in {run_directory}")
	return checkpoint_files[-1]




def load_constraint_limits(params_directory: str) -> dict:
	yaml_file = os.path.join(params_directory, 'env.yaml')
	# Read and clean YAML to remove Python-specific tags (tuples, slices, etc.)
	with open(yaml_file, 'r') as f:
    	raw_text = f.read()
	import re
	# Remove all !!python tags that safe_load cannot parse
	cleaned_text = re.sub('!!python\S*', '', raw_text)
	config = yaml.safe_load(cleaned_text)
	limits = {}
	# Try top-level constraints, fallback under scene
	constraints_cfg = config.get('constraints', config.get('scene', {}).get('constraints', {}))
	for term_name, term_config in constraints_cfg.items():
    	if isinstance(term_config, dict):
        	params = term_config.get('params', {})
        	if isinstance(params, dict) and 'limit' in params:
            	limits[term_name] = params['limit']
	return limits




def plot_time_series_with_contact(time_steps: np.ndarray, values: np.ndarray, limits: dict, label: str, output_path: str, contact_states: np.ndarray | None = None):
	"""
	Plot a time series with optional constraint limit and shaded contact regions.
	"""
	plt.figure()
	plt.plot(time_steps, values, label=label)
	if label in limits:
    	plt.hlines(limits[label], time_steps[0], time_steps[-1], linestyles='--',
               	label=f"{label}_limit={limits[label]}")
	if contact_states is not None:
    	in_contact = contact_states.any(axis=1)
    	segments = []
    	start_idx = None
    	for idx, contact in enumerate(in_contact):
        	if contact and start_idx is None:
            	start_idx = idx
        	if not contact and start_idx is not None:
            	segments.append((start_idx, idx))
            	start_idx = None
    	if start_idx is not None:
        	segments.append((start_idx, len(time_steps)))
    	for seg_idx, (s, e) in enumerate(segments):
        	plt.axvspan(time_steps[s], time_steps[e-1], color='gray', alpha=0.2,
                    	label='contact' if seg_idx == 0 else None)
	plt.title(f"{label.replace('_',' ').title()} vs Time")
	plt.xlabel('Timestep')
	plt.ylabel(label.replace('_',' ').title())
	plt.legend()
	plt.grid(True)
	plt.savefig(output_path, dpi=600)
	plt.show(block=False)




def create_height_map_animation(height_map_sequence: np.ndarray, foot_positions_sequence: np.ndarray, output_path: str, fps: int = 30):
	fig, ax = plt.subplots()
	heatmap = ax.imshow(height_map_sequence[0], origin='lower')
	scatter = ax.scatter(foot_positions_sequence[0][:, 0],
                     	foot_positions_sequence[0][:, 1], c='red', s=20)
	cbar = fig.colorbar(heatmap, ax=ax)
	cbar.set_label('Terrain Height')


	def animate_frame(frame_index):
    	heatmap.set_data(height_map_sequence[frame_index])
    	scatter.set_offsets(foot_positions_sequence[frame_index][:, :2])
    	return heatmap, scatter


	animation_obj = animation.FuncAnimation(fig, animate_frame,
                                        	frames=len(height_map_sequence),
                                        	blit=True)
	animation_obj.save(output_path, fps=fps)
	plt.close()




def plot_gait_diagram(contact_states: np.ndarray, output_path: str):
	plt.figure()
	plt.imshow(contact_states.T, aspect='auto', cmap='gray_r')
	plt.title('Gait Diagram')
	plt.ylabel('Foot Index')
	plt.xlabel('Timestep')
	plt.savefig(output_path, dpi=600)
	plt.show(block=False)




def main():
	args = parse_arguments()
	args.run_dir = os.path.abspath(args.run_dir)


	# Create output directories
	plots_directory = os.path.join(args.run_dir, "test/plots")
	trajectories_directory = os.path.join(args.run_dir, "test/trajectories")
	os.makedirs(plots_directory, exist_ok=True)
	os.makedirs(trajectories_directory, exist_ok=True)


	constraint_limits = load_constraint_limits(os.path.join(args.run_dir, 'params'))


	# Launch Isaac Lab environment
	app_launcher = AppLauncher(args)
	simulation_app = app_launcher.app
	from isaaclab.utils.dict import print_dict
	from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
	from cat_envs.tasks.utils.cleanrl.ppo import Agent
	from cat_envs.tasks.locomotion.velocity.config.solo12.cat_go2_rough_terrain_env_cfg import height_map_grid
	from isaaclab.managers import SceneEntityCfg
	import cat_envs.tasks  # noqa: F401


	env_configuration = parse_env_cfg(
    	args.task,
    	device=args.device,
    	num_envs=args.num_envs,
    	use_fabric=not args.disable_fabric
	)
	# Viewer setup
	env_configuration.viewer.origin_type = "asset_root"
	env_configuration.viewer.asset_name = "robot"
	env_configuration.viewer.eye = (0.0, -3.0, 2.0)
	env_configuration.viewer.lookat = (0.0, 0.0, 0.5)
	env_configuration.sim.render.rendering_mode = "quality"
	env_configuration.viewer.resolution = (1920, 1080)


	import cli_args  # isort: skip
	agent_configuration = cli_args.parse_clean_rl_cfg(args.task, args)


	checkpoint_path = get_latest_checkpoint(args.run_dir)
	print(f"[INFO] Loading model from: {checkpoint_path}")
	log_parent = os.path.dirname(checkpoint_path)
	model_state = torch.load(checkpoint_path)


	env = gym.make(args.task, cfg=env_configuration, render_mode="rgb_array" if args.video else None)
	if args.video:
    	video_configuration = {
        	"video_folder": os.path.join(log_parent, "videos_play"),
        	"step_trigger": lambda step: step == 0,
        	"video_length": args.video_length,
        	"disable_logger": True,
    	}
    	print_dict(video_configuration, nesting=4)
    	env = gym.wrappers.RecordVideo(env, **video_configuration)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	policy_agent = Agent(env).to(device)
	policy_agent.load_state_dict(model_state)


	# Initialize buffers for recording
	joint_positions_buffer = []
	joint_velocities_buffer = []
	contact_forces_buffer = []
	joint_torques_buffer = []
	joint_accelerations_buffer = []
	action_rate_buffer = []
	base_position_buffer = []  # [x, y, z]
	base_orientation_buffer = []  # [yaw, pitch, roll]
	base_linear_velocity_buffer = []  # [vx, vy, vz]
	base_angular_velocity_buffer = []  # [wx, wy, wz]
	commanded_velocity_buffer = []
	contact_state_buffer = []
	height_map_buffer = []
	foot_positions_buffer = []


	cumulative_reward = 0.0
	observations, info = env.reset()
	policy_observation = observations['policy']
	total_steps = args.video_length


	previous_action = None
	for t in range(total_steps):
    	with torch.no_grad():
        	action, _, _, _ = policy_agent.get_action_and_value(policy_agent.obs_rms(policy_observation, update=False))
    	next_observation, reward, done, truncated, info = env.step(action)


    	scene_data = env.unwrapped.scene['robot'].data


    	joint_positions = scene_data.joint_pos[0].cpu().numpy()
    	joint_velocities = scene_data.joint_vel[0].cpu().numpy()
    	joint_positions_buffer.append(joint_positions)
    	joint_velocities_buffer.append(joint_velocities)


    	contact_sensors = env.unwrapped.scene['contact_forces']
    	feet_ids,_ = contact_sensors.find_bodies(['FL_foot','FR_foot','RL_foot','RR_foot'], preserve_order=True)
    	net_forces = contact_sensors.data.net_forces_w_history  # [E, hist, bodies, 3]
    	forces_history = net_forces[0].cpu()[:, feet_ids, :]
    	force_magnitudes = torch.norm(forces_history,dim=-1)  # [hist, nf]
    	max_per_foot = force_magnitudes.max(dim=0)[0].numpy()
    	contact_forces_buffer.append(max_per_foot)


    	torques = scene_data.applied_torque[0].cpu().numpy()
    	joint_torques_buffer.append(torques)


    	accelerations = scene_data.joint_acc[0].cpu().numpy()
    	joint_accelerations_buffer.append(accelerations)


    	action_np = action.cpu().numpy()
    	if previous_action is None:
        	action_rate = 0.0
    	else:
        	action_rate = float(np.linalg.norm(action_np - previous_action))
    	action_rate_buffer.append(action_rate)
    	previous_action = action_np


    	position = scene_data.root_pos_w[0].cpu().numpy()
    	quaternion = scene_data.root_quat_w[0].cpu().numpy()
    	yaw, pitch, roll = quaternion_to_euler_angles(quaternion)
    	base_position_buffer.append(position)
    	base_orientation_buffer.append([yaw, pitch, roll])


    	linear_velocity = scene_data.root_lin_vel_w[0].cpu().numpy()
    	angular_velocity = scene_data.root_ang_vel_w[0].cpu().numpy()
    	base_linear_velocity_buffer.append(linear_velocity)
    	base_angular_velocity_buffer.append(angular_velocity)


    	commanded_velocity_buffer.append(env.unwrapped.command_manager.get_command("base_velocity"))
    	contact_state = (max_per_foot.sum(axis=-1) > 0).astype(int)
    	contact_state_buffer.append(contact_state)


    	height_map_sequence = height_map_grid(env.unwrapped, SceneEntityCfg(name="ray_caster")).cpu().numpy()
    	height_map_buffer.append(height_map_sequence[0])
    	foot_links = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    	foot_positions = np.stack([
        	scene_data.body_link_pos_w[0, scene_data.body_names.index(link)].cpu().numpy()
        	for link in foot_links
    	])
    	foot_positions_buffer.append(foot_positions - position)


    	cumulative_reward += reward.mean().item()
    	policy_observation = next_observation['policy']


	time_steps = np.arange(total_steps)
	joint_positions_array = np.vstack(joint_positions_buffer)
	joint_velocities_array = np.vstack(joint_velocities_buffer)
	joint_torques_array = np.vstack(joint_torques_buffer)
	joint_accelerations_array = np.vstack(joint_accelerations_buffer)
	action_rate_array = np.array(action_rate_buffer)
	contact_forces_array = np.stack(contact_forces_buffer)
	base_position_array = np.vstack(base_position_buffer)
	base_orientation_array = np.vstack(base_orientation_buffer)
	base_linear_velocity_array = np.vstack(base_linear_velocity_buffer)
	base_angular_velocity_array = np.vstack(base_angular_velocity_buffer)
	contact_state_array = np.vstack(contact_state_buffer)


	violations_percent = {}
	for term, limit in constraint_limits.items():
    	data_map = {
        	'joint_velocity': joint_velocities_array,
        	'joint_torque': joint_torques_array,
        	'joint_acceleration': joint_accelerations_array,
        	'action_rate': action_rate_array,
        	'foot_contact_force': contact_forces_array.reshape(total_steps, -1).mean(axis=1)
    	}
    	metric_array = data_map.get(term)
    	if metric_array is not None:
        	percent_violation = (metric_array > limit).sum() / metric_array.size * 100
        	violations_percent[term] = percent_violation


	summary_metrics = {
    	'cumulative_reward': cumulative_reward,
    	'violations_percent': violations_percent
	}
	summary_path = os.path.join(args.run_dir, 'test/metrics_summary.txt')
	with open(summary_path, 'w') as summary_file:
    	json.dump(summary_metrics, summary_file, indent=2)
	print(json.dumps(summary_metrics, indent=2))


	trajectories_path = os.path.join(trajectories_directory, 'trajectory.jsonl')
	with open(trajectories_path, 'w') as traj_file:
    	for idx in range(total_steps):
        	record = {
            	'step': int(idx),
            	'joint_positions': joint_positions_buffer[idx].tolist(),
            	'joint_velocities': joint_velocities_buffer[idx].tolist(),
            	'joint_accelerations': joint_accelerations_buffer[idx].tolist(),
            	'action_rate': float(action_rate_buffer[idx]),
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
	plot_time_series_with_contact(time_steps, joint_positions_array.mean(axis=1),
                              	violations_percent, 'joint_positions',
                              	os.path.join(plots_directory, 'joint_positions.pdf'),
                              	contact_state_array)
	plot_time_series_with_contact(time_steps, joint_velocities_array.mean(axis=1),
                              	violations_percent, 'joint_velocities',
                              	os.path.join(plots_directory, 'joint_velocities.pdf'),
                              	contact_state_array)
	plot_time_series_with_contact(time_steps, joint_accelerations_array.mean(axis=1),
                              	violations_percent, 'joint_accelerations',
                              	os.path.join(plots_directory, 'joint_accelerations.pdf'),
                              	contact_state_array)
	plot_time_series_with_contact(time_steps, joint_torques_array.mean(axis=1),
                              	violations_percent, 'joint_torques',
                              	os.path.join(plots_directory, 'joint_torques.pdf'),
                              	contact_state_array)
	plot_time_series_with_contact(time_steps, action_rate_array,
                              	violations_percent, 'action_rate',
                              	os.path.join(plots_directory, 'action_rate.pdf'),
                              	contact_state_array)
	foot_force_mean = contact_forces_array.reshape(total_steps, -1).mean(axis=1)
	plot_time_series_with_contact(time_steps, foot_force_mean,
                              	violations_percent, 'foot_contact_force',
                              	os.path.join(plots_directory, 'foot_contact_force.pdf'),
                              	contact_state_array)


	# Combined subplots for base position, orientation, linear and angular velocity
	FIGSIZE = (16, 9)
	fig_bp, axes_bp = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
	for i, axis_label in enumerate(['X', 'Y', 'Z']):
    	axes_bp[i].plot(time_steps, base_position_array[:, i], label=f'position_{axis_label}')
    	axes_bp[i].set_ylabel(f'Position {axis_label}')
    	axes_bp[i].legend()
    	axes_bp[i].grid(True)
	axes_bp[0].set_title('Base Position Subplots')
	axes_bp[-1].set_xlabel('Timestep')
	fig_bp.tight_layout()
	fig_bp.savefig(os.path.join(plots_directory, 'base_position_subplots.pdf'), dpi=600)
	plt.show(block=False)


	fig_bo, axes_bo = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
	for i, orient_label in enumerate(['Yaw', 'Pitch', 'Roll']):
    	axes_bo[i].plot(time_steps, base_orientation_array[:, i], label=orient_label)
    	axes_bo[i].set_ylabel(orient_label)
    	axes_bo[i].legend()
    	axes_bo[i].grid(True)
	axes_bo[0].set_title('Base Orientation Subplots')
	axes_bo[-1].set_xlabel('Timestep')
	fig_bo.tight_layout()
	fig_bo.savefig(os.path.join(plots_directory, 'base_orientation_subplots.pdf'), dpi=600)
	plt.show(block=False)


	fig_blv, axes_blv = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
	for i, vel_label in enumerate(['VX', 'VY', 'VZ']):
    	axes_blv[i].plot(time_steps, base_linear_velocity_array[:, i], label=vel_label)
    	axes_blv[i].set_ylabel(vel_label)
    	axes_blv[i].legend()
    	axes_blv[i].grid(True)
	axes_blv[0].set_title('Base Linear Velocity Subplots')
	axes_blv[-1].set_xlabel('Timestep')
	fig_blv.tight_layout()
	fig_blv.savefig(os.path.join(plots_directory, 'base_linear_velocity_subplots.pdf'), dpi=600)
	plt.show(block=False)


	fig_bav, axes_bav = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
	for i, vel_label in enumerate(['WX', 'WY', 'WZ']):
    	axes_bav[i].plot(time_steps, base_angular_velocity_array[:, i], label=vel_label)
    	axes_bav[i].set_ylabel(vel_label)
    	axes_bav[i].legend()
    	axes_bav[i].grid(True)
	axes_bav[0].set_title('Base Angular Velocity Subplots')
	axes_bav[-1].set_xlabel('Timestep')
	fig_bav.tight_layout()
	fig_bav.savefig(os.path.join(plots_directory, 'base_angular_velocity_subplots.pdf'), dpi=600)
	plt.show(block=False)


	fig_overview, overview_axes = plt.subplots(2, 2, figsize=(20, 16))
	cats = [base_position_array, base_orientation_array,
        	base_linear_velocity_array, base_angular_velocity_array]
	titles = ['Base Position', 'Base Orientation', 'Base Linear Velocity', 'Base Angular Velocity']
	labels = [['X', 'Y', 'Z'], ['Yaw', 'Pitch', 'Roll'], ['VX', 'VY', 'VZ'], ['WX', 'WY', 'WZ']]
	for ax, data_array, title, axis_labels in zip(overview_axes.flatten(), cats, titles, labels):
    	for i, lbl in enumerate(axis_labels):
        	ax.plot(time_steps, data_array[:, i], label=lbl)
    	ax.set_title(title)
    	ax.set_xlabel('Timestep')
    	ax.legend()
    	ax.grid(True)
	fig_overview.tight_layout()
	fig_overview.savefig(os.path.join(plots_directory, 'base_overview.pdf'), dpi=600)
	plt.show(block=False)


	# Height map animation and gait diagram
	create_height_map_animation(np.array(height_map_buffer),
                            	np.array(foot_positions_buffer),
                            	os.path.join(plots_directory, 'height_map.mp4'))
	plot_gait_diagram(np.array(contact_state_buffer),
                  	os.path.join(plots_directory, 'gait_diagram.pdf'))


	# Pause to keep figures open until user is ready
	input("Press Enter to close all figure windows...")
	plt.close('all')


	env.close()
	simulation_app.close()




if __name__ == "__main__":
	main()






