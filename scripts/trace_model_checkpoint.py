import torch
import os
import gymnasium as gym
import argparse
from isaaclab.app import AppLauncher
os.environ["OMNICLIENT_HUB_MODE"] = "disabled"

# Generates pt file including weights and model structure for C++ inference

def parse_arguments():
    parser = argparse.ArgumentParser(description="Use a temporary isaaclab env to generate a traced pytorch model that can be used in C++ inference.")
    parser.add_argument("--task", type=str, required=True, help="Isaac Lab environment task, used to extract observation dimension")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved model checkpoint weights. Tracing combines model structure and weights, so each checkpoint needs a separate invocation of this script.")

    arguments = parser.parse_args()
    arguments.headless = True
    arguments.num_envs = 1
    arguments.device = "cpu"
    arguments.disable_fabric = "True"
    return arguments

def main():
    args = parse_arguments()
    print(args)

    script_path = os.path.dirname(os.path.abspath(__file__))
    traced_checkpoints_dir = os.path.join(os.path.dirname(script_path), "sim2real", "traced_checkpoints")
    print(f"traced_checkpoints_dir={traced_checkpoints_dir}")
    os.makedirs(traced_checkpoints_dir, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    from isaaclab_tasks.utils import parse_env_cfg
    from cat_envs.tasks.utils.cleanrl.ppo import Agent
    
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

    agent = Agent(env).to("cpu")
    agent.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))

    example_data = torch.randn(1, *env.unwrapped.single_observation_space["policy"].shape)
    print("Single observation space shape:", *env.unwrapped.single_observation_space["policy"].shape)
    traced_actor = torch.jit.trace(agent.forward, example_data)

    checkpoint_path = args.checkpoint_path
    timestamp = os.path.basename(os.path.dirname(checkpoint_path))
    filename = os.path.basename(checkpoint_path)
    stem, _ = os.path.splitext(filename)
    # Extract numeric ID after the first underscore
    model_id = stem.split("_", 1)[1] if "_" in stem else stem
    traced_filename = f"{timestamp}_{model_id}_traced_deterministic.pt"
    traced_module_path = os.path.join(traced_checkpoints_dir, traced_filename)
    traced_actor.save(traced_module_path)
    print(f"Saved to {traced_module_path}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()