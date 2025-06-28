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

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    from isaaclab_tasks.utils import parse_env_cfg
    from cat_envs.tasks.utils.cleanrl.ppo import Agent
    
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

    agent = Agent(env).to("cpu")
    agent.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))

    example_data = torch.randn(1, *env.unwrapped.single_observation_space["policy"].shape)
    traced_actor = torch.jit.trace(agent.forward, example_data)
    deterministic_path = os.path.abspath(args.checkpoint_path + ".traced_deterministic")
    print(f"Saved to {deterministic_path}")
    traced_actor.save(deterministic_path)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()