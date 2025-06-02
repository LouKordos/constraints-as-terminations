import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from functools import partial
print = partial(print, flush=True) # For cluster runs

class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-08):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.ones(()))

        self.epsilon = epsilon

    def forward(self, obs, update=True):
        if update:
            self.update(obs)

        return (obs - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, correction=0, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.running_mean, self.running_var, self.count = (
            update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                batch_mean,
                batch_var,
                batch_count,
            )
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        SINGLE_OBSERVATION_SPACE = envs.unwrapped.single_observation_space["policy"].shape
        SINGLE_ACTION_SPACE = envs.unwrapped.single_action_space.shape
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(SINGLE_ACTION_SPACE)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(SINGLE_ACTION_SPACE)))

        self.obs_rms = RunningMeanStd(shape=SINGLE_OBSERVATION_SPACE)
        self.value_rms = RunningMeanStd(shape=())

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, use_deterministic_policy=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if use_deterministic_policy:
                action = action_mean
            else:
                action = probs.sample()
        return (action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_std)


def PPO(envs, ppo_cfg, run_path):
    print(f"Env in PPO function:\n{os.environ}")
    print("env seed in ppo.py=", envs.unwrapped.cfg.seed)
    import random
    random.seed(envs.unwrapped.cfg.seed)
    np.random.seed(envs.unwrapped.cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(envs.unwrapped.cfg.seed)
    torch.cuda.manual_seed_all(envs.unwrapped.cfg.seed)

    if ppo_cfg.logger == "wandb":
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter

        # Replace project name with timestamp + experiment name for easier cross referencing. This also corrects the timestamp so that project name and run_path use the same one
        ppo_cfg.wandb_project = ppo_cfg.experiment_name
        writer = WandbSummaryWriter(log_dir=run_path, flush_secs=10, cfg=ppo_cfg.to_dict())
    elif ppo_cfg.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

        writer = TensorboardSummaryWriter(log_dir=run_path)
    else:
        raise AssertionError("logger type not found")

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    LEARNING_RATE = ppo_cfg.learning_rate
    NUM_STEPS = ppo_cfg.num_steps
    NUM_ITERATIONS = ppo_cfg.num_iterations
    GAMMA = ppo_cfg.gamma
    GAE_LAMBDA = ppo_cfg.gae_lambda
    UPDATES_EPOCHS = ppo_cfg.updates_epochs
    MINIBATCH_SIZE = ppo_cfg.minibatch_size
    CLIP_COEF = ppo_cfg.clip_coef
    ENT_COEF = ppo_cfg.ent_coef
    VF_COEF = ppo_cfg.vf_coef
    MAX_GRAD_NORM = ppo_cfg.max_grad_norm
    NORM_ADV = ppo_cfg.norm_adv
    CLIP_VLOSS = ppo_cfg.clip_vloss
    ANNEAL_LR = ppo_cfg.anneal_lr

    NUM_ENVS = envs.unwrapped.num_envs

    SINGLE_OBSERVATION_SPACE = envs.unwrapped.single_observation_space["policy"].shape
    SINGLE_ACTION_SPACE = envs.unwrapped.single_action_space.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = int(NUM_ENVS * NUM_STEPS)

    STORE_AND_HASH_TENSORS = False
    if STORE_AND_HASH_TENSORS:
        import zarr
        from numcodecs import Blosc
        import hashlib
        snapshot_dir = os.path.join(run_path, "tensor_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        compress = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        zroot = zarr.open_group(os.path.join(snapshot_dir, f"training_data_{os.path.basename(run_path)}.zarr"), mode="w")
        signals = {
            "actions"     : ((0, NUM_ENVS) + SINGLE_ACTION_SPACE        , 'f4'),
            "rewards"     : ((0, NUM_ENVS)                              , 'f4'),
            "obs"         : ((0, NUM_ENVS) + SINGLE_OBSERVATION_SPACE   , 'f4'),
            "obs_rms"     : ((0, NUM_ENVS) + SINGLE_OBSERVATION_SPACE   , 'f4'),
            "next_dones"  : ((0, NUM_ENVS)                              , 'f4'),
            "returns"     : ((0,)                                       , 'f4'),
        }
        datasets = {}
        for name, (shape, dtype) in signals.items():
            datasets[name] = zroot.create_dataset(name, shape=shape, chunks=(128,) + shape[1:], dtype=dtype, compressor=compress, overwrite=True)

        hashers = {name: hashlib.sha512() for name in datasets}
        tensor_zarr_append_interval = 4 # Global training steps!
        zroot.attrs["append_interval"] = tensor_zarr_append_interval

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + SINGLE_OBSERVATION_SPACE, dtype=torch.float).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + SINGLE_ACTION_SPACE, dtype=torch.float).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    true_dones = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    joint_names = envs.unwrapped.scene["robot"].data.joint_names
    num_joints = len(joint_names)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(seed=envs.unwrapped.cfg.seed)[0]["policy"]
    next_obs = agent.obs_rms(next_obs)
    next_done = torch.zeros(NUM_ENVS, dtype=torch.float).to(device)
    next_true_done = torch.zeros(NUM_ENVS, dtype=torch.float).to(device)

    print(f"Starting training for {NUM_ITERATIONS} steps")

    for iteration in range(1, NUM_ITERATIONS + 1):
        ep_infos = []

        if ANNEAL_LR:
            frac = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        action_std_buffer = [] # Buffer over sim steps, then take mean across env and time steps
        # Collecting trajectories for NUM_STEPS before updating the networks
        for step in range(0, NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            true_dones[step] = next_true_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, action_std = agent.get_action_and_value(next_obs)
                action_std_buffer.append(action_std.detach().cpu().numpy())
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, timeouts, info = envs.step(action)
            # next_obs, rewards[step], next_done, timeouts, info = envs.step(torch.rand_like(action) * 0.6 - torch.ones_like(action) * 0.3)
            next_done = next_done.to(torch.float)
            next_obs = next_obs["policy"]

            if STORE_AND_HASH_TENSORS:
                # Store only first time step of each global iteration because otherwise it takes way too long.
                # To be absolutely certain two runs are identical, remove the steps == 0 and decrease tensor_zarr_append_interval, at the cost of slower training.
                if step == 0 and iteration % tensor_zarr_append_interval == 0:
                    zarr_append_start = time.time()
                    np_actions = action.detach().cpu().numpy()
                    np_rewards = rewards[step].detach().cpu().numpy()
                    np_obs = next_obs.detach().cpu().numpy()
                    np_next_dones = next_done.detach().cpu().numpy()
                    # print(f"actions zarr shape={datasets['actions'].shape}\tnp shape (pre_expand)={np_actions.shape}")
                    # print(f"rewards zarr shape={datasets['rewards'].shape}\tnp shape (pre_expand)={np_rewards.shape}")
                    # print(f"obs zarr shape={datasets['obs'].shape}\tnp shape (pre_expand)={np_obs.shape}")
                    # print(f"next_dones zarr shape={datasets['next_dones'].shape}\tnp shape (pre_expand)={np_next_dones.shape}")

                    datasets["actions"].append(np.expand_dims(np_actions, axis=0), axis=0)
                    datasets["rewards"].append(np.expand_dims(np_rewards, axis=0), axis=0)
                    datasets["obs"].append(np.expand_dims(np_obs, axis=0), axis=0)
                    datasets["next_dones"].append(np.expand_dims(np_next_dones, axis=0), axis=0)
                    print(f"iteration={iteration}\tAppending tensors to zarr file took {(time.time() - zarr_append_start):.4f} seconds")

            if torch.any(torch.isnan(next_obs)):
                print("NAN IN OBSERVATION")

            if "episode" in info:
                ep_infos.append(info["episode"])
            elif "log" in info:
                ep_infos.append(info["log"])

            info["true_dones"] = timeouts
            next_obs = agent.obs_rms(next_obs)
            if STORE_AND_HASH_TENSORS:
                if step == 0 and iteration % tensor_zarr_append_interval == 0:
                    np_obs_rms = next_obs.detach().cpu().numpy()
                    datasets["obs_rms"].append(np.expand_dims(np_obs_rms, axis=0), axis=0)

            next_true_done = info["true_dones"].float()
            if "time_outs" in info:
                if info["time_outs"].any():
                    print("time outs", info["time_outs"].sum())
                    exit(0)

        # Logging/Analytics, adapted from rslrl
        for key in ep_infos[0]: # Get keys to iterate over
            infotensor = torch.tensor([], device=device)
            for ep_info in ep_infos: # Iterate over each time step
                # handle scalar and zero dimensional tensor infos
                if key not in ep_info:
                    continue
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(device)))
            value = torch.mean(infotensor)
            if "/" in key:
                writer.add_scalar(key, value, iteration)
            else:
                writer.add_scalar("Episode/" + key, value, iteration)

        # COMMENT OUT TO DISABLE CAT, AND ALSO ADJUST STEP FUNTION FOR SIMULATION!!! CaT: must compute the CaT quantity
        not_dones = 1.0 - dones
        rewards *= not_dones

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    true_nextnonterminal = 1 - next_true_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    true_nextnonterminal = 1 - true_dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (rewards[t] + GAMMA * nextvalues * nextnonterminal * true_nextnonterminal - values[t])
                advantages[t] = lastgaelam = (delta + GAMMA * GAE_LAMBDA * nextnonterminal * true_nextnonterminal * lastgaelam)
            returns = advantages + values

            if STORE_AND_HASH_TENSORS:
                # Uses training iteration steps because it does not run in nested sim step loop
                if iteration % tensor_zarr_append_interval == 0:
                    np_returns = returns.detach().cpu().reshape(-1).numpy()
                    print(f"returns zarr shape={datasets['returns'].shape}\tnp shape={np_returns.shape}")
                    datasets["returns"].append(np_returns, axis=0)

                    hash_update_start = time.time()
                    for name, arr in [("actions", np_actions),
                    ("rewards", np_rewards),
                    ("obs", np_obs),
                    ("obs_rms", np_obs_rms),
                    ("next_dones", np_next_dones),
                    ("returns", np_returns)]:
                        hashers[name].update(arr.tobytes())
                    print(f"iteration={iteration}\tUpdating hashes took {time.time() - hash_update_start} seconds.")
                    
                    flush_consolidate_start = time.time()
                    zarr.consolidate_metadata(zroot.store)
                    # Once again a dirty hack to force flushing...
                    # Zarr v3 offers zroot.flush() but requires python>=3.11 but isaac lab python=3.10...
                    for dirpath, _, filenames in os.walk(zroot.path):
                        for fname in filenames:
                            fpath = os.path.join(dirpath, fname)
                            with open(fpath, 'rb') as f:
                                os.fsync(f.fileno())
                    print(f"zroot flush and metadata consolidation took {(time.time() - flush_consolidate_start):.4f} seconds.")

                    # Use compare_tensor_snapshots.py for torch.allclose between two runs because rounding may influence the hash
                    for name, h in hashers.items():
                        print(f"[snapshot @{global_step}] hash({name}) = {h.hexdigest()}")

        # flatten the batch
        b_obs = obs.reshape((-1,) + SINGLE_OBSERVATION_SPACE)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + SINGLE_ACTION_SPACE)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_values = agent.value_rms(b_values)
        b_returns = agent.value_rms(b_returns)

        sum_pg_loss = sum_entropy_loss = sum_v_loss = sum_surrogate_loss = 0.0

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(UPDATES_EPOCHS):
            b_inds = torch.randperm(BATCH_SIZE, device=device)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                newvalue = agent.value_rms(newvalue, update=False)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -CLIP_COEF, CLIP_COEF,)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                sum_pg_loss += pg_loss
                sum_entropy_loss += entropy_loss
                sum_v_loss += v_loss
                sum_surrogate_loss += loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        num_updates = UPDATES_EPOCHS * BATCH_SIZE / MINIBATCH_SIZE
        writer.add_scalar("Loss/mean_pg_loss", sum_pg_loss / num_updates, iteration)
        writer.add_scalar("Loss/mean_entropy_loss", sum_entropy_loss / num_updates, iteration)
        writer.add_scalar("Loss/mean_v_loss", sum_v_loss / num_updates, iteration)
        writer.add_scalar("Loss/mean_surrogate_loss", sum_surrogate_loss / num_updates, iteration)
        writer.add_scalar("Loss/learning_rate", optimizer.param_groups[0]["lr"], iteration)

        stacked_action_std_np = np.stack(action_std_buffer, axis=0).reshape(-1, num_joints)
        mean_per_joint = stacked_action_std_np.mean(axis=0)
        std_per_joint  = stacked_action_std_np.std(axis=0)
        for j, name in enumerate(joint_names):
            # Sanitize joint name to avoid W&B hierarchy issues (slashes, spaces, etc.)
            sanitized = name.replace("/", "_").replace(" ", "_")
            writer.add_scalar(f"action_std/{sanitized}/mean", mean_per_joint[j], iteration)
            writer.add_scalar(f"action_std/{sanitized}/std", std_per_joint[j], iteration)

        if (iteration + 1) % ppo_cfg.save_interval == 0:
            model_path = f"{run_path}/model_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print("Saved model")
