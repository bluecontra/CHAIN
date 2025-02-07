import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import copy

from utils.ppo_basic_utils import make_env, DSAgent, eval_policy


@dataclass
class Args:
    use_policy_ckpt: bool = False

    alg: str = 'ppo'
    scale_up_ratio: int = 1

    """the name of designated algorithm"""
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    eval_freq_timesteps: int = 40000
    """timestep to eval learned policy"""
    seed: int = 1
    """seed of the experiment"""

    gpu_no: str = '0'
    """designate the gpu with corresponding number to run the exp"""
    use_cluster: bool = False
    """0 for using normal physical workstation; 1 for using cluster"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    norm_env: bool = False
    min_lr_frac: float = 0.1

    # Algorithm specific arguments
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__ == "__main__":

    start_time = time.time()

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.alg}_{args.env_id}"
    if args.norm_env:
        run_name += '_ne'
    run_name += f"_s{args.seed}_t{int(time.time())}"
    print("---------------------------------------")
    print(f"Alg: {args.alg}, Env: {args.env_id}, Seed: {args.seed}")
    print(f"Hyperparams: {vars(args)}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./runs"):
        os.makedirs("./runs")
    # if args.save_model and not os.path.exists("./models"):
    #     os.makedirs("./models")

    # NOTE - add wandb & tensorboard logger
    if args.track:
        import wandb
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            name=run_name,
            project='policy_churn_mujcoo_ppo',
            # entity='',
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Set device
    if not args.use_cluster:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
        # - cpu occupation num
        cpu_num = 2
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu': print('- Note cpu is being used now.')

    # init env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, args.capture_video, run_name, args.gamma, args.norm_env)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # NOTE - Eval env only resets once (this is somewhat special due to the implementation in CleanRL)
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + 1234, 0, args.capture_video, run_name, args.gamma, args.norm_env)])

    agent = DSAgent(envs, scale_up_ratio=args.scale_up_ratio).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate / np.sqrt(args.scale_up_ratio), eps=1e-5)

    norm_stats = [copy.deepcopy(e.obs_rms) for e in envs.envs] if args.norm_env else None
    init_eval, init_horizon = eval_policy(agent, eval_envs, norm_stats, eval_episodes=10, device=device)
    # init_eval, init_horizon, next_obs = eval_policy(agent, envs, next_obs, eval_episodes=10, device=device)

    # NOTE - record checkpoint policies
    ckpt_agent = copy.deepcopy(agent)
    ckpt_scores = init_eval

    # NOTE - record historical policies for policy churn reduction
    his_agent_list = [copy.deepcopy(agent)]

    print("---------------------------------------")
    print(f"T: {0}, Evaluation over {len(init_eval)} episodes. "
          f"Scores: {np.mean(init_eval):.3f}, Horizons: {np.mean(init_horizon):.3f}")
    print("---------------------------------------")
    # evaluations, horizons = [np.mean(init_eval)], [np.mean(init_horizon)]
    writer.add_scalar("charts/Eval", np.mean(init_eval), 0)
    writer.add_scalar("charts/Eval_Horizon", np.mean(init_horizon), 0)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    update_cnt = 0
    eval_cnt = 0
    for iteration in range(1, args.num_iterations + 1):
        running_evals = []

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            frac = max(frac, args.min_lr_frac)
            lrnow = frac * args.learning_rate / np.sqrt(args.scale_up_ratio)
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        # NOTE - record running evals
                        running_evals.append(info["episode"]["r"])

        # NOTE - record checkpoint policies, replace if the current policy is better
        if np.mean(running_evals) > np.mean(ckpt_scores):
            ckpt_agent = copy.deepcopy(agent)
            ckpt_scores = running_evals

        # Evaluate episode
        if (global_step // args.eval_freq_timesteps) > eval_cnt:
            eval_cnt += 1
            # NOTE - use checkpoint policy for stable evaluation
            eval_agent = ckpt_agent if args.use_policy_ckpt else agent
            norm_stats = [copy.deepcopy(e.obs_rms) for e in envs.envs] if args.norm_env else None
            cur_eval, cur_horizon = eval_policy(eval_agent, eval_envs, norm_stats, eval_episodes=10, device=device)
            # NOTE - update the score of checkpoint policy? No, we should not use this information

            print("---------------------------------------")
            print(f"T: {global_step}, Evaluation over {len(cur_eval)} episodes. "
                  f"Scores: {np.mean(cur_eval):.3f}, Horizons: {np.mean(cur_horizon):.3f}")
            print("---------------------------------------")
            # evaluations.append(np.mean(cur_eval))
            # horizons.append(np.mean(cur_horizon))
            writer.add_scalar("charts/Eval", np.mean(cur_eval), eval_cnt)
            writer.add_scalar("charts/Eval_Horizon", np.mean(cur_horizon), eval_cnt)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        b_ref_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            np.random.shuffle(b_ref_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                update_cnt += 1

                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_ref_inds = b_ref_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # NOTE - Churn reduction loss
                if len(his_agent_list) > 2:
                    ref_agent = his_agent_list[-2]
                    with torch.no_grad():
                        cur_action_means = agent.actor_mean(b_obs[mb_ref_inds])
                        ref_action_means = ref_agent.actor_mean(b_obs[mb_ref_inds])
                    policy_churn = ((cur_action_means - ref_action_means) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # NOTE - add historical policies into the buffer
                his_agent_list.append(copy.deepcopy(agent))
                # NOTE - a list with 10 policies is enough
                his_agent_list = his_agent_list[-10:]

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/policy_churn", policy_churn.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    envs.close()
    writer.close()
