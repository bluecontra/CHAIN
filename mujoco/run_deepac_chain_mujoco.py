import os
import time
import random

import tyro
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from utils.basic_utils import make_env
from configs.deepac_chain_configs import Args
from stable_baselines3.common.buffers import ReplayBuffer


# Runs policy for a certain number of episodes and returns average episodic return
def eval_policy(eval_agent, eval_env, eval_episodes=10, is_deterministic=False):
    episodic_returns = []
    obs, _ = eval_env.reset()

    while len(episodic_returns) < eval_episodes:
        if is_deterministic:
            actions = eval_agent.select_action(obs, is_deterministic=True)
        else:
            actions = eval_agent.select_action(obs)
        next_obs, _, _, _, infos = eval_env.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                # print(f"- eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":

    start_time = time.time()

    args = tyro.cli(Args)
    run_name = f"{args.alg}_{args.env_id}"
    if args.alg in ['sac_pcr', 'td3_pcr', 'sac_dcr', 'td3_dcr',]:
        run_name += f"pc{args.reg_coef}"
    if args.alg in ['sac_vcr', 'td3_vcr', 'sac_dcr', 'td3_dcr',]:
        run_name += f"vc{args.v_reg_coef}"
    run_name += f"_invest_w{args.invest_window_size}i{args.invest_interval}_s{args.seed}_t{int(time.time())}"
    print("---------------------------------------")
    print(f"Alg: {args.alg}, Env: {args.env_id}, Seed: {args.seed}")
    print(f"Hyperparams: {vars(args)}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./runs"):
        os.makedirs("./runs")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # NOTE - add wandb & tensorboard logger
    if args.track:
        import wandb
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            name=run_name,
            project='policy_churn_mujoco',
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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device == 'cpu':
        print('- Note cpu is being used now.')

    # Init env
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    envs4eval = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + 1234, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    envs.single_observation_space.dtype = np.float32
    envs4eval.single_observation_space.dtype = np.float32

    # Fold hyperparams of agent
    kwargs = {
        "obs_dim": np.prod(envs.single_observation_space.shape),
        "action_dim": np.prod(envs.single_action_space.shape),
        "action_space": envs.single_action_space,
        "discount": args.gamma,
        "tau": args.tau,
        "device": device,
        "invest_window_size": args.invest_window_size,
        "invest_interval": args.invest_interval,
        "reg_his_idx": args.reg_his_idx,
    }

    # Initialize agent
    if args.alg == "td3_pcr":
        from algs.td3_pcr import TD3_PCR
        # Target policy smoothing is scaled wrt the action scale
        kwargs["lr"] = args.learning_rate
        kwargs["exploration_noise"] = args.exploration_noise
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["policy_freq"] = args.policy_frequency
        kwargs["reg_coef"] = args.reg_coef
        agent = TD3_PCR(**kwargs)
        eval_deterministic_flag = True
    elif args.alg == "sac_pcr":
        from algs.sac_pcr import SAC_PCR
        kwargs["lr"] = args.learning_rate
        kwargs["alpha"] = args.alpha
        kwargs["autotune"] = args.autotune
        kwargs["reg_coef"] = args.reg_coef
        agent = SAC_PCR(**kwargs)
        eval_deterministic_flag = False
    elif args.alg == "td3_vcr":
        from algs.td3_vcr import TD3_VCR
        # Target policy smoothing is scaled wrt the action scale
        kwargs["lr"] = args.learning_rate
        kwargs["exploration_noise"] = args.exploration_noise
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["policy_freq"] = args.policy_frequency
        kwargs["v_reg_coef"] = args.v_reg_coef
        agent = TD3_VCR(**kwargs)
        eval_deterministic_flag = True
    elif args.alg == "sac_vcr":
        from algs.sac_vcr import SAC_VCR
        kwargs["lr"] = args.learning_rate
        kwargs["alpha"] = args.alpha
        kwargs["autotune"] = args.autotune
        kwargs["v_reg_coef"] = args.v_reg_coef
        agent = SAC_VCR(**kwargs)
        eval_deterministic_flag = False
    elif args.alg == "td3_dcr":
        from algs.td3_dcr import TD3_DCR
        # Target policy smoothing is scaled wrt the action scale
        kwargs["lr"] = args.learning_rate
        kwargs["exploration_noise"] = args.exploration_noise
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["policy_freq"] = args.policy_frequency
        kwargs["reg_coef"] = args.reg_coef
        kwargs["v_reg_coef"] = args.v_reg_coef
        agent = TD3_DCR(**kwargs)
        eval_deterministic_flag = True
    elif args.alg == "sac_dcr":
        from algs.sac_dcr import SAC_DCR
        kwargs["lr"] = args.learning_rate
        kwargs["alpha"] = args.alpha
        kwargs["autotune"] = args.autotune
        kwargs["reg_coef"] = args.reg_coef
        kwargs["v_reg_coef"] = args.v_reg_coef
        agent = SAC_DCR(**kwargs)
        eval_deterministic_flag = False
    else:
        print(f'- ERROR: Unknown algorithm name:{args.alg}')
        raise NotImplementedError

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    print('=' * 50)
    print('Exp setup done. Setup time(s):', time.time() - start_time)
    run_start_time = time.time()

    # Evaluate untrained policy
    init_eval = eval_policy(agent, envs4eval, is_deterministic=eval_deterministic_flag)
    print("---------------------------------------")
    print(f"T: {0}, Evaluation over {len(init_eval)} episodes: {np.mean(init_eval):.3f}")
    print("---------------------------------------")
    evaluations = [np.mean(init_eval)]
    writer.add_scalar("charts/Eval", np.mean(init_eval), 0)

    obs, _ = envs.reset(seed=args.seed)
    episode_num = 0

    # This is more convenient than using for t in range(int(args.total_timesteps))
    for t in range(1, int(args.total_timesteps) + 1):
        # Select action randomly or according to policy
        if t <= args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = agent.select_action(obs)

        # Perform action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"T: {t}, Episodic_return: {info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], t)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], t)
                break
            episode_num += 1

        # Save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        # Store data in replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # Train agent after collecting sufficient data
        if t > args.learning_starts:
            if t % 1000 != 0:
                agent.train(rb, args.batch_size)
            else:
                agent.train(rb, args.batch_size, writer=writer, writer_timestamp=t, invest=args.invest)
                print("Step Per Second (SPS):", int(t / (time.time() - run_start_time)))
                writer.add_scalar("charts/SPS", int(t / (time.time() - run_start_time)), t)

        # Evaluate episode
        if t % args.eval_freq_timesteps == 0:
            cur_eval = eval_policy(agent, envs4eval, is_deterministic=eval_deterministic_flag)
            print("---------------------------------------")
            print(f"T: {t}, Evaluation over {len(cur_eval)} episodes: {np.mean(cur_eval):.3f}")
            print("---------------------------------------")
            evaluations.append(np.mean(cur_eval))
            writer.add_scalar("charts/Eval", np.mean(cur_eval), t)

            # np.save(f"./results/{run_name}", evaluations)
            # if args.save_model: agent.save(f"./models/{run_name}")

    envs.close()
    print('=' * 50)
    print('Run done. Run time(s):', time.time() - run_start_time)
    print('=' * 50)
    print('Exp done. Total time(s):', time.time() - start_time)
