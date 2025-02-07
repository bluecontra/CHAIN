import os
import time
import random

import tyro
import numpy as np

import gymnasium as gym
from utils.basic_utils import make_env
from configs.deepac_baseline_configs import Args


# Runs policy for a certain number of episodes and returns average episodic return
def eval_random_policy(eval_env, eval_episodes=100):
    episodic_returns = []
    obs, _ = eval_env.reset()

    while len(episodic_returns) < eval_episodes:
        actions = eval_env.action_space.sample()
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

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_name = None

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
        "invest_window_size": args.invest_window_size,
        "invest_interval": args.invest_interval,
        "reg_coef": args.reg_coef,
        "reg_his_idx": args.reg_his_idx,
    }

    print('=' * 50)
    print('Exp setup done. Setup time(s):', time.time() - start_time)
    run_start_time = time.time()

    # Evaluate untrained policy
    init_eval = eval_random_policy(envs4eval)
    print("---------------------------------------")
    print(f"Env: {args.env_id}")
    print(f"T: {0}, Evaluation over {len(init_eval)} episodes: {np.mean(init_eval):.3f}")
    print("---------------------------------------")


    envs.close()
    print('=' * 50)
    print('Run done. Run time(s):', time.time() - run_start_time)
    print('=' * 50)
    print('Exp done. Total time(s):', time.time() - start_time)
