import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def get_model_params(model):
    model_params = []
    with torch.no_grad():
        for params in model.parameters():
            model_params.append(params.data.flatten())
    return torch.cat(model_params)


# Runs policy for a certain number of episodes and returns average episodic return
def eval_policy(eval_agent, eval_env, norm_stats, eval_episodes=10, device=None):
    # NOTE - load norm stats (important)
    if norm_stats is not None:
        for e, s in zip(eval_env.envs, norm_stats):
            # NOTE - important, to make stats assignment take effect really
            # e.obs_rms = s
            e.env.env.env.obs_rms = s

    episode_returns, episode_lengths = [], []
    next_obs, _ = eval_env.reset()
    while len(episode_returns) < eval_episodes:
        action = eval_agent.select_action(torch.Tensor(next_obs).to(device))
        next_obs, _, _, _, infos = eval_env.step(action)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_returns += [info["episode"]["r"]]
                    episode_lengths += [info["episode"]["l"]]

    return episode_returns, episode_lengths


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, scale_up_ratio=1):
        super().__init__()
        self.scale_up_ratio = scale_up_ratio
        self.hidden_dim = 256 * scale_up_ratio
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def select_action(self, obs):
        with torch.no_grad():
            action_mean = self.actor_mean(obs)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()

        return action.cpu().numpy()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

