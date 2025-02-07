import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


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


def make_env(env_id, seed, idx, capture_video, run_name, gamma, norm_env=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if norm_env:
            # NOTE - this is source of why using a separate env for evaluation will be strange,
            #  because the inner normalization stats are different
            env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if norm_env:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
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

    def get_action_and_logprob(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class DSAgent(nn.Module):
    def __init__(self, envs, scale_up_ratio=1, policy_only=False):
        super().__init__()
        self.scale_up_ratio = scale_up_ratio
        if policy_only:
            self.p_hidden_units_num = 256 * self.scale_up_ratio
            self.c_hidden_units_num = 256
        else:
            self.p_hidden_units_num = 256 * self.scale_up_ratio
            self.c_hidden_units_num = 256 * self.scale_up_ratio
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.c_hidden_units_num)),
            nn.Tanh(),
            layer_init(nn.Linear(self.c_hidden_units_num, self.c_hidden_units_num)),
            nn.Tanh(),
            layer_init(nn.Linear(self.c_hidden_units_num, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.p_hidden_units_num)),
            nn.Tanh(),
            layer_init(nn.Linear(self.p_hidden_units_num, self.p_hidden_units_num)),
            nn.Tanh(),
            layer_init(nn.Linear(self.p_hidden_units_num, np.prod(envs.single_action_space.shape)), std=0.01),
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

    def get_action_and_logprob(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class DPSAgent(nn.Module):
    def __init__(self, envs, scale_up_ratio=1, policy_only=False):
        super().__init__()
        self.scale_up_ratio = scale_up_ratio
        self.p_hidden_units_num = 256
        self.c_hidden_units_num = 256
        if policy_only:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.c_hidden_units_num)),
                nn.Tanh(),
                layer_init(nn.Linear(self.c_hidden_units_num, self.c_hidden_units_num)),
                nn.Tanh(),
                layer_init(nn.Linear(self.c_hidden_units_num, 1), std=1.0),
            )
        else:
            c_seq_layers = [
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.c_hidden_units_num)),
                nn.Tanh()
            ]
            for _ in range(self.scale_up_ratio):
                c_seq_layers.append(layer_init(nn.Linear(self.c_hidden_units_num, self.c_hidden_units_num)))
                c_seq_layers.append(nn.Tanh())
            c_seq_layers.append(layer_init(nn.Linear(self.c_hidden_units_num, 1), std=1.0))
            self.critic = nn.Sequential(*c_seq_layers)

        p_seq_layers = [
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.p_hidden_units_num)),
            nn.Tanh()
        ]
        for _ in range(self.scale_up_ratio):
            p_seq_layers.append(layer_init(nn.Linear(self.p_hidden_units_num, self.p_hidden_units_num)))
            p_seq_layers.append(nn.Tanh())
        p_seq_layers.append(layer_init(nn.Linear(self.p_hidden_units_num, np.prod(envs.single_action_space.shape)), std=0.01))
        self.actor_mean = nn.Sequential(*p_seq_layers)
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

    def get_action_and_logprob(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


