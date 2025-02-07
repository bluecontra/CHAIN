import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional
import wandb
from tqdm import trange

import time, copy

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    invest: int = 1
    """if toggled, stats for policy churn investigation will be recorded"""
    invest_window_size: int = 10
    """the window size of step to investigate policy churn"""
    invest_interval: int = 2
    """the interval to keep old policies in invest window"""

    # Experiment
    gpu_no: str = '0'
    use_cluster: int = 0

    alg: str = "awac"
    env_name: str = "pen-human-v1"
    seed: int = 42
    test_seed: int = 69

    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False

    buffer_size: int = 2_000_000
    num_train_ops: int = 1_000_000
    batch_size: int = 256
    eval_frequency: int = 20000
    n_test_episodes: int = 10
    normalize_reward: bool = False

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0

    # Wandb logging
    project: str = "policy_churn_offline"
    group: str = "AWAC"

    device: str = "cuda"
    name: str = "AWAC"

    # def __post_init__(self):
    #     self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
    #     if self.checkpoints_path is not None:
    #         self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def get_mean(self, state: torch.Tensor):
        mean = self._mlp(state)
        return mean

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        action = action_t[0].cpu().numpy()
        return action


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
        device: str = "cpu",
        batch_size=256,
        invest_window_size=10,
        invest_interval=2,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

        self.device = device

        self.batch_size = batch_size
        # NOTE - record historical policies and vfs
        self.invest_window_size = invest_window_size
        self.invest_interval = invest_interval
        self.init_policy = copy.deepcopy(self._actor)
        self.init_critic = copy.deepcopy(self._critic_1)
        self.his_policy_list = [self.init_policy]
        self.his_critic_list = [self.init_critic]

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, rb, invest=False) -> Dict[str, float]:
        batch = rb.sample(self.batch_size)
        batch = [b.to(self.device) for b in batch]
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        
        # NOTE - investigate policy churn and value churn
        if invest:
            ref_batch = rb.sample(self.batch_size)
            ref_batch = [b.to(self.device) for b in ref_batch]
            self.invest_policy_and_value_churn(ref_batch, log_dict=result)

        # NOTE - record historical policies and vfs
        self.his_policy_list.append(copy.deepcopy(self._actor))
        self.his_critic_list.append(copy.deepcopy(self._critic_1))
        self.his_policy_list = self.his_policy_list[-self.invest_window_size * self.invest_interval:]
        self.his_critic_list = self.his_critic_list[-self.invest_window_size * self.invest_interval:]

        return result

    def invest_policy_and_value_churn(self, ref_data, log_dict):
        ref_observations, ref_actions, ref_rewards, ref_next_observations, ref_dones = ref_data

        # fetch the most recent one
        frozen_p = self.his_policy_list[-1]
        frozen_vf = self.his_critic_list[-1]
        ref_policy_list, ref_vf_list = [], []
        for i in range(self.invest_window_size):
            if (i + 1) * self.invest_interval > len(self.his_critic_list):
                break
            ref_policy_list.append(self.his_policy_list[-(i + 1) * self.invest_interval])
            ref_vf_list.append(self.his_critic_list[-(i + 1) * self.invest_interval])

        prior_ref_policy_list, prior_ref_q_list = [], []
        with torch.no_grad():
            # Calculate current greedy actions/qs
            cur_ref_p = self._actor.get_mean(ref_observations)
            cur_ref_q = self._critic_1(ref_observations, ref_actions)

            # Calculate prior update greedy actions/qs for reference states
            prior_update_ref_p = frozen_p.get_mean(ref_observations)
            prior_update_ref_q = frozen_vf(ref_observations, ref_actions)
            prior_ref_policy_list.append(prior_update_ref_p)
            prior_ref_q_list.append(prior_update_ref_q)

            # Calculate historical ref greedy actions/qs
            for p, vf in zip(ref_policy_list, ref_vf_list):
                his_ref_p = p.get_mean(ref_observations)
                his_ref_q = vf(ref_observations, ref_actions)
                prior_ref_policy_list.append(his_ref_p)
                prior_ref_q_list.append(his_ref_q)

            # Calculate outputs of frozen init policy
            init_ref_p = self.init_policy.get_mean(ref_observations)
            init_ref_q = self.init_critic(ref_observations, ref_actions)
            prior_ref_policy_list.append(init_ref_p)
            prior_ref_q_list.append(init_ref_q)

        # Calculate prior-post update diff
        cnt = 0
        for ref_p, ref_q in zip(prior_ref_policy_list, prior_ref_q_list):
            # NOTE - to handle the corner case
            if cnt == 0 and self.invest_interval == 1:
                cnt += 1
                continue

            invest_a_diff = (cur_ref_p - ref_p).mean()
            invest_a_diff_abs = (cur_ref_p - ref_p).abs().mean()
            invest_q_diff = (cur_ref_q - ref_q).mean()
            invest_q_diff_abs = (cur_ref_q - ref_q).abs().mean()

            if cnt == 0:
                write_suffix = ''
            elif cnt == len(prior_ref_q_list) - 1:
                write_suffix = '_init'
            else:
                write_suffix = '_' + str(cnt * self.invest_interval)
            log_dict["invest/a_diff_sig" + write_suffix] = invest_a_diff.item()
            log_dict["invest/a_diff_abs" + write_suffix] = invest_a_diff_abs.item()
            log_dict["invest/q_diff_sig" + write_suffix] = invest_q_diff.item()
            log_dict["invest/q_diff_abs" + write_suffix] = invest_q_diff_abs.item()

            cnt += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def wandb_init(config: dict, run_name) -> None:
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=run_name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env_name)
    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    # Set device
    if config.use_cluster == 0: os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_no
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu': print('- Note cpu is being used now.')

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(device)
    critic_2.to(device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
        device=device,
        invest_window_size=config.invest_window_size,
        invest_interval=config.invest_interval,
        batch_size=config.batch_size,
    )

    run_name = f"{config.alg}_{config.env_name}_invest_w{config.invest_window_size}i{config.invest_interval}"
    run_name += f"_s{config.seed}_t{int(time.time())}"
    wandb_init(asdict(config), run_name=run_name)

    for t in trange(config.num_train_ops, ncols=80):
        invest = True if config.invest == 1 and (t + 1) % 5000 == 0 else False
        update_result = awac.update(replay_buffer, invest=invest)
        if (t + 1) % 5000 == 0:
            wandb.log(update_result, step=t)
        if t == 0 or (t + 1) % config.eval_frequency == 0:
            eval_scores = eval_actor(
                env, actor, device, config.n_test_episodes, config.test_seed
            )

            wandb.log({"eval_score": eval_scores.mean()}, step=t)
            if hasattr(env, "get_normalized_score"):
                normalized_eval_scores = env.get_normalized_score(eval_scores) * 100.0
                wandb.log(
                    {"d4rl_normalized_score": normalized_eval_scores.mean()}, step=t
                )

    wandb.finish()


if __name__ == "__main__":
    train()
