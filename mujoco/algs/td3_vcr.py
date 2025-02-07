# re-construct from CleanRL

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


# Agent Construction: build agent here
# TODO - this can be further encapsulated
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_space):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


# ALGO LOGIC: initialize agent here
class TD3_VCR(object):
    def __init__(
            self,
            obs_dim,
            action_dim,
            action_space,
            discount=0.99,
            tau=0.005,
            lr=3e-4,
            exploration_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            device=None,
            invest_window_size=1,
            invest_interval=20,
            v_reg_coef=1.0,
            reg_his_idx=2,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space

        self.lr = lr
        self.discount = discount
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.v_reg_coef = v_reg_coef
        self.reg_his_idx = reg_his_idx

        self.device = device if device is not None else torch.device("cpu")

        self.actor = Actor(obs_dim, action_dim, action_space).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.lr)

        self.qf1 = QNetwork(obs_dim, action_dim).to(self.device)
        self.qf2 = QNetwork(obs_dim, action_dim).to(self.device)
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        self.critic_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.lr)

        self.invest_window_size = invest_window_size
        self.invest_interval = invest_interval
        self.frozen_init_policy = copy.deepcopy(self.actor)
        self.his_policy_list = [self.frozen_init_policy]
        self.frozen_init_qf = copy.deepcopy(self.qf1)
        self.his_qf1_list = [copy.deepcopy(self.qf1)]
        self.his_qf2_list = [copy.deepcopy(self.qf2)]

        self.total_it = 0

    def select_action(self, obs, is_deterministic=False):
        with torch.no_grad():
            action = self.actor(torch.Tensor(obs).to(self.device))
            if not is_deterministic:
                action += torch.normal(0, self.actor.action_scale * self.exploration_noise)
            action = action.cpu().numpy().clip(self.action_space.low, self.action_space.high)
        return action

    def train(self, replay_buffer, batch_size=256, writer=None, writer_timestamp=None, invest=False):
        self.total_it += 1

        # Sample replay buffer
        data = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip) * self.target_actor.action_scale

            next_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                self.action_space.low[0], self.action_space.high[0])

            # Compute the target Q value
            next_target_q1 = self.qf1_target(data.next_observations, next_actions)
            next_target_q2 = self.qf2_target(data.next_observations, next_actions)
            next_target_q = torch.min(next_target_q1, next_target_q2)
            td_target = data.rewards + (1 - data.dones) * self.discount * next_target_q

        # NOTE - Compute value regularization loss
        if len(self.his_qf1_list) < self.reg_his_idx:
            v_reg_loss = 0
        else:
            reg_data = replay_buffer.sample(batch_size)
            reg_his_qf1 = self.his_qf1_list[-self.reg_his_idx]
            reg_his_qf2 = self.his_qf2_list[-self.reg_his_idx]
            reg_q1 = self.qf1(reg_data.observations, reg_data.actions)
            reg_q2 = self.qf2(reg_data.observations, reg_data.actions)
            with torch.no_grad():
                frozen_q1 = reg_his_qf1(reg_data.observations, reg_data.actions)
                frozen_q2 = reg_his_qf2(reg_data.observations, reg_data.actions)
            qf1_reg_loss = F.mse_loss(reg_q1, frozen_q1)
            qf2_reg_loss = F.mse_loss(reg_q2, frozen_q2)
            v_reg_loss = qf1_reg_loss + qf2_reg_loss

        # Get current Q estimates
        qf1_a_values = self.qf1(data.observations, data.actions)
        qf2_a_values = self.qf2(data.observations, data.actions)
        # Compute critic loss
        qf1_loss = F.mse_loss(qf1_a_values, td_target)
        qf2_loss = F.mse_loss(qf2_a_values, td_target)
        critic_loss = qf1_loss + qf2_loss + self.v_reg_coef * v_reg_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if invest:
                # NOTE - Evaluate policy churn
                ref_data = replay_buffer.sample(batch_size)
                frozen_actor = self.his_policy_list[-1]
                # NOTE - debugged to be strict to compute the difference between current policy and \pi_t-k
                ref_policy_list = []
                for i in range(self.invest_window_size):
                    if (i + 1) * self.invest_interval > len(self.his_policy_list):
                        break
                    ref_policy_list.append(self.his_policy_list[-(i + 1) * self.invest_interval])
                prior_ref_action_list, prior_ref_q_list = [], []

                with torch.no_grad():
                    # Calculate current ref actions/qs
                    cur_ref_actions = self.actor(ref_data.observations)
                    cur_ref_qs = self.qf1(ref_data.observations, cur_ref_actions)

                    # Calculate prior update actions for reference states
                    prior_update_ref_actions = frozen_actor(ref_data.observations)
                    prior_update_ref_qs = self.qf1(ref_data.observations, prior_update_ref_actions)
                    prior_ref_action_list.append(prior_update_ref_actions)
                    prior_ref_q_list.append(prior_update_ref_qs)

                    # Calculate historical ref actions/qs
                    for idx, p in enumerate(ref_policy_list):
                        his_ref_actions = p(ref_data.observations)
                        his_ref_qs = self.qf1(ref_data.observations, his_ref_actions)
                        prior_ref_action_list.append(his_ref_actions)
                        prior_ref_q_list.append(his_ref_qs)

                    # Calculate outputs of frozen init policy
                    init_ref_actions = self.frozen_init_policy(ref_data.observations)
                    init_ref_qs = self.qf1(ref_data.observations, init_ref_actions)
                    prior_ref_action_list.append(init_ref_actions)
                    prior_ref_q_list.append(init_ref_qs)

                # Calculate prior-post update diff
                cnt = 0
                for ref_a, ref_q in zip(prior_ref_action_list, prior_ref_q_list):
                    # NOTE - to handle the corner case
                    if cnt == 0 and self.invest_interval == 1:
                        cnt += 1
                        continue

                    invest_a_diff = (cur_ref_actions - ref_a).mean()
                    invest_a_diff_abs = (cur_ref_actions - ref_a).abs().mean()
                    invest_q_diff = (cur_ref_qs - ref_q).mean()
                    invest_q_diff_abs = (cur_ref_qs - ref_q).abs().mean()

                    if cnt == 0:
                        write_suffix = ''
                    elif cnt == len(prior_ref_q_list) - 1:
                        write_suffix = '_init'
                    else:
                        write_suffix = '_' + str(cnt * self.invest_interval)
                    writer.add_scalar("invest/a_diff_sig" + write_suffix, invest_a_diff.item(), writer_timestamp)
                    writer.add_scalar("invest/a_diff_abs" + write_suffix, invest_a_diff_abs.item(), writer_timestamp)
                    writer.add_scalar("invest/q_diff_sig" + write_suffix, invest_q_diff.item(), writer_timestamp)
                    writer.add_scalar("invest/q_diff_abs" + write_suffix, invest_q_diff_abs.item(), writer_timestamp)

                    cnt += 1

                # NOTE - Evaluate value churn
                frozen_qf = self.his_qf1_list[-1]
                ref_qf_list = []
                for i in range(self.invest_window_size):
                    if (i + 1) * self.invest_interval > len(self.his_qf1_list):
                        break
                    ref_qf_list.append(self.his_qf1_list[-(i + 1) * self.invest_interval])

                prior_qas_list, prior_qpis_list = [], []
                with torch.no_grad():
                    # Calculate current ref actions/qs
                    cur_qa = self.qf1(ref_data.observations, ref_data.actions)
                    cur_qpi = self.qf1(ref_data.observations, cur_ref_actions)

                    # Calculate prior update q values for reference states
                    prior_update_qas = frozen_qf(ref_data.observations, ref_data.actions)
                    prior_update_qpis = frozen_qf(ref_data.observations, cur_ref_actions)
                    prior_qas_list.append(prior_update_qas)
                    prior_qpis_list.append(prior_update_qpis)

                    # Calculate historical ref qas and qpis
                    for idx, qf in enumerate(ref_qf_list):
                        his_qas = qf(ref_data.observations, ref_data.actions)
                        his_qpis = qf(ref_data.observations, cur_ref_actions)
                        prior_qas_list.append(his_qas)
                        prior_qpis_list.append(his_qpis)

                    # Calculate outputs of frozen init policy
                    init_qas = self.frozen_init_qf(ref_data.observations, ref_data.actions)
                    init_qpis = self.frozen_init_qf(ref_data.observations, cur_ref_actions)
                    prior_qas_list.append(init_qas)
                    prior_qpis_list.append(init_qpis)

                # Calculate prior-post update diff
                cnt = 0
                for ref_qa, ref_qpi in zip(prior_qas_list, prior_qpis_list):
                    # NOTE - to handle the corner case
                    if cnt == 0 and self.invest_interval == 1:
                        cnt += 1
                        continue

                    invest_qa_diff = (cur_qa - ref_qa).mean()
                    invest_qa_diff_abs = (cur_qa - ref_qa).abs().mean()
                    invest_qpi_diff = (cur_qpi - ref_qpi).mean()
                    invest_qpi_diff_abs = (cur_qpi - ref_qpi).abs().mean()

                    if cnt == 0:
                        write_suffix = ''
                    elif cnt == len(prior_ref_q_list) - 1:
                        write_suffix = '_init'
                    else:
                        write_suffix = '_' + str(cnt * self.invest_interval)
                    writer.add_scalar("vc_invest/qa_diff_sig" + write_suffix, invest_qa_diff.item(), writer_timestamp)
                    writer.add_scalar("vc_invest/qa_diff_abs" + write_suffix, invest_qa_diff_abs.item(),
                                      writer_timestamp)
                    writer.add_scalar("vc_invest/qpi_diff_sig" + write_suffix, invest_qpi_diff.item(), writer_timestamp)
                    writer.add_scalar("vc_invest/qpi_diff_abs" + write_suffix, invest_qpi_diff_abs.item(),
                                      writer_timestamp)

                    cnt += 1

            # NOTE - add policy to the his policy list
            self.his_policy_list.append(copy.deepcopy(self.actor))
            self.his_policy_list = self.his_policy_list[-self.invest_window_size * self.invest_interval:]

        # NOTE - add policy to the his policy list
        self.his_qf1_list.append(copy.deepcopy(self.qf1))
        self.his_qf2_list.append(copy.deepcopy(self.qf2))
        self.his_qf1_list = self.his_qf1_list[-self.invest_window_size * self.invest_interval:]
        self.his_qf2_list = self.his_qf2_list[-self.invest_window_size * self.invest_interval:]

        # Write the stats
        if writer is not None:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), writer_timestamp)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), writer_timestamp)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), writer_timestamp)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), writer_timestamp)
            writer.add_scalar("losses/v_reg_loss", v_reg_loss.item() / 2.0, writer_timestamp)
            writer.add_scalar("losses/qf_loss", critic_loss.item() / 2.0, writer_timestamp)
            if actor_loss is not None:
                writer.add_scalar("losses/actor_loss", actor_loss.item(), writer_timestamp)


    def save(self, filename):
        torch.save(self.qf1.state_dict(), filename + "_qf1")
        torch.save(self.qf2.state_dict(), filename + "_qf2")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.qf1.load_state_dict(torch.load(filename + "_qf1"))
        self.qf2.load_state_dict(torch.load(filename + "_qf2"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.target_actor = copy.deepcopy(self.actor)