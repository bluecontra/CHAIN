# re-construct from CleanRL

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


def cal_kl_div(mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None, clamp_threshold=100):
    sigma_poster = sigma_poster ** 2
    sigma_prior = sigma_prior ** 2
    sigma_poster_matrix_det = torch.prod(sigma_poster, dim=1)
    sigma_prior_matrix_det = torch.prod(sigma_prior, dim=1)

    sigma_prior_matrix_inv = 1.0 / sigma_prior
    delta_u = (mu_prior - mu_poster)
    term1 = torch.sum(sigma_poster / sigma_prior, dim=1)
    term2 = torch.sum(delta_u * sigma_prior_matrix_inv * delta_u, dim=1)
    term3 = - mu_poster.shape[-1]
    term4 = torch.log(sigma_prior_matrix_det + 1e-8) - torch.log(sigma_poster_matrix_det + 1e-8)
    kl_loss = 0.5 * (term1 + term2 + term3 + term4)
    kl_loss = torch.clamp(kl_loss, 0, clamp_threshold)

    return torch.mean(kl_loss)


def cal_jeff_div(mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None, clamp_threshold=100):
    sigma_poster = sigma_poster ** 2
    sigma_prior = sigma_prior ** 2

    sigma_prior_matrix_inv = 1.0 / sigma_prior
    sigma_poster_matrix_inv = 1.0 / sigma_poster
    delta_u = mu_prior - mu_poster
    delta_u_ = mu_poster - mu_prior
    term1 = torch.sum(sigma_poster / sigma_prior, dim=1) + torch.sum(sigma_prior / sigma_poster, dim=1)
    term2 = torch.sum(delta_u * sigma_prior_matrix_inv * delta_u, dim=1)\
            + torch.sum(delta_u_ * sigma_poster_matrix_inv * delta_u_, dim=1)
    term3 = -mu_poster.shape[-1] - mu_prior.shape[-1]
    jeff_kl_loss = 0.5 * (term1 + term2 + term3)
    jeff_kl_loss = torch.clamp(jeff_kl_loss, 0, clamp_threshold)

    return torch.mean(jeff_kl_loss)


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


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_space):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
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
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        raw_mean, raw_log_std = self(x)
        std = raw_log_std.exp()
        normal = torch.distributions.Normal(raw_mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(raw_mean) * self.action_scale + self.action_bias
        # NOTE - add additional output to save the computation below
        return action, log_prob, mean, raw_mean, raw_log_std
        # return action, log_prob, mean


# ALGO LOGIC: initialize agent here
class SAC_DCR(object):
    def __init__(
            self,
            obs_dim,
            action_dim,
            action_space,
            discount=0.99,
            tau=0.005,
            lr=3e-4,
            device=None,
            invest_window_size=1,
            invest_interval=20,
            alpha=0.2,
            autotune=False,
            reg_coef=1.0,
            v_reg_coef=1.0,
            reg_his_idx=2,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space

        self.lr = lr
        self.discount = discount
        self.tau = tau
        self.autotune = autotune

        self.reg_coef = reg_coef
        self.v_reg_coef = v_reg_coef
        self.reg_his_idx = reg_his_idx

        self.device = device if device is not None else torch.device("cpu")

        self.actor = Actor(obs_dim, action_dim, action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.lr)

        self.qf1 = QNetwork(obs_dim, action_dim).to(self.device)
        self.qf2 = QNetwork(obs_dim, action_dim).to(self.device)
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        self.critic_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.lr)

        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = alpha

        self.invest_window_size = invest_window_size
        self.invest_interval = invest_interval
        self.frozen_init_policy = copy.deepcopy(self.actor)
        self.his_policy_list = [self.frozen_init_policy]
        self.frozen_init_qf = copy.deepcopy(self.qf1)
        self.his_qf1_list = [copy.deepcopy(self.qf1)]
        self.his_qf2_list = [copy.deepcopy(self.qf2)]

        self.total_it = 0

    def select_action(self, obs):
        with torch.no_grad():
            action, _, _, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
        return action.cpu().numpy()

    def train(self, replay_buffer, batch_size=256, writer=None, writer_timestamp=None, invest=False):
        self.total_it += 1

        # Sample replay buffer
        data = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy
            next_actions, next_state_log_pi, _, _, _ = self.actor.get_action(data.next_observations)

            # Compute the target Q value
            next_target_q1 = self.qf1_target(data.next_observations, next_actions)
            next_target_q2 = self.qf2_target(data.next_observations, next_actions)
            next_target_q = torch.min(next_target_q1, next_target_q2) - self.alpha * next_state_log_pi
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

        if len(self.his_policy_list) < self.reg_his_idx:
            reg_loss = 0
        else:
            reg_his_p = self.his_policy_list[-self.reg_his_idx]
            # NOTE - Compute regularization loss
            reg_raw_means, reg_raw_logstds = self.actor(reg_data.observations)
            with torch.no_grad():
                frozen_raw_means, frozen_raw_logstds = reg_his_p(reg_data.observations)
            reg_loss = cal_kl_div(reg_raw_means, reg_raw_logstds.exp(), frozen_raw_means, frozen_raw_logstds.exp())

        # Compute actor loss
        pi, log_pi, _, _, _ = self.actor.get_action(data.observations)
        qf1_pi = self.qf1(data.observations, pi)
        qf2_pi = self.qf2(data.observations, pi)
        min_qf_pi_log = torch.min(qf1_pi, qf2_pi) - self.alpha * log_pi
        actor_loss = -min_qf_pi_log.mean()

        # NOTE - inspired by TD3+BC, but calculate adaptive coefficient while keep the scale of actor_loss
        q_coef = self.reg_coef * min_qf_pi_log.abs().mean().detach()
        total_policy_loss = actor_loss + q_coef * reg_loss

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        total_policy_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            with torch.no_grad():
                _, log_pi, _, _, _ = self.actor.get_action(data.observations)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Update the frozen target models
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if invest:
            ref_data = replay_buffer.sample(batch_size)
            frozen_actor = self.his_policy_list[-1]
            # NOTE - debugged to be strict to compute the difference between current policy and \pi_t-k
            ref_policy_list = []
            for i in range(self.invest_window_size):
                if (i + 1) * self.invest_interval > len(self.his_policy_list):
                    break
                ref_policy_list.append(self.his_policy_list[-(i + 1) * self.invest_interval])
            prior_ref_q_list, prior_ref_mean_list, prior_ref_std_list = [], [], []

            with torch.no_grad():
                # Compute current ref actions/qs
                _, _, cur_ref_actions, cur_ref_means, cur_ref_logstds = self.actor.get_action(ref_data.observations)
                cur_ref_qs = self.qf1(ref_data.observations, cur_ref_actions)

                # Compute prior-update ref actions/qs
                _, _, prior_update_ref_actions, prior_update_ref_means, prior_update_ref_logstds = frozen_actor.get_action(ref_data.observations)
                prior_update_ref_qs = self.qf1(ref_data.observations, prior_update_ref_actions)
                prior_ref_mean_list.append(prior_update_ref_means)
                prior_ref_std_list.append(prior_update_ref_logstds.exp())
                prior_ref_q_list.append(prior_update_ref_qs)

                # Calculate historical ref actions/qs
                for idx, p in enumerate(ref_policy_list):
                    _, _, his_ref_actions, his_ref_means, his_ref_logstds = p.get_action(ref_data.observations)
                    his_ref_qs = self.qf1(ref_data.observations, his_ref_actions)
                    prior_ref_mean_list.append(his_ref_means)
                    prior_ref_std_list.append(his_ref_logstds.exp())
                    prior_ref_q_list.append(his_ref_qs)

                # Calculate outputs of frozen init policy
                _, _, init_ref_actions, init_ref_means, init_ref_logstds = self.frozen_init_policy.get_action(ref_data.observations)
                init_ref_qs = self.qf1(ref_data.observations, init_ref_actions)
                prior_ref_mean_list.append(init_ref_means)
                prior_ref_std_list.append(init_ref_logstds.exp())
                prior_ref_q_list.append(init_ref_qs)

            # Calculate prior-post update diff
            cnt = 0
            for ref_m, ref_std, ref_q in zip(prior_ref_mean_list, prior_ref_std_list, prior_ref_q_list):
                # NOTE - to handle the corner case
                if cnt == 0 and self.invest_interval == 1:
                    cnt += 1
                    continue

                invest_a_diff_kl = cal_kl_div(cur_ref_means, cur_ref_logstds.exp(), ref_m, ref_std)
                invest_a_diff_jeff = cal_jeff_div(cur_ref_means, cur_ref_logstds.exp(), ref_m, ref_std)
                invest_q_diff = (cur_ref_qs - ref_q).mean()
                invest_q_diff_abs = (cur_ref_qs - ref_q).abs().mean()

                if cnt == 0:
                    write_suffix = ''
                elif cnt == len(prior_ref_mean_list) - 1:
                    write_suffix = '_init'
                else:
                    write_suffix = '_' + str(cnt * self.invest_interval)
                writer.add_scalar("invest/a_diff_kl" + write_suffix, invest_a_diff_kl.item(), writer_timestamp)
                writer.add_scalar("invest/a_diff_jeff" + write_suffix, invest_a_diff_jeff.item(), writer_timestamp)
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
            if total_policy_loss is not None:
                writer.add_scalar("losses/actor_loss", actor_loss.item(), writer_timestamp)
                writer.add_scalar("losses/reg_loss", reg_loss.item(), writer_timestamp)
                writer.add_scalar("losses/policy_loss", total_policy_loss.item(), writer_timestamp)

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
