
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        next_val = next_value if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns

def evaluate_ppo_worker(agent, env_id, seed):
    env = gym.make(env_id)
    s, _ = env.reset(seed=seed)
    done = False
    tr = 0
    while not done:
        s_t = torch.FloatTensor(s).reshape(1, -1)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(s_t)
        a = action.detach().cpu().numpy().reshape(-1)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        tr += r
    env.close()
    return tr

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, is_discrete=False,
                 action_high=None, action_low=None):
        super().__init__()
        self.is_discrete = is_discrete
        hDim = [64, 64]

        critic_layers = [nn.Linear(obs_dim, hDim[0]), nn.Tanh()]
        for i in range(len(hDim)-1):
            critic_layers += [nn.Linear(hDim[i], hDim[i+1]), nn.Tanh()]
        critic_layers += [nn.Linear(hDim[-1], 1)]
        self.critic = nn.Sequential(*critic_layers)

        actor_layers = [nn.Linear(obs_dim, hDim[0]), nn.Tanh()]
        for i in range(len(hDim)-1):
            actor_layers += [nn.Linear(hDim[i], hDim[i+1]), nn.Tanh()]
        if is_discrete:
            actor_layers += [nn.Linear(hDim[-1], act_dim)]
            self.actor = nn.Sequential(*actor_layers)
        else:
            actor_layers += [nn.Linear(hDim[-1], act_dim)]
            self.actor_mean = nn.Sequential(*actor_layers)
            self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))
            if action_high is not None:
                self.action_scale = torch.FloatTensor((action_high - action_low) / 2.0)
                self.action_bias = torch.FloatTensor((action_high + action_low) / 2.0)
            else:
                self.action_scale = torch.ones(act_dim)
                self.action_bias = torch.zeros(act_dim)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.is_discrete:
            logits = self.actor(x)
            dist = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), self.critic(x)
        else:
            mu = self.actor_mean(x)
            log_std = self.actor_log_std.expand_as(mu)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return action, log_prob, entropy, self.critic(x)

def ppo_worker(worker_id, global_agent,
               env_id, seed, num_steps, num_updates,
               shared_obs, shared_actions, shared_logprobs,
               shared_rewards, shared_dones, shared_values,
               shared_next_obs, shared_next_done,
               collect_barrier, update_barrier, ep_queue):
    torch.manual_seed(seed + worker_id)
    env = gym.make(env_id)
    env.reset(seed=seed + worker_id)
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

    s, _ = env.reset()
    done = False
    ep_reward = 0.0
    ep_steps = 0
    episode_count = 0
    total_steps = 0
    w_start = time.time()

    for update in range(num_updates):
        for step in range(num_steps):
            s_t = torch.FloatTensor(s).reshape(1, -1)
            with torch.no_grad():
                action, log_p, _, value = global_agent.get_action_and_value(s_t)
            a = action.detach().cpu().numpy().reshape(-1)
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_reward += r
            ep_steps += 1
            total_steps += 1

            shared_obs[worker_id, step] = torch.FloatTensor(s)
            shared_actions[worker_id, step] = action.squeeze(0).detach()
            shared_logprobs[worker_id, step] = log_p.squeeze().detach()
            shared_rewards[worker_id, step] = r
            shared_dones[worker_id, step] = float(done)
            shared_values[worker_id, step] = value.squeeze().detach()

            s = s_next
            if done:
                eval_reward = evaluate_ppo_worker(global_agent, env_id, seed + worker_id)
                w_end = time.time()
                episode_count += 1
                ep_queue.put((worker_id, episode_count, ep_reward, eval_reward,
                              total_steps, 0.0, w_end - w_start))
                if episode_count % 10 == 0:
                    print(f"Worker {worker_id} | Episode {episode_count} | "
                          f"Train Reward: {ep_reward:.2f} | Eval Reward: {eval_reward:.2f}")
                s, _ = env.reset()
                ep_reward = 0.0
                ep_steps = 0
                w_start = time.time()
                done = False

        shared_next_obs[worker_id] = torch.FloatTensor(s)
        shared_next_done[worker_id] = float(done)

        collect_barrier.wait()
        update_barrier.wait()

    env.close()
