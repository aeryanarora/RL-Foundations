
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time

def getStepWiseReturnsAndDiscounts(gamma, rewards):
    gammas = np.array([gamma**t for t in range(len(rewards))])
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
        returns[i] = returns[i+1]*gamma + rewards[i]
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    returns = torch.FloatTensor(returns).unsqueeze(1)
    gammas = torch.FloatTensor(gammas).unsqueeze(1)
    return returns, gammas

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

def evaluate_a2c_worker(local_agent, env_id, seed):
    env = gym.make(env_id)
    s, _ = env.reset(seed=seed)
    done = False
    tr = 0
    while not done:
        s_t = torch.FloatTensor(s).reshape(1, -1)
        with torch.no_grad():
            if local_agent.is_discrete:
                logits = local_agent.actor(s_t)
                action = torch.argmax(logits, dim=-1)
            else:
                out = local_agent.actor(s_t)
                act_dim = out.shape[-1] // 2
                action = torch.tanh(out[..., :act_dim])
        a = np.clip(action.detach().cpu().numpy().reshape(-1), env.action_space.low, env.action_space.high)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        tr += r
    env.close()
    return tr

class A2CAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        env = envs[0]
        obs_dim = env.observation_space.shape[0]
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        act_dim = env.action_space.n if self.is_discrete else env.action_space.shape[0]
        hDim = [128, 128]

        critic_layers = [nn.Linear(obs_dim, hDim[0]), nn.ReLU()]
        for i in range(len(hDim)-1):
            critic_layers += [nn.Linear(hDim[i], hDim[i+1]), nn.ReLU()]
        critic_layers += [nn.Linear(hDim[-1], 1)]
        self.critic = nn.Sequential(*critic_layers)

        actor_layers = [nn.Linear(obs_dim, hDim[0]), nn.ReLU()]
        for i in range(len(hDim)-1):
            actor_layers += [nn.Linear(hDim[i], hDim[i+1]), nn.ReLU()]
        if self.is_discrete:
            actor_layers += [nn.Linear(hDim[-1], act_dim)]
        else:
            actor_layers += [nn.Linear(hDim[-1], act_dim*2)]
        self.actor = nn.Sequential(*actor_layers)

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
            out = self.actor(x)
            act_dim = out.shape[-1] // 2
            mu = torch.tanh(out[..., :act_dim])
            logstd = torch.clamp(out[..., act_dim:], -20, 2)
            sigma = torch.exp(logstd)
            dist = torch.distributions.Normal(mu, sigma)
            if action is None:
                action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return action, log_prob, entropy, self.critic(x)

def a2c_worker(worker_id, global_agent,
               env_id, seed, num_steps, num_updates,
               shared_obs, shared_actions, shared_logprobs,
               shared_rewards, shared_dones, shared_values,
               shared_next_obs, shared_next_done,
               collect_barrier, update_barrier, ep_queue):
    torch.manual_seed(seed + worker_id)
    env = gym.make(env_id)
    env.reset(seed=seed + worker_id)

    s, _ = env.reset()
    done = False
    ep_reward = 0.0
    ep_steps = 0
    ep_train_time = 0.0
    ep_wall_time = 0.0
    w_start = time.time()
    episode_count = 0
    total_steps = 0

    for update in range(num_updates):
        for step in range(num_steps):
            s_t = torch.FloatTensor(s).reshape(1, -1)
            with torch.no_grad():
                action, log_p, _, value = global_agent.get_action_and_value(s_t)
            a = np.clip(action.detach().cpu().numpy().reshape(-1), env.action_space.low, env.action_space.high)
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
                eval_reward = evaluate_a2c_worker(global_agent, env_id, seed + worker_id)
                w_end = time.time()
                ep_wall_time = w_end - w_start
                episode_count += 1
                ep_queue.put((worker_id, episode_count, ep_reward, eval_reward,
                              total_steps, ep_train_time, ep_wall_time))
                s, _ = env.reset()
                ep_reward = 0.0
                ep_steps = 0
                ep_train_time = 0.0
                w_start = time.time()
                done = False

        shared_next_obs[worker_id] = torch.FloatTensor(s)
        shared_next_done[worker_id] = float(done)

        collect_barrier.wait()
        update_barrier.wait()

    env.close()
