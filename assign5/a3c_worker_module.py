
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
    returns = torch.FloatTensor(returns).unsqueeze(1)
    gammas = torch.FloatTensor(gammas).unsqueeze(1)
    return returns, gammas

def evaluate_a3c_worker(local_agent, env_id, seed):
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

class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["step"] = torch.zeros(1).share_memory_()
                state["exp_avg"]= torch.zeros_like(param.data).share_memory_()
                state["exp_avg_sq"] = torch.zeros_like(param.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                state["steps"] = state["step"].item()
                state["step"]+= 1
        super().step(closure)

class A3CAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, is_discrete=False):
        super().__init__()
        self.is_discrete = is_discrete
        hDim = [128, 128]

        layers = [nn.Linear(obs_dim, hDim[0]), nn.ReLU()]
        for i in range(len(hDim)-1):
            layers += [nn.Linear(hDim[i], hDim[i+1]), nn.ReLU()]
        layers += [nn.Linear(hDim[-1], 1)]
        self.critic = nn.Sequential(*layers)

        act_layers = [nn.Linear(obs_dim, hDim[0]), nn.ReLU()]
        for i in range(len(hDim)-1):
            act_layers += [nn.Linear(hDim[i], hDim[i+1]), nn.ReLU()]
        if is_discrete:
            act_layers += [nn.Linear(hDim[-1], act_dim)]
        else:
            act_layers += [nn.Linear(hDim[-1], act_dim*2)]
        self.actor = nn.Sequential(*act_layers)

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

def a3c_worker(worker_id, global_agent, global_optimizer, grad_lock,
               env_id, seed, num_episodes,
               gamma, num_steps, ent_coef, vf_coef, max_grad_norm,
               result_queue):
    torch.manual_seed(seed + worker_id)
    env = gym.make(env_id)
    env.reset(seed=seed + worker_id)
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

    local_agent = A3CAgent(obs_dim, act_dim, is_discrete)
    local_agent.load_state_dict(global_agent.state_dict())

    episode_count = 0
    total_steps = 0
    while episode_count < num_episodes:
        local_agent.load_state_dict(global_agent.state_dict())

        rewards, log_probs, entropies, values = [], [], [], []
        s, _ = env.reset()
        done = False
        tr = 0
        steps = 0
        n_steps = 0
        ep_train_time = 0.0
        w_start = time.time()

        while not done:
            steps += 1
            total_steps += 1
            s_t = torch.FloatTensor(s).reshape(1, -1)
            action, log_p, entropy, value = local_agent.get_action_and_value(s_t)
            a = np.clip(action.detach().cpu().numpy().reshape(-1), env.action_space.low, env.action_space.high)            
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            tr += r
            rewards.append(r)
            log_probs.append(log_p)
            entropies.append(entropy)
            values.append(value)

            if (steps - n_steps) == num_steps or done:
                if done:
                    bootstrap_r = 0.0
                else:
                    s_t = torch.FloatTensor(s).reshape(1, -1)
                    bootstrap_v = local_agent.get_value(s_t).detach()
                    bootstrap_r = bootstrap_v.item()

                rewards_with_boot = rewards + [bootstrap_r]
                returns, gammas = getStepWiseReturnsAndDiscounts(gamma, rewards_with_boot)
                returns = returns[:len(rewards)]
                gammas = gammas[:len(rewards)]

                values_t = torch.cat(values, dim=0)
                log_probs_t = torch.cat(log_probs, dim=0)
                entropies_t = torch.cat(entropies, dim=0)

                deltas = returns - values_t
                pLoss = torch.mean(-1.0 * gammas * deltas.detach() * log_probs_t)
                entropyLoss = -1.0 * torch.mean(entropies_t)
                vLoss = torch.mean(0.5 * deltas**2)
                loss = pLoss + ent_coef * entropyLoss + vf_coef * vLoss

                local_agent.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), max_grad_norm)

                t_start = time.time()
                with grad_lock:
                    for p_local, p_global in zip(local_agent.parameters(), global_agent.parameters()):
                        p_global._grad = p_local.grad
                    global_optimizer.step()
                    global_optimizer.zero_grad()
                    local_agent.zero_grad()

                ep_train_time += time.time() - t_start

                local_agent.load_state_dict(global_agent.state_dict())

                rewards, log_probs, entropies, values = [], [], [], []
                n_steps = steps

        w_end = time.time()
        ep_wall_time = w_end - w_start
        eval_reward = evaluate_a3c_worker(local_agent, env_id, seed + worker_id)
        episode_count += 1
        result_queue.put((worker_id, episode_count, tr, eval_reward, total_steps, ep_train_time, ep_wall_time))

    env.close()
