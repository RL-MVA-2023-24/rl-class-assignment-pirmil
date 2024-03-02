import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, nb_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim).double()
        self.fc2 = nn.Linear(hidden_dim, nb_actions).double()

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores,dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)

class ReinforceAgent:
    def __init__(self, config, policy_network: nn.Module):
        self.device = "cuda" if next(policy_network.parameters()).is_cuda else "cpu"
        self.scalar_dtype = next(policy_network.parameters()).dtype
        self.policy = policy_network
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()),lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1

    def greedy_action(self, x):
        with torch.no_grad():        
            probabilities = self.policy(torch.as_tensor(x))
            action = torch.argmax(probabilities).item()
            return action

    def sample_action_and_log_prob(self, x):
        probabilities = self.policy(torch.as_tensor(x))
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        return action.item(), log_prob
    
    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        log_probs = []
        returns = []
        for ep in range(self.nb_episodes):
            print(f"Episode: {ep}/{self.nb_episodes}")
            x,_ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while(True):
                a, log_prob = self.sample_action_and_log_prob(x)
                y,r,d,_,_ = env.step(a)
                log_probs.append(log_prob)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                if d:
                    # compute returns-to-go
                    new_returns = []
                    G_t = 0
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
        # make loss
        returns = torch.tensor(returns)
        log_probs = torch.cat(log_probs)
        loss = -(returns * log_probs).mean()
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        for rollout in range(nb_rollouts):
            print(f'Rollout: {rollout}/{nb_rollouts}')
            avg_sum_rewards.append(self.one_gradient_step(env))
        return avg_sum_rewards
    

    def act(self, state):
        return self.greedy_action(state)
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()