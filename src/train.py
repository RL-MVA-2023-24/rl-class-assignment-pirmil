from gymnasium.wrappers import TimeLimit
import torch
import torch.nn as nn
import random
import numpy as np
import os
import matplotlib.pyplot as plt

from env_hiv import HIVPatient
from dqn_agent import TargetNetwork, DeepQAgent
from replay_buffer import prefill_buffer

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.

##### HYPERPARAMETERS TO TUNE ######
agent_name = 'DQN'

train_max_episode = 4000

target_network_name = 'TargetNetwork'
target_network_activation = nn.SiLU()
target_network_hidden_dim = 512
target_network_depth = 5
target_network_normalization = None

prefill_steps = 8000
prefill_with_random = True

config = {
    'learning_rate': 0.001,
    'gamma': 0.98,
    'buffer_size': 1000000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.,
    'epsilon_decay_period': 10000,
    'epsilon_delay_decay': 400,
    'batch_size': 1024,
    'gradient_steps': 2,
    'update_target_strategy': 'ema', # or 'replace'
    'update_target_freq': 600,
    'update_target_tau': 0.001,
    'criterion': nn.SmoothL1Loss(), # or nn.HuberLoss()
    'monitoring_nb_trials': 50, 
    'monitor_every': 50,
}

##### FIXED PARAMETERS ######
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = env.observation_space.shape[0]
nb_actions = env.action_space.n

config_constant = {
    'nb_actions': nb_actions, # There are 4 possible choices for the physician at each time step: prescribe nothing, one drug, or both.
    'save_path': f'./{agent_name}.pth',
}

config.update(config_constant)

if target_network_name == 'TargetNetwork':
    model = TargetNetwork(state_dim, target_network_hidden_dim, nb_actions, target_network_depth, target_network_activation, target_network_normalization).to(device)

if agent_name == 'DQN':
    agent = DeepQAgent(config, model)

class ProjectAgent:
    def __init__(self, project_agent_name=agent_name):
        self.agent_name = project_agent_name
        if self.agent_name:
            self.agent = DeepQAgent(config, model)

    def act(self, observation, use_random=False):
        return self.agent.act(observation)

    def save(self, path):
        return self.agent.save(path)

    def load(self):
        path = f"{os.getcwd()}/src/{self.agent_name}.pth"
        self.agent.load(path)


if __name__ == "__main__":
    # Set the seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Pre-fill the buffer
    prefill_buffer(env, agent, prefill_steps, random_samples=prefill_with_random)

    ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, train_max_episode)
    agent.save(config['save_path'])
    print(f"Successfully trained {agent_name}!")

    plt.figure(figsize=(15, 5))
    plt.plot(ep_length, label="Training episode length")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{agent_name}_1.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.plot(disc_rewards, label="MC eval of discounted reward")
    plt.plot(tot_rewards, label="MC eval of total reward")
    plt.plot(V0, label="Average $max_a Q(s_0)$")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{agent_name}_2.png')
    plt.close()