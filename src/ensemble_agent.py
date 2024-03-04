import torch
import torch.nn as nn
from typing import List
from dqn_agent import TargetNetwork, ResTargetNetwork, ResTargetNetwork2
import os

class EnsembleDeepQAgent:
    """
    An ensemble agent that combines the actions of multiple Deep Q-learning agents using weighted majority voting.
    """
    def __init__(self, config, agent_configs: List[dict], weights: List[float] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = agent_configs
        self.state_dim = config['state_dim']
        self.nb_actions = config['nb_actions']
        self.save_path = 'not_relevant'
        if weights is None:
            self.weights = [1.0] * len(agent_configs)
        else:
            assert len(weights) == len(agent_configs), "Number of weights must match the number of agents"
            self.weights = weights

    def single_agent_greedy_action(self, state, i: int):
        device = "cuda" if next(self.models[i].parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.models[i](torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def ensemble_action(self, state):
        actions = [self.single_agent_greedy_action(state, i) for i in range(len(self.configs))]
        weighted_votes = {}
        for i, action in enumerate(actions):
            if action in weighted_votes.keys():
                weighted_votes[action] += self.weights[i]
            else:
                weighted_votes[action] = self.weights[i]
        max_weighted_action = max(weighted_votes, key=weighted_votes.get)
        return max_weighted_action
    
    def act(self, state):
        return self.ensemble_action(state)
    
    def save(self, path):
        pass

    def load(self, path):
        self.models = []
        for config in self.configs:
            if 'target_network_activation' not in config.keys():
                config['target_network_activation'] = nn.SiLU()
            if config['target_network_name'] == 'TargetNetwork':
                if 'target_network_normalization' not in config.keys():
                    config['target_network_normalization'] = None
                target_network = TargetNetwork(self.state_dim, config['target_network_hidden_dim'], self.nb_actions, 
                              config['target_network_depth'], config['target_network_activation'], config['target_network_normalization'])
            elif config['target_network_name'] == 'ResTargetNetwork':
                target_network = ResTargetNetwork(self.state_dim, config['target_network_hidden_dim'], self.nb_actions, 
                              config['target_network_activation'])
            elif config['target_network_name'] == 'ResTargetNetwork2':
                target_network = ResTargetNetwork2(self.state_dim, config['target_network_hidden_dim'], self.nb_actions, 
                              config['target_network_activation'])
            load_path = os.path.join(f"{os.getcwd()}/src", config['path'])         
            target_network.load_state_dict(torch.load(load_path, map_location=self.device))
            target_network.eval()
            self.models.append(target_network)
