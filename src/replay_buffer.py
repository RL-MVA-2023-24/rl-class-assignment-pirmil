import torch
import numpy as np
import random
from gymnasium.wrappers import TimeLimit
from tqdm import trange


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    

def prefill_buffer(env: TimeLimit, agent, prefill_steps, random_samples=True):
    print(f"Pre-filling the buffer with {prefill_steps} {'random' if random_samples else 'non-random (agent)'} steps")
    x, _ = env.reset()
    for _ in trange(prefill_steps):
        if random_samples:
            a = env.action_space.sample()
        else:
            a = agent.act(x)
        y, r, done, trunc, _ = env.step(a)
        agent.memory.append(x, a, r, y, done)
        if done or trunc:
            x, _ = env.reset()
        else:
            x = y