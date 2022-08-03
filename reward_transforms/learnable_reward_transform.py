import torch
from torch import nn
import numpy as np


class LearnableRewardTransform(nn.Module):
    def __init__(self, obs_space, w_rp=10., w_ri=0.1):
        super().__init__()

        self.w_rp = w_rp
        self.w_ri = w_ri

        n_in = obs_space.shape[0]
        n_out = 1
        n_h = 64

        layers = [
            nn.Linear(n_in, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_out),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*layers)

        self.cumulative_rp = 0
        self.cumulative_ri = 0
        self.cumulative_reward = 0

    def forward(self, reward, state):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state)
        
        rp = reward
        ri = self.model(state)

        r = self.w_rp * rp + self.w_ri * ri

        self.cumulative_rp += rp
        self.cumulative_ri += ri.item()
        self.cumulative_reward += r

        return r
    
    def reset(self):
        self.cumulative_rp = 0
        self.cumulative_ri = 0
        self.cumulative_reward = 0