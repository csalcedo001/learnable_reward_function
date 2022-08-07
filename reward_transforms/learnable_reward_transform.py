import torch
from torch import nn
import numpy as np


class LearnableRewardTransform(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

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

    def forward(self, state):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state)
        
        intrinsic_reward = self.model(state)

        return intrinsic_reward