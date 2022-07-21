import os

import torch
from torch import nn
import numpy as np

from .agent import Agent


class RewardLearningAgent(Agent):
    def __init__(self, agent, env, w_rp=1., w_ri=10.):
        super().__init__(env)

        self.w_rp = w_rp
        self.w_ri = w_ri


        n_in = env.observation_space.shape[0]
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
        
        self.reward_model = nn.Sequential(*layers)

        agent.optimizer.add_param_group({
            'params': self.reward_model.parameters()
        })

        self.agent = agent

    def forward(self, state):
        return self.agent.forward(state)

    def act(self, state):
        return self.agent.act(state)
    
    def train_start(self, state):
        self.agent.train_start(state)

    def train_step(self, state, action, next_state, reward):
        s = torch.from_numpy(state.astype(np.float32))

        rp = reward
        ri = self.reward_model(s)

        reward = self.w_rp * rp + self.w_ri * ri

        self.agent.train_step(state, action, next_state, reward)
    
    def train_end(self, state):
        return self.agent.train_end(state)
    
    def save(self, dict_path):
        path = os.path.join(dict_path, 'model.pt')
        torch.save(self.agent.state_dict(), path)
    
    def load(self, dict_path):
        path = os.path.join(dict_path, 'model.pt')

        if not os.path.exists(path):
            err_msg = "Error: no checkpoint '{}' in directory '{}'."
            raise Exception(err_msg.format('model.pt', dict_path))
        
        self.load_state_dict(torch.load(path))
