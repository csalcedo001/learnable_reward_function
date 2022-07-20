import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
from gym import spaces

from .agent import Agent


class ReinforceAgent(Agent):
    def __init__(self, env, gamma=0.99, lr=0.01):
        super(ReinforceAgent, self).__init__(env)
        
        self.discrete_actions = type(env.action_space) == spaces.Discrete

        n_in = env.observation_space.shape[0]

        if self.discrete_actions:
            n_out = env.action_space.n
        else:
            n_out = env.action_space.shape[0]

        
        n_h = 64

        self.gamma = gamma

        layers = [
            nn.Linear(n_in, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
        ]

        # Remember that this network outputs parameters for
        # a distribution of actions, not actions themselves
        self.base = nn.Sequential(*layers)

        if self.discrete_actions:
            self.fc = nn.Linear(n_h, n_out)
        else:
            self.fc_mean = nn.Linear(n_h, n_out)
            self.fc_std = nn.Sequential(nn.Linear(n_h, n_out), nn.Softplus())

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.onpolicy_reset()

    def forward(self, state):
        z = self.base(state)

        if self.discrete_actions:
            pdparams = self.fc(z)
        else:
            epsilon = 1e-6
            pdmean = self.fc_mean(z)
            pdstd = self.fc_std(z) + epsilon
            pdparams = pdmean, pdstd
        
        return pdparams

    def act(self, state):
        state = torch.from_numpy(state.astype(np.float32))
        # Define the probability distribution by computing
        # the probability distribution parameters with the
        # network
        pdparam = self.forward(state)

        if self.discrete_actions:
            pd = distr.Categorical(probs=None, logits=pdparam)
        else:
            pdmean, pdstd = pdparam
            pd = distr.Normal(pdmean, pdstd)

        # Sample an action according to the probability
        # distribution
        action = pd.sample()

        # Compute the log probability of that action being
        # selected, necessary for backpropagation later
        if self.training:
            log_prob = pd.log_prob(action).sum()
            self.log_probs.append(log_prob)

        return action.numpy()
    
    def train_start(self, state):
        self.onpolicy_reset()

    def train_step(self, state, reward):
        self.rewards.append(reward)
    
    def train_end(self, state):
        return self.optimize()
    
    def save(self, save_dict):
        self._save(save_dict)
    
    def load(self, save_dict):
        self._load(save_dict)
    

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def optimize(self):
        loss = 0.
        rets = 0.
        for t in reversed(range(len(self.rewards))):
            rets = self.rewards[t] + self.gamma * rets
            loss += -rets * self.log_probs[t]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()