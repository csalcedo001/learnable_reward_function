import os

import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()

        self._env = env
    
    def env(self):
        return self._env


    ### Must be implemented for evaluation
    def act(self, state):
        raise NotImplementedError
    

    ### Must be implemented for training
    def train_start(self, state):
        raise NotImplementedError

    def train_step(self, state, reward):
        raise NotImplementedError
    
    def train_end(self, state):
        raise NotImplementedError

    
    ### Must be implemented for 
    def save(self, dict_path):
        raise NotImplementedError

    def load(self, dict_path):
        raise NotImplementedError
    

    ### Functions 
    def _save(self, dict_path):
        path = os.path.join(dict_path, 'model.pt')
        torch.save(self.state_dict(), path)

    def _load(self, dict_path):
        path = os.path.join(dict_path, 'model.pt')

        if not os.path.exists(path):
            err_msg = "Error: no checkpoint '{}' in directory '{}'."
            raise Exception(err_msg.format('model.pt', dict_path))
        
        self.load_state_dict(torch.load(path))