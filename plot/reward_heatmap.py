import os
from argparse import ArgumentParser

import gym
import numpy as np
import xlab.experiment as exp
from xlab.utils import merge_dicts
import torch
import matplotlib.pyplot as plt

from utils import get_config_from_string
from agents.reinforce import ReinforceAgent
from agents.reward_learning_agent import RewardLearningAgent



parser = ArgumentParser()

parser.add_argument(
    'agent',
    type=str)
parser.add_argument(
    '--agent-config',
    type=get_config_from_string, default={})

args = parser.parse_args()



agent_name = args.agent
agent_config = args.agent_config

env = gym.make('CartPole-v1')



agent_name_type_map = {
    'reinforce': ReinforceAgent,
}

if agent_name not in agent_name_type_map:
    class_list = list(agent_name_type_map.values())
    class_names = [cls.__name__ for cls in class_list]
    
    raise Exception('Invalid agent: choose from {}.'.format(class_names))

agent_class = agent_name_type_map[agent_name]

agent = ReinforceAgent(env, **agent_config)
agent = RewardLearningAgent(agent, env)



e = exp.Experiment(
    executable='train.py',
    req_args={
        'agent': agent_name
    },
    command='python {executable} {agent} --episodes 500 --no-render'
)

e.args['episodes'] = 500
e.args['agent_config'] = agent_config

model_dir = e.get_dir()

agent.load_state_dict(torch.load(os.path.join(model_dir, 'model_9.pt')))


num_pix = 256
pos_init = -10
pos_end = 10
ang_init = -10
ang_end = 10

data = np.zeros((num_pix, num_pix))
pos_l = np.arange(num_pix) / (num_pix - 1) * (pos_end - pos_init) + pos_init
ang_l = np.arange(num_pix) / (num_pix - 1) * (ang_end - ang_init) + ang_init
for i in range(num_pix):
    ang = ang_l[i]
    for j in range(num_pix):
        pos = pos_l[j]

        x = torch.Tensor([0, pos, 0, ang])

        data[i, j] = agent.reward_model(x)


plots_dir = 'figures'
os.makedirs(plots_dir, exist_ok=True)

fig = plt.figure()
plt.imshow(
    data,
    cmap='hot',
    interpolation='nearest',
    extent=[pos_init, pos_end, ang_init, ang_end]
)
plt.title('Learnt reward for CartPole environment')
plt.xlabel('Cart position')
plt.ylabel('Pole angle')
plt.savefig(os.path.join(plots_dir, 'heatmap_reward.png'))
plt.close(fig)