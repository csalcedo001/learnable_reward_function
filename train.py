import os

import gym
import torch
from torch import optim
import pickle
import numpy as np
import xlab.experiment as exp
from xlab.utils import merge_dicts

from parser import get_parser
from utils import get_config_from_string
from agents.reinforce import ReinforceAgent
from reward_transforms.sparse_reward_transform import SparseRewardTransform
from reward_transforms.learnable_reward_transform import LearnableRewardTransform
from reward_transforms.reward_merge import RewardMerge
from difficulty_regulators.periodic_regulator import PeriodicRegulator



def episode_train_loss(env, agent, sr_transform, lr_transform, reward_merge):
    sr_transform.reset()

    agent.onpolicy_reset()
    done = False

    state = env.reset()

    total_reward_r = 0.
    total_reward_s = 0.
    total_reward_i = 0.
    total_reward = 0.
    for _ in range(max_iter):
        action = agent.act(state)

        next_state, reward_r, done, _ = env.step(action)
        reward_s, done = sr_transform(reward_r, done)
        reward_i = lr_transform(state)
        reward = reward_merge(reward_s, reward_i)


        agent.append_reward(reward)

        state = next_state

        # Collect metrics
        total_reward_r += reward_r
        total_reward_s += reward_s
        total_reward_i += reward_i.item()
        total_reward += reward.item()


        if not no_render:
            env.render()

        if done:
            break
    
    loss = agent.compute_loss()

    metrics = {
        'r_r': total_reward_r,
        'r_s': total_reward_s,
        'r_i': total_reward_i,
        'r': total_reward
    }

    return loss, metrics



parser = get_parser()

with exp.setup(parser, hash_ignore=['no_render']) as setup:
    ### Get arguments
    args = setup.args
    dir = setup.dir

    # Main arguments
    env_name = args.env
    agent_name = args.agent
    checkpoint = args.checkpoint

    # Optional arguments
    episodes = args.episodes
    num_samples = args.num_samples
    max_iter = args.max_iter
    no_render = args.no_render
    env_config = args.env_config
    agent_config = args.agent_config



    ### Process arguments

    # Validate and get agent class
    agent_name_type_map = {
        'reinforce': ReinforceAgent,
    }

    if agent_name not in agent_name_type_map:
        class_list = list(agent_name_type_map.values())
        class_names = [cls.__name__ for cls in class_list]
        
        raise Exception('Invalid agent: choose from {}.'.format(class_names))

    agent_class = agent_name_type_map[agent_name]

    # Validate env
    valid_envs = [
        'CartPole-v1',
    ]

    if env_name not in valid_envs:
        raise Exception('Invalid environment: choose from {}.'.format(
            valid_envs))

    # Validate checkpoint for agent type
    if checkpoint == None and agent_name !='random':
        warning_msg = "Warning: no checkpoint was provided for agent '{}'."
        print(warning_msg.format(agent_name))

    if checkpoint != None:
        if agent_name == 'random':
            error_msg = 'Error: random agents cannot load from a checkpoint.'
            raise Exception(error_msg)
        
        if type(checkpoint) == str and os.path.isdir(checkpoint):
            checkpoint_dir = checkpoint
        else:
            checkpoint_dict = get_config_from_string(checkpoint)
                
            executable = 'train.py'
            command = 'python -m train {agent}'
            req_args = {
                'agent': agent_name,
                'env': env_name,
            }

            checkpoint_config = merge_dicts(req_args, checkpoint_dict)
            e = exp.Experiment(executable, checkpoint_config, command=command)

            checkpoint_dir = e.get_dir()

        if not os.path.isdir(checkpoint_dir):
            error_msg = "Error: could not load checkpoint from '{}'."
            raise Exception(error_msg.format(checkpoint_dir))



    batch_size = 16


    ### Setup for training

    envs = [gym.make(env_name, **env_config) for _ in range(batch_size)]
    reward_merge = RewardMerge()

    losses = []
    rewards = []
    for sample in range(num_samples):
        agents = [agent_class(envs[i], **agent_config) for i in range(batch_size)]
        lr_transform = LearnableRewardTransform(envs[0].observation_space)

        optimizer = optim.Adam(lr_transform.parameters(), lr=0.00001)
        lr_transform.train()

        for agent in agents:
            optimizer.add_param_group({
                'params': agent.parameters(),
                'lr': 0.01
            })
            agent.train()

            if checkpoint != None:
                agent.load(checkpoint_dir)
        
        
        difficulty_regulators = []
        sr_transforms = []

        for i in range(batch_size):
            difficulty_regulator = PeriodicRegulator(
                initial=10,
                period=100,
                increment=10,
                maximum=500
            )

            reward_pass_grade = difficulty_regulator.initial
            sr_transform = SparseRewardTransform(
                reward_pass_grade=reward_pass_grade,
                max_timesteps=500
            )

            difficulty_regulators.append(difficulty_regulator)
            sr_transforms.append(sr_transform)


        sample_losses = []
        sample_rewards = []

        for episode in range(episodes):
            # Adjust environment difficulty
            for i in range(batch_size):
                reward_pass_grade = difficulty_regulators[i].next()
                sr_transforms[i].set_reward_pass_grade(reward_pass_grade)
            

            loss = 0.

            reward_r = []
            reward_s = []
            reward_i = []
            for batch_i in range(batch_size):
                loss_i, metrics_i = episode_train_loss(
                    envs[batch_i],
                    agents[batch_i],
                    sr_transforms[batch_i],
                    lr_transform,
                    reward_merge)

                loss += loss_i

                reward_r.append(metrics_i['r_r'])
                reward_s.append(metrics_i['r_s'])
                reward_i.append(metrics_i['r_i'])

            loss /= batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Print results
            loss = loss.item()
            reward_r = np.array(reward_r)
            reward_s = np.array(reward_s)
            reward_i = np.array(reward_i)

            print('Ep/diff ({:4}/{:2}). Loss: {:9.3f}. Rs: {:5.2f}. Rp: {:4.0f} Rp_min: {:2.0f} Rp_max: {:3.0f}. Ri: {:4.2f}'.format(
                episode, difficulty_regulators[-1].threshold, loss, reward_s.mean(), reward_r.mean(), reward_r.min(), reward_r.max(), reward_i.mean()))
            
            sample_losses.append(loss)
            sample_rewards.append(reward_s)
        

        torch.save(agent.state_dict(), os.path.join(dir, 'agent_model.pt'))
        torch.save(lr_transform.state_dict(), os.path.join(dir, 'reward_model.pt'))

        losses.append(sample_losses)
        rewards.append(sample_rewards)


    losses = np.array(losses)
    reward = np.array(rewards)

    losses_path = os.path.join(dir, 'losses.pkl')
    rewards_path = os.path.join(dir, 'rewards.pkl')

    with open(losses_path, 'wb') as out_file:
        pickle.dump(losses, out_file)

    with open(rewards_path, 'wb') as out_file:
        pickle.dump(rewards, out_file)