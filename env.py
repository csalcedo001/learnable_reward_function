import gym


class SparseEnvWrapper(gym.Wrapper):
    def __init__(self, env, max_timesteps):
        super().__init__(env)
        self.env = env
        self.timestep = 0
        self.min_reward = 0
        self.max_reward = 0
        self.max_timesteps = max_timesteps
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        self.timestep += 1

        if reward > self.max_reward:
            self.max_reward = reward
        if reward < self.min_reward:
            self.min_reward = reward

        # Terminate environment if it is beyond max_timesteps
        if self.timestep == self.max_timesteps:
            done = True
        
        # Normalize reward between min and max observed rewards
        reward_range = self.max_reward - self.min_reward

        if reward_range == 0:
            reward = 0
        else:
            reward = (reward + self.min_reward) * 2. / reward_range - 1.

        # Make environment sparse
        if done:
            if reward != 1:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        
        # Prepare env for next step in finished environment
        if done:
            self.timestep = 1

        return next_state, reward, done, info