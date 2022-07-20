import gym


class SparseEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_pass_grade, max_timesteps=None):
        super().__init__(env)
        self.env = env
        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.reward_pass_grade = reward_pass_grade
        self.cumulative_reward = 0
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        self.timestep += 1
        self.cumulative_reward += reward

        # Terminate environment if it is beyond max_timesteps
        if self.timestep == self.max_timesteps:
            done = True

        # Make environment sparse
        if done:
            if self.cumulative_reward >= self.reward_pass_grade:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        
        # Prepare env for next step in finished environment
        if done:
            self.timestep = 1
            self.cumulative_reward = 0

        return next_state, reward, done, info