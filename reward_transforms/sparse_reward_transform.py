from reward_transforms.reward_transform import RewardTransform

class SparseRewardTransform():
    def __init__(self, reward_pass_grade, max_timesteps=None):
        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.reward_pass_grade = reward_pass_grade
        self.cumulative_reward = 0
    
    def __call__(self, reward, done):
        # Update counters
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
        
        return reward, done
    
    def reset(self):
        self.timestep = 0
        self.cumulative_reward = 0