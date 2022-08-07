class RewardMerge:
    def __init__(self, w_ro=10., w_ri=0.1):
        self.w_ro = w_ro
        self.w_ri = w_ri
    
    def __call__(self, observed_reward, intrinsic_reward):
        reward = self.w_ro * observed_reward + self.w_ri * intrinsic_reward

        return reward