import numpy as np

from .difficulty_regulator import DifficultyRegulator

class PerformanceBasedRegulator(DifficultyRegulator):
    def __init__(self, initial=10, period=100, maximum=500):
        super().__init__()

        self.initial = initial
        self.period = period
        self.maximum = maximum

        self.threshold = initial
        self.step = 0
        self.rewards = []

    def reset(self):
        self.threshold = self.initial
        self.step = 0

    def report(self, data):
        self.rewards.append(data)

    def adjust(self):
        if self.step == 0:
            self.threshold = self.initial
        elif self.step % self.period == 0:
            rewards = np.array(self.rewards)

            weights = np.ones((self.period))
            new_estimate = np.multiply(weights, rewards).mean()

            self.threshold = min(new_estimate, self.maximum)
            self.rewards = []
        
        self.step += 1

        return self.threshold