from .difficulty_regulator import DifficultyRegulator

class PeriodicRegulator(DifficultyRegulator):
    def __init__(self, initial=10, period=100, increment=10, maximum=500):
        super().__init__()

        self.initial = initial
        self.period = period
        self.increment = increment
        self.maximum = maximum

        self.threshold = initial
        self.step = 0

    def reset(self):
        self.threshold = self.initial
        self.step = 0
    
    def report(self, data):
        pass
    
    def adjust(self):
        if self.step == 0:
            self.threshold = self.initial
        elif self.step % self.period == 0 and self.threshold < self.maximum:
            if self.threshold + self.period < self.maximum:
                self.threshold += self.increment
            else:
                self.threshold = self.maximum

        self.step += 1
        
        return self.threshold