class PeriodicRegulator:
    def __init__(self, initial=10, period=100, increment=10, maximum=500):
        self.initial = initial
        self.period = period
        self.increment = increment
        self.maximum = maximum

        self.threshold = None
        self.step = 0

    def reset(self):
        self.threshold = None
        self.step = 0
    
    def next(self):
        self.step += 1

        if self.step == 1:
            self.threshold = self.initial
        elif self.step % self.period == 0 and self.threshold < self.maximum:
            if self.threshold + self.period < self.maximum:
                self.threshold += self.increment
            else:
                self.threshold = self.maximum
        
        return self.threshold