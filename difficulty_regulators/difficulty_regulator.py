class DifficultyRegulator:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError
    
    def report(self, data):
        raise NotImplementedError
    
    def adjust(self, data):
        raise NotImplementedError