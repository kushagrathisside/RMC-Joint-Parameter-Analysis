from abc import ABC, abstractmethod

class RobotController(ABC):
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def compute(self, data, target_pos):
        pass