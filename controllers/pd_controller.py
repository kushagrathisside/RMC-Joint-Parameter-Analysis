from .base_controller import RobotController
import numpy as np

class PDController(RobotController):
    def __init__(self, model, kp=100.0, kd=10.0):
        super().__init__(model)
        self.kp = kp
        self.kd = kd
        
    def compute(self, data, target_pos):
        error = target_pos - data.qpos[:7]
        torque = self.kp * error - self.kd * data.qvel[:7]
        return np.clip(torque, -50.0, 50.0)