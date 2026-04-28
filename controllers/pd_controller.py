import numpy as np

from .base_controller import RobotController
from utils.learning_utils import DEFAULT_ACTION_SCALE, split_state


class PDController(RobotController):
    def __init__(self, model, kp=100.0, kd=10.0, torque_limit=DEFAULT_ACTION_SCALE):
        super().__init__(model)
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

    def compute_from_components(self, qpos, qvel, target_pos):
        error = np.asarray(target_pos, dtype=np.float32) - np.asarray(qpos, dtype=np.float32)
        torque = self.kp * error - self.kd * np.asarray(qvel, dtype=np.float32)
        return np.clip(torque, -self.torque_limit, self.torque_limit)

    def compute_from_state(self, state):
        qpos, qvel, target_pos = split_state(state)
        return self.compute_from_components(qpos, qvel, target_pos)

    def compute(self, data, target_pos):
        return self.compute_from_components(data.qpos[:7], data.qvel[:7], target_pos)
