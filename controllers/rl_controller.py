import tensorflow as tf
import numpy as np
from .base_controller import RobotController

class RLController(RobotController):
    def __init__(self, model):
        super().__init__(model)
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(14)
        ])
        
    def compute(self, data, target_pos):
        state = np.concatenate([data.qpos[:7], data.qvel[:7], target_pos])
        gains = self.actor(tf.expand_dims(state, 0))[0].numpy()
        kp, kd = gains[:7], gains[7:]
        error = target_pos - data.qpos[:7]
        torque = kp * error - kd * data.qvel[:7]
        return np.clip(torque, -50.0, 50.0)