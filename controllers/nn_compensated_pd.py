import tensorflow as tf
import numpy as np
from .base_controller import RobotController

class NNCompensatedPD(RobotController):
    def __init__(self, model, kp=100.0, kd=10.0):
        super().__init__(model)
        self.kp = kp
        self.kd = kd
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(7)
        ])
        
    def compute(self, data, target_pos):
        error = target_pos - data.qpos[:7]
        state = np.concatenate([data.qpos[:7], data.qvel[:7]])
        nn_torque = self.nn(tf.expand_dims(state, 0))[0].numpy()
        torque = self.kp * error - self.kd * data.qvel[:7] + nn_torque
        return np.clip(torque, -50.0, 50.0)