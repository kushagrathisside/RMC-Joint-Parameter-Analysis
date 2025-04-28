import tensorflow as tf
import numpy as np

class NeuralTrajectoryGenerator:
    def __init__(self):
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(7)
        ])
        
    def generate(self, t):
        phase = np.sin(2 * np.pi * 0.2 * t)
        return self.generator(tf.constant([[t, phase]]))[0].numpy()