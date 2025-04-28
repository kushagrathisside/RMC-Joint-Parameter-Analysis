import numpy as np

class TrajectoryGenerator:
    def __init__(self, amplitude=1.0, frequency=0.5):
        self.amplitude = amplitude
        self.frequency = frequency

    def sinusoidal(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t)
    
    def step_response(self, t, step_time=2.0):
        return self.amplitude * (t > step_time)
    
    def bezier_curve(self, t, duration=5.0):
        u = t % duration / duration
        return self.amplitude * (u**2 * (3 - 2*u))
    
    def generate(self, t):
        return np.array([
            self.sinusoidal(t),
            -1.0 + self.bezier_curve(t),
            0.0, -1.5, 0.0, 1.5, 0.5
        ])