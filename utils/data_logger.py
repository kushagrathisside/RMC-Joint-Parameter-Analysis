import numpy as np
import time

class DataLogger:
    def __init__(self):
        self.log = {'time': [], 'qpos': [], 'qvel': [], 'ctrl': []}
        
    def record(self, data):
        self.log['time'].append(time.time())
        self.log['qpos'].append(data.qpos.copy())
        self.log['qvel'].append(data.qvel.copy())
        self.log['ctrl'].append(data.ctrl.copy())
        
    def save(self, filename="rl_panda_robot_log.npz"):
        np.savez(filename, **self.log)