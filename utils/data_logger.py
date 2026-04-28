from pathlib import Path

import numpy as np

from utils.learning_utils import DATASET_PATH


class DataLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffers = {
            "time": [],
            "state": [],
            "action": [],
            "next_state": [],
            "reward": [],
            "done": [],
        }

    def log_transition(self, time, state, action, next_state, reward, done):
        self.buffers["time"].append(float(time))
        self.buffers["state"].append(np.asarray(state, dtype=np.float32))
        self.buffers["action"].append(np.asarray(action, dtype=np.float32))
        self.buffers["next_state"].append(np.asarray(next_state, dtype=np.float32))
        self.buffers["reward"].append(float(reward))
        self.buffers["done"].append(bool(done))

    def log(self, time, state, action, next_state, reward, done):
        self.log_transition(time, state, action, next_state, reward, done)

    def as_arrays(self):
        return {
            "time": np.asarray(self.buffers["time"], dtype=np.float32),
            "state": np.asarray(self.buffers["state"], dtype=np.float32),
            "action": np.asarray(self.buffers["action"], dtype=np.float32),
            "next_state": np.asarray(self.buffers["next_state"], dtype=np.float32),
            "reward": np.asarray(self.buffers["reward"], dtype=np.float32),
            "done": np.asarray(self.buffers["done"], dtype=bool),
        }

    def save(self, filename=DATASET_PATH):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.as_arrays())
        return path

    def load(self, filename=DATASET_PATH):
        with np.load(filename) as data:
            return {key: data[key] for key in data.files}
