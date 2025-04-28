import gymnasium as gym
from controllers.rl_controller import RLController
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import mujoco

class CassieEnv(gym.Env):
    def __init__(self):
        self.model = load_robot_description("cassie_mj_description")
        self.data = self.model.data()
        self.controller = RLController(self.model)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (21,))
        self.action_space = gym.spaces.Box(-1, 1, (14,))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[:7] = action
        mujoco.mj_step(self.model, self.data)
        reward = -np.sum(np.square(self.data.qpos[:7]))
        done = False
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos[:7], self.data.qvel[:7]])