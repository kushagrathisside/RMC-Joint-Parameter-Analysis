import unittest
import numpy as np
from controllers.pd_controller import PDController
from robot_descriptions.loaders.mujoco import load_robot_description

class TestPIDController(unittest.TestCase):
    def setUp(self):
        self.model = load_robot_description("cassie_mj_description")
        self.controller = PDController(self.model)

    def test_compute(self):
        class MockData:
            qpos = np.zeros(7)
            qvel = np.zeros(7)
        data = MockData()
        torque = self.controller.compute(data, np.ones(7))
        self.assertEqual(torque.shape, (7,))