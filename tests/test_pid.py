import unittest
import numpy as np

from controllers.pd_controller import PDController


class TestPIDController(unittest.TestCase):
    def setUp(self):
        self.controller = PDController(model=None, kp=100.0, kd=10.0, torque_limit=50.0)

    def test_compute(self):
        class MockData:
            qpos = np.zeros(7)
            qvel = np.zeros(7)

        data = MockData()
        torque = self.controller.compute(data, np.ones(7))
        self.assertEqual(torque.shape, (7,))
        np.testing.assert_allclose(torque, np.full(7, 50.0, dtype=np.float32))

    def test_compute_from_state(self):
        state = np.concatenate(
            [
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
                np.full(7, 0.25, dtype=np.float32),
            ]
        )
        torque = self.controller.compute_from_state(state)
        expected = np.full(7, 15.0, dtype=np.float32)
        np.testing.assert_allclose(torque, expected)
