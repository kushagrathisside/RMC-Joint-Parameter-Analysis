import unittest

import numpy as np

from utils.learning_utils import summarize_dataset, train_validation_split


class TestLearningUtils(unittest.TestCase):
    def test_train_validation_split_preserves_sample_count(self):
        states = np.arange(30, dtype=np.float32).reshape(10, 3)
        actions = np.arange(20, dtype=np.float32).reshape(10, 2)

        (train_states, train_actions), (val_states, val_actions) = train_validation_split(
            states,
            actions,
            validation_split=0.2,
            seed=7,
        )

        self.assertEqual(len(train_states), 8)
        self.assertEqual(len(val_states), 2)
        self.assertEqual(len(train_actions), 8)
        self.assertEqual(len(val_actions), 2)

    def test_summarize_dataset(self):
        dataset = {
            "time": np.array([0.0, 0.1], dtype=np.float32),
            "state": np.array(
                [
                    np.concatenate([np.zeros(7), np.zeros(7), np.ones(7)]),
                    np.concatenate([np.full(7, 0.5), np.zeros(7), np.ones(7)]),
                ],
                dtype=np.float32,
            ),
            "action": np.array(
                [
                    np.zeros(7),
                    np.ones(7),
                ],
                dtype=np.float32,
            ),
            "next_state": np.array(
                [
                    np.concatenate([np.full(7, 0.5), np.zeros(7), np.ones(7)]),
                    np.concatenate([np.full(7, 0.75), np.zeros(7), np.ones(7)]),
                ],
                dtype=np.float32,
            ),
            "reward": np.array([-1.0, -0.5], dtype=np.float32),
            "done": np.array([False, True]),
        }

        summary = summarize_dataset(dataset)

        self.assertEqual(summary["num_steps"], 2)
        self.assertAlmostEqual(summary["total_reward"], -1.5)
        self.assertAlmostEqual(summary["duration"], 0.1)
        self.assertEqual(summary["num_terminal_steps"], 1)
