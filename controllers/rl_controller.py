import os

import numpy as np
import tensorflow as tf

from .base_controller import RobotController
from utils.learning_utils import (
    DEFAULT_ACTION_SCALE,
    POLICY_MODEL_PATH,
    POLICY_STATS_PATH,
    PPO_POLICY_MODEL_PATH,
    PPO_POLICY_STATS_PATH,
    build_policy_network,
    build_state_from_data,
    load_normalization,
)


class RLController(RobotController):
    def __init__(self, model, action_scale=DEFAULT_ACTION_SCALE, load_pretrained=True, model_candidates=None):
        super().__init__(model)
        self.action_scale = action_scale
        self.actor = build_policy_network()
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.mean = None
        self.std = None
        self.model_candidates = model_candidates or [
            (POLICY_MODEL_PATH, POLICY_STATS_PATH),
            (PPO_POLICY_MODEL_PATH, PPO_POLICY_STATS_PATH),
        ]
        if load_pretrained:
            self._load_pretrained()

    def _normalize_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if self.mean is None or self.std is None:
            return state
        return (state - self.mean) / self.std

    def set_normalization(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)

    def act(self, state):
        normalized_state = self._normalize_state(state)
        state_tensor = tf.convert_to_tensor(normalized_state[None, :], dtype=tf.float32)
        normalized_action = self.actor(state_tensor, training=False)[0].numpy()
        torque = normalized_action * self.action_scale
        return np.clip(torque, -self.action_scale, self.action_scale)

    def compute(self, data, target_pos):
        state = build_state_from_data(data, target_pos)
        return self.act(state)

    def train_step(self, states, actions):
        action_targets = tf.clip_by_value(actions / self.action_scale, -1.0, 1.0)
        with tf.GradientTape() as tape:
            pred = self.actor(states, training=True)
            loss = tf.reduce_mean(tf.square(pred - action_targets))

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return float(loss.numpy())

    def _load_normalization(self, path):
        stats = load_normalization(path)
        if not stats:
            return False
        self.set_normalization(stats["mean"], stats["std"])
        return True

    def _load_pretrained(self):
        missing_models = []
        incomplete_artifacts = []
        load_errors = []

        for model_path, normalization_path in self.model_candidates:
            if not os.path.exists(model_path):
                missing_models.append(model_path)
                continue

            if not os.path.exists(normalization_path):
                incomplete_artifacts.append((model_path, normalization_path))
                continue

            try:
                self.actor = tf.keras.models.load_model(model_path, compile=False)
                if not self._load_normalization(normalization_path):
                    incomplete_artifacts.append((model_path, normalization_path))
                    continue
                print(f"Loaded policy from {model_path}")
                return
            except Exception as exc:
                load_errors.append((model_path, exc))

        if load_errors:
            formatted_errors = "; ".join(f"{path}: {exc}" for path, exc in load_errors)
            raise RuntimeError(f"Failed to load a trained RL policy artifact: {formatted_errors}")

        if incomplete_artifacts:
            formatted_pairs = ", ".join(f"{model} + {stats}" for model, stats in incomplete_artifacts)
            raise FileNotFoundError(
                "RL controller requires both a saved policy model and matching normalization stats. "
                f"Incomplete artifacts found: {formatted_pairs}. "
                "Run `python scripts/train_bc.py` or `python scripts/train_rl.py` to regenerate them."
            )

        expected_models = ", ".join(str(path) for path in missing_models) or ", ".join(
            str(model_path) for model_path, _ in self.model_candidates
        )
        raise FileNotFoundError(
            "RL controller requires a trained policy artifact before simulation. "
            f"Expected one of: {expected_models}. "
            "Run `python scripts/train_bc.py` or `python scripts/train_rl.py` first."
        )
