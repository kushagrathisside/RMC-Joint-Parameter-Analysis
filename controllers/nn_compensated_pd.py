import os

import numpy as np
import tensorflow as tf

from .base_controller import RobotController
from .pd_controller import PDController
from utils.learning_utils import (
    DEFAULT_ACTION_SCALE,
    DEFAULT_RESIDUAL_SCALE,
    RESIDUAL_MODEL_PATH,
    RESIDUAL_STATS_PATH,
    build_policy_network,
    build_state_from_data,
    load_normalization,
)


class NNCompensatedPD(RobotController):
    def __init__(
        self,
        model,
        kp=100.0,
        kd=10.0,
        torque_limit=DEFAULT_ACTION_SCALE,
        residual_scale=DEFAULT_RESIDUAL_SCALE,
        load_pretrained=True,
    ):
        super().__init__(model)
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit
        self.residual_scale = residual_scale
        self.pd_controller = PDController(model, kp=kp, kd=kd, torque_limit=torque_limit)
        self.nn = build_policy_network(hidden_units=(64, 64))
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.mean = None
        self.std = None
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

    def compute_from_state(self, state):
        pd_torque = self.pd_controller.compute_from_state(state)
        normalized_state = self._normalize_state(state)
        state_tensor = tf.convert_to_tensor(normalized_state[None, :], dtype=tf.float32)
        residual = self.nn(state_tensor, training=False)[0].numpy() * self.residual_scale
        return np.clip(pd_torque + residual, -self.torque_limit, self.torque_limit)

    def compute(self, data, target_pos):
        state = build_state_from_data(data, target_pos)
        return self.compute_from_state(state)

    def train_step(self, states, residual_targets):
        scaled_targets = tf.clip_by_value(residual_targets / self.residual_scale, -1.0, 1.0)
        with tf.GradientTape() as tape:
            pred_residual = self.nn(states, training=True)
            loss = tf.reduce_mean(tf.square(pred_residual - scaled_targets))

        grads = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
        return float(loss.numpy())

    def _load_pretrained(self):
        if not os.path.exists(RESIDUAL_MODEL_PATH):
            raise FileNotFoundError(
                "NN-compensated PD requires a trained residual policy before simulation. "
                f"Missing model artifact: {RESIDUAL_MODEL_PATH}. "
                "Run `python scripts/train_residual.py` first."
            )

        if not os.path.exists(RESIDUAL_STATS_PATH):
            raise FileNotFoundError(
                "NN-compensated PD requires matching normalization and gain metadata. "
                f"Missing stats artifact: {RESIDUAL_STATS_PATH}. "
                "Run `python scripts/train_residual.py` first."
            )

        try:
            self.nn = tf.keras.models.load_model(RESIDUAL_MODEL_PATH, compile=False)
            print(f"Loaded residual policy from {RESIDUAL_MODEL_PATH}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load residual policy from {RESIDUAL_MODEL_PATH}: {exc}") from exc

        stats = load_normalization(RESIDUAL_STATS_PATH)
        if not stats:
            raise FileNotFoundError(
                "NN-compensated PD could not read saved normalization metadata from "
                f"{RESIDUAL_STATS_PATH}. Re-run `python scripts/train_residual.py`."
            )

        self.set_normalization(stats["mean"], stats["std"])
        if "residual_scale" in stats:
            self.residual_scale = float(np.asarray(stats["residual_scale"]).reshape(-1)[0])
        if "kp" in stats:
            self.kp = float(np.asarray(stats["kp"]).reshape(-1)[0])
        if "kd" in stats:
            self.kd = float(np.asarray(stats["kd"]).reshape(-1)[0])
        self.pd_controller.kp = self.kp
        self.pd_controller.kd = self.kd
