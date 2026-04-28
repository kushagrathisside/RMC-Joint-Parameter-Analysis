import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

STATE_DIM = 21
ACTION_DIM = 7
DEFAULT_ACTION_SCALE = 50.0
DEFAULT_RESIDUAL_SCALE = 10.0
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_RANDOM_SEED = 42

DATASET_PATH = RESULTS_DIR / "dataset.npz"
POLICY_MODEL_PATH = RESULTS_DIR / "policy.h5"
POLICY_STATS_PATH = RESULTS_DIR / "policy_normalization.npz"
PPO_POLICY_MODEL_PATH = RESULTS_DIR / "ppo_policy.h5"
PPO_POLICY_STATS_PATH = RESULTS_DIR / "ppo_normalization.npz"
RESIDUAL_MODEL_PATH = RESULTS_DIR / "residual_policy.h5"
RESIDUAL_STATS_PATH = RESULTS_DIR / "residual_normalization.npz"
BC_METRICS_PATH = RESULTS_DIR / "bc_metrics.json"
RESIDUAL_METRICS_PATH = RESULTS_DIR / "residual_metrics.json"
PPO_METRICS_PATH = RESULTS_DIR / "ppo_metrics.json"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"
GA_OPTIMIZATION_PATH = RESULTS_DIR / "ga_optimization.json"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def ensure_parent_dir(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_state(qpos, qvel, target):
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    qvel = np.asarray(qvel, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    target = np.asarray(target, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    return np.concatenate([qpos, qvel, target]).astype(np.float32)


def build_state_from_data(data, target):
    return build_state(data.qpos[:ACTION_DIM], data.qvel[:ACTION_DIM], target)


def split_state(state):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.size < STATE_DIM:
        raise ValueError(f"Expected a {STATE_DIM}D state, got shape {state.shape}")
    return (
        state[:ACTION_DIM],
        state[ACTION_DIM : 2 * ACTION_DIM],
        state[2 * ACTION_DIM : 3 * ACTION_DIM],
    )


def compute_tracking_reward(qpos, qvel, target):
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    qvel = np.asarray(qvel, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    target = np.asarray(target, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    error = qpos - target
    return -float(np.dot(error, error) + 0.01 * np.linalg.norm(qvel))


def normalize_states(states, mean=None, std=None):
    states = np.asarray(states, dtype=np.float32)
    if mean is None:
        mean = states.mean(axis=0)
    if std is None:
        std = states.std(axis=0)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
    return (states - mean) / std, mean, std


def save_normalization(path, mean, std, **extra):
    ensure_results_dir()
    payload = {
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
    }
    for key, value in extra.items():
        payload[key] = np.asarray(value, dtype=np.float32)
    np.savez(path, **payload)


def save_json(path, payload):
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
    return path


def load_normalization(path):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_dataset(path=DATASET_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def train_validation_split(*arrays, validation_split=DEFAULT_VALIDATION_SPLIT, seed=DEFAULT_RANDOM_SEED):
    if not arrays:
        raise ValueError("At least one array is required for splitting")

    num_samples = len(arrays[0])
    for array in arrays[1:]:
        if len(array) != num_samples:
            raise ValueError("All arrays must have the same number of samples")

    if num_samples == 0:
        raise ValueError("Cannot split an empty dataset")

    if validation_split <= 0.0 or num_samples < 2:
        train_arrays = tuple(np.asarray(array) for array in arrays)
        val_arrays = tuple(np.asarray(array)[:0] for array in arrays)
        return train_arrays, val_arrays

    num_val = int(round(num_samples * float(validation_split)))
    num_val = min(max(1, num_val), num_samples - 1)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_arrays = tuple(np.asarray(array)[train_indices] for array in arrays)
    val_arrays = tuple(np.asarray(array)[val_indices] for array in arrays)
    return train_arrays, val_arrays


def summarize_dataset(dataset):
    states = np.asarray(dataset["state"], dtype=np.float32)
    actions = np.asarray(dataset["action"], dtype=np.float32)
    next_states = np.asarray(dataset["next_state"], dtype=np.float32)
    rewards = np.asarray(dataset["reward"], dtype=np.float32)
    dones = np.asarray(dataset["done"], dtype=bool)

    if len(states) == 0:
        raise ValueError("Cannot summarize an empty dataset")

    errors = states[:, :ACTION_DIM] - states[:, 2 * ACTION_DIM : 3 * ACTION_DIM]
    next_errors = next_states[:, :ACTION_DIM] - next_states[:, 2 * ACTION_DIM : 3 * ACTION_DIM]
    action_norms = np.linalg.norm(actions, axis=1)

    summary = {
        "num_steps": int(len(states)),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_tracking_error_l2": float(np.mean(np.linalg.norm(errors, axis=1))),
        "final_tracking_error_l2": float(np.linalg.norm(next_errors[-1])),
        "mean_action_l2": float(np.mean(action_norms)),
        "max_action_l2": float(np.max(action_norms)),
        "num_terminal_steps": int(np.count_nonzero(dones)),
    }
    if "time" in dataset:
        time = np.asarray(dataset["time"], dtype=np.float32)
        if len(time) > 1:
            summary["duration"] = float(time[-1] - time[0])
        elif len(time) == 1:
            summary["duration"] = 0.0
    return summary


def build_policy_network(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_units=(128, 128)):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to build policy networks. Run `pip install -r requirements.txt` first."
        ) from exc

    layers = [tf.keras.layers.Input(shape=(state_dim,))]
    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(action_dim, activation="tanh"))
    return tf.keras.Sequential(layers)
