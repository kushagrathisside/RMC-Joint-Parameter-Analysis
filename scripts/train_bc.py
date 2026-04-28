import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from utils.learning_utils import (
    BC_METRICS_PATH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_SPLIT,
    DATASET_PATH,
    POLICY_MODEL_PATH,
    POLICY_STATS_PATH,
    ensure_results_dir,
    load_dataset,
    normalize_states,
    save_json,
    save_normalization,
    train_validation_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy from a logged dataset.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the dataset NPZ file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of samples reserved for validation. Set to 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for splitting/shuffling.")
    parser.add_argument("--output-model", default=str(POLICY_MODEL_PATH), help="Path to save the trained model.")
    parser.add_argument(
        "--output-stats",
        default=str(POLICY_STATS_PATH),
        help="Path to save normalization statistics.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(BC_METRICS_PATH),
        help="Path to save training metrics as JSON.",
    )
    return parser.parse_args()


def evaluate_loss(agent, states, actions):
    if len(states) == 0:
        return None
    import tensorflow as tf

    state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    action_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
    action_targets = tf.clip_by_value(action_tensor / agent.action_scale, -1.0, 1.0)
    predictions = agent.actor(state_tensor, training=False)
    return float(tf.reduce_mean(tf.square(predictions - action_targets)).numpy())


def train(
    dataset_path=DATASET_PATH,
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    validation_split=DEFAULT_VALIDATION_SPLIT,
    seed=DEFAULT_RANDOM_SEED,
    model_path=POLICY_MODEL_PATH,
    stats_path=POLICY_STATS_PATH,
    metrics_path=BC_METRICS_PATH,
):
    import tensorflow as tf

    from controllers.rl_controller import RLController

    dataset = load_dataset(dataset_path)
    states = np.asarray(dataset["state"], dtype=np.float32)
    actions = np.asarray(dataset["action"], dtype=np.float32)

    (train_states, train_actions), (val_states, val_actions) = train_validation_split(
        states,
        actions,
        validation_split=validation_split,
        seed=seed,
    )

    normalized_train_states, mean, std = normalize_states(train_states)
    normalized_val_states, _, _ = normalize_states(val_states, mean=mean, std=std)
    train_state_tensor = tf.convert_to_tensor(normalized_train_states, dtype=tf.float32)
    train_action_tensor = tf.convert_to_tensor(train_actions, dtype=tf.float32)

    agent = RLController(model=None, load_pretrained=False)
    agent.optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_state_tensor, train_action_tensor))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_states), seed=seed).batch(batch_size)

    print(
        f"Training behavior cloning policy on {len(train_states)} transitions"
        + (f" with {len(val_states)} validation samples" if len(val_states) else "")
    )
    history = []
    best_weights = agent.actor.get_weights()
    best_epoch = 0
    best_score = None
    for epoch in range(epochs):
        losses = []
        for batch_states, batch_actions in train_dataset:
            losses.append(agent.train_step(batch_states, batch_actions))
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(agent, normalized_val_states, val_actions)
        score = val_loss if val_loss is not None else train_loss
        if best_score is None or score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_weights = agent.actor.get_weights()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if val_loss is None:
            print(f"BC Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f}")
        else:
            print(f"BC Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    ensure_results_dir()
    agent.actor.set_weights(best_weights)
    agent.actor.save(model_path)
    save_normalization(stats_path, mean, std)
    metrics = {
        "dataset_path": str(dataset_path),
        "num_samples": int(len(states)),
        "num_train": int(len(train_states)),
        "num_val": int(len(val_states)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "validation_split": float(validation_split),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "history": history,
        "model_path": str(model_path),
        "stats_path": str(stats_path),
    }
    save_json(metrics_path, metrics)

    print("Saved behavior cloning artifacts:")
    print(f"  - {model_path}")
    print(f"  - {stats_path}")
    print(f"  - {metrics_path}")
    print(f"Best epoch: {best_epoch}")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        seed=args.seed,
        model_path=args.output_model,
        stats_path=args.output_stats,
        metrics_path=args.metrics_output,
    )
