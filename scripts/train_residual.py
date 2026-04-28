import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from controllers.pd_controller import PDController
from utils.learning_utils import (
    DEFAULT_RANDOM_SEED,
    DATASET_PATH,
    DEFAULT_RESIDUAL_SCALE,
    DEFAULT_VALIDATION_SPLIT,
    RESIDUAL_MODEL_PATH,
    RESIDUAL_METRICS_PATH,
    RESIDUAL_STATS_PATH,
    ensure_results_dir,
    load_dataset,
    normalize_states,
    save_json,
    save_normalization,
    train_validation_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a residual model on top of the PD controller.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the dataset NPZ file.")
    parser.add_argument("--kp", type=float, default=150.0, help="PD proportional gain.")
    parser.add_argument("--kd", type=float, default=20.0, help="PD derivative gain.")
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE, help="Residual scaling factor.")
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
    parser.add_argument("--output-model", default=str(RESIDUAL_MODEL_PATH), help="Path to save the residual model.")
    parser.add_argument(
        "--output-stats",
        default=str(RESIDUAL_STATS_PATH),
        help="Path to save normalization and gain metadata.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(RESIDUAL_METRICS_PATH),
        help="Path to save training metrics as JSON.",
    )
    return parser.parse_args()


def evaluate_loss(agent, states, residual_targets):
    if len(states) == 0:
        return None
    import tensorflow as tf

    state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    target_tensor = tf.convert_to_tensor(residual_targets, dtype=tf.float32)
    scaled_targets = tf.clip_by_value(target_tensor / agent.residual_scale, -1.0, 1.0)
    predictions = agent.nn(state_tensor, training=False)
    return float(tf.reduce_mean(tf.square(predictions - scaled_targets)).numpy())


def train(
    dataset_path=DATASET_PATH,
    kp=150.0,
    kd=20.0,
    residual_scale=DEFAULT_RESIDUAL_SCALE,
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    validation_split=DEFAULT_VALIDATION_SPLIT,
    seed=DEFAULT_RANDOM_SEED,
    model_path=RESIDUAL_MODEL_PATH,
    stats_path=RESIDUAL_STATS_PATH,
    metrics_path=RESIDUAL_METRICS_PATH,
):
    import tensorflow as tf

    from controllers.nn_compensated_pd import NNCompensatedPD

    dataset = load_dataset(dataset_path)
    states = np.asarray(dataset["state"], dtype=np.float32)
    actions = np.asarray(dataset["action"], dtype=np.float32)

    (train_states, train_actions), (val_states, val_actions) = train_validation_split(
        states,
        actions,
        validation_split=validation_split,
        seed=seed,
    )

    pd_controller = PDController(model=None, kp=kp, kd=kd)
    train_pd_actions = np.stack([pd_controller.compute_from_state(state) for state in train_states], axis=0)
    val_pd_actions = np.stack([pd_controller.compute_from_state(state) for state in val_states], axis=0)
    train_residual_targets = train_actions - train_pd_actions
    val_residual_targets = val_actions - val_pd_actions

    normalized_train_states, mean, std = normalize_states(train_states)
    normalized_val_states, _, _ = normalize_states(val_states, mean=mean, std=std)
    train_state_tensor = tf.convert_to_tensor(normalized_train_states, dtype=tf.float32)
    train_residual_tensor = tf.convert_to_tensor(train_residual_targets, dtype=tf.float32)

    agent = NNCompensatedPD(
        model=None,
        kp=kp,
        kd=kd,
        residual_scale=residual_scale,
        load_pretrained=False,
    )
    agent.optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_state_tensor, train_residual_tensor))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_states), seed=seed).batch(batch_size)

    print(
        f"Training residual policy on {len(train_states)} transitions"
        + (f" with {len(val_states)} validation samples" if len(val_states) else "")
    )
    history = []
    best_weights = agent.nn.get_weights()
    best_epoch = 0
    best_score = None
    for epoch in range(epochs):
        losses = []
        for batch_states, batch_residuals in train_dataset:
            losses.append(agent.train_step(batch_states, batch_residuals))
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(agent, normalized_val_states, val_residual_targets)
        score = val_loss if val_loss is not None else train_loss
        if best_score is None or score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_weights = agent.nn.get_weights()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if val_loss is None:
            print(f"Residual Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f}")
        else:
            print(
                f"Residual Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

    ensure_results_dir()
    agent.nn.set_weights(best_weights)
    agent.nn.save(model_path)
    save_normalization(
        stats_path,
        mean,
        std,
        residual_scale=np.array([residual_scale], dtype=np.float32),
        kp=np.array([kp], dtype=np.float32),
        kd=np.array([kd], dtype=np.float32),
    )
    metrics = {
        "dataset_path": str(dataset_path),
        "num_samples": int(len(states)),
        "num_train": int(len(train_states)),
        "num_val": int(len(val_states)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "validation_split": float(validation_split),
        "kp": float(kp),
        "kd": float(kd),
        "residual_scale": float(residual_scale),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "history": history,
        "model_path": str(model_path),
        "stats_path": str(stats_path),
    }
    save_json(metrics_path, metrics)

    print("Saved residual learning artifacts:")
    print(f"  - {model_path}")
    print(f"  - {stats_path}")
    print(f"  - {metrics_path}")
    print(f"Best epoch: {best_epoch}")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset,
        kp=args.kp,
        kd=args.kd,
        residual_scale=args.residual_scale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        seed=args.seed,
        model_path=args.output_model,
        stats_path=args.output_stats,
        metrics_path=args.metrics_output,
    )
