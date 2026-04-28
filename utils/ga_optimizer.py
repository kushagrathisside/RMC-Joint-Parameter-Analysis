import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.pd_controller import PDController
from utils.learning_utils import DATASET_PATH, GA_OPTIMIZATION_PATH, load_dataset, save_json


def simulate_controller(kp, kd, dataset_path=DATASET_PATH):
    dataset = load_dataset(dataset_path)
    states = dataset["state"]
    target_actions = dataset["action"]

    controller = PDController(model=None, kp=kp, kd=kd)
    pd_actions = np.stack([controller.compute_from_state(state) for state in states], axis=0)
    imitation_error = np.mean(np.square(pd_actions - target_actions))
    return float(imitation_error)


def optimize_controller(dataset_path=DATASET_PATH, n_trials=100):
    import optuna

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: simulate_controller(
            trial.suggest_float("kp", 50.0, 200.0),
            trial.suggest_float("kd", 5.0, 30.0),
            dataset_path=dataset_path,
        ),
        n_trials=n_trials,
    )
    return {
        "kp": float(study.best_params["kp"]),
        "kd": float(study.best_params["kd"]),
        "best_objective": float(study.best_value),
        "n_trials": int(n_trials),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize PD gains against a logged dataset using Optuna.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the dataset NPZ file.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the TPE sampler.")
    parser.add_argument(
        "--output",
        default=str(GA_OPTIMIZATION_PATH),
        help="Path to save the best gain search result as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import optuna

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: simulate_controller(
            trial.suggest_float("kp", 50.0, 200.0),
            trial.suggest_float("kd", 5.0, 30.0),
            dataset_path=args.dataset,
        ),
        n_trials=args.n_trials,
    )
    result = {
        "dataset_path": str(args.dataset),
        "kp": float(study.best_params["kp"]),
        "kd": float(study.best_params["kd"]),
        "best_objective": float(study.best_value),
        "n_trials": int(args.n_trials),
        "seed": int(args.seed),
    }
    save_json(args.output, result)
    print(f"Optimization result saved to {args.output}")
    print(f"Best kp: {result['kp']:.4f}")
    print(f"Best kd: {result['kd']:.4f}")
    print(f"Best imitation error: {result['best_objective']:.6f}")


if __name__ == "__main__":
    main()
