import optuna
import numpy as np
from scripts.main import main
from utils.data_logger import DataLogger

def simulate_controller(kp, kd):
    logger = DataLogger()
    main()  # Assume main() is modified to accept kp/kd
    data = np.load("robot_log.npz")
    return np.mean(np.abs(data['qpos'] - data['ctrl']))

def optimize_controller():
    study = optuna.create_study()
    study.optimize(lambda trial: simulate_controller(
        trial.suggest_float("kp", 50.0, 200.0),
        trial.suggest_float("kd", 5.0, 30.0)
    ), n_trials=100)
    return study.best_params