import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "glfw")

import imageio.v2 as imageio
import numpy as np
import yaml

try:
    import mujoco
    from mujoco import (
        MjrContext,
        MjrRect,
        MjvCamera,
        MjvPerturb,
        MjvScene,
        mjr_readPixels,
        mjr_render,
        mjtFontScale,
        mjv_updateScene,
    )
except ModuleNotFoundError as exc:
    mujoco = None
    MjrContext = MjrRect = MjvCamera = MjvPerturb = MjvScene = None
    mjr_readPixels = mjr_render = mjtFontScale = mjv_updateScene = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

try:
    from robot_descriptions.loaders.mujoco import load_robot_description
except ModuleNotFoundError as exc:
    load_robot_description = None
    ROBOT_DESCRIPTIONS_IMPORT_ERROR = exc
else:
    ROBOT_DESCRIPTIONS_IMPORT_ERROR = None

from utils.data_logger import DataLogger
from utils.learning_utils import (
    ACTION_DIM,
    DATASET_PATH,
    build_state_from_data,
    compute_tracking_reward,
    ensure_results_dir,
    ensure_parent_dir,
    summarize_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a learning dataset from MuJoCo simulation.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to the YAML config file. Defaults to configs/config.yaml.",
    )
    parser.add_argument("--robot", default=None, help="Robot description override.")
    parser.add_argument("--robot-variant", default=None, help="Robot variant override.")
    parser.add_argument("--controller", choices=("pd", "rl", "nn_pd"), default=None, help="Controller override.")
    parser.add_argument("--trajectory", choices=("classic", "neural"), default=None, help="Trajectory override.")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration override in seconds.")
    parser.add_argument(
        "--video",
        default=None,
        help="Video output path override. Pass an empty string to disable video output.",
    )
    parser.add_argument(
        "--dataset-output",
        default=str(DATASET_PATH),
        help="Path for the generated dataset. Defaults to results/dataset.npz.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def build_controller(model, controller_type, controller_params):
    if controller_type == "pd":
        from controllers.pd_controller import PDController

        return PDController(model, **controller_params)
    if controller_type == "rl":
        from controllers.rl_controller import RLController

        return RLController(model)
    if controller_type == "nn_pd":
        from controllers.nn_compensated_pd import NNCompensatedPD

        return NNCompensatedPD(model, **controller_params)
    raise ValueError(f"Unknown controller type: {controller_type}")


def build_trajectory(trajectory_type, trajectory_params):
    if trajectory_type == "classic":
        from trajectories.trajectory_generator import TrajectoryGenerator

        return TrajectoryGenerator(**trajectory_params)
    if trajectory_type == "neural":
        from trajectories.neural_trajectory import NeuralTrajectoryGenerator

        return NeuralTrajectoryGenerator()
    raise ValueError(f"Unknown trajectory type: {trajectory_type}")


def run_simulation(model, data, controller, trajectory, duration, logger, video_path=None):
    width, height = 640, 480
    sim_dt = max(float(model.opt.timestep), 1e-3)
    total_steps = max(1, int(math.ceil(duration / sim_dt)))
    render_every = max(1, int(round(1.0 / (30.0 * sim_dt))))

    can_render = video_path is not None
    camera = None
    perturb = None
    scene = None
    context = None
    viewport = None
    video_writer = None

    if can_render:
        try:
            camera = MjvCamera()
            perturb = MjvPerturb()
            scene = MjvScene(model, maxgeom=1000)
            context = MjrContext(model, mjtFontScale.mjFONTSCALE_150)
            viewport = MjrRect(0, 0, width, height)
            video_path = ensure_parent_dir(video_path)
            video_fps = max(1, int(round(1.0 / (sim_dt * render_every))))
            video_writer = imageio.get_writer(str(video_path), fps=video_fps, codec="libx264")
        except Exception as exc:
            print(f"[Warning] Rendering disabled: {exc}")
            can_render = False

    try:
        for step in range(total_steps):
            current_time = step * sim_dt
            next_time = min((step + 1) * sim_dt, duration)

            target = np.asarray(trajectory.generate(current_time), dtype=np.float32)[:ACTION_DIM]
            state = build_state_from_data(data, target)

            action = np.asarray(controller.compute(data, target), dtype=np.float32).reshape(-1)[:ACTION_DIM]
            data.ctrl[:] = 0.0
            data.ctrl[:ACTION_DIM] = action

            mujoco.mj_step(model, data)

            next_target = np.asarray(trajectory.generate(next_time), dtype=np.float32)[:ACTION_DIM]
            next_state = build_state_from_data(data, next_target)
            reward = compute_tracking_reward(data.qpos[:ACTION_DIM], data.qvel[:ACTION_DIM], next_target)
            done = step == total_steps - 1

            logger.log_transition(current_time, state, action, next_state, reward, done)

            if can_render and step % render_every == 0:
                try:
                    mjv_updateScene(model, data, None, camera, perturb, scene)
                    mjr_render(context, viewport, scene)
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    depth = np.zeros((height, width), dtype=np.float32)
                    mjr_readPixels(frame, depth, viewport)
                    video_writer.append_data(frame)
                except Exception as exc:
                    print(f"[Warning] Frame rendering failed: {exc}")
                    can_render = False
    finally:
        if video_writer is not None:
            video_writer.close()


def main():
    args = parse_args()
    if MUJOCO_IMPORT_ERROR is not None:
        raise SystemExit("MuJoCo is not installed in the active environment. Run `pip install -r requirements.txt` first.")
    if ROBOT_DESCRIPTIONS_IMPORT_ERROR is not None:
        raise SystemExit(
            "`robot-descriptions` is not installed in the active environment. Run `pip install -r requirements.txt` first."
        )
    config = load_config(args.config)

    robot_config = config.get("robot", {})
    model = load_robot_description(
        args.robot or robot_config.get("name", "panda_mj_description"),
        variant=args.robot_variant if args.robot_variant is not None else robot_config.get("variant"),
    )

    data = mujoco.MjData(model)
    logger = DataLogger()

    controller_config = config.get("controller", {})
    controller_type = args.controller or controller_config.get("type", "pd")
    controller_params = controller_config.get("params", {})
    controller = build_controller(model, controller_type, controller_params)

    trajectory_config = config.get("trajectory", {})
    trajectory_type = args.trajectory or trajectory_config.get("type", "classic")
    trajectory_params = trajectory_config.get("params", {})
    trajectory = build_trajectory(trajectory_type, trajectory_params)

    duration = float(args.duration if args.duration is not None else config.get("simulation", {}).get("duration", 20.0))
    if args.video is not None:
        video_file = args.video or None
    else:
        video_file = config.get("output", {}).get("video", "results/simulation.mp4")

    run_simulation(model, data, controller, trajectory, duration, logger, video_path=video_file)

    ensure_results_dir()
    dataset_path = logger.save(args.dataset_output)
    summary = summarize_dataset(logger.as_arrays())
    print(f"Dataset saved to {dataset_path}")
    print(f"Total reward: {summary['total_reward']:.4f}")
    print(f"Mean reward: {summary['mean_reward']:.4f}")
    print(f"Final tracking error: {summary['final_tracking_error_l2']:.4f}")


if __name__ == "__main__":
    main()
