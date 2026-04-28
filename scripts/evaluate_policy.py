import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mujoco
except ModuleNotFoundError as exc:
    mujoco = None
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

from scripts.main import build_controller, build_trajectory, load_config, run_simulation
from utils.data_logger import DataLogger
from utils.learning_utils import EVALUATION_SUMMARY_PATH, save_json, summarize_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a controller by running a rollout and summarizing it.")
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
        help="Optional video output path override. Pass an empty string to disable video output.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(EVALUATION_SUMMARY_PATH),
        help="Path to save the evaluation summary JSON.",
    )
    parser.add_argument(
        "--rollout-output",
        default=None,
        help="Optional NPZ path to save the evaluation rollout transitions.",
    )
    return parser.parse_args()


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
    controller_config = config.get("controller", {})
    trajectory_config = config.get("trajectory", {})

    robot_name = args.robot or robot_config.get("name", "panda_mj_description")
    robot_variant = args.robot_variant if args.robot_variant is not None else robot_config.get("variant")
    controller_type = args.controller or controller_config.get("type", "pd")
    trajectory_type = args.trajectory or trajectory_config.get("type", "classic")
    duration = float(args.duration if args.duration is not None else config.get("simulation", {}).get("duration", 20.0))
    if args.video is not None:
        video_path = args.video or None
    else:
        video_path = config.get("output", {}).get("video", "results/evaluation.mp4")

    model = load_robot_description(robot_name, variant=robot_variant)
    data = mujoco.MjData(model)
    controller = build_controller(model, controller_type, controller_config.get("params", {}))
    trajectory = build_trajectory(trajectory_type, trajectory_config.get("params", {}))
    logger = DataLogger()

    run_simulation(model, data, controller, trajectory, duration, logger, video_path=video_path)

    rollout = logger.as_arrays()
    summary = summarize_dataset(rollout)
    summary.update(
        {
            "robot": robot_name,
            "robot_variant": robot_variant,
            "controller_type": controller_type,
            "trajectory_type": trajectory_type,
            "video_path": str(video_path) if video_path else None,
            "rollout_output": str(args.rollout_output) if args.rollout_output else None,
        }
    )
    save_json(args.summary_output, summary)
    if args.rollout_output:
        logger.save(args.rollout_output)

    print(f"Evaluation summary saved to {args.summary_output}")
    if args.rollout_output:
        print(f"Rollout saved to {args.rollout_output}")
    print(f"Total reward: {summary['total_reward']:.4f}")
    print(f"Mean reward: {summary['mean_reward']:.4f}")
    print(f"Mean tracking error: {summary['mean_tracking_error_l2']:.4f}")
    print(f"Final tracking error: {summary['final_tracking_error_l2']:.4f}")


if __name__ == "__main__":
    main()
