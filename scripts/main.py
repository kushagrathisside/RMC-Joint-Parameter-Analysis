import os
# Force MuJoCo to use GLFW for off-screen rendering (GPU-supported or CPU fallback)
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
import time
import yaml
from glfw import GLFWError
import imageio

from robot_descriptions.loaders.mujoco import load_robot_description
from controllers.pd_controller import PDController
from controllers.rl_controller import RLController
from trajectories.trajectory_generator import TrajectoryGenerator
from trajectories.neural_trajectory import NeuralTrajectoryGenerator
from utils.data_logger import DataLogger

# Off-screen rendering setup via Mjv and Mjr APIs
from mujoco import MjvCamera, MjvPerturb, MjvScene, MjrContext, mjv_updateScene, mjr_render, mjr_readPixels, MjrRect, mjtFontScale


def run_simulation(model, data, controller, trajectory, duration, logger, video_path=None):
    start_time = time.time()
    frames = []
    W, H = 640, 480
    can_render = True

    try:
        camera = MjvCamera()
        perturb = MjvPerturb()
        scene = MjvScene(model, maxgeom=1000)
        context = MjrContext(model, mjtFontScale.mjFONTSCALE_150)
        viewport = MjrRect(0, 0, W, H)
    except Exception as e:
        print(f"[Warning] Rendering disabled: {e}")
        can_render = False

    while time.time() - start_time < duration:
        t = time.time() - start_time
        target_pos = trajectory.generate(t)
        data.ctrl[:7] = controller.compute(data, target_pos)
        mujoco.mj_step(model, data)
        logger.record(data)

        if can_render:
            try:
                mjv_updateScene(model, data, None, camera, perturb, scene)
                mjr_render(context, viewport, scene)
                frame = np.zeros((H, W, 3), dtype=np.uint8)
                depth = np.zeros((H, W), dtype=np.float32)
                mjr_readPixels(frame, depth, viewport)
                frames.append(frame)
            except Exception as render_error:
                print(f"[Warning] Frame rendering failed: {render_error}")

        time.sleep(0.001)

    if video_path and frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=30, codec='libx264')


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model = load_robot_description(config.get("robot", {}).get("name", "panda_mj_description"))
    data = mujoco.MjData(model)
    logger = DataLogger()

    ctype = config["controller"]["type"]
    if ctype == "pd":
        controller = PDController(model, **config["controller"]["params"])
    elif ctype == "rl":
        controller = RLController(model)
    else:
        raise ValueError(f"Unknown controller type: {ctype}")

    ttype = config["trajectory"]["type"]
    if ttype == "classic":
        trajectory = TrajectoryGenerator(**config["trajectory"]["params"])
    elif ttype == "neural":
        trajectory = NeuralTrajectoryGenerator()
    else:
        raise ValueError(f"Unknown trajectory type: {ttype}")

    duration = config["simulation"]["duration"]
    video_file = config.get("output", {}).get("video", "results/simulation.mp4")
    run_simulation(model, data, controller, trajectory, duration, logger, video_path=video_file)

    logger.save()


if __name__ == "__main__":
    main()