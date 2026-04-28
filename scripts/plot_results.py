import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.learning_utils import ensure_parent_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a dataset or legacy robot log file.")
    parser.add_argument(
        "--input",
        default="results/dataset.npz",
        help="Path to the input NPZ file. Defaults to results/dataset.npz.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the figure instead of only displaying it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plotting window.",
    )
    return parser.parse_args()


def plot_results(filename="results/dataset.npz", output_path=None, show=True):
    data = np.load(filename)

    if "state" in data.files and "action" in data.files:
        time = data["time"] - data["time"][0]
        qpos = data["state"][:, :7]
        qvel = data["state"][:, 7:14]
        ctrl = data["action"]
    else:
        time = data["time"] - data["time"][0]
        qpos = np.asarray(data["qpos"])
        qvel = np.asarray(data["qvel"])
        ctrl = np.asarray(data["ctrl"])

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    for i in range(7):
        plt.plot(time, qpos[:, i])
    plt.title("Joint Positions")

    plt.subplot(3, 1, 2)
    for i in range(7):
        plt.plot(time, ctrl[:, i])
    plt.title("Control Signals")

    plt.subplot(3, 1, 3)
    for i in range(7):
        plt.plot(time, qvel[:, i])
    plt.title("Joint Velocities")

    plt.tight_layout()
    if output_path:
        ensure_parent_dir(output_path)
        plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    plot_results(filename=args.input, output_path=args.output, show=not args.no_show)
