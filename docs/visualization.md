# Visualization Guide

The repository includes a Qt5-based desktop visualizer at [`scripts/visualize_robot.py`](../scripts/visualize_robot.py).

It is intended for local desktop use when you want to inspect the Panda task scene, watch controller behavior, and record sessions without leaving the repo workflow.

## What the tool does

The Qt5 tool gives you:

- a live MuJoCo render panel
- robot and variant selection
- mode selection: `inspect`, `trajectory`, `simulate`
- controller selection: `pd`, `rl`, `nn_pd`
- trajectory selection: `classic`, `neural`
- playback rate and duration controls
- camera azimuth, elevation, and distance controls
- play, pause, reset, reload-config, and snapshot actions
- recording to timestamped `.npz` and `.mp4`

## Launching the GUI

Run from the repository root:

```bash
python scripts/visualize_robot.py
```

The window starts with values from `configs/config.yaml`.

## Useful launch examples

Static inspection:

```bash
python scripts/visualize_robot.py --mode inspect
```

Reference motion only:

```bash
python scripts/visualize_robot.py --mode trajectory --trajectory classic --duration 20
```

PD simulation:

```bash
python scripts/visualize_robot.py --mode simulate --controller pd --trajectory classic --duration 20
```

Residual controller simulation:

```bash
python scripts/visualize_robot.py --mode simulate --controller nn_pd --trajectory classic --duration 20
```

Direct torque RL simulation:

```bash
python scripts/visualize_robot.py --mode simulate --controller rl --trajectory neural --duration 20
```

## Recording behavior

Use the `Start Recording` / `Stop Recording` button in the GUI.

When recording is active, the visualizer stores:

- `time`
- `qpos`
- `qvel`
- `ctrl`
- rendered frames

On stop, it writes paired timestamped files under `results/`:

- `visualizer_recording_*.npz`
- `visualizer_recording_*.mp4`

Important distinction:

- viewer recordings are for inspection and playback
- `scripts/main.py` is what writes the learning dataset `results/dataset.npz`

## How it fits the learning pipeline

- The visualizer reads the same YAML configuration structure as `scripts/main.py`
- `simulate` mode uses the selected runtime controller
- `rl` requires `policy.h5` plus `policy_normalization.npz`, or `ppo_policy.h5` plus `ppo_normalization.npz`
- `nn_pd` requires `residual_policy.h5` plus `residual_normalization.npz`

This makes the visualizer a convenient way to inspect trained policies after running the training scripts.

## Available CLI options

- `--config`
  Path to the YAML config file

- `--mode`
  Initial mode: `inspect`, `trajectory`, or `simulate`

- `--robot`
  Initial robot description name

- `--robot-variant`
  Initial robot variant, for example `scene`

- `--controller`
  Initial controller selection: `pd`, `rl`, or `nn_pd`

- `--trajectory`
  Initial trajectory selection: `classic` or `neural`

- `--duration`
  Initial duration in seconds. `0` means continuous playback

- `--realtime-rate`
  Initial speed multiplier

- `--camera-distance`
- `--camera-azimuth`
- `--camera-elevation`
  Initial camera values

## Practical notes

- The current control stack assumes a 7-joint arm-style layout, so Panda remains the best-supported target.
- The default config points Panda at `variant: scene`, which gives you the grounded task scene with floor, table, and ball.
- The tool requires `PyQt5`, `mujoco`, and `robot-descriptions` in the active environment.
- The script defaults `MUJOCO_GL` to `glfw`, which is a sensible default for local Ubuntu desktop use.
