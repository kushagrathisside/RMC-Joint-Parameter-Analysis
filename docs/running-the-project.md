# Running the Project

This guide focuses on the current learning pipeline.

## 1. Install dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need a host environment that can run MuJoCo and, for desktop viewing, GLFW/OpenGL.

## 2. Choose a configuration

Edit [`configs/config.yaml`](../configs/config.yaml).

Recommended starting point:

- `robot.variant: scene`
- `controller.type: pd`
- `trajectory.type: classic`

Example:

```yaml
robot:
  name: panda_mj_description
  variant: scene

controller:
  type: pd
  params:
    kp: 150.0
    kd: 20.0

trajectory:
  type: classic
  params:
    amplitude: 1.0
    frequency: 0.5

simulation:
  duration: 20
```

## 3. Generate a dataset

```bash
python scripts/main.py
```

Useful overrides:

```bash
python scripts/main.py --controller pd --trajectory classic --duration 20
python scripts/main.py --video '' --dataset-output results/pd_dataset.npz
```

Expected behavior:

- MuJoCo loads the Panda task scene
- the selected controller computes a 7D action
- the selected trajectory generator produces a 7D target
- the simulator builds 21D states
- transitions are logged to a dataset
- optional video frames are rendered

Main outputs:

- learning dataset:

```text
results/dataset.npz
```

- render video, if enabled:

```text
results/simulation.mp4
```

## 4. Train a policy

### Behavior cloning

```bash
python scripts/train_bc.py
```

Outputs:

- `results/policy.h5`
- `results/policy_normalization.npz`
- `results/bc_metrics.json`

### Residual learning

```bash
python scripts/train_residual.py --kp 150 --kd 20
```

Outputs:

- `results/residual_policy.h5`
- `results/residual_normalization.npz`
- `results/residual_metrics.json`

### Minimal PPO-style actor

```bash
python scripts/train_rl.py
```

Outputs:

- `results/ppo_policy.h5`
- `results/ppo_normalization.npz`
- `results/ppo_metrics.json`

All three training scripts support `--validation-split`, `--epochs`, `--batch-size`, `--learning-rate`, and output-path overrides.

## 5. Run learned controllers

After training, switch the controller type in the config and rerun simulation or the visualizer.

Examples:

- `controller.type: rl`
  Uses the direct torque policy and requires `policy.h5` plus `policy_normalization.npz`, or `ppo_policy.h5` plus `ppo_normalization.npz`

- `controller.type: nn_pd`
  Uses PD plus a learned residual and requires `residual_policy.h5` plus `residual_normalization.npz`

## 6. Evaluate a controller rollout

```bash
python scripts/evaluate_policy.py --controller pd --trajectory classic --duration 20
python scripts/evaluate_policy.py --controller rl --trajectory neural --summary-output results/rl_eval.json
```

Main outputs:

- `results/evaluation_summary.json`
- optional rollout `.npz` if `--rollout-output` is passed

## 7. Open the Qt5 viewer

```bash
python scripts/visualize_robot.py
```

Useful launch examples:

```bash
python scripts/visualize_robot.py --mode inspect
python scripts/visualize_robot.py --mode simulate --controller pd --trajectory classic --duration 20
python scripts/visualize_robot.py --mode simulate --controller nn_pd --trajectory classic --duration 20
python scripts/visualize_robot.py --mode simulate --controller rl --trajectory neural --duration 20
```

Inside the window you can:

- switch between `inspect`, `trajectory`, and `simulate`
- select `pd`, `rl`, or `nn_pd`
- tune playback rate and duration
- adjust camera azimuth, elevation, and distance
- save a snapshot
- record a viewer session to `.npz` and `.mp4`

Note: viewer recordings are separate from `results/dataset.npz`.

## 8. Plot the dataset

```bash
python scripts/plot_results.py --input results/dataset.npz
python scripts/plot_results.py --input results/dataset.npz --output results/dataset_plot.png --no-show
```

This plots the first 7 position, velocity, and action channels from the logged dataset.

## 9. Tune PD gains with Optuna

```bash
python utils/ga_optimizer.py --dataset results/dataset.npz --n-trials 100
```

This writes the best gain search result to `results/ga_optimization.json` by default.

## 10. Legacy animation helper

`scripts/generate_video.py` still targets the older sample-log file:

```text
results/robot_log.npz
```

It is useful only if you want to animate the older repository sample log.

## 11. Run tests

```bash
pytest -q
```

What to expect:

- the current tests cover PD-controller math plus lightweight dataset helper behavior
- full training/runtime execution still depends on the MuJoCo and TensorFlow stack being installed

## Troubleshooting

### `ModuleNotFoundError: No module named 'robot_descriptions'`

Install the dependency from `requirements.txt` in the active environment.

### `ModuleNotFoundError: No module named 'tensorflow'`

Install the ML dependencies from `requirements.txt` before running `train_bc.py`, `train_residual.py`, `train_rl.py`, or learned controllers.

### A trained controller seems unchanged

Check that the expected artifact files exist in `results/`:

- `policy.h5` and `policy_normalization.npz` for `rl`
- `residual_policy.h5` and `residual_normalization.npz` for `nn_pd`
- `ppo_policy.h5` and `ppo_normalization.npz` for PPO-style actor loading

### `FileNotFoundError` when selecting `rl` or `nn_pd`

The runtime now fails fast instead of silently using random learned-controller weights.

Make sure the matching artifacts exist in `results/`:

- `policy.h5` and `policy_normalization.npz`, or `ppo_policy.h5` and `ppo_normalization.npz`, for `rl`
- `residual_policy.h5` and `residual_normalization.npz` for `nn_pd`

### No simulation MP4 is produced

The simulation can still succeed if MuJoCo rendering setup fails. Check OpenGL/GLFW availability in the environment.
