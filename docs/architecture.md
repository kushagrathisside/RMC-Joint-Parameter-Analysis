# Architecture Overview

The project is now organized around a dataset-centered control-learning loop.

## Core idea

All learning-aware components share the same state representation:

```text
[qpos(7), qvel(7), target(7)]
```

This 21D state is used consistently in:

- simulation logging
- behavior cloning training
- residual learning training
- minimal PPO-style training
- RL controller inference
- residual controller inference

## End-to-end flow

The main runtime path lives in [`scripts/main.py`](../scripts/main.py).

At a high level:

1. read `configs/config.yaml`
2. load the MuJoCo robot scene through `robot_descriptions`
3. build the selected controller
4. build the selected trajectory generator
5. construct a 21D state from the current robot state and current target
6. compute a 7D action
7. step MuJoCo
8. construct the next 21D state
9. compute reward
10. log a transition
11. optionally render a frame

The transition record is:

```text
(time, state, action, next_state, reward, done)
```

## Shared utilities

[`utils/learning_utils.py`](../utils/learning_utils.py) centralizes:

- state and action dimensions
- state construction helpers
- reward computation
- dataset/model/statistics artifact paths
- normalization helpers
- policy-network builders

This file is the contract layer that keeps training and inference aligned.

## Module map

### `controllers/`

- [`base_controller.py`](../controllers/base_controller.py)
  Defines the `compute(data, target_pos)` interface.

- [`pd_controller.py`](../controllers/pd_controller.py)
  Classical PD torque controller with helpers for both MuJoCo runtime data and 21D dataset states.

- [`rl_controller.py`](../controllers/rl_controller.py)
  Direct torque policy:
  input `21D -> 7D`
  output is scaled and clipped torque.
  It requires a trained model and matching normalization stats before simulation.

- [`nn_compensated_pd.py`](../controllers/nn_compensated_pd.py)
  Residual controller:

  ```text
  torque = PD + residual(state)
  ```

  The residual network also uses `21D -> 7D`.

### `trajectories/`

- [`trajectory_generator.py`](../trajectories/trajectory_generator.py)
  Handcrafted reference targets.

- [`neural_trajectory.py`](../trajectories/neural_trajectory.py)
  Neural target generator.

### `utils/`

- [`data_logger.py`](../utils/data_logger.py)
  Transition logger for the learning dataset.

- [`ga_optimizer.py`](../utils/ga_optimizer.py)
  Fits PD gains against the logged dataset by minimizing imitation error.

- [`learning_utils.py`](../utils/learning_utils.py)
  Shared learning contract.

### `scripts/`

- [`main.py`](../scripts/main.py)
  Generates the transition dataset and optional simulation video, with CLI overrides for controller, trajectory, duration, video, and dataset output.

- [`train_bc.py`](../scripts/train_bc.py)
  Behavior cloning on `state -> action` with a validation split and saved metrics JSON.

- [`train_residual.py`](../scripts/train_residual.py)
  Residual learning on `state -> (true_action - PD_action)` with a validation split and saved metrics JSON.

- [`train_rl.py`](../scripts/train_rl.py)
  Minimal PPO-style actor training using reward-weighted imitation loss, validation splits, and saved metrics JSON.

- [`evaluate_policy.py`](../scripts/evaluate_policy.py)
  Rollout-based evaluation that writes a summary JSON and optional rollout dataset.

- [`visualize_robot.py`](../scripts/visualize_robot.py)
  Qt5 desktop visualizer for inspection, simulation, and recording.

- [`plot_results.py`](../scripts/plot_results.py)
  Plots either the new transition dataset or the older log format.

### `tests/`

- [`test_pid.py`](../tests/test_pid.py)
  Small PD smoke test.

## Runtime selection behavior

`scripts/main.py` currently recognizes:

- controllers: `pd`, `rl`, `nn_pd`
- trajectories: `classic`, `neural`

## Reward and logging

The main simulation uses:

```text
reward = -||qpos - target||^2 - 0.01||qvel||
```

The dataset saved by `DataLogger` contains:

- `time`
- `state`
- `action`
- `next_state`
- `reward`
- `done`

Default save location:

```text
results/dataset.npz
```

## Training artifact flow

Behavior cloning writes:

- `results/policy.h5`
- `results/policy_normalization.npz`
- `results/bc_metrics.json`

Residual learning writes:

- `results/residual_policy.h5`
- `results/residual_normalization.npz`
- `results/residual_metrics.json`

Minimal PPO-style training writes:

- `results/ppo_policy.h5`
- `results/ppo_normalization.npz`
- `results/ppo_metrics.json`

Evaluation writes:

- `results/evaluation_summary.json`

Those artifacts are then picked up automatically by the runtime controllers.

## Current status

Working pipeline:

- dataset generation from simulation
- behavior cloning
- residual learning
- direct torque policy inference
- residual controller inference
- minimal PPO-style actor training

Still intentionally lightweight:

- PPO is not yet a full on-policy rollout/update implementation
- tests are still lightweight rather than end-to-end simulation coverage
