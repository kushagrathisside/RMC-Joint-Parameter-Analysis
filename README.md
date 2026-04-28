# Robotic Control Framework with Learning-Based Pipeline

A modular MuJoCo-based framework for designing, training, and evaluating robotic control policies.
This project integrates classical control, imitation learning, residual learning, and reinforcement learning into a unified pipeline built around a consistent **21D state representation**:

```
[qpos(7), qvel(7), target(7)]
```

It is designed for rapid experimentation, reproducibility, and extensibility in robotics and AI-driven control systems.

## Visualization Preview
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/f049d346-b217-47ff-8676-4b841e774119" />
## Quick Start

Use the default safe path first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/main.py --controller pd --trajectory classic --duration 20
python scripts/plot_results.py --input results/dataset.npz
```

That generates a dataset with the classical PD controller, then plots the logged joint positions, torques, and velocities.

---

# 🚀 Features

### Control Methods

* Classical **PD control**
* **Direct torque neural policies**
* **PD + neural residual compensation**
* Adaptive policies trained via imitation and reinforcement learning

### Learning Capabilities

* Dataset generation from simulation
* Behavior Cloning (BC)
* Residual learning (model correction)
* Minimal PPO-style actor warm start
* Hyperparameter tuning via genetic algorithms (Optuna)

### Tooling

* Qt5-based interactive visualizer with recording
* MuJoCo simulation with off-screen rendering
* Dataset logging in RL-compatible format
* Plotting and video generation utilities

---

# 🧠 System Overview

The framework follows a structured pipeline:

```
trajectory → controller → MuJoCo simulation → dataset → training → policy
```

State representation:

```
state = [qpos, qvel, target]  → 21D
action = torque (7D)
```

---

# 📁 Repository Structure

```
project/
├── configs/               # YAML configuration files
├── controllers/           # Control algorithms
│   ├── base_controller.py
│   ├── pd_controller.py
│   ├── rl_controller.py
│   └── nn_compensated_pd.py
├── docs/                  # Architecture, config, runtime, and visualization guides
├── trajectories/          # Motion generators
├── utils/
│   ├── data_logger.py     # Transition dataset logger
│   ├── ga_optimizer.py    # Gain optimization (Optuna)
│   └── learning_utils.py  # Shared state/reward/artifact helpers
├── scripts/
│   ├── evaluate_policy.py # Rollout-based controller evaluation
│   ├── main.py            # Simulation + dataset generation
│   ├── train_bc.py        # Behavior cloning
│   ├── train_residual.py  # Residual learning
│   ├── train_rl.py        # PPO-style actor warm start
│   ├── visualize_robot.py # Qt5 visualizer
│   ├── plot_results.py    # Plotting utilities
│   └── generate_video.py  # Legacy animation helper for older logs
├── tests/                 # Unit tests
└── results/               # Outputs (datasets, models, videos)
```

---

# ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Main dependencies

* mujoco
* numpy
* tensorflow
* gymnasium
* optuna
* imageio
* matplotlib
* PyQt5
* PyYAML
* robot-descriptions

---

# ⚙️ Configuration

All runtime behavior is defined in:

```
configs/config.yaml
```

Example:

```yaml
robot:
  name: panda_mj_description
  variant: scene

controller:
  type: pd  # pd | rl | nn_pd
  params:
    kp: 150.0
    kd: 20.0

trajectory:
  type: classic  # classic | neural
  params:
    amplitude: 1.0
    frequency: 0.5

simulation:
  duration: 20

output:
  video: results/simulation.mp4
```

Notes:

* `controller.params` are used by `pd` and for residual-model training; `rl` ignores them
* `trajectory.params` are used by `classic`; `neural` ignores them
* `rl` and `nn_pd` now fail fast if their trained artifacts are missing

---

# ▶️ Simulation & Dataset Generation

Run:

```bash
python scripts/main.py
```

### What happens:

1. Load robot model (MuJoCo)
2. Build controller + trajectory
3. Step simulation
4. Compute reward:

   ```
   reward = -||qpos - target||² - 0.01||qvel||
   ```
5. Log transitions:

   ```
   (time, state, action, next_state, reward, done)
   ```
6. Save dataset:

   ```
   results/dataset.npz
   ```
7. Optionally save render video:

   ```
   results/simulation.mp4
   ```

### Dataset format

| Key        | Shape   |
| ---------- | ------- |
| time       | (N,)    |
| state      | (N, 21) |
| action     | (N, 7)  |
| next_state | (N, 21) |
| reward     | (N,)    |
| done       | (N,)    |

---

# 🧪 Training Workflow

## 1. Behavior Cloning (Imitation Learning)

```bash
python scripts/train_bc.py
```

Trains:

```
π(state) → torque
```

Outputs:

* `results/policy.h5`
* `results/policy_normalization.npz`
* `results/bc_metrics.json`

---

## 2. Residual Learning (Best Practical Controller)

```bash
python scripts/train_residual.py
```

Learns:

```
residual = true_action - PD_action
```

Controller:

```
torque = PD + residual(state)
```

Outputs:

* `results/residual_policy.h5`
* `results/residual_normalization.npz`
* `results/residual_metrics.json`

---

## 3. PPO-Style Training (Warm Start)

```bash
python scripts/train_rl.py
```

* Lightweight reward-weighted actor training
* Dataset-based approximation of RL, not full on-policy PPO
* Uses a validation split and keeps the best checkpoint by validation score

Outputs:

* `results/ppo_policy.h5`
* `results/ppo_normalization.npz`
* `results/ppo_metrics.json`

---

# 🤖 Controllers

### PDController

Classic baseline:

```
torque = kp * error - kd * velocity
```

---

### RLController

* Direct torque policy
* Input: 21D state
* Output: 7D torque
* Requires `results/policy.h5` or `results/ppo_policy.h5` with matching normalization stats before simulation

---

### NNCompensatedPD

Hybrid controller:

```
torque = PD + neural_residual
```

* Most stable and practical approach
* Learns unmodeled dynamics
* Restores the saved training-time PD gains from `results/residual_normalization.npz`

---

# 🖥️ Visualization

Launch GUI:

```bash
python scripts/visualize_robot.py
```

Features:

* Inspect / trajectory / simulate modes
* Controller switching (pd, rl, nn_pd)
* Camera control
* Snapshot saving
* Recording to `.npz` + `.mp4`

---

# 📊 Plotting

```bash
python scripts/plot_results.py --input results/dataset.npz
```

---

# 📈 Evaluation

```bash
python scripts/evaluate_policy.py --controller pd --trajectory classic --duration 20
```

Outputs:

* `results/evaluation_summary.json`
* optional rollout `.npz` if `--rollout-output` is provided

Legacy helper:

```bash
python scripts/generate_video.py
```

This older script still expects `results/robot_log.npz`, not the current learning dataset.

---

# 🧪 Testing

```bash
pytest -q
```

---

# 🔬 AI Integrations

### Reinforcement Learning

* Policy learning via dataset and PPO-style updates
* Extendable to full RL pipelines

### Neural Residual Learning

* Compensates modeling errors
* Improves stability and accuracy

### Genetic Optimization

* Tune PD gains using Optuna
* Works on logged datasets

---

# ⚠️ Known Limitations

* PPO implementation is minimal (not full on-policy RL)
* Limited test coverage for training pipelines
* Some legacy scripts still use older log formats

---

# 🧠 Recommended Workflow

Best performance path:

```
PD → dataset → behavior cloning → residual learning → PPO fine-tune
```

Avoid:

```
training RL from scratch without prior data
```

---

# 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add changes + tests
4. Submit PR

---

# 📚 Documentation

* [docs/architecture.md](docs/architecture.md)
* [docs/configuration.md](docs/configuration.md)
* [docs/running-the-project.md](docs/running-the-project.md)
* [docs/visualization.md](docs/visualization.md)

---

# 🧭 Summary

This project evolves from a classical robotics simulator into a **learning-driven control system**, enabling:

* fast prototyping
* hybrid control strategies
* data-driven policy learning
* scalable experimentation

---

If you extend this further, the natural next steps are:

* full PPO / SAC implementation
* domain randomization
* sim-to-real transfer

---
