# Robotic Control Framework with AI Integration

A modular and extensible framework for designing, implementing, and benchmarking advanced robotic control algorithms. This project seamlessly combines proven classical control theory with cutting-edge AI methods, providing researchers and developers with the tools to:
- Rapidly prototype PD and PID controllers.
- Integrate Reinforcement Learning (RL) policies for adaptive gain tuning.
- Augment traditional controllers with neural network compensation for residual dynamics.
- Generate and compare handcrafted versus learned trajectories.
- Optimize hyperparameters automatically via genetic algorithms.

By following modern documentation standards and providing clear examples, this repository empowers users to understand, extend, and contribute to each component of the system.

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [Core Components](#core-components)
  - [Controllers](#controllers)
  - [Trajectories](#trajectories)
  - [Data Handling](#data-handling)
- [AI Integrations](#ai-integrations)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Neural Network Compensation](#neural-network-compensation)
  - [Genetic Algorithm Optimization](#genetic-algorithm-optimization)
- [How to Run Demos](#how-to-run-demos)
- [Visualization](#visualization)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Overview

This framework is built to accelerate development and evaluation of robotic controllers by providing a standardized pipeline for configuration, simulation, data logging, and result visualization. Whether you need a classical PD baseline or an AI-driven controller, the modular architecture allows you to swap components with minimal code changes. Clear interfaces, configuration-driven parameters, and extensive examples ensure that new users can quickly get up to speed, while advanced users can focus on custom algorithm development.

---

## Installation

1. **Prerequisites:**
   - Python ≥3.8
   - MuJoCo ≥2.3.3 (for simulation)
   - [Gymnasium](https://gymnasium.farama.org) for RL environments

2. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/robotic-control-framework.git
   cd robotic-control-framework
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import mujoco, torch, optuna"
   ```

---

## Quick Start

Follow these steps to run a simple PD controller on a predefined trajectory:

```bash
# Set controller and trajectory types
sed -i 's/controller.type: .*/controller.type: pd/' configs/config.yaml
sed -i 's/trajectory.type: .*/trajectory.type: classic/' configs/config.yaml

# Launch simulation
python scripts/main.py

# Generate plots
python scripts/plot_results.py
```

Expect to see time-series plots saved in the `results/` folder, illustrating controller performance metrics such as overshoot, settling time, and energy consumption.

---

## File Structure

```plaintext
project/
├── configs/               # YAML configuration files
│   └── config.yaml        # Centralized parameters
├── controllers/           # Core control algorithms
│   ├── base_controller.py # Abstract controller class
│   ├── pd_controller.py   # PD control logic
│   ├── rl_controller.py   # RL-based adaptive controller
│   └── nn_compensated_pd.py # NN-augmented PD controller
├── trajectories/          # Motion generators
│   ├── trajectory_generator.py # Handcrafted paths
│   └── neural_trajectory.py    # AI-generated trajectories
├── utils/                 # Utilities
│   ├── data_logger.py     # Data recording/export
│   └── ga_optimizer.py    # Genetic algorithm tuning
├── scripts/               # Executable scripts
│   ├── main.py            # Simulation launcher
│   ├── train_rl.py        # RL training pipeline
│   └── plot_results.py    # Data visualization
└── tests/                 # Unit tests
    └── test_pid.py        # PD controller validation
```

### Directory Descriptions

- **configs/**: Configuration files for controllers, trajectories, and simulations.
- **controllers/**: Core control algorithm implementations.
- **trajectories/**: Motion generation modules.
- **utils/**: Helper modules like data logging and optimization.
- **scripts/**: Scripts to run simulations, train models, and visualize outputs.
- **tests/**: Unit and integration tests.

---

## Core Components

### Controllers

Defines a standardized interface for control algorithms. Each controller implements a `compute(state, reference)` method that outputs actuation commands.

- **PDController**:
  - Implements a classic Proportional-Derivative controller.
  - Parameters:
    - `kp` (Proportional Gain): Controls reaction to position error.
    - `kd` (Derivative Gain): Reduces velocity oscillations.

### Trajectories

Provides both handcrafted and learned path generators:

- **TrajectoryGenerator**:
  - Offers predefined motion patterns such as sinusoidal waves, Bezier curves, and step inputs.
- **NeuralTrajectoryGenerator**:
  - Predicts time-indexed joint targets using neural networks for complex learned behaviors.

### Data Handling

Records and manages simulation data:

- **DataLogger**:
  - Captures joint positions (`qpos`), velocities (`qvel`), and torques (`ctrl`).
  - Supports export to NPZ or CSV formats.

---

## AI Integrations

### Reinforcement Learning

- **RLController**:
  - Dynamically tunes controller gains during simulation.
  - Train using Gymnasium environments:
    ```bash
    python scripts/train_rl.py --env CartPole-v1 --episodes 5000
    ```

### Neural Network Compensation

- **NNCompensatedPD**:
  - Augments classical PD output by adding a neural network-predicted compensation signal.

### Genetic Algorithm Optimization

- **ga_optimizer.py**:
  - Automatically searches optimal `kp` and `kd` values using Optuna.

---

## How to Run Demos

1. **Neural Trajectories:**
   ```bash
   sed -i 's/trajectory.type: .*/trajectory.type: neural/' configs/config.yaml
   python scripts/main.py
   ```

2. **RL Controller:**
   ```bash
   sed -i 's/controller.type: .*/controller.type: rl/' configs/config.yaml
   python scripts/main.py
   ```

3. **NN-Compensated PD:**
   ```bash
   sed -i 's/controller.type: .*/controller.type: nn_pd/' configs/config.yaml
   python scripts/main.py
   ```

---

## Visualization

Generate performance plots:

```bash
python scripts/plot_results.py --output_dir results/
```

Plots include position tracking error, control torques, and learning curves for RL.

---

## Testing

Run unit tests to validate implementation correctness:

```bash
pytest tests/ --maxfail=1 --disable-warnings -q
```

Tests are automatically triggered in CI/CD pipelines.

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Add your changes and ensure tests pass.
4. Submit a Pull Request with a detailed description.
5. Follow the [Code of Conduct](CODE_OF_CONDUCT.md) and [Contribution Guidelines](CONTRIBUTING.md).

