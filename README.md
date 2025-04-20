# Robot Dog DRL Training

This project implements a Deep Reinforcement Learning (DRL) system for training a quadruped robot dog in procedurally generated terrains. The system combines terrain generation, physics simulation, and reinforcement learning to train a robot dog to navigate complex 3D environments.

## Features

- Procedural terrain generation using fractal noise
- Biome-based environment variation
- Physics-based robot simulation using PyBullet
- Reinforcement learning using Stable Baselines3
- Support for parallel environment training
- TensorBoard integration for training visualization

## Requirements

- Python 3.8+
- numpy<2.0.0
- gymnasium>=0.29.1
- pybullet>=3.2.0
- torch==2.2.1+cpu
- opencv-python-headless>=4.8.0
- scipy>=1.11.0
- protobuf==3.20.3
- tensorboard>=2.14.0
- stable-baselines3

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `terrain_generator.py`: Implements procedural terrain generation using fractal noise
- `robot_model.py`: Defines the robot model and its physics simulation
- `minecraft_env.py`: Implements the Gymnasium environment interface for the robot dog
- `configs/default.yaml`: Default training and environment configuration
- `run.py`: Orchestrates setup, launches TensorBoard, and starts training
- `start_training.py`: Convenience script to set URDF path and launch training
- `train.py`: Main training script using PPO algorithm
- `robot_dog.urdf`: Robot model description file

## Usage

1. Create and activate a virtual environment:
```bash
python -m venv .venv
# Linux / macOS / WSL
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your robot URDF file in the project root as `robot_dog.urdf`

4. Run training with GUI (default):
```bash
python run.py
```

5. Run training without PyBullet GUI:
```bash
python run.py --no-gui
```

Alternatively, use the convenience script:
```bash
python start_training.py
```

TensorBoard will auto-open at: http://localhost:6006

## Training Configuration

The training script uses the following key parameters:
- Number of parallel environments: 4
- Total training timesteps: 1,000,000
- Learning rate: 3e-4
- Batch size: 64
- Number of epochs: 10
- Gamma: 0.99
- GAE Lambda: 0.95

## Environment Details

The environment provides the following observations:
- Joint positions
- Joint velocities
- IMU orientation
- IMU angular velocity
- IMU linear acceleration
- Foot contact states

Reward function components:
- Distance to target
- Stability (based on IMU orientation)
- Energy efficiency (based on joint torques)
- Foot contact patterns

## License

MIT License

## Acknowledgments

This project is inspired by recent advances in:
- Procedural content generation
- Physics-based simulation
- Deep reinforcement learning
- Robot locomotion 
