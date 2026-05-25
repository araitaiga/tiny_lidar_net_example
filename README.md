# tiny-lidar-net-example

A minimal example of self-driving via 2D LiDAR simulation and supervised learning.
Educational code that walks through how **TinyLidarNet** (1D CNN) works.

Reference: Zarrar et al., [*TinyLidarNet: 2D LiDAR-based End-to-End Deep Learning Model for F1TENTH Autonomous Racing*](https://arxiv.org/abs/2410.07447) (arXiv:2410.07447, 2024).

The simulation environment uses [robosim2d](../robosim2d/),
and you can experience the full flow of keyboard-driven data collection → CNN training → autonomous driving.

## Structure

```
tiny-lidar-net-example/
├── main.py                  # CLI entry point
├── commands/
│   ├── collect.py           # Collect training data via manual driving
│   ├── train.py             # Train the CNN model
│   ├── autodrive.py         # Run autonomous driving with a trained model
│   └── evaluate.py          # Headless runs across multiple worlds × start positions, aggregating driving metrics
├── tiny_lidar_net/          # TinyLidarNet package
│   ├── control.py           # Control (NamedTuple)
│   ├── model.py             # TinyLidarNet (1D CNN)
│   ├── dataset.py           # LidarDataset (NPZ loader)
│   └── trainer.py           # Training loop
├── worlds/
│   ├── circuit/             # 100×60 circuit course
│   │   ├── robot.yaml       # Robot configuration
│   │   ├── world.yaml       # World configuration
│   │   └── eval.yaml        # Start positions for evaluate (optional, both CW/CCW)
│   └── simple/              # 50×50 obstacle world
│       ├── robot.yaml
│       └── world.yaml
├── outputs/                 # Output directory (models, training data, plots)
└── pyproject.toml
```

### Role of each component

| Component | Description |
|---|---|
| **robosim2d** | 2D simulation environment. Handles vehicle physics (Ackermann/DiffDrive models), LiDAR sensor (1081 rays), collision detection, and rendering |
| **commands/** | CLI subcommands. Create and operate the robosim2d environment |
| **tiny_lidar_net/** | PyTorch-based CNN (TinyLidarNet) and training pipeline. Independent of the simulator |
| **worlds/** | World definition directory (each world stores robot.yaml + world.yaml) |

## Setup

Requires Python 3.11 or later.

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

All dependencies (numpy / robosim2d / matplotlib / torch) are installed in one step as main dependencies.

## Usage

Build a self-driving model in three steps, and use the evaluate command for quantitative comparison when needed.

### 1. Collect training data (`collect`)

Drive the vehicle manually with the keyboard, recording pairs of LiDAR scans and control inputs.

```bash
python main.py collect -w worlds/circuit -o outputs/training_data.npz
```

A simulator window opens, showing the vehicle and LiDAR.

**Controls:**

| Key | Action |
|---|---|
| `W` / `↑` | Accelerate |
| `S` / `↓` | Decelerate |
| `A` / `←` | Steer left |
| `D` / `→` | Steer right |
| `Space` | Brake (speed 0) |
| `Q` / `Esc` | Quit and save |

On exit, an `.npz` file is saved (`lidar: (N, 1081)`, `control: (N, 2)`).
Frames where `speed=0` and `steering=0` are not recorded (to prevent label distribution bias).

### 2. Train the model (`train`)

Train TinyLidarNet on the collected data.

```bash
python main.py train -d outputs/training_data.npz -o outputs/tiny_lidar_net.pth -e 100
```

Multiple data files can be combined for training:

```bash
python main.py train -d outputs/data1.npz outputs/data2.npz -o outputs/tiny_lidar_net.pth
```

**Main options:**

| Option | Default | Description |
|---|---|---|
| `-d` | (required) | Training data files (multiple allowed) |
| `-o` | `outputs/tiny_lidar_net.pth` | Model output path |
| `-e` | 100 | Number of epochs |
| `-b` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |

After training, the model (`.pth`) and loss curve (`.png`) are written out.

### 3. Autonomous driving (`autodrive`)

Use the trained model to control the vehicle automatically.

```bash
python main.py autodrive -w worlds/circuit -m outputs/tiny_lidar_net.pth
```

The LiDAR scan is fed to the model, and the vehicle is driven with the predicted steering and speed.
The run ends on collision or when the window is closed.

### 4. Evaluation (`evaluate`)

Run a trained model headlessly across multiple worlds under `worlds/` × multiple start positions, and aggregate driving metrics.
No rendering and no `plt.pause` are inserted, so the run proceeds at CPU-bound speed.

```bash
python main.py evaluate -m outputs/tiny_lidar_net.pth
```

Each world is evaluated using the start positions listed in its `eval.yaml` (two clockwise (CW) and two counter-clockwise (CCW), four trials in total).
For worlds without `eval.yaml`, `robot.yaml`'s `state` is used as a single start.

**Main options:**

| Option | Default | Description |
|---|---|---|
| `-m` | `outputs/tiny_lidar_net.pth` | Model file to evaluate |
| `-w` | All worlds | World directories to evaluate (multiple allowed, e.g. `-w worlds/circuit worlds/maze`) |
| `--max-steps` | 10000 | Step cap per trial (with DT=0.1s, equivalent to 1000s of sim time) |
| `-v`, `--visualize` | off | Render each trial with matplotlib. For debugging; slower due to rendering, runs near real-time |

**Metrics collected (per trial):**

| Column | Description |
|---|---|
| `Result` | `collision` / `stuck` (stalled, not progressing) / `survived` (reached max-steps with sufficient travel) |
| `Steps` | Number of steps executed |
| `Dist[m]` | Distance traveled until collision or termination |
| `AvgV` | Average driving speed [m/s] (computed from position differences) |
| `MaxV` | Maximum driving speed [m/s] |
| `StRMS` | RMS of steering commands [rad] (`sqrt(mean(steering^2))`. A single metric capturing both oscillation magnitude and mean offset from 0) |

**Stuck detection**: an additional classification to avoid counting degenerate "no collision but no real driving" behavior as success. Stuck is flagged if any of the following holds:

- During the run: the straight-line displacement over the last 100 steps (10 seconds) drops below 1 m (early termination)
- At the end of the run: the average speed is below 0.5 m/s

The per-world aggregation reports the number of trials, `Success%` (survived ratio), `Stuck%`, `Coll%`, mean distance, and mean speed. Stuck is treated as a failure alongside collision (not counted in Success%).

**eval.yaml format:**

```yaml
# Each entry is [x, y, yaw_rad, steering_initial]
# yaw in radians (0=east, π/2=north, π=west, -π/2=south)
starts:
  - [10.0, 50.0, 0.0, 0.0]      # West of the top corridor, facing east → CW
  - [10.0, 50.0, -1.5708, 0.0]  # West of the top corridor, facing south → CCW
  - [90.0, 10.0, 3.14159, 0.0]  # East of the bottom corridor, facing west → CW
  - [90.0, 10.0, 1.5708, 0.0]   # East of the bottom corridor, facing north → CCW
```

### Other

```bash
# List available worlds
python main.py list
```

## World definitions

Worlds are defined as subdirectories under the `worlds/` directory.
Each subdirectory contains `robot.yaml` (robot configuration) and `world.yaml` (world configuration).

**robot.yaml:**
```yaml
kinematics: {name: 'acker'}                                      # Ackermann (bicycle) model
shape: {name: 'rectangle', length: 4.0, width: 1.8, wheelbase: 2.5}
state: [5.0, 5.0, 0.0, 0.0]                                     # [x, y, yaw, steering]
vel_max: [10.0, 0.7854]                                          # [max speed, max steering angle]
sensors:
  - type: 'lidar2d'
    number: 1081           # Number of rays
    range_max: 30.0        # Max detection range [m]
    angle_range: 6.28318   # 360 degrees
```

**world.yaml:**
```yaml
world:
  height: 50.0
  width: 50.0

obstacle:
  - shape: {name: 'rectangle', length: 4.0, width: 4.0}
    state: [15.0, 15.0, 0]    # [x, y, angle]
```

## Data flow

```
collect (collect training data via manual driving)
  Keyboard → Control(steering, speed) → sim.step() → record LiDAR scan → .npz

train (train the model)
  .npz → LidarDataset → TinyLidarNet (1D CNN) → .pth

autodrive (autonomous driving)
  LiDAR scan → TinyLidarNet.predict() → Control → sim.step()

evaluate (quantitative evaluation)
  Each world × each start in eval.yaml → headless loop → aggregate distance / speed / StRMS table to stdout
```

### Control type and ordering convention

`tiny_lidar_net.Control` is a `NamedTuple` holding the steering angle and speed:

| Use | Order |
|---|---|
| Training data (`.npz`) / model output tensor | `[steering, speed]` |
| robosim2d `sim.step()` action array | `[speed, steering]` |

To avoid ordering confusion, always use `Control.to_array()` / `Control.to_action()` for conversion.

## TinyLidarNet architecture

A 1D CNN that outputs steering angle and speed from a LiDAR scan (1081 dimensions).

```
Input: (batch, 1, 1081)
  ↓ Conv1d(1→24,  k=10, s=4)   → 268
  ↓ Conv1d(24→36, k=8,  s=4)   → 66
  ↓ Conv1d(36→48, k=4,  s=2)   → 32
  ↓ Conv1d(48→64, k=3,  s=1)   → 30
  ↓ Conv1d(64→64, k=3,  s=1)   → 28
  ↓ Flatten                     → 1792
  ↓ FC(1792→100) + ReLU + Dropout
  ↓ FC(100→50)   + ReLU + Dropout
  ↓ FC(50→10)    + ReLU
  ↓ FC(10→2)
Output: (batch, 2) = [steering_angle, speed]
```


# Notes

Official implementation

https://github.com/CSL-KU/TinyLidarNet/blob/main/train.py

Activation in hidden layers: ReLU
Activation in final output layer: tanh
Dropout: none
Optimizer: Adam(5e-5)
Loss function: huber
batch_size: 64, epoch: 20

### Training data
curved_circuit
outputs/training_data_curved_circuit.npz
