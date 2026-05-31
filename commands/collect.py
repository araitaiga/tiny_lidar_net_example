"""Data collection command: drive the vehicle manually with the keyboard and record (LiDAR scan, control) pairs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import robosim2d
from robosim2d.viz import RealtimeVisualizer

from tiny_lidar_net import DT, Control
from tiny_lidar_net.control import MAX_SPEED, MAX_STEERING

SPEED_STEP = 0.5
STEERING_STEP = 0.05
MAX_STEPS = 10000


def run_collect(world_dir: str, output_file: str) -> None:
    """Collect training data via manual driving."""
    print("=" * 60)
    print("Data Collection Mode - Record training data via manual driving")
    print("=" * 60)

    world_dir = Path(world_dir)
    print(f"\nWorld: {world_dir}")
    print(f"Output file: {output_file}")

    sim = robosim2d.make(
        robot_file=world_dir / "robot.yaml",
        world_file=world_dir / "world.yaml",
        dt=DT,
        collision_mode="stop",
    )
    viz = RealtimeVisualizer(sim)

    speed = 0.0
    steering = 0.0
    is_running = True

    def on_key(event):
        nonlocal speed, steering, is_running
        key = event.key
        if key in ("w", "up"):
            speed = min(speed + SPEED_STEP, MAX_SPEED)
        elif key in ("s", "down"):
            speed = max(speed - SPEED_STEP, -MAX_SPEED)
        elif key in ("a", "left"):
            steering = min(steering + STEERING_STEP, MAX_STEERING)
        elif key in ("d", "right"):
            steering = max(steering - STEERING_STEP, -MAX_STEERING)
        elif key == " ":
            speed = 0.0
        elif key in ("q", "escape"):
            is_running = False

    sim.reset()
    viz.setup()
    viz.fig.canvas.mpl_connect("key_press_event", on_key)
    viz.render()

    print("\nControls:")
    print("  W/↑: Accelerate  |  S/↓: Decelerate")
    print("  A/←: Left        |  D/→: Right")
    print("  Space: Brake     |  Q/Esc: Quit")
    print("-" * 40)

    lidar_data: list[np.ndarray] = []
    control_data: list[np.ndarray] = []
    collided = False

    try:
        for _ in range(MAX_STEPS):
            if not is_running or not plt.fignum_exists(viz.fig.number):
                break

            control = Control(steering=steering, speed=speed)

            # Skip recording frames where speed=0 and steering=0, since
            # such "stationary samples" would otherwise dominate and skew the label distribution.
            # Also skip while in a collision state: with collision_mode="stop" the vehicle is
            # halted but key presses still produce non-zero labels, which would pair the same
            # post-collision LiDAR scan with arbitrary controls and pollute the dataset.
            if not collided and (control.speed != 0.0 or control.steering != 0.0):
                lidar_data.append(np.array(sim.get_lidar_scan(), copy=True))
                control_data.append(control.to_training_label())

            _, collision, _ = sim.step(control.to_robot_action())
            if collision and not collided:
                collided = True
                print("\nCollision detected. Recording stopped. Press Q/Esc to finish.")

            title_suffix = "  [COLLIDED]" if collided else ""
            viz.render(
                title=f"Simulation Time: {sim.time:.2f}s  |  "
                f"Speed: {speed:.2f}  Steering: {steering:.2f}{title_suffix}"
            )
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    viz.close()

    if not lidar_data:
        print("No data recorded.")
        return

    X = np.array(lidar_data)
    y = np.array(control_data)
    np.savez(output_file, lidar=X, control=y)
    print(f"\nTraining data saved: {output_file}")
    print(f"  Samples: {len(X)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")
