"""Autodrive command: control the vehicle automatically with a trained TinyLiDARNet."""

from pathlib import Path

import matplotlib.pyplot as plt
import robosim2d
from robosim2d.viz import RealtimeVisualizer

from tiny_lidar_net import DT, TinyLiDARNet

MAX_STEPS = 10000


def run_autodrive(world_dir: str, model_file: str) -> None:
    """Run autonomous driving with a trained TinyLiDARNet."""
    print("=" * 60)
    print("Autodrive Mode - Automatic control via TinyLiDARNet")
    print("=" * 60)

    if not Path(model_file).exists():
        print(f"Error: model file not found: {model_file}")
        return

    model = TinyLiDARNet.load(model_file)

    world_dir = Path(world_dir)
    sim = robosim2d.make(
        robot_file=world_dir / "robot.yaml",
        world_file=world_dir / "world.yaml",
        dt=DT,
        collision_mode="stop",
    )
    viz = RealtimeVisualizer(sim)

    print(f"\nWorld: {world_dir}")
    print(f"Model: {model_file}")
    print("\nStarting autodrive")
    print("  Close the window or press Ctrl+C to exit")
    print("-" * 40)

    sim.reset()
    viz.setup()
    viz.render()

    step = 0
    try:
        for step in range(MAX_STEPS):
            scan = sim.get_lidar_scan()
            control = model.predict(scan)

            _, collision, _ = sim.step(control.to_action())
            viz.render(
                title=f"Simulation Time: {sim.time:.2f}s  |  "
                f"Speed: {control.speed:.2f}  Steering: {control.steering:.2f}"
            )
            plt.pause(0.01)

            # Check if the window was closed
            if not plt.fignum_exists(viz.fig.number):
                break

            if collision:
                print(f"Collision! Step: {step}")
                print("Close the window to exit.")
                viz.render(
                    title=f"COLLISION at Step {step}  |  "
                    f"Time: {sim.time:.2f}s  |  Close window to exit"
                )
                while plt.fignum_exists(viz.fig.number):
                    plt.pause(0.1)
                break

    except KeyboardInterrupt:
        print("\nAutodrive interrupted.")

    print(f"Finished: {step} steps executed")
    viz.close()
