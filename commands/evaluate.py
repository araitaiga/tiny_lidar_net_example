"""Evaluation command: run autodrive across multiple worlds × multiple start positions and aggregate driving metrics.

By default it runs without rendering at CPU-bound speed. Passing ``visualize=True`` renders
each run with matplotlib (same as autodrive), which runs near real-time.

Each trial result is classified into one of three values: collision / stuck / survived.
``stuck`` is a label that captures degenerate behavior where the vehicle does not collide
but also does not drive meaningfully. It is decided by either of the following:
  - During the run: displacement over the last STUCK_WINDOW steps falls below STUCK_DISPLACEMENT → early termination
  - At the end of the run: the mean speed is below STUCK_AVG_SPEED
"""

from collections import deque
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import robosim2d
import yaml
from robosim2d.viz import RealtimeVisualizer

from tiny_lidar_net import DT, TinyLiDARNet

MAX_STEPS_DEFAULT = 10000

# Stuck-detection parameters
STUCK_WINDOW = 100          # Number of steps in the moving window (DT=0.1s → 10s)
STUCK_DISPLACEMENT = 1.0    # If straight-line displacement within the window is below this [m], judged as stuck
STUCK_AVG_SPEED = 0.5       # If mean speed at end of trial is below this [m/s], judged as stuck


def _load_starts(world_dir: Path) -> list[list[float]]:
    """Return ``starts`` from eval.yaml. If absent, return the ``state`` from robot.yaml as a single entry."""
    eval_file = world_dir / "eval.yaml"
    if eval_file.exists():
        data = yaml.safe_load(eval_file.read_text())
        return [list(s) for s in data["starts"]]
    robot_data = yaml.safe_load((world_dir / "robot.yaml").read_text())
    return [list(robot_data["state"])]


def _run_one(
    sim,
    model: TinyLiDARNet,
    start_state: list[float],
    max_steps: int,
    viz: RealtimeVisualizer | None = None,
    label: str = "",
) -> dict | None:
    """Run a single trial and return a metrics dict, or ``None`` if the visualization
    window was closed mid-run (the trial should be discarded by the caller).

    Returned keys: result ("collision" | "stuck" | "survived"), steps,
                   distance, avg_speed, max_speed, steering_rms

    When viz is provided, the trial is re-rendered at each step (runs near real-time).
    """
    sim.reset(np.asarray(start_state, dtype=float))
    if viz is not None:
        viz.render(title=f"{label}  start  Speed: 0.00  Steering: 0.00")

    prev_x, prev_y = float(start_state[0]), float(start_state[1])
    distance = 0.0
    speeds: list[float] = []
    steerings: list[float] = []
    # Keep the most recent STUCK_WINDOW (x, y) and detect stalling by straight-line displacement
    pos_window: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW)
    pos_window.append((prev_x, prev_y))

    collided = False
    stuck = False
    aborted = False
    final_step = 0

    for step in range(max_steps):
        scan = sim.get_lidar_scan()
        control = model.predict(scan)
        _, collision, info = sim.step(control.to_action())

        dx = info["x"] - prev_x
        dy = info["y"] - prev_y
        seg = (dx * dx + dy * dy) ** 0.5
        distance += seg
        speeds.append(seg / DT)
        steerings.append(float(control.steering))

        prev_x, prev_y = info["x"], info["y"]
        pos_window.append((prev_x, prev_y))
        final_step = step + 1

        if viz is not None:
            viz.render(
                title=f"{label}  t={sim.time:.1f}s  "
                f"Speed: {control.speed:.2f}  Steering: {control.steering:.2f}"
            )
            plt.pause(0.01)
            if not plt.fignum_exists(viz.fig.number):
                # Window closed mid-trial: signal abort so caller discards this incomplete run
                aborted = True
                break

        if collision:
            collided = True
            break

        # Once the window fills up, judge stalling by straight-line displacement
        if len(pos_window) == STUCK_WINDOW:
            x0, y0 = pos_window[0]
            x1, y1 = pos_window[-1]
            disp = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if disp < STUCK_DISPLACEMENT:
                stuck = True
                break

    if aborted:
        return None

    avg_speed = mean(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0
    if steerings:
        steering_rms = (sum(s * s for s in steerings) / len(steerings)) ** 0.5
    else:
        steering_rms = 0.0

    if collided:
        result = "collision"
    elif stuck or avg_speed < STUCK_AVG_SPEED:
        result = "stuck"
    else:
        result = "survived"

    return {
        "result": result,
        "steps": final_step,
        "distance": distance,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "steering_rms": steering_rms,
    }


def run_evaluate(
    world_dirs: list[str],
    model_file: str,
    max_steps: int = MAX_STEPS_DEFAULT,
    visualize: bool = False,
) -> None:
    """Run the evaluation across multiple worlds × multiple start positions and print a table to stdout.

    When visualize=True, each trial is rendered with matplotlib (runs near real-time, not CPU-bound).
    """
    print("=" * 60)
    print("Evaluate Mode" + ("  (visualize)" if visualize else ""))
    print("=" * 60)

    if not Path(model_file).exists():
        print(f"Error: model file not found: {model_file}")
        return

    model = TinyLiDARNet.load(model_file)

    print(f"Model: {model_file}")
    print(f"Max steps per run: {max_steps}  (sim time = {max_steps * DT:.0f}s)")
    print()

    rows: list[dict] = []
    aborted = False
    for world_dir_str in world_dirs:
        if aborted:
            break
        world_dir = Path(world_dir_str)
        sim = robosim2d.make(
            robot_file=world_dir / "robot.yaml",
            world_file=world_dir / "world.yaml",
            dt=DT,
            collision_mode="stop",
        )
        viz: RealtimeVisualizer | None = None
        if visualize:
            viz = RealtimeVisualizer(sim)
            viz.setup()

        starts = _load_starts(world_dir)
        for i, start in enumerate(starts):
            label = f"{world_dir.name} start={i}"
            metrics = _run_one(sim, model, start, max_steps, viz=viz, label=label)
            if metrics is None:
                # Visualization window was closed mid-trial: discard this incomplete
                # result and abort the remaining trials instead of polluting the summary.
                aborted = True
                break
            rows.append({"world": world_dir.name, "start": i, **metrics})
            print(
                f"  {world_dir.name} start={i}: {metrics['result']} "
                f"step={metrics['steps']} dist={metrics['distance']:.1f}m"
            )

        if viz is not None:
            viz.close()

    print()
    print("=" * 92)
    print(
        f"{'World':<26} {'Start':>5} {'Result':<10} {'Steps':>6} "
        f"{'Dist[m]':>9} {'AvgV':>6} {'MaxV':>6} {'StRMS':>7}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r['world']:<26} {r['start']:>5} {r['result']:<10} {r['steps']:>6} "
            f"{r['distance']:>9.1f} {r['avg_speed']:>6.2f} {r['max_speed']:>6.2f} "
            f"{r['steering_rms']:>7.4f}"
        )
    print("=" * 92)

    print()
    print(
        f"{'World':<26} {'Trials':>6} {'Success%':>9} {'Stuck%':>7} "
        f"{'Coll%':>6} {'MeanDist':>9} {'MeanAvgV':>9}"
    )
    print("-" * 80)
    by_world: dict[str, list[dict]] = {}
    for r in rows:
        by_world.setdefault(r["world"], []).append(r)
    for world, rs in by_world.items():
        n = len(rs)
        success_pct = 100.0 * sum(1 for r in rs if r["result"] == "survived") / n
        stuck_pct = 100.0 * sum(1 for r in rs if r["result"] == "stuck") / n
        coll_pct = 100.0 * sum(1 for r in rs if r["result"] == "collision") / n
        mean_dist = mean(r["distance"] for r in rs)
        mean_avg_v = mean(r["avg_speed"] for r in rs)
        print(
            f"{world:<26} {n:>6} {success_pct:>8.0f}% {stuck_pct:>6.0f}% "
            f"{coll_pct:>5.0f}% {mean_dist:>9.1f} {mean_avg_v:>9.2f}"
        )
    print("=" * 80)
