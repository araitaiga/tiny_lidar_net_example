"""Evaluation command: run autodrive across multiple worlds × multiple start positions
and aggregate driving metrics. Runs headless (no rendering) at CPU-bound speed; use the
autodrive command if you want to watch a model drive.

Each trial result is classified into one of three values: collision / stuck / survived.
``stuck`` is a label that captures degenerate behavior where the vehicle does not collide
but also does not drive meaningfully. It is decided by either of the following:
  - During the run: displacement over the last STUCK_WINDOW steps falls below STUCK_DISPLACEMENT → early termination
  - At the end of the run: the mean speed is below STUCK_AVG_SPEED
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import robosim2d
import yaml

from tiny_lidar_net import DT, TinyLidarNet

MAX_STEPS_DEFAULT = 10000

# Stuck-detection parameters
STUCK_WINDOW = 100          # Number of steps in the moving window (DT=0.1s → 10s)
STUCK_DISPLACEMENT = 1.0    # If straight-line displacement within the window is below this [m], judged as stuck
STUCK_AVG_SPEED = 0.5       # If mean speed at end of trial is below this [m/s], judged as stuck


@dataclass
class TrialResult:
    """Metrics for a single trial."""

    world: str
    start: int
    result: str  # "collision" | "stuck" | "survived"
    steps: int
    distance: float
    avg_speed: float
    max_speed: float
    steering_rms: float


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
    model: TinyLidarNet,
    world: str,
    start: int,
    start_state: list[float],
    max_steps: int,
) -> TrialResult:
    """Run a single trial and return its metrics."""
    sim.reset(np.asarray(start_state, dtype=float))

    prev_x, prev_y = float(start_state[0]), float(start_state[1])
    distance = 0.0
    speeds: list[float] = []
    steerings: list[float] = []
    # Keep the most recent STUCK_WINDOW (x, y) and detect stalling by straight-line displacement
    pos_window: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW)
    pos_window.append((prev_x, prev_y))

    collided = False
    stuck = False
    final_step = 0

    for step in range(max_steps):
        scan = sim.get_lidar_scan()
        control = model.predict(scan)
        _, collision, info = sim.step(control.to_robot_action())

        dx = info["x"] - prev_x
        dy = info["y"] - prev_y
        seg = (dx * dx + dy * dy) ** 0.5
        distance += seg
        speeds.append(seg / DT)
        steerings.append(float(control.steering))

        prev_x, prev_y = info["x"], info["y"]
        pos_window.append((prev_x, prev_y))
        final_step = step + 1

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

    return TrialResult(
        world=world,
        start=start,
        result=result,
        steps=final_step,
        distance=distance,
        avg_speed=avg_speed,
        max_speed=max_speed,
        steering_rms=steering_rms,
    )


def _pct(rows: list[TrialResult], result: str) -> float:
    """Percentage of rows whose result equals ``result``."""
    return 100.0 * sum(1 for r in rows if r.result == result) / len(rows)


def run_evaluate(
    world_dirs: list[str],
    model_file: str,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> None:
    """Run the evaluation across multiple worlds × multiple start positions and print a table to stdout."""
    print("=" * 60)
    print("Evaluate Mode")
    print("=" * 60)

    if not Path(model_file).exists():
        print(f"Error: model file not found: {model_file}")
        return

    model = TinyLidarNet.load(model_file)

    print(f"Model: {model_file}")
    print(f"Max steps per run: {max_steps}  (sim time = {max_steps * DT:.0f}s)")
    print()

    rows: list[TrialResult] = []
    for world_dir_str in world_dirs:
        world_dir = Path(world_dir_str)
        sim = robosim2d.make(
            robot_file=world_dir / "robot.yaml",
            world_file=world_dir / "world.yaml",
            dt=DT,
            collision_mode="stop",
        )

        for i, start in enumerate(_load_starts(world_dir)):
            r = _run_one(sim, model, world_dir.name, i, start, max_steps)
            rows.append(r)
            print(
                f"  {r.world} start={r.start}: {r.result} "
                f"step={r.steps} dist={r.distance:.1f}m"
            )

    print()
    print("=" * 92)
    print(
        f"{'World':<26} {'Start':>5} {'Result':<10} {'Steps':>6} "
        f"{'Dist[m]':>9} {'AvgV':>6} {'MaxV':>6} {'StRMS':>7}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r.world:<26} {r.start:>5} {r.result:<10} {r.steps:>6} "
            f"{r.distance:>9.1f} {r.avg_speed:>6.2f} {r.max_speed:>6.2f} "
            f"{r.steering_rms:>7.4f}"
        )
    print("=" * 92)

    print()
    print(
        f"{'World':<26} {'Trials':>6} {'Success%':>9} {'Stuck%':>7} "
        f"{'Coll%':>6} {'MeanDist':>9} {'MeanAvgV':>9}"
    )
    print("-" * 80)
    by_world: dict[str, list[TrialResult]] = {}
    for r in rows:
        by_world.setdefault(r.world, []).append(r)
    for world, rs in by_world.items():
        print(
            f"{world:<26} {len(rs):>6} {_pct(rs, 'survived'):>8.0f}% "
            f"{_pct(rs, 'stuck'):>6.0f}% {_pct(rs, 'collision'):>5.0f}% "
            f"{mean(r.distance for r in rs):>9.1f} "
            f"{mean(r.avg_speed for r in rs):>9.2f}"
        )
    print("=" * 80)
