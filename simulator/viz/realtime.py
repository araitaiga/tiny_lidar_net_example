"""リアルタイム可視化"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.core import Simulator

try:
    from matplotlib import patches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """matplotlibが利用可能か確認"""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


class RealtimeVisualizer:
    """リアルタイム可視化クラス

    シミュレーションをリアルタイムで描画・更新します。
    """

    def __init__(
        self,
        sim: "Simulator",
        figsize: tuple[float, float] = (10, 10),
        show_lidar: bool = True,
        show_lidar_points: bool = True,
        lidar_alpha: float = 0.3,
        update_interval: float = 0.01,
    ):
        _check_matplotlib()

        self.sim = sim
        self.figsize = figsize
        self.show_lidar = show_lidar
        self.show_lidar_points = show_lidar_points
        self.lidar_alpha = lidar_alpha
        self.update_interval = update_interval

        self.fig = None
        self.ax = None
        self._artists = {}
        self._is_running = False

    def setup(self) -> tuple:
        """描画のセットアップ"""
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
        self.ax.set_xlim(0, self.sim.world.width)
        self.ax.set_ylim(0, self.sim.world.height)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("Car Simulator")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor("#f0f0f0")

        self._draw_obstacles()
        plt.show(block=False)
        return self.fig, self.ax

    def _draw_obstacles(self):
        """障害物を描画"""
        for obs in self.sim.world.obstacles:
            rect = patches.Rectangle(
                (obs.x - obs.width / 2, obs.y - obs.height / 2),
                obs.width,
                obs.height,
                linewidth=1,
                edgecolor="#333333",
                facecolor="#666666",
            )
            self.ax.add_patch(rect)

    def _draw_vehicle(self) -> patches.Polygon:
        """車両を描画"""
        corners = self.sim.vehicle.get_corners()
        return patches.Polygon(
            corners,
            closed=True,
            linewidth=2,
            edgecolor="#0066cc",
            facecolor="#3399ff",
            alpha=0.8,
        )

    def _draw_vehicle_direction(self):
        """車両の向きを示す矢印を描画"""
        x, y, yaw = self.sim.vehicle.x, self.sim.vehicle.y, self.sim.vehicle.yaw
        length = self.sim.vehicle.length * 0.6
        dx = length * np.cos(yaw)
        dy = length * np.sin(yaw)

        (line,) = self.ax.plot(
            [x, x + dx],
            [y, y + dy],
            color="#cc0000",
            linewidth=2,
            marker=">",
            markersize=8,
            markevery=[-1],
        )
        return line

    def _draw_lidar_rays(self, ranges: np.ndarray) -> list:
        """LiDARレイを描画"""
        lines = []
        x, y, yaw = self.sim.vehicle.x, self.sim.vehicle.y, self.sim.vehicle.yaw

        step = max(1, len(ranges) // 72)

        for i in range(0, len(ranges), step):
            angle = yaw + self.sim.lidar.angles[i]
            r = ranges[i]
            end_x = x + r * np.cos(angle)
            end_y = y + r * np.sin(angle)

            (line,) = self.ax.plot(
                [x, end_x],
                [y, end_y],
                color="#00aa00",
                linewidth=0.5,
                alpha=self.lidar_alpha,
            )
            lines.append(line)

        return lines

    def _draw_lidar_points(self, ranges: np.ndarray):
        """LiDAR検出点を描画"""
        points = self.sim.lidar.scan_to_points(
            self.sim.vehicle.x,
            self.sim.vehicle.y,
            self.sim.vehicle.yaw,
            ranges,
        )

        if len(points) > 0:
            return self.ax.scatter(
                points[:, 0],
                points[:, 1],
                c="#ff6600",
                s=3,
                alpha=0.7,
                zorder=10,
            )
        return None

    def render(self, title: str | None = None) -> None:
        """現在の状態を描画"""
        if self.fig is None:
            self.setup()

        # 前回の動的要素をクリア
        for key in list(self._artists.keys()):
            artist = self._artists.pop(key)
            if isinstance(artist, list):
                for a in artist:
                    a.remove()
            elif artist is not None:
                artist.remove()

        # 車両を描画
        vehicle_patch = self._draw_vehicle()
        self.ax.add_patch(vehicle_patch)
        self._artists["vehicle"] = vehicle_patch

        # 車両の向きを描画
        direction_line = self._draw_vehicle_direction()
        self._artists["direction"] = direction_line

        # LiDARスキャン
        ranges = self.sim.get_lidar_scan()

        if self.show_lidar:
            lidar_lines = self._draw_lidar_rays(ranges)
            self._artists["lidar_rays"] = lidar_lines

        if self.show_lidar_points:
            scatter = self._draw_lidar_points(ranges)
            self._artists["lidar_points"] = scatter

        # タイトル更新
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(
                f"Car Simulator | "
                f"Time: {self.sim.time:.2f}s | "
                f"Pos: ({self.sim.vehicle.x:.1f}, {self.sim.vehicle.y:.1f})"
            )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def step_and_render(self, control) -> tuple[np.ndarray, bool, dict]:
        """シミュレーションを1ステップ進めて描画を更新"""
        result = self.sim.step(control)
        self.render()
        plt.pause(self.update_interval)
        return result

    def run(
        self,
        control_fn,
        max_steps: int = 1000,
        stop_on_collision: bool = True,
    ) -> None:
        """リアルタイムシミュレーションを実行"""
        if self.fig is None:
            self.setup()

        self._is_running = True
        step = 0

        try:
            while self._is_running and step < max_steps:
                control = control_fn(self.sim, step)
                _, collision, info = self.sim.step(control)
                step += 1

                title = (
                    f"Step: {step} | "
                    f"Time: {self.sim.time:.2f}s | "
                    f"Speed: {control.speed:.1f}m/s | "
                    f"Steering: {np.degrees(control.steering_angle):.1f}deg"
                )
                if collision:
                    title += " | COLLISION!"

                self.render(title=title)
                plt.pause(self.update_interval)

                if collision and stop_on_collision:
                    print(
                        f"Collision at step {step}! "
                        f"Position: ({info['x']:.2f}, {info['y']:.2f})"
                    )
                    break

                if not plt.fignum_exists(self.fig.number):
                    break

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")

        self._is_running = False

    def stop(self) -> None:
        """シミュレーションを停止"""
        self._is_running = False

    def close(self) -> None:
        """描画をクローズ"""
        plt.ioff()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
