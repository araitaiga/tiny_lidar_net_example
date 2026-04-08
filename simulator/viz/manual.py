"""手動操作コントローラー"""

from typing import TYPE_CHECKING

import numpy as np

from simulator.viz.realtime import RealtimeVisualizer

if TYPE_CHECKING:
    from simulator.core import Simulator

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ManualController:
    """キーボード操作による手動制御クラス

    キーボード入力で車両を操作し、学習データを記録します。

    操作方法:
        W/↑: 加速
        S/↓: 減速
        A/←: 左ステアリング
        D/→: 右ステアリング
        Space: ブレーキ
        Q/Escape: 終了
    """

    def __init__(
        self,
        sim: "Simulator",
        output_file: str = "training_data.npz",
        max_speed: float = 5.0,
        max_steering: float = 0.5,
        speed_step: float = 0.5,
        steering_step: float = 0.05,
        **visualizer_kwargs,
    ):
        self.sim = sim
        self.output_file = output_file
        self.max_speed = max_speed
        self.max_steering = max_steering
        self.speed_step = speed_step
        self.steering_step = steering_step

        self.viz = RealtimeVisualizer(sim, **visualizer_kwargs)

        self._speed = 0.0
        self._steering = 0.0
        self._is_running = False

        self._lidar_data = []
        self._control_data = []

    def _on_key_press(self, event):
        """キーボードイベントハンドラ"""
        key = event.key

        if key in ("w", "up"):
            self._speed = min(self._speed + self.speed_step, self.max_speed)
        elif key in ("s", "down"):
            self._speed = max(self._speed - self.speed_step, -self.max_speed)
        elif key in ("a", "left"):
            self._steering = min(
                self._steering + self.steering_step, self.max_steering)
        elif key in ("d", "right"):
            self._steering = max(
                self._steering - self.steering_step, -self.max_steering)
        elif key == " ":
            self._speed = 0.0
        elif key in ("q", "escape"):
            self._is_running = False

    def _record_step(self, lidar_scan: np.ndarray):
        """1ステップのデータを記録"""
        self._lidar_data.append(lidar_scan.copy())
        self._control_data.append([self._steering, self._speed])

    def _save_data(self):
        """学習データをファイルに保存"""
        if not self._lidar_data:
            print("記録データがありません。")
            return

        X = np.array(self._lidar_data)
        y = np.array(self._control_data)

        np.savez(self.output_file, lidar=X, control=y)
        print(f"\n学習データを保存しました: {self.output_file}")
        print(f"  サンプル数: {len(X)}")
        print(f"  入力形状: {X.shape}")
        print(f"  出力形状: {y.shape}")

    def run(self, max_steps: int = 10000, stop_on_collision: bool = False) -> None:
        """手動操作シミュレーションを実行"""
        from simulator.core import VehicleControl

        self.viz.setup()
        self.viz.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self._is_running = True
        self._speed = 0.0
        self._steering = 0.0
        step = 0

        print("\n操作方法:")
        print("  W/↑: 加速  |  S/↓: 減速")
        print("  A/←: 左    |  D/→: 右")
        print("  Space: ブレーキ  |  Q/Esc: 終了")
        print("-" * 40)

        try:
            while self._is_running and step < max_steps:
                control = VehicleControl(
                    steering_angle=self._steering,
                    speed=self._speed,
                )

                lidar_scan = self.sim.get_lidar_scan()
                self._record_step(lidar_scan)

                _, collision, _ = self.sim.step(control)
                step += 1

                title = (
                    f"[MANUAL] Step: {step} | "
                    f"Speed: {self._speed:.1f}m/s | "
                    f"Steering: {np.degrees(self._steering):.1f}deg | "
                    f"Samples: {len(self._lidar_data)}"
                )
                if collision:
                    title += " | COLLISION!"

                self.viz.render(title=title)
                plt.pause(self.viz.update_interval)

                if collision and stop_on_collision:
                    print(f"Collision at step {step}!")
                    break

                if not plt.fignum_exists(self.viz.fig.number):
                    break

        except KeyboardInterrupt:
            print("\n操作を中断しました。")

        self._is_running = False
        self._save_data()
        self.viz.close()

    def close(self) -> None:
        """リソースを解放"""
        self.viz.close()
