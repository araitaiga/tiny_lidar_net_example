"""自動制御コントローラー"""

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


class AutoController:
    """NNモデルによる自動制御クラス

    学習済みモデルを使用して車両を自動制御します。
    """

    def __init__(
        self,
        sim: "Simulator",
        model_path: str,
        max_speed: float = 5.0,
        max_steering: float = 0.5,
        **visualizer_kwargs,
    ):
        try:
            from simulator.ml import DrivingNet
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for auto control. "
                "Install with: pip install torch"
            ) from e

        self.sim = sim
        self.model = DrivingNet.load(model_path)
        self.max_speed = max_speed
        self.max_steering = max_steering

        self.viz = RealtimeVisualizer(sim, **visualizer_kwargs)
        self._is_running = False

    def _get_control(self, lidar_scan: np.ndarray):
        """モデルから制御入力を取得"""
        from simulator.core import VehicleControl

        steering, speed = self.model.predict(lidar_scan)

        steering = np.clip(steering, -self.max_steering, self.max_steering)
        speed = np.clip(speed, -self.max_speed, self.max_speed)

        return VehicleControl(steering_angle=steering, speed=speed)

    def run(self, max_steps: int = 10000, stop_on_collision: bool = True) -> None:
        """自動制御シミュレーションを実行"""
        self.viz.setup()
        self._is_running = True
        step = 0

        print("\n自動制御開始")
        print("  ウィンドウを閉じるか Ctrl+C で終了")
        print("-" * 40)

        try:
            while self._is_running and step < max_steps:
                lidar_scan = self.sim.get_lidar_scan()
                control = self._get_control(lidar_scan)

                _, collision, _ = self.sim.step(control)
                step += 1

                title = (
                    f"[AUTO] Step: {step} | "
                    f"Speed: {control.speed:.1f}m/s | "
                    f"Steering: {np.degrees(control.steering_angle):.1f}deg"
                )
                if collision:
                    title += " | COLLISION!"

                self.viz.render(title=title)
                plt.pause(self.viz.update_interval)

                if collision and stop_on_collision:
                    print(f"衝突! Step: {step}")
                    break

                if not plt.fignum_exists(self.viz.fig.number):
                    break

        except KeyboardInterrupt:
            print("\n自動制御を中断しました。")

        self._is_running = False
        print(f"終了: {step} ステップ実行")
        self.viz.close()

    def close(self) -> None:
        """リソースを解放"""
        self.viz.close()
