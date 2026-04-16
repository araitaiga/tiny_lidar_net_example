"""自動制御コマンド"""

from pathlib import Path

import numpy as np


def run_auto(world_dir: str, model_file: str) -> None:
    """自動制御モードを実行"""
    import matplotlib.pyplot as plt
    import robosim2d
    from robosim2d.viz import RealtimeVisualizer

    print("=" * 60)
    print("自動制御モード - NNモデルによる自動運転")
    print("=" * 60)

    if not Path(model_file).exists():
        print(f"エラー: モデルファイルが見つかりません: {model_file}")
        return

    from simulator.ml import DrivingNet

    model = DrivingNet.load(model_file)

    world_dir = Path(world_dir)
    sim = robosim2d.make(
        robot_file=world_dir / "robot.yaml",
        world_file=world_dir / "world.yaml",
        dt=0.1,
        collision_mode="stop",
    )
    viz = RealtimeVisualizer(sim)

    print(f"\nワールド: {world_dir}")
    print(f"モデル: {model_file}")
    print("\n自動制御開始")
    print("  ウィンドウを閉じるか Ctrl+C で終了")
    print("-" * 40)

    # 初期描画
    sim.reset()
    viz.setup()
    viz.render()

    step = 0
    try:
        for step in range(10000):
            scan = sim.get_lidar_scan()
            steering, speed = model.predict(scan)
            steering = float(np.clip(steering, -0.5, 0.5))
            speed = float(np.clip(speed, -5.0, 5.0))

            action = np.array([speed, steering])
            _, collision, _ = sim.step(action)
            viz.render(
                title=f"Simulation Time: {sim.time:.2f}s  |  "
                f"Speed: {speed:.2f}  Steering: {steering:.2f}"
            )
            plt.pause(0.01)

            if collision:
                print(f"衝突! Step: {step}")
                break

            if not plt.fignum_exists(viz.fig.number):
                break

    except KeyboardInterrupt:
        print("\n自動制御を中断しました。")

    print(f"終了: {step} ステップ実行")
    viz.close()
