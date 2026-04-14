"""自動制御コマンド"""

from pathlib import Path

import irsim
import matplotlib.pyplot as plt
import numpy as np


def run_auto(world_file: str, model_file: str) -> None:
    """自動制御モードを実行"""
    print("=" * 60)
    print("自動制御モード - NNモデルによる自動運転")
    print("=" * 60)

    if not Path(model_file).exists():
        print(f"エラー: モデルファイルが見つかりません: {model_file}")
        return

    from simulator.ml import DrivingNet

    model = DrivingNet.load(model_file)

    env = irsim.make(world_file)
    robot = env.robot

    print(f"\nワールド: {world_file}")
    print(f"モデル: {model_file}")
    print("\n自動制御開始")
    print("  ウィンドウを閉じるか Ctrl+C で終了")
    print("-" * 40)

    # センサー初期化
    env.step(np.zeros((2, 1)))
    env.render(0.05)

    step = 0
    try:
        for step in range(10000):
            scan = np.array(robot.get_lidar_scan()["ranges"])
            steering, speed = model.predict(scan)
            steering = float(np.clip(steering, -0.5, 0.5))
            speed = float(np.clip(speed, -5.0, 5.0))

            action = np.array([[speed], [steering]])
            env.step(action)
            env.render(0.05)

            if robot.collision:
                print(f"衝突! Step: {step}")
                break

            if not plt.get_fignums():
                break

    except KeyboardInterrupt:
        print("\n自動制御を中断しました。")

    print(f"終了: {step} ステップ実行")
    env.end(ending_time=0)
