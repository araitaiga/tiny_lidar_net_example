"""手動操作コマンド"""

from pathlib import Path

import numpy as np


def run_manual(world_dir: str, output_file: str) -> None:
    """手動操作モードを実行"""
    import matplotlib.pyplot as plt
    import robosim2d
    from robosim2d.viz import RealtimeVisualizer

    print("=" * 60)
    print("手動操作モード - 学習データ収集")
    print("=" * 60)

    world_dir = Path(world_dir)
    print(f"\nワールド: {world_dir}")
    print(f"出力ファイル: {output_file}")

    sim = robosim2d.make(
        robot_file=world_dir / "robot.yaml",
        world_file=world_dir / "world.yaml",
        dt=0.1,
        collision_mode="stop",
    )
    viz = RealtimeVisualizer(sim)

    speed = 0.0
    steering = 0.0
    is_running = True
    lidar_data: list[np.ndarray] = []
    control_data: list[list[float]] = []

    max_speed = 5.0
    max_steering = 0.5
    speed_step = 0.5
    steering_step = 0.05

    def on_key(event):
        nonlocal speed, steering, is_running
        key = event.key
        if key in ("w", "up"):
            speed = min(speed + speed_step, max_speed)
        elif key in ("s", "down"):
            speed = max(speed - speed_step, -max_speed)
        elif key in ("a", "left"):
            steering = min(steering + steering_step, max_steering)
        elif key in ("d", "right"):
            steering = max(steering - steering_step, -max_steering)
        elif key == " ":
            speed = 0.0
        elif key in ("q", "escape"):
            is_running = False

    # 初期描画
    sim.reset()
    viz.setup()
    viz.fig.canvas.mpl_connect("key_press_event", on_key)
    viz.render()

    print("\n操作方法:")
    print("  W/↑: 加速  |  S/↓: 減速")
    print("  A/←: 左    |  D/→: 右")
    print("  Space: ブレーキ  |  Q/Esc: 終了")
    print("-" * 40)

    step = 0
    try:
        while is_running and step < 10000:
            scan = sim.get_lidar_scan()
            lidar_data.append(scan)
            control_data.append([steering, speed])

            action = np.array([speed, steering])
            sim.step(action)
            viz.render(
                title=f"Simulation Time: {sim.time:.2f}s  |  "
                f"Speed: {speed:.2f}  Steering: {steering:.2f}"
            )
            plt.pause(0.01)
            step += 1

            if not plt.fignum_exists(viz.fig.number):
                break

    except KeyboardInterrupt:
        print("\n操作を中断しました。")

    viz.close()

    if lidar_data:
        X = np.array(lidar_data)
        y = np.array(control_data)
        np.savez(output_file, lidar=X, control=y)
        print(f"\n学習データを保存しました: {output_file}")
        print(f"  サンプル数: {len(X)}")
        print(f"  入力形状: {X.shape}")
        print(f"  出力形状: {y.shape}")
    else:
        print("記録データがありません。")
