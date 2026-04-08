"""手動操作コマンド"""

from simulator import ManualController
from simulator import Simulator


def run_manual(world_file: str, output_file: str) -> None:
    """手動操作モードを実行"""
    print("=" * 60)
    print("手動操作モード - 学習データ収集")
    print("=" * 60)

    sim = Simulator.from_file(world_file)

    print(f"\nワールド: {world_file}")
    print(f"出力ファイル: {output_file}")

    controller = ManualController(
        sim,
        output_file=output_file,
        update_interval=0.05,
        show_lidar=True,
        show_lidar_points=True,
    )
    controller.run()
