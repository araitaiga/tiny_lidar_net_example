"""自動制御コマンド"""

from pathlib import Path

from simulator import AutoController
from simulator import Simulator


def run_auto(world_file: str, model_file: str) -> None:
    """自動制御モードを実行"""
    print("=" * 60)
    print("自動制御モード - NNモデルによる自動運転")
    print("=" * 60)

    if not Path(model_file).exists():
        print(f"エラー: モデルファイルが見つかりません: {model_file}")
        return

    sim = Simulator.from_file(world_file)

    print(f"\nワールド: {world_file}")
    print(f"モデル: {model_file}")

    controller = AutoController(
        sim,
        model_path=model_file,
        update_interval=0.05,
        show_lidar=True,
        show_lidar_points=True,
    )
    controller.run()
