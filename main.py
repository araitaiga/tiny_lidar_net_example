"""TinyLiDARNet サンプル CLI"""

import argparse
from pathlib import Path

# ワールドファイルディレクトリ
WORLDS_DIR = Path(__file__).parent / "worlds"


def ensure_output_dir(filepath: str) -> None:
    """出力ファイルの親ディレクトリを作成"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def create_parser() -> argparse.ArgumentParser:
    """引数パーサーを作成"""
    parser = argparse.ArgumentParser(description="TinyLiDARNet example")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # 学習データ収集コマンド（手動運転で記録）
    collect_parser = subparsers.add_parser(
        "collect", help="Collect training data via manual driving"
    )
    collect_parser.add_argument(
        "--world",
        "-w",
        type=str,
        default=str(WORLDS_DIR / "simple"),
        help="Path to world config directory (containing robot.yaml + world.yaml)",
    )
    collect_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/training_data.npz",
        help="Training data output file (.npz format)",
    )

    # 学習コマンド
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data",
        "-d",
        type=str,
        nargs="+",
        required=True,
        help="Training data files (one or more, .npz format)",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/tiny_lidar_net.pth",
        help="Model output file (.pth format)",
    )
    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=20,
        help="Number of epochs (official default: 20)",
    )
    train_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Batch size (official default: 64)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (official default: 5e-5)",
    )
    train_parser.add_argument(
        "--save-best",
        type=str,
        default=None,
        help="Best-model save path (only effective when specified)",
    )

    # 自動運転コマンド（学習済みモデルで自動制御）
    autodrive_parser = subparsers.add_parser(
        "autodrive", help="Run autodrive with a trained TinyLiDARNet"
    )
    autodrive_parser.add_argument(
        "--world",
        "-w",
        type=str,
        default=str(WORLDS_DIR / "simple"),
        help="Path to world config directory (containing robot.yaml + world.yaml)",
    )
    autodrive_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="outputs/tiny_lidar_net.pth",
        help="Trained model file (.pth format)",
    )

    # ワールド一覧
    subparsers.add_parser("list", help="List available worlds")

    return parser


def show_help():
    """ヘルプを表示"""
    print("Available worlds:")
    for d in sorted(WORLDS_DIR.iterdir()):
        if d.is_dir() and (d / "world.yaml").exists():
            print(f"  {d.name}")

    print("\nUsage:")
    print("  python main.py collect   -w worlds/circuit              # Collect training data via manual driving")
    print("  python main.py train     -d outputs/data1.npz outputs/data2.npz  # Train the model (supervised)")
    print("  python main.py autodrive -w worlds/circuit -m outputs/tiny_lidar_net.pth  # Autodrive (TinyLiDARNet)")


def main():
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "list" or args.command is None:
        show_help()
        return

    if args.command == "collect":
        ensure_output_dir(args.output)
        from commands.collect import run_collect

        run_collect(args.world, args.output)

    elif args.command == "train":
        ensure_output_dir(args.output)
        from commands.train import run_train

        run_train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.save_best)

    elif args.command == "autodrive":
        from commands.autodrive import run_autodrive

        run_autodrive(args.world, args.model)


if __name__ == "__main__":
    main()
