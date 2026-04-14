"""自動運転シミュレーター CLI"""

import argparse
from pathlib import Path

# ワールドファイルディレクトリ
WORLDS_DIR = Path(__file__).parent / "worlds"


def ensure_output_dir(filepath: str) -> None:
    """出力ファイルの親ディレクトリを作成"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def create_parser() -> argparse.ArgumentParser:
    """引数パーサーを作成"""
    parser = argparse.ArgumentParser(description="自動運転シミュレーター")
    subparsers = parser.add_subparsers(dest="command", help="コマンド")

    # 手動操作コマンド
    manual_parser = subparsers.add_parser("manual", help="手動操作（学習データ収集）")
    manual_parser.add_argument(
        "--world",
        "-w",
        type=str,
        default=str(WORLDS_DIR / "simple.yaml"),
        help="ワールド設定ファイルのパス",
    )
    manual_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/training_data.npz",
        help="学習データ出力ファイル（.npz形式）",
    )

    # 学習コマンド
    train_parser = subparsers.add_parser("train", help="モデル学習")
    train_parser.add_argument(
        "--data",
        "-d",
        type=str,
        nargs="+",
        required=True,
        help="学習データファイル（複数指定可、.npz形式）",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/driving_model.pth",
        help="モデル出力ファイル（.pth形式）",
    )
    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="エポック数",
    )
    train_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="バッチサイズ",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="学習率",
    )
    train_parser.add_argument(
        "--save-best",
        type=str,
        default=None,
        help="ベストモデル保存先（指定時のみ有効）",
    )

    # 自動制御コマンド
    auto_parser = subparsers.add_parser("auto", help="自動制御（NNモデル使用）")
    auto_parser.add_argument(
        "--world",
        "-w",
        type=str,
        default=str(WORLDS_DIR / "simple.yaml"),
        help="ワールド設定ファイルのパス",
    )
    auto_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="outputs/driving_model.pth",
        help="学習済みモデルファイル（.pth形式）",
    )

    # ワールド一覧
    subparsers.add_parser("list", help="ワールドファイル一覧")

    return parser


def show_help():
    """ヘルプを表示"""
    print("利用可能なワールドファイル:")
    for f in sorted(WORLDS_DIR.glob("*.yaml")):
        print(f"  {f.name}")

    print("\n使用方法:")
    print("  python main.py manual -w worlds/circuit.yaml              # 手動操作（データ収集）")
    print("  python main.py train -d outputs/data1.npz outputs/data2.npz  # モデル学習（教師あり）")
    print("  python main.py auto -m outputs/driving_model.pth          # 自動制御（CNN）")


def main():
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "list" or args.command is None:
        show_help()
        return

    if args.command == "manual":
        ensure_output_dir(args.output)
        from commands.manual import run_manual

        run_manual(args.world, args.output)

    elif args.command == "train":
        ensure_output_dir(args.output)
        from commands.train import run_train

        run_train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.save_best)

    elif args.command == "auto":
        from commands.auto import run_auto

        run_auto(args.world, args.model)


if __name__ == "__main__":
    main()
