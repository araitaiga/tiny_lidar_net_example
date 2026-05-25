"""TinyLidarNet example CLI."""

import argparse
from pathlib import Path

# World files directory
WORLDS_DIR = Path(__file__).parent / "worlds"


def ensure_output_dir(filepath: str) -> None:
    """Create the parent directory of the output file."""
    # 再帰的に親ディレクトリを作成
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="TinyLidarNet example")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Training data collection command (record via manual driving)
    collect_parser = subparsers.add_parser(
        "collect", help="Collect training data via manual driving"
    )
    collect_parser.add_argument(
        "--world",
        "-w",
        type=str,
        required=True,
        help="Path to world config directory (containing robot.yaml + world.yaml)",
    )
    collect_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/training_data.npz",
        help="Training data output file (.npz format)",
    )

    # Training command
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

    # Autonomous driving command (control with a trained model)
    autodrive_parser = subparsers.add_parser(
        "autodrive", help="Run autodrive with a trained TinyLidarNet"
    )
    autodrive_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Trained model file (.pth format)",
    )
    autodrive_parser.add_argument(
        "--world",
        "-w",
        type=str,
        required=True,
        help="Path to world config directory (containing robot.yaml + world.yaml)",
    )

    # Evaluation command (headless autodrive across multiple worlds × start positions)
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a model across multiple worlds and start positions"
    )
    evaluate_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Trained model file (.pth format)",
    )
    evaluate_parser.add_argument(
        "--world",
        "-w",
        type=str,
        nargs="*",
        default=None,
        help="World directories to evaluate. Default: all worlds with world.yaml.",
    )
    evaluate_parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Max steps per run (default: 10000 = sim time 1000s)",
    )
    evaluate_parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Render each run with matplotlib (slower, near real-time)",
    )

    # List of worlds
    subparsers.add_parser("list", help="List available worlds")

    return parser


def show_help():
    """Show help."""
    print("Available worlds:")
    for d in sorted(WORLDS_DIR.iterdir()):
        if d.is_dir() and (d / "world.yaml").exists():
            print(f"  {d.name}")

    print("\nUsage:")
    print("  python main.py collect   -w worlds/circuit                                  # Collect training data via manual driving")
    print("  python main.py train     -d outputs/data1.npz outputs/data2.npz             # Train the model (supervised)")
    print("  python main.py autodrive -w worlds/circuit -m outputs/tiny_lidar_net.pth    # Autodrive (TinyLidarNet)")
    print("  python main.py evaluate  -m outputs/tiny_lidar_net.pth                      # Evaluate across worlds (CW+CCW, headless)")


def main():
    """Main entry."""
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

        run_train(args.data, args.output, args.epochs, args.batch_size, args.lr)

    elif args.command == "autodrive":
        from commands.autodrive import run_autodrive

        run_autodrive(args.world, args.model)

    elif args.command == "evaluate":
        from commands.evaluate import run_evaluate

        worlds = args.world
        if not worlds:
            worlds = sorted(
                str(d) for d in WORLDS_DIR.iterdir()
                if d.is_dir() and (d / "world.yaml").exists()
            )
        run_evaluate(worlds, args.model, args.max_steps, args.visualize)


if __name__ == "__main__":
    main()
