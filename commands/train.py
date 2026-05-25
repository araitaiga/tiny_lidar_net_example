"""Model training command."""

from pathlib import Path

from tiny_lidar_net import train_from_file


def run_train(
    data_files: list[str],
    model_output: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Load data files and train TinyLidarNet."""
    missing_files = [f for f in data_files if not Path(f).exists()]
    if missing_files:
        print("Error: the following files were not found:")
        for f in missing_files:
            print(f"  {f}")
        return

    train_from_file(
        data_files=data_files,
        model_output=model_output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        plot_loss=True,
    )
