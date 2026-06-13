"""TinyLidarNet training loop."""

from pathlib import Path

from tiny_lidar_net.dataset import LidarDataset
from tiny_lidar_net.model import TinyLidarNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class Trainer:
    """A simple Trainer that trains TinyLidarNet with the Huber loss.

    Hyperparameters following the official implementation:
        Optimizer: Adam(lr=5e-5),  Loss: HuberLoss(δ=1.0),
        Batch: 64,  Epoch: 20
    """

    def __init__(
        self,
        model: TinyLidarNet,
        learning_rate: float = 5e-5,
        device: str | None = None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(
        self,
        dataset: LidarDataset,
        epochs: int = 20,
        batch_size: int = 64,
        val_split: float = 0.2,
    ) -> dict:
        if len(dataset) < 2:
            raise ValueError(
                f"Dataset must contain at least 2 samples for train/val split, got {len(dataset)}"
            )
        val_size = max(int(len(dataset) * val_split), 1)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Training started: {self.device}")
        print(f"  Train data: {train_size}, Val data: {val_size}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")
        print("-" * 50)

        best_val_loss = float("inf")
        best_state: dict | None = None

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(train_loader, train_size)
            val_loss = self._validate(val_loader, val_size)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Clone the current weights so later epochs do not overwrite this snapshot.
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:4d}/{epochs}: "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print("-" * 50)
        print(f"Training complete: Best Val Loss = {best_val_loss:.6f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }

    def _train_one_epoch(self, loader: DataLoader, n_samples: int) -> float:
        # Set model to training mode (enables dropout, etc.)
        self.model.train()
        total_loss = 0.0
        for lidar, control in loader:
            # Move data to the same device as the model
            lidar = lidar.to(self.device)
            control = control.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(lidar)
            loss = self.criterion(output, control)
            loss.backward()
            self.optimizer.step()

            # Accumulate the batch loss sum (mean loss x batch size) so the final
            # average is unaffected by uneven batch sizes (e.g. a smaller last batch).
            total_loss += loss.item() * lidar.size(0)
        return total_loss / n_samples

    def _validate(self, loader: DataLoader, n_samples: int) -> float:
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for lidar, control in loader:
                lidar = lidar.to(self.device)
                control = control.to(self.device)
                output = self.model(lidar)
                loss = self.criterion(output, control)
                total_loss += loss.item() * lidar.size(0)
        return total_loss / n_samples

    def plot_loss(self, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Huber)")
        plt.title("Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            print(f"Training curve saved: {save_path}")
        else:
            plt.show()


def train_from_file(
    data_files: str | list[str],
    model_output: str = "outputs/tiny_lidar_net.pth",
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 5e-5,
    plot_loss: bool = True,
) -> TinyLidarNet:
    """Load NPZ files and train TinyLidarNet."""
    print("=" * 60)
    print("TinyLidarNet Training")
    print("=" * 60)

    if isinstance(data_files, str):
        data_files = [data_files]

    print(f"\nLoading data: {len(data_files)} file(s)")
    dataset = LidarDataset.from_files(data_files)

    print(f"  Input length (LiDAR rays): {dataset.lidar.shape[-1]}")

    model = TinyLidarNet()  # input length is fixed at 1081
    print("\nModel architecture:")
    print(model)

    print()
    trainer = Trainer(model, learning_rate=learning_rate)
    trainer.train(dataset, epochs=epochs, batch_size=batch_size)

    model.save(model_output)

    if plot_loss:
        loss_path = Path(model_output).with_suffix(".png")
        trainer.plot_loss(str(loss_path))

    return model
