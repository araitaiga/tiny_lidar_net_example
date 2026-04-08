"""モデル学習"""

from pathlib import Path

from simulator.ml.dataset import LidarDataset
from simulator.ml.model import DrivingNet

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    """PyTorchが利用可能か確認"""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for training. "
            "Install with: pip install torch"
        )


class Trainer:
    """モデル学習クラス"""

    def __init__(
        self,
        model: DrivingNet,
        learning_rate: float = 0.001,
        device: str | None = None,
    ):
        _check_torch()
        self.model = model
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def train(
        self,
        dataset: LidarDataset,
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: bool = True,
        save_best: str | None = None,
    ) -> dict:
        """モデルを学習"""
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        if verbose:
            print(f"学習開始: {self.device}")
            print(f"  学習データ: {train_size}, 検証データ: {val_size}")
            print(f"  エポック: {epochs}, バッチサイズ: {batch_size}")
            print("-" * 50)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for lidar, control in train_loader:
                lidar = lidar.to(self.device)
                control = control.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(lidar)
                loss = self.criterion(output, control)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * lidar.size(0)

            train_loss /= train_size
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for lidar, control in val_loader:
                    lidar = lidar.to(self.device)
                    control = control.to(self.device)
                    output = self.model(lidar)
                    loss = self.criterion(output, control)
                    val_loss += loss.item() * lidar.size(0)

            val_loss /= val_size
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                if save_best:
                    torch.save(best_state, save_best)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:4d}/{epochs}: "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose:
            print("-" * 50)
            print(f"学習完了: Best Val Loss = {best_val_loss:.6f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }

    def plot_loss(self, save_path: str | None = None):
        """学習曲線をプロット"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlibがインストールされていません。")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            print(f"学習曲線を保存しました: {save_path}")
        else:
            plt.show()


def train_from_file(
    data_files: str | list[str],
    model_output: str = "outputs/driving_model.pth",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    plot_loss: bool = True,
    save_best: str | None = None,
) -> DrivingNet:
    """ファイルからデータを読み込んでモデルを学習

    Args:
        data_files: 学習データファイル（単一または複数）
        model_output: モデル出力ファイル
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        plot_loss: 学習曲線をプロットするか

    Returns:
        学習済みモデル
    """
    _check_torch()

    print("=" * 60)
    print("自動運転モデル学習")
    print("=" * 60)

    # 単一ファイルの場合はリストに変換
    if isinstance(data_files, str):
        data_files = [data_files]

    print(f"\nデータ読み込み: {len(data_files)} ファイル")

    if len(data_files) == 1:
        dataset = LidarDataset.from_file(data_files[0])
        print(f"  サンプル数: {len(dataset)}")
    else:
        dataset = LidarDataset.from_files(data_files)

    model = DrivingNet()
    print("\nモデル構造:")
    print(model)

    print()
    trainer = Trainer(model, learning_rate=learning_rate)
    trainer.train(dataset, epochs=epochs, batch_size=batch_size, save_best=save_best)

    model.save(model_output)

    if plot_loss:
        loss_path = Path(model_output).with_suffix(".png")
        trainer.plot_loss(str(loss_path))

    return model
