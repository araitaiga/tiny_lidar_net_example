"""自動運転ニューラルネットワークモデル"""

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None


def _check_torch():
    """PyTorchが利用可能か確認"""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for training. "
            "Install with: pip install torch"
        )


class DrivingNet(nn.Module if HAS_TORCH else object):
    """自動運転用CNNモデル

    Architecture:
        Input: 1081*1 (LiDAR scan)
        Conv1: 24 filters, kernel=10, stride=4 → 268
        Conv2: 36 filters, kernel=8, stride=4 → 66
        Conv3: 48 filters, kernel=4, stride=2 → 32
        Conv4: 64 filters, kernel=3, stride=1 → 30
        Conv5: 64 filters, kernel=3, stride=1 → 28
        FC1: 1792 → 100
        FC2: 100 → 50
        FC3: 50 → 10
        Output: 10 → 2 (steering, speed)
    """

    def __init__(self):
        _check_torch()
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 28, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

        # Activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x: LiDARスキャン (batch, 1, 1081)

        Returns:
            制御出力 (batch, 2) [steering, speed]
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def predict(self, lidar_scan: np.ndarray) -> tuple[float, float]:
        """LiDARスキャンから制御出力を予測

        Args:
            lidar_scan: LiDARスキャン (1081,) または (N, 1081)

        Returns:
            (steering_angle, speed) または [(steering, speed), ...]
        """
        self.eval()
        with torch.no_grad():
            if lidar_scan.ndim == 1:
                x = torch.FloatTensor(lidar_scan).unsqueeze(0).unsqueeze(0)
                output = self(x)
                return float(output[0, 0]), float(output[0, 1])
            else:
                x = torch.FloatTensor(lidar_scan).unsqueeze(1)
                output = self(x)
                return output.numpy()

    def save(self, filepath: str):
        """モデルを保存"""
        torch.save(self.state_dict(), filepath)
        print(f"モデルを保存しました: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DrivingNet":
        """モデルをロード"""
        _check_torch()
        model = cls()
        model.load_state_dict(torch.load(filepath, weights_only=True))
        model.eval()
        print(f"モデルをロードしました: {filepath}")
        return model
