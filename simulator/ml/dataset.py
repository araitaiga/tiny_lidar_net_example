"""学習データセット"""

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # ダミー


def _check_torch():
    """PyTorchが利用可能か確認"""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for training. "
            "Install with: pip install torch"
        )


class LidarDataset(Dataset):
    """LiDARデータセット

    学習用のLiDARスキャンデータと制御入力のペアを管理します。
    """

    def __init__(self, lidar_data: np.ndarray, control_data: np.ndarray):
        """
        Args:
            lidar_data: LiDARスキャンデータ (N, 1081)
            control_data: 制御データ (N, 2) [steering, speed]
        """
        _check_torch()
        self.lidar = torch.FloatTensor(lidar_data)
        self.control = torch.FloatTensor(control_data)

    def __len__(self):
        return len(self.lidar)

    def __getitem__(self, idx):
        lidar = self.lidar[idx].unsqueeze(0)  # (1, 1081)
        control = self.control[idx]  # (2,)
        return lidar, control

    @classmethod
    def from_file(cls, filepath: str) -> "LidarDataset":
        """ファイルからデータセットを作成"""
        data = np.load(filepath)
        return cls(data["lidar"], data["control"])

    @classmethod
    def from_files(cls, filepaths: list[str]) -> "LidarDataset":
        """複数ファイルからデータセットを作成

        Args:
            filepaths: 学習データファイルのリスト（.npz形式）

        Returns:
            結合されたデータセット
        """
        if not filepaths:
            raise ValueError("ファイルリストが空です")

        lidar_list = []
        control_list = []

        for filepath in filepaths:
            data = np.load(filepath)
            lidar_list.append(data["lidar"])
            control_list.append(data["control"])
            print(f"  ロード: {filepath} ({len(data['lidar'])} サンプル)")

        lidar_data = np.concatenate(lidar_list, axis=0)
        control_data = np.concatenate(control_list, axis=0)

        print(f"  合計: {len(lidar_data)} サンプル")

        return cls(lidar_data, control_data)
