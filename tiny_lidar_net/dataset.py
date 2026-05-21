"""LiDAR学習データセット。"""

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_lidar_net.control import MAX_SPEED, MAX_STEERING


class LidarDataset(Dataset):
    """LiDARスキャンと制御値のペアを保持するデータセット。

    形状:
        lidar:   (N, 1081)
        control: (N, 2)  [steering, speed]   ※物理値で保存

    モデル最終層が tanh のため、制御ラベルは ``MAX_STEERING`` / ``MAX_SPEED``
    で割って ``[-1, 1]`` に正規化したテンソルを返す。
    """

    def __init__(self, lidar_data: np.ndarray, control_data: np.ndarray):
        self.lidar = torch.from_numpy(lidar_data).float()

        normalized = control_data.astype(np.float32).copy()
        normalized[:, 0] /= MAX_STEERING
        normalized[:, 1] /= MAX_SPEED
        self.control = torch.from_numpy(normalized).float()

    def __len__(self) -> int:
        return len(self.lidar)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lidar[idx].unsqueeze(0), self.control[idx]

    @classmethod
    def from_file(cls, filepath: str) -> "LidarDataset":
        data = np.load(filepath)
        return cls(data["lidar"], data["control"])

    @classmethod
    def from_files(cls, filepaths: list[str]) -> "LidarDataset":
        if not filepaths:
            raise ValueError("File list is empty")

        lidar_list = []
        control_list = []
        for filepath in filepaths:
            data = np.load(filepath)
            lidar_list.append(data["lidar"])
            control_list.append(data["control"])
            print(f"  Loaded: {filepath} ({len(data['lidar'])} samples)")

        lidar_data = np.concatenate(lidar_list, axis=0)
        control_data = np.concatenate(control_list, axis=0)
        print(f"  Total: {len(lidar_data)} samples")
        return cls(lidar_data, control_data)
