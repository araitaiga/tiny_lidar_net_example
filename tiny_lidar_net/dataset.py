"""LiDAR training dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_lidar_net.control import normalize_labels


class LidarDataset(Dataset):
    """Dataset holding pairs of LiDAR scans and control values.

    Shapes:
        lidar:   (N, 1081)
        control: (N, 2)  [steering, speed]   (stored as physical values)

    Because the model's final layer is tanh, the control labels are returned as
    tensors divided by ``MAX_STEERING`` / ``MAX_SPEED`` so that they are normalized
    into ``[-1, 1]``.
    """

    def __init__(self, lidar_data: np.ndarray, control_data: np.ndarray):
        self.lidar = torch.from_numpy(lidar_data).float()
        self.control = torch.from_numpy(normalize_labels(control_data)).float()

    def __len__(self) -> int:
        return len(self.lidar)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lidar[idx].unsqueeze(0), self.control[idx]

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
