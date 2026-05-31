"""TinyLidarNet model definition."""

import numpy as np
import torch
import torch.nn as nn

from tiny_lidar_net.control import Control


class TinyLidarNet(nn.Module):
    """A 1D CNN that regresses control values from a LiDAR scan.

    The input length is fixed at 1081 (TinyLidarNetL), so FC1 takes the flattened
    length 64 * 28 = 1792 directly. To support a variable input length, replace
    ``self.fc1`` with ``nn.LazyLinear(100)`` (in_features are inferred on the first
    forward pass, equivalent to the lazy build in the official Keras implementation).

    Architecture (input_length=1081 example):
        Input: (batch, 1, 1081)  LiDAR scan
            Conv1d(1 -> 24,  k=10, s=4) -> 268
            Conv1d(24 -> 36, k=8,  s=4) -> 66
            Conv1d(36 -> 48, k=4,  s=2) -> 32
            Conv1d(48 -> 64, k=3,  s=1) -> 30
            Conv1d(64 -> 64, k=3,  s=1) -> 28
            Flatten                      -> 1792
            FC(1792 -> 100) + ReLU + Dropout
            FC(100 -> 50)   + ReLU + Dropout
            FC(50 -> 10)    + ReLU
            FC(10 -> 2)     + tanh   # Output range [-1, 1] (training labels are normalized too)
        Output: (batch, 2)  [steering_norm, speed_norm]  ∈ [-1, 1]
            At inference time, ``Control.from_model_output`` converts back to physical values.
    """

    def __init__(self):
        super().__init__()
        self.input_length = 1081  # Fixed input length (TinyLidarNetL)

        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        # With input length 1081, the time-axis length after conv5 is 28, so the
        # flattened length is 64 * 28 = 1792. To support a variable input length,
        # use nn.LazyLinear(100) for fc1 (in_features inferred on the first forward).
        flat_len = 64 * 28
        self.fc1 = nn.Linear(flat_len, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

    def predict(self, lidar_scan: np.ndarray) -> Control:
        """Predict control values from a single LiDAR scan (de-normalize the tanh output to physical values)."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(lidar_scan).float().view(1, 1, -1).to(device)
            output = self(x)
        return Control.from_model_output(output[0].cpu().numpy())

    def save(self, filepath: str) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "input_length": self.input_length},
            filepath,
        )
        print(f"Model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "TinyLidarNet":
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        model = cls()  # input length is fixed at 1081 (checkpoint's input_length is ignored)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        print(f"Model loaded: {filepath} (input_length={model.input_length})")
        return model
